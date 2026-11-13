"""Benchmark SNN acceleration strategies on CIFAR-10 (1 epoch each).

    python examples/cifar10_benchmark.py                       # all strategies, full epoch
    python examples/cifar10_benchmark.py --max-steps 50        # quick smoke run
    python examples/cifar10_benchmark.py --only baseline fused_lif

Each strategy runs in its own subprocess (clean peak-RSS, isolated torch.compile state).
"""
import argparse
import copy
import itertools
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from soptim import SpikingCNN, accelerate, benchmark, check_close, fuse_conv_bn, lif_loop, saved_tensor_bytes
from soptim.data import FastCIFAR10Loader, torchvision_loader
from soptim.neuron import FusedLIF
from soptim.autotune import autotune_model

# Each entry inherits from `base`; strategies stack cumulatively.
STRATEGIES = {
    "baseline":      dict(kind="reference", loader="torchvision", mode="step", T=4),
    "fast_loader":   dict(base="baseline", kind="augmentation distribution", loader="fast"),
    "time_fold":     dict(base="fast_loader", kind="changes training BN", mode="multistep"),
    "encoder_cache": dict(base="time_fold", kind="changes BN running variance", cache_encoder=True),
    "fused_lif":     dict(base="encoder_cache", kind="lossless", fused_lif=True),
    "numba_lif":     dict(base="fused_lif", kind="lossless", lif_backend="numba"),
    "channels_last": dict(base="numba_lif", kind="lossless", channels_last=True),
    "fused_adam":    dict(base="channels_last", kind="lossless", fused_adam=True),
    "compile":       dict(base="fused_adam", kind="FP32-equivalent", lif_backend="fused", compile=True),
    "detach_reset":  dict(base="fused_adam", kind="slightly lossy", detach_reset=True),
    "bf16_autocast": dict(base="fused_adam", kind="slightly lossy", bf16=True),
    "T2":            dict(base="fused_adam", kind="lossy", T=2),
    "strict_reference": dict(kind="fixed recipe reference", loader="fast", mode="multistep", T=4,
                             cache_encoder=True, comparison="strict_reference"),
    "strict_numba": dict(base="strict_reference", kind="FP32-equivalent", fused_lif=True, lif_backend="numba"),
    "streaming_lif": dict(base="strict_numba", kind="FP32-equivalent", lif_backend="numba_stream"),
    "auto_lif": dict(base="strict_numba", kind="FP32-equivalent", autotune=True),
    "register_lif": dict(base="streaming_lif", lif_backend="numba_register", comparison="streaming_lif"),
    "compact_pool": dict(base="streaming_lif", compact_pool=True, comparison="streaming_lif"),
    "register_pool": dict(base="compact_pool", lif_backend="numba_register"),
    "auto_pool": dict(base="compact_pool", autotune=True,
                       search_backends=("numba", "numba_stream", "numba_register")),
}


def resolve(name):
    cfg = dict(STRATEGIES[name])
    base = cfg.pop("base", None)
    return {**resolve(base), **cfg} if base else cfg


def make_loaders(cfg, args):
    if cfg["loader"] == "fast":
        return (FastCIFAR10Loader(args.data, True, args.batch_size, seed=args.seed, drop_last=False),
                FastCIFAR10Loader(args.data, False, 500))
    return (torchvision_loader(args.data, True, args.batch_size, args.workers, drop_last=False),
            torchvision_loader(args.data, False, 500, args.workers))


def evaluate(model, loader, cfg):
    model.eval()
    correct = total = 0
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cpu", torch.bfloat16, enabled=cfg.get("bf16", False)):
        for x, y in loader:
            if cfg.get("channels_last"):
                x = x.contiguous(memory_format=torch.channels_last)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total, time.perf_counter() - t0


def run_one(name, args):
    cfg = resolve(name)
    setup_started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train_loader, test_loader = make_loaders(cfg, args)
    model = SpikingCNN(T=cfg["T"], mode=cfg["mode"], cache_encoder=cfg.get("cache_encoder", False))
    model = accelerate(model, fused_lif=cfg.get("fused_lif", False), lif_backend=cfg.get("lif_backend", "fused"),
                       channels_last=cfg.get("channels_last", False), detach_reset=cfg.get("detach_reset"),
                       compact_pool=cfg.get("compact_pool", False))
    eager = model
    if cfg.get("compile"):
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, fused=cfg.get("fused_adam", None))
    amp = torch.autocast("cpu", torch.bfloat16, enabled=cfg.get("bf16", False))

    def train_step(x, y):
        with amp:
            loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    result = dict(name=name, kind=cfg["kind"], cfg=cfg)
    model.train()
    batches = iter(train_loader)
    first_batch = next(batches)
    probe = first_batch[0]
    if cfg.get("channels_last"):
        probe = probe.contiguous(memory_format=torch.channels_last)
    if cfg.get("autotune"):
        def progress(trial):
            if trial["stage"] == "timing":
                print(f"  [autotune] {trial['config']} {trial['median_ms']:.3f} ms", file=sys.stderr, flush=True)
        result["autotune"] = autotune_model(eager, probe, backends=cfg.get("search_backends", ("numba", "numba_stream")),
                            progress=progress)
    if not cfg.get("compile"):
        with amp:
            result["saved_act_MB"] = saved_tensor_bytes(copy.deepcopy(eager), probe) / 2**20
    result["setup_s"] = time.perf_counter() - setup_started
    steps = n_img = 0
    t0 = time.perf_counter()
    for x, y in itertools.chain((first_batch,), batches):
        if cfg.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        if steps == 0:
            try:
                loss = train_step(x, y)
            except Exception as e:  # e.g. torch.compile without a C++ toolchain
                return dict(result, skipped=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
            result["first_step_s"] = time.perf_counter() - t0
        else:
            loss = train_step(x, y)
        steps += 1
        n_img += len(y)
        if steps % 50 == 0:
            print(f"  [{name}] step {steps}/{len(train_loader)} loss {loss.item():.3f} "
                  f"{n_img / (time.perf_counter() - t0):.0f} img/s", file=sys.stderr, flush=True)
        if args.max_steps and steps >= args.max_steps:
            break
    result["train_s"] = time.perf_counter() - t0
    result["train_img_s"] = n_img / result["train_s"]
    result["train_images"] = n_img
    result["train_steps"] = steps
    result["full_epoch"] = steps == len(train_loader)
    result["setup_plus_train_s"] = result["setup_s"] + result["train_s"]
    result["test_acc"], result["eval_s"] = evaluate(model, test_loader, cfg)
    if not cfg.get("compile") and cfg["mode"] == "multistep":
        fused = fuse_conv_bn(copy.deepcopy(eager).eval())
        result["test_acc_bnfused"], result["eval_bnfused_s"] = evaluate(fused, test_loader, cfg)
    result["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


def lif_microbench(threads):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    x = torch.randn(4, 128, 64, 16, 16)
    ref = lambda x: lif_loop(x)
    fused = lambda x: FusedLIF.apply(x, 2.0, 1.0, 2.0, False)
    fns = [("loop", ref), ("fused", fused)]
    try:
        from soptim._numba_lif import NumbaLIF
        fns.append(("numba", lambda x: NumbaLIF.apply(x, 2.0, 1.0, 2.0, False)))
    except ImportError:
        pass
    out = {}
    for n, fn in fns:
        xr = x.clone().requires_grad_()
        out[n] = dict(fwd_bwd_ms=benchmark(fn, [x]), saved_MB=saved_tensor_bytes(fn, xr) / 2**20)
    return out


def pool_microbench(threads):
    try:
        from soptim.pooling import CompactMaxPool2d
    except ImportError as error:
        return {"skipped": str(error)}
    torch.set_num_threads(threads)
    inputs = (torch.randn(512, 64, 32, 32, generator=torch.Generator().manual_seed(7)) > 0).float()
    results = dict(shape=list(inputs.shape), layouts={})
    reference, candidate = torch.nn.MaxPool2d(2), CompactMaxPool2d(numba_threads=threads)
    for layout in ("contiguous", "channels_last"):
        memory_format = torch.channels_last if layout == "channels_last" else torch.contiguous_format
        values = inputs.contiguous(memory_format=memory_format)
        correctness = check_close(reference, candidate, [values], rtol=0, atol=0)
        if not correctness["ok"]:
            raise RuntimeError(f"Compact pooling correctness failure: {correctness}")
        functions = {"native": reference, "compact": candidate}
        samples = {name: [] for name in functions}
        for function in functions.values():
            benchmark(function, [values], warmup=2, iters=1)
        order = list(functions)
        shuffle = random.Random(7)
        for _ in range(3):
            shuffle.shuffle(order)
            for name in order:
                samples[name].append(benchmark(functions[name], [values], warmup=0, iters=5))
        results["layouts"][layout] = dict(correctness=correctness, **{
            name: dict(samples_ms=samples[name], median_ms=statistics.median(samples[name]),
                       saved_MiB=saved_tensor_bytes(function, values.detach().requires_grad_()) / 2**20)
            for name, function in functions.items()})
    return results


def print_table(results, micro):
    print("\nLIF micro-benchmark  [T=4, B=128, C=64, 16x16], forward+backward")
    for n, r in micro.items():
        print(f"  {n:6s} {r['fwd_bwd_ms']:8.1f} ms   saved for backward: {r['saved_MB']:7.1f} MB")
    base = next((r for r in results if r["name"] == "baseline" and "skipped" not in r), None)
    hdr = ("strategy", "type", "train s", "img/s", "speedup", "test acc", "eval s", "act MB", "RSS MB")
    print("\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in results:
        if "skipped" in r:
            print(f"| {r['name']} | {r['kind']} | skipped: {r['skipped']} |")
            continue
        comparison = next((entry for entry in results
                   if entry["name"] == r["cfg"].get("comparison") and "skipped" not in entry), base)
        sp = f"{comparison['train_s'] / r['train_s']:.2f}x" if comparison else "-"
        act = f"{r['saved_act_MB']:.0f}" if "saved_act_MB" in r else "-"
        print(f"| {r['name']} | {r['kind']} | {r['train_s']:.1f} | {r['train_img_s']:.0f} | {sp} | "
              f"{r['test_acc']:.4f} | {r['eval_s']:.1f} | {act} | {r['peak_rss_MB']:.0f} |")
    fused = [r for r in results if "eval_bnfused_s" in r]
    if fused:
        print("\nInference conv+BN fusion (FP32-equivalent within rounding):")
        for r in fused:
            print(f"  {r['name']:14s} eval {r['eval_s']:.2f}s -> {r['eval_bnfused_s']:.2f}s, "
                  f"acc {r['test_acc']:.4f} -> {r['test_acc_bnfused']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full epoch")
    p.add_argument("--only", nargs="+", choices=list(STRATEGIES),
                   default=["strict_reference", "strict_numba", "streaming_lif", "auto_lif"])
    p.add_argument("--out", default="results/cifar10_results.json")
    p.add_argument("--run-one", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.run_one:
        print(json.dumps(run_one(args.run_one, args)))
        return

    import torchvision  # download once, before the subprocesses
    torchvision.datasets.CIFAR10(args.data, train=True, download=True)
    torchvision.datasets.CIFAR10(args.data, train=False, download=True)

    micro = lif_microbench(args.threads)
    pool_micro = pool_microbench(args.threads) if any(resolve(name).get("compact_pool") for name in args.only) else None
    passthrough = [f"--{k.replace('_', '-')}={v}" for k, v in vars(args).items()
                   if k not in ("only", "out", "run_one")]
    results = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    def save_results():
        with open(args.out + ".tmp", "w") as output:
            json.dump(dict(args=vars(args), micro=micro, pool_micro=pool_micro, results=results,
                           environment=dict(torch=torch.__version__, python=platform.python_version(),
                                            platform=platform.platform(), device="cpu"),
                           complete=len(results) == len(args.only)), output, indent=2)
        os.replace(args.out + ".tmp", args.out)
    save_results()
    for name in args.only:
        print(f"== {name}: {resolve(name)}", file=sys.stderr, flush=True)
        proc = subprocess.run([sys.executable, __file__, *passthrough, "--run-one", name],
                              stdout=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            results.append(dict(name=name, kind=resolve(name)["kind"], skipped=f"exit code {proc.returncode}"))
        else:
            results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        save_results()
    print_table(results, micro)


if __name__ == "__main__":
    main()
