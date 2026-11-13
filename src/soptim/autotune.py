"""Bounded, correctness-gated CPU kernel search; no remote LLM or API key required."""
import copy
import importlib.util
import itertools
import platform
import random
import statistics
import time

import torch

from .harness import benchmark, check_close
from .neuron import BACKENDS, LIF


def _configuration(layer):
    return dict(backend=layer.backend, block_size=layer.block_size, numba_threads=layer.numba_threads)


def autotune_lif(layer, example, *, block_sizes=(256, 2048, 8192), threads=(1, 2, 4, 8),
                 backends=BACKENDS, grad=True, warmup=2, iters=5, repeats=3,
                 min_speedup=1.03, progress=None):
    """Tune one LIF in place for an example shape; return the complete search report."""
    if not isinstance(layer, LIF):
        raise TypeError("autotune_lif expects soptim.LIF")
    if example.device.type != "cpu" or example.dtype != torch.float32:
        raise ValueError("The current autotuner supports CPU float32 only")
    if example.ndim < 2 or not example.numel() or example.shape[0] < 1:
        raise ValueError("example must have nonempty [T, ...] dimensions")
    if repeats < 1 or min_speedup < 1 or warmup < 0 or iters < 1:
        raise ValueError("Invalid timing or speedup settings")
    if any(backend not in BACKENDS for backend in backends):
        raise ValueError(f"backends must belong to {BACKENDS}")
    if any(not isinstance(size, int) or size < 1 for size in block_sizes):
        raise ValueError("block_sizes must contain positive integers")
    started = time.perf_counter()
    initial = _configuration(layer)
    configurations = [initial]
    numba_available = importlib.util.find_spec("numba") is not None
    max_threads = 0
    if numba_available:
        import numba
        max_threads = numba.config.NUMBA_NUM_THREADS
    for backend in backends:
        if backend.startswith("numba"):
            if not numba_available:
                continue
            for block_size, count in itertools.product(block_sizes, threads):
                if 1 <= count <= max_threads:
                    configurations.append(dict(backend=backend, block_size=block_size, numba_threads=count))
        else:
            configurations.append(dict(backend=backend, block_size=2048, numba_threads=None))
    configurations = list({tuple(config.items()): config for config in configurations}.values())
    reference = LIF(layer.tau, layer.v_th, layer.alpha, layer.detach_reset)
    generator = torch.Generator().manual_seed(1729)
    small_shape = (min(example.shape[0], 7), 3, 5, 7)
    probes = [example.detach(), torch.randn(small_shape, generator=generator) * 3,
              torch.full((1, 3, 5), layer.tau * layer.v_th)]
    static = example.stride(0) == 0
    timing_input = example[:1] if static else example

    def prepare(values):
        return values[:1].expand(example.shape[0], *values.shape[1:]) if static else values

    trials = []
    functions = []
    for configuration in configurations:
        candidate = copy.deepcopy(layer)
        for name, value in configuration.items():
            setattr(candidate, name, value)

        def candidate_fn(values, neuron=candidate):
            return neuron(prepare(values))

        trial = dict(config=configuration, accepted=False)
        try:
            checks = [check_close(lambda values: reference(prepare(values)), candidate_fn, [probe],
                                  grad=grad, seed=index) for index, probe in enumerate(probes)]
            trial["checks"] = checks
            trial["accepted"] = all(check["ok"] and check["out_max_abs_err"] == 0 for check in checks)
            if trial["accepted"]:
                benchmark(candidate_fn, [timing_input], grad=grad, warmup=warmup, iters=1)
            else:
                trial["reason"] = "spike or gradient mismatch"
        except Exception as error:
            trial["accepted"] = False
            trial["reason"] = f"{type(error).__name__}: {error}"
        trials.append(trial)
        functions.append(candidate_fn)
        if progress:
            progress(dict(stage="correctness", **trial))
    order = [index for index, trial in enumerate(trials) if trial["accepted"]]
    samples = {index: [] for index in order}
    shuffle = random.Random(1729)
    for _ in range(repeats):
        shuffle.shuffle(order)
        for index in order:
            samples[index].append(benchmark(functions[index], [timing_input], grad=grad, warmup=0, iters=iters))
    for index, values in samples.items():
        trials[index]["samples_ms"] = values
        trials[index]["median_ms"] = statistics.median(values)
        if progress:
            progress(dict(stage="timing", **trials[index]))
    valid = [trial for trial in trials if trial["accepted"]]
    if not valid:
        raise RuntimeError("No kernel candidate passed the correctness gate")
    winner = min(valid, key=lambda trial: trial["median_ms"])
    baseline = trials[0] if trials[0]["accepted"] else None
    if baseline and baseline["median_ms"] / winner["median_ms"] < min_speedup:
        winner = baseline
    for name, value in winner["config"].items():
        setattr(layer, name, value)
    report = dict(shape=list(example.shape), strides=list(example.stride()), grad=grad,
                  parameters=dict(tau=layer.tau, v_th=layer.v_th, alpha=layer.alpha, detach_reset=layer.detach_reset),
                  torch_version=torch.__version__, platform=platform.platform(),
                  torch_threads=torch.get_num_threads(), initial=initial, selected=winner["config"],
                  speedup=baseline["median_ms"] / winner["median_ms"] if baseline else None,
                  search_s=time.perf_counter() - started, trials=trials)
    layer.tuning_report = report
    return report


def autotune_model(model, example, **kwargs):
    """Tune observed multi-step LIF calls without changing weights, BN buffers, or RNG state."""
    probe_model = copy.deepcopy(model)
    examples = {}
    handles = []
    for name, layer in probe_model.named_modules():
        if isinstance(layer, LIF):
            def capture(module, inputs, layer_name=name):
                examples.setdefault(layer_name, inputs[0].detach())
            handles.append(layer.register_forward_pre_hook(capture))
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            probe_model(example)
    finally:
        for handle in handles:
            handle.remove()
    del probe_model
    if not examples:
        raise ValueError("No multi-step LIF calls observed; step() does not use these backends")
    layers = dict(model.named_modules())
    return {name: autotune_lif(layers[name], values, **kwargs) for name, values in examples.items()}


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs="+", default=[4, 128, 64, 16, 16])
    parser.add_argument("--static-input", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="results/kda_search.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    values = torch.randn(args.shape, generator=torch.Generator().manual_seed(0)) * 2
    if args.static_input:
        values = values[:1].expand_as(values)
    layer = LIF(backend="numba")

    def progress(trial):
        if trial["stage"] == "timing":
            print(f"{trial['config']}: {trial['median_ms']:.3f} ms", flush=True)
        else:
            print(f"{trial['config']}: {'pass' if trial['accepted'] else trial['reason']}", flush=True)

    report = autotune_lif(layer, values, grad=not args.inference, progress=progress)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected {report['selected']}, speedup {report['speedup']:.2f}x; saved {output}")


if __name__ == "__main__":
    main()