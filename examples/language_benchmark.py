"""One-epoch Tiny Shakespeare benchmark for a tiny causal spiking character LM."""
import argparse
import copy
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

import torch
import torch.distributed as distributed
from torch.nn import functional as functional

from soptim import TinySpikingLM, accelerate, saved_tensor_bytes
from soptim.autotune import autotune_model
from soptim.language import sequence_cross_entropy
from soptim.parallel import parallelize_language_model, synchronize_language_gradients


CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt"
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
STRATEGIES = {
    "loop": dict(backend="loop"),
    "loop1": dict(backend="loop", threads=1),
    "loop4": dict(backend="loop", threads=4),
    "fused": dict(backend="fused"),
    "fused4": dict(backend="fused", threads=4),
    "numba": dict(backend="numba"),
    "numba4": dict(backend="numba", threads=4),
    "streaming": dict(backend="numba_stream"),
    "register": dict(backend="numba_register"),
    "autotune": dict(backend="numba", tune=True),
    "threads1": dict(backend="numba_register", threads=1),
    "threads4": dict(backend="numba_register", threads=4),
    "fused_adam": dict(backend="numba_register", fused_adam=True),
    "detach_reset": dict(backend="numba_register", detach_reset=True, approximate=True),
    "bf16": dict(backend="fused", bf16=True, approximate=True),
    "compile": dict(backend="fused", compile=True),
    "data2": dict(backend="numba_register", parallel="data"),
    "tensor2": dict(backend="numba_register", parallel="tensor"),
    "sequence2": dict(backend="numba_register", parallel="sequence"),
    "model2": dict(backend="numba_register", parallel="model"),
}
STRATEGIES.update({name: dict(backend="numba", threads=4, **options) for name, options in {
    "layout_loss4": dict(layout_loss=True),
    "time_major4": dict(time_major_embedding=True),
    "vocab4": dict(vocab_projection=True),
    "packed4": dict(compact_spikes=True),
    "checkpoint4": dict(checkpoint_blocks=True),
    "prebatch4": dict(prebatch=True),
    "unpad4": dict(unpad=True),
    "foreach4": dict(foreach=True),
    "inference4": dict(inference=True),
    "tiles512_4": dict(block_size=512),
    "bundle4": dict(layout_loss=True, time_major_embedding=True, vocab_projection=True,
                    prebatch=True, unpad=True, foreach=True, inference=True),
    "selected4": dict(layout_loss=True, vocab_projection=True, block_size=512,
                      prebatch=True, unpad=True, inference=True),
    "memory4": dict(compact_spikes=True, checkpoint_blocks=True),
}.items()})
NOT_APPLICABLE = {
    "compact_pool": "No spatial pooling in this language model.",
    "conv_bn_fusion": "No convolutions or BatchNorm; token-wise LayerNorm is part of the recipe.",
    "channels_last": "Activations are [length,batch,features], not 4D image tensors.",
    "encoder_cache": "Tokens change at every timestep; static-image encoder caching would change the model.",
    "T2": "Timesteps are token positions; reducing them changes context and the dataset workload.",
    "fast_cifar_loader": "Text is already encoded and resident in memory; no image augmentation.",
    "time_fold": "Linear projections already batch all token positions; LIF still follows the ordered recurrence.",
}


def load_corpus(path, max_chars=0):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    if max_chars:
        text = text[:max_chars]
    split = int(len(text) * 0.9)
    if split < 2 or len(text) - split < 2:
        raise ValueError("Corpus must contain at least two characters in each split")
    vocabulary = ["<unk>"] + sorted(set(text[:split]))
    indices = {character: index for index, character in enumerate(vocabulary)}
    encode = lambda section: torch.tensor([indices.get(character, 0) for character in section], dtype=torch.long)
    digest = hashlib.sha256(raw).hexdigest()
    return encode(text[:split]), encode(text[split:]), dict(
        source=CORPUS_URL if digest == CORPUS_SHA256 else str(Path(path).resolve()),
        sha256=digest, characters=len(text),
        train_characters=split, validation_characters=len(text) - split, vocabulary=vocabulary,
        subset=bool(max_chars), split="chronological 90/10; training-only vocabulary")


def token_windows(tokens, length):
    if length < 1 or len(tokens) < 2:
        raise ValueError("Positive context length and at least two tokens required")
    pairs = len(tokens) - 1
    padded = math.ceil(pairs / length) * length
    inputs = torch.zeros(padded, dtype=torch.long)
    targets = torch.full((padded,), -100, dtype=torch.long)
    inputs[:pairs], targets[:pairs] = tokens[:-1], tokens[1:]
    return inputs.view(-1, length), targets.view(-1, length)


def batches(windows, batch_size, seed=None, pad_batch=True):
    inputs, targets = windows
    order = (torch.arange(len(inputs)) if seed is None else
             torch.randperm(len(inputs), generator=torch.Generator().manual_seed(seed)))
    for offset in range(0, len(inputs), batch_size):
        selection = order[offset:offset + batch_size]
        batch_inputs, batch_targets = inputs[selection], targets[selection]
        missing = batch_size - len(selection)
        if missing and pad_batch:
            batch_inputs = torch.cat((batch_inputs, torch.zeros(missing, inputs.shape[1], dtype=torch.long)))
            batch_targets = torch.cat((batch_targets, torch.full((missing, targets.shape[1]), -100, dtype=torch.long)))
        yield batch_inputs, batch_targets


def partition(inputs, targets, mode, rank, world_size):
    if mode == "data":
        return inputs.chunk(world_size, dim=0)[rank], targets.chunk(world_size, dim=0)[rank]
    if mode == "sequence":
        return inputs, targets.chunk(world_size, dim=1)[rank]
    return inputs, targets


def loss_sum(logits, targets):
    return functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten(),
                                     ignore_index=-100, reduction="sum")


def sum_metrics(values, mode):
    metrics = torch.tensor(values, dtype=torch.float64)
    if mode in ("data", "sequence"):
        distributed.all_reduce(metrics)
    return metrics.tolist()


def experiment_order(names, repeats, seed):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    runs = []
    for repeat in range(repeats):
        candidates = list(names)
        if repeats > 1:
            random.Random(seed + repeat).shuffle(candidates)
        runs.extend((name, repeat + 1) for name in candidates)
    return runs


def correctness_gate(reference, candidate, inputs, targets, mode, rank, world_size, candidate_loss=loss_sum):
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    expected = reference(inputs)
    denominator = (targets != -100).sum().item()
    (loss_sum(expected, targets) / denominator).backward()
    local_inputs, local_targets = partition(inputs, targets, mode, rank, world_size)
    actual = candidate(local_inputs)
    expected_local = expected.chunk(world_size, dim=0)[rank] if mode == "data" else expected
    if mode == "sequence":
        expected_local = expected.chunk(world_size, dim=1)[rank]
    torch.testing.assert_close(actual, expected_local, rtol=1e-4, atol=1e-5)
    multiplier = world_size if mode == "data" else 1 / world_size if mode == "model" else 1
    (candidate_loss(actual, local_targets) * multiplier / denominator).backward()
    synchronize_language_gradients(candidate, mode)
    parameters = dict(reference.named_parameters())
    unwrapped = candidate.module if mode == "data" else candidate
    unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
    gradient_error = 0.0
    for name, parameter in unwrapped.named_parameters():
        gradient = parameter.grad
        if gradient is not None and hasattr(gradient, "full_tensor"):
            gradient = gradient.full_tensor()
        if gradient is None:
            raise AssertionError(f"Missing gradient: {name}")
        torch.testing.assert_close(gradient, parameters[name].grad, rtol=1e-3, atol=1e-5)
        gradient_error = max(gradient_error, (gradient - parameters[name].grad).abs().max().item())
    errors = torch.tensor([(actual - expected_local).abs().max().item(), gradient_error])
    if mode:
        distributed.all_reduce(errors, op=distributed.ReduceOp.MAX)
    candidate.zero_grad(set_to_none=True)
    return dict(ok=True, logits_max_abs=errors[0].item(), gradients_max_abs=errors[1].item())


def run_worker(name, args):
    config = STRATEGIES[name]
    mode = config.get("parallel")
    rank = distributed.get_rank() if mode else 0
    world_size = distributed.get_world_size() if mode else 1
    threads = config.get("threads", args.threads // world_size)
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    train, validation, corpus = load_corpus(args.data, args.max_chars)
    train_windows, validation_windows = token_windows(train, args.length), token_windows(validation, args.length)
    model = TinySpikingLM(len(corpus["vocabulary"]), args.width, args.hidden_width, args.depth)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    reference = copy.deepcopy(model)
    accelerate(model, lif_backend=config["backend"], numba_threads=threads,
               detach_reset=config.get("detach_reset"), block_size=config.get("block_size", 2048),
               time_major_embedding=config.get("time_major_embedding", False),
               vocab_projection=config.get("vocab_projection", False),
               checkpoint_blocks=config.get("checkpoint_blocks", False),
               compact_spikes=config.get("compact_spikes", False))
    result = dict(name=name, config=config, status="ok", corpus=corpus, parameters=parameter_count,
                  world_size=world_size, threads_per_rank=threads, precision="bf16 autocast" if config.get("bf16") else "fp32")
    pad_batch = not config.get("unpad", False)
    prepared_train = (list(batches(train_windows, args.batch_size, args.seed, pad_batch=pad_batch))
                      if config.get("prebatch") else None)
    prepared_validation = (list(batches(validation_windows, args.batch_size, pad_batch=pad_batch))
                           if config.get("prebatch") else None)
    selected_loss = sequence_cross_entropy if config.get("layout_loss") else loss_sum
    first_inputs, first_targets = (prepared_train[0] if prepared_train is not None else
                                   next(batches(train_windows, args.batch_size, args.seed, pad_batch=pad_batch)))
    if config.get("tune"):
        result["autotune"] = autotune_model(model, first_inputs,
                                            backends=("numba", "numba_stream", "numba_register"))
    if mode:
        model = parallelize_language_model(model, mode)
    if config.get("compile"):
        model = torch.compile(model)
    if not config.get("approximate"):
        probe_length = min(args.length, 16)
        if mode == "sequence" and probe_length % world_size:
            probe_length -= probe_length % world_size
        result["correctness"] = correctness_gate(reference, model, first_inputs[:4, :probe_length],
                                                first_targets[:4, :probe_length], mode, rank, world_size,
                                                candidate_loss=selected_loss)
    else:
        result["correctness"] = {"not_equivalent": "Precision or reset gradient intentionally changed"}
    del reference
    if not mode and not config.get("compile"):
        result["saved_tensors_MiB"] = saved_tensor_bytes(model, first_inputs) / 2**20
    local_parameters = sum(parameter.to_local().numel() if hasattr(parameter, "to_local") else parameter.numel()
                           for parameter in model.parameters())
    result["local_parameters"] = local_parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=config.get("fused_adam", False),
                                  foreach=config.get("foreach", False))
    result["setup_s"] = time.perf_counter() - started
    if mode:
        distributed.barrier()
    training_started = time.perf_counter()
    train_loss, train_count, steps = 0.0, 0, 0
    model.train()
    training_batches = (prepared_train if prepared_train is not None else
                        batches(train_windows, args.batch_size, args.seed, pad_batch=pad_batch))
    for inputs, targets in training_batches:
        denominator = (targets != -100).sum().item()
        local_inputs, local_targets = partition(inputs, targets, mode, rank, world_size)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", torch.bfloat16, enabled=config.get("bf16", False)):
            logits = model(local_inputs)
            summed = selected_loss(logits, local_targets)
        multiplier = world_size if mode == "data" else 1 / world_size if mode == "model" else 1
        (summed * multiplier / denominator).backward()
        synchronize_language_gradients(model, mode)
        optimizer.step()
        train_loss += summed.item()
        train_count += (local_targets != -100).sum().item()
        steps += 1
        if rank == 0 and steps % 50 == 0:
            print(f"[{name}] step {steps}/{math.ceil(len(train_windows[0]) / args.batch_size)}", flush=True)
        if args.max_steps and steps >= args.max_steps:
            break
    if mode:
        distributed.barrier()
    result["train_s"] = time.perf_counter() - training_started
    if mode:
        elapsed = torch.tensor([result["setup_s"], result["train_s"]], dtype=torch.float64)
        distributed.all_reduce(elapsed, op=distributed.ReduceOp.MAX)
        result["setup_s"], result["train_s"] = elapsed.tolist()
    train_loss, train_count = sum_metrics([train_loss, train_count], mode)
    result.update(train_tokens=int(train_count), train_steps=steps, train_nll=train_loss / train_count,
                  train_tokens_s=train_count / result["train_s"], full_epoch=int(train_count) == len(train) - 1,
                  setup_plus_train_s=result["setup_s"] + result["train_s"])
    model.eval()
    validation_started = time.perf_counter()
    validation_loss, validation_count, correct = 0.0, 0, 0
    inference_context = torch.inference_mode() if config.get("inference") else torch.no_grad()
    validation_batches = (prepared_validation if prepared_validation is not None else
                          batches(validation_windows, args.batch_size, pad_batch=pad_batch))
    with inference_context, torch.autocast("cpu", torch.bfloat16, enabled=config.get("bf16", False)):
        for inputs, targets in validation_batches:
            local_inputs, local_targets = partition(inputs, targets, mode, rank, world_size)
            logits = model(local_inputs)
            validation_loss += selected_loss(logits, local_targets).item()
            valid = local_targets != -100
            validation_count += valid.sum().item()
            correct += ((logits.argmax(-1) == local_targets) & valid).sum().item()
    if mode:
        distributed.barrier()
    result["validation_s"] = time.perf_counter() - validation_started
    validation_loss, validation_count, correct = sum_metrics([validation_loss, validation_count, correct], mode)
    nll = validation_loss / validation_count
    result.update(validation_tokens=int(validation_count), validation_nll=nll, perplexity=math.exp(nll),
                  bits_per_character=nll / math.log(2), accuracy=correct / validation_count)
    rss = torch.tensor(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, dtype=torch.float64)
    if mode:
        distributed.all_reduce(rss, op=distributed.ReduceOp.MAX)
    result["max_rank_peak_rss_MiB"] = rss.item()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/tinyshakespeare.txt")
    parser.add_argument("--out", default="results/language_1epoch.json")
    parser.add_argument("--only", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run-one", choices=STRATEGIES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.length, args.batch_size, args.width, args.hidden_width, args.depth, args.threads) < 1:
        parser.error("Model dimensions, batch size, context length, and threads must be positive")
    if min(args.max_steps, args.max_chars) < 0:
        parser.error("max-steps and max-chars must be nonnegative")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    selected = [args.run_one] if args.run_one else args.only
    if any(STRATEGIES[name].get("parallel") for name in selected):
        if args.batch_size < 4 or args.batch_size % 2 or args.length < 2 or args.length % 2 or args.hidden_width % 2 or args.threads < 2:
            parser.error("Two-rank runs require even batch>=4, even length>=2, even hidden width, and threads>=2")
    if args.run_one:
        mode = STRATEGIES[args.run_one].get("parallel")
        try:
            if mode:
                distributed.init_process_group("gloo", timeout=timedelta(seconds=120))
            result = run_worker(args.run_one, args)
            if not mode or distributed.get_rank() == 0:
                Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        finally:
            if distributed.is_initialized():
                distributed.destroy_process_group()
        return
    path = Path(args.data)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading Tiny Shakespeare from pinned source", flush=True)
        with urlopen(CORPUS_URL, timeout=60) as response:
            contents = response.read()
        if hashlib.sha256(contents).hexdigest() != CORPUS_SHA256:
            raise RuntimeError("Downloaded corpus does not match the pinned SHA256")
        path.write_bytes(contents)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = output.parent / (output.stem + "_runs")
    workers.mkdir(exist_ok=True)
    report = dict(args=vars(args), environment=dict(torch=torch.__version__, python=platform.python_version(),
                  platform=platform.platform(), device="cpu"), not_applicable=NOT_APPLICABLE, results=[], complete=False)
    runs = experiment_order(args.only, args.repeats, args.seed)
    for name, repeat in runs:
        config = STRATEGIES[name]
        if config.get("compile") and not (shutil.which("g++") or shutil.which("clang++")):
            result = dict(name=name, status="skipped", reason="No C++ compiler for CPU Inductor")
        elif config.get("parallel") and not distributed.is_gloo_available():
            result = dict(name=name, status="skipped", reason="CPU Gloo backend unavailable")
        else:
            worker_output = workers / f"{name}_repeat{repeat}.json"
            passthrough = [f"--{key.replace('_', '-')}={value}" for key, value in vars(args).items()
                           if key not in ("only", "out", "run_one")]
            command = [sys.executable]
            if config.get("parallel"):
                command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2"]
            command += [__file__, *passthrough, "--run-one", name, "--out", str(worker_output)]
            threads = config.get("threads", args.threads // (2 if config.get("parallel") else 1))
            environment = dict(os.environ, OMP_NUM_THREADS=str(threads), NUMBA_NUM_THREADS=str(threads))
            print(f"== {name}, repeat={repeat}, compute threads/rank={threads}", flush=True)
            started = time.perf_counter()
            process = subprocess.run(command, env=environment)
            if process.returncode == 0:
                result = json.loads(worker_output.read_text())
                result["process_wall_s"] = time.perf_counter() - started
            else:
                result = dict(name=name, status="error", reason=f"Worker exit {process.returncode}; see terminal traceback")
        result["repeat"] = repeat
        report["results"].append(result)
        report["complete"] = len(report["results"]) == len(runs)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
    print("\nstrategy | train s | tokens/s | validation NLL | perplexity | setup s")
    for result in report["results"]:
        if result["status"] == "ok":
            print(f"{result['name']} | {result['train_s']:.2f} | {result['train_tokens_s']:.0f} | "
                  f"{result['validation_nll']:.4f} | {result['perplexity']:.2f} | {result['setup_s']:.2f}")
        else:
            print(f"{result['name']} | {result['status']}: {result['reason']}")
    if any(result["status"] == "error" for result in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()