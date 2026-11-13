"""Verification + benchmarking harness.

This is the contract a human or a Kernel Design Agent iterates against:
a candidate is accepted only if `check_close` passes, and ranked by `benchmark`.
"""
import time

import torch


def _clone_inputs(inputs, grad=True):
    return [x.detach().clone().requires_grad_(grad and x.is_floating_point()) if isinstance(x, torch.Tensor) else x
            for x in inputs]


def check_close(ref_fn, cand_fn, inputs, grad=True, rtol=1e-5, atol=1e-6, seed=0):
    """Compare outputs (and input grads for a random upstream grad). Returns max abs errors."""
    results = {}
    outs, grads = [], []
    for fn in (ref_fn, cand_fn):
        xs = _clone_inputs(inputs, grad=grad)
        with torch.set_grad_enabled(grad):
            y = fn(*xs)
        outs.append(y.detach())
        if grad:
            upstream = torch.randn(y.shape, generator=torch.Generator().manual_seed(seed), dtype=y.dtype).to(y.device)
            differentiable = [value for value in xs if isinstance(value, torch.Tensor) and value.requires_grad]
            gradients = torch.autograd.grad(y, differentiable, upstream, allow_unused=True) if y.requires_grad else (None,) * len(differentiable)
            grads.append(gradients)
    results["out_max_abs_err"] = (outs[0] - outs[1]).abs().max().item()
    ok = torch.allclose(outs[0], outs[1], rtol=rtol, atol=atol)
    if grad:
        pairs = list(zip(*grads))
        missing = any((reference is None) != (candidate is None) for reference, candidate in pairs)
        pairs = [(reference, candidate) for reference, candidate in pairs if reference is not None and candidate is not None]
        errs = [(reference - candidate).abs().max().item() for reference, candidate in pairs]
        results["grad_max_abs_err"] = max(errs) if errs else 0.0
        results["missing_gradient"] = missing
        ok = ok and not missing and all(torch.allclose(reference, candidate, rtol=rtol, atol=atol)
                                       for reference, candidate in pairs)
    results["ok"] = ok
    return results


def benchmark(fn, inputs, grad=True, warmup=3, iters=10):
    """Warm median wall time in ms, with fresh input gradients and CUDA synchronization."""
    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be nonnegative and iters must be positive")
    xs = _clone_inputs(inputs, grad=grad)
    devices = {value.device for value in xs if isinstance(value, torch.Tensor) and value.is_cuda}
    times = []
    for i in range(warmup + iters):
        for value in xs:
            if isinstance(value, torch.Tensor):
                value.grad = None
        for device in devices:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.set_grad_enabled(grad):
            y = fn(*xs)
            if grad:
                y.sum().backward()
        for device in devices:
            torch.cuda.synchronize(device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)[len(times) // 2]


def saved_tensor_bytes(fn, *args):
    """Bytes kept alive for backward by one call of fn(*args) (deduplicated by storage)."""
    seen = {}

    def pack(t):
        seen[(t.device, t.untyped_storage().data_ptr())] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*args)
    return sum(seen.values())
