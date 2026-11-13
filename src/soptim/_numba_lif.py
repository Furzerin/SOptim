"""Fused single-pass LIF kernels for CPU, JIT-compiled with Numba (no C++ toolchain needed).

One read of X and one write of H/S per element for the whole sequence, membrane state kept
in a thread-local buffer, parallel over neurons. Same arithmetic as `lif_loop`.
"""
import math
from contextlib import contextmanager

import numpy as np
import torch
from numba import get_num_threads, njit, prange, set_num_threads

_BLK = 2048


@njit(parallel=True, cache=True, fastmath=False)
def _fwd(x, h, s, tau, v_th, block_size):
    T, N = s.shape
    for b in prange((N + block_size - 1) // block_size):
        lo = b * block_size
        hi = min(lo + block_size, N)
        v = np.zeros(hi - lo, dtype=np.float32)
        for t in range(T):
            input_step = 0 if x.shape[0] == 1 else t
            for j in range(lo, hi):
                vj = v[j - lo]
                ht = vj + (x[input_step, j] - vj) / tau
                if h.size:
                    h[t, j] = ht
                if ht >= v_th:
                    s[t, j] = 1.0
                    v[j - lo] = 0.0
                else:
                    s[t, j] = 0.0
                    v[j - lo] = ht


@njit(parallel=True, cache=True, fastmath=False)
def _bwd(gs, h, gx, tau, v_th, half_alpha, slope, k, detach_reset, block_size):
    T, N = h.shape
    for b in prange((N + block_size - 1) // block_size):
        lo = b * block_size
        hi = min(lo + block_size, N)
        gv = np.zeros(hi - lo, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            for j in range(lo, hi):
                ht = h[t, j]
                z = slope * (ht - v_th)
                sg = half_alpha / (np.float32(1.0) + z * z)
                gh = gs[t, j] * sg
                if t < T - 1:
                    keep = np.float32(1.0) if ht < v_th else np.float32(0.0)
                    gh += gv[j - lo] * (keep if detach_reset else keep - ht * sg)
                gx[t, j] = gh / tau
                gv[j - lo] = gh * k


@njit(parallel=True, cache=True, fastmath=False)
def _fwd_register(inputs, membrane, spikes, tau, threshold, block_size):
    timesteps, neurons = spikes.shape
    for block in prange((neurons + block_size - 1) // block_size):
        start = block * block_size
        stop = min(start + block_size, neurons)
        for neuron in range(start, stop):
            voltage = np.float32(0.0)
            for step in range(timesteps):
                input_step = 0 if inputs.shape[0] == 1 else step
                charged = voltage + (inputs[input_step, neuron] - voltage) / tau
                if membrane.size:
                    membrane[step, neuron] = charged
                fired = charged >= threshold
                spikes[step, neuron] = np.float32(1.0) if fired else np.float32(0.0)
                voltage = np.float32(0.0) if fired else charged


@njit(parallel=True, cache=True, fastmath=False)
def _bwd_register(upstream, membrane, gradients, tau, threshold, half_alpha, slope, decay, detach_reset, block_size):
    timesteps, neurons = membrane.shape
    for block in prange((neurons + block_size - 1) // block_size):
        start = block * block_size
        stop = min(start + block_size, neurons)
        for neuron in range(start, stop):
            voltage_grad = np.float32(0.0)
            for step in range(timesteps - 1, -1, -1):
                charged = membrane[step, neuron]
                scaled = slope * (charged - threshold)
                surrogate = half_alpha / (np.float32(1.0) + scaled * scaled)
                charged_grad = upstream[step, neuron] * surrogate
                if step < timesteps - 1:
                    keep = np.float32(1.0) if charged < threshold else np.float32(0.0)
                    charged_grad += voltage_grad * (keep if detach_reset else keep - charged * surrogate)
                gradients[step, neuron] = charged_grad / tau
                voltage_grad = charged_grad * decay


def _is_dense_time_major(x):
    """True if x's memory is one dense block with dim 0 (time) outermost."""
    if x.storage_offset() != 0 or x.stride(0) * x.shape[0] != x.numel():
        return False
    dims = sorted((st, sz) for st, sz in zip(x.stride(), x.shape) if sz > 1)
    expected = 1
    for st, sz in dims:
        if st != expected:
            return False
        expected *= sz
    return True


def _tn(x):
    """Raw [T, N] numpy view over x's memory (layout-agnostic for element-wise ops)."""
    return x.detach().as_strided((x.shape[0], x.numel() // x.shape[0]), (x.stride(0), 1)).numpy()


def _like(x):
    return torch.empty_like(x)  # preserves dense strides (e.g. channels_last)


@contextmanager
def _threads(count):
    previous = get_num_threads()
    try:
        if count is not None and count != previous:
            set_num_threads(count)
        yield
    finally:
        if get_num_threads() != previous:
            set_num_threads(previous)


class NumbaLIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, tau, v_th, alpha, detach_reset, block_size=_BLK, streaming=False, threads=None,
                register_state=False):
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if streaming and x_seq.stride(0) == 0:
            frame = x_seq[0].unsqueeze(0)
            if not _is_dense_time_major(frame):
                frame = frame.contiguous()
            kernel_input = _tn(frame)
            output_strides = (frame.numel(), *frame.stride()[1:])
            s = torch.empty_strided(x_seq.shape, output_strides, dtype=x_seq.dtype, device=x_seq.device)
        else:
            if not _is_dense_time_major(x_seq):
                x_seq = x_seq.contiguous()
            kernel_input = _tn(x_seq)
            s = _like(x_seq)
        h = _like(s) if not streaming or ctx.needs_input_grad[0] else torch.empty(0, 0)
        f32 = np.float32
        membrane = _tn(h) if h.numel() else h.numpy()
        kernel = _fwd_register if register_state else _fwd
        with _threads(threads):
            kernel(kernel_input, membrane, _tn(s), f32(tau), f32(v_th), block_size)
        ctx.save_for_backward(h)
        ctx.cfg = (tau, v_th, alpha, detach_reset, block_size, threads, register_state)
        return s

    @staticmethod
    def backward(ctx, grad_s):
        (h,) = ctx.saved_tensors
        tau, v_th, alpha, detach_reset, block_size, threads, register_state = ctx.cfg
        if grad_s.stride() != h.stride():
            grad_s = _like(h).copy_(grad_s)
        gx = _like(h)
        f32 = np.float32
        kernel = _bwd_register if register_state else _bwd
        with _threads(threads):
            kernel(_tn(grad_s), _tn(h), _tn(gx), f32(tau), f32(v_th), f32(alpha / 2),
                   f32(math.pi / 2 * alpha), f32(1 - 1 / tau), detach_reset, block_size)
        return (gx,) + (None,) * (len(ctx.needs_input_grad) - 1)
