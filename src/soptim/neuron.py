"""LIF neuron with interchangeable backends.

All backends implement the same dynamics (hard reset to 0, decay_input=True):
    H[t] = V[t-1] + (X[t] - V[t-1]) / tau
    S[t] = Heaviside(H[t] - v_th)           (ATan surrogate gradient)
    V[t] = H[t] * (1 - S[t])
"""
import math

import torch
from torch import nn

BACKENDS = ("loop", "fused", "numba", "numba_stream", "numba_register")


def atan_surrogate_grad(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return (alpha / 2) / (1 + (math.pi / 2 * alpha * x).square())


class ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * atan_surrogate_grad(x, ctx.alpha), None


def lif_step(x, v, tau=2.0, v_th=1.0, alpha=2.0, detach_reset=False):
    """Reference single-step LIF built from autograd primitives. Returns (spike, new_v)."""
    h = x / tau if v is None else v + (x - v) / tau
    s = ATanSpike.apply(h - v_th, alpha)
    r = s.detach() if detach_reset else s
    return s, h * (1 - r)


def lif_loop(x_seq, tau=2.0, v_th=1.0, alpha=2.0, detach_reset=False):
    """Reference multi-step LIF: Python loop over T, full autograd graph."""
    v, out = None, []
    for t in range(x_seq.shape[0]):
        s, v = lif_step(x_seq[t], v, tau, v_th, alpha, detach_reset)
        out.append(s)
    return torch.stack(out)


class FusedLIF(torch.autograd.Function):
    """Multi-step LIF with a hand-derived BPTT backward.

    Lossless w.r.t. `lif_loop`: forward is bitwise identical, backward equal up to
    float rounding. Saves a single [T, ...] tensor (H) instead of ~5 per step and
    builds no per-step autograd graph.
    """

    @staticmethod
    def forward(ctx, x_seq, tau, v_th, alpha, detach_reset):
        h_seq = torch.empty_like(x_seq)
        torch.div(x_seq[0], tau, out=h_seq[0])
        for t in range(1, x_seq.shape[0]):
            v = h_seq[t - 1].masked_fill(h_seq[t - 1] >= v_th, 0.0)
            torch.sub(x_seq[t], v, out=h_seq[t]).div_(tau).add_(v)
        ctx.save_for_backward(h_seq)
        ctx.cfg = (tau, v_th, alpha, detach_reset)
        return (h_seq >= v_th).to(x_seq.dtype)

    @staticmethod
    def backward(ctx, grad_s):
        (h_seq,) = ctx.saved_tensors
        tau, v_th, alpha, detach_reset = ctx.cfg
        k = 1.0 - 1.0 / tau
        grad_x = torch.empty_like(h_seq)
        grad_v = None
        for t in reversed(range(h_seq.shape[0])):
            h = h_seq[t]
            sg = atan_surrogate_grad(h - v_th, alpha)
            grad_h = grad_s[t] * sg
            if grad_v is not None:
                keep = (h < v_th).to(h.dtype)
                dv_dh = keep if detach_reset else keep - h * sg
                grad_h = grad_h + grad_v * dv_dh
            torch.div(grad_h, tau, out=grad_x[t])
            grad_v = grad_h * k
        return grad_x, None, None, None, None


class LIF(nn.Module):
    """Multi-step LIF neuron. `forward` takes [T, ...]; `step` takes one timestep."""

    def __init__(self, tau=2.0, v_th=1.0, alpha=2.0, detach_reset=False, backend="loop",
                 block_size=2048, numba_threads=None):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.tau, self.v_th, self.alpha = tau, v_th, alpha
        self.detach_reset, self.backend = detach_reset, backend
        self.block_size, self.numba_threads = block_size, numba_threads

    def step(self, x, v):
        return lif_step(x, v, self.tau, self.v_th, self.alpha, self.detach_reset)

    def forward(self, x_seq):
        args = (x_seq, self.tau, self.v_th, self.alpha, self.detach_reset)
        if self.backend.startswith("numba") and x_seq.dtype == torch.float32 and x_seq.device.type == "cpu":
            from ._numba_lif import NumbaLIF
            return NumbaLIF.apply(*args, self.block_size, self.backend != "numba", self.numba_threads,
                                  self.backend == "numba_register")
        if self.backend in ("fused", "numba", "numba_stream", "numba_register"):
            return FusedLIF.apply(*args)
        return lif_loop(*args)

    def extra_repr(self):
        return f"tau={self.tau}, v_th={self.v_th}, detach_reset={self.detach_reset}, backend={self.backend}"
