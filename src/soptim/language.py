"""Opt-in, semantics-preserving helpers for the tiny spiking language model."""
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as functional


class _PackedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spikes, weight, bias):
        ctx.save_for_backward(spikes.to(torch.bool), weight)
        return functional.linear(spikes, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        packed, weight = ctx.saved_tensors
        flattened = upstream.reshape(-1, upstream.shape[-1])
        spike_gradient = upstream.matmul(weight) if ctx.needs_input_grad[0] else None
        weight_gradient = (flattened.transpose(0, 1).matmul(packed.to(weight.dtype).reshape(-1, weight.shape[1]))
                           if ctx.needs_input_grad[1] else None)
        bias_gradient = flattened.sum(0) if ctx.needs_input_grad[2] else None
        return spike_gradient, weight_gradient, bias_gradient


class PackedSpikeLinear(torch.nn.Module):
    """Linear fed by exact 0/1 spikes; compress only its saved input, not its weights."""

    def __init__(self, linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight, self.bias = linear.weight, linear.bias
        self.train(linear.training)

    def forward(self, spikes):
        try:
            cpu_autocast = torch.is_autocast_enabled("cpu")
        except TypeError:
            cpu_autocast = torch.is_autocast_cpu_enabled()
        if (not torch.is_grad_enabled() or cpu_autocast or spikes.device.type != "cpu"
                or spikes.dtype != torch.float32 or self.weight.dtype != torch.float32):
            return functional.linear(spikes, self.weight, self.bias)
        return _PackedProjection.apply(spikes, self.weight, self.bias)


def sequence_cross_entropy(logits, targets, reduction="sum", ignore_index=-100):
    """Cross entropy for [batch,length,vocab] logits without copying time-major logits."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch,length,vocab] and matching targets [batch,length]")
    time_major = logits.transpose(0, 1).is_contiguous()
    values = logits.transpose(0, 1) if time_major else logits
    labels = targets.transpose(0, 1) if time_major else targets
    losses = functional.cross_entropy(values.reshape(-1, values.shape[-1]), labels.reshape(-1),
                                     reduction=reduction, ignore_index=ignore_index)
    if reduction == "none":
        losses = losses.view(labels.shape)
        return losses.transpose(0, 1) if time_major else losses
    return losses