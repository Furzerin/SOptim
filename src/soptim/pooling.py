"""Optional CPU FP32 2x2 pooling with exact one-byte backward metadata."""
import numpy as np
import torch
from numba import njit, prange
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as functional

from ._numba_lif import _threads


@njit(inline="always")
def _pool_window(inputs, outputs, offsets, batch, channel, row, column):
    best = inputs[batch, channel, row * 2, column * 2]
    winner = 0
    for offset in range(1, 4):
        value = inputs[batch, channel, row * 2 + offset // 2, column * 2 + offset % 2]
        if value > best or np.isnan(value):
            best, winner = value, offset
    outputs[batch, channel, row, column] = best
    if offsets.size:
        offsets[batch, channel, row, column] = winner


@njit(parallel=True, cache=True, fastmath=False)
def _pool_forward(inputs, outputs, offsets, channels_last):
    batches, channels, height, width = outputs.shape
    planes = height if channels_last else channels
    for plane in prange(batches * planes):
        batch, outer = plane // planes, plane % planes
        if channels_last:
            for column in range(width):
                for channel in range(channels):
                    _pool_window(inputs, outputs, offsets, batch, channel, outer, column)
        else:
            for row in range(height):
                for column in range(width):
                    _pool_window(inputs, outputs, offsets, batch, outer, row, column)


@njit(inline="always")
def _gradient_window(upstream, offsets, gradients, batch, channel, row, column):
    winner = offsets[batch, channel, row, column]
    value = upstream[batch, channel, row, column]
    for offset in range(4):
        gradients[batch, channel, row * 2 + offset // 2, column * 2 + offset % 2] = (
            value if offset == winner else np.float32(0.0))


@njit(parallel=True, cache=True, fastmath=False)
def _pool_backward(upstream, offsets, gradients, channels_last):
    batches, channels, height, width = offsets.shape
    planes = height if channels_last else channels
    for plane in prange(batches * planes):
        batch, outer = plane // planes, plane % planes
        if channels_last:
            for column in range(width):
                for channel in range(channels):
                    _gradient_window(upstream, offsets, gradients, batch, channel, outer, column)
        else:
            for row in range(height):
                for column in range(width):
                    _gradient_window(upstream, offsets, gradients, batch, outer, row, column)


class _CompactPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threads):
        channels_last = inputs.is_contiguous(memory_format=torch.channels_last)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        batch, channels, height, width = inputs.shape
        outputs = torch.empty((batch, channels, height // 2, width // 2), dtype=inputs.dtype,
                              device=inputs.device, memory_format=memory_format)
        offsets = (torch.empty_like(outputs, dtype=torch.uint8) if ctx.needs_input_grad[0]
                   else torch.empty((0, 0, 0, 0), dtype=torch.uint8))
        with _threads(threads):
            _pool_forward(inputs.detach().numpy(), outputs.numpy(), offsets.numpy(), channels_last)
        ctx.save_for_backward(offsets)
        ctx.shape, ctx.channels_last, ctx.threads = inputs.shape, channels_last, threads
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (offsets,) = ctx.saved_tensors
        memory_format = torch.channels_last if ctx.channels_last else torch.contiguous_format
        gradients = torch.empty(ctx.shape, dtype=upstream.dtype, device=upstream.device,
                                memory_format=memory_format)
        if ctx.shape[-2] % 2:
            gradients[:, :, -1, :] = 0
        if ctx.shape[-1] % 2:
            gradients[:, :, :, -1] = 0
        with _threads(ctx.threads):
            _pool_backward(upstream.detach().numpy(), offsets.numpy(), gradients.numpy(), ctx.channels_last)
        return gradients, None


class CompactMaxPool2d(nn.Module):
    """MaxPool2d(2, 2), no padding or overlap; retains only uint8 winner offsets."""

    def __init__(self, numba_threads=None):
        super().__init__()
        self.numba_threads = numba_threads

    def forward(self, inputs):
        if (inputs.device.type != "cpu" or inputs.dtype != torch.float32 or inputs.ndim != 4
            or inputs.shape[1] == 0 or inputs.shape[-2] < 2 or inputs.shape[-1] < 2):
            return functional.max_pool2d(inputs, 2, 2)
        return _CompactPool.apply(inputs, self.numba_threads)