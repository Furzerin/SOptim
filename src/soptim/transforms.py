"""Model-level transforms: temporal folding, conv-BN fusion, one-call `accelerate`."""
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .neuron import BACKENDS, LIF


def fold(x_seq):
    """[T, B, ...] -> [T*B, ...] (view)."""
    return x_seq.flatten(0, 1)


def unfold(x, T):
    """[T*B, ...] -> [T, B, ...] (view)."""
    return x.unflatten(0, (T, -1))


class TimeFold(nn.Module):
    """Run a stateless module once on all timesteps by folding T into the batch dim."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x_seq):
        return unfold(self.module(fold(x_seq)), x_seq.shape[0])


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Fold BatchNorm into the preceding Conv inside every nn.Sequential (eval only)."""
    if model.training:
        raise RuntimeError("fuse_conv_bn requires model.eval()")
    for seq in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        for i in range(len(seq) - 1):
            conv, bn = seq[i], seq[i + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                seq[i] = fuse_conv_bn_eval(conv, bn)
                seq[i + 1] = nn.Identity()
    return model


def accelerate(model: nn.Module, fused_lif=True, lif_backend="fused", channels_last=False, detach_reset=None,
               compile=False, block_size=2048, numba_threads=None, compact_pool=False,
               time_major_embedding=False, vocab_projection=False, checkpoint_blocks=False, compact_spikes=False):
    """Apply drop-in accelerations to a model built from `soptim.LIF` neurons.

    FP32-equivalent within rounding: LIF backends, channels_last, compile.
    detach_reset=True changes the surrogate gradient; it is not lossless.
    """
    if fused_lif and lif_backend not in BACKENDS:
        raise ValueError(f"lif_backend must belong to {BACKENDS}")
    if time_major_embedding or vocab_projection or checkpoint_blocks or compact_spikes:
        from .models import TinySpikingLM

        if not isinstance(model, TinySpikingLM):
            raise TypeError("Language transforms require TinySpikingLM")
        if time_major_embedding:
            model.time_major_embedding = True
        if vocab_projection:
            model.vocab_projection = True
        if checkpoint_blocks:
            model.checkpoint_blocks = True
        if compact_spikes:
            from .language import PackedSpikeLinear

            for block in model.blocks:
                if type(block.down) is nn.Linear:
                    block.down = PackedSpikeLinear(block.down)
    if compact_pool:
        from .pooling import CompactMaxPool2d

        def replace_pool(module):
            if (type(module) is nn.MaxPool2d and module.kernel_size in (2, (2, 2))
                    and module.stride in (None, 2, (2, 2)) and module.padding in (0, (0, 0))
                    and module.dilation in (1, (1, 1)) and not module.ceil_mode and not module.return_indices):
                return CompactMaxPool2d(numba_threads=numba_threads).train(module.training)
            for name, child in list(module.named_children()):
                replacement = replace_pool(child)
                if replacement is not child:
                    setattr(module, name, replacement)
            return module

        model = replace_pool(model)
    for m in model.modules():
        if isinstance(m, LIF):
            if fused_lif:
                m.backend = lif_backend
                m.block_size, m.numba_threads = block_size, numba_threads
            if detach_reset is not None:
                m.detach_reset = detach_reset
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if compile:
        model = torch.compile(model)
    return model
