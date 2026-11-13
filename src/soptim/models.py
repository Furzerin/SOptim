"""Example spiking CNN for CIFAR-10 (direct encoding, T timesteps, rate-decoded logits)."""
import torch
from torch import nn

from .neuron import LIF
from .transforms import fold, unfold


class SpikingBlock(nn.Module):
    def __init__(self, c_in, c_out, pool, lif_kwargs):
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out))
        self.lif = LIF(**lif_kwargs)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()


class SpikingCNN(nn.Module):
    """
    mode="step":      naive per-timestep loop over the whole network (snnTorch style).
    mode="multistep": layer-by-layer; stateless ops see a [T*B] batch, LIF loops over T.
    cache_encoder:    (multistep) the first conv+BN sees the same static image at every
                      timestep, so compute it once and broadcast over T.
    """

    def __init__(self, channels=(32, 64, 128, 128), T=4, num_classes=10, mode="step",
                 cache_encoder=False, lif_kwargs=None):
        super().__init__()
        if mode not in ("step", "multistep"):
            raise ValueError(mode)
        self.T, self.mode, self.cache_encoder = T, mode, cache_encoder
        lif_kwargs = lif_kwargs or {}
        c_prev, blocks = 3, []
        for i, c in enumerate(channels):
            blocks.append(SpikingBlock(c_prev, c, pool=i > 0, lif_kwargs=lif_kwargs))
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(c_prev * (32 // 2 ** (len(channels) - 1)) ** 2, num_classes)

    def forward(self, x):
        return self._forward_step(x) if self.mode == "step" else self._forward_multistep(x)

    def _forward_step(self, x):
        v = [None] * len(self.blocks)
        out = 0
        for _ in range(self.T):
            y = x
            for i, b in enumerate(self.blocks):
                y, v[i] = b.lif.step(b.op(y), v[i])
                y = b.pool(y)
            out = out + self.fc(y.flatten(1))
        return out / self.T

    def _forward_multistep(self, x):
        T, first = self.T, self.blocks[0]
        if self.cache_encoder:
            y = first.op(x)
            y = y.unsqueeze(0).expand(T, *y.shape)
        else:
            y = unfold(first.op(fold(x.unsqueeze(0).expand(T, *x.shape))), T)
        s = first.lif(y)
        for b in self.blocks[1:]:
            s = b.lif(unfold(b.op(fold(s)), T))
            s = unfold(b.pool(fold(s)), T)
        return unfold(self.fc(fold(s).flatten(1)), T).mean(0)


class SpikingLanguageBlock(nn.Module):
    def __init__(self, width, hidden_width, lif_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden_width)
        self.lif = LIF(**lif_kwargs)
        self.down = nn.Linear(hidden_width, width)

    def forward(self, sequence):
        return sequence + self.down(self.lif(self.up(self.norm(sequence))))


class TinySpikingLM(nn.Module):
    """Causal character LM; token positions are LIF timesteps, not repeated encoding steps."""

    def __init__(self, vocab_size, width=64, hidden_width=128, depth=2, lif_kwargs=None):
        super().__init__()
        if min(vocab_size, width, hidden_width, depth) < 1:
            raise ValueError("All model dimensions must be positive")
        self.embedding = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList(SpikingLanguageBlock(width, hidden_width, lif_kwargs or {})
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.sequence_shards, self.sequence_rank = 1, 0
        self.time_major_embedding = False
        self.vocab_projection = False
        self.checkpoint_blocks = False

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have nonempty [batch, length] dimensions")
        sequence = (self.embedding(tokens.transpose(0, 1)) if self.time_major_embedding
                else self.embedding(tokens).transpose(0, 1))
        if self.sequence_shards > 1:
            if tokens.shape[1] % self.sequence_shards:
                raise ValueError("Sequence length must divide evenly across sequence-parallel ranks")
            sequence = sequence.chunk(self.sequence_shards, dim=0)[self.sequence_rank]
        for index, block in enumerate(self.blocks):
            if self.vocab_projection and index == 0:
                vocabulary_drive = block.up(block.norm(self.embedding.weight))
                drive = torch.nn.functional.embedding(tokens.transpose(0, 1), vocabulary_drive)
                sequence = sequence + block.down(block.lif(drive))
            elif self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                sequence = checkpoint(block, sequence, use_reentrant=False)
            else:
                sequence = block(sequence)
        return self.head(self.norm(sequence)).transpose(0, 1)
