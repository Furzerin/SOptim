"""Experimental CPU/Gloo parallel plans for TinySpikingLM, not arbitrary models."""
import torch
import torch.distributed as distributed
from torch import nn


class LayerShardedLM(nn.Module):
    """Serial layer sharding with autograd-aware broadcasts; not an overlapped pipeline."""

    def __init__(self, model):
        super().__init__()
        self.rank, self.world_size = distributed.get_rank(), distributed.get_world_size()
        if len(model.blocks) < self.world_size:
            raise ValueError("Model parallelism requires at least one layer per rank")
        self.width = model.embedding.embedding_dim
        self.vocab_size = model.head.out_features
        self.owners = [index * self.world_size // len(model.blocks) for index in range(len(model.blocks))]
        self.embedding = model.embedding if self.rank == 0 else nn.Identity()
        self.blocks = nn.ModuleList(block if owner == self.rank else nn.Identity()
                                    for block, owner in zip(model.blocks, self.owners))
        self.norm = model.norm if self.rank == self.world_size - 1 else nn.Identity()
        self.head = model.head if self.rank == self.world_size - 1 else nn.Identity()

    def forward(self, tokens):
        from torch.distributed.nn.functional import broadcast

        batch, length = tokens.shape
        if self.rank == 0:
            sequence = self.embedding(tokens).transpose(0, 1).contiguous()
        else:
            sequence = torch.zeros(length, batch, self.width, requires_grad=torch.is_grad_enabled())
        for owner in range(self.world_size):
            if self.rank == owner:
                for block, block_owner in zip(self.blocks, self.owners):
                    if block_owner == owner:
                        sequence = block(sequence)
            sequence = broadcast(sequence.contiguous(), src=owner)
        if self.rank == self.world_size - 1:
            logits = self.head(self.norm(sequence)).transpose(0, 1).contiguous()
        else:
            logits = torch.zeros(batch, length, self.vocab_size) + sequence[0, 0, 0] * 0
        return broadcast(logits, src=self.world_size - 1)


def parallelize_language_model(model, mode):
    """Apply a CPU plan after init_process_group; all ranks must start from the same weights."""
    from .models import TinySpikingLM

    if not distributed.is_initialized():
        raise RuntimeError("Initialize a Gloo process group before applying a parallel plan")
    if not isinstance(model, TinySpikingLM):
        raise TypeError("The experimental plans require TinySpikingLM")
    if model.vocab_projection or model.checkpoint_blocks or any(type(block.down) is not nn.Linear for block in model.blocks):
        raise ValueError("Vocabulary factoring, checkpointing and packed spikes are single-process experiments")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("These experimental parallel plans currently support CPU FP32 only")
    if mode == "data":
        return nn.parallel.DistributedDataParallel(model)
    if mode == "model":
        return LayerShardedLM(model)
    if mode not in ("tensor", "sequence"):
        raise ValueError(f"Unknown parallel mode: {mode}")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module

    world_size = distributed.get_world_size()
    if any(block.up.out_features % world_size for block in model.blocks):
        raise ValueError("Hidden width must divide evenly across tensor-parallel ranks")
    mesh = init_device_mesh("cpu", (world_size,))
    plan = {}
    for index in range(len(model.blocks)):
        prefix = f"blocks.{index}"
        plan[f"{prefix}.up"] = ColwiseParallel(input_layouts=Shard(0) if mode == "sequence" else Replicate())
        plan[f"{prefix}.down"] = RowwiseParallel(output_layouts=Shard(0) if mode == "sequence" else Replicate())
        if mode == "sequence":
            plan[f"{prefix}.norm"] = SequenceParallel(sequence_dim=0, use_local_output=False)
    if mode == "sequence":
        model.sequence_shards, model.sequence_rank = world_size, distributed.get_rank()
    return parallelize_module(model, mesh, plan)


def synchronize_language_gradients(model, mode):
    """Materialize partial DTensor gradients and sum replicated sequence-shard gradients."""
    if mode not in ("tensor", "sequence"):
        return
    from torch.distributed.tensor import DTensor

    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if isinstance(parameter, DTensor):
            parameter.grad = parameter.grad.redistribute(placements=parameter.placements)
        elif mode == "sequence":
            distributed.all_reduce(parameter.grad)