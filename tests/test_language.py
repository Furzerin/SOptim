import copy

import pytest
import torch

from soptim import accelerate
from soptim.models import TinySpikingLM


def test_language_model_is_causal():
    torch.manual_seed(9)
    model = TinySpikingLM(17, width=8, hidden_width=12, depth=2).eval()
    tokens = torch.randint(0, 17, (3, 11))
    modified = tokens.clone()
    modified[:, 6:] = (modified[:, 6:] + 1) % 17
    with torch.no_grad():
        expected, actual = model(tokens), model(modified)
    assert expected.shape == (3, 11, 17)
    torch.testing.assert_close(actual[:, :6], expected[:, :6], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["fused", "numba", "numba_stream", "numba_register"])
def test_language_backends_match(backend):
    if backend.startswith("numba"):
        pytest.importorskip("numba")
    torch.manual_seed(19)
    reference = TinySpikingLM(17, width=8, hidden_width=12, depth=2)
    candidate = accelerate(copy.deepcopy(reference), lif_backend=backend)
    tokens = torch.randint(0, 17, (3, 13))
    targets = torch.randint(0, 17, (3, 13))
    outputs = []
    for model in (reference, candidate):
        logits = model(tokens)
        torch.nn.functional.cross_entropy(logits.reshape(-1, 17), targets.flatten()).backward()
        outputs.append(logits.detach())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)


def test_language_validation():
    with pytest.raises(ValueError):
        TinySpikingLM(0)
    with pytest.raises(ValueError):
        TinySpikingLM(17)(torch.empty(2, 0, dtype=torch.long))


def _parallel_worker(rank, mode, rendezvous):
    from datetime import timedelta
    import torch.distributed as distributed
    from torch.distributed.tensor import DTensor
    from soptim.parallel import parallelize_language_model, synchronize_language_gradients

    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                                   timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(29)
        reference = TinySpikingLM(17, width=8, hidden_width=12, depth=2)
        with pytest.raises(ValueError, match="FP32"):
            parallelize_language_model(copy.deepcopy(reference).double(), mode)
        candidate = parallelize_language_model(copy.deepcopy(reference), mode)
        tokens = torch.randint(0, 17, (4, 8))
        targets = torch.randint(0, 17, (4, 8))
        expected = reference(tokens)
        torch.nn.functional.cross_entropy(expected.flatten(0, 1), targets.flatten()).backward()
        local_tokens = tokens.chunk(2, dim=0)[rank] if mode == "data" else tokens
        local_targets = targets.chunk(2, dim=0)[rank] if mode == "data" else targets
        if mode == "sequence":
            local_targets = targets.chunk(2, dim=1)[rank]
        actual = candidate(local_tokens)
        expected_local = expected.chunk(2, dim=0)[rank] if mode == "data" else expected
        if mode == "sequence":
            expected_local = expected.chunk(2, dim=1)[rank]
        torch.testing.assert_close(actual, expected_local, rtol=1e-5, atol=1e-6)
        scale = 2 if mode == "data" else 0.5 if mode == "model" else 1
        loss = torch.nn.functional.cross_entropy(actual.flatten(0, 1), local_targets.flatten(), reduction="sum")
        (loss * scale / targets.numel()).backward()
        synchronize_language_gradients(candidate, mode)
        unwrapped = candidate.module if mode == "data" else candidate
        expected_parameters = dict(reference.named_parameters())
        for name, parameter in unwrapped.named_parameters():
            gradient = parameter.grad.full_tensor() if isinstance(parameter, DTensor) else parameter.grad
            torch.testing.assert_close(gradient, expected_parameters[name].grad, rtol=1e-4, atol=1e-6)
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        torch.optim.SGD(candidate.parameters(), lr=0.01).step()
        for name, parameter in unwrapped.named_parameters():
            actual_parameter = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            torch.testing.assert_close(actual_parameter, expected_parameters[name], rtol=1e-5, atol=1e-6)
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("mode", ["data", "tensor", "sequence", "model"])
def test_cpu_parallel_plan(mode, tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU Gloo backend unavailable")
    pytest.importorskip("torch.distributed.tensor.parallel")
    torch.multiprocessing.spawn(_parallel_worker, args=(mode, (tmp_path / "rendezvous").as_uri()),
                                nprocs=2, join=True)


def test_language_windows_cover_every_target():
    from examples.language_benchmark import batches, token_windows

    tokens = torch.arange(20)
    windows = token_windows(tokens, length=6)
    observed = []
    for inputs, targets in batches(windows, batch_size=3):
        assert inputs.shape == targets.shape == (3, 6)
        valid = targets != -100
        torch.testing.assert_close(inputs[valid] + 1, targets[valid])
        observed.extend(targets[valid].tolist())
    assert observed == list(range(1, 20))


def test_language_padding_and_partitioned_loss():
    from examples.language_benchmark import loss_sum, partition

    torch.manual_seed(17)
    logits = torch.randn(4, 8, 17)
    targets = torch.randint(0, 17, (4, 8))
    targets[3] = -100
    targets[2, 5:] = -100
    total = loss_sum(logits, targets)
    for mode, dimension in (("data", 0), ("sequence", 1)):
        losses = []
        for rank in range(2):
            _, local_targets = partition(torch.zeros_like(targets), targets, mode, rank, 2)
            losses.append(loss_sum(logits.chunk(2, dim=dimension)[rank], local_targets))
        torch.testing.assert_close(sum(losses), total)


def test_language_vocabulary_uses_training_split(tmp_path):
    from examples.language_benchmark import load_corpus

    corpus = tmp_path / "custom.txt"
    corpus.write_text("ab" * 45 + "Z" * 10)
    train, validation, metadata = load_corpus(corpus)
    assert len(train) == 90
    assert metadata["vocabulary"] == ["<unk>", "a", "b"]
    assert validation.tolist() == [0] * 10
    assert metadata["source"] == str(corpus.resolve())


@pytest.mark.parametrize("time_major", [False, True])
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_sequence_loss_matches(time_major, reduction):
    from soptim.language import sequence_cross_entropy

    torch.manual_seed(31)
    inputs = torch.randn(3, 7, 17)
    if time_major:
        inputs = inputs.transpose(0, 1).contiguous().transpose(0, 1)
    targets = torch.randint(0, 17, (3, 7))
    targets[-1, -3:] = -100
    results, gradients = [], []
    for optimized in (False, True):
        values = inputs.detach().clone().requires_grad_()
        if optimized:
            loss = sequence_cross_entropy(values, targets, reduction=reduction)
        else:
            loss = torch.nn.functional.cross_entropy(values.reshape(-1, 17), targets.flatten(), reduction=reduction)
            if reduction == "none":
                loss = loss.view_as(targets)
        loss.sum().backward()
        results.append(loss.detach())
        gradients.append(values.grad)
    torch.testing.assert_close(results[0], results[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("options", [dict(time_major_embedding=True), dict(vocab_projection=True),
                                    dict(checkpoint_blocks=True),
                                    dict(compact_spikes=True),
                                    dict(compact_spikes=True, checkpoint_blocks=True),
                                    dict(vocab_projection=True, time_major_embedding=True, checkpoint_blocks=True)])
def test_language_factoring_and_checkpointing(options):
    pytest.importorskip("numba")
    torch.manual_seed(41)
    reference = accelerate(TinySpikingLM(17, width=8, hidden_width=12), lif_backend="numba")
    candidate = copy.deepcopy(reference)
    accelerate(candidate, fused_lif=False, **options)
    tokens = torch.randint(0, 17, (3, 19))
    targets = torch.randint(0, 17, (3, 19))
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in (reference, candidate)]
    for _ in range(3):
        outputs = []
        for model, optimizer in zip((reference, candidate), optimizers):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            torch.nn.functional.cross_entropy(logits.reshape(-1, 17), targets.flatten()).backward()
            outputs.append(logits.detach())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
        for expected, actual in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)
        for optimizer in optimizers:
            optimizer.step()


def test_packed_spikes_save_memory_without_replacing_parameters():
    from soptim import saved_tensor_bytes

    torch.manual_seed(43)
    model = TinySpikingLM(17, width=8, hidden_width=12)
    tokens = torch.randint(0, 17, (3, 19))
    original_ids = [id(parameter) for parameter in model.parameters()]
    reference_bytes = saved_tensor_bytes(model, tokens)
    accelerate(model, fused_lif=False, compact_spikes=True)
    assert [id(parameter) for parameter in model.parameters()] == original_ids
    assert saved_tensor_bytes(model, tokens) < reference_bytes
    with pytest.raises(TypeError):
        accelerate(torch.nn.Linear(8, 12), compact_spikes=True)


def test_packed_projection_observable_storage_and_autocast():
    from soptim import saved_tensor_bytes
    from soptim.language import PackedSpikeLinear

    linear = torch.nn.Linear(12, 8)
    packed = PackedSpikeLinear(linear)
    spikes = (torch.randn(3, 19, 12) > 0).float().requires_grad_()
    assert saved_tensor_bytes(packed, spikes) == spikes.numel() + linear.weight.numel() * 4
    with torch.autocast("cpu", torch.bfloat16):
        torch.testing.assert_close(packed(spikes), linear(spikes), rtol=0, atol=0)


def test_unpadded_batches_preserve_work_and_gradients():
    from examples.language_benchmark import batches, token_windows, loss_sum

    windows = token_windows(torch.arange(20) % 17, 6)
    padded = list(batches(windows, 3, seed=3))
    unpadded = list(batches(windows, 3, seed=3, pad_batch=False))
    for (_, expected), (_, actual) in zip(padded, unpadded):
        torch.testing.assert_close(expected[expected != -100], actual[actual != -100])
    torch.manual_seed(47)
    reference = TinySpikingLM(17, width=8, hidden_width=12)
    candidate = copy.deepcopy(reference)
    for model, (inputs, targets) in zip((reference, candidate), (padded[-1], unpadded[-1])):
        (loss_sum(model(inputs), targets) / (targets != -100).sum()).backward()
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)


def test_foreach_adam_updates_match():
    torch.manual_seed(53)
    reference = TinySpikingLM(17, width=8, hidden_width=12)
    candidate = copy.deepcopy(reference)
    tokens = torch.randint(0, 17, (3, 11))
    targets = torch.randint(0, 17, (3, 11))
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3, foreach=foreach)
                  for model, foreach in ((reference, False), (candidate, True))]
    for _ in range(3):
        for model, optimizer in zip((reference, candidate), optimizers):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            torch.nn.functional.cross_entropy(logits.reshape(-1, 17), targets.flatten()).backward()
            optimizer.step()
        for expected, actual in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            for name in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(optimizers[1].state[actual][name], optimizers[0].state[expected][name],
                                           rtol=1e-5, atol=1e-7)


def test_experiment_order_is_complete_and_reproducible():
    from examples.language_benchmark import experiment_order

    names = ["numba4", "vocab4", "selected4"]
    runs = experiment_order(names, repeats=3, seed=7)
    assert runs == experiment_order(names, repeats=3, seed=7)
    for repeat in (1, 2, 3):
        assert sorted(name for name, index in runs if index == repeat) == sorted(names)
    assert experiment_order(names, repeats=1, seed=7) == [(name, 1) for name in names]
    with pytest.raises(ValueError):
        experiment_order(names, repeats=0, seed=7)


def test_vocabulary_factoring_is_causal_and_stateless():
    pytest.importorskip("numba")
    torch.manual_seed(59)
    reference = TinySpikingLM(17, width=8, hidden_width=12)
    candidate = accelerate(copy.deepcopy(reference), lif_backend="numba", vocab_projection=True,
                           time_major_embedding=True, compact_spikes=True)
    tokens = torch.randint(0, 17, (3, 19))
    changed = tokens.clone()
    changed[:, 9:] = (changed[:, 9:] + 1) % 17
    candidate.eval()
    with torch.inference_mode():
        outputs, future_changed = candidate(tokens), candidate(changed)
        torch.testing.assert_close(outputs[:, :9], future_changed[:, :9], rtol=0, atol=0)
    assert set(reference.state_dict()) == set(candidate.state_dict())
    restored = TinySpikingLM(17, width=8, hidden_width=12)
    restored.load_state_dict(candidate.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens), candidate(tokens), rtol=1e-5, atol=1e-6)