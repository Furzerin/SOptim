import copy

import pytest
import torch

from soptim import SpikingCNN, accelerate, check_close, fuse_conv_bn, lif_loop
from soptim.neuron import FusedLIF


@pytest.mark.parametrize("detach_reset", [False, True])
def test_fused_lif_matches_reference(detach_reset):
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, 5, 5) * 2
    ref = lambda x: lif_loop(x, 2.0, 1.0, 2.0, detach_reset)
    cand = lambda x: FusedLIF.apply(x, 2.0, 1.0, 2.0, detach_reset)
    res = check_close(ref, cand, [x], rtol=1e-5, atol=1e-6)
    assert res["out_max_abs_err"] == 0.0
    assert res["ok"], res


@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last"])
def test_numba_lif_matches_reference(detach_reset, layout):
    pytest.importorskip("numba")
    from soptim._numba_lif import NumbaLIF

    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, 5, 5) * 2
    if layout == "channels_last":
        x = x.flatten(0, 1).contiguous(memory_format=torch.channels_last).unflatten(0, (4, 8))
    ref = lambda x: lif_loop(x, 2.0, 1.0, 2.0, detach_reset)
    cand = lambda x: NumbaLIF.apply(x, 2.0, 1.0, 2.0, detach_reset)
    res = check_close(ref, cand, [x], rtol=1e-5, atol=1e-6)
    assert res["out_max_abs_err"] == 0.0
    assert res["ok"], res


@pytest.mark.parametrize("block_size", [256, 2048, 8192])
@pytest.mark.parametrize("timesteps", [1, 7])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_numba_block_sizes(block_size, timesteps, detach_reset):
    pytest.importorskip("numba")
    from soptim._numba_lif import NumbaLIF

    inputs = torch.randn(timesteps, 3, 19, generator=torch.Generator().manual_seed(7)) * 3
    result = check_close(
        lambda values: lif_loop(values, detach_reset=detach_reset),
        lambda values: NumbaLIF.apply(values, 2.0, 1.0, 2.0, detach_reset, block_size),
        [inputs],
    )
    assert result["out_max_abs_err"] == 0.0
    assert result["ok"], result


def _logits_and_grads(model, x, y):
    model.zero_grad()
    out = model(x)
    torch.nn.functional.cross_entropy(out, y).backward()
    return out.detach(), [p.grad.clone() for p in model.parameters()]


def _make(**kw):
    torch.manual_seed(0)
    return SpikingCNN(channels=(8, 16, 16, 16), T=4, **kw)


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))


def test_step_vs_multistep_eval(batch):
    x, _ = batch
    a, b = _make(mode="step").eval(), _make(mode="multistep").eval()
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("variant", [
    dict(cache_encoder=True),
    dict(accel=dict(fused_lif=True)),
    dict(accel=dict(fused_lif=True, channels_last=True)),
    dict(accel=dict(fused_lif=True, lif_backend="numba", channels_last=True)),
    dict(cache_encoder=True, accel=dict(fused_lif=True, lif_backend="numba", channels_last=True)),
    dict(cache_encoder=True, accel=dict(fused_lif=True, lif_backend="numba_stream")),
    dict(cache_encoder=True, accel=dict(fused_lif=True, lif_backend="numba_register")),
    dict(cache_encoder=True, accel=dict(fused_lif=True, lif_backend="numba_stream", compact_pool=True)),
    dict(cache_encoder=True, accel=dict(fused_lif=True, lif_backend="numba_register", compact_pool=True)),
])
def test_lossless_variants_train(batch, variant):
    x, y = batch
    variant = dict(variant)
    accel = variant.pop("accel", None)
    ref = _make(mode="multistep", **variant).train()
    cand = _make(mode="multistep", **variant).train()
    if accel:
        cand = accelerate(cand, **accel)
        if accel.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
    out_r, g_r = _logits_and_grads(ref, x, y)
    out_c, g_c = _logits_and_grads(cand, x, y)
    torch.testing.assert_close(out_r, out_c, rtol=1e-4, atol=1e-5)
    for a, b in zip(g_r, g_c):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)
    for name, value in ref.named_buffers():
        torch.testing.assert_close(dict(cand.named_buffers())[name], value, rtol=1e-5, atol=1e-6)


def test_fuse_conv_bn(batch):
    x, y = batch
    m = _make(mode="multistep")
    _logits_and_grads(m, x, y)  # populate BN running stats
    m.eval()
    fused = fuse_conv_bn(copy.deepcopy(m))
    with torch.no_grad():
        torch.testing.assert_close(m(x), fused(x), rtol=1e-4, atol=1e-5)


def test_correctness_gate_rejects_missing_gradient():
    values = torch.ones(3)
    result = check_close(lambda first, second: first + second,
                         lambda first, second: first + second.detach(), [values, values])
    assert not result["ok"]
    assert result["missing_gradient"]


@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "expanded", "sliced"])
@pytest.mark.parametrize("tau", [2.0, 3.7])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("backend", ["numba_stream", "numba_register"])
def test_streaming_lif(layout, tau, detach_reset, backend):
    pytest.importorskip("numba")
    from soptim import LIF

    def arrange(values):
        if layout == "channels_last":
            return values.flatten(0, 1).contiguous(memory_format=torch.channels_last).unflatten(0, (5, 3))
        if layout == "expanded":
            return values[:1].expand_as(values)
        if layout == "sliced":
            return values[..., 1::2]
        return values

    values = torch.randn(5, 3, 7, 4, 9, generator=torch.Generator().manual_seed(19)) * 3
    reference = LIF(tau=tau, detach_reset=detach_reset)
    candidate = LIF(tau=tau, detach_reset=detach_reset, backend=backend, block_size=256, numba_threads=2)
    result = check_close(lambda inputs: reference(arrange(inputs)),
                         lambda inputs: candidate(arrange(inputs)), [values])
    assert result["out_max_abs_err"] == 0.0
    assert result["ok"], result


def test_streaming_inference_omits_membrane(monkeypatch):
    pytest.importorskip("numba")
    from soptim import LIF
    from soptim import _numba_lif

    original = _numba_lif._fwd
    observed = []

    def inspect(*args):
        observed.append(args[1].size)
        return original(*args)

    monkeypatch.setattr(_numba_lif, "_fwd", inspect)
    inputs = torch.randn(4, 3, 7)
    with torch.inference_mode():
        expected = LIF()(inputs)
        actual = LIF(backend="numba_stream")(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert observed == [0]


def test_benchmark_controls_grad_mode():
    from soptim import benchmark

    def inference_only(values):
        assert not torch.is_grad_enabled()
        assert not values.requires_grad
        return values * 2

    assert benchmark(inference_only, [torch.ones(3)], grad=False, warmup=0, iters=1) > 0
    with pytest.raises(ValueError):
        benchmark(inference_only, [torch.ones(3)], iters=0)


def test_autotune_preserves_model_state(batch):
    pytest.importorskip("numba")
    import numba
    from soptim.autotune import autotune_model

    model = _make(mode="multistep", cache_encoder=True)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    threads = numba.get_num_threads()
    reports = autotune_model(model, batch[0], backends=("numba_stream",), block_sizes=(256,),
                             threads=(1, 2), warmup=0, iters=1, repeats=1)
    assert len(reports) == 4
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    assert threads == numba.get_num_threads()
    assert all(any(trial["accepted"] for trial in report["trials"]) for report in reports.values())
    reference = _make(mode="multistep", cache_encoder=True)
    expected, expected_grads = _logits_and_grads(reference, *batch)
    actual, actual_grads = _logits_and_grads(model, *batch)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    for expected_grad, actual_grad in zip(expected_grads, actual_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-3, atol=1e-5)


def test_fast_loader_includes_last_batch(monkeypatch):
    import numpy as np
    import torchvision
    from types import SimpleNamespace
    from soptim.data import FastCIFAR10Loader

    dataset = SimpleNamespace(data=np.zeros((5, 32, 32, 3), dtype=np.uint8), targets=list(range(5)))
    monkeypatch.setattr(torchvision.datasets, "CIFAR10", lambda *args, **kwargs: dataset)
    loader = FastCIFAR10Loader("unused", True, batch_size=3, drop_last=False)
    batches = list(loader)
    assert len(loader) == 2
    assert [len(labels) for _, labels in batches] == [3, 2]
    assert sorted(torch.cat([labels for _, labels in batches]).tolist()) == list(range(5))


def test_accelerate_kernel_controls():
    from soptim import LIF

    layer = accelerate(LIF(), lif_backend="numba_stream", block_size=256, numba_threads=2)
    assert layer.backend == "numba_stream"
    assert layer.block_size == 256
    assert layer.numba_threads == 2
    with pytest.raises(ValueError):
        accelerate(layer, lif_backend="typo")


def test_static_kernel_preserves_channels_last():
    pytest.importorskip("numba")
    from soptim import LIF

    frame = torch.randn(3, 7, 5, 9).contiguous(memory_format=torch.channels_last)
    sequence = frame.unsqueeze(0).expand(4, *frame.shape)
    actual = LIF(backend="numba_stream")(sequence)
    torch.testing.assert_close(actual, LIF()(sequence), rtol=0, atol=0)
    assert actual.flatten(0, 1).is_contiguous(memory_format=torch.channels_last)


def test_time_folding_changes_training_batchnorm(batch):
    reference = _make(mode="step").train()
    folded = _make(mode="multistep").train()
    with torch.no_grad():
        reference(batch[0])
        folded(batch[0])
    assert reference.blocks[0].op[1].num_batches_tracked.item() == 4
    assert folded.blocks[0].op[1].num_batches_tracked.item() == 1


def test_encoder_cache_changes_running_variance(batch):
    reference = _make(mode="multistep").train()
    cached = _make(mode="multistep", cache_encoder=True).train()
    with torch.no_grad():
        reference(batch[0])
        cached(batch[0])
    assert not torch.equal(reference.blocks[0].op[1].running_var, cached.blocks[0].op[1].running_var)


def test_autotune_times_static_input_without_materializing(monkeypatch):
    from soptim import LIF
    from soptim import autotune

    frame = torch.randn(3, 7, 5, 9).contiguous(memory_format=torch.channels_last)
    sequence = frame.unsqueeze(0).expand(4, *frame.shape)
    original = autotune.benchmark
    shapes = []

    def inspect(function, inputs, **kwargs):
        shapes.append(inputs[0].shape)
        assert inputs[0].shape[0] == 1
        assert inputs[0].clone()[0].is_contiguous(memory_format=torch.channels_last)
        return original(function, inputs, **kwargs)

    monkeypatch.setattr(autotune, "benchmark", inspect)
    autotune.autotune_lif(LIF(), sequence, backends=("loop",), warmup=0, iters=1, repeats=1)
    assert shapes


@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "sliced"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("shape", [(3, 7, 8, 10), (2, 5, 7, 9)])
def test_compact_pool_exact(layout, binary, shape):
    pytest.importorskip("numba")
    from soptim.pooling import CompactMaxPool2d

    inputs = torch.randn(shape, generator=torch.Generator().manual_seed(32))
    if binary:
        inputs = (inputs > 0).float()

    def arrange(values):
        if layout == "channels_last":
            return values.contiguous(memory_format=torch.channels_last)
        if layout == "sliced":
            return values[..., ::2]
        return values

    reference = torch.nn.MaxPool2d(2)
    candidate = CompactMaxPool2d(numba_threads=2)
    result = check_close(lambda values: reference(arrange(values)),
                         lambda values: candidate(arrange(values)), [inputs], rtol=0, atol=0)
    assert result["ok"], result


def test_compact_pool_saved_metadata_and_fallback():
    pytest.importorskip("numba")
    from soptim import saved_tensor_bytes
    from soptim.pooling import CompactMaxPool2d

    inputs = torch.randn(2, 3, 8, 8, requires_grad=True)
    pool = CompactMaxPool2d()
    assert saved_tensor_bytes(pool, inputs) == inputs.numel() // 4
    values = inputs.detach().double().requires_grad_()
    result = check_close(torch.nn.MaxPool2d(2), pool, [values], rtol=0, atol=0)
    assert result["ok"], result
    with pytest.raises(RuntimeError):
        pool(torch.empty(1, 0, 4, 4))


def test_compact_pool_transform_limits():
    pytest.importorskip("numba")
    from soptim.pooling import CompactMaxPool2d

    model = torch.nn.Sequential(torch.nn.MaxPool2d(2), torch.nn.MaxPool2d(2, stride=1),
                                torch.nn.MaxPool2d(3), torch.nn.MaxPool2d(2, ceil_mode=True),
                                torch.nn.MaxPool2d(2, return_indices=True)).eval()
    unchanged = list(model.children())[1:]
    accelerate(model, fused_lif=False, compact_pool=True)
    assert isinstance(model[0], CompactMaxPool2d)
    assert not model[0].training
    assert list(model.children())[1:] == unchanged
    assert isinstance(accelerate(torch.nn.MaxPool2d(2), fused_lif=False, compact_pool=True), CompactMaxPool2d)


def test_compact_pool_preserves_training_updates(batch):
    pytest.importorskip("numba")
    reference = accelerate(_make(mode="multistep", cache_encoder=True), lif_backend="numba_stream")
    candidate = accelerate(copy.deepcopy(reference), fused_lif=False, compact_pool=True)
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in (reference, candidate)]
    for _ in range(3):
        outputs, gradients = [], []
        for model, optimizer in zip((reference, candidate), optimizers):
            output, gradient = _logits_and_grads(model, *batch)
            outputs.append(output)
            gradients.append(gradient)
            optimizer.step()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for expected, actual in zip(*gradients):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("channels_last", [False, True])
def test_compact_pool_nan_and_signed_zero(channels_last):
    pytest.importorskip("numba")
    from soptim.pooling import CompactMaxPool2d

    inputs = torch.tensor([[[[float("nan"), 1.0, -0.0, 0.0],
                              [2.0, float("nan"), 0.0, -0.0]]]])
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    results = []
    for pool in (torch.nn.MaxPool2d(2), CompactMaxPool2d()):
        values = inputs.clone().requires_grad_()
        outputs = pool(values)
        outputs.backward(torch.ones_like(outputs))
        results.append((outputs.detach(), values.grad))
    torch.testing.assert_close(results[0][0], results[1][0], equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(torch.signbit(results[0][0]), torch.signbit(results[1][0]))
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
