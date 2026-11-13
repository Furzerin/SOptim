# SOptim

Experimental PyTorch infrastructure for accelerating spiking neural networks. Includes a CIFAR-10 spiking CNN, a tiny causal spiking language model, reference LIF dynamics, fused CPU kernels, and correctness-gated automatic kernel tuning. This is a local prototype, not a published or production-certified PyPI release.

## Install

CPU environment used for the measurements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[cpu,examples,dev]"
python -m pytest -q tests
```

`torch` is the core dependency. The `cpu` extra adds Numba, `examples` adds torchvision, and `dev` adds testing dependencies. Numba uses LLVM JIT and does not require a system C++ compiler. Its first call can include compilation time.

## Use The Library

The recommended starting point on this CPU is `numba_stream`, without autotuning overhead:

```python
import torch
from soptim import LIF, SpikingCNN, accelerate

model = SpikingCNN(T=4, mode="multistep", cache_encoder=True)
model = accelerate(model, lif_backend="numba_stream")
images = torch.randn(8, 3, 32, 32)
logits = model(images)
logits.sum().backward()

neuron = LIF(backend="numba_stream", block_size=256, numba_threads=4)
sequence = torch.randn(4, 8, 32, requires_grad=True)
spikes = neuron(sequence)
spikes.sum().backward()
```

`LIF.forward` accepts `[T, ...]`, starts with zero membrane state, and uses an ATan surrogate gradient. `LIF.step` uses the reference single-step implementation, regardless of `backend`. Consequently, choose `mode="multistep"` for the example's accelerated kernels. The library does not automatically rewrite arbitrary snnTorch/SpikingJelly models.

Numba kernels support CPU FP32 and first-order gradients. Other devices/dtypes fall back to the PyTorch `fused` implementation. Install Numba explicitly when selecting a Numba backend on CPU FP32. Parameters and inputs should be finite; higher-order gradients and nonfinite-state semantics are not supported guarantees.

## What Lossless Means Here

- **FP32-equivalent:** unchanged neuron equations, reset behavior, surrogate, precision, timestep count, and training recipe; forward spikes must match exactly on the acceptance probes and gradients within `rtol=1e-5, atol=1e-6`. Model tests use appropriately looser FP32 reduction tolerances and also compare BatchNorm buffers.
- **Not bitwise training reproduction:** reassociated floating-point arithmetic and changed tensor layout can produce different weight trajectories. Exact kernel checks on finite probes are evidence, not a proof for every input. Small accuracy differences below do not establish an accuracy improvement or a universal no-loss guarantee.
- **Training BatchNorm is not stateless.** Folding time into its batch changes the normalization population and update frequency. Caching repeated encoder inputs also changes the running-variance estimator's sample count. Neither transformation is generally lossless during training. The controlled runs below hold both choices fixed in the reference.
- `detach_reset`, reduced timesteps, reduced precision, different surrogate gradients, and optimizer changes are separate accuracy/algorithm tradeoffs. They are excluded from the automatic lossless search.

## Tiny Spiking Language Model

The second example is [examples/language_benchmark.py](examples/language_benchmark.py), using `soptim.TinySpikingLM`. It is a **41,984-parameter character-level research model**, not a production LLM or a Mamba/SSM reproduction. Its two residual blocks are `LayerNorm -> Linear(64,128) -> LIF -> Linear(128,64)`, with an embedding and a next-character head. LayerNorm is token-wise; there is no BatchNorm, convolution, attention, or pooling.

Here **token positions are the LIF time axis**. Inputs are `[batch,length]`; logits are `[batch,length,vocabulary]`. All positions use batched linear projections, while the membrane recurrence remains causal. State resets at each independent 128-character window, not at each token. No streaming-state cache or incremental generation API is provided yet. Tests verify that changing future tokens cannot change earlier logits.

```python
import torch
from soptim import TinySpikingLM, accelerate

torch.set_num_threads(4)
model = TinySpikingLM(vocab_size=66, width=64, hidden_width=128, depth=2)
model = accelerate(model, lif_backend="numba", numba_threads=4)
tokens = torch.randint(0, 66, (2, 128))
logits = model(tokens)
```

### Text Protocol

- Dataset: [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), downloaded from pinned commit `370cbcd448eb7daf32f21a6be560b70e0b33c4e3` and SHA256-checked. Corpus hash: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Chronological 90/10 split of 1,115,394 characters; the vocabulary is built only from training text, with an explicit unknown token (66 entries total). A local `--data` corpus is also supported and its path/hash is recorded.
- Each full epoch covers all **1,003,853 training next-character targets**, once each, in 246 optimizer steps. Validation covers **111,539 targets**. Disjoint windows reset state, so this is windowed language modeling, not uninterrupted full-document recurrence.
- Context 128, global batch 32, seed 0, Adam 0.001. Partial windows/batches are padded, ignored in loss, and excluded from throughput. Parallel runs use the same global batches and denominator, without duplicated sampler padding.
- Reported quality is validation natural-log NLL, perplexity, bits/character, and character accuracy. One epoch is a functional/speed experiment, not evidence of convergence or downstream LLM quality.

### Language Results

Same CPU-only WSL machine as the vision experiments. Initial run: `results/language_1epoch.json`; lower-thread/parallel repeat: `results/language_thread_parallel_repeat.json`. These contain **25 successful full epochs** across configurations/repeats, plus an explicitly skipped compile attempt. All non-approximate configurations passed pre-training logits and parameter-gradient checks.

The fair four-thread comparison is more informative than the unusually slow eight-thread baseline:

| Strategy | Compute threads | Train s | Valid tokens/s | Validation NLL | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `loop1` | 1 | 18.90 | 53,113 | 2.1845 | 8.89 |
| `loop4` | 4 | 12.46 | 80,571 | 2.1833 | 8.88 |
| `fused4` | 4 | 4.92 | 203,996 | 2.1839 | 8.88 |
| `numba4` | 4 | **2.82** | **356,535** | 2.1839 | 8.88 |
| `threads1` (scalar-state Numba) | 1 | 5.76 | 174,323 | 2.1831 | 8.87 |
| `threads4` (scalar-state Numba) | 4 | 8.04 | 124,891 | 2.1839 | 8.88 |

**Start with `numba4` on this machine:** an observed 4.4x training speedup over the four-thread autograd reference. Saved tensors fall from about 22.2 to 14.3 MiB. This is not a GPU scalability claim. Timing varies: scalar-state four-thread training measured 3.49 s in the first run and 8.04 s in the repeat. These are single-seed measurements, not confidence intervals; small quality differences reflect floating-point training trajectories, not a demonstrated accuracy improvement.

Initial eight-thread experiments, retained to show unsuccessful configurations:

| Strategy | Train s | Validation NLL | Setup s | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `loop` | 733.02 | 2.1833 | 0.79 | Severe small-operation/thread overhead; not the headline baseline. |
| `fused` | 47.76 | 2.1836 | 0.71 | Reduces autograd overhead. |
| `numba` | 43.97 | 2.1836 | 0.97 | Slower than the four-thread configuration. |
| `streaming` | 46.15 | 2.1836 | 0.95 | No repeated static input to exploit. |
| `register` | 52.13 | 2.1836 | 0.84 | Neuron-major scalar-state candidate. |
| `autotune` | 44.11 | 2.1836 | 62.70 | LIF-only tuning chose four Numba threads but left PyTorch at eight; no net gain. |
| `fused_adam` | 62.42 | 2.1836 | 0.80 | Slower; optimizer-state bitwise equivalence not claimed. |
| `detach_reset` | 56.46 | 2.1998 | 0.87 | Changes gradients; not lossless. |
| `bf16` | 63.05 | 2.1818 | 0.73 | Runs on these linear layers, unlike the vision Conv backward; approximate and slower here. |
| `compile` | skipped | - | - | No C++ compiler for CPU Inductor. |

Epoch time excludes setup/search and evaluation, but includes the first training step. `process_wall_s` also captures launch/import/shutdown time. Distributed timing uses the slowest rank, not summed rank times. Peak RSS is the **maximum individual rank**, not total multi-process memory. The saved-tensor probe is FP32 and not measured for distributed/compiled models; do not use the BF16 row's probe as BF16 peak memory. Kernel-only tuning cannot substitute for tuning end-to-end thread counts.

### Parallel Experiments

Real two-process CPU/Gloo runs, launched with `torch.distributed.run`, with four PyTorch/Numba compute threads per rank (eight total, plus runtime communication threads). All use `numba_register` to keep the parallel comparison's backend fixed. The following plans are available in [src/soptim/parallel.py](src/soptim/parallel.py):

| Plan | What is partitioned | First / repeat epoch s | Repeat NLL |
| --- | --- | ---: | ---: |
| `data2` | Batch: 16 windows per rank; full model replicas, DDP gradient averaging. | 15.59 / 7.29 | 2.1823 |
| `tensor2` | Feed-forward hidden neurons: column-sharded `up`, local full-time LIF, row-sharded `down` with reduction. | 19.99 / 14.49 | 2.1826 |
| `sequence2` | Token-wise norms/residuals sharded along sequence, combined with tensor-parallel projections. Gather the full sequence before LIF, reduce-scatter after `down`. | 16.56 / 21.94 | 2.1830 |
| `model2` | Embedding/first block on rank 0; second block/final norm/head on rank 1; autograd-aware activation broadcasts. | 11.01 / 11.41 | 2.1839 |

These plans were slower than the best single-process configuration. They are experimental baselines for larger-model tests, **not recommended accelerations for this tiny CPU workload**. DDP owns all 41,984 parameters per rank, TP/SP about 25,472, and layer sharding about 21,000 per rank. Runtime/communication overhead dominates at this scale.

Important distinctions and contracts:

- `sequence2` is **sequence parallelism of token-wise operations plus tensor parallelism**, not independent temporal chunks or a parallel LIF scan. Each neuron still sees the complete ordered sequence. Resetting membrane state at rank boundaries would change the model and is not done.
- `model2` is serial layer sharding, **not overlapped pipeline parallelism**. No pipeline schedule, multiple devices, distributed dataset service, or multi-node scaling has been validated.
- `parallelize_language_model(model, mode)` requires an initialized Gloo group, the same initial weights on all ranks, a CPU FP32 `TinySpikingLM`, and compatible divisibility. The benchmark handles initialization, loss scaling and cleanup. The `parallel` extra declares `torch>=2.5`; these experimental APIs are tested here only with 2.15.0.
- After backward, TP/SP callers must invoke `synchronize_language_gradients(model, mode)`. SP sums gradients for replicated parameters and resolves DTensor partial gradients. DDP sums local-token losses with a world-size factor before its average; model-parallel replicated output losses are divided by world size. Use the example's loop rather than mixing these reduction conventions.
- The layer-sharded prototype uses PyTorch's autograd-aware `distributed.nn.functional.broadcast`, which emits a deprecation warning in 2.15. A future API migration is needed; the tested implementation is retained rather than silently replacing it with a non-autograd collective.
- Further experiments: overlapped microbatch pipelines, FSDP/ZeRO for larger optimizer states, and temporal/context sharding that communicates membrane state **and backward state**. Those are not implemented or measured here. GPU/NCCL results require GPU hardware.

### Reproduce The Language Example

```bash
pip install -e ".[cpu,parallel]"
python examples/language_benchmark.py --only loop4 fused4 numba4 --out results/language_local.json
python examples/language_benchmark.py --only threads4 data2 tensor2 sequence2 model2 --out results/language_parallel.json
python examples/language_benchmark.py --out results/language_all.json
python examples/language_benchmark.py --only numba4 data2 --max-chars 12000 --max-steps 2 --out results/language_smoke.json
```

The parent script downloads/verifies data once, starts clean workers, launches distributed workers itself, and saves JSON after each variant. Do not wrap the parent command in `torchrun`; it already does that for distributed variants. `--max-chars` creates an explicit corpus subset, while `--max-steps` truncates training. Neither smoke option reproduces the full-corpus results above. `--data`, `--width`, `--hidden-width`, `--depth`, `--length`, and `--batch-size` support further experiments. For two-rank runs use even batch/context/hidden widths and at least two blocks for layer sharding.

Vision-only methods are explicitly listed in the JSON as inapplicable: compact pooling, Conv-BN folding, channels-last image layout, static encoder caching, and CIFAR augmentation. Stateless linear projections are already batched across token positions. Shortening token length is a different context workload, not the vision `T=2` acceleration.

### Extended Language KDA Search

This round added **10 individual candidates and three combinations**, using the same fixed FP32 model, four compute threads, seed, optimizer batch boundaries, and all 1,003,853/111,539 train/validation targets. The assistant-led KDA loop used an independent research pass, a warmed training-step profile, correctness tests, a full-epoch screen, and shuffled repeated full epochs. This is measured local optimization, not an external autonomous code-generation service.

The profile attributed about 47% of CPU time to LIF forward/backward and 28% to matrix operations. That motivated factoring repeated token work, in addition to memory/layout experiments. No precision reduction, temporal skipping, changed reset, or changed surrogate is included in this matrix.

| Candidate | Call or setting | Screen train s | Saved tensors MiB | Finding |
| --- | --- | ---: | ---: | --- |
| `numba4` | Existing four-thread baseline | 2.633 | 14.27 | Fresh control for this round. |
| `layout_loss4` | `sequence_cross_entropy(logits, targets)` | 2.537 | 14.27 | Avoids copying time-major logits; initial gain did not repeat alone. |
| `time_major4` | `accelerate(..., time_major_embedding=True)` | 2.752 | 14.27 | Embedding lookup writes time-major data directly; slower here. |
| `vocab4` | `accelerate(..., vocab_projection=True)` | 2.386 | 12.27 | First normalization/projection computed once per vocabulary entry per forward, then gathered. |
| `packed4` | `accelerate(..., compact_spikes=True)` | 2.683 | 11.27 | Save binary projection inputs as bool; about 21% less model saved storage, not a speed win. |
| `checkpoint4` | `accelerate(..., checkpoint_blocks=True)` | 3.472 | 4.08 | Recompute blocks during backward; about 71% less saved storage, slower training. |
| `prebatch4` | Materialize `list(batches(...))` before training | 2.583 | 14.27 | Same shuffle; moves data gathering into separately measured setup. Increased RSS here. |
| `unpad4` | `batches(..., pad_batch=False)` | 2.611 | 14.27 | Remove only wholly padded batch rows. Every valid target is retained. |
| `foreach4` | `torch.optim.Adam(..., foreach=True)` | 2.637 | 14.27 | Three-step parameter/moment checks pass; no measured CPU training benefit. |
| `inference4` | `torch.inference_mode()` during validation | 2.634 | 14.27 | Evaluation-only change; no training acceleration claim. |
| `tiles512_4` | `accelerate(..., block_size=512)` | 2.440 | 14.27 | Smaller LIF cache/work tiles than the 2048 default. |
| `bundle4` | Layout loss + time-major lookup + vocabulary factoring + prebatch + unpad + foreach + inference | 2.360 | 12.27 | First combined candidate. |
| `memory4` | Packed spikes + checkpointing | 3.715 | 4.08 | No additional saved-storage reduction over checkpointing alone; slower. |

`sequence_cross_entropy` is imported from `soptim.language` and supports `sum`, `mean`, and `none` reductions plus `ignore_index`. It uses the contiguous physical token order for the cross-entropy call and reorders targets instead of logits. Sum reduction order may differ in FP32; no bitwise loss guarantee is claimed.

`vocab_projection=True` is specific to the unmodified `TinySpikingLM`: its first block's input depends only on token identity, with no position-dependent input or cross-token normalization. The vocabulary projection is **recomputed on every forward**, remains in autograd, and is never cached across optimizer updates. Repeated-token gradients are accumulated through the gather. Small FP32 gradient-reduction differences remain possible. This can be worse for large vocabularies; it is not a generic LLM transform.

Packed spikes use a custom first-order autograd projection, saving a one-byte bool value instead of FP32 for known 0/1 inputs. Only the language block's post-LIF `down` projections are wrapped. Parameters and state-dict keys are preserved. Do not apply `PackedSpikeLinear` to arbitrary analog activations: binary inputs are its contract. CPU FP32 is optimized; unsupported devices/dtypes and CPU autocast use native linear. Membrane states, weights, and surrogate gradients remain full precision. An initial hook-based prototype was replaced because nested hooks obscured storage measurement; the retained implementation's bool storage is explicitly observable by the memory probe.

Checkpointing uses PyTorch non-reentrant checkpointing with RNG preservation and only runs during gradient-enabled training. There is no BatchNorm in these blocks. Combined with vocabulary factoring, the specially factored first block is not checkpointed; subsequent ordinary blocks are. These new language model transforms are single-process experiments; parallel plans reject vocabulary factoring, checkpointing and packed spikes rather than silently accepting unvalidated compositions. Public APIs validate that these transforms receive a `TinySpikingLM`.

#### Repeated Results

Six candidates were run for **three shuffled full epochs each**, with identical training seed and data order within every run. `selected4` uses vocabulary factoring, 512-element LIF blocks, layout-aware loss, prebatched data, no wholly padded batch rows, and inference mode. It intentionally excludes time-major lookup, foreach Adam, packing, and checkpointing.

| Strategy | Median train s | Min-max s | Throughput vs baseline | Median setup + train s |
| --- | ---: | --- | ---: | ---: |
| `numba4` | 2.609 | 2.567-2.633 | 1.000x | 3.320 |
| `layout_loss4` | 2.772 | 2.645-2.922 | 0.941x | 3.474 |
| `vocab4` | 2.388 | 2.366-2.546 | 1.092x | 3.094 |
| `tiles512_4` | 2.512 | 2.414-2.534 | 1.039x | 3.249 |
| `bundle4` | 2.394 | 2.358-2.422 | 1.090x | 3.130 |
| **`selected4`** | **2.205** | **2.165-2.251** | **1.183x** | **2.920** |

The selected recipe shows **18.3% higher throughput / 15.5% shorter training time** than the fresh baseline in this repeat set. Including measured setup, median time drops from 3.320 to 2.920 s. Median validation time is 0.089 s versus 0.108 s. Validation NLL is 2.1834 versus 2.1839, perplexity about 8.88 for both; this small difference is not an accuracy-improvement claim. Three repeats reveal timing variability, not statistical certainty or multi-seed convergence. The layout-loss-only screen result is not used as a positive claim because it reversed on repetition.

There are **31 additional full epochs** in this round (13 screening runs plus 18 repeated runs), bringing the language-model experiment history to 56 completed full epochs. Every repeated result passed the logits/gradient gate and token-count audit. Tests also cover padded losses, three optimizer updates and Adam moments, causality after vocabulary factoring, checkpoint composition, retained parameter identity, and state-dict round-trip. Saved-tensor numbers include tensors autograd retains and are **not peak RSS**; checkpointing's 71% probe reduction did not imply a comparable reduction in process memory.

Reports: `results/language_kda_candidates_1epoch.json` and `results/language_kda_repeated.json`. Original screening order is preserved for one repeat; `--repeats N` shuffles variant order reproducibly when N > 1, keeps training seeds fixed, and saves each run separately.

```python
import torch
from soptim import TinySpikingLM, accelerate
from soptim.language import sequence_cross_entropy

torch.set_num_threads(4)
model = accelerate(TinySpikingLM(66), lif_backend="numba", numba_threads=4,
                   block_size=512, vocab_projection=True)
tokens = torch.randint(0, 66, (32, 128))
targets = torch.randint(0, 66, (32, 128))
loss = sequence_cross_entropy(model(tokens), targets) / targets.ne(-100).sum()
loss.backward()
```

The synthetic snippet demonstrates the model/loss APIs; use the benchmark for the complete selected recipe and real corpus:

```bash
python examples/language_benchmark.py --only numba4 layout_loss4 time_major4 vocab4 packed4 checkpoint4 prebatch4 unpad4 foreach4 inference4 tiles512_4 bundle4 memory4 --out results/language_kda_candidates_1epoch.json
python examples/language_benchmark.py --only numba4 layout_loss4 vocab4 tiles512_4 bundle4 selected4 --repeats 3 --out results/language_kda_repeated.json
```

Rebuild prebatched data when epoch shuffle/augmentation changes. Transform flags are not in `state_dict`, so reapply them after loading weights. Recommended choice is objective-dependent: `selected4` for the tested small-vocabulary CPU workload; checkpointing for saved-memory pressure, not raw speed. Reference guidance: [PyTorch checkpointing](https://docs.pytorch.org/docs/2.15/checkpoint.html) describes the recomputation/RNG tradeoff, and [Adam documentation](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html) explains why foreach can use more memory and is not universally faster.

## Competition-Inspired KDA Round

Continued on 2026-11-09 using the same propose/validate/time/revise loop, including independent read-only research. These are our CPU adaptations of general optimization principles, **not copied competition submissions or claimed winning techniques**.

| Primary source | Verified scope | What transfers to this library |
| --- | --- | --- |
| [GPU MODE official competition repository](https://github.com/gpu-mode/reference-kernels) and [NVIDIA problems](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia) | NVIDIA Blackwell NVFP4 GEMM/GEMV/grouped/dual-GEMM tasks, with reference implementations and evaluators. | Specialized kernels and shape-specific validation; NVFP4 itself is not a lossless FP32 SNN optimization. No competition code was copied. |
| [NVIDIA vectorized-memory guidance](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) | Memory-instruction reduction, alignment constraints, register-pressure tradeoffs. | Test loop layout, compact exact metadata, and reduced memory passes. CUDA vector instructions are not available on this CPU. |
| [SC26 Student Cluster Competition](https://sc26.supercomputing.org/students/student-cluster-competition/) at [ACM/IEEE-sponsored SC26](https://sc26.supercomputing.org/) | Architecture-specific tuning, scientifically acceptable outputs, MLPerf time-to-quality, and power constraints. The 2026 event is scheduled for November 16-18, not yet completed as of this round. | Report full-workload quality and setup costs, not just isolated kernel scores. No energy measurement or SCC benchmark submission was performed here. |

Two new lossless candidates were implemented:

1. **`numba_register`: scalar-state temporal recurrence.** Interchange the neuron/time loops so each neuron carries its voltage and backward state in a scalar variable rather than a scratch array. Arithmetic, reset, and surrogate are unchanged. This may reduce scratch traffic but sacrifice SIMD efficiency; actual register allocation is compiler-dependent. It passed checks but lost the representative 39-candidate kernel search to streaming.
2. **Compact 2x2 pooling.** Store the winning location as a uint8 offset (0-3), rather than retaining the input plus int64 indices for backward. Preserve first-maximum tie handling, NaN handling tested against PyTorch, layout, and exact gradients. Since windows do not overlap, backward writes each four-element window without atomics. An initial scalar-addressing implementation was slower; layout-specific loops and fusing gradient-zero writes into the backward kernel reversed the microbenchmark regression.

### New Full-Epoch Results

Same CPU, seed, T=4, optimizer, normalization recipe, and dataset as described below. Each row processed all 50,000 training images and evaluated all 10,000 test images. Comparisons are against the streaming baseline **in this round**, not older timings.

| Strategy | Train s | Images/s | Training speedup | Setup s | Accuracy | Saved tensors MiB | Eval s | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `streaming_lif` | 296.87 | 168.42 | 1.000x | 2.13 | 60.54% | 926.50 | 21.15 | 2125 |
| `register_lif` | 286.95 | 174.25 | 1.035x | 2.21 | 60.54% | 926.50 | 20.93 | 2177 |
| `compact_pool` | 299.73 | 166.82 | 0.990x | 2.17 | 60.54% | 627.50 | 19.77 | 2145 |
| `register_pool` | 297.76 | 167.92 | 0.997x | 2.22 | 60.54% | 627.50 | 18.28 | 2144 |
| `auto_pool` | 273.23 | 183.00 | 1.087x | 315.22 | 60.54% | 627.50 | 20.92 | 2840 |

Interpretation: compact pooling reduces model saved-tensor storage by **299 MiB (32.3%)**, but does not reduce measured peak RSS or training time in this run. The automatic combination gives an observed 8.7% training-throughput gain, **excluding search**. Setup + training is 588.45 s versus 299.00 s for the baseline, so it remains slower for a single cold run. The scalar-state 3.5% difference is too small to treat as a robust win without repeated full runs. All rows are single-seed observations, not accuracy or speed guarantees.

The model search selected `numba_stream` (block 256, eight threads) for the encoder and ordinary Numba for the other three layers; it selected no scalar-state layers. Compact pooling was an explicitly enabled candidate, not an automatically searched model rewrite. The LIF search stays correctness-gated. All five runs retain 60.54% accuracy; the compact-pool replacement additionally passes exact three-step checks of logits, gradients, weights, and BN buffers against the same streaming model.

Raw local report: `results/competition_kda_1epoch.json`. This round adds five completed full epochs to the prior twelve. Generated reports remain git-ignored; the table preserves their summary.

### New Kernel Results

Pooling forward + backward on binary `[512,64,32,32]` input, warmed, three shuffled rounds of five measurements each:

| Layout | Native pool ms | Compact pool ms | Operator speedup | Native saved MiB | Compact saved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contiguous | 50.55 | 33.51 | 1.51x | 192 | 8 |
| Channels-last | 67.65 | 42.41 | 1.59x | 192 | 8 |

The retained-storage reduction is 24x at this operator boundary: metadata itself is 8x smaller, and the original input is no longer saved. This is not a claim of 24x lower model memory. The initial prototype measured 84.59 vs 46.40 ms contiguous and 83.68 vs 68.83 ms channels-last; it was revised, not accepted as a speedup. The revised isolated search is in `results/pool_kernel_search.json`; the table above uses the microbenchmark bundled with the full-epoch report.

The LIF search (`results/kda_register_search.json`) validated all 39 candidates. Best streaming: 19.34 ms; best scalar-state: 24.14 ms; default Numba: 21.65 ms on the static-input shape `[4,128,64,16,16]`. Reducing scratch storage did not automatically improve that kernel. This is why the scalar-state backend remains opt-in.

### Call The New Strategies

```python
from soptim import SpikingCNN, accelerate
from soptim.pooling import CompactMaxPool2d
from soptim.autotune import autotune_model

model = SpikingCNN(T=4, mode="multistep", cache_encoder=True)
model = accelerate(model, lif_backend="numba_stream", compact_pool=True)
pool = CompactMaxPool2d(numba_threads=8)

scalar_model = accelerate(SpikingCNN(mode="multistep", cache_encoder=True),
                          lif_backend="numba_register")
```

`compact_pool=True` replaces only plain `nn.MaxPool2d` with kernel/stride 2, no padding, dilation 1, no ceil mode, and no returned indices. Other pooling modules remain unchanged. The compact operator supports CPU FP32 first-order backward, including odd input sizes; other dtypes/devices use native PyTorch pooling. Numba is required to import/enable it. It does not compress membrane states or change spike values. For independent use, `pool` expects `[N,C,H,W]` input.

To search LIF settings with compact pooling held fixed, call `autotune_model(model, images, backends=("numba", "numba_stream", "numba_register"))` with a representative batch. Persist the returned recipe as shown below. This search does not enable compact pooling automatically or change BN/T/reset/precision.

```bash
python -m soptim.autotune --static-input --out results/kda_register_search.json
python examples/cifar10_benchmark.py --only streaming_lif register_lif compact_pool register_pool auto_pool --out results/competition_kda_1epoch.json
```

## Earlier Full-Epoch Measurements

Measured on Intel Core Ultra 7 268V, WSL2 Linux, 8 PyTorch threads, CPU only, Python 3.14.4, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, Numba 0.68.1. No NVIDIA GPU or working C++ compiler was available.

Every completed run below trains on **all 50,000 CIFAR-10 images once**, including the final partial batch: batch size 128, 391 optimizer steps, seed 0, Adam at 0.001. All use the same four-convolution SNN `(32, 64, 128, 128)`, T=4, hard reset, ATan surrogate, cached encoder, multistep BatchNorm, and seeded fast loader. Evaluation covers all 10,000 test images. No test-set calibration is used.

| Variant | Training s | Images/s | vs reference | Setup s | Test accuracy | Eval s | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `strict_reference`: autograd LIF | 627.13 | 79.73 | 1.00x | 2.39 | 60.25% | 29.82 | 3542 |
| `strict_numba`: default fused kernel | 321.71 | 155.42 | 1.95x | 2.37 | 60.51% | 23.49 | 2730 |
| `streaming_lif`: layout-preserving | 280.29 | 178.39 | 2.24x | 2.34 | 60.54% | 20.39 | 2164 |
| `auto_lif`: before calibration correction | 302.72 | 165.17 | 2.07x | 208.96 | 60.51% | 21.34 | 2371 |

After correcting static-input calibration to preserve the physical layout, the two affected experiments were repeated:

| Variant | Training s | Images/s | Setup s | Test accuracy | Eval s | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `streaming_lif` | 277.59 | 180.12 | 2.39 | 60.54% | 19.41 | 2200 |
| `auto_lif` (current tuner) | 282.63 | 176.91 | 188.59 | 60.54% | 22.32 | 2860 |

Recommendation: **use `numba_stream` first**. In the matched four-variant run it was 2.24x faster than the reference and 1.15x faster than default Numba. Automatic selection found useful local settings but did not beat the streaming model end to end in the final repeat.

These are single-seed, single-epoch observations, not confidence intervals or convergence results. Thermal/load variation is substantial on this machine: the reference took 1042 s in an earlier run and 627 s in the final matched suite. Do not use the historical cross-round 3.65x ratio as a headline claim.

Setup includes dataset/model preparation, the activation-memory probe, JIT work encountered there, and automatic search where enabled. Training includes the first optimizer step and remaining data iteration, but excludes setup and evaluation. Latest setup + training is **279.99 s for streaming** versus **471.22 s for autotuning**. Reuse a validated tuning recipe for repeated workloads; per-layer speedups do not guarantee end-to-end gains even when search cost is excluded.

Saved tensor storage drops from 1374.50 MiB to 926.50 MiB for this model. This metric includes any tensors retained by autograd, including parameters, and is not peak activation memory. Peak RSS includes loader/runtime allocations and search allocations where applicable.

Raw local reports: `results/lossless_final_1epoch.json` (matched four-variant suite) and `results/lossless_calibrated_1epoch.json` (corrected tuner repeat). Generated results/data are git-ignored; these tables preserve the measured summary. The earlier cumulative epoch run was superseded before completion; its partial log is not a completed benchmark.

Design-loop history, all full epochs with the same seed and model recipe:

| Candidate / report | Training s | Setup s | Accuracy | Outcome |
| --- | ---: | ---: | ---: | --- |
| Original reference / `lossless_1epoch.json` | 1042.44 | 3.69 | 60.25% | Initial control, slower machine conditions. |
| Default Numba / same report | 417.29 | 2.38 | 60.51% | Initial fused baseline. |
| Streaming with layout conversion / same report | 435.57 | 2.29 | 60.51% | Training regression; improved inference/memory. |
| First autotune search / same report | 420.06 | 261.23 | 60.51% | No useful end-to-end training gain. |
| Layout-preserving streaming / `lossless_layout_1epoch.json` | 317.92 | 2.50 | 60.54% | Kept the implementation; repeated above. |
| Second search / same report | 285.60 | 195.10 | 60.54% | Local winner, not a consistent win on repetition. |

## Tested Strategies And Calls

| Strategy | How to call | Evidence and limits |
| --- | --- | --- |
| Reference autograd | `LIF(backend="loop")` | Full-epoch control. |
| Hand-derived BPTT in PyTorch | `accelerate(model, lif_backend="fused")` | Kernel/model equivalence tests; original 20-step run about 61 img/s. It reduces autograd nodes but is not one machine-code kernel. |
| Fused parallel CPU LIF | `accelerate(model, lif_backend="numba")` | Full epoch; independent neurons parallelized, time recurrence retained. |
| Broadcast-time input without materialization | `accelerate(model, lif_backend="numba_stream")` | Full epoch and expanded-input gradient tests; preserves dense physical layout. Does not itself enable encoder caching or change BN. |
| Inference without membrane-history writes | `LIF(backend="numba_stream")` inside `torch.inference_mode()` | Tested buffer omission and full test-set evaluation; gradients still save full-precision membrane history. |
| Cache-block size tuning | `LIF(backend="numba", block_size=256)` | Tested 256/2048/8192 tiles, incomplete tiles, T=1 and longer sequences. |
| Automatic CPU parallelism | `LIF(backend="numba", numba_threads=4)` | Searched 1/2/4/8 threads, limited to Numba's available pool. Thread mask restored after each forward/backward. |
| Per-layer automatic selection | `autotune_model(model, images)` | Five complete epoch evaluations across design rounds; accepts only validated candidates. Import from `soptim.autotune`. |
| Scalar-state temporal loops | `accelerate(model, lif_backend="numba_register")` | Full epochs alone/with compact pooling; useful candidate, not the kernel-search winner. |
| Compact exact pooling metadata | `accelerate(model, lif_backend="numba_stream", compact_pool=True)` | Full epochs, exact gradients and optimizer-step tests; lower saved storage, not a standalone training-speed win. |
| Channels-last model layout | `accelerate(model, channels_last=True)` | Original 20-step result about 107 img/s when stacked on Numba; not always faster. Also format input with `images.contiguous(memory_format=torch.channels_last)`. |
| Eval Conv-BN folding | `fuse_conv_bn(model.eval())` | Every controlled full run evaluated before/after fusion; latest streaming 19.41 -> 17.47 s, accuracy 60.54% both. Mutates model; use a copy if training will resume. FP32 rounding may change spikes near threshold. |
| Stateless temporal batching | `TimeFold(torch.nn.Conv2d(3, 32, 3, padding=1))` | Original 20-step model run about 51 img/s. Only safe when wrapped operations truly do not mix samples or mutate state; do not put training BN/dropout inside and call it lossless. |
| Encoder caching | `SpikingCNN(mode="multistep", cache_encoder=True)` | Original 20-step run about 50 img/s; fixed recipe in controlled suite, not a drop-in lossless training transform. |
| In-memory batch augmentation | `FastCIFAR10Loader("./data", True, 128, drop_last=False)` | Import from `soptim.data`; same augmentation distribution, different random draw/order and normalization rounding from torchvision. Original 20-step run about 51 img/s. |
| Fused Adam implementation | `torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)` | Original 20-step stacked run about 109 img/s; not validated as bitwise optimizer-state equivalent. |
| `torch.compile` | `accelerate(model, lif_backend="fused", compile=True)` | Attempted; unavailable locally without a C++ compiler. Do not expect opaque Numba/NumPy kernels to compile into one graph. |
| BF16 autocast | `with torch.autocast("cpu", torch.bfloat16): ...` | Attempted; backward unsupported by oneDNN on this CPU. Numerically approximate, not lossless. |
| Detached reset gradient | `accelerate(model, detach_reset=True)` | Original 20-step run about 65 img/s; changes BPTT, not guaranteed slightly lossy. |
| Fewer simulation steps | `SpikingCNN(T=2)` | Original 20-step run about 120 img/s; changes the model, not lossless. |

The old 20-step figures come from `results/smoke.json` and `results/smoke2.json`, with different cumulative stacks. They are **not comparable to the controlled full-epoch rows** and do not establish accuracy preservation. Several short runs scored 10% with immature/changed BN statistics; this is not a valid speed-at-equal-accuracy result. Those historical strategies have not all completed full epochs.

## Kernel Design Agent And Automatic Search

A Kernel Design Agent (KDA) is a feedback loop: propose a kernel, compile, compare with a trusted reference, measure, inspect failures/profiles, and revise. An assistant-led research/design loop was used here, including an independent read-only research agent. The reusable library component is a **bounded local autotuner**, not a connection to an external KDA service or an autonomous LLM code generator. No API key, cloud GPU, or unbounded background agent is required.

The tuner searches existing `loop`, `fused`, `numba`, `numba_stream`, and `numba_register` implementations, tile sizes, and thread counts. It validates the supplied shape plus seeded random and threshold probes, requires exact spikes and close input gradients, rejects missing gradients, warms candidates, and times them in shuffled order. Defaults are three rounds of five measurements per candidate and a 3% improvement margin. Fast-math, reduced precision, changed tau/T/reset, and surrogate substitutions are not search candidates.

```python
import torch
from soptim import SpikingCNN, accelerate
from soptim.autotune import autotune_model

model = accelerate(SpikingCNN(mode="multistep", cache_encoder=True), lif_backend="numba")
images = torch.randn(128, 3, 32, 32)
reports = autotune_model(model, images, block_sizes=(256, 2048, 8192), threads=(1, 2, 4, 8))
for name, report in reports.items():
    print(name, report["selected"], report["speedup"])
```

This mutates the selected LIF settings in place, not weights, BN buffers, or RNG state. The settings remain on each layer for subsequent calls. Training/evaluation mode is unchanged. Use `autotune_lif(neuron, sequence, grad=False)` for a standalone inference search. Tuning is CPU FP32 only and requires representative input shape/layout. Shared layers with several shapes currently use their first observed call; there is no per-shape runtime dispatch cache.

Persist/reapply a model's selected recipe without repeating the search:

```python
import json
from pathlib import Path
from soptim import accelerate

Path("tuning.json").write_text(json.dumps(reports, indent=2))
saved = json.loads(Path("tuning.json").read_text())
layers = dict(model.named_modules())
for name, report in saved.items():
    config = report["selected"]
    accelerate(layers[name], lif_backend=config["backend"],
               block_size=config["block_size"], numba_threads=config["numba_threads"])
```

Recipes are not in `state_dict`. Revalidate on different hardware, thread pools, torch/Numba versions, neuron parameters, shapes, or layouts. Changing batch size remains correct for these kernels but can invalidate the performance choice. Reports contain shapes, strides, parameters, versions, every candidate's checks/timings, and the selected configuration.

Earlier representative `[4,128,64,16,16]` searches, each with 27 candidates passing the probes before the scalar-state backend was added:

| Search | Default Numba ms | Selected ms | Observed kernel gain |
| --- | ---: | ---: | ---: |
| Forward + backward | 23.45 | 17.18 | 1.37x |
| Static expanded input, forward + backward (corrected calibration) | 34.90 | 19.06 | 1.83x |
| Inference only | 10.81 | 5.59 | 1.93x |

These are microbenchmarks, not model speedups. The corrected static-input search selected streaming at eight threads with 256-element blocks. Its earlier, superseded calibration measured 43.64 -> 22.50 ms (1.94x) but cloned/materialized the input before re-expansion and did not faithfully preserve the model's layout. That report remains in `results/kda_static.json`; the corrected one is `results/kda_static_calibrated.json`.

That earlier model search selected streaming at four threads with 2048-element blocks for the encoder; the other three layers selected ordinary Numba. Do not hard-code these hardware-specific choices as universal defaults. The repeated model runs are the acceptance criterion for performance claims.

## Reproduce

```bash
python -m soptim.autotune --out results/kda_search.json
python -m soptim.autotune --static-input --out results/kda_static.json
python -m soptim.autotune --inference --out results/kda_inference.json
python examples/cifar10_benchmark.py --out results/lossless_1epoch.json
python examples/cifar10_benchmark.py --only streaming_lif auto_lif --out results/layout_1epoch.json
python examples/cifar10_benchmark.py --only compile bf16_autocast --max-steps 1
```

The benchmark defaults to the four controlled variants and one full epoch per variant in separate processes. `--max-steps 20` is a smoke test, never a full epoch. Older recipes are available via `--only`; run `--help` for the complete list. JSON is saved after every strategy with completion, sample counts, setup costs, accuracy, and skip reasons. Run timed experiments sequentially on an otherwise idle machine.

## Further Lossless Candidates

These are researched candidates, **not implemented or claimed as tested speedups**:

| Candidate | Why try it / correctness constraint |
| --- | --- |
| Triton/CUDA/CuPy fused LIF and BPTT | Parallelize neurons while keeping each reset-dependent temporal recurrence ordered; compare against an established SNN implementation. Requires a GPU. |
| CUDA Graph replay | Reduce Python/launch overhead for static shapes and stable buffers; handle the final partial batch separately. |
| Generalize saved spike compression | Bool packing is now tested for the LM's post-LIF linear inputs. Bit packing and CNN convolution inputs remain untested. Never quantize membrane states or surrogate gradients under a lossless label. |
| Generalize activation recomputation | Block checkpointing is now tested for the BN-free LM. CNN/BN handling remains unimplemented; avoid double-updating running statistics or changing RNG. |
| Pinned-memory prefetch / transfer overlap | Hide host-to-device copies using streams and correct buffer lifetimes. CPU-only runs cannot validate this. |
| Chunked temporal batching | Tune memory versus stateless-operator throughput without mixing training BN populations; neuron state must carry across chunks exactly. |

Do not treat arbitrary time-axis parallelism, silent-neuron skipping during surrogate-gradient training, reciprocal substitution for arbitrary tau, fast-math, or a new optimizer as automatically lossless. Reset dependencies, nonzero gradients at silent neurons, and floating-point thresholds matter.

Research inspirations: [KernelBench](https://github.com/ScalingIntelligence/KernelBench) for correctness-gated generated-kernel evaluation; [CIFAR-10 Airbench](https://github.com/KellerJordan/cifar10-airbench) for batch augmentation and layout experiments; [Numba performance guidance](https://numba.readthedocs.io/en/stable/user/performance-tips.html) for parallel scheduling and fast-math caveats; [PyTorch Conv-BN recomputation tutorial](https://docs.pytorch.org/tutorials/intermediate/custom_function_conv_bn_tutorial.html) for the distinction between training-memory optimization and eval fusion. Speedrun changes to optimizers, data selection, augmentation, or model design are not lossless infrastructure changes.

Before a PyPI release: confirm package-name availability, review licensing, add a supported Python/PyTorch CI matrix, and validate GPU/device behavior. Nothing has been uploaded to PyPI.