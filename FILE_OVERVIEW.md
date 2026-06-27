# Project File Overview — WSI Patch-Filtering, CPU → GPU Study

This repository is a **research study on parallel/accelerated computing**: it takes one
fixed task — *scan a gigapixel Whole-Slide Image (WSI) with a sliding window and keep
only the tissue patches* — and re-implements it ~30 times, each version moving more of
the work onto the GPU or parallelising a different stage. Every version is measured
against a single-threaded CPU baseline (**v0a = 1.0×**), and every version must reproduce
v0a's *exact* kept-patch set (the correctness gate).

The task itself never changes. What changes between files is **where the work runs and
how it is scheduled**: CPU vs GPU, mono-core vs multi-core feed, per-patch vs batched,
synchronous vs pipelined, CPU libjpeg vs nvJPEG vs a hand-written CUDA decoder.

---

## 1. The core algorithm (shared by every version)

For each window position the loader reads the patch, converts to grayscale using PIL's
**exact integer ITU-R 601 luma**:

```
L = (R*19595 + G*38470 + B*7471 + 32768) >> 16
```

then counts white pixels (`L > 230`) and black pixels (`L < 25`). A patch is **kept**
if neither ratio exceeds `0.9`. Reproducing this formula bit-for-bit on the GPU is what
lets every version match the CPU baseline's kept coordinates.

---

## 2. Library data loaders (the original, non-numbered code)

| File | Role |
|------|------|
| [data_loader.py](data_loader.py) | The original production loader: multiprocessing `Pool`, one OpenSlide handle per worker. Source of the v0b baseline. |
| [data_loader_mono.py](data_loader_mono.py) | Single-threaded version of the same class. Source of the v0a baseline. |
| [data_loader_640.py](data_loader_640.py) | Thin subclass defaulting `patch_size`/`stride` to 640 (for DenseNet121 experiments). No algorithm change. |

---

## 3. CPU baselines (the reference points)

| Version | File | What it isolates |
|---------|------|------------------|
| `v0_ultra` | [data_loader_v0_ultra_basic.py](data_loader_v0_ultra_basic.py) | The simplest possible loader — no error handling, no optimization. Teaching scaffold only. |
| **`v0a_mono`** | [data_loader_v0a_mono_baseline.py](data_loader_v0a_mono_baseline.py) | **The 1.0× functional reference.** Single core, sequential read+filter. |
| **`v0b_multi`** | [data_loader_v0b_multi_baseline.py](data_loader_v0b_multi_baseline.py) | "How far pure CPU parallelism gets us": same code across all cores via `multiprocessing.Pool`. |

The only behavioural change v0a/v0b made versus the original library files is a
`verbose=False` flag, so per-patch print lines don't pollute the timings.

---

## 4. CuPy GPU series — v1…v10 (move the *filter* to the GPU)

These keep the **CPU JPEG decode** (OpenSlide) but progressively move filtering to the
GPU and optimize data movement. Each version adds exactly one idea on top of the last.

| Version | File | The one new idea vs the prior version |
|---------|------|----------------------------------------|
| `v1_full` | [data_loader_v1_cupy_full.py](data_loader_v1_cupy_full.py) | Naive: transfer **one patch at a time** to the GPU. Demonstrates that launch/transfer overhead swamps tiny work. |
| `v2_batch` | [data_loader_v2_cupy_batch.py](data_loader_v2_cupy_batch.py) | **Batch N patches** into one transfer + one set of kernels. Where the GPU starts to pay off. |
| `v3_hybrid` | [data_loader_v3_cupy_hybrid.py](data_loader_v3_cupy_hybrid.py) | **Smart routing**: small chunks run on CPU, big chunks on GPU (avoid paying GPU overhead for tiny jobs). |
| `v4_pinned` | [data_loader_v4_cupy_pinned_memory.py](data_loader_v4_cupy_pinned_memory.py) | **Pinned (page-locked) host buffer** + reused device buffer → faster DMA, no per-batch malloc. |
| `v5_async` | [data_loader_v5_cupy_async.py](data_loader_v5_cupy_async.py) | **Double-buffered CUDA streams**: read batch *k+1* while the GPU computes batch *k*. |
| `v6_mixed` | [data_loader_v6_cupy_mixed_precision.py](data_loader_v6_cupy_mixed_precision.py) | **fp16 luma** — exposes both the accuracy trade-off and that fp16 is *slower* on Pascal cards. |
| `v7_memopt` | [data_loader_v7_cupy_memory_optimized.py](data_loader_v7_cupy_memory_optimized.py) | **Fused uint8 luma kernel** + small chunks → tiny VRAM footprint (for the 3 GB card). |
| `v8_4060` | [data_loader_v8_cupy_optimized_4060.py](data_loader_v8_cupy_optimized_4060.py) | **Threaded OpenSlide readers** (GIL released during decode) + big batches. First attack on the *feed* side. |
| `v9_ultimate` | [data_loader_v9_ultimate_gpu.py](data_loader_v9_ultimate_gpu.py) | **Stack all 7 layers** into one config-driven loader (each toggleable for ablation). |
| `v10_par_io` | [data_loader_v10_parallel_io_gpu.py](data_loader_v10_parallel_io_gpu.py) | The **data-driven turning point**: `hardware_probe.py` showed the GPU is *starved*, not slow — decode is the bottleneck. v10 uses a reader-thread pool feeding the GPU filter; optimizing the kernel (v9) optimized the wrong thing. |

---

## 5. GPU-decode series — v11…v22 (move the *decode* to the GPU)

The key realisation: **CPU JPEG decode is ~all the wall time**, not the filter. These
Philips slides store level-0 as baseline-JPEG tiles, so the decode itself can go on the
GPU. These versions parse the TIFF directly, read raw compressed tiles, and decode on
the GPU (either nvJPEG or a hand-written CUDA kernel).

| Version | File | Decode device | Feed | New idea |
|---------|------|---------------|------|----------|
| `v11_gpudec` | [data_loader_v11_gpu_decode_5090.py](data_loader_v11_gpu_decode_5090.py) | **nvJPEG** | threaded | First GPU-decode: parse TIFF tags, splice shared JPEG tables, batch-decode via `torchvision.io.decode_jpeg(device='cuda')`. |
| `v12_dec_mono` | [data_loader_v12_gpu_decode_mono.py](data_loader_v12_gpu_decode_mono.py) | nvJPEG | mono | Ablation cell: *mono × specialized decoder*. |
| `v13_dec_multi` | [data_loader_v13_gpu_decode_multi.py](data_loader_v13_gpu_decode_multi.py) | nvJPEG | multi | Ablation cell: *multi × specialized*. v13 vs v12 = read-parallelism gain. |
| `v14_cmp_mono` | [data_loader_v14_gpu_compute_mono.py](data_loader_v14_gpu_compute_mono.py) | **custom CUDA (naive)** | mono | *mono × raw-CUDA*: decode on general CUDA cores, one thread per tile, no nvJPEG. |
| `v15_cmp_multi` | [data_loader_v15_gpu_compute_multi.py](data_loader_v15_gpu_compute_multi.py) | custom CUDA (naive) | multi | *multi × raw-CUDA*. |
| `v16_opt_mono` | [data_loader_v16_cuda_opt_mono.py](data_loader_v16_cuda_opt_mono.py) | **custom CUDA (optimized)** | mono | Same as v14 but the *optimized* decoder (bit-identical output, more throughput). |
| `v17_opt_multi` | [data_loader_v17_cuda_opt_multi.py](data_loader_v17_cuda_opt_multi.py) | custom CUDA (optimized) | multi | v16 + threaded reads. |
| `v18_ult_mono` | [data_loader_v18_cuda_ultimate_mono.py](data_loader_v18_cuda_ultimate_mono.py) | **custom CUDA (ultimate)** | mono | v16 kernel + pinned memory + double-buffered async-stream pipeline + reused buffer pool. |
| `v19_ult_multi` | [data_loader_v19_cuda_ultimate_multi.py](data_loader_v19_cuda_ultimate_multi.py) | custom CUDA (ultimate) | "multi" | v18 + a reader thread pool — but the pool **shared one file object + lock**, so reads were actually serial. |
| `v20_ult_pread` | [data_loader_v20_cuda_ultimate_pread.py](data_loader_v20_cuda_ultimate_pread.py) | custom CUDA (ultimate) | true parallel | Fixes v19 with **`os.pread`** (atomic positional reads, no lock) → genuinely parallel reads. |
| `v21_ult_pipe` | [data_loader_v21_cuda_ultimate_pipeline.py](data_loader_v21_cuda_ultimate_pipeline.py) | custom CUDA (ultimate) | producer/consumer | Moves reads **off the critical path**: a dedicated producer thread prefetches batches so the GPU never blocks on I/O. |
| `v22_par_destuff` | [data_loader_v22_cuda_parallel_destuff.py](data_loader_v22_cuda_parallel_destuff.py) | custom CUDA (ultimate) | producer/consumer | Also moves the **JPEG byte-destuffing** into the reader threads — the last serial host cost — so the pipeline is bound only by GPU decode. The fastest version. |

**The v19 → v20 → v21 → v22 arc is the most instructive part of the study**: each step
removes a hidden serialisation (shared-fd lock → reads on the main thread → destuff on
the main thread) that the previous "parallel" version still had.

---

## 6. The `dec-vN` remapping series — v23…v32

A pedagogical re-telling: these take the **conceptual ideas of v1…v10** and re-express
them in the GPU-decode world, to show the same optimization principles apply regardless
of where decode happens. Several are thin aliases pointing at an existing version.

| Version | File | Mirrors concept | Notes |
|---------|------|-----------------|-------|
| `v23_dec_v1` | [data_loader_v23_dec_v1_naive.py](data_loader_v23_dec_v1_naive.py) | v1 (per-patch) | `batch_size` hardcoded to **1** → one kernel launch per tile. Slowest GPU path, on purpose. |
| `v24_dec_v2` | [data_loader_v24_dec_v2_batch.py](data_loader_v24_dec_v2_batch.py) | v2 (batch) | Batches tiles; `batch_size=0` auto-sizes from free VRAM. |
| `v25_dec_v3` | [data_loader_v25_dec_v3_hybrid.py](data_loader_v25_dec_v3_hybrid.py) | v3 (hybrid) | GPU path + CPU/OpenSlide fallback when too few tiles or too little VRAM. |
| `v26_dec_v4` | [data_loader_v26_dec_v4_pinned.py](data_loader_v26_dec_v4_pinned.py) | v4 (pinned) | Makes the upstream read-staging buffer pinned too (end-to-end pinned). |
| `v27_dec_v5` | [data_loader_v27_dec_v5_async.py](data_loader_v27_dec_v5_async.py) | v5 (async) | Thin re-export of **v18** (double-buffered streams are already v18). |
| `v28_dec_v6` | [data_loader_v28_dec_v6_fp16.py](data_loader_v28_dec_v6_fp16.py) | v6 (fp16) | Casts the decoded Y plane to fp16 for the count. |
| `v29_dec_v7` | [data_loader_v29_dec_v7_membudget.py](data_loader_v29_dec_v7_membudget.py) | v7 (mem budget) | Explicit `vram_budget_gb`, frees the pool between chunks, tracks peak. |
| `v30_dec_v8` | [data_loader_v30_dec_v8_threaded.py](data_loader_v30_dec_v8_threaded.py) | v8 (threaded reads) | `os.pread` parallel reads + large batch. |
| `v31_dec_v9` | [data_loader_v31_dec_v9_combined.py](data_loader_v31_dec_v9_combined.py) | v9 (combine all) | Maps to **v22** (everything stacked). |
| `v32_dec_v10` | [data_loader_v32_dec_v10_pipeline.py](data_loader_v32_dec_v10_pipeline.py) | v10 (producer/consumer) | Maps to **v21** (prefetch pipeline). |

---

## 7. The hand-written CUDA JPEG decoders

These are the engines the v14–v22 loaders call. All three produce **bit-identical**
output; they differ only in throughput.

| File | What it is |
|------|------------|
| [gpu_jpeg_decoder.py](gpu_jpeg_decoder.py) | **Naive** baseline-JPEG decoder as a CuPy `RawKernel`. One CUDA thread decodes one whole tile (Huffman → dequant → IDCT → plane). Measures the parallel-across-tiles speedup, not a tuned decoder. |
| [gpu_jpeg_decoder_optimized.py](gpu_jpeg_decoder_optimized.py) | **Optimized**: register bit-buffer + 8-bit Huffman LUT, all hot tables in `__constant__` memory, DC-only IDCT skip, and a **fused YCbCr→luma→count** kernel that never materialises an RGB buffer (so a 3 GB card can run big batches). |
| [gpu_jpeg_decoder_ultimate.py](gpu_jpeg_decoder_ultimate.py) | **Ultimate**: the optimized kernel + pinned host staging + double-buffered streams + a reused two-slot buffer pool. `submit()`/`fetch()` drive the async pipeline for v18/v19. |

---

## 8. Benchmark, analysis & profiling tooling

| File | Purpose |
|------|---------|
| [test_performance_framework.py](test_performance_framework.py) | **The shared core.** Holds `IMPLEMENTATIONS` (the single registry mapping every version key → module + per-card kwargs), plus `PerformanceMetrics`, `CuPyTester`, `ValidationChecker`, `BenchmarkRunner`. Import-safe (touches no GPU until you run a version). |
| [benchmark_runner.py](benchmark_runner.py) | Time any set of versions, write `results/benchmark_<ts>.json`, optional speedup plot. |
| [benchmark_gpu_decode.py](benchmark_gpu_decode.py) | Benchmark focused on the GPU-decode series (v0a/v0b, v8, v11–v32). |
| [benchmark_decode_ablation.py](benchmark_decode_ablation.py) | The controlled **2×2 ablation**: {mono, multi} feed × {CPU, general-GPU, nvJPEG} decode, plus pure-CPU baselines. |
| [comparative_analysis.py](comparative_analysis.py) | Consolidate many benchmark JSONs into one comparison matrix + plot + a markdown snippet. |
| [plot_benchmark.py](plot_benchmark.py) | Visualise one `benchmark_gpu_decode` JSON (per-stage stacked bars). |
| [hardware_probe.py](hardware_probe.py) | **The pivotal diagnostic**: dumps GPU properties, measures PCIe bandwidth, kernel throughput, and OpenSlide read rate, then gives a roofline verdict. This is what proved the GPU was *starved* and motivated v10+. |
| [numerical_validation.py](numerical_validation.py) | Pure-data correctness primitives: compare kept-coordinate sets, grayscale fidelity, the PIL-exact reference luma. No GPU dependency. |
| [validation_suite.py](validation_suite.py) | End-to-end correctness gate: grayscale fidelity + filtering agreement (precision/recall/Jaccard) + transform-pipeline shape check. |

---

## 9. The phased test suite

A Phase 0→6 progression that builds the argument from "the speedup is real" up to "does
it matter in real training?"

| File | Phase / question |
|------|------------------|
| [test_00_baseline_comparison.py](test_00_baseline_comparison.py) | Phase 0 — proves v0a→v0b→best-GPU hierarchy is real *and* correctness-preserving. The gate. |
| [test_01_single_patch.py](test_01_single_patch.py) | Phase 1 — isolates pure filter compute (no I/O); CPU vs GPU vs patch size. |
| [test_02_batch_filtering.py](test_02_batch_filtering.py) | Phase 2 — full grid build at several strides; finds the GPU break-even point. |
| [test_03_end_to_end_dataloader.py](test_03_end_to_end_dataloader.py) | Phase 3 — the retrieval path (`__getitem__`, DataLoader workers). |
| [test_04_memory_profiling.py](test_04_memory_profiling.py) | Phase 4 — GPU memory per version; peak pool, bytes/candidate, leak check. |
| [test_05_scalability.py](test_05_scalability.py) | Phase 5 — how time/throughput scale with patch size. |
| [test_06_integration_training.py](test_06_integration_training.py) | Phase 6 — mock training loop: what fraction of a step is data loading. The bottom line. |
| [test_v9_ablation.py](test_v9_ablation.py) | v9-specific ablation: toggle each of the 7 layers, measure its contribution. |

---

## 10. Documentation & reports

| File | Contents |
|------|----------|
| [DATA_LOADER_DOCUMENTATION.md](DATA_LOADER_DOCUMENTATION.md) | Reference for the loader API / versions. |
| [CUPY_RESEARCH_PLAN.md](CUPY_RESEARCH_PLAN.md) | The v1–v9 CuPy optimization plan (the 7 layers). |
| [V9_ULTIMATE_GPU_PLAN.md](V9_ULTIMATE_GPU_PLAN.md) | Design plan for the v9 "stack everything" loader. |
| [V9_RESEARCH_POSITION.md](V9_RESEARCH_POSITION.md) | Where v9 sits in the research narrative. |
| [GPU_OPTIMIZATION_REPORT.md](GPU_OPTIMIZATION_REPORT.md) | Results write-up of the GPU optimization work. |
| [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md) | Consolidated benchmark results. |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | High-level summary of what was built. |
| [REPORT_decode_ablation.md](REPORT_decode_ablation.md) | Findings from the 2×2 decode ablation. |
| [REPORT_ultimate_v18_v19.md](REPORT_ultimate_v18_v19.md) | Findings on the ultimate v18/v19 decoders. |

---

## 11. Data & artifacts

- **[data/](data/)** — 5 Philips Her2 WSI `.tiff` slides (the test set; the default
  benchmark slide is the smallest). Git-ignored.
- **[results/](results/)** — timestamped `benchmark_*.json` runs and generated `.png`
  plots.

---

## Reading order suggestion

1. **v0a** ([data_loader_v0a_mono_baseline.py](data_loader_v0a_mono_baseline.py)) — understand the task.
2. **v1 → v2** — see why batching matters on a GPU.
3. **[hardware_probe.py](hardware_probe.py)** — the data that reframed the whole project (decode, not filter, is the bottleneck).
4. **v11** — the first GPU-decode version.
5. **v19 → v20 → v21 → v22** — the parallel-I/O debugging arc, the technical heart of the study.

---

## Note on the current working-tree changes

Three files are modified vs the last commit (`git status`): tuning/iteration on the
benchmark harness — [benchmark_gpu_decode.py](benchmark_gpu_decode.py),
[benchmark_runner.py](benchmark_runner.py), and
[test_performance_framework.py](test_performance_framework.py). Run `git diff` on these
for the exact deltas.
