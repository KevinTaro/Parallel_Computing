# CuPy Integration Research — Results

_Generated from real runs on this machine. Reproduce with the commands in
§8. Raw data: `results/benchmark_*.json`, `results/speedup_comparison.png`._

---

## 1. Executive Summary

We implemented and validated **10 runnable WSI patch-filtering data loaders** — a
mono-core CPU baseline (v0a), a multi-core CPU baseline (v0b), and seven CuPy/GPU
strategies (v1–v7) — plus a full benchmarking, validation, and reporting harness.

**Headline finding (and it contradicts the plan's hypotheses):**

> For this workload, **CPU multiprocessing (v0b) is the fastest end-to-end
> approach, and every GPU version is *slower* than even the single-core
> baseline.** The filtering computation is trivial relative to the cost of
> decoding each patch from the WSI with OpenSlide. The pipeline is **I/O-bound**,
> so moving the compute to the GPU adds transfer overhead without removing the
> real bottleneck. CPU multiprocessing wins because it parallelizes the *I/O*.

This is not a failure of the GPU code — it is the correct, measured answer to the
research question "when does GPU help here?". The compute *is* much faster on the
GPU once you isolate it (up to **8.4×** at 2048px, §3.2), but that win is invisible
behind patch-decode I/O in the real pipeline.

| Question | Answer (measured) |
|---|---|
| v0b speedup over v0a | **3.31×** (8 logical cores, sublinear ⇒ I/O contention) |
| Best GPU version end-to-end | none beat v0b; best GPU ≈ **0.85× of v0a** (slower) |
| GPU break-even on compute-only | patch ≥ **512px** (256px: 0.59×, 1024px: 7.0×, 2048px: 8.4×) |
| fp16 (v6) accuracy cost | ≤ **1 grey level**; 0 boundary patches flipped on test data |
| Most memory-efficient GPU version | **v7** at 42 MB vs v2's 637 MB (**~15× less**) |
| Training data-loading fraction | **84.6%** of wall-clock ⇒ optimize the loader, not the math |

---

## 2. System Specifications

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1060, 3 GB (Pascal, **no fast FP16**: ~1/64 FP32) |
| CUDA / CuPy | CUDA 12.6 (torch cu126) / CuPy 14.1.1 |
| CPU | 8 logical cores (`multiprocessing.cpu_count() == 8`) |
| Python | 3.12.13 |
| OpenSlide | 1.4.3 |
| PyTorch | 2.12.0+cu126 |
| Test WSI | `data/S114-80954A-Her2(3+).tiff`, level-0 = 21504 × 23040 |
| Filter params | patch 1024, white>230, black<25, rejection_ratio 0.9 |

---

## 3. Performance Results

### 3.1 End-to-end grid filtering (the real workload)

`stride=2048` ⇒ **121 candidate patches**, 32 kept; 3 iterations, warmup discarded.

| Version | min (s) | mean (s) | patch/s | vs v0a | vs v0b | Peak MB |
|---|---|---|---|---|---|---|
| **v0b_multi** | **1.002** | 1.002 | 120.8 | **3.31×** | 1.00× | – |
| v0a_mono | 3.314 | 3.322 | 36.5 | 1.00× | 0.30× | – |
| v5_async | 3.884 | 3.931 | 31.2 | 0.85× | 0.26× | – |
| v7_memopt | 3.942 | 3.993 | 30.7 | 0.84× | 0.25× | 33.6 |
| v4_pinned | 3.964 | 3.990 | 30.5 | 0.84× | 0.25× | – |
| v1_full | 4.094 | 4.113 | 29.6 | 0.81× | 0.24× | – |
| v3_hybrid | 4.329 | 4.392 | 28.0 | 0.77× | 0.23× | – |
| v6_mixed | 4.334 | 4.344 | 27.9 | 0.76× | 0.23× | – |
| v2_batch | 4.368 | 4.397 | 27.7 | 0.76× | 0.23× | – |

GPU versions cluster around 0.76–0.85× of v0a because each runs in a **single
process**: it serializes patch-decode I/O *and* pays host→device transfer. v0b
spreads the I/O across 8 cores and does nothing else.

### 3.2 Compute-only cost (I/O excluded) — where the GPU actually wins

One patch read once, compute timed 50×, GPU synchronized. (`test_01_single_patch.py`)

| Patch size | CPU ms | GPU-int ms | GPU-int + transfer ms | fp16 ms | GPU-int speedup | fp16 max err |
|---|---|---|---|---|---|---|
| 256  | 0.264 | 0.449 | 0.541 | 0.399 | 0.59× | 1.0 |
| 512  | 1.009 | 0.435 | 0.560 | 0.421 | 2.32× | 1.0 |
| 1024 | 6.370 | 0.912 | 1.194 | 0.545 | **6.98×** | 1.0 |
| 2048 | 29.868 | 3.568 | 4.691 | 2.023 | **8.37×** | 1.0 |

The GPU's compute advantage **grows with patch size** and crosses break-even at
~512px. The `+transfer` column shows the host→device tax (~0.1–1.1 ms) that
batching (v2+) is designed to amortize. fp16 is fastest in raw compute here only
because the kernels are tiny and bandwidth-bound; see §5 for why it does not help
end-to-end on Pascal.

### 3.3 Throughput vs candidate count (`test_02`) and patch size (`test_05`)

- Increasing candidate count (stride 8192→4096) raised **v0b** throughput
  (49.8→91.1 patch/s, more parallel work) but left GPU versions flat (~30 patch/s)
  — no end-to-end break-even is reached because I/O dominates.
- Across patch sizes 256/512px, v0b stayed fastest (690/301 patch/s) and v2 slowest
  (377/119 patch/s), consistent with §3.1.

---

## 4. Memory Analysis (`test_04_memory_profiling.py`)

Peak CuPy-pool usage sampled by a background poller during grid creation
(`stride=4096`, 36 candidates):

| Version | Peak MB | MB / patch | Note |
|---|---|---|---|
| v1_full | 28.3 | 0.79 | one patch on GPU at a time |
| v6_mixed | 503.3 | 13.98 | fp16 intermediate (½ of uint32) |
| v2_batch | 637.5 | 17.71 | **uint32 grayscale temp = 4× the uint8 batch** |
| v3_hybrid | 637.5 | 17.71 | same batch path as v2 |
| v4_pinned | 637.5 | 17.71 | + reusable pinned/device buffers |
| v5_async | 771.8 | 21.44 | **double-buffered** ⇒ ~2× resident |
| **v7_memopt** | **41.9** | **1.16** | fused uint8 kernel + small chunks |

**v7 uses ~15× less peak memory than v2** by eliminating the uint32 batch
temporary (a fused `ElementwiseKernel` writes uint8 grayscale directly) and
bounding residency with small chunks — the right design for the 3 GB card.

**Leak check:** pool `used_bytes` returned to 0.00 MB after each of 3 repeated
builds ⇒ no leak; v4/v5/v7 free their buffers explicitly.

---

## 5. Correctness Validation (`validation_suite.py`)

1. **Grayscale fidelity.** The GPU integer-luma kernel
   `(R·19595 + G·38470 + B·7471 + 32768) >> 16` reproduces PIL `convert('L')`
   **bit-exactly** (max abs error 0 over all sampled patches). A naive channel
   mean — what the plan's pseudocode suggested — drifts by up to ~12 grey levels,
   so it was deliberately *not* used.
2. **Filtering agreement.** All 9 non-baseline versions selected the **identical
   kept-coordinate set, in identical order**, as v0a (Jaccard = 1.0000,
   missing = extra = 0) at multiple strides.
3. **Mixed precision (v6).** fp16 grayscale differs from exact by ≤ 1 grey level;
   on the test data **no patch flipped** its keep/discard decision. On other data
   a boundary patch could flip — that is the documented precision trade-off, not a
   bug. v6 is also **not faster end-to-end** because Pascal lacks fast FP16.
4. **Transform pipeline.** `__getitem__` returns a `(3, 1024, 1024)` float32
   tensor with the expected normalization.

---

## 6. Conclusions & Recommendations

- **Know your bottleneck first.** WSI patch filtering here is dominated by
  OpenSlide patch decoding (I/O), not by the grayscale/threshold math. Profiling
  (Phase 6) shows **84.6%** of a mock training step is data loading.
- **Use v0b (CPU multiprocessing) for this task.** It is the simplest fast option
  (3.31× over mono) and parallelizes the actual bottleneck.
- **GPU pays off only when compute dominates.** The GPU filter is genuinely
  6–8× faster *as compute* (≥1024px), so GPU acceleration becomes attractive if
  (a) patches are large, (b) the per-patch operation is heavier than a threshold
  count (e.g. stain normalization, convolutions, model inference), or (c) patch
  pixels are already resident on the GPU.
- **If you must use the GPU here, combine it with parallel I/O.** None of v1–v7
  parallelize disk reads; a future version should feed a multi-worker reader into
  GPU batches so I/O and compute overlap across processes, not just streams.
- **For a memory-constrained GPU, v7's pattern is the template:** fused uint8
  kernels, small chunks, explicit pool cleanup — 42 MB vs 637 MB.
- **Skip mixed precision on pre-Volta GPUs.** v6 adds an accuracy risk with no
  speed benefit on this Pascal card.

### Decision matrix

| Scenario | Use |
|---|---|
| CPU-only box, or this exact filter | **v0b_multi** |
| Teaching the algorithm | v0_ultra_basic / v0a_mono |
| Large patches or heavy per-patch compute, ample GPU RAM | v2_batch / v4_pinned |
| Heavy compute on a small (≤4 GB) GPU | **v7_memopt** |
| Latency-sensitive mixed small/large jobs | v3_hybrid |
| Slow disk, want I/O hidden behind compute | v5_async |

---

## 7. Future Work

- Multiprocess **reader → GPU batch** pipeline (overlap parallel I/O with GPU).
- Benchmark on a Volta+ GPU to see whether fast FP16 makes v6 competitive.
- Push a heavier per-patch operation (stain norm / tiny CNN) onto the GPU, where
  the §3.2 compute advantage should finally surface end-to-end.
- Test on the larger slides in `data/` and across more rejection thresholds.

---

## 8. Reproduce

```bash
# Correctness gate (fast)
python validation_suite.py --stride 4096

# Phase 0: baseline hierarchy + validation
python test_00_baseline_comparison.py --stride 2048 --iterations 3

# Full comparative benchmark -> results/benchmark_*.json + speedup plot
python benchmark_runner.py --stride 2048 --iterations 3 --versions all
python comparative_analysis.py          # consolidates latest JSON -> md + png

# Individual phases
python test_01_single_patch.py --repeats 50           # compute-only scaling
python test_02_batch_filtering.py --strides 8192,4096,2048
python test_03_end_to_end_dataloader.py --stride 2048 --n 100
python test_04_memory_profiling.py --stride 2048      # GPU peak memory + leak check
python test_05_scalability.py --sizes 256,512,1024
python test_06_integration_training.py --stride 2048 --iters 20 --batch 4
```

All scripts accept `--wsi <path>` to target a different slide (default is the
smallest one for fast iteration).
