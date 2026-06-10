# CuPy Integration Research — Results

_Generated from real runs on this machine. Reproduce with the commands in §8.
Raw data: `results/benchmark_*.json`, `results/speedup_comparison.png`,
`results/comparative_analysis.md`._

---

## 1. Executive Summary

We implemented and validated **12 runnable WSI patch-filtering data loaders** — a
mono-core CPU baseline (v0a), a multi-core CPU baseline (v0b), seven isolated
CuPy/GPU strategies (v1–v7), an 8 GB-card tuned version (v8), and the
**all-optimizations-combined v9 "Ultimate GPU"** — plus a full benchmarking,
validation, memory-profiling, and ablation harness. All share one interface and
one registry, so any version drops into every tool automatically.

**Headline finding (contradicts the v9 plan's 8–15× hypothesis):**

> v9 **is** the fastest GPU version (it beats v1–v8), confirming that the
> optimizations combine as designed. But **every GPU version — v9 included — is
> slower end-to-end than even the single-core CPU baseline**, and CPU
> multiprocessing (v0b) wins outright at **3.0×**. The kernel timer proves why:
> v9's actual GPU work is **52 ms — just 1.1 % of its 4.7 s runtime.** The other
> **98.9 % is OpenSlide patch decoding (I/O) + the Python read loop.** You cannot
> optimize your way past a wall the GPU never touches; v0b wins because it
> parallelizes the *I/O*, which is the real bottleneck.

| Question (from the v9 plan) | Answer (measured) |
|---|---|
| Is v9 the fastest GPU version? | **Yes** — 4.73 s, ahead of v8/v5/v4/v1/v7/v6/v2/v3 |
| Does v9 hit 8–15× over v0b? | **No — 0.28× of v0b.** The task is I/O-bound, not compute-bound |
| What is the real GPU ceiling here? | **52 ms** of kernel time for 121 patches; the GPU is essentially free |
| Which layer matters most? | **Pinned memory** (kernel 52 ms vs 67 ms). Async + early-exit ≈ noise |
| Does mixed precision help? | **No — it hurts.** fp16 kernel 90 ms vs 52 ms on this Pascal card |
| Best end-to-end approach | **v0b multiprocessing (3.0×)** — parallelizes the I/O bottleneck |

---

## 2. System Specifications

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1060, 3 GB (Pascal, **no fast FP16**: ~1/64 FP32) |
| CUDA / CuPy | CUDA 12.6 (torch cu126) / CuPy 14.1.1 |
| CPU | 8 logical cores (`multiprocessing.cpu_count() == 8`) |
| Python / OpenSlide / PyTorch | 3.12.13 / 1.4.3 / 2.12.0+cu126 |
| Test WSI | `data/S114-80954A-Her2(3+).tiff`, level-0 = 21504 × 23040 |
| Filter params | patch 1024, white>230, black<25, rejection_ratio 0.9 |
| Kernel timing | CUDA events around GPU work, summed per build (`ds.kernel_time`) |

---

## 3. Performance Results

### 3.1 End-to-end grid filtering (the real workload)

`stride=2048` ⇒ **121 candidate patches**, 32 kept; 3 iterations, warmup discarded.

| Version | min (s) | mean (s) | patch/s | vs v0a | vs v0b | Self-reported peak MB |
|---|---|---|---|---|---|---|
| **v0b_multi** | **1.317** | 1.347 | 91.9 | **3.01×** | 1.00× | – |
| v0a_mono | 3.961 | 3.999 | 30.5 | 1.00× | 0.33× | – |
| **v9_ultimate** | 4.732 | 4.787 | 25.6 | 0.84× | 0.28× | 402.7 |
| v5_async | 4.744 | 4.810 | 25.5 | 0.83× | 0.28× | – |
| v4_pinned | 4.755 | 4.770 | 25.4 | 0.83× | 0.28× | – |
| v8_4060 | 4.775 | 4.836 | 25.3 | 0.83× | 0.28× | 402.7 |
| v1_full | 4.836 | 4.929 | 25.0 | 0.82× | 0.27× | – |
| v7_memopt | 4.881 | 4.959 | 24.8 | 0.81× | 0.27× | 33.6 |
| v6_mixed | 5.284 | 5.308 | 22.9 | 0.75× | 0.25× | – |
| v2_batch | 5.375 | 5.386 | 22.5 | 0.74× | 0.24× | – |
| v3_hybrid | 5.474 | 5.635 | 22.1 | 0.72× | 0.24× | – |

**v9 is the fastest GPU version** — the layered design does extract the most from
the card. It is still 0.28× of v0b because the GPU is not where the time goes.

### 3.2 Kernel-time decomposition — the decisive insight

Same run, GPU time measured with CUDA events (transfer + compute), summed over
all batches:

| Version | total (s) | kernel (s) | kernel % | overhead (s) = I/O + Python |
|---|---|---|---|---|
| v1_full | 4.836 | 1.041 | 21.5 % | 3.794 |
| v2_batch | 5.375 | 0.175 | 3.3 % | 5.200 |
| v3_hybrid | 5.474 | 0.176 | 3.2 % | 5.298 |
| v4_pinned | 4.755 | 0.139 | 2.9 % | 4.617 |
| v6_mixed | 5.284 | 0.124 | 2.3 % | 5.160 |
| **v9_ultimate** | 4.732 | **0.052** | **1.1 %** | 4.680 |
| v8_4060 | 4.775 | 0.052 | 1.1 % | 4.723 |
| v7_memopt | 4.881 | 0.033 | 0.7 % | 4.848 |
| v5_async | 4.744 | 0.024 | 0.5 % | 4.720 |

- v1's 21.5 % kernel share is **per-patch transfer/launch overhead**, not real
  compute; batching (v2 → v9) collapses it to ~1 %.
- v9/v8/v5/v7 drive GPU time to **24–52 ms**. The pipeline still takes ~4.7 s,
  so **>98 % is patch-decode I/O**. This is the quantitative proof that GPU
  optimization cannot help this workload as structured.

### 3.3 Compute-only cost (I/O excluded) — where the GPU *does* win

One patch read once, compute timed 50×, GPU synchronized (`test_01_single_patch.py`):

| Patch size | CPU ms | GPU-int ms | GPU-int + transfer ms | fp16 ms | GPU-int speedup |
|---|---|---|---|---|---|
| 256  | 0.264 | 0.449 | 0.541 | 0.399 | 0.59× |
| 512  | 1.009 | 0.435 | 0.560 | 0.421 | 2.32× |
| 1024 | 6.370 | 0.912 | 1.194 | 0.545 | **6.98×** |
| 2048 | 29.868 | 3.568 | 4.691 | 2.023 | **8.37×** |

The GPU's compute advantage is real (up to 8.4×) and grows with patch size —
it's simply invisible behind I/O in the full pipeline.

### 3.4 v9 Ablation Study (`test_v9_ablation.py`)

`stride=2048`, batch=64. Total time is noisy (disk-cache variance); **kernel time
is the stable signal**. All non-fp16 configs stay bit-exact vs v0a.

**Leave-one-out (full v9 minus one layer):**

| Config | kernel (ms) | peak MB | Reading |
|---|---|---|---|
| FULL v9 (all layers) | 51.5 | 402.7 | reference |
| − async (serial) | 51.5 | **201.3** | async adds **2× memory, no end-to-end gain** |
| − pinned memory | **67.4** | 402.7 | pinned is the **one real GPU-time win** (−24 %) |
| − early-exit | 51.9 | 402.7 | negligible — trivial compute |
| + mixed precision (fp16) | **89.9** | 402.7 | fp16 **hurts** on Pascal (+75 % kernel) |

**Layer ranking for this workload:** pinned memory (Layer 1) > batching (Layer 3)
≫ async (Layer 2) ≈ early-exit (Layer 7) ≈ 0; mixed precision (Layer 4) is
negative on Pascal. The optimizations do **not** compound here — they saturate
almost immediately because there is only 52 ms of GPU work to optimize.

---

## 4. Memory Analysis (`test_04_memory_profiling.py`)

Peak CuPy-pool usage (sampler thread, `stride=4096`), plus self-reported peaks
for v7/v8/v9:

| Version | Peak MB | Note |
|---|---|---|
| v1_full | 28.3 | one patch on GPU at a time |
| v7_memopt | 33.6–41.9 | fused uint8 kernel + small chunks — **smallest batched** |
| v6_mixed | 503.3 | fp16 intermediate (½ of uint32) |
| v2/v3/v4 | 637.5 | uint32 grayscale temp = **4× the uint8 batch** |
| v5_async | 771.8 | double-buffered ⇒ ~2× resident |
| v8_4060 | 402.7 | bs=128, fused uint8 kernel (no uint32 temp) |
| v9_ultimate | 402.7 (201.3 serial) | bs=64 × 2 stream slots; serial mode halves it |

v7's fused-uint8 + small-chunk design is **~15× leaner than v2**. v9's async
double-buffer doubles its own footprint (402 vs 201 MB) for no end-to-end gain —
on a memory-constrained card, run v9 with `enable_async=False`.

**Leak check:** pool `used_bytes` returned to 0.00 MB after 3 repeated builds — no leak.

---

## 5. Correctness Validation (`validation_suite.py`)

1. **Grayscale fidelity.** The GPU integer-luma kernel
   `(R·19595 + G·38470 + B·7471 + 32768) >> 16` reproduces PIL `convert('L')`
   **bit-exactly** (max abs error 0). A plain channel mean drifts up to ~12 grey
   levels, so it was deliberately not used.
2. **Filtering agreement.** **All 11 non-baseline versions (v0b, v1–v9) select the
   identical kept-coordinate set, in identical order**, as v0a — Jaccard 1.0000,
   0 missing / 0 extra — at every stride tested.
3. **Mixed precision (v6 and v9-fp16).** fp16 grayscale differs from exact by ≤ 1
   grey level; on the test data no patch flipped. On boundary patches it could —
   the documented precision trade-off, not a bug.
4. **Transform pipeline.** `__getitem__` returns a `(3, 1024, 1024)` float32 tensor.

---

## 6. Conclusions & Recommendations

- **Profile before optimizing.** The kernel timer shows the GPU does its job in
  **52 ms**; the pipeline spends **4.7 s decoding patches**. The bottleneck is
  OpenSlide I/O — Phase 6 independently measures **84.6 %** of a mock training
  step as data loading.
- **Use v0b (CPU multiprocessing) for this task** — simplest fast option (3.0×),
  parallelizes the actual bottleneck. No GPU version beats it.
- **v9 answers its research question honestly:** it is the GPU ceiling (fastest
  GPU version, ~52 ms kernel, near-zero overhead beyond I/O), and that ceiling is
  still below the CPU because the problem is I/O-bound. "Combine all
  optimizations" works, but the optimizations **saturate, not compound**, when
  there is almost no compute to accelerate.
- **If you must use the GPU here, parallelize the I/O too.** None of v1–v9 read
  patches in parallel; the winning design is a multi-worker reader feeding GPU
  batches so disk decode overlaps across processes — exactly what v0b does for
  the CPU path.
- **GPU pays off only when per-patch compute is heavy** (≥1024px patches, stain
  normalization, convolutions, model inference) or when pixels already live on
  the GPU. Then §3.3's 6–8× compute win would surface end-to-end.
- **On Pascal, skip mixed precision** (v6 / v9-fp16): it raises kernel time and
  risks accuracy for no benefit.
- **On a small GPU, prefer v7's pattern** (fused uint8, small chunks, 34–42 MB),
  or run v9 with `enable_async=False` to halve its footprint.

### Decision matrix

| Scenario | Use |
|---|---|
| CPU-only box, or this exact filter | **v0b_multi** |
| Teaching the algorithm | v0_ultra_basic / v0a_mono |
| Max GPU throughput / benchmarking the ceiling | **v9_ultimate** |
| 8 GB card, batched filtering | v8_4060 |
| Heavy compute on a small (≤4 GB) GPU | v7_memopt, or v9 with `enable_async=False` |
| Latency-sensitive mixed small/large jobs | v3_hybrid |

---

## 7. Future Work

- Multiprocess **reader → GPU batch** pipeline (overlap parallel I/O with GPU) —
  the only thing that could make a GPU version beat v0b here.
- Re-run v9 on a Volta+ GPU: fast FP16 should flip the mixed-precision result,
  and higher PCIe bandwidth would shrink the (already tiny) transfer share.
- Push a heavier per-patch op (stain norm / small CNN) onto the GPU, where
  §3.3's compute advantage should finally dominate end-to-end.

---

## 8. Reproduce

```bash
# Correctness gate (all 12 versions vs v0a)
python validation_suite.py --stride 2048

# Phase 0: baseline hierarchy + validation
python test_00_baseline_comparison.py --stride 1024 --iterations 3

# Full comparative benchmark (+ kernel-time decomposition) -> results/*.json + png
python benchmark_runner.py --stride 2048 --iterations 3 --versions all
python comparative_analysis.py

# v9 ablation: per-layer contribution + progressive stack
python test_v9_ablation.py --stride 2048 --iterations 3 --batch 64

# Individual phases
python test_01_single_patch.py --repeats 50           # compute-only scaling
python test_02_batch_filtering.py --strides 4096,2048,1024
python test_03_end_to_end_dataloader.py --stride 1024 --n 100
python test_04_memory_profiling.py --stride 2048      # GPU peak memory + leak check
python test_05_scalability.py --sizes 256,512,1024 --stride-factor 1.0
python test_06_integration_training.py --stride 1024 --iters 20 --batch 4
```

All scripts accept `--wsi <path>`; the default is the smallest slide for fast
iteration. Restrict versions with `--versions v0b_multi,v9_ultimate` etc.
