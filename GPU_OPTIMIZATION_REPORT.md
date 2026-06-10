# GPU Optimization Report — GTX 1060 3GB

_Data-driven analysis of the graphics card's real capabilities and the actual
bottleneck in WSI patch filtering. Reproduce every number with
`python hardware_probe.py`._

---

## 1. The card: full spec sheet (measured from the CUDA runtime)

| Property | Value | Why it matters here |
|---|---|---|
| Name / arch | GeForce GTX 1060 3GB, **sm_61** (Pascal) | dictates which features help |
| VRAM | 3.14 GB (≈2.5 GB free) | caps batch size; favors fused uint8 kernels |
| SMs × cores | 9 × 128 = 1152 FP32 cores | tiny GPU, but plenty for this task |
| Core clock | 1.734 GHz | — |
| **FP32 peak** | **3.99 TFLOP/s** | compute is not the limit |
| **FP16 peak** | **~62 GFLOP/s (1/64 of FP32)** | **mixed precision HURTS on Pascal** |
| Memory clock / bus | 4.004 GHz / 192-bit | — |
| **Memory bandwidth** | **192 GB/s** (theoretical) | the relevant ceiling for byte work |
| L2 cache | 1.5 MB | — |
| **Async copy engines** | **2** | H2D and D2H *can* overlap (Layer 2) |
| Concurrent kernels | Yes | multiple streams *can* run together |
| Max threads/block, /SM | 1024, 2048 | full occupancy easily reached |

**Takeaway:** for grayscale + threshold counting (a few bytes of arithmetic per
pixel) this is a **memory-bound** problem on a card with 192 GB/s of bandwidth.
There is no compute wall to optimize against — and FP16 is actively bad here.

---

## 2. Empirical performance (not datasheet — measured on this box)

### PCIe transfer bandwidth
| Transfer | GB/s |
|---|---|
| H2D pageable | 8.2 |
| **H2D pinned** | **11.7** (+43%) |
| D2H pageable | 8.1 |
| **D2H pinned** | **12.0** (+48%) |

→ Pinned memory is the one transfer optimization that pays off. PCIe tops out
~12 GB/s (Gen3 x16-class), well under the 192 GB/s on-card bandwidth.

### Luma kernel throughput (data already on GPU)
- batch 32 × 1024²: **1.83 ms/call → 17,517 patches/sec**, ~73 GB/s effective.
- The GPU can filter **~17.5k patches/sec**. Hold that number.

### OpenSlide decode rate — the real bottleneck
| Mode | patches/sec | speedup |
|---|---|---|
| single-thread | **27.3** | 1.00× |
| threads × 4 | 77.2 | 2.83× |
| threads × 8 | 92.9 | 3.41× |
| processes × 4 | 80.3 | 2.94× |
| processes × 8 | 94.3 | 3.46× |

→ Decoding one 1024² patch costs **36.7 ms**. **OpenSlide releases the GIL**, so
plain Python **threads** parallelize decode (3.4×), matching multiprocessing.

---

## 3. The diagnosis (roofline in one line)

```
GPU can filter : ~17,500 patches/sec   ┐
                                        ├─  the GPU is starved ~640x
disk can decode:      27 patches/sec   ┘   (one thread)
```

Your v9 run on the big slide: **137.45 s, kernel 2.1 s (1.5%), GPU util 17%,
1.3 GB / 3 GB used.** Every clue agrees: the card is idle 98% of the time waiting
for single-threaded patch decode. **No kernel/stream/precision tweak can help** —
v1–v9 all optimize the 1.5% that was never the problem. The v9 plan's 8–15×
target is unreachable because the workload is **I/O-bound, not compute-bound.**

---

## 4. The optimization that actually works — v10 (parallel I/O + GPU)

`data_loader_v10_parallel_io_gpu.py`: a pool of **reader threads decode patches
concurrently in one process** (GIL is free during OpenSlide reads; threads share
the single CUDA context), staging into a **pinned** host buffer; the GPU filters
each batch with the fused exact-integer luma kernel. The slow part (decode) is
now parallel; the fast part (filter) stays on the GPU, hidden and free.

### Result on the big slide (103424×50176, 4949 candidates, same as your v9 run)

| Version | Time | Speedup vs v9 | GPU kernel | Peak VRAM |
|---|---|---|---|---|
| your v9 (single-thread I/O) | 137.45 s | 1.0× | 2.1 s | 402 MB |
| v0b (8 CPU processes) | 37.6 s | 3.7× | — | — |
| **v10 (8 reader threads + GPU)** | **38.4 s** | **3.6×** | 2.15 s | 201 MB |
| **v10 (16 reader threads + GPU)** | **36.8 s** | **3.7×** | 2.10 s | 201 MB |

**3.6–3.7× faster, bit-exact (same 2171 patches kept as v0a).** v10 ties v0b
because both now hit the same disk/decode ceiling (~134 candidates/sec) — but
v10 does it in **one process with the GPU in the loop and half the VRAM** of v9,
so CPU cores spend their time on I/O instead of pixel counting, and the loaded
patches are already positioned for downstream GPU inference.

It is registered as `v10_par_io`, so it benchmarks/validates alongside everything
else:
```bash
python benchmark_runner.py --versions v0a_mono,v0b_multi,v9_ultimate,v10_par_io --stride 1024
```

---

## 5. Per-feature verdict (what to use, what to skip)

| GPU feature / layer | Verdict here | Evidence |
|---|---|---|
| **Parallel I/O (threads)** | ✅ **THE win** | 27 → 93 patch/s decode; 137s → 37s end-to-end |
| **Pinned memory** | ✅ keep | H2D 8.2 → 11.7 GB/s; v9 ablation kernel 67 → 52 ms |
| Fused uint8 kernel | ✅ keep | avoids 4× uint32 temp; 17.5k patch/s, 201 MB not 402 |
| Batching | ✅ keep (modest) | amortizes launch; saturates fast (only 2 ms of work) |
| Async streams (multi-stream) | ⚠️ neutral | nothing to overlap when kernel is 1.5%; doubles VRAM |
| Concurrent kernels | ⚠️ N/A | single trivial kernel; no benefit |
| **Mixed precision (FP16)** | ❌ **avoid** | Pascal FP16 = 1/64 FP32; kernel 52 → 90 ms |
| Bigger batch (v8 bs=128) | ⚠️ careful | more VRAM, no speed gain when I/O-bound; 3 GB is tight |

---

## 6. How to go faster than 37 s (beyond the GPU)

The bottleneck is now storage + JPEG/tile decode, not the GPU:

1. **Filter at a lower pyramid level.** This slide has levels at 1×…64×
   downsample. Detecting tissue at level 2 (4× down) reads **16× fewer bytes**;
   map kept tiles back to level 0. Almost always the single biggest win.
2. **Coarsen the stride** if full-resolution non-overlapping coverage isn't
   required — fewer candidates, linearly less time.
3. **Faster storage** (NVMe) and OS page-cache warming — decode is read-bound.
4. **More reader threads up to the decode plateau** (~8–16 here; beyond that the
   libjpeg/tile decoders saturate).
5. Only after the above would a faster GPU matter — and on a Volta+ card the FP16
   verdict flips, so mixed precision could finally help.
