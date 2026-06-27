# WSI Patch-Filtering: GPU Acceleration Study (v0 → v32)

**Course: Parallel Computing**

A systematic study of GPU acceleration applied to one fixed task: scanning a gigapixel
Whole-Slide Image (WSI) with a sliding window and filtering out non-tissue patches.
32 implementations are benchmarked against a single-threaded CPU baseline, each probing
a different parallelization strategy.

---

## The Task

Every version solves the same problem:

1. Slide a 1024×1024 window across a TIFF WSI at a given stride.
2. Convert each patch to grayscale using ITU-R 601 integer luma:
   ```
   L = (R×19595 + G×38470 + B×7471 + 32768) >> 16
   ```
3. Discard the patch if ≥ 90% of pixels are white (`L > 230`) or black (`L < 25`).
4. Return the coordinates of kept patches.

**Correctness gate:** every version must return the *exact same* kept-coordinate set as
the single-threaded CPU baseline (v0a), verified by Jaccard index = 1.0000.

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB VRAM |
| CPU | ~20 cores |
| PCIe Memory BW | ~11.7 GB/s (pinned H2D) |
| Test slides | 5 Philips Her2 `.tiff` WSIs (small: 462 candidates; large: 4,949 candidates) |

---

## Key Results

### Best overall: v16 — optimized hand-written CUDA decoder, mono feed

| Slide size | Speedup vs v0a (CPU mono) |
|---|---|
| Small (462 candidates) | **30.5×** |
| Large (4,949 candidates) | **62.0×** |

### Decode-stage comparison (large slide)

| Engine | Decode time | vs CPU libjpeg |
|---|---:|---:|
| CPU libjpeg (v0a baseline) | ~54 s | 1× |
| nvJPEG (v12) | 1.85 s | ~29× |
| Naive CUDA (v14) | 4.21 s | ~13× |
| **Optimized CUDA (v16)** | **1.15 s** | **~47×** |

The hand-written CUDA decoder **outperforms nvJPEG** by using register bit-buffers,
8-bit Huffman LUTs in constant memory, a DC-only IDCT fast path, and a fused
YCbCr→luma→count kernel that never materializes an RGB buffer.

### Key finding: the bottleneck was the decoder, not the filter

Profiling (via `hardware_probe.py`) showed that in the CuPy series (v1–v9), the GPU
compute kernel occupies only **~1%** of pipeline time — the GPU was starved 640× by
single-threaded OpenSlide JPEG decoding. Optimizing the filter kernel (v9) optimized the
wrong thing. Attacking the JPEG decode (v11+) is what unlocks large speedups.

---

## Implementation Map

### CPU baselines

| Version | File | Description |
|---|---|---|
| v0a | [data_loader_v0a_mono_baseline.py](data_loader_v0a_mono_baseline.py) | Single-threaded reference (1.0×) |
| v0b | [data_loader_v0b_multi_baseline.py](data_loader_v0b_multi_baseline.py) | CPU multiprocessing baseline (~8.8×) |

### Line A — CuPy filter series (v1–v10): move the *filter* to the GPU

| Version | File | New idea |
|---|---|---|
| v1 | [data_loader_v1_cupy_full.py](data_loader_v1_cupy_full.py) | Per-patch GPU transfer (naive; demonstrates launch overhead) |
| v2 | [data_loader_v2_cupy_batch.py](data_loader_v2_cupy_batch.py) | Batch N patches per transfer |
| v3 | [data_loader_v3_cupy_hybrid.py](data_loader_v3_cupy_hybrid.py) | Hybrid routing: CPU for small chunks, GPU for large |
| v4 | [data_loader_v4_cupy_pinned_memory.py](data_loader_v4_cupy_pinned_memory.py) | Pinned host buffer + reused device buffer |
| v5 | [data_loader_v5_cupy_async.py](data_loader_v5_cupy_async.py) | Double-buffered CUDA streams |
| v6 | [data_loader_v6_cupy_mixed_precision.py](data_loader_v6_cupy_mixed_precision.py) | fp16 luma (shows no benefit on RTX 5090) |
| v7 | [data_loader_v7_cupy_memory_optimized.py](data_loader_v7_cupy_memory_optimized.py) | Fused uint8 kernel; minimal VRAM footprint |
| v8 | [data_loader_v8_cupy_optimized_4060.py](data_loader_v8_cupy_optimized_4060.py) | Threaded OpenSlide readers (first attack on feed side) |
| v9 | [data_loader_v9_ultimate_gpu.py](data_loader_v9_ultimate_gpu.py) | All 7 CuPy layers stacked; each toggleable for ablation |
| v10 | [data_loader_v10_parallel_io_gpu.py](data_loader_v10_parallel_io_gpu.py) | Data-driven pivot: reader thread pool after profiling reveals I/O bottleneck |

### Line B — GPU decode series (v11–v22): move the *decoder* to the GPU

| Version | File | Decoder | Feed |
|---|---|---|---|
| v11 | [data_loader_v11_gpu_decode_5090.py](data_loader_v11_gpu_decode_5090.py) | nvJPEG | threaded |
| v12 | [data_loader_v12_gpu_decode_mono.py](data_loader_v12_gpu_decode_mono.py) | nvJPEG | mono |
| v13 | [data_loader_v13_gpu_decode_multi.py](data_loader_v13_gpu_decode_multi.py) | nvJPEG | multi |
| v14 | [data_loader_v14_gpu_compute_mono.py](data_loader_v14_gpu_compute_mono.py) | naive CUDA | mono |
| v15 | [data_loader_v15_gpu_compute_multi.py](data_loader_v15_gpu_compute_multi.py) | naive CUDA | multi |
| **v16** | [data_loader_v16_cuda_opt_mono.py](data_loader_v16_cuda_opt_mono.py) | **optimized CUDA** | mono — **best overall** |
| v17 | [data_loader_v17_cuda_opt_multi.py](data_loader_v17_cuda_opt_multi.py) | optimized CUDA | multi |
| v18 | [data_loader_v18_cuda_ultimate_mono.py](data_loader_v18_cuda_ultimate_mono.py) | ultimate CUDA | mono + pinned + async pipeline |
| v19–v22 | v19–v22 files | ultimate CUDA | Progressive removal of hidden serializations (lock → pread → prefetch → destuff) |

### Line C — GPU decode remapping series (v23–v32): replay Line A concepts on GPU decode

Pedagogical cross-validation: the same optimization ideas from v1–v10 (batch, hybrid,
pinned, async, fp16…) are re-applied in the GPU-decode context to confirm they generalize.

---

## Hand-written CUDA Decoders

Three generations of baseline-JPEG decoder, all producing bit-identical output:

| File | Description |
|---|---|
| [gpu_jpeg_decoder.py](gpu_jpeg_decoder.py) | Naive: one CUDA thread per tile, full Huffman → dequant → IDCT |
| [gpu_jpeg_decoder_optimized.py](gpu_jpeg_decoder_optimized.py) | Register bit-buffer, 8-bit Huffman LUT, constant memory, DC-only fast path, fused luma kernel |
| [gpu_jpeg_decoder_ultimate.py](gpu_jpeg_decoder_ultimate.py) | Optimized kernel + pinned staging + double-buffered async streams + buffer pool |

---

## Running Benchmarks

```bash
# Probe hardware limits and identify the bottleneck
python hardware_probe.py --wsi data/S114-80954A-Her2\(3+\).tiff

# Benchmark specific versions
python benchmark_runner.py --wsi data/S114-80954A-Her2\(3+\).tiff \
    --stride 1024 --iterations 3 \
    --versions v0a_mono,v0b_multi,v16_opt_mono

# Benchmark the full GPU-decode series
python benchmark_gpu_decode.py

# Run the 2×2 decode ablation (mono/multi × CPU/nvJPEG/CUDA)
python benchmark_decode_ablation.py

# Consolidate multiple benchmark JSON runs into a comparison matrix
python comparative_analysis.py

# Visualize a benchmark JSON (stacked-bar per stage)
python plot_benchmark.py results/benchmark_<timestamp>.json
```

Results are written to `results/benchmark_<timestamp>.json` and optional `.png` plots.

---

## Validation

```bash
# End-to-end correctness gate (Jaccard = 1.0 vs v0a)
python validation_suite.py

# Numerical correctness primitives only
python numerical_validation.py
```

---

## Test Suite (Phase 0–6)

| File | What it measures |
|---|---|
| [test_00_baseline_comparison.py](test_00_baseline_comparison.py) | Speedup hierarchy is real and correctness-preserving |
| [test_01_single_patch.py](test_01_single_patch.py) | Pure filter compute: CPU vs GPU vs patch size |
| [test_02_batch_filtering.py](test_02_batch_filtering.py) | GPU break-even point across strides |
| [test_03_end_to_end_dataloader.py](test_03_end_to_end_dataloader.py) | `__getitem__` + DataLoader worker path |
| [test_04_memory_profiling.py](test_04_memory_profiling.py) | Peak VRAM, bytes/candidate, leak check |
| [test_05_scalability.py](test_05_scalability.py) | Throughput scaling with patch size |
| [test_06_integration_training.py](test_06_integration_training.py) | Data-loading fraction of a mock training step |
| [test_v9_ablation.py](test_v9_ablation.py) | v9-specific: toggle each of 7 optimization layers |

---

## Dependencies

- Python 3.12
- [OpenSlide](https://openslide.org/) + `openslide-python`
- [CuPy](https://cupy.dev/) (CUDA 12)
- [PyTorch](https://pytorch.org/) + `torchvision` (for nvJPEG path)
- NumPy, Pillow, matplotlib

---

## Suggested Reading Order

1. [data_loader_v0a_mono_baseline.py](data_loader_v0a_mono_baseline.py) — understand the task
2. [data_loader_v1_cupy_full.py](data_loader_v1_cupy_full.py) → [data_loader_v2_cupy_batch.py](data_loader_v2_cupy_batch.py) — why batching matters
3. [hardware_probe.py](hardware_probe.py) — the data that reframed the whole project
4. [data_loader_v11_gpu_decode_5090.py](data_loader_v11_gpu_decode_5090.py) — first GPU-decode version
5. v19 → v20 → v21 → v22 files — the parallel I/O debugging arc (removing hidden serializations)
6. [FINAL_REPORT.md](FINAL_REPORT.md) — complete results and conclusions
