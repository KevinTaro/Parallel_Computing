# Where Should the Work Run? A Decode-vs-Compute Ablation for WSI Tissue Filtering

**Workload:** sliding-window tissue filtering over Philips whole-slide images
(WSI). For every 1024×1024 candidate patch, decode it and reject it if it is
>90% white (background) or >90% black (empty). The output is the set of kept
tissue coordinates.

**Goal of this study:** the earlier v1–v11 work established that the bottleneck
is *decode*, not the filter math. This report isolates **why** by separating two
independent variables and measuring each cell of the matrix:

| | **mono-core feed** | **multi-core feed** |
|---|---|---|
| **CPU only** (NumPy filter) | `v0a` | `v0b` |
| **general GPU compute** (CuPy filter, generic decode) | `v14` | `v15` |
| **specialized GPU decode** (nvJPEG) | `v12` | `v13` |

- **Mono vs multi** = does CPU read/decode parallelism help?
- **CPU-only vs general-GPU vs nvJPEG** = does moving the *filter* to general GPU
  cores help, and does a *specialized* GPU decoder help?

All six versions produce a **bit-identical kept set** (verified against v0a on
every slide), so the comparison is purely about speed and applicability.

---

## Hardware & data

- **GPU:** NVIDIA RTX 5090, 32 GB (driver 580.159)
- **Decoder libs:** `torchvision` nvJPEG (GPU), libjpeg via OpenSlide (CPU), CuPy 14.1
- **Slides:** all Philips TIFF, 512×512 **baseline JPEG (YCbCr 4:2:0)** tiles
  - small `S114-80954A` — 462 candidate patches
  - medium `S114-80969A` — 837 candidates (256 kept)
  - large `S114-82742C 20x` — ~4900 candidates (2171 kept)

Reproduce with: `python benchmark_decode_ablation.py`

---

## Results

Grid-creation wall time (best of N), speedup vs the v0a single-core reference,
and the per-stage breakdown the loaders expose.

### Large slide `S114-82742C 20x` (the representative case)

| Version | Strategy | Time | Speedup | Stage breakdown |
|---|---|---:|---:|---|
| `v0a` | mono CPU | 65.10s | 1.0× | (CPU decode+filter, sequential) |
| `v0b` | multi CPU | 5.87s | **11.1×** | multiprocessing decode+filter |
| `v14` | mono + general GPU | 64.89s | 1.0× | read 53.8s · xfer 0.9s · filter **0.08s** |
| `v15` | multi + general GPU | 8.65s | 7.5× | read 7.4s · xfer 1.1s · filter 0.07s |
| `v12` | mono + nvJPEG | **2.05s** | **31.8×** | read 0.13s · decode 1.85s · filter 0.06s |
| `v13` | multi + nvJPEG | 2.94s | 22.1× | read 1.16s · decode 1.85s · filter 0.06s |

### All three slides (speedup vs v0a)

| Version | small | medium | large |
|---|---:|---:|---:|
| `v0a` mono CPU | 1.0× | 1.0× | 1.0× |
| `v0b` multi CPU | 9.2× | 8.9× | 11.1× |
| `v14` mono + general GPU | 1.0× | 0.97× | 1.0× |
| `v15` multi + general GPU | 6.8× | 6.8× | 7.5× |
| `v12` mono + nvJPEG | **27.8×** | **27.2×** | **31.8×** |
| `v13` multi + nvJPEG | 20.1× | 21.6× | 22.1× |

---

## Analysis — five findings

**1. The filter was never the bottleneck; the decode always was.**
Across every GPU version the filter stage is **0.06–0.08s** even on the large
slide. v0a spends ~65s; essentially all of it is CPU JPEG decode. Any strategy
that does not change *where decode happens* cannot win.

**2. General GPU compute alone buys nothing (`v14` ≈ `v0a`, 1.0×).**
This is the direct answer to "how much does *just* parallel GPU compute help?"
— **none**, when the workload is decode-bound. v14 moves the filter to the GPU's
general CUDA cores (filter drops to 0.08s) but the decode is still single-core
CPU (53.8s), so the total is unchanged. The host→device transfer (0.9s) is in
fact *new* overhead v0a never paid. Parallel compute cannot remove a bottleneck
that lives in a different, serial stage.

**3. A specialized decoder is the only thing that removes the bottleneck
(`v12`, 22–32×).** nvJPEG moves the decode itself onto the GPU's fixed-function
JPEG hardware: 53.8s of CPU decode becomes 1.85s of GPU decode. This is the
single biggest lever in the entire v0–v15 series, and it dwarfs every
CPU-parallelism or general-compute approach.

**4. Two counter-intuitive inversions — more parallelism made it slower.**
   - **`v15` (8.65s) is *slower* than `v0b` (5.87s).** Both decode on multi-core
     CPU; v15 additionally ships each batch to the GPU for a 0.07s filter. The
     transfer (1.1s) costs more than the filter saves. *Offloading a cheap op to
     the GPU is a net loss once transfer is counted.*
   - **`v13` (2.94s) is *slower* than `v12` (2.05s).** Once decode is on the GPU,
     the CPU raw-tile read is no longer the bottleneck (0.13s mono). Spreading
     that trivial read across 20 threads just adds pool/lock overhead (read rises
     to 1.16s). *Parallelizing a non-bottleneck stage only adds coordination cost.*

**5. CPU multi-core is the portable middle ground (`v0b`, ~11×).** No GPU, no
codec assumptions — works on any OpenSlide-readable WSI. It is 3–6× slower than
nvJPEG but needs no special hardware or JPEG-tiled input.

---

## Applicability — the generality/speed trade-off

| Approach | Speedup | Works on | Constraint |
|---|---:|---|---|
| `v12/v13` nvJPEG | 22–32× | JPEG-tiled TIFF only | needs JPEG tiles + aligned geometry |
| `v0b` multi CPU | ~11× | **any** OpenSlide WSI | needs many CPU cores |
| `v14/v15` general GPU | 1–7× | **any** OpenSlide WSI | decode-bound; transfer overhead |

The fastest option (nvJPEG) is also the **least general**: it only works because
these slides happen to be JPEG-tiled. A hand-rolled GPU decoder would be *even
less* general. So "fits different TIFF files" and "fastest" pull in opposite
directions — the codec-generic GPU-compute path (v14/v15) is universal but cannot
beat the decode bottleneck, while the specialized path (v12/v13) wins big but
only on JPEG.

---

## Conclusions

1. **Optimize the bottleneck stage, not the convenient one.** v1–v11 and v14/v15
   all parallelized the filter — the cheap stage — and barely moved the needle.
   Only relocating *decode* (v12/v13, or CPU-parallel decode in v0b) changed the
   wall time.
2. **"Use the GPU" is not one decision.** General GPU compute (v14) and
   specialized GPU decode (v12) differ by **30×** on the same hardware. The win
   comes from fixed-function silicon doing the heavy, parallel decode — not from
   the CUDA cores doing the filter.
3. **More parallelism can be slower.** v13 < v12 and v15 < v0b: once a stage stops
   being the bottleneck, parallelizing it (or offloading it across the PCIe bus)
   only adds overhead.
4. **Recommendation:** for these JPEG WSIs, **`v12` (mono feed + nvJPEG)** is the
   best — 31.8× over baseline and *faster than the multi-threaded v13*, because
   the GPU decode makes CPU read parallelism unnecessary. For non-JPEG slides,
   fall back to **`v0b`** (portable ~11×). `v11` remains the production loader
   (nvJPEG + overlapped pipeline) for the full streaming workload.

---

## Files

| File | Cell |
|---|---|
| `data_loader_v0a_mono_baseline.py` | mono CPU (reference) |
| `data_loader_v0b_multi_baseline.py` | multi CPU |
| `data_loader_v12_gpu_decode_mono.py` | mono + nvJPEG |
| `data_loader_v13_gpu_decode_multi.py` | multi + nvJPEG |
| `data_loader_v14_gpu_compute_mono.py` | mono + general GPU compute |
| `data_loader_v15_gpu_compute_multi.py` | multi + general GPU compute |
| `data_loader_v11_gpu_decode_5090.py` | production nvJPEG streaming loader |
| `benchmark_decode_ablation.py` | this study's harness |
| `benchmark_decode_ablation_results.json` | raw numbers |
