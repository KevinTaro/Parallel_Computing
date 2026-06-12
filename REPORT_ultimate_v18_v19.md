# v18 / v19 — The "Ultimate" Stack, and the Wall It Hits

v18 (mono) and v19 (multi) take v16/v17's optimized custom-CUDA decoder and add
**every remaining transfer / overlap / memory layer** from
`CUPY_RESEARCH_PLAN.md`. They are the literal "stack every layer" versions (v9
philosophy). All output is **bit-identical** to v14/v16 and the v0a baseline on
every slide.

## Layers implemented

| Plan layer | v18/v19 implementation |
|---|---|
| v2 batch | manual `batch_size` |
| v4 pinned memory | page-locked host staging for scan upload + result read-back |
| v5 async streams | two CUDA streams, non-blocking `submit` / `fetch` |
| v7 memory pool | device + pinned buffers pre-allocated once, reused every batch |
| v9 pipeline | **double-buffered**: CPU read+destuff+upload of batch *k+1* overlaps GPU decode+count of batch *k* |
| v5/v9 kernel | inherited: bit-buffer, Huffman LUT, constant memory, DC-only skip, fused no-RGB count |
| v6 mixed precision | **deliberately omitted** — fp16 IDCT would break the bit-exact kept set |

## The overlap works — perfectly

The headline diagnostic: the GPU **never waits** for host work. The pipeline
"gpu-wait" time (the stall at each `fetch`) drops to **0.000 s** on every slide,
versus v16/v17 where the GPU sits idle during each batch's read+upload:

| slide | v16 read / gpu-wait | v18 read / gpu-wait |
|---|---|---|
| small | 0.009 / 0.184 | 0.005 / **0.000** |
| medium | 0.007 / 0.152 | 0.008 / **0.000** |
| large | 0.080 / 1.116 | 0.081 / **0.000** |

## But wall-clock does not improve

| slide | v16 | v18 | v17 | v19 |
|---|---:|---:|---:|---:|
| small | 0.195s | 0.183s | 0.246s | 0.244s |
| medium | 0.163s | 0.164s | 0.247s | 0.240s |
| large | 1.204s | 1.216s | 1.642s | 1.678s |

v18 ≈ v16; v19 ≈ v17. **The "ultimate" transfer optimizations buy ~nothing here.**

## Why — and why that's the real finding

The layers v18/v19 add all attack **host work and PCIe transfers**. But on cached
local storage this workload is **GPU-compute-bound**: on the large slide the GPU
decode is ~1.12 s while the entire host read is ~0.08 s (7%). The pipeline
*successfully* hides that 0.08 s behind the GPU (gpu-wait → 0), but the GPU still
has 1.12 s of decode work to grind through serially across batches. Hiding 7%
that was never on the critical path saves ~0%.

This is the textbook lesson the research plan was built to surface — **overlap and
transfer optimization only pay off when transfer/I-O is the bottleneck.** Here it
isn't:

- **When v18/v19 *would* win:** cold-cache / network / slow-disk storage where the
  per-batch read rivals the decode time. Then v16's serial read→decode doubles the
  wall clock while v18's pipeline stays decode-bound. The benefit scales with I/O
  cost, which is ~0 on a warm local SSD.
- **`multi` still loses to `mono`** (v19≥v18, v17≥v16): threaded reads behind a
  single file handle add lock/GIL contention, and the pipeline already hides reads,
  so the extra threads only cost.

## The remaining ceiling

With transfers fully hidden and the kept set fixed, the *only* lever left is the
**decode kernel compute itself** (the per-tile Huffman + IDCT). That is the
"optimize the decode algorithm" direction — e.g. a warp-parallel or sparse IDCT —
which trades away the bit-exactness guarantee these versions are built on. It is
out of scope for the transfer-focused plan, and is the natural next experiment.

## Bottom line

v18/v19 are the correct, complete "ultimate" implementation of the plan's
GPU layers, and they prove the overlap machinery works (GPU never idles). The
honest result is that this filtering workload is compute-bound on local storage,
so the fastest practical versions remain **v16 (mono) / v12 (nvJPEG)**; v18/v19
become the right choice the moment storage gets slow.

| File | Role |
|---|---|
| `gpu_jpeg_decoder_ultimate.py` | optimized decoder + pinned + double-buffered stream pipeline |
| `data_loader_v18_cuda_ultimate_mono.py` / `..v19_cuda_ultimate_multi.py` | mono / multi pipeline loaders |
