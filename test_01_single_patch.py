"""
test_01_single_patch.py  --  Phase 1: Single-Patch Compute Cost

Isolates the *filtering computation* (grayscale + white/black ratio) from disk
I/O so we can see the raw CPU-vs-GPU compute trade-off and how it scales with
patch size. One real patch is read per size, then the pure compute is timed
many times with the GPU properly synchronized.

Metrics per (backend, patch_size): mean time (ms), throughput, GPU peak memory,
and accuracy (grayscale max error vs the exact PIL integer luma).

    python test_01_single_patch.py --sizes 256,512,1024,2048 --repeats 50
"""
import argparse
import time

import cupy as cp
import numpy as np
import openslide

from numerical_validation import check_numerical_stability, grayscale_reference
from test_performance_framework import DEFAULT_WSI, reset_gpu_memory

_LUMA = (19595, 38470, 7471)
WT, BT, RR = 230, 25, 0.9


def cpu_filter(rgb):
    g = rgb.astype(np.uint32)
    gray = ((g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2] + 32768) >> 16).astype(np.uint8)
    n = gray.size
    white = np.count_nonzero(gray > WT) / n
    black = np.count_nonzero(gray < BT) / n
    return gray, (white < RR and black < RR)


def gpu_int_filter(rgb_gpu):
    g = rgb_gpu.astype(cp.uint32)
    gray = ((g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2] + 32768) >> 16).astype(cp.uint8)
    n = gray.size
    white = cp.count_nonzero(gray > WT) / n
    black = cp.count_nonzero(gray < BT) / n
    keep = (white < RR) & (black < RR)
    return gray, keep


def gpu_fp16_filter(rgb_gpu):
    g = rgb_gpu.astype(cp.float16)
    gray = g[..., 0] * cp.float16(0.299) + g[..., 1] * cp.float16(0.587) + g[..., 2] * cp.float16(0.114)
    n = gray.size
    white = cp.count_nonzero(gray > WT) / n
    black = cp.count_nonzero(gray < BT) / n
    keep = (white < RR) & (black < RR)
    return gray, keep


def time_cpu(rgb, repeats):
    t0 = time.perf_counter()
    for _ in range(repeats):
        cpu_filter(rgb)
    return (time.perf_counter() - t0) / repeats


def time_gpu(rgb, repeats, fn):
    rgb_gpu = cp.asarray(rgb)            # keep data resident; measure compute only
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(rgb_gpu)
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / repeats


def time_gpu_with_transfer(rgb, repeats, fn):
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(cp.asarray(rgb))             # include host->device transfer each call
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--sizes", default="256,512,1024,2048")
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    print("=" * 88)
    print(" PHASE 1: SINGLE-PATCH COMPUTE COST (I/O excluded)")
    print("=" * 88)
    print(f"{'size':>6s} | {'CPU ms':>8s} | {'GPUint ms':>9s} | {'GPUint+xfer':>11s} | "
          f"{'fp16 ms':>8s} | {'GPUint spdup':>12s} | {'fp16 maxerr':>11s}")
    print("-" * 88)

    with openslide.OpenSlide(args.wsi) as slide:
        w, h = slide.level_dimensions[0]
        for size in sizes:
            x = max(0, w // 2 - size // 2)
            y = max(0, h // 2 - size // 2)
            patch = slide.read_region((x, y), 0, (size, size))
            rgb = np.asarray(patch)[:, :, :3].copy()
            pil_L = np.asarray(patch.convert("L"))

            # warmup (JIT)
            _ = cpu_filter(rgb); _ = gpu_int_filter(cp.asarray(rgb)); _ = gpu_fp16_filter(cp.asarray(rgb))
            cp.cuda.Stream.null.synchronize()
            reset_gpu_memory()

            t_cpu = time_cpu(rgb, args.repeats)
            t_gpu = time_gpu(rgb, args.repeats, gpu_int_filter)
            t_gpu_x = time_gpu_with_transfer(rgb, args.repeats, gpu_int_filter)
            t_f16 = time_gpu(rgb, args.repeats, gpu_fp16_filter)

            # accuracy of fp16 grayscale vs exact
            gpu_fp16_gray = cp.asnumpy(cp.rint(gpu_fp16_filter(cp.asarray(rgb))[0]).astype(cp.uint8))
            acc = check_numerical_stability(pil_L, gpu_fp16_gray)
            # sanity: gpu int must be exact
            gpu_int_gray = cp.asnumpy(gpu_int_filter(cp.asarray(rgb))[0])
            assert check_numerical_stability(grayscale_reference(rgb), gpu_int_gray)["exact_match"]

            print(f"{size:>6d} | {t_cpu*1e3:8.3f} | {t_gpu*1e3:9.3f} | {t_gpu_x*1e3:11.3f} | "
                  f"{t_f16*1e3:8.3f} | {t_cpu/t_gpu:11.2f}x | {acc['max_abs_error']:11.1f}")

    print("-" * 88)
    print(" GPUint = compute only (data resident);  GPUint+xfer = includes host->device copy.")
    print(" The gap between them is the transfer tax that batching (v2+) amortizes.")


if __name__ == "__main__":
    main()
