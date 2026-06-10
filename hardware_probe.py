"""
hardware_probe.py

Dig out the GPU's real capabilities AND empirically measure where the WSI
filtering pipeline actually spends time, so optimization is data-driven rather
than guessed.

Sections:
  1. GPU device properties (full dump from the CUDA runtime)
  2. Derived theoretical limits (memory bandwidth, FP16 ratio, occupancy)
  3. Empirical PCIe transfer bandwidth (pageable vs pinned, H2D / D2H)
  4. Empirical luma-kernel throughput (achieved GB/s vs theoretical)
  5. Empirical OpenSlide read rate: single-thread vs threads vs processes
  6. Roofline-style verdict: what is the bottleneck, what to optimize

    python hardware_probe.py --wsi data/S114-80954A-Her2(3+).tiff --n-io 200
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

import cupy as cp
import numpy as np
import openslide

_LUMA = (19595, 38470, 7471)
_luma_kernel = cp.ElementwiseKernel(
    'uint8 r, uint8 g, uint8 b', 'uint8 gray',
    'gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;', 'pil_luma_probe')


# ---------------------------------------------------------------- 1. props
def dump_properties():
    print("=" * 78)
    print(" 1. GPU DEVICE PROPERTIES")
    print("=" * 78)
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    cc = (props["major"], props["minor"])
    free, total = dev.mem_info

    def g(k, default=None):
        return props.get(k, default)

    print(f"  Name                     : {name}")
    print(f"  Compute capability       : {cc[0]}.{cc[1]}  (sm_{cc[0]}{cc[1]})")
    print(f"  Total VRAM               : {total/1e9:.2f} GB   (free now: {free/1e9:.2f} GB)")
    print(f"  Multiprocessors (SM)     : {g('multiProcessorCount')}")
    print(f"  Max threads / block      : {g('maxThreadsPerBlock')}")
    print(f"  Max threads / SM         : {g('maxThreadsPerMultiProcessor')}")
    print(f"  Warp size                : {g('warpSize')}")
    print(f"  Registers / block        : {g('regsPerBlock')}")
    print(f"  Shared mem / block       : {g('sharedMemPerBlock', 0)/1024:.0f} KB")
    print(f"  L2 cache                 : {g('l2CacheSize', 0)/1024:.0f} KB")
    print(f"  Core clock               : {g('clockRate', 0)/1e6:.3f} GHz")
    print(f"  Memory clock             : {g('memoryClockRate', 0)/1e6:.3f} GHz")
    print(f"  Memory bus width         : {g('memoryBusWidth')} bit")
    print(f"  Async copy engines       : {g('asyncEngineCount')}  "
          f"(>=2 => H2D and D2H can overlap)")
    print(f"  Concurrent kernels       : {bool(g('concurrentKernels'))}")
    print(f"  Unified addressing       : {bool(g('unifiedAddressing'))}")
    print(f"  ECC enabled              : {bool(g('ECCEnabled'))}")
    return props, cc


# --------------------------------------------------- 2. theoretical limits
def theoretical_limits(props, cc):
    print("\n" + "=" * 78)
    print(" 2. DERIVED THEORETICAL LIMITS")
    print("=" * 78)
    mem_clk_hz = props.get("memoryClockRate", 0) * 1e3      # kHz -> Hz
    bus_bytes = props.get("memoryBusWidth", 0) / 8
    bw = 2.0 * mem_clk_hz * bus_bytes / 1e9                 # GDDR is DDR (x2)
    print(f"  Theoretical mem bandwidth: {bw:.1f} GB/s  (2 x memclk x bus/8)")

    sm = props.get("multiProcessorCount", 0)
    core_hz = props.get("clockRate", 0) * 1e3
    # Pascal (6.1) = 128 FP32 cores/SM; FP16 runs at 1/64 of FP32 (no fast path).
    cores_per_sm = {6: 128, 7: 64, 8: 128}.get(cc[0], 64)
    fp32 = 2.0 * sm * cores_per_sm * core_hz / 1e12         # 2 = FMA
    print(f"  Peak FP32 (FMA)          : {fp32:.2f} TFLOP/s "
          f"({sm} SM x {cores_per_sm} cores x {core_hz/1e9:.2f}GHz x2)")
    if cc == (6, 1):
        print(f"  Peak FP16                : ~{fp32/64*1000:.0f} GFLOP/s  "
              f"(Pascal sm_61 runs FP16 at 1/64 FP32 -> mixed precision HURTS)")
    print(f"  => This is a MEMORY-BOUND, not compute-bound, card for byte-shuffling")
    print(f"     work like grayscale+threshold. The luma kernel should hit ~{bw:.0f} GB/s.")
    return bw


# ---------------------------------------------- 3. transfer bandwidth
def transfer_bandwidth(mb=256, iters=10):
    print("\n" + "=" * 78)
    print(" 3. EMPIRICAL PCIe TRANSFER BANDWIDTH")
    print("=" * 78)
    n = mb * 1024 * 1024
    pageable = np.ones(n, dtype=np.uint8)
    pinned_mem = cp.cuda.alloc_pinned_memory(n)
    pinned = np.frombuffer(pinned_mem, np.uint8, n)
    pinned[:] = 1
    d = cp.empty(n, dtype=cp.uint8)

    def bench(host, direction):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            if direction == "h2d":
                d.set(host)
            else:
                d.get(out=host)
        cp.cuda.Stream.null.synchronize()
        dt = (time.perf_counter() - t0) / iters
        return mb / 1024 / dt   # GB/s

    print(f"  {'transfer':22s} | {'GB/s':>8s}")
    print("  " + "-" * 34)
    print(f"  {'H2D pageable':22s} | {bench(pageable, 'h2d'):8.1f}")
    print(f"  {'H2D pinned':22s} | {bench(pinned, 'h2d'):8.1f}")
    print(f"  {'D2H pageable':22s} | {bench(pageable, 'd2h'):8.1f}")
    print(f"  {'D2H pinned':22s} | {bench(pinned, 'd2h'):8.1f}")
    print("  Pinned should be markedly faster; that gap is why Layer 1 matters.")
    del d
    cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------- 4. kernel throughput
def kernel_throughput(ps=1024, batch=32, iters=20):
    print("\n" + "=" * 78)
    print(" 4. EMPIRICAL LUMA-KERNEL THROUGHPUT (data resident on GPU)")
    print("=" * 78)
    rgb = cp.asarray(np.random.randint(0, 256, (batch, ps, ps, 3), dtype=np.uint8))
    gray = cp.empty((batch, ps, ps), dtype=cp.uint8)
    # warmup / JIT
    _luma_kernel(rgb[..., 0], rgb[..., 1], rgb[..., 2], gray)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _luma_kernel(rgb[..., 0], rgb[..., 1], rgb[..., 2], gray)
    cp.cuda.Stream.null.synchronize()
    dt = (time.perf_counter() - t0) / iters

    pixels = batch * ps * ps
    bytes_moved = pixels * 3 + pixels  # read RGB + write gray
    print(f"  batch={batch} patch={ps}: {dt*1e3:.3f} ms/call")
    print(f"  effective bandwidth      : {bytes_moved/dt/1e9:.1f} GB/s")
    print(f"  patch throughput         : {batch/dt:,.0f} patches/sec (pure compute)")
    del rgb, gray
    cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------- 5. OpenSlide read rate
_WSI_PATH = None
def _read_one(coord):
    x, y = coord
    with openslide.OpenSlide(_WSI_PATH) as s:
        p = s.read_region((x, y), 0, (1024, 1024))
        return np.asarray(p)[:, :, :3].sum()  # touch data


def io_rate(wsi_path, n=200):
    global _WSI_PATH
    _WSI_PATH = wsi_path
    print("\n" + "=" * 78)
    print(" 5. OPENSLIDE READ RATE  (the suspected bottleneck)")
    print("=" * 78)
    with openslide.OpenSlide(wsi_path) as s:
        w, h = s.level_dimensions[0]
    ps = 1024
    xs = np.linspace(0, w - ps, int(np.sqrt(n)), dtype=int)
    ys = np.linspace(0, h - ps, int(np.sqrt(n)), dtype=int)
    coords = [(int(x), int(y)) for y in ys for x in xs][:n]
    n = len(coords)

    # single thread, one shared handle
    with openslide.OpenSlide(wsi_path) as s:
        t0 = time.perf_counter()
        for x, y in coords:
            np.asarray(s.read_region((x, y), 0, (ps, ps)))[:, :, :3].sum()
        t_single = time.perf_counter() - t0

    # threads (tests whether OpenSlide releases the GIL during decode)
    thread_rates = {}
    for nw in (2, 4, 8):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nw) as ex:
            list(ex.map(_read_one, coords))
        thread_rates[nw] = n / (time.perf_counter() - t0)

    # processes (v0b style)
    proc_rates = {}
    for nw in (4, 8):
        t0 = time.perf_counter()
        with Pool(nw) as pool:
            pool.map(_read_one, coords)
        proc_rates[nw] = n / (time.perf_counter() - t0)

    r_single = n / t_single
    print(f"  patches read: {n}, each 1024x1024 @ level 0")
    print(f"  {'mode':22s} | {'patch/s':>8s} | {'speedup':>7s}")
    print("  " + "-" * 44)
    print(f"  {'single-thread':22s} | {r_single:8.1f} | {1.0:6.2f}x")
    for nw, r in thread_rates.items():
        print(f"  {'threads x'+str(nw):22s} | {r:8.1f} | {r/r_single:6.2f}x")
    for nw, r in proc_rates.items():
        print(f"  {'processes x'+str(nw):22s} | {r:8.1f} | {r/r_single:6.2f}x")
    return r_single, thread_rates, proc_rates


def verdict(r_single, thread_rates, proc_rates):
    print("\n" + "=" * 78)
    print(" 6. VERDICT")
    print("=" * 78)
    best_thread = max(thread_rates.values())
    best_proc = max(proc_rates.values())
    thread_scales = best_thread > 1.5 * r_single
    print(f"  Single-thread I/O ceiling : {r_single:.0f} patch/s "
          f"=> {1000/r_single:.1f} ms/patch just to DECODE")
    print(f"  Threads scale?            : {'YES' if thread_scales else 'NO'} "
          f"(best {best_thread:.0f} patch/s) "
          f"-> OpenSlide {'releases' if thread_scales else 'holds'} the GIL during read")
    print(f"  Processes scale?          : best {best_proc:.0f} patch/s "
          f"({best_proc/r_single:.1f}x)")
    print()
    print("  The GPU luma kernel does ~thousands of patches/sec (section 4), but the")
    print("  pipeline feeds it at the I/O rate above. The card is STARVED, not slow.")
    if thread_scales:
        print("  => OPTIMIZE: multi-threaded reader feeding GPU batches in ONE process")
        print("     (threads share the CUDA context; GIL is free during decode).")
    else:
        print("  => OPTIMIZE: multi-PROCESS readers (threads blocked by GIL); feed GPU")
        print("     from a pool, or just use v0b. GPU compute is already free.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default="data/S114-80954A-Her2(3+).tiff")
    ap.add_argument("--n-io", type=int, default=200)
    args = ap.parse_args()
    props, cc = dump_properties()
    theoretical_limits(props, cc)
    transfer_bandwidth()
    kernel_throughput()
    rs, tr, pr = io_rate(args.wsi, args.n_io)
    verdict(rs, tr, pr)


if __name__ == "__main__":
    main()
