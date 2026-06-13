"""
benchmark_gpu_decode.py

Benchmark covering the GPU-decode research series:
  v0a / v0b  -- CPU baselines
  v8         -- optimized CPU-feed reference
  v11–v32    -- full GPU-decode ablation + dec series

CLI
---
    python benchmark_gpu_decode.py
    python benchmark_gpu_decode.py --wsi data/S114-80954A-Her2\(3+\).tiff
    python benchmark_gpu_decode.py --iterations 1 --skip-v23        # 2 runs (warmup+1)
    python benchmark_gpu_decode.py --iterations 1 --no-warmup --skip-v23  # 1 run only
    python benchmark_gpu_decode.py --iterations 3 --no-warmup
"""
import argparse
import os
import time

from test_performance_framework import (
    DEFAULT_WSI, IMPLEMENTATIONS, BenchmarkRunner, CuPyTester,
    PerformanceMetrics,
)

RESULTS_DIR = "results"

# Ordered list: baselines → CuPy reference → full GPU-decode series
VERSIONS = [
    "v0a_mono",
    "v0b_multi",
    "v8_4060",
    "v11_gpudec",
    "v12_dec_mono",
    "v13_dec_multi",
    "v14_cmp_mono",
    "v15_cmp_multi",
    "v16_opt_mono",
    "v17_opt_multi",
    "v18_ult_mono",
    "v19_ult_multi",
    "v20_ult_pread",
    "v21_ult_pipe",
    "v22_par_destuff",
    "v23_dec_v1",       # batch=1 by design -- intentionally slow; use --skip-v23 to omit
    "v24_dec_v2",
    "v25_dec_v3",
    "v26_dec_v4",
    "v27_dec_v5",
    "v28_dec_v6",
    "v29_dec_v7",
    "v30_dec_v8",
    "v31_dec_v9",
    "v32_dec_v10",
]

# Sections for the printed report
_SECTIONS = {
    "CPU baseline":    ["v0a_mono", "v0b_multi"],
    "CuPy reference":  ["v8_4060"],
    "nvJPEG decode":   ["v11_gpudec", "v12_dec_mono", "v13_dec_multi"],
    "naive CUDA":      ["v14_cmp_mono", "v15_cmp_multi"],
    "opt CUDA":        ["v16_opt_mono", "v17_opt_multi"],
    "ultimate CUDA":   ["v18_ult_mono", "v19_ult_multi",
                        "v20_ult_pread", "v21_ult_pipe", "v22_par_destuff"],
    "dec series":      ["v23_dec_v1", "v24_dec_v2", "v25_dec_v3", "v26_dec_v4",
                        "v27_dec_v5", "v28_dec_v6", "v29_dec_v7",
                        "v30_dec_v8", "v31_dec_v9", "v32_dec_v10"],
}


def _row(v: str, m: PerformanceMetrics, base_a: float, base_b: float) -> str:
    sp_a = f"{base_a / m.min:6.2f}x" if m.min > 0 else "   N/A"
    sp_b = f"{base_b / m.min:6.2f}x" if m.min > 0 else "   N/A"
    k = f"{m.kernel_min:8.3f}" if m.kernel_times else "     N/A"
    return (f"  {v:16s} | {m.min:8.3f} | {m.mean:8.3f} | {m.std:6.3f} |"
            f" {k} | {sp_a} | {sp_b} | {m.peak_gpu_bytes/1e6:7.1f}")


def report(results: dict) -> str:
    base_a = results["v0a_mono"].min if "v0a_mono" in results else None
    base_b = results["v0b_multi"].min if "v0b_multi" in results else None

    hdr = (f"  {'version':16s} | {'min(s)':>8s} | {'mean(s)':>8s} | {'std':>6s} |"
           f" {'kernel(s)':>8s} | {'vs v0a':>7s} | {'vs v0b':>7s} | {'peakMB':>7s}")
    sep = "  " + "-" * (len(hdr) - 2)

    lines = ["=" * len(hdr)]
    lines.append(hdr)

    for section, keys in _SECTIONS.items():
        present = [k for k in keys if k in results]
        if not present:
            continue
        lines.append(sep)
        lines.append(f"  -- {section} --")
        for v in present:
            lines.append(_row(v, results[v], base_a or 1.0, base_b or 1.0))

    lines.append("=" * len(hdr))

    if base_a and results:
        fastest = min(results.values(), key=lambda m: m.min)
        lines.append(f"  Fastest: {fastest.version}  "
                     f"{base_a / fastest.min:.2f}x vs v0a  |  "
                     f"{base_b / fastest.min:.2f}x vs v0b")
    return "\n".join(lines)


def run(wsi: str, patch_size: int, stride: int, iterations: int,
        warmup: bool, skip_v23: bool) -> dict:
    versions = [v for v in VERSIONS if v in IMPLEMENTATIONS]
    if skip_v23 and "v23_dec_v1" in versions:
        versions.remove("v23_dec_v1")
        print("[!] v23_dec_v1 skipped (batch=1 by design -- very slow).")

    tester = CuPyTester(wsi_path=wsi, patch_size=patch_size, stride=stride)
    results = {}

    for v in versions:
        print(f"  -> {v} ...", end="", flush=True)
        try:
            m = tester.time_grid_creation(v, iterations=iterations, warmup=warmup)
            results[v] = m
            if m.kernel_times:
                print(f"  min={m.min:.3f}s  kernel={m.kernel_min:.3f}s"
                      f"  kept={m.n_kept}", flush=True)
            else:
                print(f"  min={m.min:.3f}s  kept={m.n_kept}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    return results


def main():
    ap = argparse.ArgumentParser(description="GPU-decode benchmark (v0/v8/v11-v32)")
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--iterations", type=int, default=3,
                    help="Timed runs per version (warmup adds 1 extra)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip the untimed warmup run (first run includes JIT cost)")
    ap.add_argument("--skip-v23", action="store_true",
                    help="Omit v23_dec_v1 (batch=1, ~4x slower by design)")
    args = ap.parse_args()

    warmup = not args.no_warmup
    total_runs = args.iterations + (1 if warmup else 0)
    print(f"[*] WSI        : {args.wsi}")
    print(f"[*] patch/stride: {args.patch_size}/{args.stride}")
    print(f"[*] iterations : {args.iterations}  warmup={'yes' if warmup else 'no'}")
    print(f"[*] total runs : {total_runs} per version")
    print(f"[*] versions   : {len(VERSIONS) - (1 if args.skip_v23 else 0)}\n")

    results = run(args.wsi, args.patch_size, args.stride,
                  args.iterations, warmup, args.skip_v23)

    print("\n" + report(results))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    meta = {
        "script": "benchmark_gpu_decode.py",
        "wsi": args.wsi,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "iterations": args.iterations,
        "warmup": warmup,
        "timestamp": ts,
    }
    # convert PerformanceMetrics -> plain dict for the plotter
    results_dict = {
        v: {
            "min": m.min, "mean": m.mean, "std": m.std, "max": m.max,
            "times": m.times, "kernel_times": m.kernel_times,
            "n_candidates": m.n_candidates, "n_kept": m.n_kept,
            "throughput": m.throughput, "peak_gpu_bytes": m.peak_gpu_bytes,
        }
        for v, m in results.items()
    }
    from plot_benchmark import plot
    png = os.path.join(RESULTS_DIR, f"benchmark_gpu_decode_{ts}.png")
    plot(meta, results_dict, out_path=png)


if __name__ == "__main__":
    main()
