"""
benchmark_runner.py

Run timing benchmarks across data-loader versions and emit a report.

CLI
---
    python benchmark_runner.py --wsi data/slide.tiff --stride 2048 \
        --iterations 3 --versions v0a_mono,v0b_multi,v2_batch,v3_hybrid

    # all versions, default (smallest) slide, quick stride:
    python benchmark_runner.py --stride 4096

Outputs:
    results/benchmark_<timestamp>.json   (always)
    results/speedup_comparison.png       (if matplotlib is installed)

Functions (also importable):
    run_single_benchmark, run_comparative_benchmark,
    generate_performance_report, plot_results
"""
import argparse
import json
import os
import time
from typing import Dict, List, Optional

from test_performance_framework import (
    DEFAULT_WSI, IMPLEMENTATIONS, BenchmarkRunner, CuPyTester,
    PerformanceMetrics, list_versions,
)

RESULTS_DIR = "results"


def run_single_benchmark(version: str, wsi_path: str = DEFAULT_WSI,
                         patch_size: int = 1024, stride: int = 1024,
                         iterations: int = 3, **overrides) -> PerformanceMetrics:
    tester = CuPyTester(wsi_path=wsi_path, patch_size=patch_size, stride=stride)
    return tester.time_grid_creation(version, iterations=iterations, **overrides)


def run_comparative_benchmark(versions: List[str], wsi_path: str = DEFAULT_WSI,
                              patch_size: int = 1024, stride: int = 1024,
                              iterations: int = 3) -> Dict[str, PerformanceMetrics]:
    tester = CuPyTester(wsi_path=wsi_path, patch_size=patch_size, stride=stride)
    runner = BenchmarkRunner(tester)
    return runner.run(versions, iterations=iterations)


def generate_performance_report(results: Dict[str, PerformanceMetrics],
                                baseline: str = "v0a_mono") -> str:
    """Build a human-readable text report with speedups vs the mono baseline."""
    lines = []
    base_t = results[baseline].min if baseline in results else None
    multi_t = results["v0b_multi"].min if "v0b_multi" in results else None

    lines.append("=" * 92)
    lines.append(f"{'version':11s} | {'min(s)':>8s} | {'mean(s)':>8s} | {'std':>6s} | "
                 f"{'patch/s':>9s} | {'vs v0a':>7s} | {'vs v0b':>7s} | {'peakMB':>7s}")
    lines.append("-" * 92)
    for v, m in results.items():
        sp_a = f"{base_t / m.min:6.2f}x" if base_t and m.min > 0 else "   -  "
        sp_b = f"{multi_t / m.min:6.2f}x" if multi_t and m.min > 0 else "   -  "
        lines.append(f"{v:11s} | {m.min:8.3f} | {m.mean:8.3f} | {m.std:6.3f} | "
                     f"{m.throughput:9.1f} | {sp_a:>7s} | {sp_b:>7s} | {m.peak_gpu_bytes/1e6:7.1f}")
    lines.append("=" * 92)
    if base_t:
        fastest = min(results.values(), key=lambda m: m.min)
        lines.append(f"Fastest: {fastest.version} ({base_t / fastest.min:.2f}x vs {baseline})")
    return "\n".join(lines)


def generate_kernel_timing_report(results: Dict[str, PerformanceMetrics]) -> str:
    """Report pure kernel times vs total times (overhead analysis)."""
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'version':11s} | {'total(s)':>8s} | {'kernel(s)':>8s} | {'kernel%':>7s} | {'overhead(s)':>10s} | {'vs CPU':>7s}")
    lines.append("-" * 100)
    for v, m in results.items():
        if not m.kernel_times:
            lines.append(f"{v:11s} | {m.min:8.3f} | {'N/A':>8s} | {'N/A':>7s} | {'N/A':>10s} | {'N/A':>7s}")
        else:
            kernel = m.kernel_min
            total = m.min
            overhead = total - kernel
            kernel_pct = (kernel / total * 100) if total > 0 else 0
            lines.append(f"{v:11s} | {total:8.3f} | {kernel:8.3f} | {kernel_pct:7.1f}% | {overhead:10.3f} | {''}")
    lines.append("=" * 100)
    return "\n".join(lines)


def plot_results(results: Dict[str, PerformanceMetrics], out_path: str,
                 baseline: str = "v0a_mono") -> Optional[str]:
    """Bar chart of speedup vs baseline. Returns path, or None if no matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib not available; skipping chart.")
        return None
    if baseline not in results:
        return None

    base_t = results[baseline].min
    versions = list(results.keys())
    speedups = [base_t / results[v].min if results[v].min > 0 else 0 for v in versions]
    colors = ["#888" if IMPLEMENTATIONS[v]["category"] == "cpu" else "#1f77b4" for v in versions]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(versions, speedups, color=colors)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="v0a baseline (1.0x)")
    ax.set_ylabel(f"Speedup vs {baseline} (higher = faster)")
    ax.set_title("WSI patch-filtering: grid-creation speedup by version")
    ax.legend()
    for b, s in zip(bars, speedups):
        ax.text(b.get_x() + b.get_width() / 2, s, f"{s:.2f}x", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _serialize(results: Dict[str, PerformanceMetrics], meta: dict) -> dict:
    return {
        "meta": meta,
        "results": {v: {"min": m.min, "mean": m.mean, "std": m.std, "max": m.max,
                        "times": m.times, "n_candidates": m.n_candidates,
                        "n_kept": m.n_kept, "throughput": m.throughput,
                        "peak_gpu_bytes": m.peak_gpu_bytes, "extra": m.extra}
                    for v, m in results.items()},
    }


def main():
    ap = argparse.ArgumentParser(description="WSI data-loader benchmark runner")
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--versions", default="all",
                    help="comma-separated keys or 'all' / 'cpu' / 'gpu'")
    args = ap.parse_args()

    if args.versions == "all":
        versions = list_versions()
    elif args.versions in ("cpu", "gpu"):
        versions = list_versions(args.versions)
    else:
        versions = [v.strip() for v in args.versions.split(",")]

    print(f"[*] WSI={args.wsi}  patch={args.patch_size}  stride={args.stride}  "
          f"iters={args.iterations}")
    print(f"[*] Versions: {versions}\n")

    results = run_comparative_benchmark(versions, wsi_path=args.wsi,
                                        patch_size=args.patch_size,
                                        stride=args.stride, iterations=args.iterations)
    report = generate_performance_report(results)
    print("\n" + report)

    has_kernel = any(m.kernel_times for m in results.values())
    if has_kernel:
        kernel_report = generate_kernel_timing_report(results)
        print("\n" + kernel_report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    meta = {"wsi": args.wsi, "patch_size": args.patch_size, "stride": args.stride,
            "iterations": args.iterations, "timestamp": ts}
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(_serialize(results, meta), f, indent=2)
    print(f"\n[*] Saved {json_path}")

    png = plot_results(results, os.path.join(RESULTS_DIR, "speedup_comparison.png"))
    if png:
        print(f"[*] Saved {png}")


if __name__ == "__main__":
    main()
