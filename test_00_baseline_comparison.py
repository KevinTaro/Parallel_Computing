"""
test_00_baseline_comparison.py  --  Phase 0: Baseline Establishment (CRITICAL)

Proves the speedup hierarchy v0a -> v0b -> (best GPU) is real and measurable,
and that every version preserves filtering correctness. This is the gate the
rest of the study stands on.

    python test_00_baseline_comparison.py --stride 2048 --iterations 3
"""
import argparse

from benchmark_runner import generate_performance_report, run_comparative_benchmark
from numerical_validation import validate_against_baseline
from test_performance_framework import CuPyTester, DEFAULT_WSI, list_versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args()

    print("=" * 72)
    print(" PHASE 0: BASELINE COMPARISON")
    print(f" WSI={args.wsi} patch={args.patch_size} stride={args.stride}")
    print("=" * 72)

    versions = list_versions()

    # --- Part B/C: validate every version against the v0a mono baseline -----
    print("\n[Validation] kept-coordinate agreement vs v0a_mono")
    tester = CuPyTester(wsi_path=args.wsi, patch_size=args.patch_size, stride=args.stride)
    base = tester.build("v0a_mono").coordinates
    for v in versions:
        if v == "v0a_mono":
            continue
        verdict = validate_against_baseline(base, tester.build(v).coordinates)
        flag = "PASS" if verdict["passed"] else "FAIL"
        print(f"  [{flag}] {v:11s} missing={verdict['n_missing']} extra={verdict['n_extra']}")

    # --- Timing across all versions ----------------------------------------
    print("\n[Timing] grid creation (filtering) across versions")
    results = run_comparative_benchmark(versions, wsi_path=args.wsi,
                                        patch_size=args.patch_size,
                                        stride=args.stride, iterations=args.iterations)
    print("\n" + generate_performance_report(results))

    # --- Speedup hierarchy verdict -----------------------------------------
    base_t = results["v0a_mono"].min
    multi_t = results["v0b_multi"].min
    best_gpu = min((results[v] for v in list_versions("gpu")), key=lambda m: m.min)
    print("\n[Hierarchy]")
    print(f"  v0a mono : {base_t:.3f}s  (1.00x reference)")
    print(f"  v0b multi: {multi_t:.3f}s  ({base_t / multi_t:.2f}x vs v0a)")
    print(f"  best GPU : {best_gpu.version} {best_gpu.min:.3f}s  "
          f"({base_t / best_gpu.min:.2f}x vs v0a, {multi_t / best_gpu.min:.2f}x vs v0b)")
    print("\n  Success criteria from the plan:")
    print(f"    - v0b >= 3x v0a ?  {'YES' if base_t / multi_t >= 3 else 'NO'} "
          f"({base_t / multi_t:.2f}x)")
    print(f"    - best GPU >= 2x v0b ?  {'YES' if multi_t / best_gpu.min >= 2 else 'NO'} "
          f"({multi_t / best_gpu.min:.2f}x)")
    print("\n  (If GPU does not beat CPU here, that is a finding: the workload is")
    print("   I/O-bound -- patch decode dominates the trivial filter compute.)")


if __name__ == "__main__":
    main()
