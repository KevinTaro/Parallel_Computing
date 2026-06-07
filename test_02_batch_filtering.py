"""
test_02_batch_filtering.py  --  Phase 2: Batch / Grid Filtering Scaling

Times full grid creation (the real filtering workload) at several strides, which
yields increasing candidate-patch counts. Reports throughput (patches/sec) per
version so the break-even point -- where a GPU version overtakes the CPU
baselines -- becomes visible, and confirms every version still agrees with v0a.

    python test_02_batch_filtering.py --strides 8192,4096,2048 \
        --versions v0a_mono,v0b_multi,v2_batch,v4_pinned,v7_memopt
"""
import argparse

import openslide

from benchmark_runner import run_comparative_benchmark
from numerical_validation import validate_against_baseline
from test_performance_framework import CuPyTester, DEFAULT_WSI, list_versions


def candidate_count(wsi_path, patch_size, stride):
    with openslide.OpenSlide(wsi_path) as slide:
        w, h = slide.level_dimensions[0]
    return sum(1 for y in range(0, h, stride) for x in range(0, w, stride)
               if x + patch_size <= w and y + patch_size <= h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--strides", default="8192,4096,2048")
    ap.add_argument("--versions", default="v0a_mono,v0b_multi,v2_batch,v4_pinned,v7_memopt")
    ap.add_argument("--iterations", type=int, default=2)
    args = ap.parse_args()

    strides = [int(s) for s in args.strides.split(",")]
    versions = list_versions() if args.versions == "all" else [v.strip() for v in args.versions.split(",")]

    print("=" * 80)
    print(" PHASE 2: BATCH / GRID FILTERING SCALING")
    print(f" WSI={args.wsi} patch={args.patch_size}")
    print("=" * 80)

    throughput_table = {v: [] for v in versions}
    counts = []
    for stride in strides:
        n = candidate_count(args.wsi, args.patch_size, stride)
        counts.append(n)
        print(f"\n--- stride={stride} -> {n} candidate patches ---")

        # correctness check at this stride
        tester = CuPyTester(wsi_path=args.wsi, patch_size=args.patch_size, stride=stride)
        base = tester.build("v0a_mono").coordinates
        for v in versions:
            if v == "v0a_mono":
                continue
            ok = validate_against_baseline(base, tester.build(v).coordinates)["passed"]
            if not ok:
                print(f"    [WARN] {v} disagrees with baseline at stride {stride}")

        results = run_comparative_benchmark(versions, wsi_path=args.wsi,
                                            patch_size=args.patch_size,
                                            stride=stride, iterations=args.iterations)
        for v in versions:
            throughput_table[v].append(results[v].throughput)

    print("\n" + "=" * 80)
    print(" THROUGHPUT (patches/sec) vs candidate count")
    print("=" * 80)
    header = "version    | " + " | ".join(f"{c:>9d}" for c in counts)
    print(header)
    print("-" * len(header))
    for v in versions:
        row = " | ".join(f"{t:9.1f}" for t in throughput_table[v])
        print(f"{v:11s}| {row}")
    print("\n Rising throughput with candidate count + GPU overtaking CPU = break-even reached.")


if __name__ == "__main__":
    main()
