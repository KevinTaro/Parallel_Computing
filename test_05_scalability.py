"""
test_05_scalability.py  --  Phase 5: Scalability

Varies patch size (and stride to match) and measures how grid-creation time and
throughput scale for the CPU baselines vs a representative GPU version. Reveals
how the compute/I-O balance shifts as the per-patch work grows.

    python test_05_scalability.py --sizes 256,512,1024 \
        --versions v0a_mono,v0b_multi,v2_batch
"""
import argparse

from benchmark_runner import run_comparative_benchmark
from test_performance_framework import DEFAULT_WSI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--sizes", default="256,512,1024")
    ap.add_argument("--versions", default="v0a_mono,v0b_multi,v2_batch")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--stride-factor", type=float, default=4.0,
                    help="stride = patch_size * factor (controls candidate count)")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    versions = [v.strip() for v in args.versions.split(",")]

    print("=" * 80)
    print(" PHASE 5: SCALABILITY (vs patch size)")
    print(f" WSI={args.wsi} stride=patch*{args.stride_factor}")
    print("=" * 80)

    table = {v: [] for v in versions}
    for size in sizes:
        stride = int(size * args.stride_factor)
        print(f"\n--- patch_size={size} stride={stride} ---")
        results = run_comparative_benchmark(versions, wsi_path=args.wsi,
                                            patch_size=size, stride=stride,
                                            iterations=args.iterations)
        for v in versions:
            table[v].append((results[v].min, results[v].throughput))

    print("\n" + "=" * 80)
    print(" min time (s) by patch size")
    print("=" * 80)
    hdr = "version    | " + " | ".join(f"{s:>10d}px" for s in sizes)
    print(hdr); print("-" * len(hdr))
    for v in versions:
        print(f"{v:11s}| " + " | ".join(f"{t:10.3f}  " for t, _ in table[v]))

    print("\n throughput (patches/sec) by patch size")
    print("-" * len(hdr))
    for v in versions:
        print(f"{v:11s}| " + " | ".join(f"{tp:10.1f}  " for _, tp in table[v]))


if __name__ == "__main__":
    main()
