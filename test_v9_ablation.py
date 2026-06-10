"""
test_v9_ablation.py  --  v9 Ablation Study (per-layer contribution)

Answers the plan's question "which optimizations matter most?" by timing v9 with
individual layers toggled off, and by progressively stacking layers from a
minimal GPU baseline up to the full v9. Reports total grid-creation time, pure
GPU kernel time, and peak memory for each configuration, plus the delta each
layer contributes.

    python test_v9_ablation.py --stride 2048 --iterations 3 --batch 64
"""
import argparse

from numerical_validation import validate_against_baseline
from test_performance_framework import CuPyTester, reset_gpu_memory, DEFAULT_WSI


def _row(name, m):
    kt = m.kernel_min * 1e3 if m.kernel_times else float("nan")
    return (f"{name:26s} | {m.min:7.3f}s | {kt:8.1f}ms | "
            f"{m.throughput:8.1f} p/s | {m.peak_gpu_bytes/1e6:7.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    tester = CuPyTester(wsi_path=args.wsi, patch_size=args.patch_size, stride=args.stride)
    base_coords = tester.build("v0a_mono").coordinates
    bs = args.batch

    print("=" * 80)
    print(" v9 ABLATION STUDY")
    print(f" WSI={args.wsi} patch={args.patch_size} stride={args.stride} batch={bs}")
    print("=" * 80)

    def run(name, **overrides):
        reset_gpu_memory()
        m = tester.time_grid_creation("v9_ultimate", iterations=args.iterations,
                                      batch_size=bs, **overrides)
        # correctness must hold for every non-fp16 configuration
        ds = tester.build("v9_ultimate", batch_size=bs, **overrides)
        ok = validate_against_baseline(base_coords, ds.coordinates)["passed"]
        print(_row(name, m) + f" | {'OK' if ok else 'MISMATCH'}")
        return m

    # --- Part A: leave-one-out (full v9 minus one layer) -------------------
    print("\n[A] Leave-one-out (full v9, then remove a single layer)")
    print(f"{'config':26s} | {'total':>8s} | {'kernel':>8s} | {'throughput':>10s} | {'peak':>7s}")
    print("-" * 80)
    full = run("FULL v9 (all layers)")
    run("  - async (serial)", enable_async=False)
    run("  - pinned memory", enable_pinned_memory=False)
    run("  - early-exit", enable_early_exit=False)
    run("  + mixed precision*", enable_mixed_precision=True)

    # --- Part B: progressive stack ----------------------------------------
    print("\n[B] Progressive stack (build v9 up one layer at a time)")
    print(f"{'config':26s} | {'total':>8s} | {'kernel':>8s} | {'throughput':>10s} | {'peak':>7s}")
    print("-" * 80)
    run("L3 batch only", enable_async=False, enable_pinned_memory=False, enable_early_exit=False)
    run("+L1 pinned", enable_async=False, enable_pinned_memory=True, enable_early_exit=False)
    run("+L7 early-exit", enable_async=False, enable_pinned_memory=True, enable_early_exit=True)
    run("+L2 async (= FULL v9)", enable_async=True, enable_pinned_memory=True, enable_early_exit=True)

    print("\n Notes:")
    print("  * mixed precision is fp16 (Layer 4); it may differ from baseline on")
    print("    boundary patches -- 'MISMATCH' there is the documented trade-off, not a bug.")
    print("  Kernel time = CUDA-event GPU time (transfer+compute), summed over batches.")
    print("  If kernel << total, the run is dominated by OpenSlide I/O, not the GPU.")


if __name__ == "__main__":
    main()
