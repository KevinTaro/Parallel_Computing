"""
test_04_memory_profiling.py  --  Phase 4: GPU Memory Analysis

Profiles GPU memory for each GPU version during grid creation: peak pool usage
(sampled by a background poller while the build runs), bytes-per-candidate, and
a leak check that the pool returns to baseline after repeated builds.

Relevant because the test GPU has only 3 GB: v2's uint32 grayscale temporary is
4x the input batch, while v7's fused uint8 kernel + small chunks keep the
footprint tiny.

    python test_04_memory_profiling.py --stride 1024
"""
import argparse
import threading
import time

import cupy as cp

from test_performance_framework import (
    CuPyTester, DEFAULT_WSI, list_versions, reset_gpu_memory,
)


class PeakSampler(threading.Thread):
    """Poll the CuPy pool's used_bytes() to capture a peak during a build."""

    def __init__(self, interval=0.001):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stopped = threading.Event()   # NB: don't name this _stop (shadows Thread._stop)
        self._pool = cp.get_default_memory_pool()

    def run(self):
        while not self._stopped.is_set():
            self.peak = max(self.peak, self._pool.used_bytes())
            time.sleep(self.interval)

    def stop(self):
        self._stopped.set()
        self.join(timeout=1.0)
        self.peak = max(self.peak, self._pool.used_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    args = ap.parse_args()

    print("=" * 80)
    print(" PHASE 4: GPU MEMORY PROFILING")
    print(f" WSI={args.wsi} patch={args.patch_size} stride={args.stride}")
    print("=" * 80)

    tester = CuPyTester(wsi_path=args.wsi, patch_size=args.patch_size, stride=args.stride)
    n_candidates = len(tester.build("v0a_mono")._generate_candidate_coords())

    print(f"\n candidates={n_candidates}")
    print(f"{'version':11s} | {'peak MB':>9s} | {'MB/patch':>9s} | {'reported MB':>11s}")
    print("-" * 56)

    for v in list_versions("gpu"):
        reset_gpu_memory()
        sampler = PeakSampler()
        sampler.start()
        ds = tester.build(v)
        sampler.stop()
        peak_mb = sampler.peak / 1e6
        reported = getattr(ds, "peak_gpu_bytes", 0) / 1e6
        print(f"{v:11s} | {peak_mb:9.1f} | {peak_mb/max(1,n_candidates):9.3f} | "
              f"{(reported if reported else float('nan')):11.1f}")

    # --- leak check ---------------------------------------------------------
    print("\n[Leak check] pool used_bytes after repeated builds (should stay flat):")
    reset_gpu_memory()
    pool = cp.get_default_memory_pool()
    for i in range(3):
        tester.build("v2_batch")
        reset_gpu_memory()
        print(f"  build {i+1}: used={pool.used_bytes()/1e6:.2f} MB  total={pool.total_bytes()/1e6:.2f} MB")
    print("\n Flat used_bytes across builds => no leak. v7/v4/v5 free buffers explicitly.")


if __name__ == "__main__":
    main()
