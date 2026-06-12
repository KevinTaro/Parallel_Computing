"""
benchmark_decode_ablation.py

Controlled study of WHERE the work runs, decoupling two variables:

    CPU orchestration : mono-core      vs  multi-core
    GPU strategy      : none / general-compute / specialized nvJPEG

The 2x2 GPU-vs-orchestration matrix plus the two pure-CPU baselines:

                      mono-core feed        multi-core feed
    CPU only          v0a                   v0b
    general GPU comp.  v14                   v15
    nvJPEG decode      v12                   v13

For each (slide, version) it records the grid-creation wall time (best of N),
the per-stage breakdown the loader exposes, and the kept-patch count, then
checks every version's kept set against the v0a reference.

Usage:
    python benchmark_decode_ablation.py            # default slide set
    python benchmark_decode_ablation.py <wsi> ...  # explicit slides
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from test_performance_framework import CuPyTester, ValidationChecker  # noqa: E402

# (key, reps) -- mono CPU-decode versions are slow, so fewer reps.
MATRIX = [
    ("v0a_mono",     1),   # mono CPU only        (reference)
    ("v0b_multi",    2),   # multi CPU only
    ("v14_cmp_mono", 1),   # mono + general GPU compute
    ("v15_cmp_multi",2),   # multi + general GPU compute
    ("v12_dec_mono", 2),   # mono + nvJPEG
    ("v13_dec_multi",2),   # multi + nvJPEG
]

DEFAULT_SLIDES = [
    "data/S114-80954A-Her2(3+).tiff",   # small  (462 candidates)
    "data/S114-80969A-Her2(1+).tiff",   # medium
    "data/S114-82742C-Her2(4B5) 20x.tiff",  # large (~4900 candidates)
]


def _stage_str(ds) -> str:
    parts = []
    for attr, label in (("read_time", "read"), ("decode_time", "decode"),
                        ("transfer_time", "xfer"), ("kernel_time", "filter")):
        v = getattr(ds, attr, None)
        if v:
            parts.append(f"{label} {v:.3f}")
    return ", ".join(parts) if parts else "-"


def bench_one(tester, version, reps):
    tester.build(version)                       # warmup: CUDA ctx + JIT + nvJPEG init
    best, ds = None, None
    for _ in range(reps):
        t0 = time.perf_counter()
        ds = tester.build(version)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, ds


def main(slides):
    report = {}
    for wsi in slides:
        name = wsi.split("/")[-1]
        print(f"\n{'='*78}\n{name}\n{'='*78}")
        tester = CuPyTester(wsi_path=wsi, verbose=False)
        ref_coords = None
        rows = []
        for version, reps in MATRIX:
            try:
                best, ds = bench_one(tester, version, reps)
            except Exception as e:
                print(f"  {version:15s} FAILED: {type(e).__name__}: {str(e)[:60]}")
                continue
            if version == "v0a_mono":
                ref_coords = ds.coordinates
                corr = "reference"
            else:
                c = ValidationChecker.compare(ref_coords, ds.coordinates)
                corr = "identical" if c["identical_set"] else \
                    f"DIFF -{c['n_missing']}/+{c['n_extra']}"
            speedup = (rows[0]["time"] / best) if rows else 1.0
            row = dict(version=version, time=best, kept=len(ds.coordinates),
                       stages=_stage_str(ds), corr=corr,
                       peak_mb=getattr(ds, "peak_gpu_bytes", 0) / 1e6)
            rows.append(row)
            print(f"  {version:15s} {best:7.3f}s  x{rows[0]['time']/best:5.2f} vs v0a | "
                  f"kept {row['kept']:5d} | {corr:12s} | {row['stages']}")
        report[name] = rows
    with open("benchmark_decode_ablation_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n[*] wrote benchmark_decode_ablation_results.json")
    return report


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_SLIDES)
