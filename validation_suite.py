"""
validation_suite.py

End-to-end correctness gate for the whole framework.

Three checks:
  1. Grayscale fidelity   -- GPU integer luma vs PIL `convert('L')` (must be
     bit-exact); fp16 luma vs PIL (expected small drift).
  2. Filtering agreement  -- every version's kept-coordinate set vs the v0a
     mono baseline, with precision/recall and Jaccard.
  3. Transform pipeline   -- __getitem__ returns a correctly shaped tensor.

Run:
    python validation_suite.py                       # smallest slide, stride 1024
    python validation_suite.py --wsi ... --stride 1024
"""
import argparse

import numpy as np
import openslide
from PIL import Image

import cupy as cp

from numerical_validation import (
    check_numerical_stability, compare_filtering_results, grayscale_reference,
    print_validation, validate_against_baseline,
)
from test_performance_framework import (
    CuPyTester, DEFAULT_WSI, IMPLEMENTATIONS, list_versions,
)

_LUMA = (19595, 38470, 7471)


def _gpu_int_luma(rgb):
    g = cp.asarray(rgb).astype(cp.uint32)
    return cp.asnumpy(((g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1]
                        + g[..., 2] * _LUMA[2] + 32768) >> 16).astype(cp.uint8))


def _gpu_fp16_luma(rgb):
    g = cp.asarray(rgb).astype(cp.float16)
    gray = (g[..., 0] * cp.float16(0.299) + g[..., 1] * cp.float16(0.587)
            + g[..., 2] * cp.float16(0.114))
    return cp.asnumpy(cp.rint(gray).astype(cp.uint8))


def check_grayscale(wsi_path, patch_size, n_samples=4):
    print("\n[1] Grayscale fidelity (GPU vs PIL convert('L'))")
    print("-" * 70)
    with openslide.OpenSlide(wsi_path) as slide:
        w, h = slide.level_dimensions[0]
        coords = [(min(i * patch_size, w - patch_size), min(i * patch_size, h - patch_size))
                  for i in range(n_samples)]
        all_int_exact, fp16_stats = True, []
        for (x, y) in coords:
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            pil_L = np.asarray(patch.convert("L"))
            rgb = np.asarray(patch)[:, :, :3]

            cpu_int = grayscale_reference(rgb)
            gpu_int = _gpu_int_luma(rgb)
            gpu_fp16 = _gpu_fp16_luma(rgb)

            s_cpu = check_numerical_stability(pil_L, cpu_int)
            s_gpu = check_numerical_stability(pil_L, gpu_int)
            s_f16 = check_numerical_stability(pil_L, gpu_fp16)
            all_int_exact = all_int_exact and s_cpu["exact_match"] and s_gpu["exact_match"]
            fp16_stats.append(s_f16["max_abs_error"])
            print(f"  ({x:6d},{y:6d}) CPU-int exact={s_cpu['exact_match']}  "
                  f"GPU-int exact={s_gpu['exact_match']}  "
                  f"fp16 max|err|={s_f16['max_abs_error']:.1f} "
                  f"match={s_f16['match_rate']*100:5.1f}%")
        print(f"  => integer luma bit-exact vs PIL: {all_int_exact}")
        print(f"  => fp16 luma worst max|err| over samples: {max(fp16_stats):.1f} grey levels")
        return all_int_exact


def check_filtering(wsi_path, patch_size, stride):
    print("\n[2] Filtering agreement vs v0a mono baseline")
    print("-" * 70)
    tester = CuPyTester(wsi_path=wsi_path, patch_size=patch_size, stride=stride)
    baseline = tester.build("v0a_mono")
    base_coords = baseline.coordinates
    print(f"  baseline v0a_mono: kept {len(base_coords)} patches")

    all_pass = True
    for v in list_versions():
        if v == "v0a_mono":
            continue
        ds = tester.build(v)
        verdict = validate_against_baseline(base_coords, ds.coordinates)
        prf = compare_filtering_results(base_coords, ds.coordinates)
        print_validation(v, verdict)
        if not verdict["passed"]:
            all_pass = False
            print(f"        precision={prf['precision']:.4f} recall={prf['recall']:.4f} "
                  f"f1={prf['f1']:.4f}")
            if verdict["missing_sample"]:
                print(f"        missing e.g. {verdict['missing_sample']}")
            if verdict["extra_sample"]:
                print(f"        extra   e.g. {verdict['extra_sample']}")
    return all_pass


def check_transform(wsi_path, patch_size, stride):
    print("\n[3] Transform pipeline (__getitem__ shape/dtype)")
    print("-" * 70)
    tester = CuPyTester(wsi_path=wsi_path, patch_size=patch_size, stride=stride)
    ds = tester.build("v0a_mono")
    tensor, coords = ds[0]
    ok = tuple(tensor.shape) == (3, patch_size, patch_size)
    print(f"  v0a_mono[0]: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
          f"coords={coords} -> {'OK' if ok else 'BAD SHAPE'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    args = ap.parse_args()

    print("=" * 70)
    print(" VALIDATION SUITE")
    print(f" WSI={args.wsi} patch={args.patch_size} stride={args.stride}")
    print("=" * 70)

    gray_ok = check_grayscale(args.wsi, args.patch_size)
    filt_ok = check_filtering(args.wsi, args.patch_size, args.stride)
    tf_ok = check_transform(args.wsi, args.patch_size, args.stride)

    print("\n" + "=" * 70)
    print(f" RESULT: grayscale={'PASS' if gray_ok else 'FAIL'}  "
          f"filtering={'PASS' if filt_ok else 'FAIL'}  "
          f"transform={'PASS' if tf_ok else 'FAIL'}")
    print("=" * 70)
    print(" Note: v6 (fp16) may legitimately differ on patches whose ratio sits")
    print("       on the rejection boundary; that is the documented precision")
    print("       trade-off, not a bug.")


if __name__ == "__main__":
    main()
