"""
numerical_validation.py

Numerical correctness primitives used by the validation suite and tests.

These functions answer one question precisely: "does an optimized version make
the *same* keep/discard decisions as the mono-core baseline, and is its
intermediate arithmetic faithful?" They operate on plain data (coordinate lists
and arrays) so they have no GPU dependency themselves.

Functions
---------
validate_against_baseline(coords_baseline, coords_test)
    Compare two kept-coordinate sets; returns a structured verdict.
compare_filtering_results(coords_baseline, coords_test)
    Precision/recall-style breakdown of agreement between two versions.
check_numerical_stability(gray_baseline, gray_test, tolerance=0)
    Element-wise comparison of two grayscale arrays (L1 / Linf / match rate).
grayscale_reference(rgb)
    PIL-exact integer luma on the CPU, the ground truth all versions target.
"""
from typing import Iterable, List, Sequence, Tuple

import numpy as np

_LUMA = (19595, 38470, 7471)
_LUMA_ROUND = 32768


def grayscale_reference(rgb: np.ndarray) -> np.ndarray:
    """PIL `convert('L')`-exact integer luma. rgb: (..., 3) uint8 -> (...) uint8."""
    g = rgb.astype(np.uint32)
    gray = (g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2]
            + _LUMA_ROUND) >> 16
    return gray.astype(np.uint8)


def validate_against_baseline(coords_baseline: Sequence[Tuple[int, int]],
                              coords_test: Sequence[Tuple[int, int]]) -> dict:
    """Structured verdict on whether `coords_test` matches `coords_baseline`."""
    b, t = set(map(tuple, coords_baseline)), set(map(tuple, coords_test))
    missing = sorted(b - t)
    extra = sorted(t - b)
    n_union = len(b | t)
    jaccard = (len(b & t) / n_union) if n_union else 1.0
    return {
        "passed": b == t,
        "identical_order": list(map(tuple, coords_baseline)) == list(map(tuple, coords_test)),
        "n_baseline": len(b),
        "n_test": len(t),
        "n_agree": len(b & t),
        "n_missing": len(missing),     # baseline kept, test dropped
        "n_extra": len(extra),         # test kept, baseline dropped
        "jaccard": jaccard,
        "missing_sample": missing[:10],
        "extra_sample": extra[:10],
    }


def compare_filtering_results(coords_baseline: Sequence[Tuple[int, int]],
                              coords_test: Sequence[Tuple[int, int]]) -> dict:
    """Precision/recall of `coords_test` treating baseline as ground truth."""
    b, t = set(map(tuple, coords_baseline)), set(map(tuple, coords_test))
    tp = len(b & t)
    fp = len(t - b)
    fn = len(b - t)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1}


def check_numerical_stability(gray_baseline: np.ndarray, gray_test: np.ndarray,
                              tolerance: float = 0.0) -> dict:
    """Element-wise comparison of two grayscale arrays of equal shape."""
    a = gray_baseline.astype(np.float64)
    b = gray_test.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    n = a.size
    return {
        "max_abs_error": float(diff.max()) if n else 0.0,
        "mean_abs_error": float(diff.mean()) if n else 0.0,
        "l2_error": float(np.sqrt(np.sum(diff ** 2))),
        "match_rate": float(np.mean(diff <= tolerance)) if n else 1.0,
        "exact_match": bool(np.array_equal(a, b)),
        "within_tolerance": bool(np.all(diff <= tolerance)),
    }


def print_validation(version: str, verdict: dict) -> None:
    status = "PASS" if verdict["passed"] else "FAIL"
    print(f"  [{status}] {version:11s} "
          f"agree={verdict['n_agree']:4d}/{verdict['n_baseline']:<4d} "
          f"missing={verdict['n_missing']} extra={verdict['n_extra']} "
          f"jaccard={verdict['jaccard']:.4f}")


if __name__ == "__main__":
    # Self-check on synthetic data.
    base = [(0, 0), (0, 1024), (1024, 0)]
    same = [(1024, 0), (0, 0), (0, 1024)]    # same set, different order
    drop = [(0, 0), (0, 1024)]               # one missing
    print("identical set:", validate_against_baseline(base, same))
    print("one dropped  :", validate_against_baseline(base, drop))
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    g = grayscale_reference(rgb)
    print("grayscale self-stability:", check_numerical_stability(g, g))
