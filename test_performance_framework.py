"""
test_performance_framework.py

Core testing infrastructure shared by every benchmark / validation script.

Provides:
  - IMPLEMENTATIONS : the single source of truth registry of all 10 runnable
    data-loader versions (v0a, v0b, v1-v7), how to import them and their
    version-specific kwargs.
  - PerformanceMetrics : timing/memory/throughput record for one measurement.
  - CuPyTester         : build a dataset from any version and time grid creation.
  - ValidationChecker  : compare kept-coordinate sets between versions.
  - BenchmarkRunner    : run a version several times and aggregate metrics.

Everything is import-safe: importing this module does NOT touch the GPU or read
any WSI. GPU work only happens when you actually run a version.
"""
import importlib
import statistics
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Registry: the one place that knows about every runnable version.
# key -> (module, class, category, version-specific default kwargs)
# ---------------------------------------------------------------------------
"""IMPLEMENTATIONS: Dict[str, dict] = {
    "v0a_mono":   dict(module="data_loader_v0a_mono_baseline",       category="cpu", kwargs={}),
    "v0b_multi":  dict(module="data_loader_v0b_multi_baseline",      category="cpu", kwargs={}),
    "v1_full":    dict(module="data_loader_v1_cupy_full",            category="gpu", kwargs={}),
    "v2_batch":   dict(module="data_loader_v2_cupy_batch",           category="gpu", kwargs={"batch_size": 32}),
    "v3_hybrid":  dict(module="data_loader_v3_cupy_hybrid",          category="gpu", kwargs={"batch_size": 32, "gpu_threshold": 16}),
    "v4_pinned":  dict(module="data_loader_v4_cupy_pinned_memory",   category="gpu", kwargs={"batch_size": 32}),
    "v5_async":   dict(module="data_loader_v5_cupy_async",           category="gpu", kwargs={"batch_size": 32}),
    "v6_mixed":   dict(module="data_loader_v6_cupy_mixed_precision", category="gpu", kwargs={"batch_size": 32}),
    "v7_memopt":  dict(module="data_loader_v7_cupy_memory_optimized", category="gpu", kwargs={"chunk_size": 8}),
    "v8_4060":    dict(module="data_loader_v8_cupy_optimized_4060",   category="gpu", kwargs={"batch_size": 512}),
    "v9_ultimate": dict(module="data_loader_v9_ultimate_gpu",          category="gpu", kwargs={"batch_size": 32}),
    "v10_par_io": dict(module="data_loader_v10_parallel_io_gpu",       category="gpu", kwargs={"batch_size": 64, "num_readers": os.cpu_count()}),
    "v11_gpudec": dict(module="data_loader_v11_gpu_decode_5090",       category="gpu", kwargs={"batch_size": 2048}),
    "v12_dec_mono": dict(module="data_loader_v12_gpu_decode_mono",     category="gpu", kwargs={"batch_size": 2048}),
    "v13_dec_multi": dict(module="data_loader_v13_gpu_decode_multi",   category="gpu", kwargs={"batch_size": 2048}),
    "v14_cmp_mono": dict(module="data_loader_v14_gpu_compute_mono",    category="gpu", kwargs={"batch_size": 1024}),
    "v15_cmp_multi": dict(module="data_loader_v15_gpu_compute_multi",  category="gpu", kwargs={"batch_size": 1024}),
    "v16_opt_mono": dict(module="data_loader_v16_cuda_opt_mono",       category="gpu", kwargs={"batch_size": 2048}),
    "v17_opt_multi": dict(module="data_loader_v17_cuda_opt_multi",     category="gpu", kwargs={"batch_size": 2048}),
    "v18_ult_mono": dict(module="data_loader_v18_cuda_ultimate_mono",  category="gpu", kwargs={"batch_size": 2048}),
    "v19_ult_multi": dict(module="data_loader_v19_cuda_ultimate_multi", category="gpu", kwargs={"batch_size": 2048}),
}"""

### for 5090
IMPLEMENTATIONS: Dict[str, dict] = {
    "v0a_mono":   dict(module="data_loader_v0a_mono_baseline",       category="cpu", kwargs={}),
    "v0b_multi":  dict(module="data_loader_v0b_multi_baseline",      category="cpu", kwargs={}),
    "v12_dec_mono": dict(module="data_loader_v12_gpu_decode_mono",     category="gpu", kwargs={"batch_size": 12288}),
    "v13_dec_multi": dict(module="data_loader_v13_gpu_decode_multi",   category="gpu", kwargs={"batch_size": 12288}),
    "v14_cmp_mono": dict(module="data_loader_v14_gpu_compute_mono",    category="gpu", kwargs={"batch_size": 6144}),
    "v15_cmp_multi": dict(module="data_loader_v15_gpu_compute_multi",  category="gpu", kwargs={"batch_size": 6144}),
    "v16_opt_mono": dict(module="data_loader_v16_cuda_opt_mono",       category="gpu", kwargs={"batch_size": 131072}),
    "v17_opt_multi": dict(module="data_loader_v17_cuda_opt_multi",     category="gpu", kwargs={"batch_size": 131072}),
    "v18_ult_mono": dict(module="data_loader_v18_cuda_ultimate_mono",  category="gpu", kwargs={"batch_size": 4096}),
    "v19_ult_multi": dict(module="data_loader_v19_cuda_ultimate_multi", category="gpu", kwargs={"batch_size": 4096}),
}


CLASS_NAME = "WSISlidingWindowDataset"
DEFAULT_WSI = "data/S114-82742C-Her2(4B5) 20x.tiff"   # smallest slide -> fast iteration


def list_versions(category: Optional[str] = None) -> List[str]:
    """Return version keys, optionally filtered by 'cpu' / 'gpu'."""
    return [k for k, v in IMPLEMENTATIONS.items()
            if category is None or v["category"] == category]


def load_dataset_class(version: str):
    """Import a version module and return its WSISlidingWindowDataset class."""
    if version not in IMPLEMENTATIONS:
        raise KeyError(f"Unknown version '{version}'. Known: {list(IMPLEMENTATIONS)}")
    module = importlib.import_module(IMPLEMENTATIONS[version]["module"])
    return getattr(module, CLASS_NAME)


# ---------------------------------------------------------------------------
# GPU memory helpers (no-op if CuPy / CUDA unavailable)
# ---------------------------------------------------------------------------
def get_gpu_memory_used() -> int:
    """Bytes currently allocated by the CuPy default pool (0 if unavailable)."""
    try:
        import cupy as cp
        return int(cp.get_default_memory_pool().used_bytes())
    except Exception:
        return 0


def reset_gpu_memory() -> None:
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


@dataclass
class PerformanceMetrics:
    """Aggregated result of timing one version on one configuration."""
    version: str
    n_candidates: int = 0
    n_kept: int = 0
    times: List[float] = field(default_factory=list)
    kernel_times: List[float] = field(default_factory=list)
    peak_gpu_bytes: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times) if self.times else float("nan")

    @property
    def std(self) -> float:
        return statistics.pstdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else float("nan")

    @property
    def max(self) -> float:
        return max(self.times) if self.times else float("nan")

    @property
    def kernel_mean(self) -> float:
        return statistics.mean(self.kernel_times) if self.kernel_times else float("nan")

    @property
    def kernel_min(self) -> float:
        return min(self.kernel_times) if self.kernel_times else float("nan")

    @property
    def throughput(self) -> float:
        """Candidate patches filtered per second (uses best/min time)."""
        return self.n_candidates / self.min if self.times and self.min > 0 else 0.0

    def summary(self) -> str:
        return (f"{self.version:11s} | mean {self.mean:7.3f}s  min {self.min:7.3f}s  "
                f"std {self.std:6.3f}  | kept {self.n_kept}/{self.n_candidates} | "
                f"{self.throughput:7.1f} patch/s | peak {self.peak_gpu_bytes/1e6:6.1f}MB")

    def summary_with_kernel(self) -> str:
        if not self.kernel_times:
            return self.summary()
        kernel_pct = (self.kernel_min / self.min * 100) if self.min > 0 else 0
        return (f"{self.version:11s} | total {self.min:7.3f}s  kernel {self.kernel_min:7.3f}s "
                f"({kernel_pct:5.1f}%) | peak {self.peak_gpu_bytes/1e6:6.1f}MB")


class CuPyTester:
    """Unified interface to instantiate and time any registered version."""

    def __init__(self, wsi_path: str = DEFAULT_WSI, patch_size: int = 1024,
                 stride: int = 1024, verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.verbose = verbose

    def build(self, version: str, **overrides):
        """Instantiate a dataset for `version`; returns the dataset object."""
        cls = load_dataset_class(version)
        kwargs = dict(IMPLEMENTATIONS[version]["kwargs"])
        kwargs.update(overrides)
        return cls(wsi_path=self.wsi_path, patch_size=self.patch_size,
                   stride=self.stride, verbose=self.verbose, **kwargs)

    def time_grid_creation(self, version: str, iterations: int = 3,
                           warmup: bool = True, **overrides) -> PerformanceMetrics:
        """Build the dataset `iterations` times, timing grid creation each time."""
        if IMPLEMENTATIONS[version]["category"] == "gpu":
            reset_gpu_memory()
        if warmup:
            # First build pays CUDA-context init + kernel JIT; don't count it.
            ds = self.build(version, **overrides)
            n_candidates = len(ds._generate_candidate_coords())
        else:
            n_candidates = 0

        metrics = PerformanceMetrics(version=version)
        peak = 0
        n_kept = 0
        for _ in range(iterations):
            reset_gpu_memory()
            t0 = time.perf_counter()
            ds = self.build(version, **overrides)
            metrics.times.append(time.perf_counter() - t0)
            if hasattr(ds, "kernel_time"):
                metrics.kernel_times.append(ds.kernel_time)
            n_kept = len(ds.coordinates)
            peak = max(peak, get_gpu_memory_used(),
                       getattr(ds, "peak_gpu_bytes", 0))
        metrics.n_candidates = n_candidates or len(ds._generate_candidate_coords())
        metrics.n_kept = n_kept
        metrics.peak_gpu_bytes = peak
        return metrics


class ValidationChecker:
    """Verify that two versions select the same patches."""

    @staticmethod
    def coords_of(dataset) -> List[Tuple[int, int]]:
        return list(dataset.coordinates)

    @staticmethod
    def compare(baseline_coords, test_coords) -> dict:
        b, t = set(baseline_coords), set(test_coords)
        missing = b - t          # in baseline, not in test (false negatives)
        extra = t - b            # in test, not in baseline (false positives)
        return {
            "identical_set": b == t,
            "identical_order": list(baseline_coords) == list(test_coords),
            "n_baseline": len(b),
            "n_test": len(t),
            "n_missing": len(missing),
            "n_extra": len(extra),
            "missing": sorted(missing)[:10],
            "extra": sorted(extra)[:10],
        }


class BenchmarkRunner:
    """Run a set of versions through CuPyTester and collect metrics."""

    def __init__(self, tester: CuPyTester):
        self.tester = tester

    def run(self, versions: List[str], iterations: int = 3,
            per_version_overrides: Optional[Dict[str, dict]] = None
            ) -> Dict[str, PerformanceMetrics]:
        per_version_overrides = per_version_overrides or {}
        results: Dict[str, PerformanceMetrics] = {}
        for v in versions:
            overrides = per_version_overrides.get(v, {})
            print(f"  -> timing {v} ...", flush=True)
            results[v] = self.tester.time_grid_creation(v, iterations=iterations, **overrides)
            if results[v].kernel_times:
                print("     " + results[v].summary_with_kernel(), flush=True)
            else:
                print("     " + results[v].summary(), flush=True)
        return results


if __name__ == "__main__":
    # Smoke test: confirm every version imports and exposes the class.
    print("Registered versions:")
    for key, info in IMPLEMENTATIONS.items():
        try:
            load_dataset_class(key)
            ok = "ok"
        except Exception as e:  # pragma: no cover
            ok = f"FAILED: {e}"
        print(f"  {key:11s} <- {info['module']:38s} [{ok}]")
