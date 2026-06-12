"""
data_loader_v15_gpu_compute_multi.py

v15: GENERAL GPU COMPUTE on a MULTI-CORE feed  (ablation cell: multi x general)
===============================================================================
Research question: *with no specialized decoder and the filter on the GPU's
general CUDA cores, how far does CPU multi-core parallelism push a codec-generic
pipeline -- the portable "works on any TIFF" upper bound?*

Base: data_loader_v0b_multi_baseline.py (all-cores CPU parallelism).
Identical to v14 EXCEPT the generic decode (``openslide.read_region``) is run by
a pool of reader threads -- libopenslide releases the GIL inside the decode, so
the CPU cores decode patches in parallel. Each filled batch is then transferred
to the GPU and filtered with the same general-compute CuPy kernel. No nvJPEG, so
it stays codec-generic (JPEG, LZW, deflate, uncompressed -- anything OpenSlide
reads).

This is the portable counterpart to the specialized v13: v15 attacks the decode
bottleneck with CPU cores (general, universal) while v13 attacks it with the
GPU's fixed-function JPEG unit (faster, JPEG-only).

Compare v15 vs v14 -> multi-core vs mono-core decode (the pure CPU-parallelism win).
Compare v15 vs v13 -> general CPU-parallel decode vs specialized GPU decode.
Compare v15 vs v0b -> what moving just the filter to the GPU adds over pure CPU.

Results are bit-identical to v0a/v0b (same CPU decode, same luma arithmetic).
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torchvision import transforms

_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b',
    out_params='uint8 gray',
    operation='gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;',
    name='pil_luma_uint8',
)


class WSISlidingWindowDataset(Dataset):
    """Multi-core generic decode + general GPU-compute filter (v0b base)."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 512,
                 num_readers: Optional[int] = None,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
        self.num_readers = num_readers or (os.cpu_count() or 4)
        self.verbose = verbose
        self.read_time = 0.0        # parallel CPU decode (wall time of the read stage)
        self.transfer_time = 0.0
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            print(f"[*] v15 grid done in {self.grid_creation_time:.2f}s "
                  f"(cpu-decode {self.read_time:.3f} + xfer {self.transfer_time:.3f} + "
                  f"gpu-filter {self.kernel_time:.3f}); kept {len(self.coordinates)}")
        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    coords.append((x, y))
        return coords

    def _filter_batch_gpu(self, host_batch: np.ndarray, n: int) -> np.ndarray:
        ps = self.patch_size
        total_pixels = ps * ps
        t0 = time.perf_counter()
        dev = cp.asarray(host_batch[:n])
        cp.cuda.Stream.null.synchronize()
        self.transfer_time += time.perf_counter() - t0

        e0, e1 = cp.cuda.Event(), cp.cuda.Event(); e0.record()
        gray = cp.empty((n, ps, ps), dtype=cp.uint8)
        _luma_kernel(dev[..., 0], dev[..., 1], dev[..., 2], gray)
        w = cp.count_nonzero(gray > self.white_pixel_threshold, axis=(1, 2))
        b = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2))
        keep = (w.astype(cp.float64) / total_pixels < self.rejection_ratio) & \
               (b.astype(cp.float64) / total_pixels < self.rejection_ratio)
        e1.record(); e1.synchronize()
        self.kernel_time += cp.cuda.get_elapsed_time(e0, e1) / 1000.0
        self.peak_gpu_bytes = max(self.peak_gpu_bytes,
                                  cp.get_default_memory_pool().used_bytes())
        return cp.asnumpy(keep)

    def _create_grid(self) -> List[Tuple[int, int]]:
        ps, bs = self.patch_size, self.batch_size
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v15 (multi CPU decode x{self.num_readers} + general GPU filter, "
                  f"batch={bs}): {len(candidates)} candidate patches")

        host_batch = np.empty((bs, ps, ps, 3), dtype=np.uint8)
        coordinates = []

        # Thread-local OpenSlide handles: read_region releases the GIL inside
        # libopenslide, so the CPU cores decode patches truly in parallel.
        tls = threading.local()
        handles, handles_lock = [], threading.Lock()

        def decode_into(buf: np.ndarray, slot: int, x: int, y: int) -> None:
            slide = getattr(tls, "slide", None)
            if slide is None:
                slide = openslide.OpenSlide(self.wsi_path)
                tls.slide = slide
                with handles_lock:
                    handles.append(slide)
            buf[slot] = np.asarray(slide.read_region((x, y), 0, (ps, ps)))[:, :, :3]

        batches = [candidates[s:s + bs] for s in range(0, len(candidates), bs)]
        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                for chunk in batches:
                    # ---- MULTI-CORE decode: pool fills the batch in parallel ----
                    t0 = time.perf_counter()
                    list(pool.map(lambda a: decode_into(host_batch, a[0], a[1][0], a[1][1]),
                                  list(enumerate(chunk))))
                    self.read_time += time.perf_counter() - t0
                    keep = self._filter_batch_gpu(host_batch, len(chunk))
                    coordinates.extend(c for c, k in zip(chunk, keep) if k)
        finally:
            for h in handles:
                h.close()
        return coordinates

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        with openslide.OpenSlide(self.wsi_path) as slide:
            x, y = self.coordinates[idx]
            patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert('RGB')
            t = self.transform(patch) if self.transform else transforms.ToTensor()(patch)
            return t, (x, y)


def run_test(wsi_path: str = "data/S114-80954A-Her2(3+).tiff"):
    print("==== v15 general GPU compute (multi-core decode) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | cpu-decode {ds.read_time:.3f}s xfer {ds.transfer_time:.3f}s "
          f"gpu-filter {ds.kernel_time:.3f}s")


if __name__ == '__main__':
    run_test()
