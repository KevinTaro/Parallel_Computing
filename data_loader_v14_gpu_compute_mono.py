"""
data_loader_v14_gpu_compute_mono.py

v14: GENERAL GPU COMPUTE on a MONO-CORE feed  (ablation cell: mono x general)
=============================================================================
Research question: *without any specialized decoder (no nvJPEG / NVDEC), using
ONLY the GPU's general-purpose CUDA cores for the compute, and keeping the
decode codec-generic so it fits ANY TIFF -- how much can pure parallelism help
when the CPU still does the decode on a single core?*

Base: data_loader_v0a_mono_baseline.py (single CPU thread, sequential scan).
Change vs v0a: the per-pixel tissue filter (luma + white/black thresholding)
moves from NumPy-on-CPU to a CuPy kernel on the GPU's general compute units. The
decode stays on the generic CPU path (``openslide.read_region``), so v14 works
on JPEG, LZW, deflate, uncompressed -- *any* OpenSlide-readable WSI, unlike the
nvJPEG versions (v12/v13) which require JPEG tiles.

What this isolates: the value of general GPU compute alone. Because the decode
is still single-threaded CPU work -- and that is the bottleneck -- the GPU only
accelerates the (already cheap) filter, and the host->device transfer of decoded
patches is pure overhead v0a never paid. So v14 is expected to land *near* v0a,
sometimes slightly slower: that null/!negative result IS the finding -- general
parallel compute cannot remove a decode bottleneck a specialized unit removes.

Results are bit-identical to v0a (same CPU decode, same luma arithmetic).
"""
import time
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
    """Mono-core generic decode + general GPU-compute filter (v0a base)."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 256,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
        self.verbose = verbose
        self.read_time = 0.0        # CPU decode (openslide, single thread)
        self.transfer_time = 0.0    # host -> device
        self.kernel_time = 0.0      # GPU general-compute filter
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
            print(f"[*] v14 grid done in {self.grid_creation_time:.2f}s "
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
        """Transfer n patches to GPU and run the general-compute luma filter."""
        ps = self.patch_size
        total_pixels = ps * ps
        t0 = time.perf_counter()
        dev = cp.asarray(host_batch[:n])             # host -> device (general copy)
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
            print(f"[*] v14 (mono CPU decode + general GPU filter, batch={bs}): "
                  f"{len(candidates)} candidate patches")

        host_batch = np.empty((bs, ps, ps, 3), dtype=np.uint8)
        coordinates = []
        # ---- MONO-CORE: one handle, strictly sequential decode ----
        with openslide.OpenSlide(self.wsi_path) as slide:
            chunk = []
            for (x, y) in candidates:
                t0 = time.perf_counter()
                arr = np.asarray(slide.read_region((x, y), 0, (ps, ps)))[:, :, :3]
                self.read_time += time.perf_counter() - t0
                host_batch[len(chunk)] = arr
                chunk.append((x, y))
                if len(chunk) == bs:
                    keep = self._filter_batch_gpu(host_batch, len(chunk))
                    coordinates.extend(c for c, k in zip(chunk, keep) if k)
                    chunk = []
            if chunk:
                keep = self._filter_batch_gpu(host_batch, len(chunk))
                coordinates.extend(c for c, k in zip(chunk, keep) if k)
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
    print("==== v14 general GPU compute (mono-core decode) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | cpu-decode {ds.read_time:.3f}s xfer {ds.transfer_time:.3f}s "
          f"gpu-filter {ds.kernel_time:.3f}s")


if __name__ == '__main__':
    run_test()
