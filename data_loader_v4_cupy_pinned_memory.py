"""
data_loader_v4_cupy_pinned_memory.py

v4: PINNED MEMORY OPTIMIZATION
==============================
v2/v3 let CuPy allocate a fresh device array and copy from pageable host memory
on every batch. Two costs hide there: (1) pageable->device copies are slower
than page-locked (pinned) ones because the driver has to stage them, and (2)
re-allocating device memory each batch thrashes the allocator.

v4 fixes both:
  - A reusable **pinned (page-locked) host staging buffer** that patches are
    copied into. Transfers from pinned memory hit higher PCIe bandwidth and can
    be issued asynchronously on a stream.
  - A reusable **pre-allocated device buffer** so no per-batch device malloc.
  - A CuPy ``PinnedMemoryPool`` so the pinned buffer itself is recycled.

The arithmetic is identical to v2; only the data-movement pipeline changes.
"""
import time
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_LUMA = (19595, 38470, 7471)
_LUMA_ROUND = 32768

# Recycle pinned host allocations across batches.
cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)


def gpu_grayscale_uint8(rgb_gpu: cp.ndarray) -> cp.ndarray:
    g = rgb_gpu.astype(cp.uint32)
    gray = (g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2]
            + _LUMA_ROUND) >> 16
    return gray.astype(cp.uint8)


def _alloc_pinned(shape, dtype=np.uint8) -> np.ndarray:
    """Allocate a page-locked host ndarray of the given shape."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(mem, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset using pinned-memory staging for fast transfers."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 32,
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
        self.kernel_time = 0.0

        if self.verbose:
            print(f"[*] Initializing dataset for WSI: {self.wsi_path}")

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
                if self.verbose:
                    print(f"    - WSI dimensions (level 0): {self.wsi_width}x{self.wsi_height}")
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        start_time = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start_time
        if self.verbose:
            print(f"\n[*] Grid creation finished in {self.grid_creation_time:.2f} seconds.")

        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

        if self.verbose:
            print(f"[*] Found {len(self.coordinates)} tissue-containing patches.")

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        potential_coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    potential_coords.append((x, y))
        return potential_coords

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v4 (pinned memory, bs={self.batch_size}): scanning "
                  f"{len(potential_coords)} candidates...")

        ps = self.patch_size
        total_pixels = ps * ps
        # Reusable pinned host buffer + device buffer + dedicated transfer stream.
        pinned_host = _alloc_pinned((self.batch_size, ps, ps, 3), np.uint8)
        device_buf = cp.empty((self.batch_size, ps, ps, 3), dtype=cp.uint8)
        stream = cp.cuda.Stream(non_blocking=True)

        coordinates = []
        with openslide.OpenSlide(self.wsi_path) as slide:
            for start in range(0, len(potential_coords), self.batch_size):
                chunk = potential_coords[start:start + self.batch_size]
                valid_coords = []
                n = 0
                for x, y in chunk:
                    try:
                        patch = slide.read_region((x, y), 0, (ps, ps))
                        # Copy straight into the page-locked staging buffer.
                        pinned_host[n] = np.asarray(patch)[:, :, :3]
                        valid_coords.append((x, y))
                        n += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"    - patch at ({x},{y}): discarded (error: {e})")
                if n == 0:
                    continue

                e_start = cp.cuda.Event()
                e_end = cp.cuda.Event()
                e_start.record()

                with stream:
                    # Async pinned host -> device copy, then compute on the stream.
                    device_buf[:n].set(pinned_host[:n], stream=stream)
                    gray = gpu_grayscale_uint8(device_buf[:n])
                    white_ratio = cp.sum(gray > self.white_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
                    black_ratio = cp.sum(gray < self.black_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
                    keep = (white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio)
                    keep_host = cp.asnumpy(keep)
                stream.synchronize()
                e_end.record()
                e_end.synchronize()
                self.kernel_time += cp.cuda.get_elapsed_time(e_start, e_end) / 1000.0

                for (x, y), k in zip(valid_coords, keep_host):
                    if k:
                        coordinates.append((x, y))

        # Release the big buffers promptly.
        del device_buf
        cp.get_default_memory_pool().free_all_blocks()

        if self.verbose:
            print(f"\n[*] Scanned {len(potential_coords)} patches. Kept {len(coordinates)}.")
        return coordinates

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        with openslide.OpenSlide(self.wsi_path) as slide:
            x, y = self.coordinates[idx]
            patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')
            if self.transform:
                patch_tensor = self.transform(patch)
            else:
                patch_tensor = transforms.ToTensor()(patch)
            return patch_tensor, (x, y)


def run_test(wsi_path: str = "data/S114-82742C-Her2(4B5) 20x.tiff"):
    print("=====================================================")
    print(" v4 CuPy Pinned Memory - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, batch_size=32, verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")


if __name__ == '__main__':
    run_test()
