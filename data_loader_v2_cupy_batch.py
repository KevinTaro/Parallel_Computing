"""
data_loader_v2_cupy_batch.py

v2: CuPy BATCH PROCESSING
=========================
v1 paid a transfer + several kernel launches *per patch*. v2 amortises that by
reading ``batch_size`` patches on the CPU, stacking them into a single
(N, H, W, 4) array, doing **one** host->device transfer, and computing the
grayscale + white/black ratios for the whole batch with vectorised reductions
over the spatial axes. One transfer and a handful of kernels now cover N
patches instead of one.

This is where the GPU typically starts to pay off: larger batches push the
arithmetic-to-overhead ratio in the GPU's favour. The batch size is the key
tunable and is deliberately modest by default because the target GPU only has
3 GB of memory (a batch of N 1024x1024 patches needs ~N*12 MB just for the
uint32 grayscale intermediate).
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


def gpu_grayscale_uint8(rgb_gpu: cp.ndarray) -> cp.ndarray:
    """Exact PIL-equivalent grayscale on the GPU. ``rgb_gpu`` is (..., 3) uint8."""
    g = rgb_gpu.astype(cp.uint32)
    gray = (g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2]
            + _LUMA_ROUND) >> 16
    return gray.astype(cp.uint8)


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset filtered in GPU batches."""

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

    def _filter_batch(self, batch_rgb: np.ndarray) -> np.ndarray:
        """Filter a stacked (N, H, W, 3) uint8 batch. Returns a bool keep-mask."""
        total_pixels = self.patch_size * self.patch_size
        gpu = cp.asarray(batch_rgb)                       # one host -> device copy
        gray = gpu_grayscale_uint8(gpu)                   # (N, H, W) uint8

        white_counts = cp.sum(gray > self.white_pixel_threshold, axis=(1, 2))
        black_counts = cp.sum(gray < self.black_pixel_threshold, axis=(1, 2))
        white_ratio = white_counts.astype(cp.float64) / total_pixels
        black_ratio = black_counts.astype(cp.float64) / total_pixels

        keep = (white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio)
        return cp.asnumpy(keep)                           # N booleans back to host

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v2 (CuPy batch, bs={self.batch_size}): scanning "
                  f"{len(potential_coords)} candidates...")

        coordinates = []
        with openslide.OpenSlide(self.wsi_path) as slide:
            for start in range(0, len(potential_coords), self.batch_size):
                chunk = potential_coords[start:start + self.batch_size]
                patches, valid_coords = [], []
                for x, y in chunk:
                    try:
                        patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
                        patches.append(np.asarray(patch)[:, :, :3])
                        valid_coords.append((x, y))
                    except Exception as e:
                        if self.verbose:
                            print(f"    - patch at ({x},{y}): discarded (error: {e})")
                if not patches:
                    continue
                batch_rgb = np.stack(patches, axis=0)
                keep_mask = self._filter_batch(batch_rgb)
                for (x, y), keep in zip(valid_coords, keep_mask):
                    if keep:
                        coordinates.append((x, y))

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
    print(" v2 CuPy Batch - Test Run")
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
