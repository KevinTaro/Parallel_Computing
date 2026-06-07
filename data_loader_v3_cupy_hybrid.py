"""
data_loader_v3_cupy_hybrid.py

v3: HYBRID (SMART TRANSFER)
===========================
Neither pure-CPU nor pure-GPU is best at every scale: for a handful of patches
the host<->device transfer and kernel-launch overhead is not worth it, while
for many patches the GPU's throughput dominates. v3 chooses per chunk:

    if len(chunk) >= gpu_threshold:  process the chunk on the GPU (like v2)
    else:                            process it on the CPU (like v0a)

So small jobs and the small trailing chunk run on the CPU, while the bulk runs
on the GPU. This mirrors how a real deployment would avoid paying GPU overhead
for latency-sensitive / small workloads. The counts of CPU- vs GPU-processed
patches are recorded on ``self.cpu_patches`` / ``self.gpu_patches`` for the
benchmark report.

Both paths use the identical integer luma + threshold arithmetic, so the kept
coordinates are independent of which path a patch happened to take.
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
    g = rgb_gpu.astype(cp.uint32)
    gray = (g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2]
            + _LUMA_ROUND) >> 16
    return gray.astype(cp.uint8)


def cpu_grayscale_uint8(rgb_np: np.ndarray) -> np.ndarray:
    """Same integer luma on the CPU, kept identical to the GPU path."""
    g = rgb_np.astype(np.uint32)
    gray = (g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2]
            + _LUMA_ROUND) >> 16
    return gray.astype(np.uint8)


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset that switches between CPU and GPU per chunk."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 32,
                 gpu_threshold: int = 16,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
        self.gpu_threshold = gpu_threshold
        self.verbose = verbose
        self.cpu_patches = 0
        self.gpu_patches = 0

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
            print(f"    - CPU-processed: {self.cpu_patches}, GPU-processed: {self.gpu_patches}")

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

    def _filter_gpu(self, batch_rgb: np.ndarray) -> np.ndarray:
        total_pixels = self.patch_size * self.patch_size
        gpu = cp.asarray(batch_rgb)
        gray = gpu_grayscale_uint8(gpu)
        white_ratio = cp.sum(gray > self.white_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
        black_ratio = cp.sum(gray < self.black_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
        keep = (white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio)
        return cp.asnumpy(keep)

    def _filter_cpu(self, batch_rgb: np.ndarray) -> np.ndarray:
        total_pixels = self.patch_size * self.patch_size
        gray = cpu_grayscale_uint8(batch_rgb)
        white_ratio = np.sum(gray > self.white_pixel_threshold, axis=(1, 2)) / total_pixels
        black_ratio = np.sum(gray < self.black_pixel_threshold, axis=(1, 2)) / total_pixels
        return (white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio)

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v3 (hybrid, gpu_threshold={self.gpu_threshold}): scanning "
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

                # The smart-transfer decision.
                if len(patches) >= self.gpu_threshold:
                    keep_mask = self._filter_gpu(batch_rgb)
                    self.gpu_patches += len(patches)
                else:
                    keep_mask = self._filter_cpu(batch_rgb)
                    self.cpu_patches += len(patches)

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
    print(" v3 CuPy Hybrid - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, batch_size=32, gpu_threshold=16,
                                      verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")


if __name__ == '__main__':
    run_test()
