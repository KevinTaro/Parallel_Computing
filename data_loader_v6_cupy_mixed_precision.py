"""
data_loader_v6_cupy_mixed_precision.py

v6: MIXED PRECISION (float16 grayscale)
=======================================
Idea: do the grayscale weighting in float16 instead of integer/float32 to halve
memory bandwidth and (on GPUs with fast FP16) double arithmetic throughput. The
inputs stay uint8 and the final counts are integers; only the intermediate
luma computation is float16.

Two things this version is designed to expose:

1. **Accuracy trade-off.** float16 has ~11 bits of mantissa, so the weighted sum
   R*0.299 + G*0.587 + B*0.114 is no longer exact. Patches whose white/black
   ratio sits right on the rejection threshold can flip decision versus the
   exact integer baseline. The validation suite quantifies how many (usually a
   tiny boundary set, if any).

2. **Hardware dependence.** On Pascal-class cards (e.g. GTX 1060) native FP16
   throughput is ~1/64 of FP32, so this can be *slower* than v2 despite moving
   less data -- a concrete reminder that "mixed precision" is only a win on
   hardware with real FP16/tensor-core support (Volta+).

The decision rule and thresholds are otherwise identical to the other versions.
"""
import time
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Standard ITU-R 601-2 luma weights (float form of the integer baseline).
_LUMA_F = (0.299, 0.587, 0.114)


def gpu_grayscale_fp16(rgb_gpu: cp.ndarray) -> cp.ndarray:
    """Approximate grayscale computed in float16. ``rgb_gpu`` is (..., 3) uint8."""
    g = rgb_gpu.astype(cp.float16)
    gray = (g[..., 0] * cp.float16(_LUMA_F[0])
            + g[..., 1] * cp.float16(_LUMA_F[1])
            + g[..., 2] * cp.float16(_LUMA_F[2]))
    return gray  # float16, ~[0, 255]


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset filtered with float16 grayscale (mixed precision)."""

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
        total_pixels = self.patch_size * self.patch_size
        gpu = cp.asarray(batch_rgb)
        gray = gpu_grayscale_fp16(gpu)
        # Threshold comparisons promote to a common type; counts stay exact ints.
        white_ratio = cp.sum(gray > self.white_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
        black_ratio = cp.sum(gray < self.black_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
        keep = (white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio)
        return cp.asnumpy(keep)

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v6 (mixed precision fp16, bs={self.batch_size}): scanning "
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
                keep_mask = self._filter_batch(np.stack(patches, axis=0))
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
    print(" v6 CuPy Mixed Precision (fp16) - Test Run")
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
