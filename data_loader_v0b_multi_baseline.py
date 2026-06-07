"""
data_loader_v0b_multi_baseline.py

v0b: MULTI-CORE CPU BASELINE (Multiprocessing)
==============================================
Source: data_loader.py

Original CPU implementation parallelised across all cores with
``multiprocessing.Pool``. Each worker opens its own OpenSlide handle (handles
are not fork-safe to share), reads its patch, and runs the identical NumPy
filtering arithmetic used by v0a. This is the reference for "how far pure CPU
parallelism gets us" before reaching for the GPU.

Only behavioural change versus the original data_loader.py is a ``verbose``
flag (default ``False``) so the per-patch log lines don't pollute benchmarks.
"""
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Optional, Tuple

import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WSISlidingWindowDataset(Dataset):
    """Multiprocessing WSI patch dataset (multi-core CPU baseline)."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 num_workers: Optional[int] = None,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.num_workers = num_workers or cpu_count()
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

        if self.verbose:
            print("[*] Generating virtual grid with high-precision filtering (Level 0)...")
        start_time = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start_time
        if self.verbose:
            print(f"\n[*] Grid creation finished in {self.grid_creation_time:.2f} seconds.")

        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI. "
                             "Consider adjusting the filtering thresholds.")

        if self.verbose:
            print(f"[*] Found {len(self.coordinates)} tissue-containing patches.")

    @staticmethod
    def _process_patch(coords: Tuple[int, int], wsi_path: str, patch_size: int,
                       white_pixel_threshold: int, black_pixel_threshold: int,
                       rejection_ratio: float) -> Tuple[int, int, bool]:
        """Worker: decide whether a single patch is kept. Runs in a child process."""
        x, y = coords
        try:
            with openslide.OpenSlide(wsi_path) as slide:
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                patch_gray_np = np.array(patch.convert('L'))
                total_pixels = patch_size * patch_size

                white_ratio = np.sum(patch_gray_np > white_pixel_threshold) / total_pixels
                if white_ratio >= rejection_ratio:
                    return (x, y, False)

                black_ratio = np.sum(patch_gray_np < black_pixel_threshold) / total_pixels
                if black_ratio >= rejection_ratio:
                    return (x, y, False)

                return (x, y, True)
        except Exception:
            return (x, y, False)

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        potential_coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    potential_coords.append((x, y))
        return potential_coords

    def _create_grid(self) -> List[Tuple[int, int]]:
        """Filter all candidate patches in parallel across CPU cores."""
        potential_coords = self._generate_candidate_coords()

        if self.verbose:
            print(f"[*] 偵測到 {self.num_workers} 個 CPU 核心。掃描 {len(potential_coords)} 個候選區塊...")

        process_func = partial(
            WSISlidingWindowDataset._process_patch,
            wsi_path=self.wsi_path,
            patch_size=self.patch_size,
            white_pixel_threshold=self.white_pixel_threshold,
            black_pixel_threshold=self.black_pixel_threshold,
            rejection_ratio=self.rejection_ratio,
        )

        with Pool(processes=self.num_workers) as pool:
            results = pool.map(process_func, potential_coords)

        # Preserve row-major (y, then x) ordering identical to v0a.
        coordinates = [(x, y) for (x, y, keep) in results if keep]

        if self.verbose:
            for x, y, keep in results:
                status = "selected" if keep else "discarded"
                print(f"    - patch at ({x},{y}): {status}")
            print(f"\n[*] Scanned {len(potential_coords)} potential patches. Kept {len(coordinates)} tissue patches.")
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
    print(" v0b Multi-Core CPU Baseline - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")
    if len(dataset) > 0:
        patch, coords = dataset[0]
        print(f"[*] First patch shape {tuple(patch.shape)} at {coords}")


if __name__ == '__main__':
    run_test()
