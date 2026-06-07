"""
data_loader_v0a_mono_baseline.py

v0a: MONO-CORE BASELINE (Single-Threaded CPU)
=============================================
Source: data_loader_mono.py

Pure sequential processing on a single CPU core. This is the *functional*
reference for the whole research framework: every other version (v0b, v1-v7)
is measured as a speedup relative to this one.

Characteristics:
  - Single OpenSlide handle, sequential read + filter loop
  - Proper resource management (context manager) and per-patch error handling
  - NumPy on CPU only, no parallelism whatsoever
  - Reliable but slow -> the "1.0x" reference

The only behavioural change versus the original data_loader_mono.py is a
``verbose`` flag (default ``False``). The originals printed one line per patch,
which dominates wall-clock time and makes benchmark numbers meaningless. The
filtering arithmetic is byte-for-byte identical, so the kept coordinates match.
"""
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WSISlidingWindowDataset(Dataset):
    """Single-threaded, sequential WSI patch dataset (mono-core baseline)."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.verbose = verbose

        if self.verbose:
            print(f"[*] Initializing dataset for WSI: {self.wsi_path}")

        # Read WSI metadata without loading the full image
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

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        """Build the list of in-bounds (x, y) candidate coordinates."""
        potential_coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    potential_coords.append((x, y))
        return potential_coords

    def _create_grid(self) -> List[Tuple[int, int]]:
        """Sequentially filter every candidate patch on a single CPU core."""
        potential_coords = self._generate_candidate_coords()

        if self.verbose:
            print(f"[*] 開始以單核心順序掃描 {len(potential_coords)} 個候選區塊...")

        coordinates = []
        with openslide.OpenSlide(self.wsi_path) as slide:
            for x, y in potential_coords:
                try:
                    patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
                    patch_gray_np = np.array(patch.convert('L'))
                    total_pixels = self.patch_size * self.patch_size

                    white_ratio = np.sum(patch_gray_np > self.white_pixel_threshold) / total_pixels
                    if white_ratio >= self.rejection_ratio:
                        if self.verbose:
                            print(f"    - patch at ({x},{y}): discarded (white_ratio {white_ratio:.2f} >= {self.rejection_ratio})")
                        continue

                    black_ratio = np.sum(patch_gray_np < self.black_pixel_threshold) / total_pixels
                    if black_ratio >= self.rejection_ratio:
                        if self.verbose:
                            print(f"    - patch at ({x},{y}): discarded (black_ratio {black_ratio:.2f} >= {self.rejection_ratio})")
                        continue

                    if self.verbose:
                        print(f"    - patch at ({x},{y}): selected (white: {white_ratio:.2f}, black: {black_ratio:.2f})")
                    coordinates.append((x, y))

                except Exception as e:
                    if self.verbose:
                        print(f"    - patch at ({x},{y}): discarded (error: {e})")

        if self.verbose:
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
    print(" v0a Mono-Core Baseline - Test Run")
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
