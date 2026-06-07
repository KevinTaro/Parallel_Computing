"""
data_loader_v1_cupy_full.py

v1: CuPy FULL FILTERING (per-patch GPU)
=======================================
Replace every NumPy operation in the per-patch filter with CuPy. Each patch is
read on the CPU (OpenSlide I/O is unavoidable), transferred to the GPU, and the
grayscale conversion + white/black ratio reductions all happen on the device.
Only two scalars come back per patch.

This is the most naive GPU strategy: one host->device transfer and a handful of
tiny kernel launches *per patch*. It exists to demonstrate that for small,
independent units of work the transfer + launch overhead can swamp the GPU's
compute advantage. v2 (batching) is the direct response to what this version
reveals.

Correctness note
----------------
PIL's ``Image.convert('L')`` uses the integer ITU-R 601-2 luma transform
    L = (R*19595 + G*38470 + B*7471 + 32768) >> 16
We replicate it *exactly* on the GPU so the uint8 grayscale - and therefore the
threshold pixel counts and the kept coordinates - match the CPU baselines
bit-for-bit. (A plain channel mean would drift by up to ~12 grey levels.)
"""
import time
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Integer luma weights matching PIL's L24 macro (R, G, B).
_LUMA = (19595, 38470, 7471)
_LUMA_ROUND = 32768  # 0x8000, for round-to-nearest before the >>16 shift


def gpu_grayscale_uint8(rgb_gpu: cp.ndarray) -> cp.ndarray:
    """Exact PIL-equivalent grayscale on the GPU. ``rgb_gpu`` is (..., 3) uint8."""
    g = rgb_gpu.astype(cp.uint32)
    gray = (g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2]
            + _LUMA_ROUND) >> 16
    return gray.astype(cp.uint8)


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset filtered fully on the GPU, one patch at a time."""

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

    def _keep_patch(self, rgba_np: np.ndarray) -> bool:
        """Run the full grayscale + ratio decision on the GPU for one patch."""
        total_pixels = self.patch_size * self.patch_size
        e_start = cp.cuda.Event()
        e_end = cp.cuda.Event()
        e_start.record()

        rgb_gpu = cp.asarray(rgba_np[:, :, :3])          # host -> device
        gray = gpu_grayscale_uint8(rgb_gpu)              # GPU grayscale
        white_ratio = float(cp.sum(gray > self.white_pixel_threshold)) / total_pixels
        if white_ratio >= self.rejection_ratio:
            e_end.record()
            e_end.synchronize()
            self.kernel_time += cp.cuda.get_elapsed_time(e_start, e_end) / 1000.0
            return False
        black_ratio = float(cp.sum(gray < self.black_pixel_threshold)) / total_pixels

        e_end.record()
        e_end.synchronize()
        self.kernel_time += cp.cuda.get_elapsed_time(e_start, e_end) / 1000.0

        if black_ratio >= self.rejection_ratio:
            return False
        return True

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v1 (CuPy full): scanning {len(potential_coords)} candidate patches one-by-one on GPU...")

        coordinates = []
        with openslide.OpenSlide(self.wsi_path) as slide:
            for x, y in potential_coords:
                try:
                    patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
                    rgba_np = np.asarray(patch)          # (H, W, 4) uint8
                    if self._keep_patch(rgba_np):
                        coordinates.append((x, y))
                except Exception as e:
                    if self.verbose:
                        print(f"    - patch at ({x},{y}): discarded (error: {e})")

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
    print(" v1 CuPy Full (per-patch GPU) - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")


if __name__ == '__main__':
    run_test()
