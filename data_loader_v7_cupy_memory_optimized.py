"""
data_loader_v7_cupy_memory_optimized.py

v7: MEMORY-OPTIMIZED (for constrained GPUs)
===========================================
Target: a small GPU (the test card has only 3 GB). v2's straightforward batch
path quietly allocates a uint32 array four times the size of the uint8 input
just to hold the grayscale intermediate -- for a batch of 64 1024x1024 patches
that is ~800 MB of scratch. v7 attacks peak memory directly:

  1. **Fused luma kernel.** A single ``ElementwiseKernel`` reads uint8 R/G/B and
     writes uint8 grayscale in one pass, so the 4x uint32 intermediate never
     exists. Peak scratch drops from ~5x to ~(1 + 0.33)x the input batch.
  2. **Small chunks.** A modest ``chunk_size`` bounds how much is resident at
     once, trading a little throughput for a low memory ceiling.
  3. **Buffer reuse + explicit cleanup.** A single device input buffer is
     reused across chunks and the memory pool is trimmed between chunks so
     fragmentation cannot creep up over a long scan.

Peak GPU memory actually observed is recorded on ``self.peak_gpu_bytes``.
Arithmetic matches the integer baseline exactly (the fused kernel uses the same
PIL L24 formula), so kept coordinates are identical to v0a/v1/v2.
"""
import time
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Fused PIL-equivalent luma: reads uint8 R,G,B -> writes uint8 gray, no uint32
# batch temporary. (r*19595 + g*38470 + b*7471 + 32768) >> 16.
_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b',
    out_params='uint8 gray',
    operation='gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;',
    name='pil_luma_uint8',
)


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset tuned for low peak GPU memory."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 chunk_size: int = 8,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.peak_gpu_bytes = 0

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
            print(f"    - Peak GPU memory: {self.peak_gpu_bytes / 1e6:.1f} MB")

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
            print(f"[*] v7 (memory-optimized, chunk={self.chunk_size}): scanning "
                  f"{len(potential_coords)} candidates...")

        ps = self.patch_size
        total_pixels = ps * ps
        mempool = cp.get_default_memory_pool()
        # Reusable device input buffer (uint8) + grayscale output buffer.
        device_in = cp.empty((self.chunk_size, ps, ps, 3), dtype=cp.uint8)
        gray_buf = cp.empty((self.chunk_size, ps, ps), dtype=cp.uint8)

        coordinates = []
        with openslide.OpenSlide(self.wsi_path) as slide:
            for start in range(0, len(potential_coords), self.chunk_size):
                chunk = potential_coords[start:start + self.chunk_size]
                valid_coords, n = [], 0
                for x, y in chunk:
                    try:
                        device_in[n].set(np.ascontiguousarray(
                            np.asarray(slide.read_region((x, y), 0, (ps, ps)))[:, :, :3]))
                        valid_coords.append((x, y))
                        n += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"    - patch at ({x},{y}): discarded (error: {e})")
                if n == 0:
                    continue

                sub = device_in[:n]
                # Fused grayscale into the preallocated buffer (no uint32 temp).
                _luma_kernel(sub[..., 0], sub[..., 1], sub[..., 2], gray_buf[:n])
                gray = gray_buf[:n]
                white_ratio = cp.count_nonzero(gray > self.white_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
                black_ratio = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
                keep = cp.asnumpy((white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio))

                for (x, y), k in zip(valid_coords, keep):
                    if k:
                        coordinates.append((x, y))

                self.peak_gpu_bytes = max(self.peak_gpu_bytes, mempool.used_bytes())
                # Trim transient allocations (ratio temporaries) every chunk.
                mempool.free_all_blocks()

        del device_in, gray_buf
        mempool.free_all_blocks()

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
    print(" v7 CuPy Memory-Optimized - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, chunk_size=8, verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")


if __name__ == '__main__':
    run_test()
