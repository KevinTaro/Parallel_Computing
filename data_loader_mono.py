"""
core/data_loader.py

Mono core data loader for Whole Slide Images (WSIs).

This module implements a PyTorch Dataset for efficiently loading patches
from WSIs without pre-splitting them into individual files on disk.
It dynamically reads regions from the WSI, enabling a "zero-I/O overhead"
approach, which is crucial for handling large gigapixel images.
"""
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WSISlidingWindowDataset(Dataset):
    """
    A PyTorch Dataset for loading patches from a WSI using a sliding window.

    This dataset implements a high-precision background filtering method by
    analyzing each potential patch at level 0 for its pixel content.

    Args:
        wsi_path (str): Path to the Whole Slide Image file.
        patch_size (int): The height and width of the patches to extract (default: 1024).
        stride (int): The step size between patches (default: 1024).
        transform (Callable, optional): A function/transform to apply to the patches.
        white_pixel_threshold (int): Pixel intensity value to be considered 'white' (0-255).
        black_pixel_threshold (int): Pixel intensity value to be considered 'black' (0-255).
        rejection_ratio (float): Ratio of white or black pixels to discard a patch (0.0-1.0).
    """

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio

        print(f"[*] Initializing dataset for WSI: {self.wsi_path}")

        # Read WSI metadata without loading the full image
        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
                print(f"    - WSI dimensions (level 0): {self.wsi_width}x{self.wsi_height}")
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        # Generate a virtual grid of coordinates
        print("[*] Generating virtual grid with high-precision filtering (Level 0)...")
        start_time = time.time()
        self.coordinates = self._create_grid()
        end_time = time.time()
        print(f"\n[*] Grid creation finished in {end_time - start_time:.2f} seconds.")

        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI. "
                             "Consider adjusting the filtering thresholds.")

        print(f"[*] Found {len(self.coordinates)} tissue-containing patches.")

    def _create_grid(self) -> List[Tuple[int, int]]:
        """
        Generates a grid of (x, y) coordinates by performing high-precision
        filtering on every potential patch at level 0 using sequential processing.
        """
        potential_coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    potential_coords.append((x, y))

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
                        print(f"    - patch at ({x},{y}): discarded (reason: white_ratio {white_ratio:.2f} >= {self.rejection_ratio})")
                        continue

                    black_ratio = np.sum(patch_gray_np < self.black_pixel_threshold) / total_pixels
                    if black_ratio >= self.rejection_ratio:
                        print(f"    - patch at ({x},{y}): discarded (reason: black_ratio {black_ratio:.2f} >= {self.rejection_ratio})")
                        continue

                    print(f"    - patch at ({x},{y}): selected (white: {white_ratio:.2f}, black: {black_ratio:.2f})")
                    coordinates.append((x, y))

                except Exception as e:
                    print(f"    - patch at ({x},{y}): discarded (reason: error processing patch: {e})")

        print(f"\n[*] Scanned {len(potential_coords)} potential patches. Kept {len(coordinates)} tissue patches.")
        return coordinates

    def __len__(self) -> int:
        """Returns the total number of patches in the dataset."""
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Retrieves a patch and its coordinates at the given index.
        """
        # Lazy initialization of the slide object for multiprocessing compatibility
        with openslide.OpenSlide(self.wsi_path) as slide:
            x, y = self.coordinates[idx]
            patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')

            if self.transform:
                patch_tensor = self.transform(patch)
            else:
                patch_tensor = transforms.ToTensor()(patch)

            return patch_tensor, (x, y)

def run_test():
    """
    Example of how to use the dataset, demonstrating the new high-precision filtering.
    """
    WSI_PATH = "data/S114-82742C-Her2(4B5) 20x.tiff"

    try:
        print("=====================================================")
        print(" WSI Sliding Window Dataset - High-Precision Test Run")
        print("=====================================================")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = WSISlidingWindowDataset(
            wsi_path=WSI_PATH,
            patch_size=1024,
            stride=1024,
            transform=transform,
            white_pixel_threshold=230,
            black_pixel_threshold=25,
            rejection_ratio=0.9
        )

        print(f"\n[*] Dataset created successfully for WSI: {WSI_PATH}")
        print(f"[*] Total patches with tissue: {len(dataset)}")

        if len(dataset) > 0:
            print("\n[*] Loading a single patch (index 0)...")
            start_time = time.time()
            patch, coords = dataset[0]
            end_time = time.time()
            print(f"    - Patch loaded in {end_time - start_time:.4f} seconds.")
            print(f"    - Shape of the first patch: {patch.shape}")
            print(f"    - Coordinates of the first patch: {coords}")

            print("\n[*] Testing with PyTorch DataLoader...")
            data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            print("    - Fetching a batch of 4 patches...")
            start_time = time.time()
            batch_patches, batch_coords = next(iter(data_loader))
            end_time = time.time()
            print(f"    - Batch loaded in {end_time - start_time:.4f} seconds.")
            print(f"    - Batch of patches shape: {batch_patches.shape}")
            print("    - Batch of coordinates (x, y):")
            for x_coord, y_coord in zip(*batch_coords):
                print(f"      - ({x_coord.item()}, {y_coord.item()})")

        print("\n=====================================================")
        print("          Test Run Completed Successfully")
        print("=====================================================")

    except Exception as e:
        print(f"\n[!!!] An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_test()
