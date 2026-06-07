"""
data_loader_v5_cupy_async.py

v5: ASYNC GPU PROCESSING (double-buffered streams)
==================================================
v4 still ran strictly serially: read a batch, transfer it, compute it,
synchronize, repeat. The GPU sat idle while the CPU read the next batch off
disk, and the CPU sat idle while the GPU computed.

v5 overlaps the two with a classic **double buffer** across two CUDA streams:

    slot 0: [transfer+compute batch i  ] on stream 0
    slot 1:                 [read batch i+1 from disk] (CPU, concurrent)
             [transfer+compute batch i+1] on stream 1
    ...

We launch the async transfer + compute for a batch and immediately go read the
next batch from disk into the *other* pinned buffer instead of blocking. The
disk read + host->device copy of batch i+1 thus overlaps the GPU compute of
batch i. Each slot owns its own pinned host buffer, device buffer, and stream,
so the two never alias.

Arithmetic is identical to v2/v4; only the scheduling changes. On a slow disk
the win comes mostly from hiding I/O behind compute.
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

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)


def gpu_grayscale_uint8(rgb_gpu: cp.ndarray) -> cp.ndarray:
    g = rgb_gpu.astype(cp.uint32)
    gray = (g[..., 0] * _LUMA[0] + g[..., 1] * _LUMA[1] + g[..., 2] * _LUMA[2]
            + _LUMA_ROUND) >> 16
    return gray.astype(cp.uint8)


def _alloc_pinned(shape, dtype=np.uint8) -> np.ndarray:
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(mem, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset overlapping I/O + transfer + compute via 2 streams."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 32,
                 num_streams: int = 2,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
        self.num_streams = max(2, num_streams)
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

    def _launch_batch(self, slot: dict, n: int):
        """Issue the async transfer + filter for the n patches staged in a slot."""
        total_pixels = self.patch_size * self.patch_size
        stream = slot['stream']
        slot['e_start'] = cp.cuda.Event()
        slot['e_end'] = cp.cuda.Event()
        slot['e_start'].record()
        with stream:
            slot['device'][:n].set(slot['pinned'][:n], stream=stream)
            gray = gpu_grayscale_uint8(slot['device'][:n])
            white_ratio = cp.sum(gray > self.white_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
            black_ratio = cp.sum(gray < self.black_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
            slot['keep_gpu'] = (white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio)
        slot['e_end'].record()

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v5 (async, {self.num_streams} streams, bs={self.batch_size}): scanning "
                  f"{len(potential_coords)} candidates...")

        ps = self.patch_size
        # One independent buffer set per stream.
        slots = []
        for _ in range(self.num_streams):
            slots.append({
                'stream': cp.cuda.Stream(non_blocking=True),
                'pinned': _alloc_pinned((self.batch_size, ps, ps, 3), np.uint8),
                'device': cp.empty((self.batch_size, ps, ps, 3), dtype=cp.uint8),
                'keep_gpu': None,
                'coords': None,
                'busy': False,
                'e_end': None,
            })

        coordinates = []

        def drain(slot):
            """Block until a slot's work is done and collect its kept coords."""
            if not slot['busy']:
                return
            slot['stream'].synchronize()
            if slot['e_end']:
                slot['e_end'].synchronize()
                self.kernel_time += cp.cuda.get_elapsed_time(slot['e_start'], slot['e_end']) / 1000.0
            keep_host = cp.asnumpy(slot['keep_gpu'])
            for (x, y), k in zip(slot['coords'], keep_host):
                if k:
                    coordinates.append((x, y))
            slot['busy'] = False

        chunk_starts = list(range(0, len(potential_coords), self.batch_size))
        with openslide.OpenSlide(self.wsi_path) as slide:
            for i, start in enumerate(chunk_starts):
                slot = slots[i % self.num_streams]
                # Reclaim this slot's previous in-flight batch before overwriting it.
                drain(slot)

                chunk = potential_coords[start:start + self.batch_size]
                valid_coords, n = [], 0
                for x, y in chunk:
                    try:
                        slot['pinned'][n] = np.asarray(slide.read_region((x, y), 0, (ps, ps)))[:, :, :3]
                        valid_coords.append((x, y))
                        n += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"    - patch at ({x},{y}): discarded (error: {e})")
                if n == 0:
                    continue
                slot['coords'] = valid_coords
                slot['busy'] = True
                self._launch_batch(slot, n)
                # NOTE: we do NOT synchronize here -> the next loop iteration
                # reads the next batch off disk while this one runs on the GPU.

            # Drain whatever is still in flight, in submission order.
            for j in range(self.num_streams):
                drain(slots[(len(chunk_starts) + j) % self.num_streams])

        for slot in slots:
            del slot['device']
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
    print(" v5 CuPy Async (double-buffered streams) - Test Run")
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
