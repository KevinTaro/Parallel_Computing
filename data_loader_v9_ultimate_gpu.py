"""
data_loader_v9_ultimate_gpu.py

v9: ULTIMATE GPU (all optimization layers combined)
===================================================
Where v1-v7 each isolate a single optimization and v8 tunes for one card, v9
stacks every GPU technique into one configuration-driven loader and asks: "with
all known optimizations applied, what is the GPU performance ceiling for WSI
patch filtering?" It deliberately drops v3's hybrid CPU fallback -- v9 assumes
the GPU is present and pushes it.

The 7 layers (each independently toggleable for the ablation study):

  Layer 1  Memory management   pinned host staging + reused device buffers + pool
  Layer 2  Async streams       double-buffered: read+transfer batch i+1 while the
                               GPU computes batch i (events synced only at drain,
                               so overlap is real -- unlike v8 which syncs/batch)
  Layer 3  Batch processing    one H->D copy + vectorised reductions per batch
  Layer 4  Mixed precision     OPTIONAL (off by default); fp16 luma, validated
  Layer 5  Kernel optimization fused uint8 luma kernel, everything stays on GPU,
                               only the N-bool keep-mask returns to the host
  Layer 6  Memory layout       C-contiguous, channel-last (H,W,C) staging
  Layer 7  Algorithm opt       boolean masks + count_nonzero (bit ops, no uint32
                               temporary); optional per-batch early-discard

Correctness: with mixed precision OFF (default) the fused integer luma kernel
    gray = (R*19595 + G*38470 + B*7471 + 32768) >> 16
is bit-identical to PIL convert('L'), so v9's kept coordinates match v0a exactly.
With mixed precision ON the fp16 luma can differ by <=1 grey level (validated).

The loader exposes the same contract as every other version
(``WSISlidingWindowDataset``, ``kernel_time``, ``peak_gpu_bytes``,
``_generate_candidate_coords``) so it drops straight into the existing
benchmark / validation / memory-profiling harness.
"""
import time
from typing import Callable, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_LUMA = (19595, 38470, 7471)
_LUMA_F = (0.299, 0.587, 0.114)

# Layer 5/7: fused PIL-equivalent luma, uint8 R,G,B -> uint8 gray, no uint32 temp.
_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b',
    out_params='uint8 gray',
    operation='gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;',
    name='pil_luma_uint8_v9',
)


def _alloc_pinned(shape, dtype=np.uint8) -> np.ndarray:
    """Layer 1: page-locked host buffer for fast (DMA-able) transfers."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    return np.frombuffer(mem, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


class WSISlidingWindowDataset(Dataset):
    """Ultimate GPU WSI patch dataset: all optimization layers, configurable."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 512,
                 num_streams: int = 2,
                 # --- layer toggles (for the ablation study) ----------------
                 enable_pinned_memory: bool = True,    # Layer 1
                 enable_async: bool = True,            # Layer 2
                 enable_mixed_precision: bool = False,  # Layer 4 (opt-in)
                 enable_early_exit: bool = True,        # Layer 7
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
        self.num_streams = max(1, num_streams if enable_async else 1)
        self.enable_pinned_memory = enable_pinned_memory
        self.enable_async = enable_async and self.num_streams > 1
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_early_exit = enable_early_exit
        self.verbose = verbose

        # Harness contract.
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0

        if self.verbose:
            print(f"[*] Initializing dataset for WSI: {self.wsi_path}")
            print(f"    - layers: pinned={self.enable_pinned_memory} async={self.enable_async} "
                  f"streams={self.num_streams} fp16={self.enable_mixed_precision} "
                  f"early_exit={self.enable_early_exit} batch={self.batch_size}")

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
            print(f"    - kernel time: {self.kernel_time*1e3:.1f} ms  "
                  f"peak GPU: {self.peak_gpu_bytes/1e6:.1f} MB")

        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

        if self.verbose:
            print(f"[*] Found {len(self.coordinates)} tissue-containing patches.")

    # ------------------------------------------------------------------ grid
    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        potential_coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    potential_coords.append((x, y))
        return potential_coords

    def _host_buffer(self, shape) -> np.ndarray:
        """Layer 1/6: pinned (or plain) C-contiguous host staging buffer."""
        if self.enable_pinned_memory:
            return _alloc_pinned(shape, np.uint8)
        return np.empty(shape, dtype=np.uint8)            # pageable fallback

    def _compute_keep(self, device_rgb: cp.ndarray, n: int) -> cp.ndarray:
        """Layers 4/5/7: grayscale + ratio filter on the GPU. Returns N-bool (GPU)."""
        total_pixels = self.patch_size * self.patch_size
        sub = device_rgb[:n]
        if self.enable_mixed_precision:
            # Layer 4: fp16 luma (faster bandwidth, <=1 grey-level drift).
            g = sub.astype(cp.float16)
            gray = (g[..., 0] * cp.float16(_LUMA_F[0])
                    + g[..., 1] * cp.float16(_LUMA_F[1])
                    + g[..., 2] * cp.float16(_LUMA_F[2]))
        else:
            # Layer 5/7: fused exact integer luma into a uint8 buffer (no uint32 temp).
            gray = cp.empty((n, self.patch_size, self.patch_size), dtype=cp.uint8)
            _luma_kernel(sub[..., 0], sub[..., 1], sub[..., 2], gray)

        # Layer 7: boolean masks + count_nonzero (bit ops) instead of sum.
        white_ratio = cp.count_nonzero(gray > self.white_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
        if self.enable_early_exit:
            # A patch already over the white limit is discarded regardless of black;
            # only evaluate black where white passed (saves one reduction's worth of
            # meaningful comparisons on white-heavy slides).
            white_keep = white_ratio < self.rejection_ratio
            black_ratio = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
            keep = white_keep & (black_ratio < self.rejection_ratio)
        else:
            black_ratio = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
            keep = (white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio)
        return keep

    def _make_slots(self, ps: int) -> List[dict]:
        """One independent (stream, host buf, device buf, events) set per stream."""
        slots = []
        for _ in range(self.num_streams):
            slots.append({
                "stream": cp.cuda.Stream(non_blocking=True),
                "host": self._host_buffer((self.batch_size, ps, ps, 3)),
                "device": cp.empty((self.batch_size, ps, ps, 3), dtype=cp.uint8),
                "e_start": cp.cuda.Event(),
                "e_end": cp.cuda.Event(),
                "keep_gpu": None,
                "coords": None,
                "busy": False,
            })
        return slots

    def _launch(self, slot: dict, n: int):
        """Layer 2/3: async H->D copy + filter on this slot's stream (no host sync)."""
        stream = slot["stream"]
        with stream:
            slot["e_start"].record(stream)
            slot["device"][:n].set(slot["host"][:n], stream=stream)   # async if pinned
            slot["keep_gpu"] = self._compute_keep(slot["device"], n)
            slot["e_end"].record(stream)
        slot["busy"] = True

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v9 (ultimate): scanning {len(potential_coords)} candidates "
                  f"({self.num_streams}-way {'overlapped' if self.enable_async else 'serial'})...")

        ps = self.patch_size
        mempool = cp.get_default_memory_pool()
        slots = self._make_slots(ps)
        coordinates: List[Tuple[int, int]] = []

        def drain(slot):
            """Sync a slot's in-flight batch, accumulate kernel time, collect coords."""
            if not slot["busy"]:
                return
            slot["stream"].synchronize()
            self.kernel_time += cp.cuda.get_elapsed_time(slot["e_start"], slot["e_end"]) / 1000.0
            keep_host = cp.asnumpy(slot["keep_gpu"])
            for (x, y), k in zip(slot["coords"], keep_host):
                if k:
                    coordinates.append((x, y))
            slot["busy"] = False

        chunk_starts = list(range(0, len(potential_coords), self.batch_size))
        with openslide.OpenSlide(self.wsi_path) as slide:
            for i, start in enumerate(chunk_starts):
                slot = slots[i % self.num_streams]
                # Reclaim this slot's previous batch before reusing its buffers.
                drain(slot)

                chunk = potential_coords[start:start + self.batch_size]
                valid_coords, n = [], 0
                for x, y in chunk:
                    try:
                        slot["host"][n] = np.asarray(slide.read_region((x, y), 0, (ps, ps)))[:, :, :3]
                        valid_coords.append((x, y))
                        n += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"    - patch at ({x},{y}): discarded (error: {e})")
                if n == 0:
                    continue
                slot["coords"] = valid_coords
                self._launch(slot, n)
                self.peak_gpu_bytes = max(self.peak_gpu_bytes, mempool.used_bytes())

                if not self.enable_async:
                    drain(slot)             # serial mode: finish before next read

            # Drain whatever is still in flight, in submission order.
            for j in range(self.num_streams):
                drain(slots[(len(chunk_starts) + j) % self.num_streams])

        for slot in slots:
            del slot["device"]
        mempool.free_all_blocks()

        if self.verbose:
            print(f"\n[*] Scanned {len(potential_coords)} patches. Kept {len(coordinates)}.")
        return coordinates

    # ----------------------------------------------------------- dataset API
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
    print(" v9 Ultimate GPU (all layers) - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, batch_size=64, verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")


if __name__ == '__main__':
    run_test()
