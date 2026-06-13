"""
data_loader_v29_dec_v7_membudget.py

v29: GPU-decode translation of the v7 "memory-optimized for constrained VRAM" concept
======================================================================================
v7 (data_loader_v7_cupy_memory_optimized.py) targeted machines where VRAM is
scarce: it computed an explicit memory budget, sized the batch accordingly, and
aggressively freed the CuPy memory pool between chunks to prevent OOM.

Translation to the GPU-decode world:
    The same constraint applies to the GPU JPEG decoder: each tile occupies
    roughly ``512*512*3 + 96*1024`` bytes of VRAM during decode (RGB output buffer
    + compressed scan staging). On a 4 GB GPU running other workloads, the default
    batch_size of 2048 could OOM. This version:

    1. Accepts an explicit ``vram_budget_gb`` parameter (default 1.0 GB).
    2. Auto-sizes batch from the budget: ``batch_size = max(32, budget_bytes // per_tile)``.
       (If ``batch_size`` is specified explicitly (>0), that overrides the auto-size.)
    3. After EACH chunk, calls ``cp.get_default_memory_pool().free_all_blocks()``
       to return fragmented device memory to the OS, preventing pool growth across
       chunks.
    4. Tracks ``self.peak_gpu_bytes`` as the high-water mark of ``used_bytes()``
       across all chunks.

    Memory budget formula:
        budget_bytes = vram_budget_gb * 1e9
        per_tile     = 512*512*3     (uint8 RGB decode output)
                     + 96*1024       (compressed JPEG scan, 96 KB worst case)
        batch_size   = max(32, budget_bytes // per_tile)

    Example: 1.0 GB budget -> ~1e9 / (786432 + 98304) = ~1130 tiles per batch.

    Comparison:
        v24 (no budget, default 2048)  -- may OOM on constrained GPUs
        v29 (budget=1.0 GB)            -- safe, slower on large GPUs (smaller batches),
                                          but prevents OOM and keeps peak VRAM bounded

Parameters:
    vram_budget_gb  float = 1.0    maximum VRAM to use for decode buffers (GB)
    batch_size      int   = 0      0 = auto-compute from vram_budget_gb
"""
import time
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader_v11_gpu_decode_5090 import _read_tiff_tiles
from gpu_jpeg_decoder_optimized import GpuJpegDecoderOptimized


class WSISlidingWindowDataset(Dataset):
    """VRAM-budget-constrained GPU tile decode -- translates v7 memory budget concept."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 vram_budget_gb: float = 1.0,
                 batch_size: int = 0,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.vram_budget_gb = vram_budget_gb
        self.verbose = verbose
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0

        # Compute batch_size from explicit VRAM budget
        budget_bytes = int(vram_budget_gb * 1e9)
        per_tile = 512 * 512 * 3 + 96 * 1024
        auto_bs = max(32, budget_bytes // per_tile)
        self.batch_size = batch_size if batch_size > 0 else auto_bs

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v29 requires JPEG tiles (compression == 7).")

        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start

        if self.verbose:
            d = self._decoder.device
            print(f"[v29] {d['name']} (cc{d['cc']}) budget={vram_budget_gb:.1f}GB "
                  f"batch={self.batch_size} (auto={auto_bs})")
            print(f"[v29] grid done in {self.grid_creation_time:.2f}s "
                  f"kernel_time={self.kernel_time:.3f}s "
                  f"peak_gpu={self.peak_gpu_bytes/1e6:.1f}MB "
                  f"kept={len(self.coordinates)}")

        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

    def _build_decoder(self) -> GpuJpegDecoderOptimized:
        offs, bcs = self._tiff["offsets"], self._tiff["bytecounts"]
        with open(self.wsi_path, 'rb') as fh:
            tid = next(i for i, b in enumerate(bcs) if b > 0)
            fh.seek(offs[tid]); sample = fh.read(bcs[tid])
        return GpuJpegDecoderOptimized(self._tiff["jpegtables"], sample)

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    coords.append((x, y))
        return coords

    def _create_grid(self) -> List[Tuple[int, int]]:
        ps = self.patch_size
        tw, th = self._tiff["tile_width"], self._tiff["tile_height"]
        iw, ih = self._tiff["image_width"], self._tiff["image_height"]

        if not (self.stride == ps and ps % tw == 0 and ps % th == 0
                and iw % tw == 0 and ih % th == 0):
            raise ValueError("requires tile-aligned geometry.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()

        if self.verbose:
            print(f"[v29] {len(candidates)} candidates, "
                  f"{len(self._tiff['offsets'])} tiles, batch={self.batch_size} "
                  f"(budget={self.vram_budget_gb:.1f}GB)")

        patch_tiles, needed = [], {}
        for (x, y) in candidates:
            col0, row0 = x // tw, y // th
            ids = []
            for dr in range(typ):
                base = (row0 + dr) * tiles_per_row + col0
                for dc in range(txp):
                    tid = base + dc; ids.append(tid)
                    needed.setdefault(tid, len(needed))
            patch_tiles.append((x, y, ids))

        tile_ids = list(needed.keys())
        offsets, bytecounts = self._tiff["offsets"], self._tiff["bytecounts"]
        white_per_tile = np.zeros(len(tile_ids), dtype=np.int64)
        black_per_tile = np.zeros(len(tile_ids), dtype=np.int64)
        TILE_PX = tw * th
        decode_idx = [ci for ci, tid in enumerate(tile_ids) if bytecounts[tid] > 0]
        for ci, tid in enumerate(tile_ids):
            if bytecounts[tid] == 0: black_per_tile[ci] = TILE_PX

        wt, bt = self.white_pixel_threshold, self.black_pixel_threshold
        bs = self.batch_size

        with open(self.wsi_path, 'rb') as fh:
            for chunk_start in range(0, len(decode_idx), bs):
                chunk = decode_idx[chunk_start: chunk_start + bs]
                tiles = []
                for ci in chunk:
                    tid = tile_ids[ci]
                    fh.seek(offsets[tid])
                    tiles.append(fh.read(bytecounts[tid]))

                ev_start = cp.cuda.Event()
                ev_end = cp.cuda.Event()
                ev_start.record()
                w, b = self._decoder.count_batch(tiles, wt, bt)
                ev_end.record()
                ev_end.synchronize()
                self.kernel_time += cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e-3

                for local_i, ci in enumerate(chunk):
                    white_per_tile[ci] = int(w[local_i])
                    black_per_tile[ci] = int(b[local_i])

                # v7 concept: free pool blocks after each chunk to cap peak VRAM
                self.peak_gpu_bytes = max(self.peak_gpu_bytes,
                                          cp.get_default_memory_pool().used_bytes())
                cp.get_default_memory_pool().free_all_blocks()

        coordinates, rr = [], self.rejection_ratio
        for (x, y, ids) in patch_tiles:
            wc = sum(int(white_per_tile[needed[t]]) for t in ids)
            bc = sum(int(black_per_tile[needed[t]]) for t in ids)
            if (wc / total_pixels) < rr and (bc / total_pixels) < rr:
                coordinates.append((x, y))
        return coordinates

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        with openslide.OpenSlide(self.wsi_path) as slide:
            x, y = self.coordinates[idx]
            patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert('RGB')
            t = self.transform(patch) if self.transform else transforms.ToTensor()(patch)
            return t, (x, y)


def run_test(wsi_path: str = "data/S114-80954A-Her2(3+).tiff"):
    print("==== v29 GPU-decode v7-membudget - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, vram_budget_gb=1.0, verbose=True)
    print(f"[v29] kept={len(ds)} batch={ds.batch_size} "
          f"peak_gpu={ds.peak_gpu_bytes/1e6:.1f}MB "
          f"kernel_time={ds.kernel_time:.3f}s "
          f"grid_time={ds.grid_creation_time:.2f}s")


if __name__ == '__main__':
    run_test()
