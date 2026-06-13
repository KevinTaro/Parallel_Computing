"""
data_loader_v24_dec_v2_batch.py

v24: GPU-decode translation of the v2 "batch N patches together" concept
=========================================================================
v2 (data_loader_v2_cupy_batch.py) introduced the key insight that GPU throughput
is dominated by launch overhead when batches are tiny. The fix: accumulate N
patches and send them as one CuPy operation. On the GPU-decode pipeline the same
principle applies directly: batch multiple compressed JPEG tiles into a single
``count_batch(tiles, ...)`` call.

Translation to the GPU-decode world:
    - ``batch_size`` tiles are read from the file sequentially.
    - A single ``GpuJpegDecoderOptimized.count_batch(tiles, wt, bt)`` processes
      all of them in one fused CUDA kernel (decode JPEG -> Y plane -> threshold ->
      count), returning two int64 arrays of length N.
    - ``batch_size=0`` triggers auto-sizing from free VRAM: enough to keep the GPU
      ~35% busy with decode buffers, clamped to [64, 8192].
    - Expected: significant speedup over v23 (batch=1). The first batch call shows
      the full kernel-launch + H2D latency; subsequent calls amortise it over N tiles.

Auto batch sizing formula:
    free_bytes = cp.cuda.Device().mem_info[0]
    per_tile   = 512*512*3   (RGB decode buffer)
               + 96*1024     (compressed scan staging, ~96 KB worst case)
    batch_size = clamp(int(free_bytes * 0.35 / per_tile), 64, 8192)

Comparison guide:
    v23 (batch=1)  -> baseline, all kernel-launch overhead
    v24 (batch=N)  -> this file, N-tile amortisation
    v25 (hybrid)   -> adds CPU fallback for low-VRAM or few-tile cases
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
    """Batched GPU tile decode -- translates v2 batch-N concept."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 0,
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
        self.peak_gpu_bytes = 0

        # Auto batch_size from free VRAM when 0
        if batch_size == 0:
            free = cp.cuda.Device().mem_info[0]
            per_tile = 512 * 512 * 3 + 96 * 1024
            self.batch_size = int(min(max(64, free * 0.35 // per_tile), 8192))
        else:
            self.batch_size = batch_size

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v24 requires JPEG tiles (compression == 7).")

        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start

        if self.verbose:
            d = self._decoder.device
            print(f"[v24] {d['name']} (cc{d['cc']}) batch={self.batch_size}")
            print(f"[v24] grid done in {self.grid_creation_time:.2f}s "
                  f"kernel_time={self.kernel_time:.3f}s kept={len(self.coordinates)}")

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
            print(f"[v24] {len(candidates)} candidate patches, "
                  f"{len(self._tiff['offsets'])} tiles total, batch={self.batch_size}")

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

        # v2 concept: batch N tiles per GPU call
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

        self.peak_gpu_bytes = max(self.peak_gpu_bytes,
                                  cp.get_default_memory_pool().used_bytes())

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
    print("==== v24 GPU-decode v2-batch - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v24] kept={len(ds)} kernel_time={ds.kernel_time:.3f}s "
          f"batch={ds.batch_size} grid_time={ds.grid_creation_time:.2f}s")


if __name__ == '__main__':
    run_test()
