"""
data_loader_v23_dec_v1_naive.py

v23: GPU-decode translation of the v1 "naive per-patch" concept
===============================================================
v1 (data_loader_v1_cupy_full.py) processed one patch at a time: open the slide,
decode just the pixels for that patch, run luma+threshold, move on. The defining
characteristic was ONE computation unit per patch -- no batching at all.

Translation to the GPU-decode world:
    batch_size is HARDCODED to 1. Every tile in the WSI is sent to the GPU as a
    single-element list, so there is exactly ONE ``count_batch([raw_tile], ...)``
    call per tile. This means:
        - One CUDA kernel launch per tile (~2-8 us overhead per launch)
        - One H2D memcpy per tile (pinned -> device, ~1-5 us per transfer)
        - One device->host result copy per tile (2 int64 scalars)
    For a typical 40x WSI with ~100,000 tiles this accumulates to:
        - ~100,000 kernel launches * ~5 us = ~0.5 s of pure launch overhead
        - GPU utilisation <10%: the GPU is idle waiting for the next single-tile
          kernel to arrive while the CPU prepares the next read
    Expected outcome: SLOWEST GPU-decode version. The GPU buys nothing here;
    a plain CPU OpenSlide decode would likely be faster because it avoids the H2D
    transfer overhead entirely.

    DO NOT use this as a production path. It exists solely to demonstrate WHY
    batching matters -- compare v23 (batch=1) vs v24 (batch=N) to see the speedup.

Note: batch_size=1 is NOT a parameter. It is an architectural decision fixed at
the class level to faithfully represent the v1 concept.
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
    """Per-tile GPU decode (batch_size=1 hardcoded) -- translates v1 naive concept."""

    # batch_size is intentionally NOT a constructor parameter.
    # It is fixed at 1 to replicate the v1 "one unit at a time" architecture.
    _BATCH_SIZE: int = 1

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
        # Required instance attributes
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v23 requires JPEG tiles (compression == 7).")

        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start

        if self.verbose:
            d = self._decoder.device
            print(f"[v23] {d['name']} (cc{d['cc']}) batch=1 (hardcoded, v1 naive)")
            print(f"[v23] grid done in {self.grid_creation_time:.2f}s "
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
            print(f"[v23] {len(candidates)} candidate patches, "
                  f"{len(self._tiff['offsets'])} tiles total (batch=1 per tile)")

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

        # v1 concept: one GPU call PER TILE (batch_size=1 hardcoded)
        with open(self.wsi_path, 'rb') as fh:
            for ci in decode_idx:
                tid = tile_ids[ci]
                fh.seek(offsets[tid])
                raw = fh.read(bytecounts[tid])

                # CUDA event timing around each single-tile decode
                ev_start = cp.cuda.Event()
                ev_end = cp.cuda.Event()
                ev_start.record()
                w, b = self._decoder.count_batch([raw], wt, bt)
                ev_end.record()
                ev_end.synchronize()
                self.kernel_time += cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e-3

                white_per_tile[ci] = int(w[0])
                black_per_tile[ci] = int(b[0])

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
    print("==== v23 GPU-decode v1-naive (batch=1 hardcoded) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v23] kept={len(ds)} kernel_time={ds.kernel_time:.3f}s "
          f"grid_time={ds.grid_creation_time:.2f}s")
    print("NOTE: This is the slowest GPU-decode version by design (v1 concept).")


if __name__ == '__main__':
    run_test()
