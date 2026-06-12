"""
data_loader_v19_cuda_ultimate_multi.py

v19: ULTIMATE custom CUDA decode on a MULTI-CORE feed  (multi x ultimate-CUDA)
=============================================================================
The end of the line for the multi-feed custom-CUDA branch. Identical to v18
(optimized decoder + pinned memory + double-buffered async-stream pipeline +
reused buffer pool, all bit-identical to v14/v16) EXCEPT the raw compressed
tiles are read by a pool of threads instead of one.

So v19 stacks every available parallelism axis: parallel CPU reads, async H2D
transfer overlapped with compute, and the massively parallel custom CUDA decode
-- the literal "stack every layer" version of CUPY_RESEARCH_PLAN.md's v9.

Compare v19 vs v18 for the read-parallelism delta; v19 vs v17 for the
pinned+pipeline overlap win; v19 vs v13 for ultimate-CUDA vs nvJPEG.
"""
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader_v11_gpu_decode_5090 import _read_tiff_tiles
from gpu_jpeg_decoder_ultimate import GpuJpegDecoderUltimate


class WSISlidingWindowDataset(Dataset):
    """Multi-core feed + ultimate pipelined custom CUDA decode (v0b base)."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 2048,
                 n_slots: int = 2,
                 num_readers: Optional[int] = None,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
        self.n_slots = n_slots
        self.num_readers = num_readers or (os.cpu_count() or 4)
        self.verbose = verbose
        self.read_time = 0.0
        self.decode_time = 0.0
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v19 custom CUDA decoder needs JPEG tiles (compression 7).")
        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            d = self._decoder.device
            print(f"[*] v19 on {d['name']} (cc{d['cc']}) batch={self.batch_size} "
                  f"slots={self.n_slots} readers={self.num_readers}")
            print(f"[*] v19 grid done in {self.grid_creation_time:.2f}s "
                  f"(read {self.read_time:.3f} + gpu-wait {self.decode_time:.3f}); "
                  f"kept {len(self.coordinates)}")
        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

    def _build_decoder(self) -> GpuJpegDecoderUltimate:
        offs, bcs = self._tiff["offsets"], self._tiff["bytecounts"]
        with open(self.wsi_path, 'rb') as fh:
            tid = next(i for i, b in enumerate(bcs) if b > 0)
            fh.seek(offs[tid]); sample = fh.read(bcs[tid])
        return GpuJpegDecoderUltimate(self._tiff["jpegtables"], sample,
                                      max_batch=self.batch_size, n_slots=self.n_slots)

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
            raise ValueError("v19 requires tile-aligned geometry.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v19 (multi feed x{self.num_readers} + ultimate CUDA pipeline): "
                  f"{len(candidates)} patches over {len(self._tiff['offsets'])} tiles")

        patch_tiles, needed = [], {}
        for (x, y) in candidates:
            col0, row0 = x // tw, y // th
            ids = []
            for dr in range(typ):
                base = (row0 + dr) * tiles_per_row + col0
                for dc in range(txp):
                    tid = base + dc
                    ids.append(tid)
                    needed.setdefault(tid, len(needed))
            patch_tiles.append((x, y, ids))

        tile_ids = list(needed.keys())
        offsets, bytecounts = self._tiff["offsets"], self._tiff["bytecounts"]
        white_per_tile = np.zeros(len(tile_ids), dtype=np.int64)
        black_per_tile = np.zeros(len(tile_ids), dtype=np.int64)

        TILE_PX = tw * th
        decode_idx = []
        for ci, tid in enumerate(tile_ids):
            if bytecounts[tid] > 0:
                decode_idx.append(ci)
            else:
                black_per_tile[ci] = TILE_PX

        chunks = [decode_idx[i:i + self.batch_size]
                  for i in range(0, len(decode_idx), self.batch_size)]
        wt, bt = self.white_pixel_threshold, self.black_pixel_threshold

        fh = open(self.wsi_path, 'rb')
        read_lock = threading.Lock()

        def read_tile(tid: int) -> bytes:
            with read_lock:
                fh.seek(offsets[tid])
                return fh.read(bytecounts[tid])

        pending: "deque" = deque()
        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                for i, cidx in enumerate(chunks):
                    # ---- MULTI-CORE read: thread pool over the raw tiles ----
                    t0 = time.perf_counter()
                    tiles = list(pool.map(read_tile, (tile_ids[ci] for ci in cidx)))
                    self.read_time += time.perf_counter() - t0

                    slot = i % self.n_slots
                    if len(pending) == self.n_slots:
                        ps, pcidx = pending.popleft()
                        t0 = time.perf_counter()
                        w, b = self._decoder.fetch(ps)
                        self.decode_time += time.perf_counter() - t0
                        white_per_tile[pcidx] = w
                        black_per_tile[pcidx] = b

                    self._decoder.submit(slot, tiles, wt, bt)
                    pending.append((slot, cidx))

                while pending:
                    ps, pcidx = pending.popleft()
                    t0 = time.perf_counter()
                    w, b = self._decoder.fetch(ps)
                    self.decode_time += time.perf_counter() - t0
                    white_per_tile[pcidx] = w
                    black_per_tile[pcidx] = b
        finally:
            fh.close()
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
    print("==== v19 ultimate CUDA pipeline (multi-core feed) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | read {ds.read_time:.3f}s gpu-wait {ds.decode_time:.3f}s")


if __name__ == '__main__':
    run_test()
