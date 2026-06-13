"""
data_loader_v30_dec_v8_threaded.py

v30: GPU-decode translation of the v8 "ThreadPoolExecutor parallel reads" concept
==================================================================================
v8 (data_loader_v8_cupy_optimized_4060.py) used a ThreadPoolExecutor to issue
parallel file reads, accumulating a large batch before handing it to the GPU.
The theory: parallel reads saturate the storage bandwidth, filling the batch
faster than sequential reads.

Translation to the GPU-decode world:
    The same pattern is applied here using ``os.pread`` (which is GIL-releasing
    and positionally atomic, unlike ``fh.seek + fh.read`` which requires locking):

        fd = os.open(path, os.O_RDONLY)
        read_tile = lambda tid: os.pread(fd, bytecounts[tid], offsets[tid])
        tiles = list(pool.map(read_tile, [tile_ids[ci] for ci in chunk]))

    A single file descriptor is shared across all reader threads; ``os.pread``
    releases the GIL per syscall, allowing genuine parallelism. This is the
    v8 "parallel reads + large batch" concept correctly implemented.

    Honest expectation (GPU-decode world):
        On a WARM-CACHE local SSD the GPU JPEG decode dominates: it is ~90-93%
        of wall time. The raw tile reads are ~7%. Therefore:
            - Parallelising reads reduces the 7% to maybe 3-4% (2x read speedup)
            - Total speedup: at most 4% faster than v24 (sequential reads + same batch)
            - The ThreadPoolExecutor dispatch + result collection adds its own
              overhead (Python thread wakeup, future collection) which is NOT on the
              GPU's critical path but IS on the CPU path between batches.
            - Net result: v30 is likely EQUAL to or SLIGHTLY SLOWER than v24 on
              warm local storage, because the pool overhead roughly cancels the
              read parallelism gain.
        On cold storage (SSD cold cache, NFS, HDD) where reads are 50%+ of wall
        time, v30 will clearly beat v24.
        Compare v30 vs v24 to measure the pool overhead vs read-parallelism tradeoff.

    Why os.pread fixes v19's mistake:
        v19 ("multi") shared one file OBJECT (``open()`` fd) across threads with a
        mutex, so reads were effectively serialised. ``os.pread`` is a POSIX syscall
        that takes an explicit offset, bypassing the file position state entirely --
        all threads can issue concurrent syscalls on the same fd without any lock.

Parameters:
    batch_size   int  = 2048   tiles per count_batch call
    num_readers  int  = None   thread pool size; None -> os.cpu_count()
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
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
    """ThreadPool parallel tile reads + large-batch GPU decode -- translates v8 concept."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 2048,
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
        self.num_readers = num_readers if num_readers is not None else os.cpu_count()
        self.verbose = verbose
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v30 requires JPEG tiles (compression == 7).")

        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start

        if self.verbose:
            d = self._decoder.device
            print(f"[v30] {d['name']} (cc{d['cc']}) batch={self.batch_size} "
                  f"readers={self.num_readers}")
            print(f"[v30] grid done in {self.grid_creation_time:.2f}s "
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
            print(f"[v30] {len(candidates)} candidates, "
                  f"{len(self._tiff['offsets'])} tiles, "
                  f"batch={self.batch_size} readers={self.num_readers}")

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

        # Open a single fd for all reader threads (os.pread is positionally atomic)
        fd = os.open(self.wsi_path, os.O_RDONLY)
        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                for chunk_start in range(0, len(decode_idx), bs):
                    chunk = decode_idx[chunk_start: chunk_start + bs]
                    chunk_tids = [tile_ids[ci] for ci in chunk]

                    # v8 concept: parallel reads via os.pread (GIL-releasing)
                    def read_tile(tid: int) -> bytes:
                        return os.pread(fd, bytecounts[tid], offsets[tid])

                    tiles = list(pool.map(read_tile, chunk_tids))

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
        finally:
            os.close(fd)

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
    print("==== v30 GPU-decode v8-threaded (os.pread pool.map) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v30] kept={len(ds)} kernel_time={ds.kernel_time:.3f}s "
          f"batch={ds.batch_size} readers={ds.num_readers} "
          f"grid_time={ds.grid_creation_time:.2f}s")
    print("NOTE: Compare v30 vs v24 -- read parallelism vs pool overhead tradeoff.")
    print("On warm-cache SSD: v30 ~= v24 (GPU dominates; pool overhead cancels gain).")
    print("On cold/NFS: v30 should clearly beat v24 (reads become the bottleneck).")


if __name__ == '__main__':
    run_test()
