"""
data_loader_v20_cuda_ultimate_pread.py

v20: ULTIMATE custom CUDA decode with TRUE parallel reads (pread multi-feed)
============================================================================
v19 advertised a "multi-core feed" but did not actually read in parallel. Its
reader threads all shared ONE file object and a ``threading.Lock``:

    fh = open(path, 'rb'); read_lock = threading.Lock()
    def read_tile(tid):
        with read_lock:              # <-- every thread serialises here
            fh.seek(offsets[tid]); return fh.read(bytecounts[tid])

A single file object has one OS file position, so concurrent ``seek``+``read``
is unsafe and the lock is mandatory -- which means only one thread ever reads at
a time. v19's reads were therefore *serial with thread-pool overhead on top*
(its ``read_time`` matched mono v18 almost exactly, confirming no parallelism).

v20 fixes this with **positional reads**: ``os.pread(fd, nbytes, offset)`` is a
single atomic syscall that does not touch the shared file position, so many
threads can call it on the SAME file descriptor concurrently with NO lock. The
GIL is released for the duration of each syscall, so the reads genuinely run in
parallel (as parallel as the storage allows). This is the read-parallelism v19
claimed but never delivered.

Everything else is identical to v18/v19: the optimized custom-CUDA decoder
(register bit-buffer, 8-bit Huffman LUT, constant-memory tables, DC-only IDCT
skip, fused YCbCr->luma->count with no RGB buffer), pinned host staging, the
double-buffered async-stream pipeline, and the reused buffer pool. Output is
**bit-identical** to v14/v16/v18/v19 and the v0a baseline.

When this actually helps (and when it does not)
-----------------------------------------------
On a WARM-CACHE local SSD the raw-tile read is ~7% of the wall time and is
already hidden behind the GPU decode by the pipeline (v18's gpu-wait is 0.000s).
So on a fast warm disk v20 ~= v18 ~= v19: the parallel read shrinks a cost that
was not on the critical path. v20 pays off when the read is genuinely slow and
NOT hidden -- cold cache, network/NFS storage, or a slow HDD -- where serial
reads (v18) or lock-serialised reads (v19) would stall the GPU. Then true
parallel pread keeps the decode pipeline fed. ``read_time`` is reported so you
can see, on your storage, whether the read was ever the bottleneck.

NOTE: ``os.pread`` is POSIX (Linux/macOS). On Windows fall back to per-thread
file handles (each its own position, no lock) -- same idea, more handles.
"""
import os
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
    """True-parallel-read feed (os.pread) + ultimate pipelined custom CUDA decode."""

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
            raise ValueError("v20 custom CUDA decoder needs JPEG tiles (compression 7).")
        if not hasattr(os, "pread"):
            raise OSError("v20 needs os.pread (POSIX). On Windows use per-thread "
                          "file handles instead (see module docstring).")
        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            d = self._decoder.device
            print(f"[*] v20 on {d['name']} (cc{d['cc']}) batch={self.batch_size} "
                  f"slots={self.n_slots} readers={self.num_readers} (os.pread)")
            print(f"[*] v20 grid done in {self.grid_creation_time:.2f}s "
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
            raise ValueError("v20 requires tile-aligned geometry.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v20 (os.pread parallel feed + ultimate CUDA pipeline): "
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

        # ---- TRUE parallel reads: one fd, many threads, os.pread (no lock) ----
        # os.pread(fd, n, offset) is an atomic positional syscall that does not
        # use the shared file position, so concurrent calls on the same fd need
        # no lock and the GIL is released during each read -> reads run in
        # parallel. (v19 shared one file object + a lock and was serial.)
        fd = os.open(self.wsi_path, os.O_RDONLY)

        def read_tile(tid: int) -> bytes:
            return os.pread(fd, bytecounts[tid], offsets[tid])

        pending: "deque" = deque()
        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                for i, cidx in enumerate(chunks):
                    # Parallel decode-free read of the batch's raw JPEG tiles.
                    t0 = time.perf_counter()
                    tiles = list(pool.map(read_tile, (tile_ids[ci] for ci in cidx)))
                    self.read_time += time.perf_counter() - t0

                    slot = i % self.n_slots
                    if len(pending) == self.n_slots:
                        ps_slot, pcidx = pending.popleft()
                        t0 = time.perf_counter()
                        w, b = self._decoder.fetch(ps_slot)
                        self.decode_time += time.perf_counter() - t0
                        white_per_tile[pcidx] = w
                        black_per_tile[pcidx] = b

                    self._decoder.submit(slot, tiles, wt, bt)
                    pending.append((slot, cidx))

                while pending:
                    ps_slot, pcidx = pending.popleft()
                    t0 = time.perf_counter()
                    w, b = self._decoder.fetch(ps_slot)
                    self.decode_time += time.perf_counter() - t0
                    white_per_tile[pcidx] = w
                    black_per_tile[pcidx] = b
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
    print("==== v20 ultimate CUDA pipeline (true parallel pread feed) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | read {ds.read_time:.3f}s gpu-wait {ds.decode_time:.3f}s")


if __name__ == '__main__':
    run_test()
