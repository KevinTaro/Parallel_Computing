"""
data_loader_v13_gpu_decode_multi.py

v13: nvJPEG DECODE on a MULTI-CORE feed  (ablation cell: multi x specialized)
=============================================================================
Research question: *with the decode already on the GPU's specialized JPEG unit,
how much does parallelising the CPU-side raw-tile reads (v0b-style multi-core)
add on top of the mono-core v12?*

Base: data_loader_v0b_multi_baseline.py (all-cores CPU parallelism).
Identical to v12 EXCEPT the raw compressed tiles are read by a pool of threads
instead of one. nvJPEG releases the work to the GPU regardless; the only thing
parallelised here is the file I/O that feeds it. (Threads, not processes: the
read is I/O-bound and the GPU context is not fork-safe across processes, so
threads are the right tool for a GPU-fed pipeline.)

Compare v13 vs v12 -> the benefit of read parallelism once decode is on the GPU.
Compare v13 vs v0b   -> specialized GPU decode vs multi-core CPU decode.

Same geometry / codec requirements and arithmetic notes as v12.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_jpeg

from data_loader_v11_gpu_decode_5090 import _read_tiff_tiles

_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b',
    out_params='uint8 gray',
    operation='gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;',
    name='pil_luma_uint8',
)


class WSISlidingWindowDataset(Dataset):
    """Multi-core feed + nvJPEG GPU decode (v0b base, specialized-decode cell)."""

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
            raise ValueError("v13 needs JPEG tiles (compression 7); use v15 instead.")

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            print(f"[*] v13 grid done in {self.grid_creation_time:.2f}s "
                  f"(read {self.read_time:.3f} + decode {self.decode_time:.3f} + "
                  f"filter {self.kernel_time:.3f}); kept {len(self.coordinates)}")
        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    coords.append((x, y))
        return coords

    def _splice(self, tile: bytes) -> bytes:
        jt = self._tiff["jpegtables"]
        return tile[:2] + jt[2:-2] + tile[2:] if jt else tile

    def _create_grid(self) -> List[Tuple[int, int]]:
        ps = self.patch_size
        tw, th = self._tiff["tile_width"], self._tiff["tile_height"]
        iw, ih = self._tiff["image_width"], self._tiff["image_height"]
        if not (self.stride == ps and ps % tw == 0 and ps % th == 0
                and iw % tw == 0 and ih % th == 0):
            raise ValueError("v13 requires tile-aligned geometry; use v15.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v13 (multi feed x{self.num_readers} + nvJPEG, batch={self.batch_size}): "
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

        fh = open(self.wsi_path, 'rb')
        read_lock = threading.Lock()

        def read_tile(tid: int) -> bytes:
            with read_lock:                      # single fh; serialise the seek+read
                fh.seek(offsets[tid])
                return self._splice(fh.read(bytecounts[tid]))

        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                for start in range(0, len(decode_idx), self.batch_size):
                    cidx = decode_idx[start:start + self.batch_size]
                    chunk = [tile_ids[ci] for ci in cidx]

                    # ---- MULTI-CORE read: thread pool over the raw tiles ----
                    t0 = time.perf_counter()
                    streams = list(pool.map(read_tile, chunk))
                    buffers = [torch.frombuffer(bytearray(s), dtype=torch.uint8) for s in streams]
                    self.read_time += time.perf_counter() - t0

                    # ---- GPU nvJPEG decode ----
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    stacked = torch.stack(decode_jpeg(buffers, mode=ImageReadMode.RGB, device='cuda'))
                    torch.cuda.synchronize(); self.decode_time += time.perf_counter() - t0

                    # ---- GPU filter ----
                    e0, e1 = cp.cuda.Event(), cp.cuda.Event(); e0.record()
                    ct = cp.asarray(stacked); n = ct.shape[0]
                    gray = cp.empty((n, th, tw), dtype=cp.uint8)
                    _luma_kernel(ct[:, 0], ct[:, 1], ct[:, 2], gray)
                    w = cp.count_nonzero(gray > self.white_pixel_threshold, axis=(1, 2))
                    b = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2))
                    e1.record(); e1.synchronize()
                    self.kernel_time += cp.cuda.get_elapsed_time(e0, e1) / 1000.0
                    white_per_tile[cidx] = cp.asnumpy(w)
                    black_per_tile[cidx] = cp.asnumpy(b)
                    self.peak_gpu_bytes = max(self.peak_gpu_bytes,
                        cp.get_default_memory_pool().used_bytes() + torch.cuda.max_memory_allocated())
                    del stacked, ct, gray
        finally:
            fh.close()

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
    print("==== v13 nvJPEG decode (multi-core feed) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | read {ds.read_time:.3f}s decode {ds.decode_time:.3f}s "
          f"filter {ds.kernel_time:.3f}s")


if __name__ == '__main__':
    run_test()
