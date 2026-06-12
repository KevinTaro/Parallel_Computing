"""
data_loader_v17_cuda_opt_multi.py

v17: OPTIMIZED custom CUDA decode on a MULTI-CORE feed  (multi x tuned-CUDA)
===========================================================================
Same ablation cell as v15 (multi-threaded feed, hand-written CUDA decode, no
nvJPEG) but using the optimized decoder ``gpu_jpeg_decoder_optimized``.
Bit-identical output to v15/v14; the difference is throughput.

Identical to v16 EXCEPT the raw compressed tiles are read by a pool of threads
instead of one. See v16 for the decoder optimizations (register bit-buffer,
8-bit Huffman LUT, constant-memory tables, DC-only IDCT skip, fused
luma-counting with no RGB buffer, and a fixed, manually-set ``batch_size`` that
can be tuned per card -- RTX 5090 / RTX 4060 / GTX 1060).

Base: data_loader_v0b_multi_baseline.py. Compare v17 vs v15 for the optimization
win; v17 vs v16 for read-parallelism; v17 vs v13 for tuned-CUDA vs nvJPEG.
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

from data_loader_v11_gpu_decode_5090 import _read_tiff_tiles
from gpu_jpeg_decoder_optimized import GpuJpegDecoderOptimized


class WSISlidingWindowDataset(Dataset):
    """Multi-core feed + optimized custom CUDA decode (v0b base)."""

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
            raise ValueError("v17 custom CUDA decoder needs JPEG tiles (compression 7).")
        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            d = self._decoder.device
            print(f"[*] v17 on {d['name']} (cc{d['cc']}, {d['free']/1e9:.1f}GB free) "
                  f"batch={self.batch_size}, readers={self.num_readers}")
            print(f"[*] v17 grid done in {self.grid_creation_time:.2f}s "
                  f"(read {self.read_time:.3f} + cuda-decode+count {self.decode_time:.3f}); "
                  f"kept {len(self.coordinates)}")
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
            raise ValueError("v17 requires tile-aligned geometry.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v17 (multi feed x{self.num_readers} + optimized CUDA decode): "
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
            with read_lock:
                fh.seek(offsets[tid])
                return fh.read(bytecounts[tid])

        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                for start in range(0, len(decode_idx), self.batch_size):
                    cidx = decode_idx[start:start + self.batch_size]
                    chunk = [tile_ids[ci] for ci in cidx]

                    # ---- MULTI-CORE read: thread pool over the raw tiles ----
                    t0 = time.perf_counter()
                    tiles = list(pool.map(read_tile, chunk))
                    self.read_time += time.perf_counter() - t0

                    # ---- optimized CUDA decode + fused white/black count ----
                    cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
                    w, b = self._decoder.count_batch(tiles, self.white_pixel_threshold,
                                                     self.black_pixel_threshold)
                    cp.cuda.Stream.null.synchronize(); self.decode_time += time.perf_counter() - t0

                    white_per_tile[cidx] = w
                    black_per_tile[cidx] = b
                    self.peak_gpu_bytes = max(self.peak_gpu_bytes,
                                              cp.get_default_memory_pool().used_bytes())
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
    print("==== v17 optimized CUDA decode (multi-core feed) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | read {ds.read_time:.3f}s cuda-decode+count {ds.decode_time:.3f}s")


if __name__ == '__main__':
    run_test()
