"""
data_loader_v25_dec_v3_hybrid.py

v25: GPU-decode translation of the v3 "hybrid CPU/GPU threshold" concept
=========================================================================
v3 (data_loader_v3_cupy_hybrid.py) recognised that sending tiny workloads to the
GPU costs more (in H2D + kernel launch + sync overhead) than simply running on the
CPU. The hybrid strategy was: if the workload is large enough AND sufficient GPU
memory is available, use the GPU path; otherwise fall back to the CPU path.

Translation to the GPU-decode world:
    - GPU path: ``GpuJpegDecoderOptimized.count_batch`` in batches (same as v24).
    - CPU fallback path: read patches via OpenSlide, convert to numpy, compute luma
      using the integer approximation formula from v6:
          gray = (r*19595 + g*38470 + b*7471 + 32768) >> 16
      then threshold and count white/black pixels.
    - Two guard conditions trigger the CPU fallback:
        1. ``len(decode_idx) < self.gpu_threshold`` (too few tiles to amortise GPU overhead)
        2. ``free_mb < self.min_vram_mb`` (not enough VRAM for even one batch buffer)
    - ``self.cpu_tiles`` and ``self.gpu_tiles`` track how many tiles went each way.

Parameters:
    batch_size     int = 2048    tiles per GPU count_batch call
    gpu_threshold  int = 128     min non-empty tiles to engage GPU decoder
    min_vram_mb    float = 500   min free VRAM (MB) to engage GPU decoder

Luma formula (integer, BT.601 coefficients, same as v6):
    gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16

Note: The CPU fallback reads PATCHES (via OpenSlide) rather than individual tiles,
so tile-level counts are approximated by counting over the full patch and dividing
by tiles-per-patch. This is consistent with the v3 "best effort" philosophy.
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


def _cpu_luma_counts(arr: np.ndarray,
                     white_thr: int,
                     black_thr: int) -> Tuple[int, int]:
    """Count white/black pixels using integer BT.601 luma approximation."""
    r = arr[:, :, 0].astype(np.int32)
    g = arr[:, :, 1].astype(np.int32)
    b = arr[:, :, 2].astype(np.int32)
    gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16
    white = int(np.count_nonzero(gray > white_thr))
    black = int(np.count_nonzero(gray < black_thr))
    return white, black


class WSISlidingWindowDataset(Dataset):
    """Hybrid CPU/GPU decode -- translates v3 hybrid concept."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 2048,
                 gpu_threshold: int = 128,
                 min_vram_mb: float = 500.0,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
        self.gpu_threshold = gpu_threshold
        self.min_vram_mb = min_vram_mb
        self.verbose = verbose
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0
        self.cpu_tiles = 0
        self.gpu_tiles = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v25 requires JPEG tiles (compression == 7).")

        self._decoder: Optional[GpuJpegDecoderOptimized] = None

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start

        if self.verbose:
            path_used = "GPU" if self.gpu_tiles > 0 else "CPU"
            print(f"[v25] grid done in {self.grid_creation_time:.2f}s path={path_used} "
                  f"gpu_tiles={self.gpu_tiles} cpu_tiles={self.cpu_tiles} "
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

        # Hybrid decision: check VRAM and tile count
        free_mb = cp.cuda.Device().mem_info[0] / 1e6
        use_gpu = (len(decode_idx) >= self.gpu_threshold and free_mb >= self.min_vram_mb)

        if self.verbose:
            print(f"[v25] decode_idx={len(decode_idx)} free_mb={free_mb:.0f} "
                  f"gpu_threshold={self.gpu_threshold} min_vram_mb={self.min_vram_mb} "
                  f"-> {'GPU' if use_gpu else 'CPU'} path")

        wt, bt = self.white_pixel_threshold, self.black_pixel_threshold

        if use_gpu:
            # ---- GPU path: same as v24 ----
            self._decoder = self._build_decoder()
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
                    self.gpu_tiles += len(chunk)

            self.peak_gpu_bytes = max(self.peak_gpu_bytes,
                                      cp.get_default_memory_pool().used_bytes())
        else:
            # ---- CPU fallback path: OpenSlide + numpy luma ----
            # We count at the patch level and distribute equally across tiles in
            # each patch (v3 "best effort" approximation for the fallback path).
            tiles_per_patch = txp * typ

            # Build a mapping from tile_id to list of patch indices that contain it
            tile_to_patches: dict = {}
            for pi, (x, y, ids) in enumerate(patch_tiles):
                for t in ids:
                    tile_to_patches.setdefault(t, []).append(pi)

            # Per-patch accumulators
            patch_white = np.zeros(len(patch_tiles), dtype=np.int64)
            patch_black = np.zeros(len(patch_tiles), dtype=np.int64)

            with openslide.OpenSlide(self.wsi_path) as slide:
                for pi, (x, y, ids) in enumerate(patch_tiles):
                    region = slide.read_region((x, y), 0,
                                               (self.patch_size, self.patch_size))
                    arr = np.array(region.convert('RGB'))
                    wc, bc = _cpu_luma_counts(arr, wt, bt)
                    patch_white[pi] = wc
                    patch_black[pi] = bc

            # Distribute patch-level counts to constituent tiles (even split)
            for pi, (x, y, ids) in enumerate(patch_tiles):
                n = len(ids)
                if n == 0:
                    continue
                wc_per = patch_white[pi] // n
                bc_per = patch_black[pi] // n
                for t in ids:
                    ci = needed[t]
                    if bytecounts[t] > 0:
                        white_per_tile[ci] += wc_per
                        black_per_tile[ci] += bc_per

            self.cpu_tiles = len(decode_idx)

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
    print("==== v25 GPU-decode v3-hybrid - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v25] kept={len(ds)} gpu_tiles={ds.gpu_tiles} cpu_tiles={ds.cpu_tiles} "
          f"kernel_time={ds.kernel_time:.3f}s grid_time={ds.grid_creation_time:.2f}s")


if __name__ == '__main__':
    run_test()
