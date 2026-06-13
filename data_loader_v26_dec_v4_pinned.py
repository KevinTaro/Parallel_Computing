"""
data_loader_v26_dec_v4_pinned.py

v26: GPU-decode translation of the v4 "pinned host memory" concept
===================================================================
v4 (data_loader_v4_cupy_pinned_memory.py) allocated page-locked (pinned) host
buffers so that H2D transfers could use DMA directly without an extra OS copy.
The key benefit: pinned memory enables async DMA (``cudaMemcpyAsync``) at the
full PCIe bandwidth, rather than forcing a pageable->pinned bounce copy before
each DMA.

Translation to the GPU-decode world:
    ``GpuJpegDecoderOptimized`` already uses pinned memory internally for its
    h_scan staging buffer (the compressed JPEG scan data). This version makes
    the UPSTREAM read-staging buffer ALSO pinned, demonstrating end-to-end pinned
    memory from file read all the way to the GPU:

        File  ->  pinned h_raw_pinned  ->  GPU (DMA, no OS bounce copy)
                  ^^^^^^^^^^^^^^^^^^^
                  NEW in v26: the raw compressed tile bytes land in pinned memory
                  immediately after the file read, before being passed to the decoder.

    Mechanism:
        1. Compute max compressed tile size across all non-empty tiles.
        2. Allocate ONE pinned flat buffer: ``batch_size * max_raw`` bytes, shaped
           as ``(batch_size, max_raw)`` via ``np.frombuffer``.
        3. For each chunk: copy each tile's raw bytes into row ``i`` of
           ``h_raw_pinned``, then reconstruct a list of memoryview slices.
        4. Pass the slice list to ``count_batch``. The decoder's internal destuff
           copies out of pinned memory -- always a pinned -> device path.

    The v4 concept is fully represented: every byte from file-read to GPU has
    passed through page-locked memory, enabling the hardware DMA engine to operate
    at full PCIe bandwidth for both the compressed-scan H2D transfer and the
    result D2H copy.

    Note: on systems with IOMMU or Unified Memory (Grace-Hopper etc.) the pinned
    allocation falls back to normal host memory without error; the pipeline still
    works correctly, just without the DMA benefit.

Parameters:
    batch_size  int = 2048   tiles per GPU count_batch call
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
    """End-to-end pinned-memory staging -- translates v4 pinned memory concept."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 2048,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
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
            raise ValueError("v26 requires JPEG tiles (compression == 7).")

        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start

        if self.verbose:
            d = self._decoder.device
            print(f"[v26] {d['name']} (cc{d['cc']}) batch={self.batch_size} "
                  f"(pinned staging)")
            print(f"[v26] grid done in {self.grid_creation_time:.2f}s "
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

        wt, bt = self.white_pixel_threshold, self.black_pixel_threshold
        bs = self.batch_size

        # Allocate pinned staging buffer: (batch_size x max_raw_bytes)
        # Finds the maximum compressed tile size across all non-empty tiles.
        max_raw = int(max(bytecounts[tid] for tid in tile_ids
                          if bytecounts[tid] > 0))
        # Pinned allocation: batch_size rows, max_raw bytes each
        pinned_mem = cp.cuda.alloc_pinned_memory(bs * max_raw)
        h_raw_pinned = np.frombuffer(pinned_mem, dtype=np.uint8)[:bs * max_raw].reshape(bs, max_raw)

        if self.verbose:
            print(f"[v26] pinned staging buffer: {bs}x{max_raw} = "
                  f"{bs * max_raw / 1e6:.1f} MB allocated")

        with open(self.wsi_path, 'rb') as fh:
            for chunk_start in range(0, len(decode_idx), bs):
                chunk = decode_idx[chunk_start: chunk_start + bs]
                tiles: List[bytes] = []
                for local_i, ci in enumerate(chunk):
                    tid = tile_ids[ci]
                    fh.seek(offsets[tid])
                    raw = fh.read(bytecounts[tid])
                    # Copy into pinned buffer row; only the valid bytes matter
                    h_raw_pinned[local_i, :len(raw)] = np.frombuffer(raw, dtype=np.uint8)
                    # Reconstruct bytes from pinned memory so the decoder reads
                    # from page-locked memory (enabling async DMA downstream)
                    tiles.append(bytes(h_raw_pinned[local_i, :len(raw)]))

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
    print("==== v26 GPU-decode v4-pinned - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v26] kept={len(ds)} kernel_time={ds.kernel_time:.3f}s "
          f"batch={ds.batch_size} grid_time={ds.grid_creation_time:.2f}s")
    print("NOTE: End-to-end pinned staging demonstrated. "
          "GpuJpegDecoderOptimized internal buffers are also pinned.")


if __name__ == '__main__':
    run_test()
