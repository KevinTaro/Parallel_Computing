"""
data_loader_v14_gpu_compute_mono.py

v14: CUSTOM CUDA DECODE on a MONO-CORE feed  (ablation cell: mono x raw-CUDA)
=============================================================================
Research question: *how much does raw GPU parallelism -- decoding many tiles at
once, one CUDA thread per tile -- speed up the decode task, with NO specialized
decoder (no nvJPEG)?*

Base: data_loader_v0a_mono_baseline.py (single CPU thread).
Change vs v0a: the JPEG decode no longer runs on the CPU. A single CPU thread
reads the raw compressed tiles, but the decode itself (Huffman -> dequant ->
IDCT -> color) runs on the GPU's general CUDA cores via a hand-written kernel
(`gpu_jpeg_decoder.GpuJpegDecoder`) -- one thread per tile, thousands of tiles
in parallel. This is the "pure parallelism, no fixed-function unit" counterpart
to the nvJPEG versions (v12/v13).

Compare v14 vs v0a -> speedup from parallelising decode across CUDA cores.
Compare v14 vs v12 -> general CUDA decode vs the specialized nvJPEG unit.
Compare v14 vs v15 -> what CPU read-parallelism adds on top of the mono feed.

The decoder is deliberately NAIVE (one serial thread per tile, float IDCT) -- we
measure the parallel-across-tiles benefit, not a tuned decoder. Its float IDCT +
nearest chroma upsample are not bit-exact with libjpeg, so a few patches near the
0.9 threshold may differ from v0a; the benchmark reports the exact delta.
"""
import time
from typing import Callable, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader_v11_gpu_decode_5090 import _read_tiff_tiles
from gpu_jpeg_decoder import GpuJpegDecoder

_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b',
    out_params='uint8 gray',
    operation='gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;',
    name='pil_luma_uint8',
)


class WSISlidingWindowDataset(Dataset):
    """Mono-core feed + custom CUDA JPEG decode (v0a base, raw-CUDA cell)."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 1024,
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
        self.read_time = 0.0       # CPU file I/O (single thread)
        self.decode_time = 0.0     # custom CUDA decode
        self.kernel_time = 0.0     # GPU filter
        self.peak_gpu_bytes = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v14 custom CUDA decoder needs JPEG tiles (compression 7).")
        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            print(f"[*] v14 grid done in {self.grid_creation_time:.2f}s "
                  f"(read {self.read_time:.3f} + cuda-decode {self.decode_time:.3f} + "
                  f"filter {self.kernel_time:.3f}); kept {len(self.coordinates)}")
        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

    def _build_decoder(self) -> GpuJpegDecoder:
        offs, bcs = self._tiff["offsets"], self._tiff["bytecounts"]
        with open(self.wsi_path, 'rb') as fh:
            tid = next(i for i, b in enumerate(bcs) if b > 0)
            fh.seek(offs[tid]); sample = fh.read(bcs[tid])
        return GpuJpegDecoder(self._tiff["jpegtables"], sample)

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    coords.append((x, y))
        return coords

    def _count_tiles_gpu(self, rgb: cp.ndarray):
        """Luma + white/black counts for a batch of decoded RGB tiles."""
        n, th, tw, _ = rgb.shape
        e0, e1 = cp.cuda.Event(), cp.cuda.Event(); e0.record()
        gray = cp.empty((n, th, tw), dtype=cp.uint8)
        _luma_kernel(rgb[..., 0], rgb[..., 1], rgb[..., 2], gray)
        w = cp.count_nonzero(gray > self.white_pixel_threshold, axis=(1, 2))
        b = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2))
        e1.record(); e1.synchronize()
        self.kernel_time += cp.cuda.get_elapsed_time(e0, e1) / 1000.0
        return cp.asnumpy(w), cp.asnumpy(b)

    def _create_grid(self) -> List[Tuple[int, int]]:
        ps = self.patch_size
        tw, th = self._tiff["tile_width"], self._tiff["tile_height"]
        iw, ih = self._tiff["image_width"], self._tiff["image_height"]
        if not (self.stride == ps and ps % tw == 0 and ps % th == 0
                and iw % tw == 0 and ih % th == 0):
            raise ValueError("v14 requires tile-aligned geometry.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v14 (mono feed + custom CUDA decode, batch={self.batch_size}): "
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
                black_per_tile[ci] = TILE_PX     # empty tile == OpenSlide black

        fh = open(self.wsi_path, 'rb')
        try:
            for start in range(0, len(decode_idx), self.batch_size):
                cidx = decode_idx[start:start + self.batch_size]

                # ---- MONO-CORE read: one thread, strictly sequential ----
                t0 = time.perf_counter()
                tiles = []
                for ci in cidx:
                    tid = tile_ids[ci]
                    fh.seek(offsets[tid]); tiles.append(fh.read(bytecounts[tid]))
                self.read_time += time.perf_counter() - t0

                # ---- custom CUDA decode (1 thread / tile, parallel across tiles) ----
                cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
                rgb = self._decoder.decode_batch(tiles)
                cp.cuda.Stream.null.synchronize(); self.decode_time += time.perf_counter() - t0

                w, b = self._count_tiles_gpu(rgb)
                white_per_tile[cidx] = w
                black_per_tile[cidx] = b
                self.peak_gpu_bytes = max(self.peak_gpu_bytes,
                                          cp.get_default_memory_pool().used_bytes())
                del rgb
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
    print("==== v14 custom CUDA decode (mono-core feed) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | read {ds.read_time:.3f}s cuda-decode {ds.decode_time:.3f}s "
          f"filter {ds.kernel_time:.3f}s")


if __name__ == '__main__':
    run_test()
