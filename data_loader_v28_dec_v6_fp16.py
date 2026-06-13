"""
data_loader_v28_dec_v6_fp16.py

v28: GPU-decode translation of the v6 "fp16 mixed precision" concept
=====================================================================
v6 (data_loader_v6_cupy_mixed_precision.py) used fp16 (half-precision float) for
the CuPy luma computation after CPU JPEG decode. The argument: fp16 halves memory
bandwidth for the luma pass because each channel value goes from 8-bit (uint8)
through a 32-bit float intermediate down to a 16-bit accumulation.

Translation to the GPU-decode world:
    The custom CUDA decoder (``GpuJpegDecoderOptimized``) uses float32 internally
    for its IDCT and YCbCr operations to maintain precision. The decoded data is
    returned as uint8 RGB or as separate Y/Cb/Cr planes. This version:

    1. Calls ``decoder._decode_planes(tiles)`` to get the Y (luminance) plane
       directly as ``Yp`` shape ``(N, 512, 512)`` uint8 cupy array.
    2. Casts to fp16: ``Yf = Yp.astype(cp.float16)``
    3. Counts white/black pixels using fp16 CuPy comparisons:
           w = cp.count_nonzero(Yf > wp, axis=(1,2))
           b = cp.count_nonzero(Yf < bp, axis=(1,2))

    Why Y plane directly?
        In YCbCr, the Y channel IS the luminance signal (BT.601 luma). For
        white/black tissue detection, comparing Y against a threshold is equivalent
        to computing full RGB luma -- and more direct, because:
          - No 3-channel memory bandwidth (only Y plane needed, 1/3 the data of RGB)
          - No luma formula computation needed; Y is already luminance
          - fp16 is sufficient because the thresholds (230, 25) are exact integers
            representable in fp16, and the Y values are integers 0..255, also exact.

    Important distinction from v6:
        v6 applied fp16 AFTER CPU JPEG decode (the decode itself was CPU float32).
        v28 keeps the GPU IDCT in float32 (the decoder's internal precision), and
        applies fp16 ONLY at the count/threshold step -- so "mixed precision" here
        means: float32 decode + fp16 count. This is MORE correct than v6 in that
        the decode precision is not compromised; only the cheap count uses fp16.

    Boundary pixel note:
        fp16 has 10 mantissa bits (~3 decimal digits). For Y values in [0,255],
        fp16 is exact (integers up to 2048 are exact in fp16). However, if the
        decoder's internal pipeline produces non-integer Y values before rounding
        to uint8, the cast path is uint8->fp16, which is always exact. No boundary
        pixel flip risk in this specific pipeline.

Parameters:
    batch_size  int = 2048   tiles per _decode_planes call
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
    """fp16 CuPy luma counting on the Y plane -- translates v6 fp16 concept."""

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
            raise ValueError("v28 requires JPEG tiles (compression == 7).")

        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start

        if self.verbose:
            d = self._decoder.device
            print(f"[v28] {d['name']} (cc{d['cc']}) batch={self.batch_size} "
                  f"(fp16 Y-plane count)")
            print(f"[v28] grid done in {self.grid_creation_time:.2f}s "
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
        wp = cp.float16(wt)
        bp = cp.float16(bt)

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

                # Get Y plane directly (luminance = Y in YCbCr)
                # _decode_planes returns Yp(N,512,512), Cbp(N,256,256), Crp(N,256,256)
                Yp, _Cbp, _Crp = self._decoder._decode_planes(tiles)

                # fp16 luma approximation: Y channel IS luminance
                Yf = Yp.astype(cp.float16)

                # Count white and black pixels using fp16 comparisons
                w = cp.count_nonzero(Yf > wp, axis=(1, 2)).astype(np.int64)
                b = cp.count_nonzero(Yf < bp, axis=(1, 2)).astype(np.int64)

                ev_end.record()
                ev_end.synchronize()
                self.kernel_time += cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e-3

                w_cpu = cp.asnumpy(w)
                b_cpu = cp.asnumpy(b)
                for local_i, ci in enumerate(chunk):
                    white_per_tile[ci] = int(w_cpu[local_i])
                    black_per_tile[ci] = int(b_cpu[local_i])

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
    print("==== v28 GPU-decode v6-fp16 (Y-plane fp16 count) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v28] kept={len(ds)} kernel_time={ds.kernel_time:.3f}s "
          f"batch={ds.batch_size} grid_time={ds.grid_creation_time:.2f}s")
    print("NOTE: Decode is float32 (IDCT); only the count step uses fp16 on Y plane.")


if __name__ == '__main__':
    run_test()
