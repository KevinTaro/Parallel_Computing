"""
data_loader_v16_cuda_opt_mono.py

v16: OPTIMIZED custom CUDA decode on a MONO-CORE feed  (mono x tuned-CUDA)
=========================================================================
Same ablation cell as v14 (mono feed, hand-written CUDA decode, no nvJPEG) but
using the *optimized, auto-tuning* decoder ``gpu_jpeg_decoder_optimized`` instead
of the naive one. Bit-identical output to v14; the difference is throughput and
portability.

Optimizations inherited from the decoder (see that module for detail):
  - register bit-buffer + 8-bit Huffman LUT (vs per-bit global reads),
  - all hot tables in __constant__ memory,
  - DC-only blocks skip the IDCT,
  - **fused YCbCr->luma->count** -> no (N,512,512,3) RGB buffer is materialised,
    which both cuts memory traffic and lets the batch fit small-VRAM cards,
  - **GPU auto-tuning**: the batch size is derived from free VRAM, so the same
    code adapts across e.g. RTX 5090 32 GB / RTX 4060 8 GB / GTX 1060 3 GB.

Base: data_loader_v0a_mono_baseline.py (single CPU read thread). Compare v16 vs
v14 for the optimization win; v16 vs v12 for tuned-CUDA vs nvJPEG.
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
from gpu_jpeg_decoder_optimized import GpuJpegDecoderOptimized


class WSISlidingWindowDataset(Dataset):
    """Mono-core feed + optimized auto-tuning custom CUDA decode (v0a base)."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: Optional[int] = None,   # None -> auto from VRAM
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.verbose = verbose
        self.read_time = 0.0
        self.decode_time = 0.0      # GPU decode + fused count
        self.kernel_time = 0.0      # filter is fused into decode_time
        self.peak_gpu_bytes = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v16 custom CUDA decoder needs JPEG tiles (compression 7).")
        self._decoder = self._build_decoder()
        self.batch_size = batch_size or self._decoder.recommended_batch

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            d = self._decoder.device
            print(f"[*] v16 on {d['name']} (cc{d['cc']}, {d['free']/1e9:.1f}GB free) "
                  f"auto batch={self.batch_size}")
            print(f"[*] v16 grid done in {self.grid_creation_time:.2f}s "
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
            raise ValueError("v16 requires tile-aligned geometry.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v16 (mono feed + optimized CUDA decode): {len(candidates)} "
                  f"patches over {len(self._tiff['offsets'])} tiles")

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
    print("==== v16 optimized CUDA decode (mono-core feed) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | read {ds.read_time:.3f}s cuda-decode+count {ds.decode_time:.3f}s")


if __name__ == '__main__':
    run_test()
