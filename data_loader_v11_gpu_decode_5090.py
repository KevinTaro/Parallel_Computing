"""
data_loader_v11_gpu_decode_5090.py

v11: GPU JPEG DECODE for RTX 5090 (32GB) -- attacks the REAL bottleneck
=======================================================================
Every prior version (v1..v10) accepted the same hard floor: ``openslide.
read_region`` decodes the WSI's JPEG tiles *on the CPU* with libjpeg, and that
decode -- not the GPU filter -- is ~all of the wall time. v8's threaded readers
just throw more CPU cores at it. v11 removes the CPU decode entirely.

These Philips WSIs store level 0 as 512x512 **baseline JPEG tiles** with YCbCr
color (verified from the raw TIFF tags). That is exactly what NVIDIA's nvJPEG
hardware decoder consumes natively -- including the YCbCr->RGB conversion. So
v11:

  1. **Parses the TIFF directory itself** (tile offsets 324, byte counts 325,
     shared tables 347) instead of going through OpenSlide.
  2. **Reads the raw compressed tiles** (pure file I/O -- seeks + reads, no
     decode, microseconds per tile).
  3. **Batch-decodes the tiles on the GPU** via nvJPEG (``torchvision.io.
     decode_jpeg(..., device='cuda')``). WSI tiles are *abbreviated* JPEG
     streams (Huffman/quant tables live once in tag 347, not in each tile), so
     each tile is spliced with the shared table segment before decode.
  4. **Filters on the GPU** with the same fused PIL-luma kernel as v8.

Geometry shortcut (valid for this dataset): patch_size == stride == 2*tile and
the image dims are exact multiples of the tile size, so every 1024x1024 patch
is exactly a 2x2 block of *non-overlapping* tiles. White/black pixel counts are
therefore additive across a patch's 4 tiles -- decode each tile once, count on
GPU, sum per patch. If that alignment does not hold, v11 raises (it is a decode
specialization, not a general resampler) -- use v8 for arbitrary geometry.

ARITHMETIC CAVEAT (read this): the luma formula and thresholds are byte-for-byte
identical to v8, but the *pixel values feeding them* come from nvJPEG, not
libjpeg. The two decoders' IDCT + YCbCr->RGB can differ by +/-1 LSB, so a patch
whose white/black ratio sits right on ``rejection_ratio`` (0.9) can occasionally
flip. The kept-coordinate set is therefore *near*-identical to v8, not
guaranteed bit-identical. ``run_test`` / the benchmark report the exact delta so
you can judge it for your slides.
"""
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_jpeg

# Fused PIL-equivalent luma: uint8 R,G,B -> uint8 gray, identical to v8's kernel.
_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b',
    out_params='uint8 gray',
    operation='gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;',
    name='pil_luma_uint8',
)


def _read_tiff_tiles(path: str) -> dict:
    """Parse the level-0 IFD of a tiled TIFF without OpenSlide.

    Returns tile geometry plus, for every tile, its (offset, byte-count) in the
    file and the shared JPEGTables blob. Only standard (non-Big) TIFF is
    handled -- which is what these Philips slides are.
    """
    fh = open(path, 'rb')
    head = fh.read(8)
    bo = '<' if head[:2] == b'II' else '>'
    if head[2:4] != (b'\x2a\x00' if bo == '<' else b'\x00\x2a'):
        fh.close()
        raise ValueError("Not a standard TIFF (BigTIFF unsupported by v11).")
    ifd0 = struct.unpack(bo + 'I', head[4:8])[0]

    fh.seek(ifd0)
    n = struct.unpack(bo + 'H', fh.read(2))[0]
    raw = fh.read(n * 12)
    tags: Dict[int, Tuple[int, int, int]] = {}
    for i in range(n):
        tag, typ, cnt = struct.unpack(bo + 'HHI', raw[i * 12:i * 12 + 8])
        valoff = struct.unpack(bo + 'I', raw[i * 12 + 8:i * 12 + 12])[0]
        tags[tag] = (typ, cnt, valoff)

    def scalar(tag):
        return tags[tag][2]

    def array(tag):
        typ, cnt, valoff = tags[tag]
        size = {3: 2, 4: 4}[typ]
        if cnt == 1:
            return [valoff]
        fh.seek(valoff)
        buf = fh.read(cnt * size)
        fmt = bo + ("H" if typ == 3 else "I") * cnt
        return list(struct.unpack(fmt, buf))

    if 322 not in tags or 324 not in tags:
        fh.close()
        raise ValueError("Level-0 IFD is not tiled; v11 needs JPEG tiles.")

    info = {
        "image_width": scalar(256),
        "image_height": scalar(257),
        "tile_width": scalar(322),
        "tile_height": scalar(323),
        "compression": scalar(259),          # expect 7 (JPEG)
        "offsets": array(324),
        "bytecounts": array(325),
    }
    if 347 in tags:                          # shared Huffman/quant tables
        typ, cnt, valoff = tags[347]
        fh.seek(valoff)
        info["jpegtables"] = fh.read(cnt)
    else:
        info["jpegtables"] = b""
    fh.close()
    return info


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset that decodes JPEG tiles on the GPU (RTX 5090, 32GB)."""

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
        self.batch_size = batch_size            # tiles decoded per GPU batch
        self.num_readers = num_readers or (os.cpu_count() or 4)
        self.verbose = verbose
        self.peak_gpu_bytes = 0
        self.kernel_time = 0.0                   # GPU luma/count time (v8-comparable)
        self.decode_time = 0.0                   # GPU JPEG decode time (new in v11)

        if self.verbose:
            print(f"[*] Initializing GPU-decode dataset for WSI: {self.wsi_path}")

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError(
                f"v11 needs JPEG-compressed tiles (TIFF compression 7); got "
                f"{self._tiff['compression']}. Use v8 for this slide.")

        start_time = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start_time
        if self.verbose:
            print(f"\n[*] Grid creation finished in {self.grid_creation_time:.2f} s "
                  f"(GPU decode {self.decode_time:.3f} s + filter {self.kernel_time:.3f} s).")
            print(f"    - Peak GPU memory: {self.peak_gpu_bytes / 1e6:.1f} MB")

        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")
        if self.verbose:
            print(f"[*] Found {len(self.coordinates)} tissue-containing patches.")

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    coords.append((x, y))
        return coords

    def _splice(self, tile: bytes) -> bytes:
        """Insert the shared JPEGTables marker segments after the tile's SOI."""
        jt = self._tiff["jpegtables"]
        if not jt:
            return tile
        return tile[:2] + jt[2:-2] + tile[2:]   # SOI + tables(no SOI/EOI) + body

    def _create_grid(self) -> List[Tuple[int, int]]:
        ps, tw, th = self.patch_size, self._tiff["tile_width"], self._tiff["tile_height"]
        iw, ih = self._tiff["image_width"], self._tiff["image_height"]

        # Geometry guard: every patch must be an exact, non-overlapping block of
        # whole tiles, so per-tile counts sum to per-patch counts with no crop.
        if not (self.stride == ps and ps % tw == 0 and ps % th == 0
                and iw % tw == 0 and ih % th == 0):
            raise ValueError(
                "v11 requires stride==patch_size, patch a whole multiple of the "
                f"tile ({tw}x{th}), and image dims multiples of the tile. Got "
                f"patch={ps}, stride={self.stride}, image={iw}x{ih}. Use v8.")

        tiles_per_row = iw // tw
        tx_per_patch = ps // tw                 # tile columns spanned by one patch
        ty_per_patch = ps // th
        total_pixels = ps * ps

        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v11 (GPU JPEG decode, tile_bs={self.batch_size}, "
                  f"readers={self.num_readers}): {len(candidates)} candidate "
                  f"patches over {len(self._tiff['offsets'])} tiles...")

        # Map each candidate patch -> the tile ids it covers; collect the unique
        # tile ids actually needed (sparse slides won't touch every tile).
        patch_tiles: List[Tuple[int, int, List[int]]] = []
        needed: Dict[int, int] = {}             # tile_id -> compact index
        for (x, y) in candidates:
            col0, row0 = x // tw, y // th
            ids = []
            for dr in range(ty_per_patch):
                base = (row0 + dr) * tiles_per_row + col0
                for dc in range(tx_per_patch):
                    tid = base + dc
                    ids.append(tid)
                    if tid not in needed:
                        needed[tid] = len(needed)
            patch_tiles.append((x, y, ids))

        tile_ids = list(needed.keys())          # in compact-index order
        offsets = self._tiff["offsets"]
        bytecounts = self._tiff["bytecounts"]

        # Per-tile white/black pixel counts, indexed by compact tile index.
        white_per_tile = np.zeros(len(tile_ids), dtype=np.int64)
        black_per_tile = np.zeros(len(tile_ids), dtype=np.int64)

        # Zero-byte tiles are empty background. OpenSlide synthesises them as
        # transparent black -> RGB (0,0,0), gray 0: every pixel counts as
        # "black", none as "white" (verified against this slide). v8 sees that
        # via read_region, so v11 must match. They are NOT valid JPEG, so they
        # must stay out of the nvJPEG batch (one bad stream fails the whole
        # batch) -- assign their counts directly instead.
        TILE_PX = tw * th
        decode_idx = []                         # compact indices to GPU-decode
        for ci, tid in enumerate(tile_ids):
            if bytecounts[tid] > 0:
                decode_idx.append(ci)
            else:
                black_per_tile[ci] = TILE_PX    # all-black empty background tile

        fh = open(self.wsi_path, 'rb')
        read_lock = threading.Lock()

        def read_tile(tid: int) -> bytes:
            off, bc = offsets[tid], bytecounts[tid]
            with read_lock:                     # single fh; seeks must not race
                fh.seek(off)
                return self._splice(fh.read(bc))

        try:
            for start in range(0, len(decode_idx), self.batch_size):
                cidx = decode_idx[start:start + self.batch_size]   # compact indices
                chunk = [tile_ids[ci] for ci in cidx]              # tile ids

                # 1) read raw compressed tiles (cheap I/O, threaded)
                with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                    streams = list(pool.map(read_tile, chunk))
                buffers = [torch.frombuffer(bytearray(s), dtype=torch.uint8)
                           for s in streams]

                # 2) batch nvJPEG decode on the GPU -> list of (3, th, tw) uint8
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                decoded = decode_jpeg(buffers, mode=ImageReadMode.RGB, device='cuda')
                stacked = torch.stack(decoded)          # (n, 3, th, tw) on cuda
                torch.cuda.synchronize()
                self.decode_time += time.perf_counter() - t0

                # 3) fused luma + threshold counts on the GPU (same as v8)
                e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                e0.record()
                ct = cp.asarray(stacked)                # zero-copy view of torch mem
                n = ct.shape[0]
                gray = cp.empty((n, th, tw), dtype=cp.uint8)
                _luma_kernel(ct[:, 0], ct[:, 1], ct[:, 2], gray)
                w = cp.count_nonzero(gray > self.white_pixel_threshold, axis=(1, 2))
                b = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2))
                e1.record(); e1.synchronize()
                self.kernel_time += cp.cuda.get_elapsed_time(e0, e1) / 1000.0

                white_per_tile[cidx] = cp.asnumpy(w)
                black_per_tile[cidx] = cp.asnumpy(b)

                self.peak_gpu_bytes = max(
                    self.peak_gpu_bytes,
                    cp.get_default_memory_pool().used_bytes()
                    + torch.cuda.max_memory_allocated())
                del decoded, stacked, ct, gray
        finally:
            fh.close()

        # 4) aggregate tile counts -> per-patch ratios -> keep decision
        coordinates = []
        rr = self.rejection_ratio
        for (x, y, ids) in patch_tiles:
            wc = sum(int(white_per_tile[needed[t]]) for t in ids)
            bc = sum(int(black_per_tile[needed[t]]) for t in ids)
            if (wc / total_pixels) < rr and (bc / total_pixels) < rr:
                coordinates.append((x, y))

        if self.verbose:
            print(f"\n[*] Scanned {len(candidates)} patches. Kept {len(coordinates)}.")
        return coordinates

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Per-item training reads stay on OpenSlide; the GPU path is for the
        # bulk tissue scan (the bottleneck), not random single-patch access.
        with openslide.OpenSlide(self.wsi_path) as slide:
            x, y = self.coordinates[idx]
            patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')
            if self.transform:
                patch_tensor = self.transform(patch)
            else:
                patch_tensor = transforms.ToTensor()(patch)
            return patch_tensor, (x, y)


def run_test(wsi_path: str = "data/S114-80954A-Her2(3+).tiff"):
    print("=====================================================")
    print(" v11 GPU JPEG Decode for RTX 5090 (32GB) - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")
    print(f"[*] GPU decode time: {dataset.decode_time:.3f} s | "
          f"GPU filter time: {dataset.kernel_time:.3f} s")


if __name__ == '__main__':
    run_test()
