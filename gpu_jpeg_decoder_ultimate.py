"""
gpu_jpeg_decoder_ultimate.py

ULTIMATE custom-CUDA JPEG decoder: the optimized decoder plus the transfer /
overlap layers from CUPY_RESEARCH_PLAN.md (v4 pinned memory, v5 async streams,
v7 memory pool, v9 "stack every layer"). Bit-identical output to the optimized
and naive decoders -- the gains are pure throughput, from hiding host work and
transfers behind GPU compute.

Layers added on top of ``GpuJpegDecoderOptimized``:
  - **Pinned (page-locked) host staging** for the scan blob and the result
    read-back -> true async DMA instead of pageable copies.
  - **Double-buffered CUDA streams** -> the CPU destuff + H2D upload of batch k+1
    overlaps the GPU decode+count of batch k. Submit is non-blocking; results are
    collected later via ``fetch``.
  - **Pre-allocated, reused device + pinned buffers** (a fixed buffer pool, two
    "slots") -> no per-batch allocation churn.

Mixed precision (v6) is intentionally NOT used: the float IDCT must stay fp32 to
remain bit-identical to the CPU baseline's kept set; an fp16 IDCT would move
patches across the rejection threshold.

The pipeline is driven by the caller (v18/v19): submit(slot, tiles) enqueues GPU
work and returns immediately; fetch(slot) blocks for that slot's stream and
returns its per-tile white/black counts.
"""
from typing import List

import cupy as cp
import numpy as np

from gpu_jpeg_decoder import destuff_tile_scan
from gpu_jpeg_decoder_optimized import GpuJpegDecoderOptimized


def _pinned(nbytes: int, dtype=np.uint8) -> np.ndarray:
    """Page-locked host buffer of `nbytes` items of `dtype`."""
    itemsize = np.dtype(dtype).itemsize
    mem = cp.cuda.alloc_pinned_memory(nbytes * itemsize)
    return np.frombuffer(mem, dtype=dtype, count=nbytes)


class GpuJpegDecoderUltimate(GpuJpegDecoderOptimized):
    """Optimized decoder + pinned memory + double-buffered async stream pipeline."""

    def __init__(self, jpegtables: bytes, sample_tile: bytes,
                 max_batch: int, n_slots: int = 2, decode_threads: int = 32,
                 scan_bytes_per_tile: int = 96 * 1024):
        super().__init__(jpegtables, sample_tile, decode_threads)
        self.max_batch = int(max_batch)
        self.n_slots = max(2, n_slots)
        self._slots = [self._make_slot(self.max_batch, scan_bytes_per_tile)
                       for _ in range(self.n_slots)]

    # -- buffer pool -------------------------------------------------------
    def _make_slot(self, mb: int, scan_per_tile: int) -> dict:
        cap = mb * scan_per_tile
        return {
            "stream": cp.cuda.Stream(non_blocking=True),
            "scan_cap": cap,
            "h_scan": _pinned(cap, np.uint8),
            "d_scan": cp.empty(cap, dtype=cp.uint8),
            "d_bs": cp.empty(mb, dtype=cp.int64),
            "d_be": cp.empty(mb, dtype=cp.int64),
            "Yp": cp.empty((mb, 512, 512), dtype=cp.uint8),
            "Cbp": cp.empty((mb, 256, 256), dtype=cp.uint8),
            "Crp": cp.empty((mb, 256, 256), dtype=cp.uint8),
            "white": cp.empty(mb, dtype=cp.int32),
            "black": cp.empty(mb, dtype=cp.int32),
            "h_white": _pinned(mb, np.int32),
            "h_black": _pinned(mb, np.int32),
            "n": 0,
        }

    def _grow_scan(self, slot: dict, total: int) -> None:
        cap = int(total * 1.5)
        slot["scan_cap"] = cap
        slot["h_scan"] = _pinned(cap, np.uint8)
        slot["d_scan"] = cp.empty(cap, dtype=cp.uint8)

    # -- pipeline API ------------------------------------------------------
    def submit(self, si: int, tiles: List[bytes], white_thr: int, black_thr: int) -> None:
        """Enqueue decode+count for `tiles` on slot `si`'s stream (non-blocking).

        The CPU destuff + pinned staging here overlaps whatever GPU work is
        already in flight on the *other* slot(s).
        """
        slot = self._slots[si]
        st = slot["stream"]
        n = len(tiles)
        slot["n"] = n

        # CPU: destuff + offsets (overlaps in-flight GPU on other streams).
        scans = [destuff_tile_scan(t) for t in tiles]
        lengths = np.fromiter((len(s) for s in scans), dtype=np.int64, count=n)
        off = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=off[1:])
        total = int(off[-1])
        if total > slot["scan_cap"]:
            self._grow_scan(slot, total)
        slot["h_scan"][:total] = np.frombuffer(b"".join(scans), dtype=np.uint8)

        with st:
            # pinned -> device, async on this slot's stream. NOTE: the optimized
            # kernel's refill() indexes scan by BYTE (s[pos]), so these are byte
            # offsets -- not bit offsets (do not multiply by 8).
            slot["d_scan"][:total].set(slot["h_scan"][:total], stream=st)
            slot["d_bs"][:n].set(off[:n], stream=st)
            slot["d_be"][:n].set(off[1:], stream=st)

            thr = self.decode_threads
            self._k_decode(((n + thr - 1) // thr,), (thr,),
                           (slot["d_scan"], slot["d_bs"], slot["d_be"],
                            slot["Yp"], slot["Cbp"], slot["Crp"],
                            np.int32(n), np.int32(32), np.int32(32)))
            slot["white"][:n].fill(0)
            slot["black"][:n].fill(0)
            total_px = n * 512 * 512
            self._k_count(((total_px + 255) // 256,), (256,),
                          (slot["Yp"], slot["Cbp"], slot["Crp"],
                           slot["white"], slot["black"],
                           np.int32(n), np.int32(white_thr), np.int32(black_thr)))
            # device -> pinned host, async
            slot["white"][:n].get(stream=st, out=slot["h_white"][:n])
            slot["black"][:n].get(stream=st, out=slot["h_black"][:n])

    def fetch(self, si: int):
        """Block on slot `si`'s stream and return (white[n], black[n]) int64."""
        slot = self._slots[si]
        slot["stream"].synchronize()
        n = slot["n"]
        return (slot["h_white"][:n].astype(np.int64).copy(),
                slot["h_black"][:n].astype(np.int64).copy())
