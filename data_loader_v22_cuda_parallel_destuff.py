"""
data_loader_v22_cuda_parallel_destuff.py

v22: parallel READ **and** DESTUFF -- take the last serial host work off the
     main thread so the pipeline is bound only by the GPU decode
=============================================================================
v21 proved the architecture point (reads run parallel, off the main thread,
prefetched ahead). But it still did not beat v18 by much on warm storage, and
the reason is in the decoder's ``submit``:

    scans = [destuff_tile_scan(t) for t in tiles]   # <-- serial Python, main thread
    ...
    h_scan[:total] = np.frombuffer(b"".join(scans), ...)

``destuff_tile_scan`` does a full ``bytes.replace(b'\\xff\\x00', b'\\xff')`` over
each tile's ~96 KB scan blob. For a batch of 2048 tiles that is a pure-Python
pass over ~190 MB, on the **main thread**, only partially hidden behind the
in-flight GPU. Once reads are parallel (v20/v21), THIS becomes the dominant
host-side serial cost on the critical path -- which is why v21 only matched v18.

v22 moves it where it belongs. Each reader thread, right after it ``os.pread``s
a tile, also **destuffs it** -- so the read and the destuff are both done in
parallel across ``num_readers`` threads, ahead of the GPU. The main thread's
``submit_scans`` then only joins the (already destuffed) scans into the pinned
buffer and launches the async H2D + decode + count. The per-tile Python work is
gone from the critical path entirely; the loop is now bound by the GPU decode.

Mechanism
---------
  * reader task  : ``tid -> destuff_tile_scan(os.pread(fd, ...))``  (parallel)
  * producer     : keeps the reader pool's FIFO stocked across batch boundaries
  * ready_q      : hands completed (destuffed-scan) batches to the consumer
  * consumer     : ``decoder.submit_scans`` (no destuff) + ``decoder.fetch`` only

``_PipelineUltimateDecoder`` subclasses the shared ``GpuJpegDecoderUltimate`` and
adds ``submit_scans`` -- byte-for-byte the same GPU work as ``submit`` minus the
internal destuff loop. The shared decoder (used by v18/v19/v20/v21) is untouched,
so all of them keep their exact behaviour. Output here stays **bit-identical** to
the v0a baseline.

Honest ceiling
--------------
On warm local storage the workload is GPU-compute-bound: once read AND destuff
are off the critical path, the wall clock equals the serial GPU decode time, and
v22 should *clearly* beat v18 (which paid both serial read and serial destuff) --
but it cannot go below the single-GPU decode floor. To break THAT floor you need
data-parallelism across the decode itself: shard the tile batches across multiple
GPUs (tiles are independent -> near-linear), or a faster decode kernel (which
trades away bit-exactness). The multi-GPU path is the next version; this one
extracts everything a single GPU + parallel host feed can give.
"""
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Semaphore
from typing import Callable, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader_v11_gpu_decode_5090 import _read_tiff_tiles
from gpu_jpeg_decoder import destuff_tile_scan
from gpu_jpeg_decoder_ultimate import GpuJpegDecoderUltimate


class _PipelineUltimateDecoder(GpuJpegDecoderUltimate):
    """GpuJpegDecoderUltimate + a submit that takes ALREADY-DESTUFFED scans.

    Identical GPU work to the parent's ``submit`` (same staging, same kernels,
    same async read-back) -- the ONLY difference is that the per-tile
    ``destuff_tile_scan`` loop has been hoisted out to the reader threads, so it
    is not repeated here on the main thread. The parent class is left unchanged.
    """

    def submit_scans(self, si: int, scans: List[bytes],
                     white_thr: int, black_thr: int) -> None:
        slot = self._slots[si]
        st = slot["stream"]
        n = len(scans)
        slot["n"] = n

        # scans are already destuffed (done in parallel by the reader pool).
        lengths = np.fromiter((len(s) for s in scans), dtype=np.int64, count=n)
        off = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=off[1:])
        total = int(off[-1])
        if total > slot["scan_cap"]:
            self._grow_scan(slot, total)
        slot["h_scan"][:total] = np.frombuffer(b"".join(scans), dtype=np.uint8)

        with st:
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
            slot["white"][:n].get(stream=st, out=slot["h_white"][:n])
            slot["black"][:n].get(stream=st, out=slot["h_black"][:n])


class WSISlidingWindowDataset(Dataset):
    """Parallel read+destuff prefetch feed + ultimate pipelined custom CUDA decode."""

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
                 prefetch_batches: int = 3,
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
        self.prefetch_batches = max(n_slots + 1, prefetch_batches)
        self.verbose = verbose
        self.read_time = 0.0       # main-thread wait on read+destuff futures
        self.decode_time = 0.0     # main-thread wait on the GPU
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v22 custom CUDA decoder needs JPEG tiles (compression 7).")
        if not hasattr(os, "pread"):
            raise OSError("v22 needs os.pread (POSIX). On Windows use per-thread "
                          "file handles instead (see module docstring).")
        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            d = self._decoder.device
            print(f"[*] v22 on {d['name']} (cc{d['cc']}) batch={self.batch_size} "
                  f"slots={self.n_slots} readers={self.num_readers} "
                  f"prefetch={self.prefetch_batches} (parallel read+destuff)")
            print(f"[*] v22 grid done in {self.grid_creation_time:.2f}s "
                  f"(read+destuff-wait {self.read_time:.3f} + gpu-wait "
                  f"{self.decode_time:.3f}); kept {len(self.coordinates)}")
        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

    def _build_decoder(self) -> _PipelineUltimateDecoder:
        offs, bcs = self._tiff["offsets"], self._tiff["bytecounts"]
        with open(self.wsi_path, 'rb') as fh:
            tid = next(i for i, b in enumerate(bcs) if b > 0)
            fh.seek(offs[tid]); sample = fh.read(bcs[tid])
        return _PipelineUltimateDecoder(self._tiff["jpegtables"], sample,
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
            raise ValueError("v22 requires tile-aligned geometry.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v22 (parallel read+destuff pipeline + ultimate CUDA): "
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

        # ===================================================================
        # Parallel read+destuff prefetch pipeline.
        #   reader task : tid -> destuff_tile_scan(os.pread(fd, ...))
        #                 (BOTH the read and the destuff run in the pool, in
        #                  parallel, ahead of the GPU)
        #   producer    : pool.submit keeps the FIFO stocked across batches
        #   consumer    : only decoder.submit_scans / decoder.fetch (GPU work)
        # ===================================================================
        fd = os.open(self.wsi_path, os.O_RDONLY)

        def read_destuff(tid: int) -> bytes:
            return destuff_tile_scan(os.pread(fd, bytecounts[tid], offsets[tid]))

        free = Semaphore(self.prefetch_batches)
        ready_q: "Queue" = Queue()

        def producer(pool: ThreadPoolExecutor):
            for i, cidx in enumerate(chunks):
                free.acquire()
                futs = [pool.submit(read_destuff, tile_ids[ci]) for ci in cidx]
                ready_q.put((i, cidx, futs))
            ready_q.put(None)

        pending: "deque" = deque()
        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                prod = threading.Thread(target=producer, args=(pool,), daemon=True)
                prod.start()

                while True:
                    item = ready_q.get()
                    if item is None:
                        break
                    i, cidx, futs = item

                    t0 = time.perf_counter()
                    scans = [f.result() for f in futs]   # already destuffed
                    self.read_time += time.perf_counter() - t0

                    slot = i % self.n_slots
                    if len(pending) == self.n_slots:
                        ps_slot, pcidx = pending.popleft()
                        t0 = time.perf_counter()
                        w, b = self._decoder.fetch(ps_slot)
                        self.decode_time += time.perf_counter() - t0
                        white_per_tile[pcidx] = w
                        black_per_tile[pcidx] = b

                    self._decoder.submit_scans(slot, scans, wt, bt)
                    pending.append((slot, cidx))
                    del scans
                    free.release()

                while pending:
                    ps_slot, pcidx = pending.popleft()
                    t0 = time.perf_counter()
                    w, b = self._decoder.fetch(ps_slot)
                    self.decode_time += time.perf_counter() - t0
                    white_per_tile[pcidx] = w
                    black_per_tile[pcidx] = b

                prod.join()
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
    print("==== v22 ultimate CUDA pipeline (parallel read+destuff) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | read+destuff-wait {ds.read_time:.3f}s "
          f"gpu-wait {ds.decode_time:.3f}s")


if __name__ == '__main__':
    run_test()
