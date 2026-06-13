"""
data_loader_v21_cuda_ultimate_pipeline.py

v21: ULTIMATE custom CUDA decode with a TRUE prefetch pipeline (producer/consumer)
==================================================================================
This is the version that finally reads in parallel *and off the critical path*.

The story so far:
  - v18 (mono): one thread, serial reads. Read overlaps the in-flight GPU decode,
    so on warm local storage gpu-wait -> 0 and it is already good.
  - v19 ("multi"): a thread pool BUT all threads shared one file object + a lock,
    so reads were actually serial -- pool overhead on top of v18. Slower.
  - v20 (pread): real parallel reads via ``os.pread`` (no lock), but still called
    ``pool.map`` **on the main thread, per batch**. The thread-dispatch + result
    collection sits on the GPU submit/fetch critical path, so between batches the
    GPU streams briefly starve. On warm storage (where the read was never the
    bottleneck) that overhead made it *slower than mono v18*.

The real problem in v19/v20 was never just "is the read parallel" -- it was
**where the read runs**. As long as the read happens inline in the main loop, it
competes with GPU orchestration. v21 fixes the architecture:

  * A dedicated **producer thread** drives a ``ThreadPoolExecutor`` of readers
    that issue ``os.pread`` calls (atomic positional reads on one shared fd, no
    lock, GIL released per syscall -> genuinely parallel).
  * The producer keeps the pool's FIFO **stocked across batch boundaries** using
    ``pool.submit`` (not ``pool.map``), so a reader finishing batch k's last tile
    immediately starts batch k+1's tiles -- the readers never drain at a boundary.
  * A bounded ``ready_q`` (a ``Semaphore`` = prefetch-depth backpressure) hands
    completed raw-tile batches to the **main thread**, which does ONLY the GPU
    work: ``decoder.submit`` (CPU destuff + async H2D + decode+count launch) and
    ``decoder.fetch``. No ``pool.map`` ever runs on the main thread.

Net effect: parallel reads run continuously, fully ahead of and overlapped with
the GPU decode, and the main GPU loop is never blocked waiting on a per-batch
read dispatch. This is the producer/consumer design v8 used for the CPU-decode
feed, now applied to the custom-CUDA decode pipeline.

Output is **bit-identical** to v14/v16/v18/v19/v20 and the v0a baseline -- only
the read scheduling changes; the bytes read and decoded are the same.

Honest expectation
------------------
On a WARM-CACHE local SSD this workload is GPU-compute-bound: the GPU decode is
~all the wall time and the read was already hidden. So v21 should *match* v18
(it removes v20's overhead, it does not invent new GPU speed) -- the goal here is
to stop being slower, and to actually pull ahead when the read is slow and NOT
hidden: cold cache, network/NFS, or a slow HDD, where a serial (v18) or
lock-serialised (v19) read would stall the GPU and only a continuous parallel
prefetch keeps the decode pipeline fed. ``read_time`` / ``decode_time`` are
reported so you can see which regime your storage is in.

NOTE: ``os.pread`` is POSIX (Linux/macOS). On Windows give each reader its own
file handle (per-thread position, no lock) -- same idea, more handles.
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
from gpu_jpeg_decoder_ultimate import GpuJpegDecoderUltimate


class WSISlidingWindowDataset(Dataset):
    """Continuous parallel-prefetch feed (os.pread producer/consumer) + ultimate
    pipelined custom CUDA decode."""

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
        # How many batches of raw tiles may sit read-ahead in RAM (backpressure).
        # >= n_slots + 1 so a batch is always ready when a GPU slot frees up.
        self.prefetch_batches = max(n_slots + 1, prefetch_batches)
        self.verbose = verbose
        self.read_time = 0.0       # wall time the main thread waits on reads
        self.decode_time = 0.0     # wall time the main thread waits on the GPU
        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        self._tiff = _read_tiff_tiles(self.wsi_path)
        if self._tiff["compression"] != 7:
            raise ValueError("v21 custom CUDA decoder needs JPEG tiles (compression 7).")
        if not hasattr(os, "pread"):
            raise OSError("v21 needs os.pread (POSIX). On Windows use per-thread "
                          "file handles instead (see module docstring).")
        self._decoder = self._build_decoder()

        start = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start
        if self.verbose:
            d = self._decoder.device
            print(f"[*] v21 on {d['name']} (cc{d['cc']}) batch={self.batch_size} "
                  f"slots={self.n_slots} readers={self.num_readers} "
                  f"prefetch={self.prefetch_batches} (os.pread pipeline)")
            print(f"[*] v21 grid done in {self.grid_creation_time:.2f}s "
                  f"(read-wait {self.read_time:.3f} + gpu-wait {self.decode_time:.3f}); "
                  f"kept {len(self.coordinates)}")
        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

    def _build_decoder(self) -> GpuJpegDecoderUltimate:
        offs, bcs = self._tiff["offsets"], self._tiff["bytecounts"]
        with open(self.wsi_path, 'rb') as fh:
            tid = next(i for i, b in enumerate(bcs) if b > 0)
            fh.seek(offs[tid]); sample = fh.read(bcs[tid])
        return GpuJpegDecoderUltimate(self._tiff["jpegtables"], sample,
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
            raise ValueError("v21 requires tile-aligned geometry.")

        tiles_per_row = iw // tw
        txp, typ = ps // tw, ps // th
        total_pixels = ps * ps
        candidates = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v21 (os.pread prefetch pipeline + ultimate CUDA): "
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
        # TRUE prefetch pipeline.
        #
        #   producer thread  -> ThreadPoolExecutor of readers issuing os.pread
        #                       (atomic, lock-free, parallel) for ALL tiles of
        #                       upcoming batches; FIFO stays stocked across batch
        #                       boundaries so readers never drain mid-scan.
        #   ready_q          -> hands (chunk_idx, cidx, futures) to the consumer.
        #   Semaphore(free)  -> bounds batches read-ahead in RAM (backpressure).
        #   consumer (here)  -> ONLY GPU work: decoder.submit / decoder.fetch.
        #                       No pool.map ever runs on this thread.
        # ===================================================================
        fd = os.open(self.wsi_path, os.O_RDONLY)

        def read_tile(tid: int) -> bytes:
            return os.pread(fd, bytecounts[tid], offsets[tid])

        free = Semaphore(self.prefetch_batches)     # read-ahead RAM backpressure
        ready_q: "Queue" = Queue()                  # producer -> consumer handoff

        def producer(pool: ThreadPoolExecutor):
            for i, cidx in enumerate(chunks):
                free.acquire()                      # wait for a read-ahead slot
                # submit (not map): the pool keeps draining these in parallel and
                # rolls straight into the next batch's tiles with no boundary stall.
                futs = [pool.submit(read_tile, tile_ids[ci]) for ci in cidx]
                ready_q.put((i, cidx, futs))
            ready_q.put(None)                       # sentinel: no more batches

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

                    # Wait only for THIS batch's reads (already running ahead).
                    t0 = time.perf_counter()
                    tiles = [f.result() for f in futs]
                    self.read_time += time.perf_counter() - t0

                    slot = i % self.n_slots
                    if len(pending) == self.n_slots:
                        ps_slot, pcidx = pending.popleft()
                        t0 = time.perf_counter()
                        w, b = self._decoder.fetch(ps_slot)
                        self.decode_time += time.perf_counter() - t0
                        white_per_tile[pcidx] = w
                        black_per_tile[pcidx] = b

                    # submit copies tiles into the slot's pinned buffer + launches
                    # async; the raw `tiles` bytes are no longer needed afterwards.
                    self._decoder.submit(slot, tiles, wt, bt)
                    pending.append((slot, cidx))
                    del tiles
                    free.release()                  # let the producer prefetch more

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
    print("==== v21 ultimate CUDA pipeline (true prefetch producer/consumer) - Test Run ====")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[*] kept {len(ds)} | read-wait {ds.read_time:.3f}s gpu-wait {ds.decode_time:.3f}s")


if __name__ == '__main__':
    run_test()
