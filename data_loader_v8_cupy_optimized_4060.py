"""
data_loader_v8_cupy_optimized_4060.py

v8: OPTIMIZED FOR RTX 4060 (8GB)
================================
The honest profile of this workload: GPU filter compute is <1s; nearly all
wall time is CPU-side TIFF decode in ``slide.read_region``. So v8 attacks the
*feed* side as well as the GPU side:

  1. **Threaded patch decode**: a pool of reader threads (one OpenSlide handle
     each -- libopenslide releases the GIL) fills the pinned staging buffer in
     parallel. This is where the real speedup over v1/v2 comes from.
  2. **Large batches**: batch_size=512 (~2 GiB RGBA on device) -- the 8 GB
     card allows it, and bigger batches amortise transfer + launch overhead.
  3. **Pinned host memory**: page-locked staging buffer, DMA transfer.
  4. **Fused uint8 luma kernel**: no 4x uint32 grayscale temporary.

Peak VRAM at bs=512: ~2 GiB device input + ~0.5 GiB grayscale = ~2.6 GiB.
Note ``peak_gpu_bytes`` reports the CuPy *device pool* (VRAM); the pinned
staging buffer (~2 GiB at bs=512) lives in host RAM and is not counted.

Arithmetic is identical to v0a/v2/v7 (same PIL luma formula).
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_LUMA = (19595, 38470, 7471)
_LUMA_ROUND = 32768

# Fused PIL-equivalent luma: uint8 R,G,B -> uint8 gray, no uint32 temporary.
_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b',
    out_params='uint8 gray',
    operation='gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;',
    name='pil_luma_uint8',
)


def _alloc_pinned(shape, dtype=np.uint8) -> np.ndarray:
    """Allocate page-locked host memory for DMA."""
    mem = cp.cuda.alloc_pinned_memory(np.prod(shape) * np.dtype(dtype).itemsize)
    return np.frombuffer(mem, dtype=dtype).reshape(shape)


class WSISlidingWindowDataset(Dataset):
    """WSI patch dataset optimized for RTX 4060 (8GB VRAM)."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 512,
                 num_readers: Optional[int] = None,
                 num_stage_buffers: int = 2,
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
        self.num_stage_buffers = max(2, num_stage_buffers)
        self.verbose = verbose
        self.peak_gpu_bytes = 0
        self.kernel_time = 0.0

        if self.verbose:
            print(f"[*] Initializing dataset for WSI: {self.wsi_path}")

        try:
            with openslide.OpenSlide(self.wsi_path) as slide:
                self.wsi_width, self.wsi_height = slide.level_dimensions[0]
                if self.verbose:
                    print(f"    - WSI dimensions (level 0): {self.wsi_width}x{self.wsi_height}")
        except openslide.OpenSlideError:
            raise OSError(f"Could not open WSI file: {self.wsi_path}")

        start_time = time.time()
        self.coordinates = self._create_grid()
        self.grid_creation_time = time.time() - start_time
        if self.verbose:
            print(f"\n[*] Grid creation finished in {self.grid_creation_time:.2f} seconds.")
            print(f"    - Peak GPU memory: {self.peak_gpu_bytes / 1e6:.1f} MB")

        if not self.coordinates:
            raise ValueError("No valid tissue regions found in the WSI.")

        if self.verbose:
            print(f"[*] Found {len(self.coordinates)} tissue-containing patches.")

    def _generate_candidate_coords(self) -> List[Tuple[int, int]]:
        potential_coords = []
        for y in range(0, self.wsi_height, self.stride):
            for x in range(0, self.wsi_width, self.stride):
                if x + self.patch_size <= self.wsi_width and y + self.patch_size <= self.wsi_height:
                    potential_coords.append((x, y))
        return potential_coords

    def _filter_batch_gpu(self, batch_rgba_gpu: cp.ndarray) -> np.ndarray:
        """Filter a batch already on GPU. Input is (N, H, W, 4) RGBA cp.ndarray."""
        n = batch_rgba_gpu.shape[0]
        total_pixels = self.patch_size * self.patch_size
        e_start = cp.cuda.Event()
        e_end = cp.cuda.Event()
        e_start.record()

        # Fused kernel directly on the GPU buffer — zero extra host→device transfer.
        gray = cp.empty((n, self.patch_size, self.patch_size), dtype=cp.uint8)
        _luma_kernel(batch_rgba_gpu[..., 0], batch_rgba_gpu[..., 1], batch_rgba_gpu[..., 2], gray)

        white_counts = cp.count_nonzero(gray > self.white_pixel_threshold, axis=(1, 2))
        black_counts = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2))
        white_ratio = white_counts.astype(cp.float64) / total_pixels
        black_ratio = black_counts.astype(cp.float64) / total_pixels

        keep = (white_ratio < self.rejection_ratio) & (black_ratio < self.rejection_ratio)
        # Sample peak while gray + temporaries are still alive (honest peak).
        self.peak_gpu_bytes = max(self.peak_gpu_bytes,
                                  cp.get_default_memory_pool().used_bytes())
        result = cp.asnumpy(keep)

        e_end.record()
        e_end.synchronize()
        self.kernel_time += cp.cuda.get_elapsed_time(e_start, e_end) / 1000.0

        return result

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v8 (4060-optimized, bs={self.batch_size}, "
                  f"readers={self.num_readers}): scanning "
                  f"{len(potential_coords)} candidates...")

        ps = self.patch_size
        bs = self.batch_size
        n_buf = self.num_stage_buffers
        mempool = cp.get_default_memory_pool()

        # CONTINUOUS DECODE (producer/consumer, no per-batch read barrier).
        # Decode is the bottleneck (~6.7s floor). The previous double-buffer
        # version still waited for *every* read of batch k before queuing batch
        # k+1, so at each batch boundary the early-finishing readers idled while
        # the slowest straggler completed (~1s wasted over 10 batches).
        #
        # Here a producer thread keeps the pool's FIFO queue stocked with up to
        # ``n_buf`` batches of read tasks (bounded by a semaphore = host-RAM
        # backpressure). A reader that finishes batch k's last straggler
        # immediately picks up batch k+1's already-queued tasks, so the 24
        # readers stay saturated until the final patch. The consumer (main
        # thread) drains completed batches in order: one host->device copy, GPU
        # filter, then releases the buffer back to the producer. Only ONE device
        # buffer is reused serially, so VRAM is unchanged.
        #
        # n_buf=2 is optimal here: each pinned buffer is ~2 GiB and page-locked
        # allocation is counted in grid_creation_time, so n_buf=3/4 measured
        # *slower* (8.1/8.3s vs 7.8s) -- the extra alloc cost outweighs the tiny
        # extra runway. 2 buffers already keep the readers saturated because the
        # consumer (transfer+GPU, ~0.2s/batch) is far faster than a batch decode.
        pinned_hosts = [_alloc_pinned((bs, ps, ps, 4), np.uint8) for _ in range(n_buf)]
        device_buf = cp.empty((bs, ps, ps, 4), dtype=cp.uint8)
        xfer_stream = cp.cuda.Stream(non_blocking=True)

        # Reader threads each own an OpenSlide handle (handles are cheap;
        # read_region releases the GIL inside libopenslide, so decode runs
        # truly in parallel). This is the actual bottleneck of the workload.
        tls = threading.local()
        handles: List[openslide.OpenSlide] = []
        handles_lock = threading.Lock()

        def read_into(buf: np.ndarray, slot: int, x: int, y: int) -> bool:
            slide = getattr(tls, "slide", None)
            if slide is None:
                slide = openslide.OpenSlide(self.wsi_path)
                tls.slide = slide
                with handles_lock:
                    handles.append(slide)
            try:
                buf[slot] = np.asarray(slide.read_region((x, y), 0, (ps, ps)))
                return True
            except Exception as e:
                if self.verbose:
                    print(f"    - patch at ({x},{y}): discarded (error: {e})")
                return False

        batches = [potential_coords[s:s + bs]
                   for s in range(0, len(potential_coords), bs)]
        coordinates = []

        free_buffers = threading.Semaphore(n_buf)   # host-RAM backpressure
        ready_q: "Queue" = Queue()                   # producer -> consumer handoff

        def producer(pool: ThreadPoolExecutor):
            for bi, chunk in enumerate(batches):
                free_buffers.acquire()               # wait for a free staging buffer
                buf = pinned_hosts[bi % n_buf]
                futs = [pool.submit(read_into, buf, i, x, y)
                        for i, (x, y) in enumerate(chunk)]
                ready_q.put((bi, chunk, buf, futs))
            ready_q.put(None)                        # sentinel: no more batches

        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                prod = threading.Thread(target=producer, args=(pool,), daemon=True)
                prod.start()

                while True:
                    item = ready_q.get()
                    if item is None:
                        break
                    bi, chunk, buf, futures = item
                    ok = [f.result() for f in futures]   # wait this batch's decode
                    if any(ok):
                        n = len(chunk)
                        with xfer_stream:
                            device_buf[:n].set(buf[:n], stream=xfer_stream)
                        xfer_stream.synchronize()
                        # Failed slots hold stale bytes; their mask entry is ignored.
                        keep_mask = self._filter_batch_gpu(device_buf[:n])
                        for i, (x, y) in enumerate(chunk):
                            if ok[i] and keep_mask[i]:
                                coordinates.append((x, y))
                    free_buffers.release()               # buffer reusable by producer

                prod.join()
        finally:
            for h in handles:
                h.close()

        del pinned_hosts, device_buf, xfer_stream
        mempool.free_all_blocks()

        if self.verbose:
            print(f"\n[*] Scanned {len(potential_coords)} patches. Kept {len(coordinates)}.")
        return coordinates

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        with openslide.OpenSlide(self.wsi_path) as slide:
            x, y = self.coordinates[idx]
            patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')
            if self.transform:
                patch_tensor = self.transform(patch)
            else:
                patch_tensor = transforms.ToTensor()(patch)
            return patch_tensor, (x, y)


def run_test(wsi_path: str = "data/S114-82742C-Her2(4B5) 20x.tiff"):
    print("=====================================================")
    print(" v8 CuPy Optimized for RTX 4060 (8GB) - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")
    print(f"[*] Pure GPU kernel time: {dataset.kernel_time:.3f} s "
          f"(everything else is CPU decode + transfer)")


if __name__ == '__main__':
    run_test()
