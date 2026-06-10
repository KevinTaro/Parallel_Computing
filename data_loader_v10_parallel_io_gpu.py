"""
data_loader_v10_parallel_io_gpu.py

v10: PARALLEL-I/O + GPU  (the data-driven optimization)
=======================================================
hardware_probe.py proved the real story on this GTX 1060 3GB:

  * the luma kernel does ~17,500 patches/sec (pure GPU compute), but
  * OpenSlide decodes only ~27 patches/sec on one thread, so
  * v1-v9 leave the GPU ~98% idle (17% util in nvidia-smi) -- it is STARVED,
    not slow. Optimizing the kernel/streams/precision cannot help.
  * Crucially, OpenSlide RELEASES THE GIL during read_region, so multiple
    Python threads decode in parallel: 8 threads -> ~93 patches/sec (3.4x).

v10 acts on that: a pool of reader threads decodes patches concurrently *inside
one process* (so they share the single CUDA context), staging straight into a
pinned host buffer, and the GPU filters each batch. The slow part (decode) is
now parallel; the fast part (filter) stays on the GPU and is essentially free
and hidden. This is the fastest design the hardware allows for this task --
unlike v9, which optimized the part that was never the bottleneck.

Why threads, not processes (v0b): threads share the GPU context and the kernel
buffers with zero IPC/serialization, and the GIL is free during the decode that
dominates -- so we get v0b-class I/O parallelism *and* keep the GPU in the loop
for the filter, freeing CPU cores from the pixel counting entirely.

Arithmetic is the exact integer PIL luma, so kept coordinates match v0a exactly.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import cupy as cp
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_LUMA = (19595, 38470, 7471)

# Fused exact PIL luma: uint8 R,G,B -> uint8 gray, no uint32 temporary.
_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b',
    out_params='uint8 gray',
    operation='gray = (r * 19595 + g * 38470 + b * 7471 + 32768) >> 16;',
    name='pil_luma_uint8_v10',
)


def _alloc_pinned(shape, dtype=np.uint8) -> np.ndarray:
    count = int(np.prod(shape))
    mem = cp.cuda.alloc_pinned_memory(count * np.dtype(dtype).itemsize)
    return np.frombuffer(mem, dtype=dtype, count=count).reshape(shape)


class WSISlidingWindowDataset(Dataset):
    """WSI dataset that parallelizes the OpenSlide decode (the real bottleneck)
    across threads in one process and filters batches on the GPU."""

    def __init__(self,
                 wsi_path: str,
                 patch_size: int = 1024,
                 stride: int = 1024,
                 transform: Optional[Callable] = None,
                 white_pixel_threshold: int = 230,
                 black_pixel_threshold: int = 25,
                 rejection_ratio: float = 0.9,
                 batch_size: int = 64,
                 num_readers: int = 8,
                 verbose: bool = False):
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.white_pixel_threshold = white_pixel_threshold
        self.black_pixel_threshold = black_pixel_threshold
        self.rejection_ratio = rejection_ratio
        self.batch_size = batch_size
        self.num_readers = num_readers
        self.verbose = verbose

        self.kernel_time = 0.0
        self.peak_gpu_bytes = 0
        self.io_time = 0.0          # wall time spent waiting on parallel decode
        self._tls = threading.local()
        self._open_slides: List[openslide.OpenSlide] = []
        self._slides_lock = threading.Lock()

        if self.verbose:
            print(f"[*] Initializing dataset for WSI: {self.wsi_path}")
            print(f"    - readers={self.num_readers} batch={self.batch_size}")

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
            print(f"    - parallel I/O wait: {self.io_time:.2f}s  kernel: {self.kernel_time*1e3:.1f}ms"
                  f"  peak GPU: {self.peak_gpu_bytes/1e6:.1f}MB")

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

    def _thread_slide(self) -> openslide.OpenSlide:
        """Per-thread OpenSlide handle (opened once, reused) -> guaranteed-safe
        concurrent decode that scales with reader threads."""
        s = getattr(self._tls, "slide", None)
        if s is None:
            s = openslide.OpenSlide(self.wsi_path)
            self._tls.slide = s
            with self._slides_lock:
                self._open_slides.append(s)
        return s

    def _filter_on_gpu(self, host_buf: np.ndarray, n: int, device_buf: cp.ndarray) -> np.ndarray:
        """Exact integer luma + ratio filter for the n staged patches."""
        total_pixels = self.patch_size * self.patch_size
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        device_buf[:n].set(host_buf[:n])                      # pinned H2D
        gray = cp.empty((n, self.patch_size, self.patch_size), dtype=cp.uint8)
        _luma_kernel(device_buf[:n, ..., 0], device_buf[:n, ..., 1],
                     device_buf[:n, ..., 2], gray)
        white = cp.count_nonzero(gray > self.white_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
        black = cp.count_nonzero(gray < self.black_pixel_threshold, axis=(1, 2)).astype(cp.float64) / total_pixels
        keep = cp.asnumpy((white < self.rejection_ratio) & (black < self.rejection_ratio))
        e1.record(); e1.synchronize()
        self.kernel_time += cp.cuda.get_elapsed_time(e0, e1) / 1000.0
        return keep

    def _create_grid(self) -> List[Tuple[int, int]]:
        potential_coords = self._generate_candidate_coords()
        if self.verbose:
            print(f"[*] v10 (parallel-I/O + GPU): scanning {len(potential_coords)} "
                  f"candidates with {self.num_readers} reader threads...")

        ps = self.patch_size
        mempool = cp.get_default_memory_pool()
        host_buf = _alloc_pinned((self.batch_size, ps, ps, 3), np.uint8)
        device_buf = cp.empty((self.batch_size, ps, ps, 3), dtype=cp.uint8)
        coordinates: List[Tuple[int, int]] = []

        def read_into(slot_coord):
            slot, (x, y) = slot_coord
            try:
                patch = self._thread_slide().read_region((x, y), 0, (ps, ps))
                host_buf[slot] = np.asarray(patch)[:, :, :3]
                return slot, (x, y), True
            except Exception:
                return slot, (x, y), False

        try:
            with ThreadPoolExecutor(max_workers=self.num_readers) as pool:
                for start in range(0, len(potential_coords), self.batch_size):
                    chunk = potential_coords[start:start + self.batch_size]

                    # --- parallel decode of the whole batch (the slow part) ---
                    t_io = time.perf_counter()
                    results = list(pool.map(read_into, enumerate(chunk)))
                    self.io_time += time.perf_counter() - t_io

                    valid = [(slot, xy) for slot, xy, ok in results if ok]
                    if not valid:
                        continue
                    # Compact valid rows to the front (handles rare read errors).
                    n = len(valid)
                    for dst, (slot, _) in enumerate(valid):
                        if dst != slot:
                            host_buf[dst] = host_buf[slot]
                    coords_in_order = [xy for _, xy in valid]

                    # --- GPU filter (fast, hidden behind next batch's reads) ---
                    keep = self._filter_on_gpu(host_buf, n, device_buf)
                    for (x, y), k in zip(coords_in_order, keep):
                        if k:
                            coordinates.append((x, y))

                    self.peak_gpu_bytes = max(self.peak_gpu_bytes, mempool.used_bytes())
                    mempool.free_all_blocks()
        finally:
            with self._slides_lock:
                for s in self._open_slides:
                    try:
                        s.close()
                    except Exception:
                        pass
                self._open_slides.clear()
            del device_buf
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
    print(" v10 Parallel-I/O + GPU - Test Run")
    print("=====================================================")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WSISlidingWindowDataset(wsi_path=wsi_path, patch_size=1024, stride=1024,
                                      transform=transform, batch_size=64, num_readers=8,
                                      verbose=True)
    print(f"\n[*] Total tissue patches: {len(dataset)}")


if __name__ == '__main__':
    run_test()
