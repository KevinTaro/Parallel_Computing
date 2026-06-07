"""
test_03_end_to_end_dataloader.py  --  Phase 3: End-to-End DataLoader

Measures the *retrieval* path: __getitem__ per-patch load time and PyTorch
DataLoader batch-load time at several batch sizes / worker counts.

Note: patch retrieval (read_region + transform) is identical across all
versions -- the versions differ only in how the grid is *filtered* during
__init__ (Phases 0-2). So this phase characterizes the shared data-loading
cost and how DataLoader parallelism helps, using the v0a dataset to provide the
coordinate list.

    python test_03_end_to_end_dataloader.py --stride 2048 --n 100
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from test_performance_framework import CuPyTester, DEFAULT_WSI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    print("=" * 72)
    print(" PHASE 3: END-TO-END DATALOADER")
    print("=" * 72)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    t0 = time.perf_counter()
    cls = CuPyTester(wsi_path=args.wsi, patch_size=args.patch_size, stride=args.stride)
    ds = cls.build("v0a_mono")
    ds.transform = transform
    init_t = time.perf_counter() - t0
    print(f"[*] Dataset init (filtering): {init_t:.2f}s, {len(ds)} patches")

    if len(ds) == 0:
        print("[!] No patches; choose a finer stride.")
        return

    # --- per-patch __getitem__ ---------------------------------------------
    n = min(args.n, len(ds))
    idxs = [i % len(ds) for i in range(n)]
    t0 = time.perf_counter()
    shape = None
    for i in idxs:
        patch, _ = ds[i]
        shape = patch.shape
    per_patch = (time.perf_counter() - t0) / n
    print(f"\n[*] __getitem__: {per_patch*1e3:.2f} ms/patch over {n} loads, shape={tuple(shape)}")

    # --- DataLoader batching -----------------------------------------------
    print("\n[*] DataLoader batch-load time:")
    print(f"    {'batch':>6s} | {'workers':>7s} | {'batch ms':>9s} | {'ms/patch':>9s}")
    print("    " + "-" * 42)
    for batch_size in (1, 4, 8, 16):
        for workers in (0, 2):
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=workers)
            it = iter(loader)
            next(it)  # warmup (worker spawn / first read)
            t0 = time.perf_counter()
            batches = 0
            for _ in range(min(5, len(loader) - 1)):
                xb, _ = next(it)
                batches += 1
            if batches == 0:
                continue
            dt = (time.perf_counter() - t0) / batches
            print(f"    {batch_size:>6d} | {workers:>7d} | {dt*1e3:9.2f} | {dt*1e3/batch_size:9.2f}")
            del loader, it

    print("\n Per-patch cost dominated by read_region decode; more workers parallelize it.")


if __name__ == "__main__":
    main()
