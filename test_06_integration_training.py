"""
test_06_integration_training.py  --  Phase 6: Integration / Mock Training

Runs a realistic training loop (DataLoader -> mock CNN forward/backward on GPU)
and reports wall-clock time split into data-loading vs model-step, i.e. what
fraction of training is spent waiting on the data pipeline. This is the bottom
line: filtering strategy only matters at the margin if data loading already
dominates the step.

    python test_06_integration_training.py --stride 2048 --iters 20 --batch 4
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from test_performance_framework import CuPyTester, DEFAULT_WSI


class MockModel(nn.Module):
    """Tiny CNN stand-in for a real classifier head."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(" PHASE 6: INTEGRATION / MOCK TRAINING")
    print(f" device={device} batch={args.batch} workers={args.workers}")
    print("=" * 72)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = CuPyTester(wsi_path=args.wsi, patch_size=args.patch_size, stride=args.stride).build("v0a_mono")
    ds.transform = transform
    print(f"[*] Dataset: {len(ds)} patches")
    if len(ds) == 0:
        return

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    model = MockModel().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    data_time = compute_time = 0.0
    n = 0
    t_load = time.perf_counter()
    it = iter(loader)
    for step in range(args.iters):
        try:
            xb, _ = next(it)
        except StopIteration:
            it = iter(loader)
            xb, _ = next(it)
        data_time += time.perf_counter() - t_load

        t0 = time.perf_counter()
        xb = xb.to(device, non_blocking=True)
        yb = torch.randint(0, 2, (xb.size(0),), device=device)  # mock labels
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_time += time.perf_counter() - t0
        n += 1
        t_load = time.perf_counter()

    total = data_time + compute_time
    print(f"\n[*] {n} iterations")
    print(f"    data-load time : {data_time:7.3f}s  ({100*data_time/total:5.1f}%)")
    print(f"    model-step time: {compute_time:7.3f}s  ({100*compute_time/total:5.1f}%)")
    print(f"    per-iteration  : {total/n*1e3:7.1f} ms")
    print(f"\n Data-loading fraction = {100*data_time/total:.1f}% -> "
          f"{'I/O-bound (optimize loader)' if data_time > compute_time else 'compute-bound'}")


if __name__ == "__main__":
    main()
