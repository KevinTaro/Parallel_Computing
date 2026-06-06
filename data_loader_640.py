"""
core/data_loader_640.py

640x640 專用的 WSI 動態滑動視窗資料載入器。

設計目標：
1. 保留 core/data_loader.py 的既有 1024x1024 預設行為，不做破壞性修改。
2. 提供一個預設 patch_size/stride=640 的版本給 DenseNet121 相關實驗使用。
3. 演算法與過濾邏輯沿用原始 WSISlidingWindowDataset。
"""

from __future__ import annotations

from typing import Callable, Optional

try:
    from .data_loader import WSISlidingWindowDataset
except ImportError:
    # 支援 `python core/data_loader_640.py` 直接執行
    from data_loader import WSISlidingWindowDataset


class WSISlidingWindowDataset640(WSISlidingWindowDataset):
    """
    640x640 版本的 WSI 資料集。

    注意：
    - 預設 patch_size 與 stride 為 640。
    - 其餘白/黑背景過濾流程、讀圖流程完全沿用基底類別。
    - 可透過參數覆寫尺寸（例如 patch_size=512），但預設是 640。
    """

    def __init__(
        self,
        wsi_path: str,
        patch_size: int = 640,
        stride: int = 640,
        transform: Optional[Callable] = None,
        white_pixel_threshold: int = 230,
        black_pixel_threshold: int = 25,
        rejection_ratio: float = 0.9,
    ):
        super().__init__(
            wsi_path=wsi_path,
            patch_size=patch_size,
            stride=stride,
            transform=transform,
            white_pixel_threshold=white_pixel_threshold,
            black_pixel_threshold=black_pixel_threshold,
            rejection_ratio=rejection_ratio,
        )


# 向後相容：允許舊程式仍以 WSISlidingWindowDataset 名稱匯入 640 版本
WSISlidingWindowDataset = WSISlidingWindowDataset640


def run_test_640() -> None:
    """640 版本最小測試入口（僅示範，不會更動原始 1024 版）。"""
    from torchvision import transforms

    wsi_path = "data/raw_wsi/S114-82742C-Her2(4B5) 20x.tiff"

    dataset = WSISlidingWindowDataset640(
        wsi_path=wsi_path,
        patch_size=640,
        stride=640,
        transform=transforms.ToTensor(),
        white_pixel_threshold=230,
        black_pixel_threshold=25,
        rejection_ratio=0.9,
    )

    print(f"[640 loader] dataset size = {len(dataset)}")
    if len(dataset) > 0:
        patch, coords = dataset[0]
        print(f"[640 loader] first patch shape = {tuple(patch.shape)}, coords = {coords}")


if __name__ == "__main__":
    run_test_640()
