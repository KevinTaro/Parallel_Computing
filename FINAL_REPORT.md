# WSI Patch-Filtering GPU 加速研究：從 v0 到 v32 的系統性並行最佳化

**課程：平行計算** ｜ 日期：2026-06-27

---

## 一、研究概述

本研究以一項固定任務為核心——對 Gigapixel 全切片影像（Whole-Slide Image, WSI）
執行滑動視窗（sliding window）組織篩選——系統性地實作 32 種平行化策略，
探討從純 CPU 到 GPU 全管線的加速極限。

**篩選演算法（所有版本共用）：**
每個 1024×1024 候選 patch 以 ITU-R 601 整數 luma 轉灰階：

```
L = (R×19595 + G×38470 + B×7471 + 32768) >> 16
```

若白像素比（L > 230）或黑像素比（L < 25）≥ 0.9，則丟棄。
**正確性門檻：** 所有版本的保留座標集合必須與 v0a 逐位元完全一致（Jaccard = 1.0000）。

**硬體與測試資料：**

| 項目 | 規格 |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB |
| CPU | ~20 核心 |
| PCIe Memory BW | ~11.7 GB/s (pinned H2D) |
| WSI – 小切片 | S114-80954A，462 候選，127 kept |
| WSI – 大切片 | S114-82742C 20x，4,949 候選，2,171 kept，~19,800 tiles |

---

## 二、演進主線與量測數據

### 線 A：CuPy 計算線（v1–v10）

**策略：** 把像素過濾（grayscale + threshold）從 CPU NumPy 移到 GPU CuPy，
逐步加入批次（batch）、pinned memory、雙緩衝 CUDA stream、mixed precision 等優化層。

**小切片端到端量測（stride=1024，462 candidates，實測最新 benchmark）：**

| 版本 | 策略 | 時間 (s) | vs v0a | vs v0b |
|---|---|---:|---:|---:|
| **v0a** | CPU mono（基準）| 5.832 | **1.00×** | — |
| **v0b** | CPU multiprocessing | **0.666** | **8.76×** | **1.00×** |
| v1 | 逐 patch 傳輸 GPU | 6.105 | 0.96× | 0.11× |
| v2 | Batch H2D | 7.038 | 0.83× | 0.09× |
| v3 | Hybrid routing | 6.949 | 0.84× | 0.10× |
| v4 | Pinned memory | 5.840 | 1.00× | 0.11× |
| v5 | Double-buffered stream | 5.755 | 1.01× | 0.12× |
| v6 | fp16 luma | 6.820 | 0.85× | 0.10× |
| v7 | Fused uint8 kernel | 5.822 | 1.00× | 0.12× |
| v8 | Threaded OpenSlide + GPU | **0.791** | **7.37×** | **0.84×** |
| **v9** | 全層疊加 Ultimate | 5.739 | 1.01× | 0.12× |
| **v10** | Parallel I/O + GPU | **0.862** | **6.77×** | **0.77×** |

> **關鍵發現（Profile 結果）：** v9 Ultimate 叫用 CUDA events 量測，GPU kernel 僅佔約
> 1% 的 pipeline 時間。另外 99% 是 OpenSlide 單執行緒 JPEG 解碼（CPU I/O-bound）。
> 瓶頸不是計算，而是解碼。v8/v10 透過多執行緒餵料而非優化 kernel，才顯著縮短時間。

**compute-only 縮放（data 預讀入後，單 patch 計時 50 次，GPU 已同步）：**

| Patch size | CPU ms | GPU int ms | GPU 加速 |
|---|---:|---:|---:|
| 256 | 0.26 | 0.45 | 0.59× |
| 512 | 1.01 | 0.44 | 2.32× |
| 1024 | 6.37 | 0.91 | **6.98×** |
| 2048 | 29.87 | 3.57 | **8.37×** |

GPU 計算優勢（6–8×）真實存在且隨 patch size 增長，但在完整 pipeline 中被 I/O 掩蓋。

---

### 線 B：GPU 解碼線（v11–v22）

**策略：** 直接解析 TIFF raw bytes，跳過 OpenSlide，以 GPU 解碼 JPEG tile。
設計 4 引擎 × 2 feed 的消融矩陣，並手寫三代 CUDA 解碼器（輸出逐位元一致）。

**手寫 CUDA 解碼器架構（`gpu_jpeg_decoder_optimized.py`，共 5 項技術）：**

| # | 技術 | naive 的問題 | 效果 |
|---|---|---|---|
| 1 | **Register bit-buffer** | 每 bit 一次 global memory load | 降至每 byte 一次，流量砍 ~8× |
| 2 | **8-bit Huffman LUT** | 每碼跑 canonical 逐位元比對（≤16 次）| 大多數碼一次 constant memory 查表解出 |
| 3 | **Constant memory** | 所有熱表走 global load | Warp 廣播讀取，省大量頻寬 |
| 4 | **DC-only fast path** | 平滑區塊仍跑完整 1024-MAC IDCT | 無 AC 係數時直接填常數，跳過整個 transform |
| 5 | **Fused color→luma→count** | 先寫出完整 (N,512,512,3) RGB buffer | 單 kernel 完成 YCbCr→luma→白黑計數，**RGB 從不寫出**，VRAM 大降 |

**完整 benchmark（大切片，4,949 candidates，stride=1024，實測最新數據）：**

| 版本 | 引擎 × Feed | 時間 (s) | vs v0a (64.81s) | Throughput (patches/s) |
|---|---|---:|---:|---:|
| v0a | CPU libjpeg × mono | 64.81 | 1.00× | 76 |
| v0b | CPU libjpeg × multi | 6.004 | **10.8×** | 824 |
| v12 | nvJPEG × mono | 2.081 | 31.1× | 2,378 |
| v13 | nvJPEG × multi | 3.077 | 21.1× | 1,608 |
| v14 | naive CUDA × mono | 1.549 | 41.8× | 3,194 |
| v15 | naive CUDA × multi | 2.473 | 26.2× | 2,001 |
| **v16** | **opt CUDA × mono** | **1.045** | **62.0×** | **4,734** |
| v17 | opt CUDA × multi | 1.618 | 40.1× | 3,059 |
| v18 | ultimate × mono | 1.524 | 42.5× | 3,247 |
| v21 | ultimate + pipeline | 1.622 | 39.9× | 3,051 |
| v22 | ultimate + par-destuff | 2.123 | 30.5× | 2,331 |

**解碼階段時間（大切片）：**

| Engine | 解碼時間 | vs CPU libjpeg |
|---|---:|---:|
| CPU libjpeg (v0a) | ~54 s | 1× |
| naive CUDA (v14) | 4.21 s | ~13× |
| **optimized CUDA (v16)** | **1.15 s** | **~47×** |
| nvJPEG (v12) | 1.85 s | ~29× |

> **optimized 手寫 CUDA 在解碼階段達到 47× 加速，超越 nvJPEG 固定功能硬體（29×）。**
> v16 端到端達到 **62.0×**（大切片），是本研究最佳版本。
> ultimate（v18–v22）在 optimized 之上幾乎無再提升，因為傳輸層已不在關鍵路徑。

**v19→v22 的隱性序列化去除（每步移除一個 bottleneck）：**

```
v19: thread pool 共用 file object + lock → 實際序列讀取
v20: 改用 os.pread（原子定址讀取）→ 真正並行 I/O
v21: producer thread 預取批次 → 讀取移出 critical path
v22: destuffing 移入 reader threads → 主執行緒無剩餘序列成本
```

---

### 線 C：GPU 解碼重演線（v23–v32）

在 GPU decode 管線上**重演 CuPy 計算線各概念**（per-tile, batch, hybrid, pinned, async…），
交叉驗證先前結論在 GPU 解碼背景下是否仍成立。

**小切片代表性數據（462 candidates，實測）：**

| 版本 | 對應概念 | 時間 (s) | vs v0a |
|---|---|---:|---:|
| v23 (dec_v1) | 逐 tile GPU（無 batch）| 18.460 | 0.32× |
| **v24 (dec_v2)** | **Batch decode** | **0.182** | **32.1×** |
| v25 (dec_v3) | Hybrid routing | 0.181 | 32.2× |
| v26 (dec_v4) | Pinned memory | 0.204 | 28.6× |
| v27 (dec_v5) | Async stream | 0.200 | 29.2× |
| v28 (dec_v6) | fp16 luma | 0.186 | 31.3× |
| v32 (dec_v10) | Parallel I/O | 0.231 | 25.2× |

v23 的 18.46 s（比 v0a 慢 3×）再次驗證：逐 tile 的 GPU kernel launch overhead 在無 batch 時極為懲罰性。
v24 引入 batch decode 後立刻跳至 32.1×，與線 A v1→v2 的規律完全一致。

---

## 三、綜合結論

| 命題 | 實測結論 |
|---|---|
| GPU 過濾比 CPU 快？| 計算本身 6–8×，但完整 pipeline 在 CPU I/O-bound 下 GPU 版反比 CPU 慢 |
| 手寫 CUDA 能超越 nvJPEG？| **是：47× vs 29×（解碼階段）**，端到端 62× vs 31× |
| fp16 有幫助？| 無論在 CuPy 過濾或解碼端均無明顯優勢（RTX 5090 上影響小） |
| async stream 值得嗎？| 解碼 compute-bound 情境下效益有限；傳輸已非瓶頸 |
| 最佳策略？| **v16**（optimized CUDA × mono decode）：大切片 62.0×，小切片 30.5× |

**研究主旨——Profile first 的實證：**
線 A（v1–v9）將工程完整度做到位，卻優化了只佔 1% 的 GPU kernel；
硬體探針量出「GPU 被餓死 640×」後，才轉向攻擊真正瓶頸（JPEG 解碼），
最終 v16 在相同正確性保證下達到 62× 端對端加速。
所有 32 個版本均通過正確性驗證（Jaccard = 1.0000）。
