# Data Loader v1 ~ v32 版本演進全紀錄

> **主題**:WSI(Whole Slide Image)切片的 **patch 組織篩選** GPU 加速研究框架。
> 所有版本「功能」相同 — 用整數 luma 灰階 + 白/黑像素比例,判斷每個 patch 要保留或丟棄。
> 差異全部在 **如何把工作丟給 GPU、以及瓶頸到底在哪**。

---

## 0. 全系列共同背景

### 共同任務
1. 在 WSI 上用滑動視窗產生候選座標 `(x, y)`。
2. 讀出 patch 像素(JPEG 解碼)。
3. 轉灰階 → 算白/黑像素比例。
4. 任一比例 ≥ `rejection_ratio`(0.9)就丟掉(過濾純白背景 / 純黑空洞)。
5. 留下的座標進 `self.coordinates`,供 `__getitem__` 訓練取用。

### 共同正確性約束:灰階必須與 PIL 逐位元相同
所有版本都用整數運算精確複製 PIL `convert('L')`,確保結果與 CPU baseline **位元級一致**:

```python
_LUMA = (19595, 38470, 7471)   # R,G,B 整數權重(ITU-R 601-2)
_LUMA_ROUND = 32768            # 0x8000,>>16 前四捨五入
# gray = (R*19595 + G*38470 + B*7471 + 32768) >> 16
```

### 三條演進主線(全系列地圖)

| 線 | 版本 | 核心問題 | 一句話 |
|----|------|----------|--------|
| **A. CuPy 計算線** | v1–v10 | 把「過濾計算」搬上 GPU,並逐步發現真正瓶頸是 **CPU 解碼** | 從「優化計算」走到「優化餵料」 |
| **B. GPU 解碼線** | v11–v22 | 把 **JPEG 解碼本身**也搬上 GPU(nvJPEG / 手寫 CUDA),並做 mono×multi × 解碼器的消融矩陣 | 拔掉 CPU 解碼這個地板 |
| **C. 解碼世界重演線** | v23–v32 | 用 GPU 解碼管線,**重新演一次 v1–v10 的每個概念**(batch / hybrid / pinned / async…) | 把計算線的教訓在解碼線複製驗證 |

> **基準(baseline)**:`v0a` 單執行緒 CPU(1.0×)、`v0b` 多核 CPU。所有版本相對 v0a 衡量加速。

---
---

# 線 A:CuPy 計算線(v1 ~ v10)

---

## v1 — CuPy Full(逐 patch GPU)
📄 `data_loader_v1_cupy_full.py`

**在做什麼**:最天真的 GPU 策略 — 每張 patch 各自做「1 次 H2D 傳輸 + 幾個小 kernel」,只把 2 個純量拉回。存在目的是**示範反例**。

```python
def _keep_patch(self, rgba_np):
    rgb_gpu = cp.asarray(rgba_np[:, :, :3])      # 每張各自 host->device
    gray = gpu_grayscale_uint8(rgb_gpu)
    white_ratio = float(cp.sum(gray > self.white_pixel_threshold)) / total_pixels
    if white_ratio >= self.rejection_ratio: return False   # 白底 early-return
    black_ratio = float(cp.sum(gray < self.black_pixel_threshold)) / total_pixels
    return black_ratio < self.rejection_ratio
```

**程式碼介紹**:`for x,y in coords` 逐張呼叫 `_keep_patch`;`float(cp.sum(...))` 會強制同步把純量拉回 host。

**重點**:⚠️ 每張 = 1 次傳輸 + 多次 kernel + 同步拉回,開銷被放到最大,對「小而多」可能**比 CPU 還慢**。整個系列的「慢的對照組」。

---

## v2 — CuPy Batch(整批 GPU)
📄 `data_loader_v2_cupy_batch.py`

**在做什麼**:讀滿 `batch_size` 張堆成 `(N,H,W,4)`,做**一次** H2D,沿空間軸向量化算完整批。一次傳輸 + 幾個 kernel 服務 N 張。

```python
def _filter_batch(self, batch_rgb):
    gpu = cp.asarray(batch_rgb)                              # 整批一次 H2D
    gray = gpu_grayscale_uint8(gpu)                          # (N,H,W)
    white_counts = cp.sum(gray > self.white_pixel_threshold, axis=(1, 2))  # 沿空間軸 reduce
    black_counts = cp.sum(gray < self.black_pixel_threshold, axis=(1, 2))
    keep = (white_counts/total < r) & (black_counts/total < r)
    return cp.asnumpy(keep)                                  # N 個布林一次拉回
```

**程式碼介紹**:`axis=(1,2)` 向量化 reduction 一次算完 N 張;`np.stack` 疊批讓傳輸只觸發一次。

**重點**:✅ 攤平開銷,GPU 開始划算。⚠️ 記憶體是天花板(3GB 卡,uint32 灰階中間值要 `N×12MB`)。**核心 takeaway:關鍵不是用不用 GPU,而是有沒有把資料批起來餵。**

---

## v3 — CuPy Hybrid(聰明切換 CPU/GPU)
📄 `data_loader_v3_cupy_hybrid.py`

**在做什麼**:逐 chunk 決策 — 大批走 GPU、小批/尾巴走 CPU,因為小工作付不起 GPU overhead。

```python
if len(patches) >= self.gpu_threshold:
    keep_mask = self._filter_gpu(batch_rgb); self.gpu_patches += len(patches)
else:
    keep_mask = self._filter_cpu(batch_rgb); self.cpu_patches += len(patches)
```

**程式碼介紹**:新增 `_filter_cpu`(NumPy)與 `cpu_grayscale_uint8`,與 GPU 路用**同一組** `_LUMA`,故結果與走哪條路無關;`gpu_threshold`(預設16)是決策門檻。

**重點**:✅ 自適應路由,最貼近真實部署;✅ 結果與路徑無關。

---

## v4 — Pinned Memory(頁鎖定記憶體)
📄 `data_loader_v4_cupy_pinned_memory.py`

**在做什麼**:v2/v3 每批都從**可分頁(pageable)** host 記憶體複製、且每批重新配置 device 記憶體。v4 改用**可重用的 pinned(page-locked)staging buffer + 預配置 device buffer + PinnedMemoryPool**。算術與 v2 完全相同,只改資料搬移管線。

```python
cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
pinned_host = _alloc_pinned((bs, ps, ps, 3), np.uint8)   # 可重用頁鎖定 host buffer
device_buf  = cp.empty((bs, ps, ps, 3), dtype=cp.uint8)  # 預配置 device buffer
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    device_buf[:n].set(pinned_host[:n], stream=stream)   # async pinned->device DMA
```

**程式碼介紹**:`_alloc_pinned` 用 `cp.cuda.alloc_pinned_memory` 配頁鎖定 host 陣列;patch 直接讀進 pinned buffer,再以 stream async 上傳。

**重點**:✅ pinned 記憶體吃滿 PCIe 頻寬、可非同步傳輸;✅ buffer 重用避免每批 malloc 抖動。為 v5 的非同步重疊鋪路。

---

## v5 — Async Streams(雙緩衝重疊 I/O 與計算)
📄 `data_loader_v5_cupy_async.py`

**在做什麼**:v4 仍是嚴格序列(讀→傳→算→同步→重複),GPU 與 CPU 互相空等。v5 用 **雙緩衝 + 兩條 CUDA stream** 把「讀下一批」與「算這一批」重疊。

```python
def _launch_batch(self, slot, n):       # 非阻塞:在 slot 自己的 stream 上發 async 傳輸+計算
    with slot['stream']:
        slot['device'][:n].set(slot['pinned'][:n], stream=slot['stream'])
        gray = gpu_grayscale_uint8(slot['device'][:n])
        slot['keep_gpu'] = ...           # 不在這裡同步
# 主迴圈:launch 完立刻去讀下一批 → 下批的讀+傳重疊本批 GPU 計算
```

**程式碼介紹**:每個 slot 各擁有 stream / pinned buffer / device buffer,互不別名;`drain()` 在重用 slot 前才同步。`num_streams≥2`。

**重點**:✅ 慢碟上的主要收益來自把 I/O 藏在計算後面;發 launch 後**不同步**是重疊的關鍵。

---

## v6 — Mixed Precision(fp16 灰階)
📄 `data_loader_v6_cupy_mixed_precision.py`

**在做什麼**:把灰階加權改用 **float16**,省一半記憶體頻寬。刻意用來暴露兩個取捨。

```python
_LUMA_F = (0.299, 0.587, 0.114)
def gpu_grayscale_fp16(rgb_gpu):
    g = rgb_gpu.astype(cp.float16)
    return (g[...,0]*cp.float16(_LUMA_F[0]) + g[...,1]*cp.float16(_LUMA_F[1])
            + g[...,2]*cp.float16(_LUMA_F[2]))
```

**程式碼介紹**:唯一改動是灰階用 fp16 近似(非整數精確),其餘與 v2 相同。

**重點**:⚠️ **精度取捨** — fp16 約 11 bit 尾數,卡在門檻邊緣的 patch 可能翻轉決策(validation 量化有幾個)。⚠️ **硬體相依** — Pascal 卡(GTX1060)fp16 吞吐僅 fp32 的 1/64,反而**比 v2 慢**;只有 Volta+ 才划算。

---

## v7 — Memory-Optimized(低 VRAM 友善)
📄 `data_loader_v7_cupy_memory_optimized.py`

**在做什麼**:針對小 VRAM(3GB)。v2 的 uint32 灰階中間值是輸入的 4 倍大。v7 用 **fused kernel** 直接 uint8→uint8,讓 4× uint32 中間值**從不存在**。

```python
_luma_kernel = cp.ElementwiseKernel(
    in_params='uint8 r, uint8 g, uint8 b', out_params='uint8 gray',
    operation='gray = (r*19595 + g*38470 + b*7471 + 32768) >> 16;',  # 一次讀寫,無 uint32 暫存
    name='pil_luma_uint8')
# 小 chunk + 重用 device buffer + 每 chunk free_all_blocks()
```

**程式碼介紹**:`_luma_kernel` 是融合 elementwise kernel;`device_in`/`gray_buf` 預配置重用;`chunk_size`(預設8)限制常駐量;`self.peak_gpu_bytes` 記錄峰值。

**重點**:✅ 峰值 scratch 從 ~5× 降到 ~1.33× 輸入;✅ fused kernel 仍位元精確(同 PIL 公式)。三招:融合 kernel + 小 chunk + buffer 重用/清理。

---

## v8 — Optimized for RTX 4060(轉折點:攻擊「餵料側」)
📄 `data_loader_v8_cupy_optimized_4060.py`

**在做什麼**:誠實剖析後發現 — **GPU 過濾 <1s,幾乎全部 wall time 是 CPU 端 TIFF 解碼**。v8 第一次同時攻擊餵料側:多執行緒解碼 + 大批(512)+ pinned + fused kernel。

```python
def read_into(buf, slot, x, y):           # 每個 reader thread 自己一個 OpenSlide handle
    slide = tls.slide or openslide.OpenSlide(self.wsi_path)   # read_region 釋放 GIL → 真並行
    buf[slot] = np.asarray(slide.read_region((x,y),0,(ps,ps)))[:, :, :3]
# producer/consumer:producer 用 Semaphore 維持 n_buf 個批的讀任務在 FIFO,
# consumer 主執行緒只做 H2D + GPU filter,讀者永不在批邊界 drain。
```

**程式碼介紹**:`ThreadPoolExecutor` 多 reader 並行解碼填 pinned buffer;`free_buffers` semaphore 做 host-RAM 背壓;`_filter_batch_gpu` 直接在 device buffer 上跑 fused kernel(零額外傳輸)。

**重點**:✅ **真正的提速來源是多執行緒解碼**(libopenslide 釋放 GIL);✅ `n_buf=2` 實測最佳(pinned buffer ~1.5GiB,配置成本算進 grid time)。**這版起,焦點從 GPU 轉到 CPU 解碼。**

---

## v9 — Ultimate GPU(所有計算層堆疊 + 消融開關)
📄 `data_loader_v9_ultimate_gpu.py`

**在做什麼**:把 v1–v7 每個優化堆成一個**設定驅動**的 loader,問「所有已知優化全開,GPU 過濾的效能天花板在哪」。刻意丟掉 v3 的 CPU fallback。

```python
# 7 個可獨立開關的層(供消融研究)
enable_pinned_memory=True   # Layer1 記憶體管理
enable_async=True           # Layer2 雙緩衝 stream
enable_mixed_precision=False# Layer4 fp16(opt-in)
enable_early_exit=True      # Layer7 布林 mask + count_nonzero
# Layer3 batch / Layer5 fused kernel / Layer6 channel-last 內建
```

**程式碼介紹**:`_compute_keep` 依開關走 fp16 或 fused 整數 kernel;`_make_slots`/`_launch`/`drain` 實作雙 stream 重疊;暴露與全系列相同的 `kernel_time`/`peak_gpu_bytes` 契約,可直接套進 benchmark/validation harness。

**重點**:✅ 集大成 + **消融研究**載體(每層可單獨關掉量化貢獻)。但 — 它優化的是**從來就不是瓶頸**的那一側(計算),這正是 v10 的反思。

---

## v10 — Parallel-I/O + GPU(資料驅動的正解)
📄 `data_loader_v10_parallel_io_gpu.py`

**在做什麼**:`hardware_probe.py` 給出鐵證(GTX1060 3GB):luma kernel ~17500 patch/s,但 OpenSlide 單執行緒只 ~27 patch/s → v1–v9 讓 GPU **98% 閒置**(被餓死,不是慢)。8 執行緒 → ~93 patch/s(3.4×)。v10 據此行動。

```python
def _thread_slide(self):                  # 每執行緒一個 OpenSlide handle,重用
    s = getattr(self._tls, "slide", None)
    if s is None: s = openslide.OpenSlide(self.wsi_path); self._tls.slide = s
    return s                               # read_region 釋 GIL → 多執行緒真並行解碼
```

**程式碼介紹**:一池 reader thread 在**單一行程內**並行解碼(共用同一 CUDA context),staging 進 pinned buffer,GPU 過濾每批。慢的部分(解碼)並行了,快的部分(過濾)留在 GPU 幾乎免費且被藏住。

**重點**:✅ **這是硬體允許的最快設計** — 與 v9「優化從不是瓶頸的部分」對比鮮明。✅ 用 thread 不用 process:共用 GPU context、零 IPC、解碼時 GIL free。

---
---

# 線 B:GPU 解碼線(v11 ~ v22)

> v1–v10 都接受同一個地板:`openslide.read_region` 在 **CPU** 用 libjpeg 解碼 JPEG tile,而那幾乎是全部 wall time。線 B **把解碼本身移上 GPU**,並做消融矩陣:`{mono, multi} 餵料 × {nvJPEG, 手寫CUDA naive, 手寫CUDA optimized, ultimate pipeline}`。

---

## v11 — nvJPEG Decode for RTX 5090(拔掉 CPU 解碼)
📄 `data_loader_v11_gpu_decode_5090.py`

**在做什麼**:這些 Philips WSI 的 level-0 是 512×512 **baseline JPEG tile(YCbCr)** — 正好是 NVIDIA nvJPEG 硬體解碼器原生吃的格式。v11 **繞過 OpenSlide**:自己解析 TIFF directory → 讀原始壓縮 tile(純檔案 I/O)→ nvJPEG 批次解碼 → GPU 過濾。

```python
def _read_tiff_tiles(path):              # 自己解析 TIFF IFD(tag 324 offsets / 325 bytecounts / 347 JPEGTables)
    ...                                  # 不經 OpenSlide
# WSI tile 是「縮略 JPEG」(Huffman/quant 表只在 tag347),解碼前要把共用表段拼回每個 tile
# decode_jpeg(..., device='cuda')  ← nvJPEG 硬體解碼 + YCbCr->RGB
```

**程式碼介紹**:幾何捷徑 — `patch=stride=2×tile` 且影像尺寸是 tile 整數倍,故每個 1024² patch 恰是 2×2 個不重疊 tile,白/黑計數可跨 4 tile 相加。幾何不符就 raise(這是解碼特化,非通用 resampler)。

**重點**:✅ 完全移除 CPU 解碼。⚠️ **算術警告**:像素值來自 nvJPEG 而非 libjpeg,IDCT+YCbCr→RGB 可能差 ±1 LSB,卡門檻的 patch 偶爾翻轉 → kept set 與 v8 **近乎**相同而非保證位元相同。

---

## v12 / v13 — nvJPEG × {mono / multi} 餵料(消融格)
📄 `data_loader_v12_gpu_decode_mono.py` ・ `data_loader_v13_gpu_decode_multi.py`

**在做什麼**:把 v11 的 nvJPEG 解碼拆成兩個消融格,隔離「CPU 讀取並行度」的貢獻。
- **v12**(base v0a):單執行緒讀原始 tile + nvJPEG 解碼 → 隔離「純特化 GPU 解碼」效益。
- **v13**(base v0b):一池 thread 並行讀原始 tile + nvJPEG 解碼 → 隔離「讀取並行」加成。

```python
from data_loader_v11_gpu_decode_5090 import _read_tiff_tiles   # 共用驗證過的 TIFF 解析
# v12: 單執行緒讀 → decode_jpeg(cuda)
# v13: ThreadPoolExecutor 並行讀(thread 而非 process:I/O-bound 且 GPU context 不可跨 fork)
```

**程式碼介紹**:兩者解碼/過濾完全相同,唯一差別是「讀 tile 用 1 條還是多條 thread」。

**重點**:消融設計 — `v12 vs v0a` = 純 GPU 特化解碼;`v13 vs v12` = 讀取並行加成;`v13 vs v0b` = 特化 GPU 解碼 vs 多核 CPU 解碼。

---

## v14 / v15 — 手寫 CUDA naive 解碼 × {mono / multi}(消融格)
📄 `data_loader_v14_gpu_compute_mono.py` ・ `data_loader_v15_gpu_compute_multi.py`

**在做什麼**:不用 nvJPEG 固定功能單元,改用**手寫 CUDA kernel**(`gpu_jpeg_decoder.GpuJpegDecoder`)做解碼 — 一個 CUDA thread 一個 tile,數千 tile 並行。這是「純並行、無固定功能單元」的對照。

```python
from gpu_jpeg_decoder import GpuJpegDecoder      # 手寫:Huffman -> dequant -> IDCT -> color
# v14: 單執行緒讀;v15: 多執行緒讀。decode 都在 CUDA 一般核心。
```

**程式碼介紹**:解碼器刻意 **naive**(一 thread 序列處理一 tile、float IDCT),量的是「跨 tile 並行」效益而非調優解碼器。

**重點**:⚠️ float IDCT + nearest 色度上採樣**非**位元精確,門檻附近少數 patch 會與 v0a 不同。對照:`v14 vs v12` = 通用 CUDA 解碼 vs 特化 nvJPEG;`v14 vs v0a` = CUDA 核心並行解碼的提速。

---

## v16 / v17 — 手寫 CUDA **optimized** 解碼 × {mono / multi}
📄 `data_loader_v16_cuda_opt_mono.py` ・ `data_loader_v17_cuda_opt_multi.py`

**在做什麼**:與 v14/v15 同格,但換上**優化版解碼器** `gpu_jpeg_decoder_optimized`。輸出與 v14 **位元相同**,差別純粹是吞吐。

```python
from gpu_jpeg_decoder_optimized import GpuJpegDecoderOptimized
# 優化:register bit-buffer + 8-bit Huffman LUT、熱表進 __constant__ 記憶體、
#       DC-only block 跳過 IDCT、fused YCbCr->luma->count(不產生 (N,512,512,3) RGB buffer)
```

**程式碼介紹**:最關鍵的 fused「YCbCr→luma→count」讓 RGB 中間 buffer**從不物化** — 既省頻寬又讓小 VRAM 卡放得下;`batch_size` 可per-card 手調。

**重點**:✅ 位元等同 naive 但更快、更省記憶體。對照:`v16 vs v14` = 優化win;`v16 vs v12` = 調優CUDA vs nvJPEG。

---

## v18 / v19 — 手寫 CUDA **ultimate** pipeline × {mono / multi}
📄 `data_loader_v18_cuda_ultimate_mono.py` ・ `data_loader_v19_cuda_ultimate_multi.py`

**在做什麼**:各分支的「終點」 — v16 優化解碼器 **+** 傳輸/重疊層(pinned + 雙緩衝 async stream pipeline + 重用 buffer pool)。輸出仍位元等同 v14/v16。

```python
from gpu_jpeg_decoder_ultimate import GpuJpegDecoderUltimate
# v16 kernel wins + v4 pinned + v5 async streams + v9 pipeline + v7 memory pool
# 兩 slot/兩 stream:批 k+1 的 讀+destuff+upload 重疊批 k 的 GPU decode+count
# v6 fp16 刻意省略(fp32 IDCT 保持 kept set 精確)
```

**程式碼介紹**:`v18` 單執行緒讀但 pipeline 仍把讀/destuff 重疊在前一批 GPU 解碼後;`v19` 改多執行緒讀。

**重點**:✅ 收益來自把 host 工作與傳輸藏在 GPU 計算後面。⚠️ **(伏筆)v19 的「多核」其實沒真的並行讀** → 見 v20。

---

## v20 — True Parallel Reads via `os.pread`(揭穿 v19 的假並行)
📄 `data_loader_v20_cuda_ultimate_pread.py`

**在做什麼**:v19 的 reader thread 全部共用**一個 file object + 一把 lock**,單一 OS 檔案位置使 `seek+read` 必須序列化 → v19 的讀其實是「序列 + thread pool 開銷」。v20 改用 **positional read**。

```python
# v19 的問題:
#   with read_lock: fh.seek(off); return fh.read(bc)   # 每個 thread 在此序列化
# v20 的修正:
os.pread(fd, nbytes, offset)   # 單一原子 syscall,不碰共用檔案位置 → 多 thread 同 fd 無鎖並行,GIL 釋放
```

**程式碼介紹**:其餘與 v18/v19 完全相同(優化解碼器 + pinned + pipeline),輸出**位元相同** v0a。

**重點**:⚠️ 暖快取本地 SSD 上,讀只佔 ~7% 且已被 pipeline 藏住 → v20 ≈ v18 ≈ v19。v20 只在**冷快取 / NFS / 慢 HDD** 才贏。`read_time` 報出來讓你判斷。

---

## v21 — True Prefetch Pipeline(producer/consumer,讀離開關鍵路徑)
📄 `data_loader_v21_cuda_ultimate_pipeline.py`

**在做什麼**:v20 雖並行讀,但仍在主執行緒**每批** `pool.map`,thread 派發+收集落在 GPU submit/fetch 關鍵路徑上 → 批間 GPU 短暫挨餓。v21 修架構:**真正的問題不是「讀有沒有並行」,而是「讀在哪裡跑」**。

```python
# 專屬 producer thread 用 pool.submit(非 pool.map)讓 reader FIFO 跨批邊界不 drain
# os.pread 無鎖並行;bounded ready_q(Semaphore = prefetch 深度背壓)交給主執行緒
# 主執行緒只做 GPU 工作:decoder.submit(async H2D+decode+count) + decoder.fetch
```

**程式碼介紹**:這就是 v8 對 CPU 解碼用過的 producer/consumer,移植到 custom-CUDA 解碼管線。輸出位元等同 v0a。

**重點**:✅ 讀持續並行、完全領先並重疊 GPU 解碼,主迴圈不再卡在每批讀派發。誠實預期:暖快取下 GPU-bound,v21 目標是「不再比 v18 慢」並在慢碟拉開。

---

## v22 — Parallel Read **and** Destuff(把最後的序列 host 工作也搬走)
📄 `data_loader_v22_cuda_parallel_destuff.py`

**在做什麼**:v21 仍在主執行緒序列做 `destuff_tile_scan`(對每 tile ~96KB 做 `bytes.replace(b'\xff\x00', b'\xff')`,一批 2048 tile = ~190MB 純 Python pass)。讀並行後,**這成了關鍵路徑上的主要序列成本** → 所以 v21 只追平 v18。v22 把 destuff 也丟給 reader thread。

```python
# reader task:  tid -> destuff_tile_scan(os.pread(fd, ...))   # 讀+destuff 都並行、領先 GPU
# 主執行緒只:   decoder.submit_scans(已destuff,無 destuff 迴圈) + decoder.fetch
class _PipelineUltimateDecoder(GpuJpegDecoderUltimate):
    def submit_scans(...): ...   # 與 submit 位元相同的 GPU 工作,但拿掉內部 destuff
```

**程式碼介紹**:每 tile 的 Python 工作從關鍵路徑**完全消失**,迴圈只受 GPU 解碼吞吐限制。輸出位元等同 v0a。

**重點**:✅ 暖快取下應**明顯勝過 v18**(v18 同時付序列讀與序列 destuff)。⚠️ **單 GPU 解碼地板**:要再突破得多 GPU 分片(tile 獨立 → 近線性)或更快(犧牲位元精確)的 kernel。這是「單 GPU + 並行 host 餵料」的極限。

---
---

# 線 C:解碼世界重演線(v23 ~ v32)

> 把 **v1–v10 的每個概念**,在 GPU 解碼管線(`GpuJpegDecoderOptimized` / `GpuJpegDecoderUltimate`)上**重新實作一次**,完成 `v1..v10 → v23..v32` 的對照。其中 v27/v31/v32 是對既有版本的**薄包裝(thin re-export)**。

| 版本 | 對應概念 | 等同/實作 |
|------|----------|-----------|
| v23 | v1 naive 逐單元 | batch=1 硬編碼 |
| v24 | v2 批次 | `count_batch` + 自動批量 |
| v25 | v3 hybrid | GPU 解碼 + CPU fallback |
| v26 | v4 pinned | 端到端 pinned 讀緩衝 |
| v27 | v5 async | **= v18**(薄包裝) |
| v28 | v6 fp16 | Y plane + fp16 count |
| v29 | v7 memory budget | VRAM 預算自動批量 |
| v30 | v8 threaded reads | `os.pread` 並行讀 |
| v31 | v9 all-combined | **= v22**(薄包裝) |
| v32 | v10 producer/consumer | **= v21**(薄包裝) |

---

## v23 — dec_v1 naive(batch=1 硬編碼)
📄 `data_loader_v23_dec_v1_naive.py`

**在做什麼**:重演 v1「每單元一次計算」。`batch_size` **硬編碼為 1**,每個 tile 單獨送 GPU:1 次 kernel launch + 1 次 H2D + 1 次結果回拷。

**程式碼介紹**:`batch=1` 不是參數而是 class 層級的架構決定,忠實代表 v1 概念。~100k tile 累積 ~0.5s 純 launch 開銷,GPU 利用率 <10%。

**重點**:⚠️ 最慢的 GPU 解碼版,GPU 在此毫無收益(連純 CPU OpenSlide 都可能更快)。**唯一目的:展示為何要 batching**(對照 v23 vs v24)。

---

## v24 — dec_v2 batch(批 N tile + 自動批量)
📄 `data_loader_v24_dec_v2_batch.py`

**在做什麼**:重演 v2。把 N 個壓縮 JPEG tile 批進一次 `count_batch(tiles, wt, bt)`(融合 kernel:decode→Y plane→threshold→count),回傳兩個長度 N 的 int64 陣列。

```python
# batch_size=0 → 從 free VRAM 自動估,夾在 [64, 8192]
per_tile = 512*512*3 + 96*1024
batch_size = clamp(int(free_bytes * 0.35 / per_tile), 64, 8192)
```

**程式碼介紹**:第一批顯示完整 launch+H2D 延遲,後續批攤平。

**重點**:✅ 相對 v23 顯著提速;自動批量讓 GPU 維持 ~35% 解碼 buffer 佔用。

---

## v25 — dec_v3 hybrid(GPU 解碼 + CPU fallback)
📄 `data_loader_v25_dec_v3_hybrid.py`

**在做什麼**:重演 v3。兩個 guard 觸發 CPU fallback:`tile 數 < gpu_threshold`(128)或 `free_mb < min_vram_mb`(500)。

**程式碼介紹**:GPU 路用 `count_batch`;CPU fallback 經 OpenSlide 讀 patch,用 v6 的整數 luma 公式 threshold/count;`cpu_tiles`/`gpu_tiles` 記錄分流。CPU fallback 讀的是 patch 不是 tile,故 tile 計數用「整 patch 計數 ÷ tiles-per-patch」近似(v3 best-effort 哲學)。

**重點**:✅ 低 VRAM / 少 tile 時不付 GPU overhead。

---

## v26 — dec_v4 pinned(端到端頁鎖定)
📄 `data_loader_v26_dec_v4_pinned.py`

**在做什麼**:重演 v4。解碼器內部 h_scan 已是 pinned;v26 讓**上游讀取 staging buffer 也 pinned**,達成 `File → pinned h_raw → GPU(DMA,無 OS bounce copy)` 端到端頁鎖定。

```python
# 1. 算所有非空 tile 的最大壓縮尺寸 max_raw
# 2. 配置一塊 pinned flat buffer:(batch_size, max_raw)
# 3. 每 tile 原始 bytes 複製進 row i,再以 memoryview slice 傳給 count_batch
```

**程式碼介紹**:每個 byte 從檔案讀到 GPU 全程經頁鎖定記憶體,DMA 引擎全速。IOMMU/Unified Memory 系統上會無錯退回一般 host 記憶體。

**重點**:✅ 壓縮 scan H2D 與結果 D2H 都走 full PCIe 頻寬。

---

## v27 — dec_v5 async(= v18,薄包裝)
📄 `data_loader_v27_dec_v5_async.py`

**在做什麼**:重演 v5 雙緩衝 async stream。在 GPU 解碼世界,這個概念**正好就是 v18**(`GpuJpegDecoderUltimate` 的兩 slot/兩 stream:`submit` 非阻塞、`fetch` 阻塞;主迴圈交錯 submit(k) 與 fetch(k-1))。

```python
from data_loader_v18_cuda_ultimate_mono import WSISlidingWindowDataset
class V27AsyncDataset(WSISlidingWindowDataset): ...   # 純別名,使 v5→v27 對應顯式化
```

**重點**:薄 re-export。對照 `v24(非async) vs v27/v18(async)`。

---

## v28 — dec_v6 fp16(Y plane + fp16 count)
📄 `data_loader_v28_dec_v6_fp16.py`

**在做什麼**:重演 v6 fp16,但更聰明。直接取解碼器的 **Y(luminance)plane**(YCbCr 的 Y 就是 BT.601 luma),轉 fp16 再 count。

```python
Yp = decoder._decode_planes(tiles)        # (N,512,512) uint8,Y plane 直接是亮度
Yf = Yp.astype(cp.float16)
w = cp.count_nonzero(Yf > wp, axis=(1,2)); b = cp.count_nonzero(Yf < bp, axis=(1,2))
```

**程式碼介紹**:用 Y plane 的三個好處 — 只需 1/3 資料(無 RGB 三通道頻寬)、不需算 luma 公式、fp16 對 0..255 整數精確(2048 內精確)。

**重點**:✅ 與 v6 的關鍵區別:**float32 decode + fp16 count**(只在便宜的 count 用 fp16),decode 精度不妥協 → 比 v6 更正確。

---

## v29 — dec_v7 membudget(VRAM 預算自動批量)
📄 `data_loader_v29_dec_v7_membudget.py`

**在做什麼**:重演 v7。接受 `vram_budget_gb`(預設 1.0),從預算反推批量,每 chunk 後 `free_all_blocks()` 防止 pool 成長。

```python
budget_bytes = vram_budget_gb * 1e9
per_tile = 512*512*3 + 96*1024
batch_size = max(32, budget_bytes // per_tile)   # 1.0GB → ~1130 tile/batch
```

**程式碼介紹**:`batch_size>0` 時覆寫自動值;`peak_gpu_bytes` 記錄 high-water mark。

**重點**:✅ 受限 GPU 安全不 OOM、峰值 VRAM 有界;大 GPU 上因批較小會稍慢。

---

## v30 — dec_v8 threaded(`os.pread` 並行讀)
📄 `data_loader_v30_dec_v8_threaded.py`

**在做什麼**:重演 v8 並行讀。用 `os.pread`(釋 GIL、位置原子)而非 `seek+read`(需鎖)。

```python
fd = os.open(path, os.O_RDONLY)
read_tile = lambda tid: os.pread(fd, bytecounts[tid], offsets[tid])
tiles = list(pool.map(read_tile, [tile_ids[ci] for ci in chunk]))   # 共用 fd,無鎖並行
```

**程式碼介紹**:解釋了 `os.pread` 如何修正 v19 的錯(v19 共用 file object+mutex 使讀序列化)。

**重點**:⚠️ 暖快取本地 SSD 上 GPU 解碼佔 ~90-93%,讀僅 ~7% → v30 約等於或略慢於 v24(pool 開銷抵銷讀並行)。冷儲存才明顯贏。

---

## v31 — dec_v9 combined(= v22,薄包裝)
📄 `data_loader_v31_dec_v9_combined.py`

**在做什麼**:重演 v9「全部一起開」。GPU 解碼世界的「everything combined」正是 **v22**(batch + pinned + async + memory pool + 並行讀 + 並行 destuff)。

```python
from data_loader_v22_cuda_parallel_destuff import WSISlidingWindowDataset   # 薄 re-export
```

**重點**:薄包裝。架構:reader threads 做 `os.pread → destuff → ready_q`;主執行緒只 `submit_scans + fetch`。所有 host 工作離開關鍵路徑,主迴圈只受 GPU 解碼吞吐限制。

---

## v32 — dec_v10 pipeline(= v21,薄包裝)
📄 `data_loader_v32_dec_v10_pipeline.py`

**在做什麼**:重演 v10 producer/consumer I/O pipeline。GPU 解碼世界的等價實作正是 **v21**(專屬 producer thread + reader pool → bounded ready_q → consumer 只做 GPU 工作)。

```python
from data_loader_v21_cuda_ultimate_pipeline import WSISlidingWindowDataset   # 薄 re-export
```

**程式碼介紹**:與 v31 的關鍵區別 — `v21/v32` 只並行讀(無並行 destuff,純 v10 架構);`v22/v31` 讀+destuff 都並行(v22 在其上多疊一層)。

**重點**:薄包裝。冷儲存/NFS 上讀慢時,v21 持續餵 GPU 不停頓。

---
---

# 全系列總覽

## 三線一圖

```
線 A(計算側 / CuPy)        線 B(解碼側 / GPU decode)      線 C(解碼世界重演 v1–v10)
v1  逐patch GPU ───────────┐
v2  批次 ──────────────────┼──────────────────────────────► v24 dec_v2 批次
v3  hybrid ────────────────┼──────────────────────────────► v25 dec_v3 hybrid
v4  pinned ────────────────┼──────────────────────────────► v26 dec_v4 pinned
v5  async streams ─────────┼──────────────────────────────► v27 (= v18)
v6  fp16 ──────────────────┼──────────────────────────────► v28 dec_v6 fp16
v7  memory-opt(fused k)────┼──────────────────────────────► v29 dec_v7 membudget
v8  4060:多執行緒解碼 ⚡轉折 ┘
v9  ultimate(集大成+消融)
v10 parallel-IO ⚡正解
                            v11 nvJPEG(拔CPU解碼)
                            v12/v13 nvJPEG × mono/multi
                            v14/v15 手寫CUDA naive × mono/multi
                            v16/v17 手寫CUDA opt   × mono/multi
                            v18/v19 ultimate pipeline × mono/multi ──► v27/v23..
                            v20 真並行讀(os.pread)
                            v21 prefetch pipeline ─────────────────► v32
                            v22 並行讀+destuff(單GPU極限)─────────► v31
                                                                     v23 dec_v1 batch=1
                                                                     v30 dec_v8 os.pread
```

## 核心領悟(整個研究的主線)

1. **批次化 > 用不用 GPU**(v1→v2):開銷攤平才是 GPU 划算的前提。
2. **優化要打在瓶頸上**(v9 vs v10):v1–v9 拼命優化 GPU 計算,但 `hardware_probe` 證明 GPU 98% 閒置 — 瓶頸是 **CPU 解碼**。v10 並行解碼才是正解。
3. **拔掉地板**(v11–v22):把 JPEG 解碼也搬上 GPU(nvJPEG 或手寫 CUDA),消滅 CPU 解碼這個硬地板。
4. **「並行」不只是有沒有,而是在哪跑**(v19→v20→v21→v22):假並行(共用 fd+lock)→ 真並行(os.pread)→ 離開關鍵路徑(producer/consumer)→ 連 destuff 都並行。逐步把序列 host 工作搬離 GPU 關鍵路徑。
5. **單 GPU 的極限**(v22):host 餵料全並行後,wall clock = 單 GPU 解碼時間;再快只能多 GPU 分片或犧牲位元精確。
6. **正確性貫穿全程**:整數 luma 路徑位元等同 PIL;fp16 / nvJPEG / 手寫 float-IDCT 會有 ±1 LSB 漂移,門檻邊緣 patch 可能翻轉,benchmark 都報出 kept-set delta。

## 消融矩陣(線 B)

| 餵料 ＼ 解碼器 | nvJPEG(特化) | 手寫CUDA naive | 手寫CUDA opt | ultimate pipeline |
|---------------|---------------|----------------|--------------|-------------------|
| **mono**(v0a base) | v12 | v14 | v16 | v18 |
| **multi**(v0b base) | v13 | v15 | v17 | v19 / v20 / v21 / v22 |

> v11 = 多核 nvJPEG 的 RTX5090 特化首版;v20–v22 是 v19 多核分支在「讀取並行度與排程」上的逐步修正。
