# GPU JPEG 解碼器逐行詳解

本文件詳細說明三個手寫 CUDA baseline-JPEG 解碼器的**每一段程式碼在做什麼**，以及**用了哪些加速技巧**。三個檔案是同一條演進線：

```
gpu_jpeg_decoder.py            naive   一個 thread 解一張 tile，最樸素，當基準
        │  (相同演算法、相同輸出)
        ▼
gpu_jpeg_decoder_optimized.py  優化     暫存器 bit-buffer + Huffman LUT + 常數記憶體 + 融合 count
        │  (繼承，bit-for-bit 相同輸出)
        ▼
gpu_jpeg_decoder_ultimate.py   終極     再加 pinned 記憶體 + 雙緩衝 stream pipeline + buffer pool
```

三者輸出**逐位元相同**（不和 libjpeg 完全相同，因為用浮點 IDCT + 最近鄰 chroma 升採樣，差幾個 LSB），所以呼叫端篩選出的 patch 集合不會變。差別純粹是吞吐量。

---

## 0. 背景知識：這些 tile 是什麼

這些 Philips WSI 把 level-0 存成 **512×512 的 baseline JPEG tile**，色彩是 YCbCr 4:2:0：
- **Y（亮度）**：2×2 取樣 → 每張 tile 完整 512×512。
- **Cb / Cr（色度）**：1×1 取樣 → 各 256×256（長寬各砍半）。

JPEG 的最小編碼單位 **MCU** 在 4:2:0 下是 16×16 像素，內含 6 個 8×8 區塊：4 個 Y + 1 Cb + 1 Cr。512/16 = 32，所以一張 tile 是 **32×32 = 1024 個 MCU**（程式裡傳進 kernel 的 `mcux=mcuy=32`）。

關鍵設計：這些是 **abbreviated（縮寫）JPEG 串流** —— Huffman/量化表只存一份在 TIFF 的 `JPEGTables` tag，每張 tile 只帶自己的 SOF0/SOS + 熵編碼資料。所以解碼器先解析「共用表」一次，再批次解每張 tile 的掃描資料。

JPEG 解碼一個區塊的標準流程：
```
熵編碼位元流 → Huffman 解碼 → 反量化(dequant) → 反zigzag → 反離散餘弦(IDCT) → 加128 → 像素
```

---

# 一、gpu_jpeg_decoder.py（naive 版）

## CPU 端：表格解析（小、序列，不平行化）

### `_ZIGZAG`（L29-34）
JPEG 把 8×8 係數用 zig-zag 順序存放（低頻在前）。這個陣列把「zig-zag 索引 k」對應回「自然 row-major 索引」。解碼時 `F[zz[k]] = 值`，把係數放回正確的二維位置。

### `_idct_cos_table()`（L37-44）
預先算好 IDCT 的餘弦基底表 `cs[k*8+i] = a(k)·cos((2i+1)kπ/16)`，其中正規化係數 `a(0)=√(1/8)`、`a(k>0)=1/2`。
**技巧**：把三角函數預算成表，kernel 裡只做乘加，不在 GPU 上算 `cos`。

### `_build_huffman(counts, vals)`（L50-71）
把 JPEG DHT 的 `(每長度的碼數 counts[16], 符號 vals[])` 轉成標準 **canonical Huffman 解碼表**：
- `codes[]`：依長度遞增、同長度遞增地指派位元碼（標準 Annex C 規則：`code` 每個符號 +1，每升一個長度左移一位）。
- `mincode[L]/maxcode[L]`：長度 L 的最小/最大碼值。
- `valptr[L]`：長度 L 的第一個符號在 `hv[]` 裡的位置。
- `maxcode[L] = -1` 表示沒有該長度的碼。
這套表讓解碼時可以「逐位元累積 code，一旦 `code <= maxcode[L]` 就命中」。

### `parse_shared_tables(jpegtables)`（L74-109）
掃過共用表 blob，逐個 JPEG marker 解析：
- L80-88：標準的 marker 掃描（`0xFF` 開頭，跳過 SOI/EOI，讀 `seg_len`）。
- L90-98 **DQT（量化表）**：`pq_tq` 高 4 bit 是精度、低 4 bit 是表 id；8-bit 精度讀 64 bytes（zig-zag 順序）。
- L99-107 **DHT（Huffman 表）**：`tc_th` 高 4 bit 是類別（0=DC,1=AC）、低 4 bit 是 id；`slot = 類別*2 + id` → 0..3 對應 DC0,DC1,AC0,AC1。
回傳 `{quant, huff}`。

### `parse_tile_frame(tile)`（L112-134）
解析單張 tile 的 **SOF0**（取得每個分量的取樣比 h/v、量化表 id `tq`）與 **SOS**（取得每分量的 DC/AC Huffman 表選擇器 `td/ta`），合併進 `comps`。所有 tile 的 frame 結構相同，所以只需解析一張當樣本。

### `destuff_tile_scan(tile)`（L137-145）★重要技巧：byte de-stuffing
JPEG 熵資料裡若出現 `0xFF`，編碼端會塞一個 `0x00` 進去（避免和 marker 混淆）。解碼前要把 `0xFF 0x00` 還原回 `0xFF`：
```python
return tile[start:end].replace(b'\xff\x00', b'\xff')
```
這行在 optimized/ultimate 版會變成效能瓶頸（見後面 v22 的平行 destuff）。同時這裡切出 SOS 之後、EOI(`0xFFD9`) 之前的掃描資料。

---

## GPU 端 kernel：`_DECODE_SRC`（L151-273）

### 位元讀取（最樸素的做法）
```cuda
getbit(d, bp, be)   // L154-159
```
**這是 naive 版的核心弱點**：`bp` 是「位元指標」，每讀 1 bit 就做一次 **global memory 讀取** `d[byte]`，再位移取出該 bit。一個區塊要讀成千上萬 bit，等於成千上萬次 global load。

- `receive(d,bp,be,s)`（L160-164）：連讀 s 個 bit 組成數值。
- `extend(v,s)`（L165-169）：JPEG 的有號數還原（若最高位是 0 代表負數，補成負值）。
- `dhuff(...)`（L170-178）：canonical Huffman 解碼 —— 逐位元把 `code` 左移累積，一旦 `code <= maxcode[l]` 就用 `hv[valptr[l] + code - mincode[l]]` 查出符號。

### `decode_tiles(...)`（L180-252）★核心技巧：一個 thread 解一整張 tile
```cuda
int t = blockIdx.x * blockDim.x + threadIdx.x;  // 第 t 個 thread = 第 t 張 tile
if (t >= ntile) return;
```
這就是整個研究的核心命題：**用「跨 tile 的大規模平行」取代「單張 tile 內的平行」**。幾千張 tile 同時各自被一個 thread 序列解碼。

迴圈結構（L197-251）：對每個 MCU(`my,mx`) → 每個分量(`c`) → 該分量的子區塊(`by,bx`)：
1. **清零係數** `F[64]`（L207）。
2. **解 DC**（L209-212）：Huffman 解出大小 `s` → 讀 s bit → `extend` → 累加到 `dcpred[c]`（DC 是差分編碼，要累加前一個區塊的值）→ `F[0] = dcpred·Q[0]`（順便反量化）。
3. **解 AC**（L214-223）：迴圈讀 `(run, size)`；`size==0 && run==15` 是 ZRL（跳 16 個零），`size==0` 否則是 EOB（區塊結束）；否則跳過 `run` 個零、`extend` 出值、`F[zz[k]] = val·Q[k]`（反量化 + 反 zig-zag 一次到位）。
4. **2D IDCT**（L224-230, L232-246）：可分離式 IDCT，先對「列」做一維 IDCT 存進 `tmp`，再對「行」做一維 IDCT。每點是 8 次乘加。
5. **寫像素**（L236-246）：`lrintf(a)+128` 還原位準、夾到 [0,255]。Y 寫進 512×512 平面對應位置；Cb/Cr 寫進各自的 256×256 平面。

### `ycbcr_to_rgb(...)`（L254-271）
**第二個 pass**：把 Y/Cb/Cr 三個平面轉成 RGB。注意 chroma 用 `row/2, col/2` 取值 —— 這就是**最近鄰升採樣**（naive，不做雙線性），也是和 libjpeg 不 bit-exact 的原因之一。標準 BT.601 反矩陣：
```
R = Y + 1.402·Cr
G = Y − 0.344136·Cb − 0.714136·Cr
B = Y + 1.772·Cb
```

### Python 包裝 `GpuJpegDecoder`（L280-346）
- `__init__`：解析共用表 + 樣本 frame，**驗證假設**（512×512、3 分量、4:2:0），把所有表 `cp.asarray` 上傳成 device 陣列。
- `decode_batch(tiles)`（L317-346）：
  - L320 對每張 tile destuff。
  - L321-324 算出每張掃描資料的長度與**前綴和偏移** `byte_off`，再 `b"".join` 串成一條大 blob 一次上傳。
  - L326-327 **注意**：naive 版傳的是 **bit 偏移**（`*8`），因為 `getbit` 用位元指標。
  - L335-340 啟動 decode kernel（threads=64）。
  - L342-345 啟動第二個 kernel 轉 RGB，回傳 `(N,512,512,3)`。

**naive 版的兩大缺點**：(1) 每 bit 一次 global load；(2) 一定要實際生出整個 `(N,512,512,3)` RGB 緩衝（佔大量 VRAM）。optimized 版把這兩點都解掉。

---

# 二、gpu_jpeg_decoder_optimized.py（優化版）

CPU 端解析直接**重用 naive 版**（L38-41 import `parse_shared_tables` 等），所以只看新增的優化。

## 技巧 1：8-bit Huffman 快速查表 LUT

### `_build_lut8(counts, vals)`（L44-58）
為長度 ≤ 8 的 Huffman 碼建一張 256 項的查表：把「碼左補到 8 bit 的所有可能後綴」全部填成 `(長度<<16) | 符號`。
**效果**：絕大多數 JPEG 碼 ≤ 8 bit，解一個符號只要**一次常數記憶體查表**，不用逐位元迴圈。

### `_canonical(counts, vals)`（L61-75）
為長度 > 8 的少數長碼，保留 naive 版的 canonical 表（mincode/maxcode/valptr/huffval），當 LUT 落空時的後備路徑。

## 技巧 2：常數記憶體存放所有熱表

### `_SRC` 開頭的 `__constant__` 宣告（L81-89）
所有「每個 thread 都會狂讀、但全程不變」的表（LUT、canonical、量化表、zig-zag、IDCT 餘弦、分量選擇器）都放進 `__constant__`。
**效果**：常數記憶體有專屬快取且支援 **broadcast**（一個 warp 內同位址只讀一次），比放 global memory 快得多。

## 技巧 3：暫存器 bit-buffer（最關鍵的加速）

### `refill(...)`（L91-98）
```cuda
while (*cnt <= 24){
    unsigned int b = (*pos < end) ? s[*pos] : 0u;
    *buf |= b << (24 - *cnt);
    *cnt += 8; (*pos)++;
}
```
每個 thread 在**暫存器**裡維護一個 32-bit 累積器 `buf` 和有效位元數 `cnt`，一次補一個 **byte**（不是一個 bit）。
**對比 naive**：naive 是「每 1 bit 做 1 次 global load」；這裡是「每 1 byte 做 1 次 global load」，Huffman 熱迴圈的記憶體存取量降到約 **1/8**。`pos` 是 **byte 指標**（不是 bit 指標）—— 這是和 naive 版傳偏移方式不同之處（ultimate 版 L104-106 特別註解提醒不要乘 8）。

### `dhuff(...)`（L99-117）
1. `refill` 補滿 buffer。
2. 取最高 8 bit `buf>>24` 去查 LUT（L102-104）：命中（`len>0`）就消耗 `len` 個 bit（`buf<<=len; cnt-=len`）直接回傳符號 —— **快路徑**。
3. 沒命中（長碼）才走 canonical 逐位元後備（L105-116）。

### `receive_ext(...)`（L118-126）
從 buffer 頂端取 `sz` 個 bit，做 JPEG 有號數還原。一樣只在暫存器上位移，無 global 存取。

## 技巧 4：DC-only 區塊跳過 IDCT

### L168-190
若整個區塊**沒有任何 AC 係數**（`had_ac==false`，常見於平滑/背景區），IDCT 結果是一個常數，不必做 1024 次乘加的完整 IDCT：
```cuda
float t0 = c_cs[0] * F[0];
int p = (int)lrintf(c_cs[0] * t0) + 128;   // 兩次「分開的」浮點乘
```
**重要細節（L164-167 註解）**：必須**精確複製完整 IDCT 的浮點運算順序** —— 寫成 `a0*(a0*F[0])` 兩次四捨五入的乘法，而**不能**寫成 `0.125*F[0]`，因為 float32 下 `a0*a0 ≠ 0.125`，否則就會和完整路徑差 1 LSB、破壞 bit-exact。

## 技巧 5：融合 color+luma+count，完全不生 RGB 緩衝（省 VRAM 的關鍵）

### `count_tiles(...)`（L241-273）★
一個 kernel 同時做完：YCbCr→RGB→整數亮度→門檻判斷→**每張 tile 的白/黑像素計數**，**中間完全不寫出 RGB 緩衝**。
- L244-245：每個 block 用 shared memory `sw[256]/sb[256]` 做區塊內歸約。
- 設計巧思（L238-240 註解）：一個 block = 連續 256 個像素，因為 `512*512 % 256 == 0`，這 256 個像素**保證落在同一張 tile** → 可以安全地用 shared-memory reduce，每個 block 對該 tile 只做**一次 atomicAdd**。
- L250-261：算出 R/G/B → 整數亮度 `(R*19595+G*38470+B*7471+32768)>>16`（**和 PIL `convert('L')` 完全一致**）→ 比門檻得到白/黑各 0/1。
- L262-272：shared memory 樹狀歸約 → thread 0 用 `atomicAdd` 累加到該 tile 的計數。

**效果**：naive 要實際生出 `(N,512,512,3)` 的 RGB（巨大 VRAM）再第二趟篩選；這裡完全不生 RGB → **VRAM 大降**（這正是 3GB 顯卡能跑大 batch 的原因）、記憶體流量也少一趟。

### `to_rgb`（L223-236）
保留一個會真的輸出 RGB 的 kernel，只用於**驗證 / 和 naive 對拍**。

## Python 包裝 `GpuJpegDecoderOptimized`（L294-415）
- `_detect_device()`（L278-291）：抓顯卡名稱、compute capability、SM 數、可用 VRAM（給呼叫端調 batch 用）。
- `__init__`（L297-343）：同樣驗證 4:2:0 假設；`_raw_huff` 重抽每個 Huffman slot 的 `(counts,vals)`，同時建 **LUT + canonical 兩套**；用 `_set_const` 把所有表寫進常數記憶體。
- `_set_const`（L346-349）：透過 `get_global` 拿到 `__constant__` 變數位址，用 `cp.ndarray(..., memptr=ptr)` 包起來再 `.set()` 上傳。
- `_upload`（L376-382）：destuff + 前綴和偏移 + 串成 blob，回傳 device 上的 `(scan, 起始偏移, 結束偏移)` —— 注意這裡是 **byte 偏移**（配合 register bit-buffer）。
- `_decode_planes`（L384-394）：配置 Y/Cb/Cr 平面，啟動 `decode_tiles`。
- `decode_batch`（L396-403）：解碼 + `to_rgb`（驗證用）。
- `count_batch`（L405-415）★：解碼 + `count_tiles`，直接回傳每張 tile 的 `(white[N], black[N])`，**不經過 RGB**。這是 loader 實際走的路徑。

---

# 三、gpu_jpeg_decoder_ultimate.py（終極版）

繼承 `GpuJpegDecoderOptimized`（L43），**kernel 完全不變、輸出 bit-identical**。新增的全是「把主機端工作和傳輸藏到 GPU 計算背後」的排程層（來自 CUPY_RESEARCH_PLAN.md 的 v4/v5/v7/v9）。

## 技巧 6：Pinned（鎖頁）主機記憶體

### `_pinned(nbytes, dtype)`（L36-40）
用 `cp.cuda.alloc_pinned_memory` 配置**鎖頁主機緩衝**，再用 `np.frombuffer` 包成 numpy 陣列。
**效果**：鎖頁記憶體可做**真正的非同步 DMA**（`cudaMemcpyAsync`），不必先從 pageable 記憶體做一次 OS 反彈拷貝，PCIe 頻寬也更高。

## 技巧 7：預配置、可重用的雙緩衝 buffer pool

### `_make_slot(...)`（L56-73）
每個「slot」是一整組預先配好的緩衝：一條 **non-blocking CUDA stream**、pinned 的 `h_scan`、device 的 `d_scan/d_bs/d_be`、Y/Cb/Cr 平面、white/black 計數、以及 pinned 的結果回讀緩衝 `h_white/h_black`。
**效果**：所有緩衝**只配置一次、跨 batch 重用**，消除每個 batch 反覆 malloc/free 的開銷（v7 memory pool 的概念）。預設 2 個 slot 做雙緩衝。

### `_grow_scan(...)`（L75-79）
萬一某 batch 的掃描資料超過預留容量，放大到 1.5 倍重配（很少發生）。

## 技巧 8：雙緩衝 stream pipeline（submit/fetch 非同步）

### `submit(si, tiles, ...)`（L82-125）★ 非阻塞
在 slot `si` 的 stream 上排入「解碼 + 計數」，**立刻回傳**：
1. **CPU 段**（L94-101）：destuff + 算偏移 + 把掃描資料填進 pinned `h_scan`。這段 CPU 工作會**和其他 slot 上正在跑的 GPU 工作重疊**。
2. **GPU 段**（L103-125，全在 `with st:` 這條 stream 上非同步）：
   - pinned→device 非同步上傳 `d_scan/d_bs/d_be`。
   - 啟動 `decode_tiles`。
   - `white/black` 清零、啟動 `count_tiles`。
   - device→pinned **非同步回讀**計數。
   - L104-106 註解再次提醒：偏移是 **byte** 偏移（因為 optimized kernel 的 `refill` 用 `s[pos]` 逐 byte 索引）。

### `fetch(si)`（L127-133）★ 阻塞
等該 slot 的 stream 跑完，回傳 `(white, black)`。

**整體流程**（由 v18/v19 loader 驅動）：主迴圈交錯呼叫 `submit(k)` 與 `fetch(k-1)` → 當 GPU 在解 batch k 時，CPU 同時在讀/destuff/上傳 batch k+1 → **計算與傳輸完全重疊**。

## 技巧 9：刻意不用混合精度（fp16）
L19-21 註解：IDCT 的浮點運算**必須維持 fp32** 才能和 CPU 基準的 kept set bit-identical；fp16 IDCT 會把臨界 patch 推過篩選門檻，所以這裡**故意不採用 v6 的 fp16**。

---

# 技巧總表

| # | 技巧 | 出現於 | 解決的問題 |
|---|------|--------|-----------|
| 1 | **跨 tile 大規模平行**（1 thread = 1 tile） | naive 起 | 用平行 thread 數取代逐張序列解碼 |
| 2 | **共用 JPEG 表 + abbreviated 串流** | 全部 | tile 只帶熵資料，表只解析一次 |
| 3 | **byte de-stuffing** (`FF00→FF`) | 全部 | 還原 JPEG 填充位元組 |
| 4 | **預算 IDCT 餘弦表 / zig-zag 表** | 全部 | kernel 內不算三角函數 |
| 5 | **canonical Huffman 解碼** | 全部 | 標準 Annex C 解碼 |
| 6 | **暫存器 bit-buffer + byte refill** | optimized | global load 由「每 bit」降到「每 byte」(~1/8) |
| 7 | **8-bit Huffman LUT** | optimized | 短碼一次查表，不逐位元迴圈 |
| 8 | **`__constant__` 常數記憶體** | optimized | 熱表 broadcast 快取讀取 |
| 9 | **DC-only 區塊跳過 IDCT** | optimized | 平滑區省掉 1024 次乘加（需保浮點運算順序） |
| 10 | **融合 color+luma+count，不生 RGB** | optimized | VRAM 大降、少一趟記憶體流量（3GB 卡能跑大 batch） |
| 11 | **shared-memory 區塊歸約 + 單次 atomicAdd** | optimized | 每 tile 計數只一次原子操作 |
| 12 | **呼叫端手動指定 batch size** | optimized | 依顯卡 VRAM 調（5090/4060/1060） |
| 13 | **pinned（鎖頁）主機記憶體** | ultimate | 真非同步 DMA、更高 PCIe 頻寬 |
| 14 | **預配置可重用 buffer pool（slots）** | ultimate | 消除每 batch malloc/free 開銷 |
| 15 | **雙緩衝 stream pipeline（submit/fetch）** | ultimate | CPU 讀取/傳輸與 GPU 計算重疊 |
| 16 | **刻意維持 fp32（不用 fp16）** | ultimate | 保持與 CPU 基準 bit-identical |

> 一句話總結三版差異：**naive** 證明「跨 tile 平行」可行；**optimized** 把每個 thread 的解碼變快、並融合計數省掉 RGB 緩衝；**ultimate** 在 kernel 不變的前提下，用 pinned 記憶體 + 雙 stream pipeline 把主機工作與傳輸藏到 GPU 計算背後。
