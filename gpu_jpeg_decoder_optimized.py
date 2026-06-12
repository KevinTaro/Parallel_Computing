"""
gpu_jpeg_decoder_optimized.py

OPTIMIZED hand-written CUDA baseline-JPEG decoder (no nvJPEG, no codec library).
Same algorithm and (bit-for-bit) same output as ``gpu_jpeg_decoder.py``, but
tuned to extract the card's throughput, and it AUTO-ADAPTS to the GPU it finds
(tested target envelope: RTX 5090 32 GB, RTX 4060 8 GB, GTX 1060 3 GB).

What changed vs the naive decoder
---------------------------------
1. **Register bit-buffer.** The naive ``getbit`` did one *global* load per bit.
   Here each thread keeps a 32-bit accumulator in registers and refills a byte
   at a time -> ~1 load per byte instead of per bit (the Huffman hot loop).
2. **8-bit Huffman LUT.** Codes <=8 bits (the overwhelming majority) decode in a
   single constant-memory lookup; only longer codes hit the canonical fallback.
3. **Constant memory.** All hot tables (LUT, canonical Huffman, quant, zig-zag,
   IDCT cosines, component selectors) live in ``__constant__`` -> broadcast reads.
4. **Fused color+luma+count.** Instead of writing a full (N,512,512,3) RGB buffer
   to global memory and filtering it in a second pass, one kernel converts
   YCbCr->RGB, computes the PIL luma, and block-reduces white/black pixel counts
   per tile. No RGB buffer is ever materialised -> far less VRAM (this is what
   lets a 3 GB card run large batches) and less memory traffic.
5. **Device auto-tuning.** Batch size is derived from *free* VRAM so the same
   code saturates a 32 GB 5090 and still fits a 3 GB 1060.

Output remains NOT bit-exact with libjpeg (float IDCT, nearest chroma upsample),
but it is identical to the naive decoder, so callers' kept sets are unchanged.
"""
import struct
from typing import List, Tuple

import cupy as cp
import numpy as np

from gpu_jpeg_decoder import (  # reuse the verified CPU-side parsing
    _ZIGZAG, _idct_cos_table, destuff_tile_scan, parse_shared_tables,
    parse_tile_frame,
)


def _build_lut8(counts: List[int], vals: List[int]) -> List[int]:
    """8-bit fast LUT: prefix -> (length<<16)|symbol for codes of length <=8."""
    lut = [0] * 256
    code, k = 0, 0
    for L in range(1, 17):
        for _ in range(counts[L - 1]):
            sym = vals[k]
            if L <= 8:
                pref = code << (8 - L)
                for j in range(1 << (8 - L)):
                    lut[pref + j] = (L << 16) | sym
            code += 1
            k += 1
        code <<= 1
    return lut


def _canonical(counts: List[int], vals: List[int]):
    """mincode/maxcode/valptr (len 17) + padded huffval, for the >8-bit path."""
    codes = [0] * sum(counts)
    code, k = 0, 0
    for L in range(1, 17):
        for _ in range(counts[L - 1]):
            codes[k] = code; code += 1; k += 1
        code <<= 1
    mincode = [0] * 17; maxcode = [-1] * 17; valptr = [0] * 17
    k = 0
    for L in range(1, 17):
        if counts[L - 1] > 0:
            valptr[L] = k; mincode[L] = codes[k]
            maxcode[L] = codes[k + counts[L - 1] - 1]; k += counts[L - 1]
    return mincode, maxcode, valptr, list(vals) + [0] * (256 - len(vals))


_SRC = r'''
extern "C" {

__constant__ int   c_maxcode[68];   // 4 tables * 17
__constant__ int   c_mincode[68];
__constant__ int   c_valptr[68];
__constant__ int   c_huffval[1024]; // 4 * 256
__constant__ int   c_lut[1024];     // 4 * 256  (len<<16)|sym, len 0 => >8 bits
__constant__ int   c_quant[128];    // 2 * 64 (zig-zag order)
__constant__ int   c_zz[64];
__constant__ float c_cs[64];
__constant__ int   c_cdc[3], c_cac[3], c_cq[3], c_ch[3], c_cv[3];

__device__ __forceinline__ void refill(const unsigned char* s, long* pos, long end,
                                        unsigned int* buf, int* cnt){
    while (*cnt <= 24){
        unsigned int b = (*pos < end) ? s[*pos] : 0u;
        *buf |= b << (24 - *cnt);
        *cnt += 8; (*pos)++;
    }
}
__device__ __forceinline__ int dhuff(int tbl, const unsigned char* s, long* pos,
                                      long end, unsigned int* buf, int* cnt){
    refill(s, pos, end, buf, cnt);
    int e = c_lut[tbl*256 + ((*buf) >> 24)];
    int len = e >> 16;
    if (len > 0){ *buf <<= len; *cnt -= len; return e & 0xffff; }
    int code = 0, used = 0;
    const int *mx = c_maxcode + tbl*17, *mn = c_mincode + tbl*17, *vp = c_valptr + tbl*17;
    #pragma unroll
    for (int l = 1; l <= 16; l++){
        code = (code << 1) | (int)(((*buf) >> (31 - used)) & 1u);
        used++;
        if (mx[l] >= 0 && code <= mx[l]){
            *buf <<= used; *cnt -= used;
            return c_huffval[tbl*256 + vp[l] + code - mn[l]];
        }
    }
    *buf <<= used; *cnt -= used; return 0;
}
__device__ __forceinline__ int receive_ext(const unsigned char* s, long* pos, long end,
                                            unsigned int* buf, int* cnt, int sz){
    if (sz == 0) return 0;
    refill(s, pos, end, buf, cnt);
    int v = (int)((*buf) >> (32 - sz));
    *buf <<= sz; *cnt -= sz;
    if (v < (1 << (sz - 1))) v += (-(1 << sz)) + 1;
    return v;
}

__global__ void decode_tiles(const unsigned char* __restrict__ scan,
                             const long* __restrict__ bstart, const long* __restrict__ bend,
                             unsigned char* Yp, unsigned char* Cbp, unsigned char* Crp,
                             int ntile, int mcux, int mcuy)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntile) return;

    unsigned int buf = 0; int cnt = 0;
    long pos = bstart[t], end = bend[t];
    int dc[3] = {0, 0, 0};
    float F[64], tmp[64];

    for (int my = 0; my < mcuy; my++)
      for (int mx = 0; mx < mcux; mx++)
        for (int c = 0; c < 3; c++){
            int dcs = c_cdc[c], acs = c_cac[c], qbase = c_cq[c]*64;
            int hh = c_ch[c], vv = c_cv[c];
            for (int by = 0; by < vv; by++)
              for (int bx = 0; bx < hh; bx++){
                #pragma unroll
                for (int i = 0; i < 64; i++) F[i] = 0.0f;
                int s = dhuff(dcs, scan, &pos, end, &buf, &cnt);
                dc[c] += receive_ext(scan, &pos, end, &buf, &cnt, s);
                F[0] = (float)(dc[c] * c_quant[qbase]);
                int k = 1; bool had_ac = false;
                while (k < 64){
                    int rs = dhuff(acs, scan, &pos, end, &buf, &cnt);
                    int r = rs >> 4, sa = rs & 15;
                    if (sa == 0){ if (r == 15){ k += 16; continue; } else break; }
                    k += r; if (k > 63) break;
                    int val = receive_ext(scan, &pos, end, &buf, &cnt, sa);
                    F[c_zz[k]] = (float)(val * c_quant[qbase + k]);
                    had_ac = true;
                    k++;
                }
                // DC-only block (no AC) -> IDCT is a constant. Common in smooth/
                // background regions; skips the 1024-MAC IDCT. Must reproduce the
                // full IDCT's exact float op-order -- a0*(a0*F[0]), two rounded
                // multiplies -- because a0*a0 != 0.125 exactly in float32.
                if (!had_ac){
                    float t0 = c_cs[0] * F[0];
                    int p = (int)lrintf(c_cs[0] * t0) + 128;
                    p = p < 0 ? 0 : (p > 255 ? 255 : p);
                    if (c == 0){
                        int r0 = (my*2 + by)*8, c0 = (mx*2 + bx)*8;
                        #pragma unroll
                        for (int y = 0; y < 8; y++)
                          #pragma unroll
                          for (int x = 0; x < 8; x++)
                            Yp[((long)t*512 + r0+y)*512 + c0+x] = (unsigned char)p;
                    } else {
                        int r0 = my*8, c0 = mx*8;
                        #pragma unroll
                        for (int y = 0; y < 8; y++)
                          #pragma unroll
                          for (int x = 0; x < 8; x++){
                            long off = ((long)t*256 + r0+y)*256 + c0+x;
                            if (c == 1) Cbp[off] = (unsigned char)p; else Crp[off] = (unsigned char)p;
                          }
                    }
                    continue;
                }
                #pragma unroll
                for (int v = 0; v < 8; v++)
                  #pragma unroll
                  for (int x = 0; x < 8; x++){
                    float a = 0.f;
                    #pragma unroll
                    for (int u = 0; u < 8; u++) a += c_cs[u*8+x] * F[v*8+u];
                    tmp[v*8+x] = a;
                  }
                #pragma unroll
                for (int y = 0; y < 8; y++)
                  #pragma unroll
                  for (int x = 0; x < 8; x++){
                    float a = 0.f;
                    #pragma unroll
                    for (int v = 0; v < 8; v++) a += c_cs[v*8+y] * tmp[v*8+x];
                    int p = (int)lrintf(a) + 128;
                    p = p < 0 ? 0 : (p > 255 ? 255 : p);
                    if (c == 0){
                        int row = (my*2 + by)*8 + y, col = (mx*2 + bx)*8 + x;
                        Yp[((long)t*512 + row)*512 + col] = (unsigned char)p;
                    } else {
                        int row = my*8 + y, col = mx*8 + x;
                        long off = ((long)t*256 + row)*256 + col;
                        if (c == 1) Cbp[off] = (unsigned char)p; else Crp[off] = (unsigned char)p;
                    }
                  }
              }
        }
}

// RGB output path (used for validation / parity with the naive decoder).
__global__ void to_rgb(const unsigned char* Yp, const unsigned char* Cbp,
                       const unsigned char* Crp, unsigned char* rgb, int ntile){
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)ntile * 512 * 512;
    if (idx >= total) return;
    int col = idx % 512; long r0 = idx / 512; int row = r0 % 512; int t = r0 / 512;
    float Y = Yp[idx];
    float Cb = Cbp[((long)t*256 + row/2)*256 + col/2] - 128.f;
    float Cr = Crp[((long)t*256 + row/2)*256 + col/2] - 128.f;
    int R = (int)lrintf(Y + 1.402f*Cr);        R = R<0?0:(R>255?255:R);
    int G = (int)lrintf(Y - 0.344136f*Cb - 0.714136f*Cr); G = G<0?0:(G>255?255:G);
    int B = (int)lrintf(Y + 1.772f*Cb);        B = B<0?0:(B>255?255:B);
    long o = idx*3; rgb[o]=(unsigned char)R; rgb[o+1]=(unsigned char)G; rgb[o+2]=(unsigned char)B;
}

// Fused: YCbCr->RGB->luma + per-tile white/black counts. No RGB buffer.
// One block = 256 consecutive pixels, which (since 512*512 % 256 == 0) always
// lie within a single tile -> shared-reduce, one atomicAdd per block per tile.
__global__ void count_tiles(const unsigned char* Yp, const unsigned char* Cbp,
                            const unsigned char* Crp, int* white, int* black,
                            int ntile, int wt, int bt){
    __shared__ int sw[256];
    __shared__ int sb[256];
    int tid = threadIdx.x;
    long idx = (long)blockIdx.x * blockDim.x + tid;
    long total = (long)ntile * 512 * 512;
    int w = 0, b = 0;
    if (idx < total){
        int col = idx % 512; long r0 = idx / 512; int row = r0 % 512; int t = r0 / 512;
        float Y = Yp[idx];
        float Cb = Cbp[((long)t*256 + row/2)*256 + col/2] - 128.f;
        float Cr = Crp[((long)t*256 + row/2)*256 + col/2] - 128.f;
        int R = (int)lrintf(Y + 1.402f*Cr);        R = R<0?0:(R>255?255:R);
        int G = (int)lrintf(Y - 0.344136f*Cb - 0.714136f*Cr); G = G<0?0:(G>255?255:G);
        int B = (int)lrintf(Y + 1.772f*Cb);        B = B<0?0:(B>255?255:B);
        int gray = (R*19595 + G*38470 + B*7471 + 32768) >> 16;
        w = (gray > wt) ? 1 : 0;
        b = (gray < bt) ? 1 : 0;
    }
    sw[tid] = w; sb[tid] = b;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (tid < s){ sw[tid] += sw[tid+s]; sb[tid] += sb[tid+s]; }
        __syncthreads();
    }
    if (tid == 0){
        int t = (int)(((long)blockIdx.x * blockDim.x) / (512 * 512));
        atomicAdd(&white[t], sw[0]);
        atomicAdd(&black[t], sb[0]);
    }
}
}
'''


def _detect_device() -> dict:
    dev = cp.cuda.Device()
    free, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="ignore")
    return {
        "name": name,
        "cc": dev.compute_capability,          # e.g. '61', '89', '120'
        "sm": props["multiProcessorCount"],
        "free": int(free),
        "total": int(total),
    }


class GpuJpegDecoderOptimized:
    """Optimized custom-CUDA JPEG decoder with GPU auto-tuning."""

    # decoded planes (Y + Cb + Cr) per tile; the fused counter needs no RGB.
    _PLANE_BYTES = 512 * 512 + 2 * 256 * 256        # ~384 KiB
    _SCAN_EST = 96 * 1024                           # generous per-tile scan estimate

    def __init__(self, jpegtables: bytes, sample_tile: bytes,
                 vram_fraction: float = 0.40, decode_threads: int = 32):
        tables = parse_shared_tables(jpegtables)
        frame = parse_tile_frame(sample_tile)
        comps = frame["comps"]
        if frame["width"] != 512 or frame["height"] != 512 or len(comps) != 3:
            raise ValueError("optimized decoder assumes 512x512 / 3-component tiles")
        if not (comps[0]["h"] == 2 and comps[0]["v"] == 2 and
                comps[1]["h"] == 1 and comps[2]["h"] == 1):
            raise ValueError("optimized decoder assumes YCbCr 4:2:0 sampling")

        # Re-derive counts/vals per Huffman slot to build both LUT and canonical.
        raw = self._raw_huff(jpegtables)
        maxc = np.full((4, 17), -1, np.int32); minc = np.zeros((4, 17), np.int32)
        vptr = np.zeros((4, 17), np.int32); hval = np.zeros((4, 256), np.int32)
        lut = np.zeros((4, 256), np.int32)
        for slot, (counts, vals) in raw.items():
            mn, mx, vp, hv = _canonical(counts, vals)
            minc[slot] = mn; maxc[slot] = mx; vptr[slot] = vp; hval[slot] = hv
            lut[slot] = _build_lut8(counts, vals)

        quant = np.zeros((2, 64), np.int32)
        for tq, q in tables["quant"].items():
            quant[tq] = q

        self._mod = cp.RawModule(code=_SRC)
        self._k_decode = self._mod.get_function("decode_tiles")
        self._k_rgb = self._mod.get_function("to_rgb")
        self._k_count = self._mod.get_function("count_tiles")

        # Upload all hot tables into __constant__ memory.
        self._set_const("c_maxcode", maxc.ravel())
        self._set_const("c_mincode", minc.ravel())
        self._set_const("c_valptr", vptr.ravel())
        self._set_const("c_huffval", hval.ravel())
        self._set_const("c_lut", lut.ravel())
        self._set_const("c_quant", quant.ravel())
        self._set_const("c_zz", _ZIGZAG.astype(np.int32))
        self._set_const("c_cs", _idct_cos_table().astype(np.float32))
        self._set_const("c_cdc", np.array([c["td"] for c in comps], np.int32))
        self._set_const("c_cac", np.array([2 + c["ta"] for c in comps], np.int32))
        self._set_const("c_cq", np.array([c["tq"] for c in comps], np.int32))
        self._set_const("c_ch", np.array([c["h"] for c in comps], np.int32))
        self._set_const("c_cv", np.array([c["v"] for c in comps], np.int32))

        self.decode_threads = decode_threads
        self.device = _detect_device()
        self.recommended_batch = self._auto_batch(vram_fraction)

    # -- constant-memory upload -------------------------------------------
    def _set_const(self, name: str, arr: np.ndarray) -> None:
        ptr = self._mod.get_global(name)
        view = cp.ndarray(arr.shape, dtype=arr.dtype, memptr=ptr)
        view.set(np.ascontiguousarray(arr))

    @staticmethod
    def _raw_huff(jpegtables: bytes) -> dict:
        """Return slot -> (counts[16], vals[]) for each DHT in the tables blob."""
        out = {}
        i, n = 2, len(jpegtables)
        while i < n - 1:
            if jpegtables[i] != 0xFF:
                i += 1; continue
            m = jpegtables[i + 1]
            if m in (0xD8, 0xD9):
                i += 2; continue
            seg_len = struct.unpack('>H', jpegtables[i + 2:i + 4])[0]
            body = jpegtables[i + 4:i + 2 + seg_len]
            if m == 0xC4:
                p = 0
                while p < len(body):
                    tc_th = body[p]; p += 1
                    counts = list(body[p:p + 16]); p += 16
                    total = sum(counts)
                    vals = list(body[p:p + total]); p += total
                    out[(tc_th >> 4) * 2 + (tc_th & 0x0F)] = (counts, vals)
            i += 2 + seg_len
        return out

    def _auto_batch(self, frac: float) -> int:
        """Tiles per GPU batch sized to *free* VRAM, clamped to a sane range."""
        per_tile = self._PLANE_BYTES + self._SCAN_EST
        budget = int(self.device["free"] * frac)
        batch = budget // per_tile
        return int(max(128, min(batch, 8192)))

    # -- decoding ----------------------------------------------------------
    def _upload(self, tiles: List[bytes]):
        scans = [destuff_tile_scan(t) for t in tiles]
        lengths = np.fromiter((len(s) for s in scans), dtype=np.int64, count=len(scans))
        off = np.zeros(len(scans) + 1, dtype=np.int64)
        np.cumsum(lengths, out=off[1:])
        blob = np.frombuffer(b"".join(scans), dtype=np.uint8)
        return cp.asarray(blob), cp.asarray(off[:-1]), cp.asarray(off[1:])

    def _decode_planes(self, tiles: List[bytes]):
        n = len(tiles)
        d_scan, d_bs, d_be = self._upload(tiles)
        Yp = cp.empty((n, 512, 512), dtype=cp.uint8)
        Cbp = cp.empty((n, 256, 256), dtype=cp.uint8)
        Crp = cp.empty((n, 256, 256), dtype=cp.uint8)
        thr = self.decode_threads
        self._k_decode(((n + thr - 1) // thr,), (thr,),
                       (d_scan, d_bs, d_be, Yp, Cbp, Crp,
                        np.int32(n), np.int32(32), np.int32(32)))
        return Yp, Cbp, Crp

    def decode_batch(self, tiles: List[bytes]) -> cp.ndarray:
        """Decode -> (N,512,512,3) uint8 RGB (parity path for validation)."""
        n = len(tiles)
        Yp, Cbp, Crp = self._decode_planes(tiles)
        rgb = cp.empty((n, 512, 512, 3), dtype=cp.uint8)
        total = n * 512 * 512
        self._k_rgb(((total + 255) // 256,), (256,), (Yp, Cbp, Crp, rgb, np.int32(n)))
        return rgb

    def count_batch(self, tiles: List[bytes], white_thr: int, black_thr: int):
        """Decode + fused luma-count -> (white[N], black[N]) int64, no RGB buffer."""
        n = len(tiles)
        Yp, Cbp, Crp = self._decode_planes(tiles)
        white = cp.zeros(n, dtype=cp.int32)
        black = cp.zeros(n, dtype=cp.int32)
        total = n * 512 * 512
        self._k_count(((total + 255) // 256,), (256,),
                      (Yp, Cbp, Crp, white, black, np.int32(n),
                       np.int32(white_thr), np.int32(black_thr)))
        return cp.asnumpy(white).astype(np.int64), cp.asnumpy(black).astype(np.int64)
