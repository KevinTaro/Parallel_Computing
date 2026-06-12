"""
gpu_jpeg_decoder.py

A *naive* baseline-JPEG decoder that runs on the GPU's general CUDA cores --
NO nvJPEG, no codec library, just a hand-written CuPy RawKernel. Its only job is
to answer one research question: *how much does raw parallelism speed up the
decode task?* One CUDA thread decodes one whole tile (Huffman -> dequant ->
IDCT -> plane), and thousands of tiles decode at once. It is deliberately
UNOPTIMIZED (one serial thread per tile, float IDCT); the point is the
parallel-across-tiles speedup, not a fast decoder.

Scope / assumptions (true for these Philips WSIs, asserted at parse time):
  - baseline sequential JPEG, 8-bit, 3 components, YCbCr 4:2:0 (Y=2x2, C=1x1)
  - shared DQT/DHT tables (from the TIFF JPEGTables tag); each tile carries its
    own SOF0 + SOS + entropy data
  - no restart markers (DRI = 0)

NOT bit-exact with libjpeg: the float IDCT and nearest-neighbour chroma upsample
differ by a few LSB, so a handful of patches near the rejection threshold may
flip versus the CPU baseline. The callers measure and report that delta.
"""
import struct
from typing import Dict, List, Tuple

import cupy as cp
import numpy as np

# Standard JPEG zig-zag: zigzag index k -> natural (row-major) index.
_ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
], dtype=np.int32)


def _idct_cos_table() -> np.ndarray:
    """cs[k*8+i] = a(k) * cos((2i+1)k*pi/16), a(0)=sqrt(1/8), a(k>0)=1/2."""
    cs = np.zeros((8, 8), dtype=np.float32)
    for k in range(8):
        a = np.sqrt(1.0 / 8.0) if k == 0 else 0.5
        for i in range(8):
            cs[k, i] = a * np.cos((2 * i + 1) * k * np.pi / 16.0)
    return cs.reshape(-1)


# --------------------------------------------------------------------------
# CPU-side JPEG header parsing (small, serial -- not the work we parallelise)
# --------------------------------------------------------------------------
def _build_huffman(counts: List[int], vals: List[int]):
    """Annex C/F canonical decode tables: mincode/maxcode/valptr (len 17)."""
    codes = [0] * sum(counts)
    code, k = 0, 0
    for l in range(1, 17):
        for _ in range(counts[l - 1]):
            codes[k] = code
            code += 1
            k += 1
        code <<= 1
    mincode = [0] * 17
    maxcode = [-1] * 17
    valptr = [0] * 17
    k = 0
    for l in range(1, 17):
        if counts[l - 1] > 0:
            valptr[l] = k
            mincode[l] = codes[k]
            maxcode[l] = codes[k + counts[l - 1] - 1]
            k += counts[l - 1]
    hv = list(vals) + [0] * (256 - len(vals))
    return mincode, maxcode, valptr, hv


def parse_shared_tables(jpegtables: bytes) -> dict:
    """Parse DQT (2) + DHT (DC0,DC1,AC0,AC1) from the shared JPEGTables blob."""
    quant = {}                      # tq -> 64 ints (zig-zag order)
    huff = {}                       # slot (cls*2+id) -> (mincode,maxcode,valptr,hv)
    i = 2                           # skip SOI
    n = len(jpegtables)
    while i < n - 1:
        if jpegtables[i] != 0xFF:
            i += 1
            continue
        m = jpegtables[i + 1]
        if m in (0xD8, 0xD9):
            i += 2
            continue
        seg_len = struct.unpack('>H', jpegtables[i + 2:i + 4])[0]
        body = jpegtables[i + 4:i + 2 + seg_len]
        if m == 0xDB:               # DQT (may pack multiple tables)
            p = 0
            while p < len(body):
                pq_tq = body[p]; p += 1
                tq = pq_tq & 0x0F
                if (pq_tq >> 4) == 0:
                    quant[tq] = list(body[p:p + 64]); p += 64
                else:               # 16-bit quant (not expected here)
                    quant[tq] = list(struct.unpack('>64H', body[p:p + 128])); p += 128
        elif m == 0xC4:             # DHT (may pack multiple tables)
            p = 0
            while p < len(body):
                tc_th = body[p]; p += 1
                counts = list(body[p:p + 16]); p += 16
                total = sum(counts)
                vals = list(body[p:p + total]); p += total
                slot = (tc_th >> 4) * 2 + (tc_th & 0x0F)   # DC0,DC1,AC0,AC1 -> 0..3
                huff[slot] = _build_huffman(counts, vals)
        i += 2 + seg_len
    return {"quant": quant, "huff": huff}


def parse_tile_frame(tile: bytes) -> dict:
    """Parse SOF0 (component sampling/tables) from one tile -- same for all."""
    p = tile.find(b'\xff\xc0')
    if p < 0:
        raise ValueError("no SOF0 in tile")
    prec, h, w, ncomp = tile[p + 4], struct.unpack('>H', tile[p + 5:p + 7])[0], \
        struct.unpack('>H', tile[p + 7:p + 9])[0], tile[p + 9]
    comps = []
    q = p + 10
    for _ in range(ncomp):
        cid = tile[q]; hv = tile[q + 1]; tq = tile[q + 2]; q += 3
        comps.append({"id": cid, "h": hv >> 4, "v": hv & 0x0F, "tq": tq})
    # SOS gives the DC/AC table selectors per component
    s = tile.find(b'\xff\xda')
    ns = tile[s + 4]
    sel = {}
    q = s + 5
    for _ in range(ns):
        cs = tile[q]; tdta = tile[q + 1]; q += 2
        sel[cs] = {"td": tdta >> 4, "ta": tdta & 0x0F}
    for c in comps:
        c.update(sel[c["id"]])
    return {"prec": prec, "height": h, "width": w, "comps": comps}


def destuff_tile_scan(tile: bytes) -> bytes:
    """Extract + byte-de-stuff a tile's entropy-coded scan data (no restart)."""
    s = tile.find(b'\xff\xda')
    seg_len = struct.unpack('>H', tile[s + 2:s + 4])[0]
    start = s + 2 + seg_len
    end = tile.find(b'\xff\xd9', start)          # EOI
    if end < 0:
        end = len(tile)
    return tile[start:end].replace(b'\xff\x00', b'\xff')


# --------------------------------------------------------------------------
# The CUDA kernel: one thread per tile.  Huffman -> dequant -> IDCT -> planes.
# --------------------------------------------------------------------------
_DECODE_SRC = r'''
extern "C" {

__device__ __forceinline__ int getbit(const unsigned char* d, long* bp, long be){
    if (*bp >= be) return 0;
    long byte = (*bp) >> 3; int sh = 7 - (int)((*bp) & 7);
    (*bp)++;
    return (d[byte] >> sh) & 1;
}
__device__ __forceinline__ int receive(const unsigned char* d, long* bp, long be, int s){
    int v = 0;
    for (int i = 0; i < s; i++) v = (v << 1) | getbit(d, bp, be);
    return v;
}
__device__ __forceinline__ int extend(int v, int s){
    if (s == 0) return 0;
    if (v < (1 << (s - 1))) v += (-(1 << s)) + 1;
    return v;
}
__device__ int dhuff(const unsigned char* d, long* bp, long be,
                     const int* mn, const int* mx, const int* vp, const int* hv){
    int code = 0;
    for (int l = 1; l <= 16; l++){
        code = (code << 1) | getbit(d, bp, be);
        if (mx[l] >= 0 && code <= mx[l]) return hv[vp[l] + code - mn[l]];
    }
    return 0;
}

__global__ void decode_tiles(
    const unsigned char* scan, const long* bit_start, const long* bit_end,
    const int* mincode, const int* maxcode, const int* valptr, const int* huffval,
    const int* quant,                              // [2*64] zig-zag order
    const int* zz, const float* cs,
    const int* comp_dc, const int* comp_ac, const int* comp_q,
    const int* comp_h, const int* comp_v,
    unsigned char* Yp, unsigned char* Cbp, unsigned char* Crp,
    int ntile, int mcux, int mcuy)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntile) return;

    long bp = bit_start[t], be = bit_end[t];
    int dcpred[3] = {0, 0, 0};
    float F[64], tmp[64];

    for (int my = 0; my < mcuy; my++){
      for (int mx = 0; mx < mcux; mx++){
        for (int c = 0; c < 3; c++){
            int ds = comp_dc[c], as = comp_ac[c], qi = comp_q[c];
            int hh = comp_h[c], vv = comp_v[c];
            const int *mnD=mincode+ds*17,*mxD=maxcode+ds*17,*vpD=valptr+ds*17,*hvD=huffval+ds*256;
            const int *mnA=mincode+as*17,*mxA=maxcode+as*17,*vpA=valptr+as*17,*hvA=huffval+as*256;
            const int *Q = quant + qi*64;
            for (int by = 0; by < vv; by++){
              for (int bx = 0; bx < hh; bx++){
                for (int i = 0; i < 64; i++) F[i] = 0.0f;
                // DC
                int s = dhuff(scan,&bp,be,mnD,mxD,vpD,hvD);
                int diff = extend(receive(scan,&bp,be,s), s);
                dcpred[c] += diff;
                F[0] = (float)(dcpred[c] * Q[0]);
                // AC
                int k = 1;
                while (k < 64){
                    int rs = dhuff(scan,&bp,be,mnA,mxA,vpA,hvA);
                    int r = rs >> 4, sa = rs & 15;
                    if (sa == 0){ if (r == 15){ k += 16; continue; } else break; }
                    k += r; if (k > 63) break;
                    int val = extend(receive(scan,&bp,be,sa), sa);
                    F[zz[k]] = (float)(val * Q[k]);
                    k++;
                }
                // 2D IDCT (separable, orthonormal): tmp = cols, out = rows
                for (int v = 0; v < 8; v++)
                  for (int x = 0; x < 8; x++){
                    float a = 0.f;
                    for (int u = 0; u < 8; u++) a += cs[u*8+x] * F[v*8+u];
                    tmp[v*8+x] = a;
                  }
                // write pixels
                for (int y = 0; y < 8; y++)
                  for (int x = 0; x < 8; x++){
                    float a = 0.f;
                    for (int v = 0; v < 8; v++) a += cs[v*8+y] * tmp[v*8+x];
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
      }
    }
}

__global__ void ycbcr_to_rgb(const unsigned char* Yp, const unsigned char* Cbp,
                             const unsigned char* Crp, unsigned char* rgb,
                             int ntile)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)ntile * 512 * 512;
    if (idx >= total) return;
    int col = idx % 512; long r0 = idx / 512; int row = r0 % 512; int t = r0 / 512;
    float Y  = Yp[idx];
    float Cb = Cbp[((long)t*256 + row/2)*256 + col/2] - 128.0f;
    float Cr = Crp[((long)t*256 + row/2)*256 + col/2] - 128.0f;
    int R = (int)lrintf(Y + 1.402f*Cr);
    int G = (int)lrintf(Y - 0.344136f*Cb - 0.714136f*Cr);
    int B = (int)lrintf(Y + 1.772f*Cb);
    R = R<0?0:(R>255?255:R); G = G<0?0:(G>255?255:G); B = B<0?0:(B>255?255:B);
    long o = idx*3;
    rgb[o] = (unsigned char)R; rgb[o+1] = (unsigned char)G; rgb[o+2] = (unsigned char)B;
}
}
'''

_module = cp.RawModule(code=_DECODE_SRC)
_k_decode = _module.get_function("decode_tiles")
_k_color = _module.get_function("ycbcr_to_rgb")


class GpuJpegDecoder:
    """Holds the shared tables on-device; decodes batches of tiles to RGB."""

    def __init__(self, jpegtables: bytes, sample_tile: bytes):
        tables = parse_shared_tables(jpegtables)
        frame = parse_tile_frame(sample_tile)
        comps = frame["comps"]
        if frame["width"] != 512 or frame["height"] != 512 or len(comps) != 3:
            raise ValueError("custom CUDA decoder assumes 512x512 / 3-component tiles")
        if not (comps[0]["h"] == 2 and comps[0]["v"] == 2 and
                comps[1]["h"] == 1 and comps[2]["h"] == 1):
            raise ValueError("custom CUDA decoder assumes YCbCr 4:2:0 sampling")

        # Flatten Huffman tables (slots 0..3 = DC0,DC1,AC0,AC1) to device arrays.
        mn = np.zeros((4, 17), np.int32); mx = np.full((4, 17), -1, np.int32)
        vp = np.zeros((4, 17), np.int32); hv = np.zeros((4, 256), np.int32)
        for slot, (mnc, mxc, vpc, hvc) in tables["huff"].items():
            mn[slot] = mnc; mx[slot] = mxc; vp[slot] = vpc; hv[slot] = hvc
        self.d_mn, self.d_mx = cp.asarray(mn.ravel()), cp.asarray(mx.ravel())
        self.d_vp, self.d_hv = cp.asarray(vp.ravel()), cp.asarray(hv.ravel())

        quant = np.zeros((2, 64), np.int32)
        for tq, q in tables["quant"].items():
            quant[tq] = q
        self.d_quant = cp.asarray(quant.ravel())
        self.d_zz = cp.asarray(_ZIGZAG)
        self.d_cs = cp.asarray(_idct_cos_table())

        # Per-component selectors: DC slot = td, AC slot = 2+ta, quant = tq.
        cdc = np.array([c["td"] for c in comps], np.int32)
        cac = np.array([2 + c["ta"] for c in comps], np.int32)
        cq = np.array([c["tq"] for c in comps], np.int32)
        ch = np.array([c["h"] for c in comps], np.int32)
        cv = np.array([c["v"] for c in comps], np.int32)
        self.d_cdc, self.d_cac, self.d_cq = cp.asarray(cdc), cp.asarray(cac), cp.asarray(cq)
        self.d_ch, self.d_cv = cp.asarray(ch), cp.asarray(cv)

    def decode_batch(self, tiles: List[bytes]) -> cp.ndarray:
        """Decode a list of raw JPEG tile byte-strings -> (N,512,512,3) uint8 RGB."""
        n = len(tiles)
        scans = [destuff_tile_scan(t) for t in tiles]
        lengths = np.array([len(s) for s in scans], dtype=np.int64)
        byte_off = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=byte_off[1:])
        blob = np.frombuffer(b"".join(scans), dtype=np.uint8)
        d_scan = cp.asarray(blob)
        d_bstart = cp.asarray(byte_off[:n] * 8)
        d_bend = cp.asarray(byte_off[1:] * 8)

        Yp = cp.empty((n, 512, 512), dtype=cp.uint8)
        Cbp = cp.empty((n, 256, 256), dtype=cp.uint8)
        Crp = cp.empty((n, 256, 256), dtype=cp.uint8)

        threads = 64
        blocks = (n + threads - 1) // threads
        _k_decode((blocks,), (threads,), (
            d_scan, d_bstart, d_bend,
            self.d_mn, self.d_mx, self.d_vp, self.d_hv,
            self.d_quant, self.d_zz, self.d_cs,
            self.d_cdc, self.d_cac, self.d_cq, self.d_ch, self.d_cv,
            Yp, Cbp, Crp, np.int32(n), np.int32(32), np.int32(32)))

        rgb = cp.empty((n, 512, 512, 3), dtype=cp.uint8)
        total = n * 512 * 512
        tpb = 256
        _k_color(((total + tpb - 1) // tpb,), (tpb,), (Yp, Cbp, Crp, rgb, np.int32(n)))
        return rgb
