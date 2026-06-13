"""
data_loader_v31_dec_v9_combined.py

v31: GPU-decode translation of the v9 "all optimization layers combined" concept
=================================================================================
v9 (data_loader_v9_ultimate_gpu.py) was the culmination of the v1..v8 CuPy
series: it combined every optimisation layer into a single class --
batching (v2), hybrid fallback (v3), pinned memory (v4), async streams (v5),
fp16 (v6), memory budget (v7), and parallel reads (v8). The philosophy:
"turn everything on at once and let them compound".

Translation to the GPU-decode world:
    The equivalent "everything combined" implementation in the GPU-decode series is
    ``data_loader_v22_cuda_parallel_destuff.py`` (v22).

    v22 combines:
        - v2 (batching): large tile batches per GPU call
        - v4 (pinned memory): GpuJpegDecoderUltimate pre-allocates pinned h_scan
        - v5 (async double-buffer): GpuJpegDecoderUltimate two slots / two streams
        - v7 (memory pool): device + pinned buffers pre-allocated once, reused
        - v8 (parallel reads): ThreadPoolExecutor with os.pread, off the main thread
        - v8+ (destuff parallel): EACH reader thread also calls destuff_tile_scan,
          so the destuff (v22's key innovation) runs parallel and off the critical path

    v22 represents the same "combine everything" philosophy as v9, applied to the
    custom CUDA decode pipeline. v31 is a thin re-export to complete the v1..v10
    mapping.

    Architecture summary of v22 (== v31 concept):
        reader threads: os.pread(tid) -> destuff_tile_scan(raw) -> ready_q
        main thread:    fetch from ready_q -> submit_scans (async H2D + decode + count)
                        -> fetch results (non-blocking until ready)
        Result: all host-side work (read + destuff) is off the critical path.
                The main loop is bound only by GPU decode throughput.

See ``data_loader_v22_cuda_parallel_destuff.py`` for full implementation details.
"""
from data_loader_v22_cuda_parallel_destuff import WSISlidingWindowDataset  # noqa: F401


def run_test(wsi_path: str = "data/S114-80954A-Her2(3+).tiff"):
    print("==== v31 GPU-decode v9-combined (== v22 parallel destuff) - Test Run ====")
    print("Note: v31 is a thin wrapper over v22 (WSISlidingWindowDataset).")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v31/v22] kept={len(ds)} grid_time={ds.grid_creation_time:.2f}s")
    print("v31 concept: v9 'all optimisations combined' -> v22 parallel read+destuff "
          "pipeline.")
    print("Layer stack: batch (v2) + pinned (v4) + async (v5) + "
          "mempool (v7) + threaded-reads+destuff (v8+).")


if __name__ == '__main__':
    run_test()
