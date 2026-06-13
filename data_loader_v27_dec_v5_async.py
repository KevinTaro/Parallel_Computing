"""
data_loader_v27_dec_v5_async.py

v27: GPU-decode translation of the v5 "async double-buffer streams" concept
============================================================================
v5 (data_loader_v5_cupy_async.py) introduced double-buffered asynchronous CUDA
streams: while the GPU processes batch k, the CPU prepares batch k+1 in a
second stream buffer, so neither the CPU nor the GPU ever idles waiting for the
other.

Translation to the GPU-decode world:
    This concept, applied to the custom GPU JPEG decoder, is EXACTLY v18:
    ``data_loader_v18_cuda_ultimate_mono.py``.

    v18 implements the double-buffered pipeline via ``GpuJpegDecoderUltimate``'s
    two slots and two CUDA streams:
        - ``decoder.submit(slot, tiles, wt, bt)`` -- non-blocking: launches async
          H2D DMA + decode kernel + count kernel on stream ``slot``.
        - ``decoder.fetch(slot)`` -- blocks until the GPU work on ``slot`` is done
          and returns the white/black counts.
    The main loop interleaves submit(k) with fetch(k-1), so the GPU decode of
    batch k runs in parallel with the CPU reading tiles for batch k+1.

    v27 is a thin re-export of v18 to complete the v1..v10 mapping. It adds the
    ``V27AsyncDataset`` alias to make the versioning explicit.

    To compare async vs non-async in the GPU-decode world:
        v24 (non-async batched)  vs  v27/v18 (double-buffered async)
    Expected: v27 matches or beats v24 because the GPU never stalls waiting for
    the CPU to prepare the next batch.

See ``data_loader_v18_cuda_ultimate_mono.py`` for full implementation details.
"""
from data_loader_v18_cuda_ultimate_mono import WSISlidingWindowDataset  # noqa: F401


class V27AsyncDataset(WSISlidingWindowDataset):
    """
    v27 alias for WSISlidingWindowDataset from v18.

    Inherits the full double-buffered async CUDA pipeline. No behaviour change;
    this class exists only to make the v5-concept -> v27 mapping explicit for
    benchmarking and documentation purposes.
    """
    pass


def run_test(wsi_path: str = "data/S114-80954A-Her2(3+).tiff"):
    print("==== v27 GPU-decode v5-async (== v18 double-buffer) - Test Run ====")
    print("Note: v27 is a thin wrapper over v18 (WSISlidingWindowDataset).")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v27/v18] kept={len(ds)} kernel_time={ds.kernel_time:.3f}s "
          f"read_time={ds.read_time:.3f}s grid_time={ds.grid_creation_time:.2f}s")
    print("v27 concept: v5 async double-buffer streams -> v18 GpuJpegDecoderUltimate "
          "pipeline.")


if __name__ == '__main__':
    run_test()
