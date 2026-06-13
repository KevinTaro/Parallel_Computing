"""
data_loader_v32_dec_v10_pipeline.py

v32: GPU-decode translation of the v10 "producer/consumer parallel IO pipeline"
=================================================================================
v10 (data_loader_v10_parallel_io_gpu.py) introduced a proper producer/consumer
architecture for the IO pipeline: a dedicated producer thread drives a pool of
reader threads, stocking a bounded queue with pre-fetched batches. The consumer
(main thread) only does GPU work -- it never blocks on IO. This decouples IO
from compute completely, allowing them to run at full throughput simultaneously.

Translation to the GPU-decode world:
    The exact same producer/consumer architecture, applied to the custom CUDA JPEG
    decoder, is ``data_loader_v21_cuda_ultimate_pipeline.py`` (v21).

    v21 producer/consumer design:
        producer thread:
            - Drives a ``ThreadPoolExecutor`` of readers via ``pool.submit``
              (NOT pool.map -- submit is non-blocking, so the pool stays stocked
               across batch boundaries without draining between batches)
            - Each reader issues ``os.pread(fd, bytecounts[tid], offsets[tid])``
              (GIL-releasing, positionally atomic, no lock needed on shared fd)
            - Completed batches are placed on ``ready_q`` (bounded by a Semaphore
              for backpressure)
        main thread (consumer):
            - Pops completed raw-tile batches from ``ready_q``
            - Calls ``decoder.submit(slot, tiles, wt, bt)`` (async H2D + GPU decode)
            - Calls ``decoder.fetch(slot)`` (blocks until GPU done, returns counts)
            - Never dispatches pool.map; never blocks on IO
        Result: IO and GPU decode run fully in parallel. On cold storage / NFS,
                where reads are slow, v21 keeps the GPU fed without stalling.

    v32 is a thin re-export to complete the v1..v10 -> v23..v32 mapping.

    The key distinction from v31 (v9/v22):
        v21/v32: producer/consumer, reads-only parallel (no parallel destuff)
        v22/v31: producer/consumer, reads AND destuff parallel (v22's innovation)
        v21 is the pure v10 architecture; v22 adds the extra layer on top.

See ``data_loader_v21_cuda_ultimate_pipeline.py`` for full implementation details.
"""
from data_loader_v21_cuda_ultimate_pipeline import WSISlidingWindowDataset  # noqa: F401


def run_test(wsi_path: str = "data/S114-80954A-Her2(3+).tiff"):
    print("==== v32 GPU-decode v10-pipeline (== v21 producer/consumer) - Test Run ====")
    print("Note: v32 is a thin wrapper over v21 (WSISlidingWindowDataset).")
    ds = WSISlidingWindowDataset(wsi_path=wsi_path, verbose=True)
    print(f"[v32/v21] kept={len(ds)} grid_time={ds.grid_creation_time:.2f}s")
    print("v32 concept: v10 producer/consumer IO pipeline -> v21 prefetch pipeline.")
    print("Architecture: dedicated producer thread + reader pool -> bounded ready_q "
          "-> consumer (main thread, GPU-only work).")


if __name__ == '__main__':
    run_test()
