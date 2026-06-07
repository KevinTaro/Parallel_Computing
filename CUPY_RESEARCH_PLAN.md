# CuPy Integration Research Plan
## Performance Testing & Validation Framework

---

## 1. Research Objectives

### 📌 Why Mono-Core Baseline Matters

The **mono-core baseline (v0a)** is crucial because it:
- Shows the **true computational bottleneck** without any parallelization
- Reveals **per-patch processing cost** - the atomic unit of work
- Establishes why CPU multiprocessing (v0b) helps or doesn't
- Provides a reality check: GPU benefit must overcome transfer + compute overhead
- Enables fair comparison: all GPU versions must beat sequential baseline by significant margin

**Research Principle**: Before optimizing with GPU, understand the CPU baseline bottleneck.

### Primary Goals
- **Establish baseline performance** across mono-core CPU (v0a), multi-core CPU (v0b), and GPU (v1-v7)
- **Explore** different CuPy integration strategies for WSI patch filtering
- **Measure** performance improvements/trade-offs systematically
- **Validate** numerical correctness across all 10 implementations
- **Document** findings and learnings for educational purposes

### Core Research Questions
1. **CPU Parallelization**: How much does multiprocessing (v0b) improve over mono-core (v0a)?
2. **GPU Benefit**: At what point does GPU (v1-v7) outperform CPU parallelization (v0b)?
3. **Optimal Strategy**: Which GPU implementation (v1-v7) is best for realistic WSI workloads?
4. **Transfer Overhead**: When does GPU memory transfer kill performance gains?
5. **Scaling**: How do speedups change with patch size, batch size, WSI resolution?

---

## 2. Implementation Strategies

### 🔴 **CPU BASELINES** (Educational & Reference)

#### Strategy v0: Ultra-Basic Baseline (Pure Concept)
**File**: `data_loader_v0_ultra_basic.py`
- **SIMPLEST possible implementation** - pedagogical baseline only
- No memory optimization
- No error handling
- No resource cleanup
- Shows bare-minimum algorithm
- **NOT guaranteed to run** - for understanding only
- Expected: Educational clarity, slowest if runs

#### Strategy v0a: Mono-Core Baseline (Single-Threaded CPU)
**File**: `data_loader_v0a_mono_baseline.py` (Source: `data_loader_mono.py`)
- **Pure sequential processing** - single CPU core only
- Single-threaded OpenSlide reading with proper resource management
- No multiprocessing, no parallelism
- Simple for-loop patch filtering with error handling
- Serves as **functional baseline** to show CPU bottleneck
- Expected: slowest but reliable, highest initialization time

#### Strategy v0b: Multi-Core CPU Baseline (Multiprocessing)
**File**: `data_loader_v0b_multi_baseline.py` (Source: `data_loader.py`)
- Original implementation with **multiprocessing on CPU**
- Uses `Pool(cpu_count())` for parallel filtering
- NumPy on CPU only
- Reference for CPU parallelization speedup
- Expected: faster than v0a, but still CPU-limited

### 🔵 **GPU IMPLEMENTATIONS** (CuPy Accelerated)

#### Strategy v1: CuPy Full Filtering
**File**: `data_loader_v1_cupy_full.py`
- Replace ALL NumPy operations with CuPy in `_process_patch()`
- Grayscale conversion on GPU
- Pixel ratio calculations on GPU
- Minimal data transfer
- Shows GPU benefit for pure computation

#### Strategy v2: CuPy Batch Processing
**File**: `data_loader_v2_cupy_batch.py`
- Load multiple patches together (batched)
- Process entire batch on GPU
- Reduce function call overhead
- Test different batch sizes
- Amortizes GPU kernel launch cost

#### Strategy v3: Hybrid (Smart Transfer)
**File**: `data_loader_v3_cupy_hybrid.py`
- Selective GPU usage: only transfer when batch size exceeds threshold
- CPU for small batches, GPU for large batches
- Optimize for latency-sensitive operations
- Real-world optimal strategy

#### Strategy v4: Pinned Memory Optimization
**File**: `data_loader_v4_cupy_pinned_memory.py`
- Use CUDA pinned (page-locked) memory for faster CPU↔GPU transfers
- Pre-allocate GPU memory pools
- Optimize data transfer pipeline
- Minimizes transfer bottleneck

#### Strategy v5: Async GPU Processing
**File**: `data_loader_v5_cupy_async.py`
- Asynchronous GPU operations
- Stream-based processing for overlapping compute and transfer
- Multiple CUDA streams for parallelism
- Hidden transfer latency

#### Strategy v6: Mixed Precision
**File**: `data_loader_v6_cupy_mixed_precision.py`
- Use float32 for input, float16 for intermediate calculations
- Reduce memory bandwidth requirements
- Validate precision requirements for filtering
- Shows trade-off: speed vs accuracy

#### Strategy v7: Memory-Optimized
**File**: `data_loader_v7_cupy_memory_optimized.py`
- In-place operations where possible
- Chunked processing to minimize peak memory
- Memory pool management
- For memory-constrained GPUs

---

## 2.1 Architecture Comparison: v0 → v0a → v0b → v1-v7

```
┌──────────────────────────────────────────────────────────────────┐
│ v0: ULTRA-BASIC BASELINE (Pure Concept)                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  slide = openslide.OpenSlide(path)                               │
│  for y in range(0, height, stride):                              │
│    for x in range(0, width, stride):                             │
│      patch = slide.read_region((x,y), 0, (1024, 1024))          │
│      gray = np.array(patch.convert('L'))                        │
│      white = np.sum(gray > threshold) / total                    │
│      if white < ratio: keep_coordinate((x,y))                    │
│  slide.close()                                                   │
│                                                                   │
│  ❌ No error handling, no resource management                    │
│  ❌ Resources may leak on crash                                  │
│  ✅ Shows CORE ALGORITHM clearly                                 │
│  🎓 Educational: understand workflow first                       │
│  ⚠️  May NOT run reliably                                        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ v0a: MONO-CORE BASELINE (Single-Threaded, Functional)            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  with openslide.OpenSlide(path) as slide:  # Proper cleanup      │
│    for x, y in potential_coords:                                 │
│      try:                                                        │
│        patch = slide.read_region((x,y), 0, (1024, 1024))        │
│        patch_gray = np.array(patch.convert('L'))                │
│        white_ratio = np.sum(patch_gray > threshold) / total      │
│        if white_ratio >= rejection_ratio: discard                │
│      except Exception: handle_error                              │
│                                                                   │
│  ✅ Proper error handling & cleanup (context manager)            │
│  ✅ Reliable baseline for performance comparison                 │
│  ⚠️  Bottleneck: Sequential I/O + sequential computation         │
│  📊 Reference: 1.0x speedup (baseline)                           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ v0b: MULTI-CORE CPU BASELINE (Multiprocessing, Optimized)        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  with Pool(processes=cpu_count()) as pool:                       │
│    results = pool.map(_process_patch, potential_coords)          │
│                                                                   │
│  _process_patch (worker):                                        │
│    with openslide.OpenSlide(path) as slide:                      │
│      patch = slide.read_region((x,y), 0, (1024, 1024))          │
│      patch_gray = np.array(patch.convert('L'))                  │
│      white_ratio = np.sum(patch_gray > threshold) / total        │
│                                                                   │
│  ✅ Parallelizes I/O and computation across cores                │
│  ✅ Each worker has own OpenSlide instance                       │
│  📊 Expected speedup: ~4-8x over v0a (N_cores dependent)        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ v1-v7: GPU IMPLEMENTATIONS (CuPy)                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  patch = slide.read_region((x,y), 0, (1024, 1024))   # CPU       │
│  patch_gpu = cp.asarray(patch)                       # Transfer  │
│  patch_gray = cp.mean(patch_gpu, axis=2)             # GPU       │
│  white_ratio = cp.sum(patch_gray > threshold) / N    # GPU       │
│  result_cpu = cp.asnumpy(white_ratio)                # Transfer  │
│                                                                   │
│  💡 Strategy differences (v1-v7):                                 │
│    - Batch size optimization (v2)                                │
│    - Hybrid CPU/GPU selection (v3)                              │
│    - Transfer optimization (v4, v5)                             │
│    - Precision trade-offs (v6)                                  │
│    - Memory management (v7)                                     │
│                                                                   │
│  ✅ Leverages GPU compute (100x+ FLOPS)                         │
│  ⚠️  Transfer overhead can dominate for small patches            │
│  📊 Break-even at batch_size > 50-200 patches (expected)        │
└──────────────────────────────────────────────────────────────────┘
```

### Key Comparison Insights
| Aspect | v0 (Ultra) | v0a (Mono) | v0b (Multi) | v1-v7 (GPU) |
|--------|-----------|-----------|-----------|-----------|
| **Code Clarity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Error Handling** | ❌ None | ✅ Full | ✅ Full | ✅ Full |
| **Resource Cleanup** | ❌ Manual | ✅ Auto | ✅ Auto | ✅ Auto |
| **Reliability** | ❌ Unsafe | ✅ Safe | ✅ Safe | ✅ Safe |
| **Parallelization** | None | None | Processes | CUDA Streams |
| **Memory Model** | Shared | Shared | Separate | Pinned + GPU |
| **I/O Bottleneck** | Critical | Critical | Reduced | Depends |
| **Transfer Overhead** | N/A | N/A | N/A | Yes (large) |
| **Overhead** | Minimal | Minimal | Serialization | GPU init |
| **Best for** | Learning | Baseline | Real CPU | Large batch |
| **Expected Speedup** | N/A | 1.0x | ~4-8x | ~2-10x |

---

## 3. Test Framework Structure

### Core Testing Infrastructure

#### File: `test_performance_framework.py`
```
Classes:
- PerformanceMetrics: collect timing, memory, accuracy data
- CuPyTester: unified interface to test any implementation
- ValidationChecker: verify numerical correctness
- BenchmarkRunner: execute controlled experiments
```

#### File: `benchmark_runner.py`
```
Functions:
- run_single_benchmark(implementation, wsi_path, iterations)
- run_comparative_benchmark(implementations, wsi_path, iterations)
- generate_performance_report(results)
- plot_results(results)
```

#### File: `numerical_validation.py`
```
Functions:
- validate_against_baseline(v_baseline, v_test, tolerance=1e-6)
- compare_filtering_results(coords_baseline, coords_test)
- check_numerical_stability(patch_data_baseline, patch_data_cupy)
```

---

## 4. Test Execution Plan

### Phase 0: Baseline Establishment & Educational Validation (CRITICAL)
**File**: `test_00_baseline_comparison.py`

```
OBJECTIVE: Establish baselines from educational → functional → optimized
           Prove v0b > v0a > v1-v7 has measurable gains

Part A: Educational Baseline (v0)
  1. Code review: Is v0 clear and understandable?
  2. Algorithm verification: Does it show core concept?
  3. Workflow documentation: Is it educational?
  ✅ SUCCESS: v0 used for teaching algorithm to new developers

Part B: Functional Baselines (v0a, v0b)
  For each version (v0a, v0b):
    1. Load single patch 100 times
    2. Measure: mean time, std dev, min, max
    3. Record CPU utilization
    4. Validate: same patches filtered as v0a
    
  VALIDATION:
    - Confirm v0a is slowest (1.0x reference)
    - Confirm v0b is ~Nx faster (N = cpu_count, expect 4-8x)
    - Confirm both are reliable (no crashes/errors)

Part C: GPU Baselines (v1-v7)
  1. Load single patch 100 times
  2. Measure: mean time, std dev, GPU memory
  3. Record GPU utilization
  4. Validate: same filtering results as v0a
  
  VALIDATION:
    - Confirm GPU versions don't crash
    - Confirm all produce identical filtering results
    - Identify fastest GPU variant
```

**Output**: 
- Speedup hierarchy: v0a (baseline 1.0x) → v0b (CPU ~5x) → vX_best (GPU ~2-10x)
- CPU/GPU utilization profiles
- Code clarity assessment for v0
- Early indicator of which GPU version is most promising
- Validation that algorithm correctness is preserved

---

### Phase 1: Single-Patch Testing
**File**: `test_01_single_patch.py`

```
For each implementation version (v0-v7):
  1. Load a single patch
  2. Time the pixel ratio calculation
  3. Validate against baseline (v0)
  4. Record: time, memory, accuracy
  5. Test with varying patch sizes (256, 512, 1024, 2048)
```

**Metrics**:
- Execution time (ms)
- Peak GPU memory (MB)
- Accuracy (relative error vs baseline)
- Speedup ratio (baseline_time / version_time)

---

### Phase 2: Batch Filtering Testing
**File**: `test_02_batch_filtering.py`

```
For each implementation version:
  1. Test grid filtering with increasing patch counts
  2. Batch sizes: 10, 50, 100, 500, 1000
  3. Time the entire _create_grid() operation
  4. Memory profiling throughout execution
  5. Validate coordinate output matches baseline
```

**Metrics**:
- Grid creation time (seconds)
- Memory usage over time
- Filtering accuracy (% patches correctly classified)
- Throughput (patches/second)

---

### Phase 3: End-to-End DataLoader Testing
**File**: `test_03_end_to_end_dataloader.py`

```
For each implementation version:
  1. Initialize dataset with real WSI
  2. Load 100 random patches via __getitem__()
  3. Profile each patch load operation
  4. Test with DataLoader batching
  5. Measure batch load time with varying batch sizes
```

**Metrics**:
- Per-patch load time (ms)
- Batch load time (ms) for different batch sizes
- Total dataset initialization time
- Cumulative memory usage

---

### Phase 4: GPU Memory Analysis
**File**: `test_04_memory_profiling.py`

```
For each implementation:
  1. Profile GPU memory allocation patterns
  2. Track memory fragmentation
  3. Test memory cleanup/deallocation
  4. Simulate long training runs (10000+ iterations)
  5. Check for memory leaks
```

**Metrics**:
- Peak memory (MB)
- Memory fragmentation %
- Memory per-patch (MB)
- Sustained memory usage (OOM risk?)

---

### Phase 5: Scalability Testing
**File**: `test_05_scalability.py`

```
For different hardware configurations:
  1. Test on different GPU types (if available)
  2. Vary patch sizes: 256px, 512px, 1024px, 2048px
  3. Vary stride: full overlap vs no overlap
  4. Different WSI resolutions
```

**Metrics**:
- Performance scaling curve
- GPU utilization %
- Bottleneck identification

---

### Phase 6: Integration Testing
**File**: `test_06_integration_training.py`

```
Simulate actual training loop:
  1. Create dataset with v0 and best-performing vX
  2. Load 1000 patches each
  3. Feed to mock model training
  4. Measure wall-clock time
  5. Measure GPU utilization during training
```

**Metrics**:
- Training iteration time
- GPU utilization
- Data loading bottleneck % of total time

---

## 4.5 Comparative Analysis Strategy

### Performance Comparison Methodology

After all phases complete, run:

**File**: `comparative_analysis.py`

```python
# Build performance comparison matrix
results = {
    'v0a_mono': {...},          # Baseline reference
    'v0b_multi': {...},         # CPU parallelization baseline
    'v1_cupy_full': {...},
    'v2_cupy_batch': {...},
    ...
    'v7_cupy_memory_opt': {...}
}

# Key analyses:
1. Speedup vs v0a (mono baseline)
2. Speedup vs v0b (CPU baseline)
3. Efficiency ratio: speedup / overhead
4. Memory scaling: peak memory vs batch size
5. Accuracy comparison: L1/L2 norm vs baseline
6. Break-even analysis: when does GPU pay for itself?
```

### Visualization Outputs

```
1. speedup_comparison.png
   - Bar chart: all versions vs v0a
   - Shows v0b intermediate performance

2. scaling_curves.png
   - Line plots: speedup vs batch_size
   - Separate lines for v0a, v0b, v1-v7
   - Reveals break-even points

3. memory_profile.png
   - Stacked area: v0a, v0b memory over time
   - GPU memory for v1-v7
   - Peak memory comparison

4. accuracy_comparison.png
   - Numerical error vs v0a for all versions
   - Precision degradation for v6

5. bottleneck_analysis.png
   - Time breakdown: I/O, compute, transfer
   - Shows where GPU helps most
```

### Report Sections (in RESEARCH_RESULTS.md)

```
## Baseline Performance
- v0a (mono) as absolute reference: X seconds for 1000 patches
- v0b (multi) shows CPU parallelization value: 4-8x speedup
- Validates multiprocessing overhead is justified

## GPU Performance Summary
- Which GPU version wins? (probably v3 or v4)
- How much better than v0b?
- At what batch size?

## Trade-off Analysis
- Speed vs memory (all versions)
- Speed vs accuracy (v6 mixed precision)
- Speed vs implementation complexity

## Recommendations
- For small systems: use v0a or v0b
- For GPUs: use vX (recommended)
- For different workloads: use vY
```

---

## 5. Validation & Correctness

### Correctness Checks
**File**: `validation_suite.py`

```python
Tests for each version:
1. Grayscale conversion correctness
   - Compare pixel values: NumPy vs CuPy
   - Tolerance: exact match (uint8 range)

2. Ratio calculation correctness
   - White ratio, black ratio
   - Tolerance: 1e-6 (floating point)

3. Filtering logic consistency
   - Same patches selected/rejected
   - Coordinate ordering

4. Transform pipeline correctness
   - Tensor shape and values
   - Normalize transform output
```

### Edge Cases
```
- Empty WSI (no tissue patches)
- Fully white/black WSI
- Patch size = WSI size
- Single-pixel patches
- Very large patches (> 4GB in memory)
```

---

## 6. Output & Reporting

### Generated Test Files

#### Data Loader Implementations (11 versions)
```
CPU BASELINES (Educational & Reference):
  data_loader_v0_ultra_basic.py         (Pure concept, no optimization)
  data_loader_v0a_mono_baseline.py      (from data_loader_mono.py)
  data_loader_v0b_multi_baseline.py     (from data_loader.py)

GPU IMPLEMENTATIONS (CuPy Accelerated):
  data_loader_v1_cupy_full.py
  data_loader_v2_cupy_batch.py
  data_loader_v3_cupy_hybrid.py
  data_loader_v4_cupy_pinned_memory.py
  data_loader_v5_cupy_async.py
  data_loader_v6_cupy_mixed_precision.py
  data_loader_v7_cupy_memory_optimized.py
```

#### Testing Infrastructure (3 files)
```
test_performance_framework.py
benchmark_runner.py
numerical_validation.py
```

#### Test Suites (6 phases)
```
test_01_single_patch.py
test_02_batch_filtering.py
test_03_end_to_end_dataloader.py
test_04_memory_profiling.py
test_05_scalability.py
test_06_integration_training.py
```

#### Validation & Reporting
```
validation_suite.py
RESEARCH_RESULTS.md
```

### Report Structure

#### Report File: `RESEARCH_RESULTS.md`
```
1. Executive Summary
   - Key findings
   - Best performing strategy
   - Trade-offs discovered

2. Implementation Details
   - Each strategy explained
   - Design choices and rationale
   - Code patterns used

3. Performance Results
   - Comparative tables
   - Performance graphs
   - Speedup analysis

4. Memory Analysis
   - GPU memory usage patterns
   - Transfer overhead breakdown
   - Peak memory requirements

5. Correctness Validation
   - Numerical accuracy results
   - Edge case handling
   - Precision analysis

6. Conclusions
   - When to use each approach
   - Recommendations
   - Future optimization directions

7. Appendix
   - Raw benchmark data
   - System specifications
   - Code snippets
```

---

## 7. Hardware & Dependencies

### Requirements
```
- NVIDIA GPU (RTX 3060+, or similar)
- CUDA 11.x or 12.x
- CuPy (installed for your CUDA version)
- NumPy, PyTorch, OpenSlide
- Memory Profiler (memory_profiler)
- Timing: time, timeit modules
```

### Installation
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# Testing dependencies
pip install memory-profiler
pip install matplotlib  # for plotting results
```

---

## 8. Research Milestones

### Week 1: Setup & Baseline
- [ ] Create v0_baseline (reference)
- [ ] Build performance_framework.py
- [ ] Establish testing infrastructure

### Week 2: Basic CuPy Implementations
- [ ] Create v1_cupy_full
- [ ] Create v2_cupy_batch
- [ ] Run test_01 and test_02

### Week 3: Advanced Optimizations
- [ ] Create v3-v7 variations
- [ ] Run test_03, test_04
- [ ] Initial performance comparison

### Week 4: Validation & Analysis
- [ ] Complete validation_suite.py
- [ ] Run test_05, test_06
- [ ] Generate RESEARCH_RESULTS.md

### Week 5: Documentation & Conclusions
- [ ] Polish implementations
- [ ] Create detailed report
- [ ] Write recommendations

---

## 9. Key Research Questions to Answer

1. **CPU Parallelism vs GPU**: How much speedup does multiprocessing (v0b) give over mono-core (v0a)?
   - Validates that WSI filtering is CPU-bound before GPU consideration
   
2. **GPU Break-Even Point**: At what patch count/batch size does GPU (v1-v7) outperform CPU (v0b)?
   - Quantifies GPU transfer overhead vs computation gains
   
3. **Optimal GPU Strategy**: Which CuPy implementation (v1-v7) is best for typical WSI workloads?
   - Practical recommendation for real deployments
   
4. **Memory Trade-off**: Is GPU memory a bottleneck for large batches?
   - Peak memory usage and scaling limits
   
5. **Precision vs Performance**: Do float16 calculations (v6) affect filtering accuracy?
   - Trade-off validation for production use
   
6. **Scaling Characteristics**: How do all strategies scale with patch size and WSI dimensions?
   - Understanding algorithmic complexity

7. **End-to-End Impact**: What's the actual wall-clock improvement in realistic training?
   - Measures practical benefit beyond theoretical speedup

---

## 10. Expected Findings (Hypotheses)

| Strategy | Expected Behavior | Speedup |
|----------|-------------------|---------|
| **v0 (Ultra Basic)** | Pure concept; educational only; may crash | N/A (not runnable) |
| **v0a (Mono CPU)** | Single-core baseline, shows bottleneck | 1.0x (reference) |
| **v0b (Multi CPU)** | Multiprocessing speedup across cores | 4-8x vs v0a |
| **v1 (Full CuPy)** | GPU transfer overhead vs small gains | 1.5-2.5x vs v0b |
| **v2 (Batch CuPy)** | Amortizes kernel overhead; better scaling | 3-5x vs v0b |
| **v3 (Hybrid)** | **Best for real-world** mixed workloads | 3-6x vs v0b |
| **v4 (Pinned Memory)** | Better sustained transfers; reduces bottleneck | 4-7x vs v0b |
| **v5 (Async)** | Marginal gains from pipelining | 2.5-5x vs v0b |
| **v6 (Mixed Precision)** | Faster but may affect filtering accuracy | 5-8x vs v0b |
| **v7 (Memory Opt)** | Better for memory-constrained systems | 2-4x vs v0b |

**Key Insights**: 
- v0 (ultra-basic) shows algorithm clearly but not production-ready
- v0a (mono) is reliable baseline: **1.0x reference**
- v0b (multi) shows CPU parallelization value: **~5x typical speedup**
- GPU becomes beneficial at **batch size > 50-200 patches** (estimated)

---

## 11. Success Criteria

✅ **Research Success if**:
- **Educational foundation**: v0 (ultra-basic) clearly shows the core algorithm
- **Baselines established**: v0a (mono) ↔ v0b (multi) performance with clear speedup ratio
- All 11 implementations are properly characterized:
  - v0: Educational clarity ✅
  - v0a/v0b: Functional & reliable ✅
  - v1-v7: Functionally correct (pass validation_suite) ✅
- Performance differences are quantified and explained across all versions
- Clear understanding of **where GPU becomes beneficial** vs CPU parallelization
- Report clearly documents trade-offs, use cases, and recommendations
- Code is reproducible, well-documented, and educational at each level

### Success Metrics
1. ✅ v0 effectively teaches the algorithm (code review)
2. ✅ v0b speedup ≥ 3x over v0a (confirms CPU parallelization value)
3. ✅ Best GPU version ≥ 2x over v0b (justifies GPU complexity)
4. ✅ Numerical accuracy within 1e-6 across all functional versions
5. ✅ Memory profiles complete for all strategies
6. ✅ Scaling analysis: performance vs batch_size, patch_size, WSI_size

### Bonus Findings (High Value)
- Discovery of true bottleneck (I/O bandwidth vs compute)
- Optimization insights specific to WSI filtering characteristics
- GPU portability: performance across different GPU models
- **Decision matrix**: when to use v0/v0a/v0b/vX for different scenarios
- Unexpected discoveries or edge cases where one strategy wins/loses

---

This plan provides a **structured, educational research framework** to explore CuPy integration strategies, measure their effectiveness, and document the learning process. The multiple test versions allow you to understand *where* GPU helps and *why*.
