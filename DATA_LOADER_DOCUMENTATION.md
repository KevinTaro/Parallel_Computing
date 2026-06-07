# WSI Sliding Window Data Loader - Technical Documentation

## Overview

The `data_loader.py` module implements a **dynamic sliding window data loader** for Whole Slide Images (WSIs). It efficiently extracts patches from gigapixel-scale medical images without pre-splitting them into individual files, enabling a "zero-I/O overhead" approach suitable for handling very large histopathology images.

---

## Purpose & Use Case

- **Problem**: WSIs can be several gigabytes in size. Pre-processing them into individual patch files requires massive disk space and I/O overhead.
- **Solution**: Dynamically load patches on-the-fly from the original WSI file using OpenSlide, filtering out background/non-tissue regions in real-time.
- **Target Application**: Training deep learning models (e.g., DenseNet121) on histopathology data with automatic tissue detection.

---

## Core Architecture

### Main Class: `WSISlidingWindowDataset`

A PyTorch `Dataset` subclass that implements the standard interface for use with `DataLoader`.

#### Constructor Parameters
```python
WSISlidingWindowDataset(
    wsi_path,              # Path to .tiff/.svs WSI file
    patch_size=1024,       # Height/width of each patch (pixels)
    stride=1024,           # Step size between patches
    transform=None,        # Optional augmentation pipeline
    white_pixel_threshold=230,   # Grayscale value for "white" detection
    black_pixel_threshold=25,    # Grayscale value for "black" detection
    rejection_ratio=0.9    # Ratio threshold for discarding patches
)
```

---

## Technical Implementations

### 1. **OpenSlide Integration**
- Uses `openslide` library to read WSI files without loading entire image into memory
- Reads metadata at initialization: WSI dimensions at level 0 (full resolution)
- Lazy initialization in `__getitem__`: opens slide fresh for each patch request to ensure multiprocessing compatibility

```python
with openslide.OpenSlide(wsi_path) as slide:
    patch = slide.read_region((x, y), 0, (patch_size, patch_size))
```

### 2. **High-Precision Tissue Filtering**
Automatically detects and discards patches that are mostly background (white) or non-tissue (black).

**Algorithm**:
- Convert each patch to grayscale (8-bit)
- Count white pixels: `grayscale > white_pixel_threshold` (default: 230)
- Count black pixels: `grayscale < black_pixel_threshold` (default: 25)
- Discard if either ratio ≥ `rejection_ratio` (default: 0.9 = 90%)

**Example**: If 90% of a patch is white pixels, it's discarded as background.

### 3. **Multiprocessing for Grid Generation**
During dataset initialization, scans all potential patches in parallel:

```python
num_workers = cpu_count()  # Detect available cores
with Pool(processes=num_workers) as pool:
    results = pool.map(process_func, potential_coords)
```

**Why Multiprocessing?**
- Grid generation requires scanning every potential patch location
- I/O bound operation (disk reads via OpenSlide)
- Parallelization dramatically reduces initialization time
- Static method `_process_patch()` designed for worker process serialization

### 4. **Coordinate-Based Indexing**
- Pre-computes valid patch coordinates during `__init__`
- Stores only `(x, y)` coordinates of tissue patches
- Returns both patch tensor and coordinates on access

### 5. **PyTorch Integration**
- Standard `Dataset` interface: `__len__()` and `__getitem__()`
- Compatible with `DataLoader` for batch processing with `num_workers`
- Supports optional transform pipeline (e.g., normalization)
- Returns `Tuple[torch.Tensor, Tuple[int, int]]`: patch and coordinates

### 6. **Grid Sliding Window Logic**
```python
for y in range(0, wsi_height, stride):
    for x in range(0, wsi_width, stride):
        if x + patch_size <= wsi_width and y + patch_size <= wsi_height:
            potential_coords.append((x, y))
```

- Sliding window with configurable `stride` (overlap control)
- Boundary checking: ensures patches don't exceed image dimensions
- Default 1:1 stride-to-patch_size ratio = non-overlapping tiles

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Initialization: WSISlidingWindowDataset(wsi_path, ...)      │
├─────────────────────────────────────────────────────────────┤
│ 1. Read WSI metadata → get dimensions (wsi_width, wsi_height)
│ 2. Generate potential coordinates grid                       
│ 3. Launch multiprocessing pool                              
│ 4. Each worker: Load patch → Analyze tissue (gray + ratios) 
│ 5. Filter: Keep only tissue-containing patches              
│ 6. Store valid (x,y) coordinates → self.coordinates         
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ During Training: dataset[idx]                                │
├─────────────────────────────────────────────────────────────┤
│ 1. Fetch coordinates: (x, y) = self.coordinates[idx]        │
│ 2. Lazy load: Open WSI → read_region((x,y), level=0, size)│
│ 3. Convert RGBA → RGB → Apply transforms                    
│ 4. Return (patch_tensor, (x, y))                            
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Benefit |
|---------|---------|
| **Lazy Loading** | Only loads patches when accessed; minimal memory footprint |
| **Multiprocessing Grid** | Scales with CPU cores; fast initialization |
| **Automatic Tissue Detection** | No manual annotation needed; filters background automatically |
| **Configurable Thresholds** | Adapt to different stain/image quality |
| **Coordinate Tracking** | Know exact position of each patch in slide |
| **PyTorch Compatible** | Drop-in for standard training pipelines |
| **No Pre-processing** | Work directly with original WSI files |

---

## Example Usage

```python
from torchvision import transforms
from torch.utils.data import DataLoader

# Create dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = WSISlidingWindowDataset(
    wsi_path="path/to/slide.tiff",
    patch_size=1024,
    stride=1024,
    transform=transform,
    white_pixel_threshold=230,
    black_pixel_threshold=25,
    rejection_ratio=0.9
)

# Create DataLoader for batching
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0  # Use 0 to avoid pickling issues with OpenSlide
)

# Train loop
for batch_patches, batch_coords in dataloader:
    # batch_patches: (4, 3, 1024, 1024) - 4 RGB patches
    # batch_coords: tuple of x and y tensors
    model_output = model(batch_patches)
```

---

## Performance Characteristics

- **Initialization**: Multiprocessing scans all candidate patches in parallel
- **Per-Patch Load**: ~10-50ms depending on storage I/O speed
- **Memory**: Only active patches in GPU memory; coordinates stored as integers
- **Scalability**: Handles gigapixel WSIs (e.g., 100,000x100,000 pixels)

---

## Configuration for Different Models

**For DenseNet121** (640x640 input):
```python
dataset = WSISlidingWindowDataset(
    wsi_path=path,
    patch_size=640,   # Match model input size
    stride=640,       # Non-overlapping tiles
)
```

**For sliding overlap** (50% overlap):
```python
dataset = WSISlidingWindowDataset(
    wsi_path=path,
    patch_size=1024,
    stride=512,       # 50% stride relative to patch_size
)
```

---

## Summary

This loader bridges the gap between gigapixel medical images and efficient deep learning training by:
1. **Dynamically loading** patches on demand
2. **Intelligently filtering** background/non-tissue regions
3. **Leveraging multiprocessing** for fast initialization
4. **Integrating seamlessly** with PyTorch's standard data pipeline

No pre-processing, no massive disk storage, no I/O bottleneck—just efficient patch-based training from raw WSI files.
