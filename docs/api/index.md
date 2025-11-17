# API Reference

## Overview

`cslrtools2` provides three main subpackages:

- [**lmpipe**](lmpipe.md) - Landmark extraction pipeline
- [**sldataset**](sldataset.md) - Sign language dataset management
- [**convsize**](convsize.md) - PyTorch convolution utilities

## Package Structure

```
cslrtools2/
├── lmpipe/              # Landmark extraction pipeline
│   ├── estimator.py     # Base estimator classes
│   ├── collector/       # Output collectors
│   ├── interface/       # High-level interface
│   └── app/            # CLI application
├── sldataset/          # Dataset management
│   ├── dataset.py      # Dataset classes
│   ├── array_loader.py # Array format loaders
│   └── app/           # CLI application
├── convsize.py         # Convolution size utilities
├── plugins/            # Plugin implementations
│   ├── mediapipe/     # MediaPipe estimators
│   └── fluentsigners50/ # FluentSigners50 dataset
└── typings/           # Type definitions

```

## Quick Reference

### LMPipe

```python
from cslrtools2.lmpipe import LMPipeInterface
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipeHolistic

# Create pipeline
estimator = MediaPipeHolistic()
interface = LMPipeInterface(estimator)

# Process video
interface.run_single_video("video.mp4", "output.npz")
```

[Full LMPipe API Reference →](lmpipe.md)

### SLDataset

```python
from cslrtools2.sldataset import ZarrSLDataset

# Load dataset
dataset = ZarrSLDataset.from_zarr("dataset.zarr")

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8)
```

[Full SLDataset API Reference →](sldataset.md)

### ConvSize

```python
from cslrtools2.convsize import conv_size
import torch

# Calculate output size
output_size = conv_size(
    size=torch.tensor([224, 224]),
    kernel_size=torch.tensor([3, 3]),
    stride=torch.tensor([2, 2]),
    padding=torch.tensor([1, 1]),
    dilation=torch.tensor([1, 1])
)
```

[Full ConvSize API Reference →](convsize.md)

## Detailed References

- [LMPipe API](lmpipe.md) - Landmark extraction pipeline
- [SLDataset API](sldataset.md) - Dataset management utilities
- [ConvSize API](convsize.md) - Convolution size helpers
- [Plugins](plugins.md) - Plugin implementations (MediaPipe, datasets)
- [Type Definitions](types.md) - Type annotations and protocols

## CLI Tools

### lmpipe

Extract landmarks from videos:

```bash
lmpipe mediapipe.holistic video.mp4 -o landmarks.npz
```

[CLI Reference →](cli/lmpipe.md)

### sldataset2

Manage sign language datasets:

```bash
sldataset2 load dataset.zarr
```

[CLI Reference →](cli/sldataset.md)

---

## Navigation

- **[← Back to Home](../index.md)**
- **Next**: [LMPipe API](lmpipe.md)
