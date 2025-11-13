# Examples

## Overview

This section provides practical examples for using `cslrtools2` in various scenarios.

## Quick Examples

### Extract Landmarks from Video

```python
from cslrtools2.lmpipe import LMPipeInterface
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipeHolistic

# Initialize estimator
estimator = MediaPipeHolistic()

# Create pipeline
interface = LMPipeInterface(estimator, workers=1)

# Process single video
interface.run_single_video("input.mp4", "output.npz")
```

### Batch Process Videos

```python
from pathlib import Path
from cslrtools2.lmpipe import LMPipeInterface
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipePose

# Initialize estimator
estimator = MediaPipePose()

# Create pipeline with multiple workers
interface = LMPipeInterface(estimator, workers=4)

# Process all videos in directory
interface.run_directory(
    input_dir="videos/",
    output_dir="landmarks/",
    collector_type="npz",
    pattern="*.mp4"
)
```

### Use with PyTorch DataLoader

```python
from torch.utils.data import DataLoader
from cslrtools2.sldataset import ZarrSLDataset

# Load dataset
dataset = ZarrSLDataset.from_zarr("dataset.zarr")

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch in loader:
    videos = batch.videos
    landmarks = batch.landmarks
    targets = batch.targets
    # ... training code
```

### Calculate Convolution Output Size

```python
import torch
from cslrtools2.convsize import conv_size

# Calculate 2D convolution output size
output_size = conv_size(
    size=torch.tensor([224, 224]),
    kernel_size=torch.tensor([3, 3]),
    stride=torch.tensor([2, 2]),
    padding=torch.tensor([1, 1]),
    dilation=torch.tensor([1, 1])
)
print(output_size)  # tensor([112, 112])
```

---

## Detailed Examples

### [Landmark Extraction Pipeline](lmpipe_examples.md)

- Extract pose, hands, and face landmarks
- Process videos in parallel
- Export annotated frames
- Custom output formats

### [Dataset Management](sldataset_examples.md)

- Create Zarr-based datasets
- Load and iterate datasets
- Custom array loaders
- Integration with PyTorch

### [ConvSize Utilities](convsize_examples.md)

- Track output shapes through networks
- Calculate transposed convolution sizes
- Verify network architectures

---

## CLI Examples

### LMPipe

Extract holistic landmarks:

```bash
lmpipe mediapipe.holistic video.mp4 -o landmarks.npz
```

Process directory with 4 workers:

```bash
lmpipe mediapipe.pose videos/ -o output/ --workers 4
```

Extract hands only, save as Zarr:

```bash
lmpipe mediapipe.both_hands video.mp4 -o hands.zarr --output-format zarr
```

---

## Jupyter Notebooks

Coming soon: Interactive Jupyter notebooks demonstrating:

- End-to-end landmark extraction
- Dataset creation and management
- Training a simple CSLR model
- Visualization and analysis

---

## Navigation

- **[← Back to Home](../index.md)**
- **Next**: [LMPipe Examples](lmpipe_examples.md)
