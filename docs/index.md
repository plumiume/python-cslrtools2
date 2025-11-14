# cslrtools2 Documentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/cslrtools2.svg)](https://pypi.org/project/cslrtools2/)

**Comprehensive toolkit for Continuous Sign Language Recognition (CSLR) research**

---

## Overview

`cslrtools2` is a Python toolkit designed for continuous sign language recognition research, providing:

- 📹 **LMPipe**: Landmark extraction pipeline using MediaPipe
- 📊 **SLDataset**: Sign language dataset management with Zarr
- 🧮 **ConvSize**: PyTorch convolution size calculation utilities

---

## Quick Links

- [Installation Guide](installation.md)
- [API Reference](api/index.md)
- [User Guide](guide/index.md)
- [Examples](examples/index.md)
- [GitHub Repository](https://github.com/ikegami-yukino/python-cslrtools2)

---

## Quick Start

### Installation

```bash
pip install cslrtools2
```

### Extract Landmarks from Video

```bash
lmpipe mediapipe.holistic input_video.mp4 -o landmarks.npz
```

### Use in Python

```python
from cslrtools2.lmpipe import LMPipeInterface
from cslrtools2.plugins.mediapipe.lmpipe import MediaPipeHolistic

# Initialize estimator
estimator = MediaPipeHolistic()

# Create pipeline
interface = LMPipeInterface(estimator)

# Process video
interface.run_single_video("video.mp4", "output.npz")
```

---

## Features

### LMPipe - Landmark Extraction Pipeline

- Multiple landmark types (pose, hands, face, holistic)
- Parallel processing with progress tracking
- Multiple output formats (NPY, NPZ, Zarr, SafeTensors, PyTorch, JSON, CSV)
- Video and image sequence support
- Annotated frame visualization

### SLDataset - Dataset Management

- Zarr-based efficient storage
- PyTorch Dataset integration
- Flexible schema for videos, landmarks, and targets
- Plugin system for dataset-specific loaders

### ConvSize - PyTorch Utilities

- Convolution output size calculation
- Transpose convolution support
- Layer shape tracking helpers

---

## Getting Help

- [GitHub Issues](https://github.com/ikegami-yukino/python-cslrtools2/issues)
- [Discussions](https://github.com/ikegami-yukino/python-cslrtools2/discussions)

---

## License

Licensed under the [Apache License 2.0](https://github.com/ikegami-yukino/python-cslrtools2/blob/main/LICENSE)

---

## Navigation

- **Next**: [Installation Guide](installation.md)
- **See Also**: [API Reference](api/index.md) | [Examples](examples/index.md)
