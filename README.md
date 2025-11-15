# cslrtools2

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/yourusername/python-cslrtools2)

**Comprehensive toolkit for Continuous Sign Language Recognition (CSLR) research**, providing landmark extraction pipelines, dataset management utilities, and PyTorch helpers for sign language video analysis.

---

## 🌟 Features

### 📹 LMPipe - Landmark Extraction Pipeline

Efficient pipeline for extracting pose/hand/face landmarks from sign language videos using MediaPipe.

- **Multiple landmark types**: Pose, hands (left/right/both), face, holistic
- **Parallel processing**: Multi-process execution with progress tracking
- **Flexible output formats**: NPY, NPZ, Zarr, SafeTensors, PyTorch, JSON, CSV
- **Video & image support**: Process videos or image sequences
- **Annotated frame export**: Visualize landmarks with OpenCV, matplotlib, PIL, or torchvision
- **Plugin architecture**: Extensible design for custom estimators

### 📊 SLDataset - Sign Language Dataset Management

Utilities for managing and loading sign language datasets with efficient storage.

- **Zarr-based storage**: Efficient array storage for large datasets
- **PyTorch integration**: Seamless `torch.utils.data.Dataset` compatibility
- **Flexible schema**: Support for videos, landmarks, and custom targets
- **Array loaders**: Built-in support for various formats (NPY, NPZ, SafeTensors, etc.)
- **Plugin system**: Easy integration with specific datasets (e.g., FluentSigners50)

### 🧮 ConvSize - PyTorch Convolution Utilities

Helper functions for calculating convolution output sizes in PyTorch models.

- **Size calculation**: Compute output dimensions for conv/pooling layers
- **Transpose convolution**: Support for deconvolution size calculation
- **Layer tracking**: Utilities for tracking tensor shapes through networks

---

## 📦 Installation

### Basic Installation

```bash
pip install cslrtools2
```

### With MediaPipe Support

For landmark extraction features, install with MediaPipe:

```bash
pip install cslrtools2
pip install --upgrade mediapipe  # Install MediaPipe separately
```

Or using dependency groups (with `uv`):

```bash
uv pip install cslrtools2 --group mediapipe
```

### From Source

```bash
git clone https://github.com/yourusername/python-cslrtools2.git
cd python-cslrtools2
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/python-cslrtools2.git
cd python-cslrtools2
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

### LMPipe - Extract Landmarks from Videos

#### Command Line Interface

```bash
# Extract holistic landmarks (pose + hands + face)
lmpipe mediapipe.holistic input_video.mp4 -o landmarks.npz

# Extract only pose landmarks
lmpipe mediapipe.pose video_directory/ -o pose_data.zarr

# Process with parallel workers
lmpipe mediapipe.both_hands videos/ -o hands.safetensors --workers 4

# Export annotated frames
lmpipe mediapipe.holistic input.mp4 -o output/ --annotated-frames
```

#### Python API

```python
from cslrtools2.lmpipe.interface import LMPipeExecutor
from cslrtools2.plugins.mediapipe.lmpipe.holistic import HolisticEstimator

# Create estimator
estimator = HolisticEstimator(
    static_image_mode=False,
    model_complexity=1,
)

# Execute pipeline
executor = LMPipeExecutor(estimator=estimator)
results = executor.process("input_video.mp4", output_path="landmarks.npz")
```

### SLDataset - Load Sign Language Datasets

```python
from cslrtools2.sldataset import SLDataset
import zarr

# Open a Zarr-based dataset
store = zarr.open("path/to/dataset.zarr", mode="r")
dataset = SLDataset(store)

# Access items
item = dataset[0]
video = item["videos"]["rgb"]        # Video data
landmarks = item["landmarks"]["pose"] # Pose landmarks
target = item["targets"]["label"]     # Label/annotation

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # Your training code
    pass
```

### ConvSize - Calculate Convolution Output Sizes

```python
from cslrtools2.convsize import conv_size, conv_transpose_size
import torch

# Calculate conv2d output size
input_size = torch.tensor([224, 224])
output_size = conv_size(
    size=input_size,
    kernel_size=torch.tensor([3, 3]),
    stride=torch.tensor([2, 2]),
    padding=torch.tensor([1, 1]),
    dilation=torch.tensor([1, 1])
)
print(output_size)  # tensor([112, 112])

# Calculate transpose convolution size
deconv_size = conv_transpose_size(
    size=output_size,
    kernel_size=torch.tensor([3, 3]),
    stride=torch.tensor([2, 2]),
    padding=torch.tensor([1, 1]),
    output_padding=torch.tensor([1, 1]),
    dilation=torch.tensor([1, 1])
)
print(deconv_size)  # tensor([224, 224])
```

---

## 📚 Documentation

### LMPipe Available Plugins

Built-in MediaPipe estimators:

- `mediapipe.pose` - Body pose landmarks (33 points)
- `mediapipe.left_hand` - Left hand landmarks (21 points)
- `mediapipe.right_hand` - Right hand landmarks (21 points)
- `mediapipe.both_hands` - Both hands landmarks (42 points)
- `mediapipe.face` - Face mesh landmarks (478 points)
- `mediapipe.holistic` - Combined pose + hands + face (543 points)

### Output Formats

LMPipe supports multiple output formats:

- **NumPy** (`.npy`, `.npz`) - Standard NumPy arrays
- **Zarr** (`.zarr`) - Chunked, compressed array storage
- **SafeTensors** (`.safetensors`) - Safe PyTorch tensor format
- **PyTorch** (`.pt`, `.pth`) - PyTorch native format
- **JSON** (`.json`) - Human-readable format
- **CSV** (`.csv`) - Tabular format for spreadsheets

### SLDataset Schema

Default dataset structure in Zarr:

```
dataset.zarr/
├── metadata/          # Dataset-level metadata
├── connections/       # Landmark connectivity information
└── items/            # Individual samples
    ├── 0/
    │   ├── videos/   # Video data (e.g., RGB frames)
    │   ├── landmarks/ # Extracted landmarks
    │   └── targets/   # Annotations/labels
    ├── 1/
    └── ...
```

---

## 🔧 Requirements

- **Python**: >= 3.12
- **Core Dependencies**:
  - PyTorch >= 2.0.0
  - NumPy >= 2.0.0
  - OpenCV >= 4.5.0
  - Zarr >= 3.0.0
  - Rich >= 13.0.0 (for CLI progress display)

- **Optional**:
  - MediaPipe >= 0.10.14 (for landmark extraction)
  - matplotlib >= 3.5.0 (for visualization)
  - SafeTensors >= 0.3.0 (for safe tensor storage)

See `pyproject.toml` for complete dependency list.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/python-cslrtools2.git
cd python-cslrtools2

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=cslrtools2 --cov-report=html
```

### Docker Test Environments

Multiple PyTorch+CUDA environments for testing. See [`tests/build/`](tests/build/) for details.

```bash
cd tests/build

# Build and test specific environment
docker compose build pytorch-cu128
docker compose run --rm pytorch-cu128 uv run python tests/build/test_pytorch_cuda.py

# Test all environments
foreach ($env in @('pytorch-cu128', 'pytorch-cu126', 'pytorch-cu130', 'pytorch-cpu')) {
    docker compose run --rm $env uv run python tests/build/test_pytorch_cuda.py
}
```

For more information, see:
- [`tests/build/README.md`](tests/build/README.md) - Quick start guide
- [`tests/build/DOCKER_STRATEGY.md`](tests/build/DOCKER_STRATEGY.md) - Detailed strategy

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 cslrtools2 contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## � Technical Details

### Type System

This project uses modern Python type hints with the following features:

- **Python 3.12+ compatibility** with PEP 585+ generic syntax
- **`from __future__ import annotations`** for cleaner forward references
- **Fully typed codebase** with Pyright type checking
- **Generic classes** using PEP 695 syntax (e.g., `class MyClass[T: Bound]`)

### Code Quality

- ✅ **100% test coverage** for core modules
- ✅ **Zero type errors** from static analysis
- ✅ **Modular architecture** with clear separation of concerns
- ✅ **Comprehensive logging** with structured error handling

For detailed changes, see [CHANGELOG.md](CHANGELOG.md).

---

## �🙏 Acknowledgments

- **MediaPipe**: Google's MediaPipe framework for landmark detection
- **PyTorch**: Deep learning framework
- **Zarr**: Chunked, compressed array storage

---

## 📮 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/python-cslrtools2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/python-cslrtools2/discussions)

---

## 🗺️ Roadmap

- [ ] Comprehensive API documentation (Sphinx)
- [ ] Example notebooks and tutorials
- [ ] Pre-trained model zoo
- [ ] Support for additional landmark detectors
- [ ] Web-based visualization tools
- [ ] Dataset format converters

---

**Note**: This project is in **alpha stage** (v0.1.0). APIs may change in future releases. For production use, please pin to a specific version.
