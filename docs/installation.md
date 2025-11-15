# Installation Guide

## Requirements

- Python 3.12 or higher
- `uv` (recommended) or `pip` package manager

## Recommended: Using `uv`

[`uv`](https://github.com/astral-sh/uv) is the recommended package manager for faster dependency resolution and installation:

### Install `uv`

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip
pip install uv
```

### Install cslrtools2

```bash
# Core package
uv pip install cslrtools2

# With MediaPipe support
uv pip install cslrtools2 --group mediapipe

# With development tools
uv pip install cslrtools2 --group dev

# All optional dependencies
uv pip install cslrtools2 --group mediapipe --group dev
```

## Using pip

### Basic Installation

Install the core package:

```bash
pip install cslrtools2
```

## Optional Dependencies

### MediaPipe Support

For landmark extraction features using MediaPipe:

```bash
pip install mediapipe>=0.10.14
```

### Development Tools

For development and testing:

```bash
pip install pytest pytest-cov pyright flake8 black sphinx sphinx-rtd-theme
```

## PyTorch with CUDA Support

For optimal performance with GPU acceleration:

### Prerequisites

- **CUDA Toolkit**: 11.8 or 12.8 (check with `nvidia-smi`)
- **NVIDIA GPU**: Compatible with CUDA

### Installation

```bash
# CUDA 12.8 (recommended for newer GPUs)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 11.8 (for older GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only version
pip install torch torchvision
```

See [PyTorch Getting Started](https://pytorch.org/get-started/locally/) for platform-specific instructions.

## From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ikegami-yukino/python-cslrtools2.git
cd python-cslrtools2

# Using uv (recommended)
uv pip install -e .

# With optional dependencies
uv pip install -e . --group mediapipe --group dev

# Using pip
pip install -e .
pip install -e ".[dev,mediapipe]"  # Note: this syntax doesn't work with current setup
```

## Dependencies

### Core Dependencies

- `torch >= 2.0.0` - PyTorch for tensor operations
- `numpy >= 2.0.0` - Numerical computing
- `opencv-python >= 4.5.0` - Video and image processing
- `zarr >= 3.0.0` - Array storage
- `safetensors >= 0.3.0` - Tensor serialization
- `matplotlib >= 3.5.0` - Visualization
- `rich >= 13.0.0` - Terminal UI
- `clipar` - CLI argument parsing
- `loky >= 3.0.0` - Parallel processing
- `requests >= 2.28.0` - HTTP requests

### Optional Dependencies

- `mediapipe` - MediaPipe framework for landmark extraction

## Verification

Verify your installation:

```bash
python -c "import cslrtools2; print(cslrtools2.__version__)"
```

Check CLI tools:

```bash
lmpipe --help
sldataset2 --help
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### MediaPipe Issues

MediaPipe may require additional system libraries on Linux:

```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install mesa-libGL glib2
```

### CUDA/GPU Issues

Check CUDA availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## Next Steps

- [Quick Start Guide](guide/index.md)
- [API Reference](api/index.md)
- [Examples](examples/index.md)
