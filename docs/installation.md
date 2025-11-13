# Installation Guide

## Requirements

- Python 3.12 or higher
- pip or uv package manager

## Basic Installation

Install the core package:

```bash
pip install cslrtools2
```

Or using `uv`:

```bash
uv pip install cslrtools2
```

## Optional Dependencies

### MediaPipe Support

For landmark extraction features using MediaPipe:

```bash
pip install mediapipe
```

Or install from the optional dependency group:

```bash
pip install cslrtools2[mediapipe]
```

### Development Tools

For development and testing:

```bash
pip install cslrtools2[dev]
```

This includes:
- pytest (testing framework)
- pytest-cov (coverage reporting)
- type stubs for dependencies

## From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ikegami-yukino/python-cslrtools2.git
cd python-cslrtools2
pip install -e .
```

### With Optional Dependencies

```bash
pip install -e ".[mediapipe,dev]"
```

## GPU/CUDA Requirements

For optimal performance with PyTorch-based operations:

- **PyTorch**: >= 2.0.0 (with CUDA support recommended)
- **CUDA**: 11.8 or higher (for GPU acceleration)

Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

See [PyTorch Getting Started](https://pytorch.org/get-started/locally/) for platform-specific instructions.

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
