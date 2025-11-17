# Dataset Enhancement Workspace - Code Agent Instructions

## 🎯 Workspace Context

**Branch**: `dev-ai/dataset-enhancement`  
**Location**: `C:\Users\ikeda\Workspace\1github\cslrtools2-dataset`  
**Purpose**: SLDataset feature enhancements, test infrastructure, and documentation

## 📋 Current State

### Completed
- ✅ Workspace setup with independent `.venv`
- ✅ 44 packages installed via `uv sync --all-groups`
- ✅ Python 3.12.11 + PyTorch 2.9.0+cu128
- ✅ Clean working tree

### Pending Integration
**Branch to merge**: `origin/dev-ai/merge-integration`
- Contains: utilities-expansion, dependencies-update, gitignore-cleanup
- Includes: `tests/` directory with core import tests
- Status: Ready to merge

## 🚀 Priority Tasks (in order)

### 1. Merge Integration Branch
```powershell
git fetch origin
git merge origin/dev-ai/merge-integration --no-edit
```

**Expected changes**:
- New `tests/` directory with `test_core_imports.py`, `test_mediapipe_constants.py`
- Updated exception handling and utilities
- Apache 2.0 license headers in new files

### 2. Add Test Dependencies
Edit `pyproject.toml`:
```toml
[dependency-groups]
test = [
    "pytest>=9.0.0",
    "pytest-cov>=7.0.0",
]
```

Then run:
```powershell
uv sync --all-groups
```

### 3. Create SLDataset Tests

**File**: `tests/test_sldataset.py`

Test coverage:
- ✅ Dataset creation with Zarr backend
- ✅ Item addition (videos, landmarks, targets)
- ✅ Metadata management
- ✅ `__getitem__` and `__len__` operations
- ✅ PyTorch DataLoader compatibility
- ✅ Connection graph storage

**Example structure**:
```python
# tests/test_sldataset.py
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from cslrtools2.sldataset.dataset import SLDataset

@pytest.fixture
def temp_dataset_path(tmp_path):
    return tmp_path / "test_dataset.zarr"

def test_create_dataset(temp_dataset_path):
    """Test basic dataset creation."""
    dataset = SLDataset.create(
        path=temp_dataset_path,
        metadata={"name": "test", "version": "0.1"}
    )
    assert dataset.path.exists()
    assert len(dataset) == 0

def test_add_item(temp_dataset_path):
    """Test adding items to dataset."""
    # Implementation needed
    ...
```

### 4. Create Array Loader Tests

**File**: `tests/test_array_loader.py`

Test coverage:
- ✅ Load from NumPy arrays
- ✅ Load from videos (cv2)
- ✅ Load from Zarr arrays
- ✅ Load from PyTorch tensors
- ✅ Format auto-detection

### 5. Implement sldataset2 CLI Commands

**File**: `src/cslrtools2/sldataset/app/cli.py`

Add subcommands:

```python
@namespace
class SLDatasetArgs:
    command: Literal["info", "validate", "convert"] = "info"
    
    # info command
    dataset_path: Path | None = None
    
    # validate command
    check_integrity: bool = False
    
    # convert command
    input_path: Path | None = None
    output_path: Path | None = None
    output_format: Literal["zarr", "hdf5"] | None = None
```

**Implementation targets**:
1. `sldataset2 info <dataset.zarr>`: Show dataset statistics
   - Number of items
   - Metadata summary
   - Size on disk
   - Video/landmark/target shapes

2. `sldataset2 validate <dataset.zarr>`: Check integrity
   - Verify all items have required keys
   - Check array shapes consistency
   - Validate connection graphs

3. `sldataset2 convert <input> <output>`: Format conversion
   - Support zarr → zarr (optimization)
   - Future: zarr → hdf5, hdf5 → zarr

### 6. Documentation Updates

**File**: `README.md`

Add sections:

1. **FluentSigners50 Plugin Usage**:
```python
from cslrtools2.plugins.fluentsigners50.sldataset import load_fluentsigners50

# Load dataset
dataset = load_fluentsigners50(
    data_dir="/path/to/fluentsigners50",
    split="train"
)

# Use with PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

2. **sldataset2 CLI Examples**:
```bash
# Show dataset info
uv run sldataset2 info my_dataset.zarr

# Validate dataset
uv run sldataset2 validate my_dataset.zarr --check-integrity

# Convert dataset
uv run sldataset2 convert input.zarr output_optimized.zarr
```

3. **Dataset Creation Tutorial**:
- Step-by-step guide
- Code examples for each stage
- Best practices

### 7. Type Stub Completion

**File**: `typings/zarr/api/synchronous.pyi`

Complete TODO items:
- `fill_value` parameter types
- `object_codec` types
- `write_empty_chunks` types
- `meta_array` return types

Reference: Zarr v3 API documentation

## 🔧 Development Workflow

### Always use `uv run python`
```powershell
# ❌ Wrong
python script.py

# ✅ Correct
uv run python script.py
```

### Testing
```powershell
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=cslrtools2 --cov-report=html

# Run specific test file
uv run pytest tests/test_sldataset.py -v
```

### Type Checking
```powershell
uv run pyright src/
```

### Code Formatting (if needed)
```powershell
uv run ruff check src/
uv run ruff format src/
```

## 📐 Architecture Patterns to Follow

### 1. Generic Key Types
```python
class SLDataset[Kmeta: str, Kvid: str, Klm: str, Ktgt: str]:
    # K types preserve through operations
    def __getitem__(self, idx: int) -> SLDatasetItem[Kmeta, Kvid, Klm, Ktgt]:
        ...
```

### 2. Entry Point Plugins
All plugins register via `pyproject.toml`:
```toml
[project.entry-points."cslrtools2.sldataset.plugins"]
"fluentsigners50" = "cslrtools2.plugins.fluentsigners50.sldataset:fluentsigners50_loader"
```

### 3. Type Annotations
Always use:
- `from __future__ import annotations`
- PEP 695 generic syntax: `class MyClass[T]:`
- Explicit return types on all functions

### 4. Error Handling
Use custom exceptions from `cslrtools2.exceptions`:
```python
from cslrtools2.exceptions import CSLRToolsError, DatasetError

raise DatasetError(f"Invalid dataset path: {path}")
```

## ⚠️ Important Notes

### MediaPipe is Optional
- MediaPipe tests should skip gracefully if not installed
- Use `pytest.mark.skipif` for MediaPipe-dependent tests
- Check `tests/test_mediapipe_constants.py` for example

### Zarr Storage Format
Dataset structure:
```
dataset.zarr/
├── metadata/          # Dataset-level info
├── connections/       # Landmark connectivity graphs
└── items/{idx}/
    ├── videos/{kvid}  # Video arrays
    ├── landmarks/{klm}  # Landmark matrices
    └── targets/{ktgt}  # Labels/annotations
```

### Commit Message Format
Follow Conventional Commits:
```
feat: Add sldataset2 info command
fix: Correct array loader video detection
test: Add SLDataset CRUD tests
docs: Add FluentSigners50 usage examples
refactor: Improve type hints in dataset.py
```

## 🎯 Success Criteria

Before merging to `dev-ai/merge-integration`:
- ✅ All tests pass (`uv run pytest tests/ -v`)
- ✅ No type errors (`uv run pyright src/`)
- ✅ At least 80% test coverage for `sldataset/` module
- ✅ `sldataset2` CLI commands functional
- ✅ Documentation updated with examples
- ✅ Clean commit history (squash if needed)

## 📞 Next Steps After Completion

1. Push to remote:
```powershell
git push origin dev-ai/dataset-enhancement
```

2. Merge into integration branch:
```powershell
cd C:\Users\ikeda\Workspace\1github\cslrtools2-merge
git merge dev-ai/dataset-enhancement
uv run pytest tests/ -v
```

3. Create PR to `main` (if all tests pass)

---

**Last Updated**: 2025-11-14  
**Agent Mode**: Autonomous Development  
**Python Version**: 3.12.11  
**Package Manager**: uv (with `uv.lock`)
