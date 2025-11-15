# cslrtools2 AI Coding Agent Instructions

**Project**: Continuous Sign Language Recognition (CSLR) toolkit for landmark extraction and dataset management  
**Language**: Python 3.12+ with PEP 695 generics and modern type hints  
**Package Manager**: `uv` (preferred for all operations)

---

## 🏗️ Architecture Overview

This project is a **domain-specific application framework** with two major subsystems:

### 1. LMPipe (Landmark Pipeline) - ETL Framework
- **Purpose**: Extract skeletal landmarks from sign language videos using MediaPipe
- **Pattern**: Pipeline (Extract → Transform → Load) with plugin architecture
- **Key abstraction**: `Estimator[K]` ABC - implement `process()` and `configure_estimator_name()`
- **Plugin system**: Entry points in `pyproject.toml` register estimators (e.g., `mediapipe.holistic`, `mediapipe.pose`)
- **Execution modes**: Sequential, parallel (multiprocessing), or thread-based via `LMPipeOptions`
- **CLI**: `lmpipe mediapipe.holistic input.mp4 -o output.npz --workers 4`

**Core flow**: `RunSpec` (job spec) → `Estimator` (landmark detection) → `Collector` (save to NPY/Zarr/SafeTensors) → Progress tracking via Rich

### 2. SLDataset - Data Access Layer
- **Purpose**: Unified storage/loading for sign language datasets with PyTorch compatibility
- **Pattern**: Repository pattern with Zarr-backed storage
- **Schema**: `dataset.zarr/{metadata/, connections/, items/[N]/{videos/, landmarks/, targets/}}`
- **Key classes**: `SLDataset` (PyTorch Dataset), `SLDatasetItem` (DTO), `IterableSLDataset` (streaming)
- **CLI**: `sldataset2 info dataset.zarr`

---

## 🔧 Development Workflow

### Essential Commands

**All operations use `uv`** - never use `pip` directly:

```powershell
# Install dependencies (first time)
uv sync

# Run CLI tools (no install needed)
uv run lmpipe mediapipe.pose video.mp4 -o output.npz
uv run sldataset2 info dataset.zarr

# Run Python scripts
uv run python script_name.py

# Test in Docker (multiple PyTorch+CUDA environments)
cd tests/build
docker compose run --rm pytorch-cu128 uv run python tests/build/test_pytorch_cuda.py
```

**MediaPipe installation**: Separate dependency group - `uv sync --group mediapipe`

### Build & Test Matrix

- **Primary**: PyTorch 2.9.0 + CUDA 12.8 (from `https://download.pytorch.org/whl/cu128`)
- **Docker envs**: `pytorch-cu128`, `pytorch-cu126`, `pytorch-cu130`, `pytorch-cpu` (see `tests/build/`)
- **UV cache**: Shared volume (`uv-cache`) across containers for fast installs (~0.5s after first run)

---

## 📐 Code Style & Conventions

### Type System (CRITICAL)
- **Python 3.12+ only**: Use PEP 695 generic syntax: `class MyClass[T: Bound]: ...`, `def func[T](x: T) -> T: ...`
- **Always import annotations**: `from __future__ import annotations` in every file
- **No legacy typing**: Avoid `typing.Generic`, `typing.TypeVar` - use built-in generics
- **Type stubs**: Custom stubs in `typings/` for mediapipe, zarr, safetensors (incomplete upstream types)
- **Pyright-compliant**: All code must pass Pyright static analysis (zero type errors)

### Docstrings (Google Style + Sphinx)
**Required format** (see `DOCSTRING_STYLE.md`):
- Args: Type names in backticks `Type` for VSCode hover support
- Body text: Use Sphinx roles - `:class:`ClassName``, `:func:`function_name``, `:obj:`None``/`True`/`False``
- Returns/Raises: Use roles - `:class:`Tensor``, `:exc:`ValueError``
- Example:
  ```python
  def process(frame: MatLike) -> ProcessResult:
      """Process a single frame.
      
      Args:
          frame (`MatLike`): Input frame. Must not be :obj:`None`.
      
      Returns:
          :class:`ProcessResult`: Processed landmarks via :meth:`detect_landmarks`.
      
      Raises:
          :exc:`ValueError`: When frame is :obj:`None`.
      """
  ```

### Exception Handling
Follow `EXCEPTION_LOGGING_STYLE_GUIDE.md`:
- **Custom exceptions**: Inherit from `CSLRToolsError` (to be implemented - see guide)
- **Planned hierarchy**: `LMPipeError`, `EstimatorError`, `SLDatasetError`, `DataLoadError`, etc.
- **Current state**: Uses standard exceptions (`ValueError`, `TypeError`, `RuntimeError`) - gradually migrate
- **Logging**: Use named loggers (`logging.getLogger("cslrtools2.lmpipe")`) with structured messages

### Git Workflow
See `BRANCHING_STRATEGY.md`:
- **AI development**: `dev-ai/<task-name>` branches (frequent commits encouraged)
- **Commit format**: Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`)
- **Merge to main**: Squash merge preferred (cleans up AI commit history)
- **Integration branch**: `main-ai` for combining multiple `dev-ai/*` branches before merging to `main`

---

## 🧩 Plugin System Architecture

### Creating a Custom Estimator

1. **Subclass `Estimator[K]`** where `K` is your landmark key type (e.g., `Literal['my_estimator']`)
2. **Implement required methods**:
   ```python
   from cslrtools2.lmpipe.estimator import Estimator, ProcessResult
   
   class MyEstimator(Estimator[Literal['my_key']]):
       def process(self, frame: MatLike) -> ProcessResult[Literal['my_key']]:
           # Detect landmarks from frame
           return ProcessResult(
               frame_id=0,
               headers={'my_key': np.array(['x', 'y', 'z'])},
               landmarks={'my_key': np.array([[1.0, 2.0, 3.0]])},
               annotated_frame=frame
           )
       
       def configure_estimator_name(self) -> Literal['my_key']:
           return 'my_key'
   ```

3. **Register in `pyproject.toml`**:
   ```toml
   [project.entry-points."cslrtools2.lmpipe.plugins"]
   "my_estimator" = "mypackage.estimator_module:estimator_info"
   ```

### MediaPipe Constants
**Use `mp_constants.py`** for all MediaPipe landmark definitions:
- **Landmark enums**: `PoseLandmark`, `HandLandmark` (re-exported from MediaPipe - DO NOT redefine)
- **Connection constants**: `POSE_CONNECTIONS`, `HAND_CONNECTIONS`, `FACEMESH_*` (14 types)
- **Deprecated**: `MediaPipePoseNames`, `MediaPipeHandNames` (aliases kept for compatibility)
- **Import**: `from cslrtools2.plugins.mediapipe.lmpipe.mp_constants import PoseLandmark, POSE_CONNECTIONS`

---

## 🎯 Common Tasks

### Adding a New Landmark Estimator
1. Create estimator class in `src/cslrtools2/plugins/<plugin_name>/lmpipe/`
2. Define `<name>_args.py` with CLI argument groups (see `mediapipe/lmpipe/pose_args.py`)
3. Register entry point in `pyproject.toml`
4. Update `CHANGELOG.md` under `[Unreleased]`

### Adding a New Output Format (Collector)
1. Subclass `Collector` in `src/cslrtools2/lmpipe/collector/`
2. Implement `collect()` method for accumulating results
3. Implement `save()` method for writing to disk
4. Add CLI option in `lmpipe/app/args.py`

### Working with Zarr Datasets
```python
import zarr
from cslrtools2.sldataset import SLDataset

# Read
root = zarr.open("dataset.zarr", mode="r")
dataset = SLDataset.from_zarr(root)
item = dataset[0]  # Returns SLDatasetItem

# Write
new_root = zarr.open("output.zarr", mode="w")
dataset.to_zarr(new_root)
```

---

## ⚠️ Critical Integration Points

### 1. PyTorch Custom Index URLs
**Torch/torchvision MUST use custom index** (`https://download.pytorch.org/whl/cu128`):
- Defined in `pyproject.toml`: `[[tool.uv.index]]` sections
- Never install from PyPI - builds are incompatible
- `torchvision==0.24.0` is PINNED (known issue with 0.24.1 - see `github-issue-torchvision==0.24.1-markdown.md`)

### 2. MediaPipe Separate Install
MediaPipe is **optional** (separate dependency group) because:
- Not needed for dataset loading (`sldataset` works standalone)
- Users may use custom estimators without MediaPipe
- Install: `uv sync --group mediapipe`

### 3. Type Stub Files
Custom stubs in `typings/` because upstream packages lack complete types:
- `mediapipe.pyi`: Landmark enums, connections
- `zarr/api/synchronous.pyi`: Zarr v3 API
- `safetensors/*.pyi`: Save/load functions
- **DO NOT** modify stubs lightly - they're carefully aligned with runtime behavior

### 4. License Headers
**Apache 2.0 license required** on all `.py` files (use `add_license_headers.py` script):
```python
# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

---

## 📚 Key Documentation Files

- `LMPIPE_FRAMEWORK_CLASSIFICATION.md`: Deep dive into LMPipe architecture patterns
- `DOCSTRING_STYLE.md`: Complete docstring formatting guide
- `EXCEPTION_LOGGING_STYLE_GUIDE.md`: Exception hierarchy design (implementation pending)
- `MEDIAPIPE_CONSTANTS_INTEGRATION.md`: MediaPipe constants refactoring history
- `BRANCHING_STRATEGY.md`: Git workflow for AI agents
- `tests/build/DOCKER_STRATEGY.md`: Multi-environment testing setup

---

## 🚨 Common Pitfalls

1. **Don't use `pip`** - always `uv run` or `uv sync`
2. **Don't redefine MediaPipe constants** - import from `mp_constants.py`
3. **Don't use legacy `typing.Generic[T]`** - use PEP 695 `class MyClass[T]:`
4. **Don't forget `from __future__ import annotations`** at top of files
5. **Don't mix PyPI and custom PyTorch indexes** - index config is critical
6. **Don't skip Sphinx roles in docstrings** - `:class:`, `:func:`, `:obj:` are required
7. **Don't create estimators without entry points** - CLI won't discover them

---

## 🎯 Current Focus Areas (as of 2025-11-15)

- Exception hierarchy implementation (`EXCEPTION_LOGGING_TODO.md`)
- Expanding test coverage (currently minimal - see `tests/` directory)
- API documentation with Sphinx (structure in `sphinx/`, incomplete)
- MediaPipe annotation improvements using connection constants

---

**Quick Reference**: For immediate context on any module, read the docstring in `src/cslrtools2/<module>/__init__.py` - they contain comprehensive architectural overviews.
