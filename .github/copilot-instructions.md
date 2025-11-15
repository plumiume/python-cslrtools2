# Copilot Instructions for cslrtools2

## Project Overview

**cslrtools2** is a Continuous Sign Language Recognition (CSLR) toolkit providing:
- **LMPipe**: Landmark extraction pipeline using MediaPipe
- **SLDataset**: PyTorch-compatible dataset management with Zarr storage
- **ConvSize**: PyTorch convolution size calculation utilities

Python 3.12+ required with modern type hints (PEP 695 generics).

## Architecture Patterns

### Plugin System via Entry Points

The project uses `pyproject.toml` entry points for extensibility. All MediaPipe estimators are registered as plugins:

```toml
[project.entry-points."cslrtools2.lmpipe.plugins"]
"mediapipe.holistic" = "cslrtools2.plugins.mediapipe.lmpipe.holistic_args:holistic_info"
```

Each plugin returns a tuple `(NamespaceWrapper, EstimatorCreator)`. See `src/cslrtools2/lmpipe/app/plugins.py` for the loader pattern.

**When adding new estimators:**
1. Create `{name}_args.py` with a `@namespace` class and `{name}_info` tuple
2. Implement estimator in `{name}.py` extending `Estimator[K]`
3. Register in `pyproject.toml` entry points

### Type System: Generic Key Types

The codebase uses string literal types extensively for type-safe key handling:

```python
class Estimator[K: str]:  # K is a string literal type like "pose" | "left_hand"
    def process(self) -> ProcessResult[K]:  # Keys preserved through pipeline
        ...
```

**Critical pattern**: The `K` type parameter flows through:
- `Estimator[K]` → `ProcessResult[K]` → `Collector[K]` → output files

Dataset items use 4 generic keys: `[Kmeta, Kvid, Klm, Ktgt]` for metadata/videos/landmarks/targets.

### Collector Pattern: Output Format Abstraction

Collectors handle result persistence. Two categories:

**LandmarkMatrixSaveCollector** (per-key files):
- `csv_lmsc.py`, `npy_lmsc.py`, `json_lmsc.py`
- Each landmark key (e.g., "pose", "left_hand") → separate file
- Base class in `src/cslrtools2/lmpipe/collector/landmark_matrix/base.py`

**Container collectors** (single file for all keys):
- `npz_lmsc.py`, `zarr_lmsc.py`, `safetensors_lmsc.py`, `torch_lmsc.py`
- All landmarks in one container with keys as internal structure

**When adding new formats**: Extend `LandmarkMatrixSaveCollector[K]` and implement:
- `is_perkey` / `is_container` properties
- `_open_file()`, `_save_landmark_matrix()`, `_close_file()`

### CLI Architecture: clipar + Nested Commands

Uses `clipar` library for CLI argument parsing with nested command structures:

```python
# In holistic_args.py
@namespace
class MediaPipeHolisticArgs(MediaPipeBaseArgs, mixin.ReprMixin):
    model_complexity: int = 1
    smooth_landmarks: bool = True
```

CLI pattern: `lmpipe mediapipe.holistic input.mp4 -o output.npz --model-complexity 2`

Command resolution in `cli.py` uses dynamic plugin lookup - args determine which estimator plugin loads.

## Development Workflows

### Build & Run

```powershell
# Install with uv (recommended)
uv pip install -e .

# With MediaPipe support
uv pip install -e . --group mediapipe

# Run Python with uv (CRITICAL - always use this)
uv run python script.py

# Run landmark extraction
lmpipe mediapipe.holistic video.mp4 -o landmarks.zarr --workers 4

# Use dataset tools
sldataset2 <command>
```

**IMPORTANT**: Always use `uv run python` instead of `python` directly to ensure proper virtual environment and dependency resolution.

### Type Checking

Project uses Pyright with strict mode. Key conventions:
- All files use `from __future__ import annotations` for forward references
- Generic classes use PEP 695 syntax: `class MyClass[T: Bound]:`
- Type stubs in `typings/` directory for external libraries (MediaPipe, Zarr, etc.)

### Testing

No test files currently exist (alpha stage). When adding tests:
- Use pytest
- Mock MediaPipe dependencies (heavy external dependency)
- Test collector outputs by verifying file formats

### Dependency Management

- **uv** is the package manager (see `pyproject.toml`)
- PyTorch installed from custom index: `https://download.pytorch.org/whl/cu128`
- `clipar` from git: `https://github.com/plumiume/python-clipar.git`

MediaPipe is optional (separate dependency group) - estimators gracefully import error if missing.

## Key File References

- **Entry point definitions**: `pyproject.toml` (lines 24-29)
- **Plugin loader**: `src/cslrtools2/lmpipe/app/plugins.py`
- **Estimator base**: `src/cslrtools2/lmpipe/estimator.py`
- **Collector base**: `src/cslrtools2/lmpipe/collector/base.py`
- **Dataset core**: `src/cslrtools2/sldataset/dataset.py`
- **CLI entry**: `src/cslrtools2/lmpipe/app/cli.py`
- **Options schema**: `src/cslrtools2/lmpipe/options.py` (uses TypedDict + clipar groups)

## Project-Specific Conventions

### Branching Strategy

- `main` - stable production
- `dev-ai/*` - AI agent collaborative development branches
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, etc.
- Squash merge from `dev-ai/*` to `main` to consolidate AI agent commits

See `BRANCHING_STRATEGY.md` for full rules.

### File Organization

- Executable logic: `src/cslrtools2/{module}/app/`
- Public interfaces: `src/cslrtools2/{module}/interface/`
- Plugins: `src/cslrtools2/plugins/{provider}/{module}/`
- Type stubs: `typings/` (for external libraries without stubs)

### Naming Patterns

- Estimators: `{Provider}{Part}Estimator` (e.g., `MediaPipeHolisticEstimator`)
- Args classes: `{Provider}{Part}Args` with `{part}_info` tuple export
- Collectors: `{Format}LandmarkMatrixSaveCollector` → alias `{Format}LMSC`

## Integration Points

### MediaPipe Models

Models auto-download on first use to `src/cslrtools2/assets/{part}/{size}.task`. See `src/cslrtools2/plugins/mediapipe/lmpipe/base.py` `get_mediapipe_model()`.

Available models:
- Pose: lite/full/heavy
- Hand: full
- Face: full

### Zarr Storage Schema

Datasets follow this structure:

```
dataset.zarr/
├── metadata/          # Dataset-level info
├── connections/       # Landmark connectivity graphs
└── items/{idx}/
    ├── videos/{kvid}  # Video arrays
    ├── landmarks/{klm}  # Landmark matrices
    └── targets/{ktgt}  # Labels/annotations
```

Access via `SLDataset` class which implements `torch.utils.data.Dataset`.

### Parallel Processing

Uses `loky` for multiprocessing (not stdlib `multiprocessing`). Custom `ProcessPoolExecutor` in `src/cslrtools2/lmpipe/interface/executor.py` adds `cancel_futures` support to shutdown.

`DummyExecutor` available for single-threaded debugging.
