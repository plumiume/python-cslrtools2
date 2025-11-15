# cslrtools2 例外・ログ改善 適用TODOリスト

**作成日**: 2025年11月13日  
**参照**: `EXCEPTION_LOGGING_STYLE_GUIDE.md`

---

## 📊 実装概要

**目標**: 
- カスタム例外階層導入
- 統一されたログ出力
- より良いエラーメッセージ

**期待効果**:
- ライブラリチェックリスト: +5点（エラーハンドリング）
- デバッグ性向上
- ユーザーエクスペリエンス改善

---

## 🏗️ Phase 1: 基盤整備（優先度: 🔴 HIGH）

### Task 1.1: 例外クラス定義ファイル作成

**ファイル**: `src/cslrtools2/exceptions.py` (新規作成)

**内容**:
```python
# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom exceptions for cslrtools2.

This module defines the exception hierarchy for cslrtools2, providing
better error classification and handling capabilities.

Exception Hierarchy::

    CSLRToolsError (base)
    ├── ConfigurationError
    ├── ValidationError
    ├── LMPipeError
    │   ├── EstimatorError
    │   ├── CollectorError
    │   ├── VideoProcessingError
    │   └── ModelDownloadError
    └── SLDatasetError
        ├── DataLoadError
        ├── DataFormatError
        └── PluginError

Example:
    Catch all cslrtools2 errors::

        >>> from cslrtools2.exceptions import CSLRToolsError
        >>> try:
        ...     # Some operation
        ...     pass
        ... except CSLRToolsError as e:
        ...     print(f"Error: {e}")

    Use specific exceptions::

        >>> from cslrtools2.exceptions import ValidationError
        >>> if value < 0:
        ...     raise ValidationError(f"Expected positive value, got {value}")
"""

__all__ = [
    "CSLRToolsError",
    "ConfigurationError",
    "ValidationError",
    "LMPipeError",
    "EstimatorError",
    "CollectorError",
    "VideoProcessingError",
    "ModelDownloadError",
    "SLDatasetError",
    "DataLoadError",
    "DataFormatError",
    "PluginError",
]


class CSLRToolsError(Exception):
    """Base exception for all cslrtools2 errors.
    
    All custom exceptions in cslrtools2 inherit from this class.
    This allows users to catch all cslrtools2-specific errors with
    a single except clause.
    
    Example:
        >>> try:
        ...     # Some cslrtools2 operation
        ...     pass
        ... except CSLRToolsError as e:
        ...     print(f"cslrtools2 error: {e}")
    """
    pass


# ============================================================================
# Common Exceptions
# ============================================================================

class ConfigurationError(CSLRToolsError):
    """Raised when configuration is invalid or inconsistent.
    
    This includes:
    - Invalid option combinations
    - Missing required configuration
    - Malformed configuration files
    
    Example:
        >>> raise ConfigurationError(
        ...     "Invalid estimator configuration: missing model_path"
        ... )
    """
    pass


class ValidationError(CSLRToolsError):
    """Raised when input validation fails.
    
    This includes:
    - Invalid argument values
    - Type mismatches
    - Out-of-range values
    
    Example:
        >>> raise ValidationError(
        ...     f"Expected positive integer, got {value}"
        ... )
    """
    pass


# ============================================================================
# LMPipe Exceptions
# ============================================================================

class LMPipeError(CSLRToolsError):
    """Base exception for landmark pipeline errors.
    
    All lmpipe-specific exceptions inherit from this class.
    """
    pass


class EstimatorError(LMPipeError):
    """Raised when landmark estimation fails.
    
    This includes:
    - Model initialization failures
    - Estimation computation errors
    - Invalid estimator state
    
    Example:
        >>> raise EstimatorError(
        ...     "MediaPipe model initialization failed: invalid model file"
        ... )
    """
    pass


class CollectorError(LMPipeError):
    """Raised when result collection fails.
    
    This includes:
    - Output file write errors
    - Format conversion failures
    - Collector initialization errors
    
    Example:
        >>> raise CollectorError(
        ...     f"Failed to write output to {path}: {reason}"
        ... )
    """
    pass


class VideoProcessingError(LMPipeError):
    """Raised when video processing fails.
    
    This includes:
    - Cannot open video file
    - Video decode errors
    - Frame extraction failures
    
    Example:
        >>> raise VideoProcessingError(
        ...     f"Cannot open video file: {path}. "
        ...     f"Ensure the file exists and is a valid video format."
        ... )
    """
    pass


class ModelDownloadError(LMPipeError):
    """Raised when model download fails.
    
    This includes:
    - Network errors
    - HTTP errors
    - File write errors
    
    Example:
        >>> raise ModelDownloadError(
        ...     f"Failed to download model from {url}. "
        ...     f"Status code: {status_code}. "
        ...     f"Ensure you have internet connectivity."
        ... )
    """
    pass


# ============================================================================
# SLDataset Exceptions
# ============================================================================

class SLDatasetError(CSLRToolsError):
    """Base exception for dataset errors.
    
    All sldataset-specific exceptions inherit from this class.
    """
    pass


class DataLoadError(SLDatasetError):
    """Raised when data loading fails.
    
    This includes:
    - File not found
    - Data key not found
    - File format errors
    
    Example:
        >>> raise DataLoadError(
        ...     f"Failed to load array from {path}: {reason}"
        ... )
    """
    pass


class DataFormatError(SLDatasetError):
    """Raised when data format is unexpected.
    
    This includes:
    - Unexpected data types
    - Invalid data shapes
    - Missing required fields
    
    Example:
        >>> raise DataFormatError(
        ...     f"Expected Tensor in file {path}, got {type(data).__name__}. "
        ...     f"Ensure the file was saved with torch.save(tensor, path)."
        ... )
    """
    pass


class PluginError(SLDatasetError):
    """Raised when plugin loading or execution fails.
    
    This includes:
    - Invalid plugin structure
    - Plugin load failures
    - Plugin execution errors
    
    Example:
        >>> raise PluginError(
        ...     f"Plugin entry point {name} does not return a valid processor. "
        ...     f"Expected a callable, got {type(processor).__name__}."
        ... )
    """
    pass
```

**所要時間**: 15分  
**テスト**: `pytest tests/test_exceptions.py`（新規作成）

---

### Task 1.2: 共通ロガー設定ファイル作成

**ファイル**: `src/cslrtools2/logger.py` (新規作成)

**内容**:
```python
# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging utilities for cslrtools2.

This module provides unified logging configuration for all cslrtools2
subpackages.

Logger Hierarchy::

    cslrtools2 (root)
    ├── cslrtools2.lmpipe
    └── cslrtools2.sldataset

Example:
    Get a logger::

        >>> import logging
        >>> logger = logging.getLogger("cslrtools2.mymodule")
        >>> logger.info("Starting operation")

    Configure logging level::

        >>> import logging
        >>> logging.getLogger("cslrtools2").setLevel(logging.DEBUG)
"""

import logging

__all__ = [
    "root_logger",
    "standard_formatter",
    "detailed_formatter",
]

# Root logger for cslrtools2
root_logger = logging.getLogger("cslrtools2")

# Standard formatter for production use
standard_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Detailed formatter for debugging
detailed_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s (%(pathname)s:%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def configure_logger(
    logger: logging.Logger,
    level: int = logging.INFO,
    formatter: logging.Formatter = standard_formatter,
    handler: logging.Handler | None = None,
) -> None:
    """Configure a logger with standard settings.
    
    Args:
        logger: Logger instance to configure.
        level: Logging level (default: INFO).
        formatter: Formatter to use (default: standard_formatter).
        handler: Handler to use (default: StreamHandler to stdout).
    
    Example:
        >>> import logging
        >>> from cslrtools2.logger import configure_logger, detailed_formatter
        >>> logger = logging.getLogger("cslrtools2.mymodule")
        >>> configure_logger(logger, level=logging.DEBUG, formatter=detailed_formatter)
    """
    logger.setLevel(level)
    
    if handler is None:
        handler = logging.StreamHandler()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
```

**所要時間**: 10分

---

### Task 1.3: sldatasetロガー作成

**ファイル**: `src/cslrtools2/sldataset/logger.py` (新規作成)

**内容**:
```python
# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logger for sldataset module."""

import logging

sldataset_logger = logging.getLogger("cslrtools2.sldataset")

__all__ = ["sldataset_logger"]
```

**所要時間**: 5分

---

### Task 1.4: lmpipeロガー更新

**ファイル**: `src/cslrtools2/lmpipe/logger.py` (既存更新)

**変更内容**:
```python
# 変更前
lmpipe_logger = logging.getLogger("lmpipe")

# 変更後
lmpipe_logger = logging.getLogger("cslrtools2.lmpipe")
```

**所要時間**: 2分

---

### Task 1.5: __init__.pyに例外をエクスポート

**ファイル**: `src/cslrtools2/__init__.py`

**追加内容**:
```python
# 既存の__all__に追加
__all__ = [
    "__version__",
    # Exceptions (便利のため再エクスポート)
    "CSLRToolsError",
    "ConfigurationError",
    "ValidationError",
    "LMPipeError",
    "SLDatasetError",
]

# Import exceptions for re-export
from .exceptions import (
    CSLRToolsError,
    ConfigurationError,
    ValidationError,
    LMPipeError,
    SLDatasetError,
)
```

**所要時間**: 3分

---

## 🔧 Phase 2: lmpipeモジュール適用（優先度: 🟡 MEDIUM）

### Task 2.1: plugins/mediapipe/lmpipe/base.py

**変更箇所1**: `get_mediapipe_model`関数

```python
# 変更前
from typing import Literal
import requests
from pathlib import Path

def get_mediapipe_model(...):
    ...
    if part_map is None:
        raise ValueError(
            f"Invalid model part: {part}. Available parts: {list(MODELS.keys())}"
        )
    
    if model_url is None:
        raise ValueError(
            f"Invalid model size: {size} for part {part}. Available sizes: {list(part_map.keys())}"
        )
    
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download model from {model_url}. Status code: {response.status_code}"
        )

# 変更後
from typing import Literal
import requests
from pathlib import Path
from cslrtools2.exceptions import ValidationError, ModelDownloadError

def get_mediapipe_model(...):
    ...
    if part_map is None:
        raise ValidationError(
            f"Invalid model part: {part}. Available parts: {list(MODELS.keys())}"
        )
    
    if model_url is None:
        raise ValidationError(
            f"Invalid model size: {size} for part {part}. "
            f"Available sizes: {list(part_map.keys())}"
        )
    
    if response.status_code != 200:
        raise ModelDownloadError(
            f"Failed to download model from {model_url}. "
            f"Status code: {response.status_code}. "
            f"Reason: {response.reason}. "
            f"Ensure you have internet connectivity."
        )
```

**所要時間**: 5分

---

### Task 2.2: plugins/mediapipe/lmpipe/holistic.py

**変更箇所**: `keys`プロパティ

```python
# 変更前
def keys(self) -> list[K]:
    raise ValueError("Holistic estimator does not have a single key.")

# 変更後
from cslrtools2.exceptions import ValidationError

def keys(self) -> list[K]:
    raise ValidationError(
        "Holistic estimator does not have a single key. "
        "Use estimator.pose_key, estimator.left_hand_key, etc. instead."
    )
```

**所要時間**: 3分

---

### Task 2.3: plugins/mediapipe/lmpipe/hand.py

**変更箇所**: `keys`プロパティ

```python
# 変更前
def keys(self) -> list[K]:
    raise ValueError(
        "BothHandsEstimator does not have a single key."
        " Use estimator.left_hand_key and estimator.right_hand_key instead."
    )

# 変更後
from cslrtools2.exceptions import ValidationError

def keys(self) -> list[K]:
    raise ValidationError(
        "BothHandsEstimator does not have a single key. "
        "Use estimator.left_hand_key and estimator.right_hand_key instead."
    )
```

**所要時間**: 3分

---

### Task 2.4: lmpipe/runspec.py

**変更箇所**: `create_runspec`関数

```python
# 変更前
from pathlib import Path
from ..logger import lmpipe_logger

def create_runspec(...):
    ...
    if not src_path.exists():
        lmpipe_logger.error(f"Video file does not exist: {src_path}")
        raise FileNotFoundError(f"Video file does not exist: {src_path}")

# 変更後
from pathlib import Path
from cslrtools2.exceptions import VideoProcessingError
from ..logger import lmpipe_logger

def create_runspec(...):
    ...
    if not src_path.exists():
        lmpipe_logger.error(f"Source path does not exist: {src_path}")
        raise VideoProcessingError(
            f"Source path does not exist: {src_path}. "
            f"Ensure the path is correct and the file is accessible."
        )
```

**所要時間**: 5分

---

### Task 2.5: lmpipe/interface/__init__.py

**変更箇所**: エラーハンドリング改善

```python
# ファイル先頭にimport追加
from cslrtools2.exceptions import VideoProcessingError, CollectorError

# run_videoメソッド内
# 変更前
cap = cv2.VideoCapture(str(runspec.src))
if not cap.isOpened():
    lmpipe_logger.error(f"Cannot open video file: {runspec.src}")
    return

# 変更後
cap = cv2.VideoCapture(str(runspec.src))
if not cap.isOpened():
    error_msg = f"Cannot open video file: {runspec.src}"
    lmpipe_logger.error(error_msg)
    raise VideoProcessingError(
        f"{error_msg}. "
        f"Ensure the file exists and is in a supported format (mp4, avi, etc.)."
    )

# run_streamメソッド内も同様
```

**所要時間**: 10分

---

## 📦 Phase 3: sldatasetモジュール適用（優先度: 🟡 MEDIUM）

### Task 3.1: sldataset/utils.py

**変更箇所**: 例外の置き換え

```python
# 変更前
def get_array_or_group(...):
    ...
    if key not in zarr_array:
        raise KeyError(f"Array not found at path: {path}")
    
def get_group(...):
    ...
    if key not in zarr_group:
        raise KeyError(f"Group not found at path: {path}")

# 変更後
from cslrtools2.exceptions import DataLoadError

def get_array_or_group(...):
    ...
    if key not in zarr_array:
        raise DataLoadError(
            f"Array not found at path: {path}. "
            f"Available keys: {list(zarr_array.keys())}"
        )
    
def get_group(...):
    ...
    if key not in zarr_group:
        raise DataLoadError(
            f"Group not found at path: {path}. "
            f"Available keys: {list(zarr_group.keys())}"
        )
```

**所要時間**: 5分

---

### Task 3.2: sldataset/array_loader.py

**変更箇所**: ValueError → DataFormatError

```python
# 変更前
from typing import Any
import numpy as np
import torch

def load_npy_single(...):
    ...
    if isinstance(result, dict):
        raise ValueError(f"Expected a single array in NPY file at {path}, got a mapping")

def load_npz_single(...):
    ...
    if not isinstance(npz, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a NPZ file at {path}, got {type(npz)}")

def load_torch_single(...):
    ...
    if not isinstance(result, torch.Tensor):
        raise ValueError(f"Expected a Tensor in Torch file at {path}, got {type(result)}")

def load_torch_multi(...):
    ...
    if not isinstance(result, dict):
        raise ValueError(f"Expected a dict of str to ArrayLike in Torch file at {path}")

def load_safetensors_multi(...):
    ...
    if not isinstance(result, dict):
        raise ValueError(f"Expected a dict of str to ArrayLike in Safetensors file at {path}")

# 変更後
from typing import Any
import numpy as np
import torch
from cslrtools2.exceptions import DataFormatError

def load_npy_single(...):
    ...
    if isinstance(result, dict):
        raise DataFormatError(
            f"Expected a single array in NPY file at {path}, got a mapping. "
            f"Use load_npy_multi() for multi-array files."
        )

def load_npz_single(...):
    ...
    if not isinstance(npz, np.lib.npyio.NpzFile):
        raise DataFormatError(
            f"Expected a NPZ file at {path}, got {type(npz).__name__}. "
            f"Ensure the file has .npz extension and was saved with np.savez()."
        )

def load_torch_single(...):
    ...
    if not isinstance(result, torch.Tensor):
        raise DataFormatError(
            f"Expected a Tensor in Torch file at {path}, got {type(result).__name__}. "
            f"Ensure the file was saved with torch.save(tensor, path)."
        )

def load_torch_multi(...):
    ...
    if not isinstance(result, dict):
        raise DataFormatError(
            f"Expected a dict of str to ArrayLike in Torch file at {path}, "
            f"got {type(result).__name__}. "
            f"Use load_torch_single() for single tensor files."
        )

def load_safetensors_multi(...):
    ...
    if not isinstance(result, dict):
        raise DataFormatError(
            f"Expected a dict of str to ArrayLike in Safetensors file at {path}, "
            f"got {type(result).__name__}."
        )
```

**所要時間**: 10分

---

### Task 3.3: sldataset/app/plugins.py

**変更箇所**: TypeError/ValueError → PluginError

```python
# 変更前
import importlib.metadata

def loader() -> dict[str, PluginInfo[Any]]:
    ...
    for ep in entry_points:
        info = ep.load()
        
        if not _is_tuple(info):
            raise TypeError(
                f"Plugin entry point {ep.name} does not return a tuple"
            )
        if len(info) != 2:
            raise ValueError(
                f"Plugin entry point {ep.name} does not return a tuple of length 2"
            )
        
        nswrapper, processor = info
        
        if not _is_nswrapper(nswrapper):
            raise TypeError(
                f"First element of plugin entry point {ep.name} is not a NamespaceWrapper"
            )
        
        if not _is_processor(processor):
            raise TypeError(
                f"Second element of plugin entry point {ep.name} is not a processor callable"
            )

# 変更後
import importlib.metadata
from cslrtools2.exceptions import PluginError

def loader() -> dict[str, PluginInfo[Any]]:
    ...
    for ep in entry_points:
        info = ep.load()
        
        if not _is_tuple(info):
            raise PluginError(
                f"Plugin entry point {ep.name} does not return a tuple. "
                f"Expected (NamespaceWrapper, processor), got {type(info).__name__}."
            )
        if len(info) != 2:
            raise PluginError(
                f"Plugin entry point {ep.name} does not return a tuple of length 2. "
                f"Expected (NamespaceWrapper, processor), got tuple of length {len(info)}."
            )
        
        nswrapper, processor = info
        
        if not _is_nswrapper(nswrapper):
            raise PluginError(
                f"First element of plugin entry point {ep.name} is not a NamespaceWrapper. "
                f"Got {type(nswrapper).__name__}."
            )
        
        if not _is_processor(processor):
            raise PluginError(
                f"Second element of plugin entry point {ep.name} is not a processor callable. "
                f"Expected a callable, got {type(processor).__name__}."
            )
```

**所要時間**: 8分

---

### Task 3.4: sldataset/app/cli.py

**変更箇所**: エラーハンドリング追加

```python
# 変更前
def main(args: MainArgs.T) -> None:
    ...
    if args.command == "convert":
        ...
    else:
        raise ValueError(f"Unknown command: {args.command}")

# 変更後
from cslrtools2.exceptions import ConfigurationError
from ..logger import sldataset_logger

def main(args: MainArgs.T) -> None:
    sldataset_logger.info(f"Starting sldataset command: {args.command}")
    
    if args.command == "convert":
        sldataset_logger.info("Running convert subcommand")
        ...
    else:
        error_msg = f"Unknown command: {args.command}"
        sldataset_logger.error(error_msg)
        raise ConfigurationError(
            f"{error_msg}. Available commands: convert"
        )
```

**所要時間**: 5分

---

### Task 3.5: sldataset/dataset.pyにログ追加

**追加箇所**: 主要メソッド

```python
# ファイル先頭
from .logger import sldataset_logger

class SLDataset:
    def __init__(...):
        sldataset_logger.debug(
            f"Initializing SLDataset: root={root}, "
            f"lazy_load={lazy_load}, transform={transform is not None}"
        )
        ...
    
    def __getitem__(self, idx: int):
        sldataset_logger.debug(f"Loading item {idx}")
        ...
```

**所要時間**: 10分

---

## 📝 Phase 4: テスト作成（優先度: 🟢 LOW）

### Task 4.1: 例外テスト

**ファイル**: `tests/test_exceptions.py` (新規作成)

**内容**:
```python
"""Tests for cslrtools2.exceptions module."""

import pytest
from cslrtools2.exceptions import (
    CSLRToolsError,
    ValidationError,
    LMPipeError,
    ModelDownloadError,
    SLDatasetError,
    DataFormatError,
)


def test_exception_hierarchy() -> None:
    """Test exception inheritance hierarchy."""
    # All custom exceptions inherit from CSLRToolsError
    assert issubclass(ValidationError, CSLRToolsError)
    assert issubclass(LMPipeError, CSLRToolsError)
    assert issubclass(SLDatasetError, CSLRToolsError)
    
    # LMPipe exceptions inherit from LMPipeError
    assert issubclass(ModelDownloadError, LMPipeError)
    
    # SLDataset exceptions inherit from SLDatasetError
    assert issubclass(DataFormatError, SLDatasetError)


def test_catch_all_exceptions() -> None:
    """Test catching all cslrtools2 exceptions."""
    with pytest.raises(CSLRToolsError):
        raise ValidationError("test error")
    
    with pytest.raises(CSLRToolsError):
        raise ModelDownloadError("test error")


def test_exception_messages() -> None:
    """Test exception messages are preserved."""
    msg = "Test error message"
    
    with pytest.raises(ValidationError, match=msg):
        raise ValidationError(msg)
    
    with pytest.raises(DataFormatError, match=msg):
        raise DataFormatError(msg)
```

**所要時間**: 15分

---

## 📊 実装スケジュール

### Week 1: 基盤整備
- [ ] Day 1: Phase 1完了（Task 1.1 - 1.5）
- [ ] Day 2: Phase 2開始（Task 2.1 - 2.3）
- [ ] Day 3: Phase 2完了（Task 2.4 - 2.5）

### Week 2: モジュール適用
- [ ] Day 4: Phase 3開始（Task 3.1 - 3.2）
- [ ] Day 5: Phase 3完了（Task 3.3 - 3.5）
- [ ] Day 6: Phase 4（Task 4.1）
- [ ] Day 7: テスト・レビュー

---

## ✅ 完了チェックリスト

### Phase 1: 基盤整備
- [ ] `src/cslrtools2/exceptions.py` 作成完了
- [ ] `src/cslrtools2/logger.py` 作成完了
- [ ] `src/cslrtools2/sldataset/logger.py` 作成完了
- [ ] `src/cslrtools2/lmpipe/logger.py` 更新完了
- [ ] `src/cslrtools2/__init__.py` 更新完了

### Phase 2: lmpipe適用
- [ ] `plugins/mediapipe/lmpipe/base.py` 更新完了
- [ ] `plugins/mediapipe/lmpipe/holistic.py` 更新完了
- [ ] `plugins/mediapipe/lmpipe/hand.py` 更新完了
- [ ] `lmpipe/runspec.py` 更新完了
- [ ] `lmpipe/interface/__init__.py` 更新完了

### Phase 3: sldataset適用
- [ ] `sldataset/utils.py` 更新完了
- [ ] `sldataset/array_loader.py` 更新完了
- [ ] `sldataset/app/plugins.py` 更新完了
- [ ] `sldataset/app/cli.py` 更新完了
- [ ] `sldataset/dataset.py` 更新完了

### Phase 4: テスト
- [ ] `tests/test_exceptions.py` 作成完了
- [ ] すべてのテストが通過

### 最終確認
- [ ] pyright strict チェック通過
- [ ] 既存の14テストすべて通過
- [ ] 新規テストすべて通過
- [ ] ドキュメント更新完了

---

## 🎯 期待効果

### 定量的効果
- **ライブラリチェックリスト**: 100/130 → 105/130 (+5点)
- **テストケース**: 14 → 20+ (+6テスト以上)

### 定性的効果
- より明確なエラーメッセージ
- デバッグ性の向上
- ユーザーエクスペリエンスの改善
- プロフェッショナルなライブラリ品質

---

**最終更新**: 2025年11月13日  
**次のアクション**: Phase 1から順次実装開始
