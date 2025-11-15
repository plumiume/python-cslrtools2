# cslrtools2 Coding Style Guide

**作成日**: 2025年11月16日  
**対象**: src/cslrtools2/ 以下の全Pythonコード  
**目的**: プロジェクト全体で統一されたコーディングスタイルの確立

---

## 📋 目次

1. [基本原則](#基本原則)
2. [ファイル構造](#ファイル構造)
3. [型ヒントとジェネリクス](#型ヒントとジェネリクス)
4. [命名規則](#命名規則)
5. [インポート](#インポート)
6. [クラス設計](#クラス設計)
7. [関数設計](#関数設計)
8. [ドキュメント](#ドキュメント)
9. [例外処理](#例外処理)
10. [ツール設定](#ツール設定)

---

## 🎯 基本原則

### 必須要件

- **Python 3.12以降**: PEP 695のネイティブジェネリクス構文を使用
- **型安全性**: すべての公開APIに完全な型ヒント
- **明示的 > 暗黙的**: 動作を推測させない明確なコード
- **ドキュメント駆動**: すべての公開APIに詳細なdocstring

### コードフォーマット

- **Black**: 自動フォーマッター（設定はデフォルト）
- **行長**: 最大88文字（Blackデフォルト）
- **インデント**: スペース4つ
- **クォート**: ダブルクォート優先（Blackに従う）

### 型チェック

- **Pyright**: 厳密モード（strict）
- **型スタブ**: `typings/` ディレクトリに外部ライブラリ用スタブ配置
- **型チェック無効化**: 必要最小限、コメントで理由を明記

---

## 📁 ファイル構造

### ファイルヘッダー

すべての`.py`ファイルに以下のヘッダーを含める：

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

# pyright: ... (必要な場合のみ)

from __future__ import annotations  # ← 必須
```

**構造**:
1. Apacheライセンスヘッダー（必須）
2. Pyright指示コメント（必要な場合のみ）
3. `from __future__ import annotations`（必須）
4. 空行
5. その他のimport文

### Pyright指示（必要に応じて）

```python
# pyright: reportUnnecessaryIsInstance=false
# pyright: reportWildcardImportFromLibrary=false
```

**ルール**: 
- ファイルレベルで無効化する場合のみ使用
- 理由が明確な場合に限定
- 広範囲な無効化は避ける
- **`# type: ignore` ではなく `# pyright: ignore` を使用**

```python
# ✅ Pyright専用の無効化
value = get_value()  # pyright: ignore[reportUnknownVariableType]

# ❌ 汎用的な type: ignore は使用しない
value = get_value()  # type: ignore
```

**理由**:
- Pyright固有のエラーコードを指定できる
- 他の型チェッカー（mypy等）との混乱を避ける
- より明確なエラー抑制

### モジュールdocstring

```python
"""Brief module description.

More detailed explanation of the module's purpose, key classes,
and usage patterns.

Example:
    Basic usage::

        >>> from cslrtools2.lmpipe import Estimator
        >>> estimator = Estimator()
"""
```

---

## 🔤 型ヒントとジェネリクス

### PEP 695 ジェネリクス構文（必須）

**✅ 正しい（Python 3.12+）:**

```python
from __future__ import annotations

# 型エイリアス
type PathLike = _PathLike[str] | str

# ジェネリッククラス
class Estimator[K: str]:
    """Estimator for landmark detection."""
    
    def process(self) -> ProcessResult[K]:
        ...

# ジェネリック関数
def transform[T](data: T) -> T:
    ...

# 複数型パラメータ
class SLDataset[Kmeta: str, Kvid: str, Klm: str, Ktgt: str]:
    ...
```

**❌ 古い構文（使用禁止）:**

```python
from typing import Generic, TypeVar

K = TypeVar("K", bound=str)

class Estimator(Generic[K]):  # ❌ 古い構文
    ...
```

### 型制約

```python
# 文字列リテラル型制約
class ProcessResult[K: str]:
    ...

# Literal型の使用
from typing import Literal

type ExecutorMode = Literal["batch", "frames"] | None
type ExistRule = Literal["skip", "overwrite", "suffix", "error"]
```

### Union型の表記

```python
# ✅ Python 3.10+ の | 演算子を使用
def func(value: int | str | None) -> bool:
    ...

# ✅ None との Union は明示的に記述
def func(value: int | None) -> str | None:
    ...

# ❌ typing.Union は使用しない
from typing import Union
def func(value: Union[int, str, None]) -> bool:  # ❌
    ...

# ❌ typing.Optional は使用しない
from typing import Optional
def func(value: Optional[int]) -> Optional[str]:  # ❌
    ...
```

**ルール**: `Optional[T]` の代わりに `T | None` を常に使用

**理由**:
- 明示的で読みやすい
- 他のUnion型と一貫性がある
- Python 3.10+ の標準スタイル

### `from __future__ import annotations`

**ルール**: **すべての`.py`ファイルで必須**

前方参照の有無にかかわらず、常にファイルの先頭（ライセンスヘッダーとPyright指示の後）に配置：

```python
# Copyright header...
# pyright: ... (if needed)

from __future__ import annotations  # ← 必須

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .options import LMPipeOptions  # 循環import回避

class Collector[K: str]:
    def configure_from_options(self, options: LMPipeOptions) -> None:  # クォート不要
        ...
```

**理由**:
- 型ヒントの文字列化により前方参照の問題を回避
- すべてのファイルで統一された動作
- 型チェックとランタイムの一貫性

### TYPE_CHECKINGブロック

**用途**: 型チェック専用のimport（循環import回避）

**重要**: `TYPE_CHECKING`でインポートした型は、ランタイムで解決できるように`else`ブロックで定義すること

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..options import LMPipeOptions  # 循環import回避
    from typing import Protocol  # 型チェックのみで使用
else:
    # ランタイムでの型解決用（文字列として評価される）
    LMPipeOptions = "LMPipeOptions"
    Protocol = object  # または適切なフォールバック
```

**循環importが発生しない場合は通常のimportを使用**:

```python
# ✅ 循環importなし → 通常のimport
from ..options import LMPipeOptions

class Collector[K: str]:
    def configure_from_options(self, options: LMPipeOptions) -> None:
        ...
```

**理由**:
- `from __future__ import annotations` により型ヒントは文字列化される
- ランタイムでアクセスされる可能性のある型（`__annotations__`など）のため
- 型チェックとランタイムの一貫性を保つ

---

## 🏷️ 命名規則

### 基本ルール

| 種類 | 命名規則 | 例 |
|------|---------|-----|
| モジュール | `lowercase` | `estimator.py`, `array_loader.py` |
| パッケージ | `lowercase` | `lmpipe`, `sldataset` |
| クラス | `PascalCase` | `Estimator`, `ProcessResult` |
| 関数/メソッド | `snake_case` | `conv_size`, `process_frame` |
| 変数 | `snake_case` | `frame_id`, `landmark_matrix` |
| 定数 | `UPPER_SNAKE_CASE` | `DEFAULT_OPTIONS`, `MAX_WORKERS` |
| 型エイリアス | `PascalCase` | `PathLike`, `ArrayLike` |
| 型パラメータ | `K`, `T`, `Kmeta` | `K: str`, `T: Bound` |

### プラグイン命名

```python
# プロバイダープレフィックス + 機能名
class MediaPipeHolisticEstimator(Estimator[MediaPipeHolisticKey]):
    ...

class MediaPipePoseEstimator(Estimator[MediaPipePoseKey]):
    ...

# プラグインエントリーポイント名
"mediapipe.holistic" = "cslrtools2.plugins.mediapipe.lmpipe.holistic_args:holistic_info"
```

### キー型命名

```python
# {Provider}{Part}Key パターン
type MediaPipePoseKey = Literal["mediapipe.pose"]
type MediaPipeHandKey = Literal["mediapipe.left_hand", "mediapipe.right_hand"]
type MediaPipeHolisticKey = (
    MediaPipePoseKey |
    MediaPipeHandKey |
    MediaPipeFaceKey
)
```

### プライベートメンバー

```python
class Example:
    # プライベート（1つのアンダースコア）
    def _internal_method(self) -> None:
        ...
    
    # 名前修飾が必要な場合（2つのアンダースコア）
    def __mangled_method(self) -> None:
        ...
```

---

## 📦 インポート

### インポート順序

**1. 標準ライブラリ**
**2. サードパーティライブラリ**
**3. ローカルパッケージ（絶対import）**
**4. ローカルパッケージ（相対import）**

各グループは空行で区切る：

```python
# Copyright header...
# pyright: ... (if needed)

from __future__ import annotations  # 必須、最初のimport

# 標準ライブラリ
from abc import ABC, abstractmethod
from typing import Any, Mapping, Callable
from dataclasses import dataclass
from pathlib import Path

# サードパーティライブラリ
import numpy as np
import torch
from torch import Tensor

# ローカルパッケージ（絶対import）
from cslrtools2.typings import ArrayLike, PathLike

# ローカルパッケージ（相対import）
from ..estimator import ProcessResult
from ..options import LMPipeOptions
from .base import Collector
```

### 絶対import vs 相対import

**絶対importを使用する場合**:
- **pluginsモジュール**: 外部プロジェクトから呼び出される可能性があるコード
- **公開API**: 他のプロジェクトで使用されるモジュール
- **再利用可能なコンポーネント**: 独立して使用できるユーティリティ

```python
# plugins/mediapipe/lmpipe/holistic.py
from __future__ import annotations

# ✅ 絶対import（外部から参照される可能性）
from cslrtools2.lmpipe.estimator import Estimator
from cslrtools2.typings import MatLike

class MediaPipeHolisticEstimator(Estimator[MediaPipeHolisticKey]):
    ...
```

**相対importを使用する場合**:
- **内部実装**: パッケージ内部でのみ使用されるコード
- **密結合なモジュール**: 同じサブパッケージ内の関連モジュール
- **プライベートAPI**: 外部に公開されない内部機能

```python
# lmpipe/collector/landmark_matrix/csv_lmsc.py
from __future__ import annotations

# ✅ 相対import（内部実装）
from ..base import LandmarkMatrixSaveCollector
from ...options import LMPipeOptions

class CSVLandmarkMatrixSaveCollector(LandmarkMatrixSaveCollector[K]):
    ...
```

**判断基準**:
- 外部プロジェクトから `from cslrtools2.plugins.X import Y` される？ → 絶対import
- パッケージ内部でのみ使用？ → 相対import
- 迷ったら絶対importを使用（安全側）

### インポートスタイル

```python
# ✅ 推奨: 明示的インポート
from typing import Any, Mapping, Callable

# ✅ OK: モジュールインポート（長い名前の場合）
import numpy as np
import torch.nn as nn

# ⚠️ 注意: * インポートは避ける（特殊なケースのみ）
from typing import *  # pyright: ignore[reportWildcardImportFromLibrary]

# ❌ 非推奨: 不要なインポート
from typing import List, Dict  # Python 3.9+ では list, dict を使用
```

### 型チェック専用インポート

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .options import LMPipeOptions  # 循環import回避
    from typing import Protocol  # 型チェックのみで使用
```

---

## 🏛️ クラス設計

### 抽象基底クラス（ABC）

```python
from abc import ABC, abstractmethod

class Estimator[K: str](ABC):
    """Abstract base class for estimators.
    
    Type Parameters:
        K: String literal type for landmark keys.
    """
    
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Return expected output shape."""
        ...
    
    @abstractmethod
    def estimate(self, frame: MatLike) -> Mapping[K, NDArrayFloat]:
        """Estimate landmarks from a frame."""
        ...
```

### データクラス

```python
from dataclasses import dataclass

@dataclass
class ProcessResult[K: str]:
    """Result of frame processing.
    
    Attributes:
        frame_id (`int`): Frame identifier.
        headers (`Mapping[K, NDArrayStr]`): Landmark headers.
        landmarks (`Mapping[K, NDArrayFloat]`): Landmark coordinates.
        annotated_frame (`MatLike`): Annotated frame.
    """
    frame_id: int
    headers: Mapping[K, NDArrayStr]
    landmarks: Mapping[K, NDArrayFloat]
    annotated_frame: MatLike
```

### Protocolクラス（構造的型付け）

```python
from typing import Protocol

class SupportsArray[T](Protocol):
    """Protocol for objects supporting __array__."""
    
    def __array__(self) -> T:
        ...
```

### clipar namespace（CLI引数定義）

```python
from clipar import namespace, mixin

@namespace
class MediaPipeHolisticArgs(MediaPipeBaseArgs, mixin.ReprMixin):
    """Arguments for MediaPipe Holistic estimator."""
    
    model_complexity: int = 1
    "The complexity of the model: 0, 1, or 2."
    
    smooth_landmarks: bool = True
    "Whether to smooth the landmarks."
```

**特徴**:
- クラス変数に型と初期値を指定
- ドキュメントは文字列リテラルで直後に記述
- `mixin.ReprMixin` で自動的に `__repr__` 実装

---

## 🔧 関数設計

### 型ヒント

```python
def conv_size(
    size: Tensor,
    kernel_size: Tensor,
    stride: Tensor,
    padding: Tensor,
    dilation: Tensor
) -> Tensor:
    """Calculate convolution output size.
    
    Args:
        size (`Tensor`): Input spatial dimensions.
        kernel_size (`Tensor`): Kernel size.
        stride (`Tensor`): Stride.
        padding (`Tensor`): Padding.
        dilation (`Tensor`): Dilation.
    
    Returns:
        :class:`Tensor`: Output spatial dimensions.
    """
    return torch.floor_divide(
        size + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride
    ) + 1
```

### オーバーロード

```python
from typing import overload

class ConvSize:
    @overload
    def forward(self, size: Tensor) -> Tensor: ...
    
    @overload
    def forward(self, size: int, *sizes: int) -> Tensor: ...
    
    def forward(self, size: Tensor | int, *sizes: int):
        """Forward pass with multiple signatures.
        
        Note: 戻り値の型ヒントは省略可能（オーバーロードで定義済み）
        """
        if isinstance(size, Tensor):
            return self._forward_tensor(size)
        return self._forward_ints(size, *sizes)
```

**ルール**:
- `@overload` で全署名を定義（戻り値を含む完全な型情報）
- **実装メソッドの戻り値型ヒントは省略可能**（オーバーロードで定義されているため）
- 実装の引数型ヒントは記述する（可能な限りエラーを解消）
- `reportInconsistentOverload` の無効化は**極力避ける**
- どうしてもエラーを解消できない場合のみ、**理由をコメントで明記**:

```python
# ❌ 理由なしの無効化（避ける）
def forward(  # pyright: ignore[reportInconsistentOverload]
    self,
    size: Tensor | int,
    *sizes: int
) -> Tensor:
    ...

# ✅ やむを得ない場合は理由を記述
def forward(
    # pyright: ignore[reportInconsistentOverload]
    # Reason: Cannot express "first arg int requires *sizes" constraint in Python type system.
    # Overloads guarantee correct usage, runtime validation handles invalid calls.
    self,
    size: Tensor | int,
    *sizes: int
) -> Tensor:
    ...
```

**推奨アプローチ**:
1. オーバーロードのシグネチャを調整してエラーを解消
2. 実装の型ヒントを調整（Union型、Protocol使用など）
3. それでも解消できない場合のみ、理由を明記して無効化

### デフォルト引数

```python
# ✅ イミュータブル
def process(value: int = 0, name: str = "") -> None:
    ...

# ✅ None でデフォルト化
def process(data: list[int] | None = None) -> None:
    if data is None:
        data = []
    ...

# ❌ ミュータブルなデフォルト引数は禁止
def process(data: list[int] = []) -> None:  # ❌ バグの原因
    ...
```

---

## 📝 ドキュメント

### Docstringスタイル

**Google Style + Sphinx** を使用。詳細は [DOCSTRING_STYLE_GUIDE.md](DOCSTRING_STYLE_GUIDE.md) 参照。

**キーポイント**:

```python
def function(param1: Type1, param2: Type2) -> ReturnType:
    """Brief description (one line).
    
    Detailed description with references to :class:`OtherClass`
    and :func:`other_function`.
    
    Args:
        param1 (`Type1`): Description. Use backticks for types in Args.
        param2 (`Type2`): Description. Returns :obj:`None` if invalid.
    
    Returns:
        :class:`ReturnType`: Description. Use Sphinx roles for Returns.
    
    Raises:
        :exc:`ValueError`: When param1 is :obj:`None`.
    
    Example:
        Usage example::
        
            >>> result = function(value1, value2)
            >>> print(result)
            expected_output
    """
```

### モジュールレベルdocstring

```python
"""Landmark extraction pipeline components.

This module provides the core abstractions for building landmark
detection pipelines, including :class:`Estimator` and :class:`Collector`.

The estimator processes video frames to extract body landmarks using
models like MediaPipe, while collectors handle result storage in
various formats (CSV, JSON, Zarr, etc.).

Example:
    Basic pipeline usage::
    
        >>> from cslrtools2.lmpipe import Estimator
        >>> estimator = MediaPipeHolisticEstimator(args)
        >>> results = estimator.process(video_path)
"""
```

---

## 🚨 例外処理

### カスタム例外階層

詳細は [EXCEPTION_LOGGING_STYLE_GUIDE.md](EXCEPTION_LOGGING_STYLE_GUIDE.md) 参照。

```python
from cslrtools2.exceptions import (
    ValidationError,
    EstimatorError,
    DataLoadError
)

# ✅ 適切な例外使用
def validate_input(value: int) -> None:
    if value < 0:
        raise ValidationError(
            f"Expected non-negative integer, got {value}"
        )

# ✅ コンテキスト情報を含める
def load_model(path: Path) -> Model:
    try:
        return Model.load(path)
    except FileNotFoundError as e:
        raise EstimatorError(
            f"Model file not found: {path}. "
            f"Ensure the model has been downloaded."
        ) from e
```

### 例外チェーン

```python
# ✅ from e で原因を保持
try:
    data = load_data(path)
except IOError as e:
    raise DataLoadError(f"Failed to load {path}") from e

# ❌ 原因を破棄しない
try:
    data = load_data(path)
except IOError:
    raise DataLoadError(f"Failed to load {path}")  # ❌ 原因が失われる
```

---

## 🛠️ ツール設定

### pyproject.toml 抜粋

```toml
[project]
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.black]
line-length = 88
target-version = ['py312']

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"
reportMissingTypeStubs = false
```

### Pyright設定

**厳密モード有効**:
- すべての型ヒント必須
- `Any` の使用を最小化
- 型安全性の最大化

**無効化する場合のみコメント**:

```python
# pyright: reportUnnecessaryIsInstance=false
def func(value: int | str) -> int:
    if isinstance(value, int):  # Pyrightは不要と判断するが、ランタイムで必要
        return value
    return int(value)
```

### pytest設定

```python
import pytest

# MediaPipe必須のテストにマーカー
@pytest.mark.mediapipe
def test_mediapipe_estimator():
    ...

# テストスキップ
@pytest.mark.skipif(
    not HAS_MEDIAPIPE,
    reason="MediaPipe not installed"
)
def test_with_mediapipe():
    ...
```

---

## ✅ チェックリスト

### 新規ファイル作成時

- [ ] Apacheライセンスヘッダー追加
- [ ] `from __future__ import annotations` 追加（**必須**）
- [ ] Pyright指示コメント（必要な場合のみ）
- [ ] モジュールdocstring記述
- [ ] インポート順序確認（標準→サードパーティ→ローカル）
- [ ] 絶対import vs 相対importの判断（plugins等は絶対import）
- [ ] すべての公開APIに型ヒント
- [ ] PEP 695ジェネリクス構文使用
- [ ] `Optional[T]` ではなく `T | None` を使用
- [ ] `TYPE_CHECKING` ブロックには `else` でランタイム定義

### クラス作成時

- [ ] 抽象基底クラスの場合、`ABC` と `@abstractmethod` 使用
- [ ] データクラスの場合、`@dataclass` 使用
- [ ] ジェネリック型パラメータを適切に定義
- [ ] クラスdocstring記述（Attributes含む）
- [ ] すべての公開メソッドにdocstring

### 関数作成時

- [ ] 完全な型ヒント（引数と戻り値）
- [ ] `Optional[T]` ではなく `T | None` を使用
- [ ] docstring記述（Args, Returns, Raises, Example）
- [ ] デフォルト引数はイミュータブル
- [ ] 例外は適切なカスタム例外クラス使用
- [ ] オーバーロード使用時は `reportInconsistentOverload` の無効化を避ける

### コミット前

- [ ] `black .` でフォーマット
- [ ] `pyright` でエラーなし
- [ ] `pytest` ですべてのテストパス
- [ ] 新規公開APIにはテスト追加
- [ ] ドキュメント更新

---

## 📚 参考資料

### プロジェクト内ガイド

- [DOCSTRING_STYLE_GUIDE.md](DOCSTRING_STYLE_GUIDE.md) - Docstring詳細
- [EXCEPTION_LOGGING_STYLE_GUIDE.md](EXCEPTION_LOGGING_STYLE_GUIDE.md) - 例外とログ
- [BRANCHING_STRATEGY.md](BRANCHING_STRATEGY.md) - ブランチ戦略

### 外部リソース

- [PEP 8](https://peps.python.org/pep-0008/) - Python Style Guide
- [PEP 695](https://peps.python.org/pep-0695/) - Type Parameter Syntax
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pyright Documentation](https://github.com/microsoft/pyright)
- [Black Code Style](https://black.readthedocs.io/en/stable/)

---

## 🔄 移行ガイド（既存コードの更新）

### 1. `from __future__ import annotations` の追加

**すべての`.py`ファイルに追加**:

```python
# Copyright header...
# pyright: ... (if needed)

from __future__ import annotations  # ← 追加

# 以下、既存のimport文
```

### 2. 旧型ヒント → PEP 695

**Before (Python 3.11以前):**

```python
from typing import Generic, TypeVar

K = TypeVar("K", bound=str)
T = TypeVar("T")

class Estimator(Generic[K]):
    def process(self) -> ProcessResult[K]:
        ...
```

**After (Python 3.12+):**

```python
from __future__ import annotations

class Estimator[K: str]:
    def process(self) -> ProcessResult[K]:
        ...
```

### 3. Union → | 演算子、Optional 排除

**Before:**

```python
from typing import Union, Optional

def func(value: Union[int, str]) -> Optional[bool]:
    ...
```

**After:**

```python
from __future__ import annotations

def func(value: int | str) -> bool | None:
    ...
```

### 4. 型エイリアス

**Before:**

```python
from typing import TypeAlias

PathLike: TypeAlias = _PathLike[str] | str
```

**After:**

```python
from __future__ import annotations

type PathLike = _PathLike[str] | str
```

### 5. TYPE_CHECKING ブロックの更新

**Before:**

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .options import LMPipeOptions
```

**After:**

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .options import LMPipeOptions
else:
    LMPipeOptions = "LMPipeOptions"  # ランタイム解決用
```

### 6. type: ignore → pyright: ignore

**Before:**

```python
value = func()  # type: ignore
result = process(data)  # type: ignore[arg-type]
```

**After:**

```python
from __future__ import annotations

value = func()  # pyright: ignore[reportUnknownVariableType]
result = process(data)  # pyright: ignore[reportArgumentType]
```

### 7. 絶対import への変更（plugins等）

**Before:**

```python
# plugins/mediapipe/lmpipe/holistic.py
from ...lmpipe.estimator import Estimator
```

**After:**

```python
# plugins/mediapipe/lmpipe/holistic.py
from __future__ import annotations

from cslrtools2.lmpipe.estimator import Estimator  # 絶対import
```

---

**最終更新**: 2025年11月16日  
**適用範囲**: src/cslrtools2/ 全モジュール
