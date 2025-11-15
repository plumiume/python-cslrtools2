# cslrtools2 例外処理とログ出力スタイルガイド

**作成日**: 2025年11月13日  
**目的**: プロジェクト全体で統一された例外処理とログ出力のパターンを確立

---

## 📋 現状分析

### 現在の例外使用パターン

#### 使用されている標準例外
- `ValueError`: 不正な値や引数（最も多用）
- `TypeError`: 型の不一致
- `RuntimeError`: 実行時エラー（モデルダウンロード失敗等）
- `KeyError`: キーが見つからない（zarr, dict）
- `FileNotFoundError`: ファイルが存在しない
- `NotImplementedError`: 未実装機能

#### 問題点
- ❌ カスタム例外クラスが存在しない
- ❌ 例外階層が設計されていない
- ❌ エラーメッセージの書式が統一されていない
- ❌ コンテキスト情報の提供が不十分

### 現在のログ使用パターン

#### lmpipeモジュール
- ✅ `logging.getLogger("lmpipe")` を使用
- ✅ 構造化されたログフォーマット:
  ```
  %(asctime)s [%(levelname)s] %(name)s (%(pathname)s:%(lineno)d): %(message)s
  ```
- ✅ ログレベルの使い分け:
  - `debug`: 詳細な実行情報（フレーム処理、executor設定）
  - `info`: 重要な処理開始/完了（ファイル処理、バッチ処理）
  - `warning`: 警告（ユーザー割り込み）
  - `error`: エラー（ファイルが開けない、パス不正）

#### sldatasetモジュール
- ⚠️ ログ出力なし（例外のみ）

#### CLIアプリケーション
- ✅ Rich consoleを使用（進捗表示、視覚的フィードバック）
- ⚠️ 一部`print()`の使用

---

## 🎯 改善方針

### 1. カスタム例外階層の導入

**目的**: 
- エラーの分類を明確化
- より適切なエラーハンドリング
- ユーザーへの情報提供を改善

**設計原則**:
- すべてのカスタム例外は`CSLRToolsError`を継承
- サブパッケージごとに専用例外クラスを用意
- 既存の標準例外も適切に使い分け

### 2. ログ出力の標準化

**目的**:
- 全モジュールで統一されたログフォーマット
- 適切なログレベルの使用
- デバッグ性の向上

**設計原則**:
- 各サブパッケージで名前付きロガーを使用
- 構造化されたログメッセージ
- コンテキスト情報の提供

---

## 📐 例外クラス設計

### 基本階層

```python
"""Custom exceptions for cslrtools2."""

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


class ConfigurationError(CSLRToolsError):
    """Raised when configuration is invalid or inconsistent.
    
    This includes:
    - Invalid option combinations
    - Missing required configuration
    - Malformed configuration files
    
    Example:
        >>> raise ConfigurationError("Invalid estimator configuration: missing model_path")
    """
    pass


class ValidationError(CSLRToolsError):
    """Raised when input validation fails.
    
    This includes:
    - Invalid argument values
    - Type mismatches
    - Out-of-range values
    
    Example:
        >>> raise ValidationError(f"Expected positive integer, got {value}")
    """
    pass
```

### LMPipeサブパッケージ例外

```python
class LMPipeError(CSLRToolsError):
    """Base exception for landmark pipeline errors."""
    pass


class EstimatorError(LMPipeError):
    """Raised when landmark estimation fails.
    
    Example:
        >>> raise EstimatorError("MediaPipe model initialization failed")
    """
    pass


class CollectorError(LMPipeError):
    """Raised when result collection fails.
    
    Example:
        >>> raise CollectorError(f"Failed to write output to {path}: {reason}")
    """
    pass


class VideoProcessingError(LMPipeError):
    """Raised when video processing fails.
    
    Example:
        >>> raise VideoProcessingError(f"Cannot open video file: {path}")
    """
    pass


class ModelDownloadError(LMPipeError):
    """Raised when model download fails.
    
    Example:
        >>> raise ModelDownloadError(
        ...     f"Failed to download model from {url}. Status: {status_code}"
        ... )
    """
    pass
```

### SLDatasetサブパッケージ例外

```python
class SLDatasetError(CSLRToolsError):
    """Base exception for dataset errors."""
    pass


class DataLoadError(SLDatasetError):
    """Raised when data loading fails.
    
    Example:
        >>> raise DataLoadError(f"Failed to load array from {path}: {reason}")
    """
    pass


class DataFormatError(SLDatasetError):
    """Raised when data format is unexpected.
    
    Example:
        >>> raise DataFormatError(
        ...     f"Expected Tensor in file {path}, got {type(data)}"
        ... )
    """
    pass


class PluginError(SLDatasetError):
    """Raised when plugin loading or execution fails.
    
    Example:
        >>> raise PluginError(f"Plugin {name} is not a valid processor")
    """
    pass
```

---

## 📝 ログ出力設計

### ロガー構成

各サブパッケージで名前付きロガーを作成:

```python
# src/cslrtools2/lmpipe/logger.py (既存)
import logging

lmpipe_logger = logging.getLogger("cslrtools2.lmpipe")
```

```python
# src/cslrtools2/sldataset/logger.py (新規)
import logging

sldataset_logger = logging.getLogger("cslrtools2.sldataset")
```

```python
# src/cslrtools2/logger.py (新規 - 共通)
import logging

# Root logger for cslrtools2
root_logger = logging.getLogger("cslrtools2")

# Unified formatter
standard_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Detailed formatter (for debug)
detailed_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s (%(pathname)s:%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

### ログレベル使い分け

#### DEBUG
**用途**: 詳細な実行フロー、内部状態

**メッセージパターン**:
```python
logger.debug(f"Initializing estimator with config: {config}")
logger.debug(f"Processing frame {idx}/{total}")
logger.debug(f"Executor type: {type(executor).__name__}")
```

#### INFO
**用途**: 重要な処理の開始・完了、ユーザーアクション

**メッセージパターン**:
```python
logger.info(f"Starting video processing: {video_path}")
logger.info(f"Batch processing completed: {count} files processed")
logger.info(f"Output saved to: {output_path}")
```

#### WARNING
**用途**: 警告、非推奨機能、回復可能なエラー

**メッセージパターン**:
```python
logger.warning(f"File already exists, skipping: {path}")
logger.warning(f"Task interrupted by user")
logger.warning(f"Using fallback method due to: {reason}")
```

#### ERROR
**用途**: エラー、失敗した操作（例外を投げない場合）

**メッセージパターン**:
```python
logger.error(f"Failed to open video: {path}", exc_info=True)
logger.error(f"Invalid configuration: {reason}")
logger.error(f"Model download failed: {url} (status: {status})")
```

#### CRITICAL
**用途**: 致命的エラー、アプリケーション停止

**メッセージパターン**:
```python
logger.critical(f"System resource exhausted: {resource}")
logger.critical(f"Unrecoverable error in pipeline: {error}")
```

---

## 🎨 メッセージフォーマット規則

### 例外メッセージ

#### 基本パターン
```python
# ❌ 悪い例
raise ValueError("invalid value")

# ✅ 良い例
raise ValidationError(f"Invalid frame index: expected 0-{max_idx}, got {idx}")
```

#### 詳細情報を含める
```python
# ❌ 悪い例
raise RuntimeError("download failed")

# ✅ 良い例
raise ModelDownloadError(
    f"Failed to download model from {url}. "
    f"Status code: {response.status_code}. "
    f"Reason: {response.reason}"
)
```

#### 解決策を示唆
```python
# ✅ 良い例
raise DataFormatError(
    f"Expected a Tensor in file {path}, got {type(data).__name__}. "
    f"Ensure the file was saved with torch.save(tensor, path)"
)
```

### ログメッセージ

#### 構造化された情報
```python
# ❌ 悪い例
logger.info("processing video")

# ✅ 良い例
logger.info(f"Processing video: path={video_path}, frames={frame_count}, fps={fps}")
```

#### 進捗情報
```python
# ✅ 良い例
logger.info(f"Batch progress: {current}/{total} files ({current/total*100:.1f}%)")
```

#### コンテキスト付きエラー
```python
# ✅ 良い例
logger.error(
    f"Failed to process file: {file_path}",
    exc_info=True,  # スタックトレース付加
    extra={"file_path": file_path, "attempt": retry_count}
)
```

---

## 🔄 既存コード移行パターン

### パターン1: ValueError → ValidationError

**Before**:
```python
if not isinstance(value, int):
    raise ValueError(f"Expected int, got {type(value)}")
```

**After**:
```python
if not isinstance(value, int):
    raise ValidationError(f"Expected int, got {type(value).__name__}")
```

### パターン2: RuntimeError → 専用例外

**Before**:
```python
if response.status_code != 200:
    raise RuntimeError(f"Failed to download model from {url}")
```

**After**:
```python
if response.status_code != 200:
    raise ModelDownloadError(
        f"Failed to download model from {url}. "
        f"Status code: {response.status_code}"
    )
```

### パターン3: KeyError → DataLoadError

**Before**:
```python
if key not in data:
    raise KeyError(f"Array not found at path: {path}")
```

**After**:
```python
if key not in data:
    raise DataLoadError(f"Array not found at path: {path}")
```

### パターン4: ログ追加

**Before**:
```python
def process_file(path):
    # 処理...
    return result
```

**After**:
```python
def process_file(path):
    logger.info(f"Processing file: {path}")
    try:
        # 処理...
        logger.debug(f"File processed successfully: {path}")
        return result
    except Exception as e:
        logger.error(f"Failed to process file: {path}", exc_info=True)
        raise
```

---

## ✅ 実装チェックリスト

### Phase 1: 基盤整備
- [ ] `src/cslrtools2/exceptions.py` 作成
  - [ ] `CSLRToolsError` 基底クラス
  - [ ] `ConfigurationError`, `ValidationError` 共通例外
  - [ ] `LMPipeError`, `EstimatorError`, `CollectorError`, `VideoProcessingError`, `ModelDownloadError`
  - [ ] `SLDatasetError`, `DataLoadError`, `DataFormatError`, `PluginError`
  
- [ ] `src/cslrtools2/logger.py` 作成
  - [ ] ルートロガー設定
  - [ ] 標準フォーマッター
  - [ ] 詳細フォーマッター

- [ ] `src/cslrtools2/sldataset/logger.py` 作成
  - [ ] `sldataset_logger` 定義

### Phase 2: lmpipeモジュール移行
- [ ] `lmpipe/estimator.py` - エラーハンドリング改善
- [ ] `lmpipe/interface/__init__.py` - ログメッセージ改善
- [ ] `lmpipe/runspec.py` - `FileNotFoundError` → `VideoProcessingError`
- [ ] `lmpipe/app/cli.py` - エラーハンドリング追加
- [ ] `plugins/mediapipe/lmpipe/base.py` - `RuntimeError` → `ModelDownloadError`
- [ ] `plugins/mediapipe/lmpipe/holistic.py` - `ValueError` → `ValidationError`
- [ ] `plugins/mediapipe/lmpipe/hand.py` - `ValueError` → `ValidationError`

### Phase 3: sldatasetモジュール移行
- [ ] `sldataset/utils.py` - `KeyError` → `DataLoadError`
- [ ] `sldataset/array_loader.py` - `ValueError` → `DataFormatError`
- [ ] `sldataset/dataset.py` - ログ追加
- [ ] `sldataset/app/plugins.py` - `TypeError`/`ValueError` → `PluginError`
- [ ] `sldataset/app/cli.py` - エラーハンドリング改善

### Phase 4: ドキュメント更新
- [ ] 各例外クラスのdocstring充実
- [ ] API docに例外情報追加
- [ ] READMEにエラーハンドリング例追加

---

**最終更新**: 2025年11月13日  
**次のステップ**: 適用TODOリスト作成 → 実装実行
