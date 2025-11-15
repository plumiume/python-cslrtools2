# コーディングスタイルガイド違反リスト

**作成日**: 2025年11月16日  
**目的**: 現在のコードベースでスタイルガイドに準拠していない箇所を列挙  
**次のステップ**: これらの違反を修正してコードベース全体を標準化

---

## 📊 違反概要

| 違反タイプ | 件数 | 重要度 |
|-----------|------|--------|
| `from __future__ import annotations` 欠落 | 56ファイル | 🔴 高 |
| `# type: ignore` の使用（`# pyright: ignore` を使うべき） | 15箇所 | 🟡 中 |
| `TYPE_CHECKING` で `else` ブロック欠落 | 3ファイル | 🟡 中 |
| `Optional[T]` の使用（`T \| None` を使うべき） | 0件 | ✅ なし |

---

## 🔴 高優先度: `from __future__ import annotations` 欠落

**違反**: すべての `.py` ファイルで必須だが、以下56ファイルで欠落

### コアモジュール (6ファイル)

```
src/cslrtools2/__init__.py
src/cslrtools2/_root.py
src/cslrtools2/_version.py
src/cslrtools2/convsize.py
src/cslrtools2/exceptions.py
src/cslrtools2/logger.py
```

### lmpipe モジュール (25ファイル)

```
src/cslrtools2/lmpipe/__init__.py
src/cslrtools2/lmpipe/logger.py
src/cslrtools2/lmpipe/options.py
src/cslrtools2/lmpipe/typings.py
src/cslrtools2/lmpipe/utils.py
src/cslrtools2/lmpipe/app/args.py
src/cslrtools2/lmpipe/app/cli.py
src/cslrtools2/lmpipe/app/mp_rich.py
src/cslrtools2/lmpipe/app/plugins.py
src/cslrtools2/lmpipe/app/runner.py
src/cslrtools2/lmpipe/app/holistic/estimator.py
src/cslrtools2/lmpipe/app/holistic/roi.py
src/cslrtools2/lmpipe/collector/__init__.py
src/cslrtools2/lmpipe/collector/base.py
src/cslrtools2/lmpipe/collector/annotated_frames/__init__.py
src/cslrtools2/lmpipe/collector/annotated_frames/base.py
src/cslrtools2/lmpipe/collector/annotated_frames/cv2_af.py
src/cslrtools2/lmpipe/collector/annotated_frames/matplotlib_af.py
src/cslrtools2/lmpipe/collector/annotated_frames/pil_af.py
src/cslrtools2/lmpipe/collector/annotated_frames/torchvision_af.py
src/cslrtools2/lmpipe/collector/landmark_matrix/__init__.py
src/cslrtools2/lmpipe/collector/landmark_matrix/base.py
src/cslrtools2/lmpipe/collector/landmark_matrix/csv_lmsc.py
src/cslrtools2/lmpipe/collector/landmark_matrix/json_lmsc.py
src/cslrtools2/lmpipe/collector/landmark_matrix/npy_lmsc.py
src/cslrtools2/lmpipe/collector/landmark_matrix/npz_lmsc.py
src/cslrtools2/lmpipe/collector/landmark_matrix/safetensors_lmsc.py
src/cslrtools2/lmpipe/collector/landmark_matrix/torch_lmsc.py
src/cslrtools2/lmpipe/collector/landmark_matrix/zarr_lmsc.py
src/cslrtools2/lmpipe/interface/executor.py
```

### plugins モジュール (14ファイル)

```
src/cslrtools2/plugins/__init__.py
src/cslrtools2/plugins/fluentsigners50/sldataset/__init__.py
src/cslrtools2/plugins/fluentsigners50/sldataset/main.py
src/cslrtools2/plugins/mediapipe/lmpipe/base_args.py
src/cslrtools2/plugins/mediapipe/lmpipe/base.py
src/cslrtools2/plugins/mediapipe/lmpipe/face_args.py
src/cslrtools2/plugins/mediapipe/lmpipe/face.py
src/cslrtools2/plugins/mediapipe/lmpipe/hand_args.py
src/cslrtools2/plugins/mediapipe/lmpipe/hand.py
src/cslrtools2/plugins/mediapipe/lmpipe/holistic_args.py
src/cslrtools2/plugins/mediapipe/lmpipe/holistic.py
src/cslrtools2/plugins/mediapipe/lmpipe/pose_args.py
src/cslrtools2/plugins/mediapipe/lmpipe/pose.py
src/cslrtools2/typings/__init__.py
```

### sldataset モジュール (8ファイル)

```
src/cslrtools2/sldataset/__init__.py
src/cslrtools2/sldataset/array_loader.py
src/cslrtools2/sldataset/logger.py
src/cslrtools2/sldataset/utils.py
src/cslrtools2/sldataset/app/args.py
src/cslrtools2/sldataset/app/cli.py
src/cslrtools2/sldataset/app/plugins.py
```

**修正方法**:
各ファイルの先頭（ライセンスヘッダーとPyright指示の後）に追加：

```python
# Copyright header...
# pyright: ... (if needed)

from __future__ import annotations  # ← 追加
```

---

## 🟡 中優先度: `# type: ignore` を `# pyright: ignore` に変更

**違反**: Pyright専用の無効化構文を使用すべき

### 15箇所で `# type: ignore` を使用

#### src/cslrtools2/convsize.py (2箇所)

```python
Line 213: def forward( # type: ignore[reportInconsistentOverload]
Line 317: def forward( # type: ignore[reportInconsistentOverload]
```

**修正**: 
```python
# Before
def forward( # type: ignore[reportInconsistentOverload]

# After (理由も追加)
def forward(
    # pyright: ignore[reportInconsistentOverload]
    # Reason: Cannot express "first arg Tensor XOR first arg int with *sizes" in Python type system.
```

#### src/cslrtools2/lmpipe/app/mp_rich.py (1箇所)

```python
Line 177: return renderable # type: ignore
```

**修正**:
```python
# Before
return renderable # type: ignore

# After
return renderable  # pyright: ignore[reportReturnType]
```

#### src/cslrtools2/lmpipe/collector/annotated_frames/matplotlib_af.py (5箇所)

```python
Line 110: self._ax.axis('off') # type: ignore
Line 114: self._im = self._ax.imshow(result.annotated_frame) # type: ignore
Line 118: self._ax.set_title(f"Frame {result.frame_id}") # type: ignore
Line 119: self._fig.canvas.draw() # type: ignore
Line 120: self._fig.canvas.flush_events() # type: ignore
```

**修正**:
```python
# Before
self._ax.axis('off') # type: ignore

# After
self._ax.axis('off')  # pyright: ignore[reportAttributeAccessIssue]
```

#### src/cslrtools2/lmpipe/collector/annotated_frames/cv2_af.py (6箇所)

```python
Line 113: return self._cv2.VideoWriter.fourcc(*self.fourcc)  # type: ignore
Line 118: return self._cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
Line 120: return self._cv2.VideoWriter.fourcc(*"XVID")  # type: ignore
Line 122: return self._cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
Line 124: return self._cv2.VideoWriter.fourcc(*"X264")  # type: ignore
Line 127: return self._cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore
```

**修正**:
```python
# Before
return self._cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore

# After
return self._cv2.VideoWriter.fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
```

#### src/cslrtools2/sldataset/transform/frozen.py (1箇所)

```python
Line 86: bc_type=self.bc_type, # type: ignore
```

**修正**:
```python
# Before
bc_type=self.bc_type, # type: ignore

# After
bc_type=self.bc_type,  # pyright: ignore[reportArgumentType]
```

---

## 🟡 中優先度: `TYPE_CHECKING` ブロックに `else` 定義が欠落

**違反**: `TYPE_CHECKING` でインポートした型はランタイム解決用に `else` ブロックでも定義すべき

### 3ファイルで `else` ブロック欠落

#### src/cslrtools2/lmpipe/collector/base.py

```python
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ..options import LMPipeOptions
```

**修正**:
```python
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ..options import LMPipeOptions
else:
    LMPipeOptions = "LMPipeOptions"
```

#### src/cslrtools2/lmpipe/collector/annotated_frames/matplotlib_af.py

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.pyplot as plt
```

**修正**:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.pyplot as plt
else:
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
```

または、ランタイムで実際にインポートするなら `TYPE_CHECKING` は不要：

```python
# TYPE_CHECKING 不要（オプショナル依存でなければ）
import matplotlib
import matplotlib.pyplot as plt
```

#### src/cslrtools2/typings/__init__.py

```python
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
```

**修正**:
```python
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
else:
    np = None  # type: ignore
    npt = None  # type: ignore
```

---

## ✅ 準拠している項目

### Optional[T] の使用

**検索結果**: 0件  
**状態**: ✅ すべてのファイルで `T | None` を使用（または未使用）

---

## 📝 修正の優先順位

### フェーズ1: 自動修正可能（スクリプトで一括処理）

1. **`from __future__ import annotations` の追加** (56ファイル)
   - 各ファイルの先頭に機械的に追加可能
   - エディタのマクロまたはスクリプトで一括処理

### フェーズ2: 手動レビュー必要

2. **`# type: ignore` → `# pyright: ignore`** (15箇所)
   - 適切なエラーコードを特定する必要がある
   - `reportInconsistentOverload`, `reportAttributeAccessIssue`, `reportArgumentType` など

3. **`TYPE_CHECKING` の `else` ブロック追加** (3ファイル)
   - ランタイムでの使用状況を確認
   - オプショナル依存か必須依存かを判断

---

## 🔧 一括修正スクリプト例

### PowerShell: `from __future__ import annotations` 追加

```powershell
Get-ChildItem -Path "src/cslrtools2" -Filter "*.py" -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    
    # すでに含まれている場合はスキップ
    if ($content -match 'from __future__ import annotations') {
        return
    }
    
    # ライセンスヘッダーと pyright 指示の後に挿入
    if ($content -match '(?s)(# Copyright.*?limitations under the License\.\s*\n)(# pyright:.*?\n)?') {
        $header = $Matches[1]
        $pyright = $Matches[2]
        $rest = $content.Substring($Matches[0].Length)
        
        $newContent = $header
        if ($pyright) { $newContent += $pyright }
        $newContent += "`nfrom __future__ import annotations`n"
        $newContent += $rest
        
        Set-Content -Path $_.FullName -Value $newContent -NoNewline
        Write-Host "Updated: $($_.FullName)"
    }
}
```

### Python: type: ignore 置換

```python
import re
from pathlib import Path

replacements = {
    r'# type: ignore\[reportInconsistentOverload\]': '# pyright: ignore[reportInconsistentOverload]',
    r'# type: ignore(?!\[)': '# pyright: ignore',
}

for py_file in Path('src/cslrtools2').rglob('*.py'):
    content = py_file.read_text(encoding='utf-8')
    modified = content
    
    for pattern, replacement in replacements.items():
        modified = re.sub(pattern, replacement, modified)
    
    if modified != content:
        py_file.write_text(modified, encoding='utf-8')
        print(f'Updated: {py_file}')
```

---

## 📋 修正チェックリスト

### フェーズ1: 自動修正
- [ ] `from __future__ import annotations` を56ファイルに追加
- [ ] Pyright/pytest で構文エラーがないことを確認
- [ ] コミット: `style: add 'from __future__ import annotations' to all Python files`

### フェーズ2: 手動修正
- [ ] `# type: ignore` を15箇所で `# pyright: ignore[ErrorCode]` に変更
- [ ] 適切なエラーコードを特定
- [ ] コミット: `style: replace 'type: ignore' with 'pyright: ignore'`

### フェーズ3: TYPE_CHECKING修正
- [ ] `lmpipe/collector/base.py` に `else: LMPipeOptions = "LMPipeOptions"` 追加
- [ ] `matplotlib_af.py` のTYPE_CHECKING使用を再検討
- [ ] `typings/__init__.py` に適切なフォールバック追加
- [ ] コミット: `style: add runtime fallbacks for TYPE_CHECKING imports`

### 最終確認
- [ ] `pyright src/cslrtools2` でエラーなし
- [ ] `pytest tests/` ですべてのテストパス
- [ ] スタイルガイドに準拠

---

**次のアクション**: フェーズ1の自動修正スクリプトを実行し、56ファイルに `from __future__ import annotations` を追加
