# 型スタブ検証レポート

**検証日**: 2025年11月14日  
**対象**: MediaPipe connections モジュールの型定義  
**結論**: ✅ 型スタブは不要

---

## 🔍 検証プロセス

### 1. 初期状態のエラー
型スタブなしで `connections.py` をインポート:
```
❌ "HAND_CONNECTIONS" の種類が部分的に不明です
   型は "frozenset[Unknown]" です
❌ "FACEMESH_CONTOURS" の種類が部分的に不明です
   型は "frozenset[Unknown | tuple[int, int]]" です
```

### 2. 型スタブを作成 (第1案)
以下の5ファイルを作成:
- `typings/mediapipe/python/solutions/pose.pyi` (38行)
- `typings/mediapipe/python/solutions/hands.pyi` (26行)
- `typings/mediapipe/python/solutions/face_mesh.pyi` (18行)
- `typings/mediapipe/python/solutions/__init__.pyi`
- `typings/mediapipe/python/__init__.pyi`

**結果**: エラーは継続 (型スタブが認識されない)

### 3. 型スタブを簡略化 (第2案)
Enum定義を削除し、接続定数のみに:
```python
# hands.pyi (4行に削減)
HAND_CONNECTIONS: frozenset[tuple[int, int]]
```

**結果**: エラーは継続

### 4. 明示的な型アノテーション (最終解)
`connections.py` 内で直接型を宣言:
```python
# インポート時にプライベート名で取得
from mediapipe.python.solutions.pose import POSE_CONNECTIONS as _POSE_CONNECTIONS

# 明示的な型アノテーション付きで再エクスポート
POSE_CONNECTIONS: "frozenset[tuple[int, int]]" = _POSE_CONNECTIONS
```

**結果**: ✅ すべてのエラーが解消

### 5. 型スタブ削除の検証
`typings/mediapipe/` ディレクトリ全体を削除:
```powershell
Remove-Item -Recurse -Force "typings/mediapipe"
```

**結果**: ✅ エラーなし → **型スタブは不要**

---

## ✅ 最終的な解決策

### connections.py の型定義パターン

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import FrozenSet, Tuple

# プライベート名でインポート
from mediapipe.python.solutions.pose import (
    POSE_CONNECTIONS as _POSE_CONNECTIONS,
)

# 明示的な型アノテーション付きで再エクスポート
POSE_CONNECTIONS: "frozenset[tuple[int, int]]" = _POSE_CONNECTIONS

if TYPE_CHECKING:
    # 型エイリアス (オプション)
    ConnectionSet = FrozenSet[Tuple[int, int]]
```

### 重要なポイント

1. **`as _PRIVATE_NAME` パターン**
   - MediaPipeからプライベート名でインポート
   - 型情報がない状態でインポート

2. **明示的な型アノテーション**
   - `CONSTANT: "frozenset[tuple[int, int]]" = _PRIVATE_CONSTANT`
   - 文字列リテラル型を使用 (PEP 563対応)

3. **`from __future__ import annotations`**
   - 型アノテーションを文字列として評価
   - Python 3.12で必須

4. **`# pyright: ignore[reportMissingTypeStubs]`**
   - インポート行の警告のみ抑制
   - 実際の使用箇所では型チェックが効く

---

## 📊 型スタブ vs 明示的アノテーション

| 項目 | 型スタブ (`.pyi`) | 明示的アノテーション |
|------|------------------|---------------------|
| **ファイル数** | 5ファイル | 0ファイル (connections.pyのみ) |
| **コード行数** | 82行 | 14行 |
| **保守性** | 別ファイル管理 | 同一ファイルで完結 |
| **Pyright認識** | ❌ 認識しない | ✅ 完全認識 |
| **IDE補完** | ❌ 不安定 | ✅ 動作 |
| **メリット** | 理論上は型定義分離 | シンプル・確実 |
| **デメリット** | 複雑・動作不安定 | なし |

---

## 🎯 推奨事項

### ✅ 採用する方法
**明示的な型アノテーション** (connections.py内)
- 型スタブは削除済み
- すべてのエラーが解消
- テストも正常に動作

### ❌ 採用しない方法
**外部型スタブ (.pyi ファイル)**
- MediaPipe のパッケージ構造が複雑
- Pyright が認識しない
- 保守コストが高い

---

## 📝 実装後の確認

### エラーチェック結果
```bash
$ get_errors connections.py
No errors found ✓
```

### テスト結果
```bash
$ python test_connections.py
✅ All tests passed!
  - 35 pose skeleton connections
  - 21 hand skeleton connections
  - 2556 face mesh tesselation connections
  - 124 face contour connections (recommended)
```

### 削除されたファイル
- `typings/mediapipe/python/solutions/pose.pyi`
- `typings/mediapipe/python/solutions/hands.pyi`
- `typings/mediapipe/python/solutions/face_mesh.pyi`
- `typings/mediapipe/python/solutions/__init__.pyi`
- `typings/mediapipe/python/__init__.pyi`

**削減**: 5ファイル、82行 → 0ファイル

---

## 🧪 技術的な学び

### Pyrightの型推論の限界
MediaPipeのような stub がないサードパーティライブラリからのインポートは:
1. 型情報が `Unknown` になる
2. 外部 `.pyi` を配置しても認識されない場合がある
3. **解決策**: モジュール内で明示的に型を宣言

### 最適な型宣言パターン (Python 3.12+)
```python
from __future__ import annotations  # 必須

# 型なしでインポート
from external_lib import CONST as _CONST

# 明示的に型を付与
CONST: "frozenset[tuple[int, int]]" = _CONST
```

このパターンは:
- ✅ Pyright が完全認識
- ✅ IDE 補完が動作
- ✅ 型チェックが効く
- ✅ 実行時オーバーヘッドなし

---

## ✨ まとめ

**結論**: 型スタブは完全に不要

**理由**:
1. connections.py 内の明示的な型アノテーションで十分
2. 型スタブは Pyright に認識されなかった
3. 保守コストの削減 (5ファイル削除)
4. すべてのテストが正常動作

**成果**:
- ✅ 0個のエラー
- ✅ 完全な型チェック
- ✅ IDE自動補完サポート
- ✅ コードの簡潔性向上

**教訓**: 外部ライブラリの型定義は、可能な限りモジュール内で明示的に行うべき。
