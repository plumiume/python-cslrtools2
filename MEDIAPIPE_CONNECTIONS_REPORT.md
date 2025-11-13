# MediaPipe Connections 統合レポート

**作成日**: 2025年11月14日  
**対象**: `cslrtools2/plugins/mediapipe`  
**MediaPipeバージョン**: 0.10.14

## 📊 エグゼクティブサマリー

MediaPipeライブラリは骨格接続を定義する**14個の定数**を提供しており、これらを`cslrtools2`プラグインに統合することで:

- ✅ **保守性向上**: 自前の接続定義が不要
- ✅ **一貫性保証**: MediaPipe公式定義を使用
- ✅ **可視化機能**: アノテーション描画に直接利用可能
- ✅ **拡張性**: 将来のMediaPipeアップデートに自動対応

---

## 🔍 利用可能な接続定数

### 1. Pose (姿勢)

| 定数名 | 型 | 接続数 | 説明 |
|--------|-----|--------|------|
| `POSE_CONNECTIONS` | `frozenset[tuple[int, int]]` | 35 | 全身骨格の接続 |

**詳細**:
```python
from mediapipe.python.solutions.pose import POSE_CONNECTIONS

# 接続例: (0,1), (0,4), (1,2), (2,3), (3,7), (4,5), (5,6), (6,8), ...
# 顔特徴点 → 肩 → 腕 → 手 → 腰 → 脚 → 足を接続
```

**対応ランドマーク**: 33点 (MediaPipePoseNames に対応)

---

### 2. Hand (手)

| 定数名 | 型 | 接続数 | 説明 |
|--------|-----|--------|------|
| `HAND_CONNECTIONS` | `frozenset[tuple[int, int]]` | 21 | 手の骨格接続 |

**詳細**:
```python
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

# 接続例: (0,1), (0,5), (0,17), (1,2), (2,3), (3,4), ...
# 手首 → 親指/人差し指/中指/薬指/小指の各関節を接続
```

**対応ランドマーク**: 21点 (MediaPipeHandNames に対応)

---

### 3. Face Mesh (顔メッシュ)

| 定数名 | 型 | 接続数 | 説明 |
|--------|-----|--------|------|
| `FACEMESH_TESSELATION` | `frozenset[tuple[int, int]]` | 2,556 | 顔全体の三角形メッシュ |
| `FACEMESH_CONTOURS` | `frozenset[tuple[int, int]]` | 124 | 顔の輪郭線 |
| `FACEMESH_IRISES` | `frozenset[tuple[int, int]]` | 8 | 虹彩 (左右各4) |
| `FACEMESH_FACE_OVAL` | `frozenset[tuple[int, int]]` | 36 | 顔の楕円形輪郭 |
| `FACEMESH_LEFT_EYE` | `frozenset[tuple[int, int]]` | 16 | 左目 |
| `FACEMESH_RIGHT_EYE` | `frozenset[tuple[int, int]]` | 16 | 右目 |
| `FACEMESH_LEFT_EYEBROW` | `frozenset[tuple[int, int]]` | 8 | 左眉 |
| `FACEMESH_RIGHT_EYEBROW` | `frozenset[tuple[int, int]]` | 8 | 右眉 |
| `FACEMESH_LEFT_IRIS` | `frozenset[tuple[int, int]]` | 4 | 左虹彩 |
| `FACEMESH_RIGHT_IRIS` | `frozenset[tuple[int, int]]` | 4 | 右虹彩 |
| `FACEMESH_LIPS` | `frozenset[tuple[int, int]]` | 40 | 唇 |
| `FACEMESH_NOSE` | `frozenset[tuple[int, int]]` | 25 | 鼻 |

**詳細**:
```python
from mediapipe.python.solutions.face_mesh import (
    FACEMESH_TESSELATION,  # 高精度メッシュ (2556接続)
    FACEMESH_CONTOURS,     # 輪郭のみ (124接続) - 軽量版
    FACEMESH_IRISES,       # 虹彩検出用
    # ... その他10定数
)
```

**対応ランドマーク**: 468点 (通常) / 478点 (虹彩あり)

---

## 🎯 プラグインへの統合推奨事項

### Priority 1: 定数モジュールの作成 (必須)

**新規ファイル**: `src/cslrtools2/plugins/mediapipe/lmpipe/connections.py`

```python
"""MediaPipe connection constants for skeleton visualization.

This module re-exports official MediaPipe connection constants
for use in annotation and visualization tasks.

Example:
    >>> from cslrtools2.plugins.mediapipe.lmpipe.connections import (
    ...     POSE_CONNECTIONS,
    ...     HAND_CONNECTIONS,
    ...     FACEMESH_CONTOURS
    ... )
    >>> # Use in drawing functions
"""

from typing import FrozenSet, Tuple

# Pose connections
from mediapipe.python.solutions.pose import POSE_CONNECTIONS

# Hand connections
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

# Face mesh connections
from mediapipe.python.solutions.face_mesh import (
    FACEMESH_TESSELATION,
    FACEMESH_CONTOURS,
    FACEMESH_IRISES,
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
)

# Type alias for clarity
ConnectionSet = FrozenSet[Tuple[int, int]]

__all__ = [
    # Pose
    "POSE_CONNECTIONS",
    # Hand
    "HAND_CONNECTIONS",
    # Face Mesh
    "FACEMESH_TESSELATION",
    "FACEMESH_CONTOURS",
    "FACEMESH_IRISES",
    "FACEMESH_FACE_OVAL",
    "FACEMESH_LEFT_EYE",
    "FACEMESH_RIGHT_EYE",
    "FACEMESH_LEFT_EYEBROW",
    "FACEMESH_RIGHT_EYEBROW",
    "FACEMESH_LEFT_IRIS",
    "FACEMESH_RIGHT_IRIS",
    "FACEMESH_LIPS",
    "FACEMESH_NOSE",
    # Type
    "ConnectionSet",
]
```

**理由**:
- 🎯 単一のインポート元
- 📚 型ヒント付きで IDE サポート向上
- 🔄 MediaPipe バージョンアップ時の変更が1箇所で済む

---

### Priority 2: アノテーション機能の拡張 (推奨)

#### 2.1 Pose Estimator (`pose.py`)

**現在の状況**:
```python
# pose.py の @annotate デコレータ使用箇所を確認
# → 接続線描画機能が未実装または独自実装の可能性
```

**推奨実装**:
```python
# src/cslrtools2/plugins/mediapipe/lmpipe/pose.py

from .connections import POSE_CONNECTIONS

class MediaPipePoseEstimator:
    
    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        landmarks: NDArrayFloat,
        show_connections: bool = True,  # NEW parameter
        connection_color: tuple[int, int, int] = (0, 255, 0),  # NEW
        connection_thickness: int = 2,  # NEW
    ) -> MatLike:
        """Annotate pose landmarks on frame.
        
        Args:
            frame_src: Input frame
            landmarks: Pose landmarks (33, channels)
            show_connections: Whether to draw skeleton connections
            connection_color: RGB color for connection lines
            connection_thickness: Line thickness in pixels
            
        Returns:
            Annotated frame with landmarks and connections
        """
        frame = frame_src.copy()
        
        # Draw connections first (so landmarks appear on top)
        if show_connections:
            self._draw_connections(
                frame, landmarks, 
                POSE_CONNECTIONS, 
                connection_color, 
                connection_thickness
            )
        
        # Draw landmarks
        self._draw_landmarks(frame, landmarks)
        
        return frame
    
    def _draw_connections(
        self,
        frame: MatLike,
        landmarks: NDArrayFloat,
        connections: frozenset[tuple[int, int]],
        color: tuple[int, int, int],
        thickness: int
    ) -> None:
        """Draw skeleton connections between landmarks."""
        height, width = frame.shape[:2]
        
        for start_idx, end_idx in connections:
            # Get normalized coordinates
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            # Skip if either landmark is invalid (NaN or visibility < threshold)
            if np.isnan(start_lm).any() or np.isnan(end_lm).any():
                continue
            
            # Convert to pixel coordinates
            start_point = (
                int(start_lm[0] * width),
                int(start_lm[1] * height)
            )
            end_point = (
                int(end_lm[0] * width),
                int(end_lm[1] * height)
            )
            
            # Draw line
            cv2.line(frame, start_point, end_point, color, thickness)
```

---

#### 2.2 Hand Estimator (`hand.py`)

**推奨実装**:
```python
# src/cslrtools2/plugins/mediapipe/lmpipe/hand.py

from .connections import HAND_CONNECTIONS

class MediaPipeHandEstimator:
    
    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        landmarks: NDArrayFloat,
        show_connections: bool = True,
        connection_color: tuple[int, int, int] = (255, 0, 0),  # Red for hands
        connection_thickness: int = 2,
    ) -> MatLike:
        # Similar to pose implementation
        # Use HAND_CONNECTIONS
```

---

#### 2.3 Face Estimator (`face.py`)

**推奨実装**:
```python
# src/cslrtools2/plugins/mediapipe/lmpipe/face.py

from .connections import (
    FACEMESH_CONTOURS,      # Default: lightweight
    FACEMESH_TESSELATION,   # Option: high-quality mesh
    FACEMESH_IRISES,
    # ... other face connections
)

class MediaPipeFaceEstimator:
    
    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        landmarks: NDArrayFloat,
        mesh_mode: Literal["contours", "tesselation", "none"] = "contours",
        show_irises: bool = True,
        connection_color: tuple[int, int, int] = (0, 255, 255),  # Yellow
        connection_thickness: int = 1,  # Thinner for face
    ) -> MatLike:
        """Annotate face landmarks with customizable mesh density.
        
        Args:
            mesh_mode: "contours" (124 lines), "tesselation" (2556 lines), or "none"
            show_irises: Whether to draw iris connections
        """
        frame = frame_src.copy()
        
        # Choose connection set based on mode
        if mesh_mode == "contours":
            connections = FACEMESH_CONTOURS
        elif mesh_mode == "tesselation":
            connections = FACEMESH_TESSELATION
        else:
            connections = frozenset()
        
        # Draw mesh
        if connections:
            self._draw_connections(frame, landmarks, connections, ...)
        
        # Draw irises separately
        if show_irises and len(landmarks) >= 478:  # Check for iris landmarks
            self._draw_connections(
                frame, landmarks, FACEMESH_IRISES, 
                (255, 0, 255), 2  # Magenta, thicker
            )
        
        return frame
```

---

#### 2.4 Holistic Estimator (`holistic.py`)

**推奨実装**:
```python
# src/cslrtools2/plugins/mediapipe/lmpipe/holistic.py

from .connections import (
    POSE_CONNECTIONS,
    HAND_CONNECTIONS,
    FACEMESH_CONTOURS,
)

class MediaPipeHolisticEstimator:
    
    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        landmarks: NDArrayFloat,  # Combined: pose + hands + face
        show_pose_connections: bool = True,
        show_hand_connections: bool = True,
        show_face_mesh: bool = True,
    ) -> MatLike:
        """Annotate holistic landmarks (pose + hands + face)."""
        frame = frame_src.copy()
        
        # Extract landmark subsets
        pose_lms = landmarks[:33]  # Pose: 0-32
        left_hand_lms = landmarks[33:54]  # Left hand: 33-53
        right_hand_lms = landmarks[54:75]  # Right hand: 54-74
        face_lms = landmarks[75:]  # Face: 75-542 (468 landmarks)
        
        # Draw pose skeleton
        if show_pose_connections:
            self._draw_connections(
                frame, pose_lms, POSE_CONNECTIONS, (0, 255, 0), 2
            )
        
        # Draw hand skeletons
        if show_hand_connections:
            self._draw_connections(
                frame, left_hand_lms, HAND_CONNECTIONS, (255, 0, 0), 2
            )
            self._draw_connections(
                frame, right_hand_lms, HAND_CONNECTIONS, (0, 0, 255), 2
            )
        
        # Draw face mesh
        if show_face_mesh:
            self._draw_connections(
                frame, face_lms, FACEMESH_CONTOURS, (0, 255, 255), 1
            )
        
        return frame
```

---

### Priority 3: Type Stubs の追加 (オプション)

**新規ファイル**: `typings/mediapipe/python/solutions/__init__.pyi`

```python
"""Type stubs for MediaPipe connection constants."""

from typing import FrozenSet, Tuple

# Pose
POSE_CONNECTIONS: FrozenSet[Tuple[int, int]]

# Hands
HAND_CONNECTIONS: FrozenSet[Tuple[int, int]]

# Face Mesh
FACEMESH_TESSELATION: FrozenSet[Tuple[int, int]]
FACEMESH_CONTOURS: FrozenSet[Tuple[int, int]]
FACEMESH_IRISES: FrozenSet[Tuple[int, int]]
# ... (other 9 constants)
```

**理由**: Pyright の型チェックエラーを解消

---

## 📝 実装チェックリスト

### Phase 1: 基盤整備
- [ ] `connections.py` を作成
- [ ] 型スタブを追加 (オプション)
- [ ] 既存コードで独自定義の接続を検索・削除

### Phase 2: アノテーション強化
- [ ] `pose.py`: `show_connections` パラメータ追加
- [ ] `hand.py`: `show_connections` パラメータ追加
- [ ] `face.py`: `mesh_mode` パラメータ追加
- [ ] `holistic.py`: 統合アノテーション実装

### Phase 3: テスト・ドキュメント
- [ ] 各接続描画のユニットテスト
- [ ] ビジュアル回帰テスト (画像比較)
- [ ] README に可視化オプション追加
- [ ] Sphinx docstring に Examples 追加

---

## 🔬 技術的考慮事項

### メモリ効率
```python
# frozenset は immutable → 複数インスタンスで共有しても安全
# 2556接続 × 2要素 × 8バイト(int64) ≈ 40KB (FACEMESH_TESSELATION)
# → メモリ影響は無視できるレベル
```

### パフォーマンス
```python
# 接続線描画の計算量: O(N) where N = 接続数
# POSE: 35接続 → 高速
# HAND: 21接続 → 高速
# FACEMESH_CONTOURS: 124接続 → 許容範囲
# FACEMESH_TESSELATION: 2556接続 → リアルタイム処理では注意が必要

# 最適化案: Numba/Cython化、GPU加速 (将来)
```

### 後方互換性
```python
# デフォルト値で既存動作を維持
@annotate
def annotate(
    self,
    frame_src: MatLike,
    landmarks: NDArrayFloat,
    show_connections: bool = False,  # Default: OFF (backward compatible)
) -> MatLike:
    pass
```

---

## 🎓 参考資料

- [MediaPipe Pose Landmark Model](https://google.github.io/mediapipe/solutions/pose.html)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [MediaPipe Solutions Drawing Utils (GitHub)](https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py)

---

## 📌 アクションアイテム

1. **即座に実施可能**:
   - `connections.py` を作成して定数を re-export
   - 型スタブ追加でコンパイルエラー解消

2. **2-3日で実装**:
   - `pose.py` と `hand.py` にシンプルな接続描画を追加
   - ユニットテスト作成

3. **1週間で完成**:
   - `face.py` と `holistic.py` の高度な描画オプション実装
   - ドキュメント整備
   - ビジュアル回帰テスト

---

## ✅ 結論

MediaPipeの公式接続定数を統合することで:
- **保守コストを削減** (自前実装不要)
- **品質を向上** (公式定義の正確性)
- **拡張性を確保** (将来のMediaPipe更新に追従)

**推奨アクション**: Priority 1 (connections.py作成) を即座に実装し、Priority 2 (アノテーション強化) を段階的に展開。
