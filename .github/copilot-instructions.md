# Copilot Instructions for cslrtools2-sldataset-metrics

## Project Overview

このプロジェクトは、**cslrtools2**の`sldataset`モジュールに評価指標（metrics）機能を追加するための専用ブランチです。

## 重要な参照ドキュメント

### 必読: メトリクス設計ドキュメント

実装前に以下のドキュメントを必ず参照してください:

- **メインドキュメント**: `c:\Users\ikeda\Workspace\1github\cslrtools2-dataset-2\pose_estimation_metrics_analysis.md`
  - Ground Truthなしで使用できる評価指標
  - 異なるランドマーク推定エンジン間の比較に適した指標
  - 実装推奨度マトリックス
  - モジュール構成案

### 親プロジェクトのルール

- **親プロジェクト**: `c:\Users\ikeda\Workspace\1github\cslrtools2-dataset-2`
- **親プロジェクトの指示**: `.github/copilot-instructions.md`
  - プロジェクト全体のアーキテクチャ
  - 型システム、プラグインシステム
  - コーディング規約

## 実装方針

### 1. メトリクス実装の優先順位

`pose_estimation_metrics_analysis.md` の「6. cslrtools2への実装提案」に従って実装してください:

#### Phase 1: 基本指標（最優先）
- ✅ NaN/欠損率計算
- ✅ 基本統計量（平均、標準偏差）
- **推奨度**: ⭐⭐⭐⭐⭐（最高）
- **エンジン間比較**: ✅ 適している

#### Phase 2: 時間的一貫性
- フレーム間変化量
- 滑らかさ指標（加速度の標準偏差）
- 加速度分析
- **推奨度**: ⭐⭐⭐⭐⭐（最高）
- **エンジン間比較**: ✅ 適している

#### Phase 3: 骨格制約
- 骨の長さの一貫性
- 関節角度の妥当性
- 左右対称性
- **推奨度**: ⭐⭐⭐⭐
- **エンジン間比較**: ✅ 適している

#### Phase 4: 高度な指標（オプション）
- 多視点一貫性（多視点データがある場合）
- Ground Truthとの比較（利用可能な場合）

### 2. 実装すべきでない指標

以下の指標は**エンジン間比較に向いていない**ため、実装優先度を下げてください:

❌ **信頼度スコア (Confidence Scores)**
- 理由: 各エンジン(MediaPipe, OpenPose等)で計算方法が異なる
- 用途: 同一エンジン内での品質評価に限定

❌ **再構成誤差 (Reconstruction Error)**
- 理由: モデルアーキテクチャに依存
- 用途: Ground Truthがある場合のみ

## モジュール構成

`pose_estimation_metrics_analysis.md` の推奨構成に従ってください:

```
src/cslrtools2/sldataset/
    metrics/
        __init__.py          # モジュールエクスポート
        base.py              # 共通インターフェース・抽象クラス
        completeness.py      # NaN/欠損率計算（Phase 1）
        temporal.py          # 時間的一貫性（Phase 2）
        anatomical.py        # 骨格制約違反（Phase 3）
        geometric.py         # 幾何学的一貫性（Phase 4, オプション）
        utils.py             # ユーティリティ関数
```

### 命名規約

#### クラス名
- 各指標に対応するクラス: `{MetricName}Metric`
- 例: `CompletenessMetric`, `TemporalConsistencyMetric`, `AnatomicalConstraintMetric`

#### メソッド名
- `calculate(data, **kwargs)`: 指標を計算するメインメソッド
- `validate_inputs(data)`: 入力データの検証
- `get_description()`: 指標の説明を返す

### 共通インターフェース (base.py)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Metric(ABC):
    """メトリクス計算の基底クラス"""
    
    @abstractmethod
    def calculate(self, data: Any, **kwargs) -> Dict[str, float]:
        """指標を計算する
        
        Args:
            data: 入力データ（ランドマーク配列など）
            **kwargs: 追加パラメータ
            
        Returns:
            計算結果の辞書（指標名: 値）
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """指標の説明を返す"""
        pass
    
    def validate_inputs(self, data: Any) -> bool:
        """入力データの検証（オプション）"""
        return True
```

## 実装ガイドライン

### 1. エンジン非依存性の確保

**CRITICAL**: すべての実装は特定のランドマーク推定エンジン（MediaPipe, OpenPose等）に依存しないようにしてください。

```python
# ✅ Good: エンジン非依存
def calculate_nan_rate(landmarks: np.ndarray) -> float:
    """任意のランドマーク配列のNaN率を計算"""
    return np.isnan(landmarks).sum() / landmarks.size

# ❌ Bad: エンジン依存
def calculate_mediapipe_confidence(results: MediaPipeResults) -> float:
    """MediaPipe特有のデータ構造に依存"""
    return results.pose_landmarks.landmark[0].visibility
```

### 2. 入力データ形式

cslrtools2の型システムに従ってください:

```python
# ランドマークデータの想定形状
# (frames, keypoints, coordinates)
# 例: (300, 33, 3) = 300フレーム、33キーポイント、xyz座標
landmarks: np.ndarray  # shape: (T, K, D)
```

### 3. 時間的一貫性の実装例

`pose_estimation_metrics_analysis.md` の計算方法を参照:

```python
# フレーム間の変化量
velocity = landmarks[1:] - landmarks[:-1]  # shape: (T-1, K, D)

# 加速度
acceleration = velocity[1:] - velocity[:-1]  # shape: (T-2, K, D)

# 滑らかさの指標（加速度の標準偏差）
smoothness = np.std(acceleration)
```

### 4. 骨格制約の実装例

```python
# 骨の長さの変動係数
def calculate_bone_length_variation(landmarks: np.ndarray, 
                                     bone_pairs: List[Tuple[int, int]]) -> float:
    """骨の長さの一貫性を評価
    
    Args:
        landmarks: shape (T, K, D)
        bone_pairs: [(joint_a, joint_b), ...] 骨を構成する関節ペア
    """
    bone_lengths = []
    for i, j in bone_pairs:
        # 各フレームでの骨の長さ
        lengths = np.linalg.norm(landmarks[:, i] - landmarks[:, j], axis=1)
        bone_lengths.append(lengths)
    
    bone_lengths = np.array(bone_lengths)  # shape: (num_bones, T)
    
    # 変動係数 = std / mean
    variation_coef = np.std(bone_lengths, axis=1) / np.mean(bone_lengths, axis=1)
    return np.mean(variation_coef)
```

## テスト戦略

### テストデータの準備

親プロジェクトの `tests/resource/` ディレクトリを参照してください:
- `tests/resource/setup_resources.py`: テストリソースのダウンロード
- `tests/resource/README.md`: リソース管理ガイド

### テストケース

各メトリクスに対して以下をテスト:

1. **正常系**:
   - 典型的なランドマークデータでの計算
   - 結果が期待される範囲内にあるか

2. **異常系**:
   - NaNを含むデータ
   - 空配列
   - 形状が不正なデータ

3. **エッジケース**:
   - 単一フレーム（時間的一貫性計算不可）
   - 全てNaN
   - 全て同じ値（変動ゼロ）

## CLI統合

`sldataset` コマンドへの統合方法:

```bash
# 使用例
sldataset calculate-metrics \
    --dataset path/to/dataset.zarr \
    --metrics completeness temporal anatomical \
    --output metrics_report.json
```

実装場所: `src/cslrtools2/sldataset/app/cli.py`

## ドキュメント要件

各メトリクスには以下を含むドキュメントを記述してください:

1. **指標の説明**: 何を測定するか
2. **計算方法**: 数式または疑似コード
3. **解釈方法**: 値の範囲と意味
4. **制約事項**: 使用上の注意点
5. **参考文献**: 関連する論文（`pose_estimation_metrics_analysis.md` 参照）

例:

```python
class TemporalConsistencyMetric(Metric):
    """時間的一貫性を評価する指標
    
    フレーム間のランドマーク位置変化の滑らかさを測定します。
    
    計算方法:
        1. 速度 v[t] = x[t+1] - x[t]
        2. 加速度 a[t] = v[t+1] - v[t]
        3. 滑らかさ = std(a)
    
    解釈:
        - 低い値: 滑らかな動き（良好）
        - 高い値: ジッタが多い（要改善）
    
    参考文献:
        Liu & Yuan, "Recognizing Human Actions as the Evolution 
        of Pose Estimation Maps", CVPR 2018
    """
```

## 開発ワークフロー

### 1. ブランチ戦略

親プロジェクトの `guides/BRANCHING_STRATEGY.md` に従ってください:

- このワークツリーは `dev-ai/sldataset-metrics` ブランチに対応
- 実装完了後、`integration-ai` または `main-ai` にマージ
- Conventional Commits を使用: `feat:`, `test:`, `docs:` など

### 2. 実装の進め方

1. **Phase 1から順に実装**
   - まず `completeness.py` (NaN/欠損率)
   - 次に `temporal.py` (時間的一貫性)

2. **各Phaseごとにテストを追加**
   - `tests/unit/sldataset/metrics/` にテストファイル配置

3. **ドキュメント更新**
   - `docs/api/sldataset.md` にAPI追加
   - 使用例を `docs/examples/` に追加

### 3. 依存関係管理

- `pyproject.toml` に新しい依存関係を追加する場合は最小限に
- NumPy, PyTorchなど既存の依存関係を優先的に使用

## 重要な注意事項

### ❌ やってはいけないこと

1. **エンジン特有の機能を実装しない**
   - MediaPipe, OpenPoseなど特定エンジンに依存するコード

2. **信頼度スコアを主要指標として実装しない**
   - エンジン間比較には不適切（`pose_estimation_metrics_analysis.md` 参照）

3. **Ground Truth必須の指標を優先しない**
   - このプロジェクトはGround Truthなし評価が目的

### ✅ 推奨される実装

1. **エンジン非依存の指標を優先**
   - NaN率、時間的一貫性、骨格制約

2. **NumPy配列ベースの実装**
   - 汎用性と計算効率を重視

3. **段階的な実装**
   - Phase 1 → Phase 2 → Phase 3 の順

## 参考リソース

### 親プロジェクト

- メインREADME: `c:\Users\ikeda\Workspace\1github\cslrtools2-dataset-2\README.md`
- コーディングスタイル: `guides/CODING_STYLE_GUIDE.md`
- テスト戦略: `guides/INTEGRATION_TEST_STRATEGY.md`

### メトリクス設計

- **必読**: `pose_estimation_metrics_analysis.md`
  - セクション3: 異なるランドマーク推定エンジンの比較に推奨される指標
  - セクション4: 実装推奨度マトリックス
  - セクション6: cslrtools2への実装提案

### 学術文献

`pose_estimation_metrics_analysis.md` のセクション5「参考文献」を参照してください。

---

## クイックスタートチェックリスト

新しい機能を実装する際は、以下を確認してください:

- [ ] `pose_estimation_metrics_analysis.md` を読んだ
- [ ] 実装する指標が推奨リスト（✅）に含まれている
- [ ] エンジン非依存の実装になっている
- [ ] 入力データ形式が親プロジェクトと整合している
- [ ] テストケースを作成した
- [ ] ドキュメント（docstring）を記述した
- [ ] 型ヒントを追加した（PEP 695形式）
- [ ] Conventional Commitsでコミットした

---

## 更新履歴

- 2025-11-27: 初版作成
  - メトリクス実装の優先順位と方針を定義
  - `pose_estimation_metrics_analysis.md` への参照を追加
