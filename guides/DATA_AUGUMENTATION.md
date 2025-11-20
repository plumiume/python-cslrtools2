# Data Augmentation for Continuous Sign Language Recognition (CSLR)

> **関連ドキュメント**: このガイドを使用する際は、以下のドキュメントも併せて参照してください。
> - [コーディングスタイルガイド](CODING_STYLE_GUIDE.md) - Transform実装規約
> - [Docstringスタイルガイド](DOCSTRING_STYLE_GUIDE.md) - 変換処理のドキュメント化
> - [統合テスト戦略](INTEGRATION_TEST_STRATEGY.md) - Data Augmentationのテスト方法
> - [例外処理・ログスタイルガイド](EXCEPTION_LOGGING_STYLE_GUIDE.md) - 変換エラーのハンドリング

## 連続手話認識（CSLR）で採用する変換手法

以下は上記調査結果から、CSLR タスクの特性を考慮して実際に実装する手法を選定したものです。

| カテゴリ        | 採用手法                                  | 実装場所                     | 優先度  | 備考                                                       |
| ----------- | ------------------------------------- | ------------------------ | ---- | -------------------------------------------------------- |
| 時間変換        | Uniform Speed Change                  | `transform/dynamic.py`   | High | 時間軸の一様スケール変換（補間ベース）                                      |
|             | Random Temporal Crop                  | `transform/dynamic.py`   | Mid  | ランダムな時間窓切り出し（窓サイズ制約必須：大きすぎると言語構造破壊）                      |
| 空間変換（同期必須）  | Random Resize + Padding/Crop          | `transform/dynamic.py`   | High | 中心基準のスケール変換（affine使用）                                    |
| 空間変換（RGB単独） | Color Jitter                          | `transform/dynamic.py`   | Mid  | 明度・彩度・コントラスト変化                                           |
|             | Random Grayscale                      | `transform/dynamic.py`   | Low  | グレースケール化                                                 |
| ランドマーク固有    | Joint Coordinate Noise                | `transform/dynamic.py`   | Mid  | 座標への小さなガウスノイズ付加                                          |
|             | Drop Joint / Mask Joint               | `transform/dynamic.py`   | Low  | 関節の欠損シミュレーション（冗長性評価用）                                    |
| 前処理（frozen） | Missing Value Imputation              | `transform/frozen.py`    | High | 欠損値の線形/最近傍補間                                             |
|             | Landmark Normalization                | `transform/frozen.py`    | High | 座標正規化（中心化・スケール調整）                                        |
| グラフ構造       | Symmetric Adjacency Normalization     | `transform/dynamic.py`   | Mid  | GCN用の隣接行列正規化 (D^{-1/2} A D^{-1/2})；PyG sparse不要時のみ有効 |
| 型変換         | ToTensor / Type Conversion            | `dataset/loader.py`      | High | NumPy/PIL → Tensor変換（Transform外で実施）                     |
| 複合          | Compose (Sequential Transform Chain)  | `transform/core.py`      | High | 複数変換の連鎖実行（`__call__`でパイプライン構築）                           |
| 将来実装候補      | Bone-Length Constrained Perturbation  | TBD                      | Low  | 制約条件下での骨格長変動；実装難易度が高いため将来目標                              |

---

## 連続手話認識（CSLR）で不採用とした変換手法

以下は一般的な動画認識では有用だが、CSLRの文脈では不適切と判断した手法です。

| 不採用手法                          | 不採用理由                                                                        | カテゴリ     |
| ------------------------------ | ---------------------------------------------------------------------------- | -------- |
| **Temporal Jitter**            | フレーム順序の微小な入れ替えは手話の時間的文法を破壊する可能性が高い（音素の順序変化に相当）                             | 時間変換     |
| **Frame Drop (Random)**        | ランダムなフレーム削除は手話動作の連続性を損なう；Speed Changeで代替可能                                  | 時間変換     |
| **Time Warp (Non-linear)**     | 非線形な時間歪みは手話の自然な動作速度分布を崩す；Uniform Speed Changeで十分                              | 時間変換     |
| **Random Start Offset**        | データセット設計で既に対応済み（データローダーでのランダムクロップ）；二重適用のリスク                                 | 時間変換     |
| **RandomResizedCrop**          | アスペクト比変更を伴うため手話者の体形が不自然に変形；Resize + Padding/Cropで代替                          | 空間変換     |
| **RandomRotation**             | 手話は重力方向（上下）が意味を持つため回転は不自然；カメラ角度のバリエーションは別手法で対応                             | 空間変換     |
| **RandomAffine (Shear)**       | せん断変形は手の形状認識に悪影響；scaleのみの使用で十分                                               | 空間変換     |
| **Random Horizontal Flip**     | **手話は左右の手の使い分けが意味を持つため左右反転は言語的意味を破壊**；一般画像認識とは異なり利用不可                       | 空間変換     |
| **GaussianBlur**               | 手の形状や指の細部がぼやけると認識精度が低下；手話では高周波成分が重要                                         | 画質・ノイズ   |
| **Random Erasing**             | ランドマークベースの認識では画像の局所欠損よりも関節欠損（Drop Joint）の方が適切                                | 見た目破壊    |
| **Edge Drop/Perturb (PyG)**    | PyG sparse graphでは不要；素朴な実装では有効だがdynamic.pyの計算コスト制約に抵触                       | グラフ構造    |
| **Multi-scale Graph**          | 計算コスト増大に対して精度向上が限定的；単一スケールで十分                                               | グラフ構造    |
| **CenterCrop**                 | 手話者が中央にいる保証がない；Random Cropの方が汎化性能が高い                                        | 空間変換     |
| **ToTensor (in Transform)**    | Transformクラスの型制約（Tensor only）により別ファイルで実装する方が適切；dataset/loader.pyで対応          | 型変換      |

### 不採用の主要な判断基準

1. **時間的文法の保持**: 手話は時間順序が意味を持つため、順序破壊的な変換は避ける
2. **空間的意味の保持**: **左右の手の使い分けが言語的意味を持つため左右反転は利用不可**（一般画像認識との重要な違い）
3. **物理的制約の尊重**: 人体の自然な動作範囲・骨格比率を維持
4. **細部情報の保存**: 手の形状・指の配置など高周波成分が重要
5. **計算効率**: `dynamic.py` では軽量な変換のみ、重い処理は `frozen.py` で事前実行
6. **アーキテクチャ適合性**: PyG sparse graph使用時はEdge Drop等は不要
7. **型制約の遵守**: Transform クラスは Tensor 型専用；型変換は別レイヤーで実施
8. **データセット設計との整合性**: 既存の前処理パイプラインとの重複回避

### 将来実装候補

以下は技術的には有用だが実装難易度が高いため、将来の目標として記録：

| 手法                                     | 実装難易度 | 必要な制約条件                                   | 期待効果                   |
| -------------------------------------- | ----- | ----------------------------------------- | ---------------------- |
| **Bone-Length Constrained Perturb**    | Very High | 人体骨格の物理制約（関節角度・骨長比率）を満たす最適化ソルバーが必要       | ランドマークのロバスト性向上         |
| **Temporal Crop (Large Window)**       | Mid   | 言語構造を破壊しない最大窓サイズの決定；文節境界検出アルゴリズムとの統合    | データ多様性向上（現在は小窓のみ実装可能） |
| **Symmetric Adjacency Norm (Dense)**   | Low   | PyG sparseを使わない素朴なGCN実装での利用を想定          | dense GCN での学習安定化      |

---

## 参考：動画認識・手話認識で検討される変換手法の網羅的調査

以下は torchvision および一般的な動画認識タスクで使用される変換手法の調査結果です。CSLR での採用判断の根拠として参照してください。

| カテゴリ            |                                             具体例 | torchvision 内の実装（モジュール）                                                                                       | 同期の必要性                             |
| --------------- | ----------------------------------------------: | ------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 時間変換            |                                   Temporal Crop | 自作（torchvision: 公式の temporal augment は限定的／非同期）                                                                | 必須（RGB⇄Landmark）                   |
|                 |                    Temporal Jitter / Frame Drop | 自作                                                                                                            | 必須                                 |
|                 |                        Speed Change / Time Warp | 自作                                                                                                            | 必須                                 |
|                 |                             Random Start Offset | 自作                                                                                                            | 必須                                 |
| 空間変換（画像側）       |                               RandomResizedCrop | `torchvision.transforms.RandomResizedCrop` / `transforms.v2`                                                  | 必須（対応する Landmark 変換を同時適用）          |
|                 |                RandomCrop / CenterCrop / Resize | `torchvision.transforms.RandomCrop`, `CenterCrop`, `Resize` / `transforms.v2`                                 | 必須（同上）                             |
|                 |                            RandomHorizontalFlip | `torchvision.transforms.RandomHorizontalFlip` / `transforms.v2` ; video: `transforms._functional_video.hflip` | 必須（LR 関節入れ替え等）                     |
|                 |                   RandomRotation / RandomAffine | `torchvision.transforms.RandomRotation`, `RandomAffine` / `transforms.v2`                                     | 必須（ランドマークに同一アフィン適用）                |
| 動画向け低レベル        |                    Resize / Crop / HFlip（video） | `torchvision.transforms._functional_video`（内部 API） / `torchvision.io` 補助                                      | 必須（時空間同期）                          |
| 色変換（RGB）        |                                     ColorJitter | `torchvision.transforms.ColorJitter` / `transforms.v2.ColorJitter`                                            | 不要（Landmark には無関係）                 |
|                 |                                 RandomGrayscale | `torchvision.transforms.RandomGrayscale`                                                                      | 不要                                 |
| 画質・ノイズ          |                                    GaussianBlur | `torchvision.transforms.GaussianBlur` / `transforms.v2`                                                       | 不要                                 |
|                 |                                         Blur（他） | 一部 in `transforms` / 自作                                                                                       | 不要                                 |
| 見た目破壊           |                                   RandomErasing | `torchvision.transforms.RandomErasing`（Tensor 対応）                                                             | 不要（Landmark には無関係）                 |
| Tensor / 型変換    | ToTensor / PILToTensor / ToPILImage / Normalize | `torchvision.transforms.ToTensor`, `PILToTensor`, `ToPILImage`, `Normalize` / `transforms.v2`                 | -                                  |
| ランドマーク固有        |                             Joint Jitter（座標ノイズ） | 自作                                                                                                            | —（Landmark 単独だが RGB と同期不要な場合あり）    |
|                 |                             Bone-Length Perturb | 自作                                                                                                            | —                                  |
|                 |                         Drop-Joint / Mask-Joint | 自作                                                                                                            | —（欠損シミュレーション）                      |
|                 |                    Normalize / Centering（座標正規化） | 自作（単純算術）                                                                                                      | —                                  |
| 隣接行列（Adjacency） |   Symmetric Normalization (D^{-1/2} A D^{-1/2}) | 自作（GCN 前処理）                                                                                                   | 不要（ただし Landmark の変更に伴い再計算が必要な場合あり） |
|                 |                        Edge Drop / Edge Perturb | 自作                                                                                                            | 不要（ただし構造変化の意味合いを確認）                |
|                 |                 Multi-scale Graph / Multi-hop A | 自作                                                                                                            | 不要                                 |
| 複合（同期必須）        |                  RandomCrop + Landmark アフィン（同時） | 自作（JointTransform：`Callable[[Dict], Dict]`）                                                                   | 必須                                 |
|                 |                   Horizontal Flip + LR-swap（同時） | 自作（JointTransform）                                                                                            | 必須                                 |
|                 |               Temporal + Spatial Joint（同一 seed） | 自作（JointTransform）                                                                                            | 必須                                 |
