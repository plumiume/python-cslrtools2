# Metrics Calculator 設計メモ

## 目次
- [概要](#概要)
- [アーキテクチャ](#アーキテクチャ)
- [設計決定事項](#設計決定事項)
- [実装仕様](#実装仕様)
- [設計検討記録](#設計検討記録)

---

## 概要

### 目的
SLDataset上で複数のメトリクス（評価指標）を効率的に計算するためのフレームワーク。

### 主要な設計方針
1. **責任分離**: Metricはデータ処理のみ、Calculatorがデータ抽出とオーケストレーション
2. **柔軟性**: YAML/TOML設定 + Pythonコード両対応
3. **効率性**: サンプル並列処理、ストリーミング集約
4. **拡張性**: プラグインシステム（Entry Points）

### パフォーマンス考慮
- **I/Oボトルネック**: zarr/NTFS制約により、並列読み込みが重要
- **並列戦略**: idx+getitem SLDataset、ユーザー指定の並列数（lmpipe Executor相当）
- **実績**: 40,214サンプル × 5カテゴリー処理可能

---

## アーキテクチャ

### コンポーネント構成

```
MetricCalculator
├─ _resolve_categories()     # カテゴリー名前解決
├─ _filter()                  # データ抽出 (SLDatasetItem → NDArray)
└─ calculate()                # メトリクス実行・集約

Metric (interface)
├─ calculate(data: NDArray) → MetricResult
└─ [Optional] init/update/finalize_aggregator()

Aggregator
├─ init() → state
├─ update(state, result) → state
└─ finalize(state) → MetricResult
```

### データフロー

```
SLDataset → [並列] → SLDatasetItem 
                     ↓
                     _filter(categories, item)
                     ↓
                     NDArray (連結されたランドマークデータ)
                     ↓
                     Metric.calculate(data)
                     ↓
                     MetricResult
                     ↓
                     Aggregator (ストリーミング集約)
                     ↓
                     最終集約結果 (JSON出力)
```

### カテゴリーシステム

```python
Category = (
    str |                      # "pose" (単一カテゴリー)
    list[str] |                # ["left_hand", "right_hand"] (無名組み合わせ)
    dict[str, list[str]]       # {"all": ["pose", "left_hand", "right_hand"]} (名前付き)
)

# 名前解決: Calculator初期化時に一括解決（前方参照対応）
# 複数マッチ: 正規表現 ".*\.{category}" でマッチ、プレフィックス優先度で選択
```

### CalculationSpec スキーマ

```python
class CalculationSpec(BaseModel):
    method: str                       # メトリクス名 or FQN
    categories: list[Category]        # 対象カテゴリー
    model_config = ConfigDict(extra='allow')  # メトリクス固有パラメータ
    # 任意フィールド（型チェックなし、厳格な命名規則で管理）
```

---

## 設計決定事項

### フェーズ1: 基本設計 (Q1-Q10)

| 項目 | 決定内容 |
|------|----------|
| **初期化** | 両方サポート (YAML/TOML + Pythonコード) |
| **インスタンス管理** | 1spec = 1instance (パラメータ差異時は独立) |
| **名前解決** | Calculator初期化時に一括解決 |
| **プラグインシステム** | Entry Points (静的リスト → 動的、セキュリティはFuture) |
| **パラメータ** | `Any` で許可、厳格な命名規則必要 |
| **並列化** | サンプル並列 (Process/Thread選択可) |
| **集約** | Metric.aggregate()優先、なければデフォルト統計 |
| **エラー処理** | 警告ログでスキップ（中間生成物保存対応） |
| **データ抽出** | Calculator._filter()が担当 |
| **設定形式** | JSON/YAML/TOML (Pydantic BaseModel) |

### フェーズ2: 詳細仕様 (Q11-Q20)

| 項目 | 決定内容 |
|------|----------|
| **セキュリティ** | Future (承認フローが簡単) |
| **_filter()責任** | 全担当、ヘルパーで分離 |
| **カテゴリー変換** | 正規表現 `".*\.{category}"` |
| **前処理** | ユーザー責任 (将来sldataset.transform?) |
| **Aggregatorインターフェース** | init/update/finalize優先 |
| **エラー分類** | 全て警告ログ、中間生成物保存 |
| **型安全性** | `Any` (厳格な命名規則で対応) |
| **順序保証** | 必須 (SLDataset idx順) |
| **名前付きスコープ** | 混在サポート (副作用なければ) |
| **インスタンスキャッシュ** | なし (ユーザーがパラメータで制御) |
| **出力形式** | まずJSON、将来複数形式 |
| **プログレス表示** | `rich.progress` (lmpipeで使用中) |

### フェーズ3: 実装詳細 (Q21-Q30)

| 項目 | 暫定決定 | 備考 |
|------|----------|------|
| **チェックポイント** | TBD | 案1:ファイル、案2:JSONL、案3:SQLite |
| **複数マッチ** | TBD | プレフィックス優先度が有力 |
| **MetricResultスキーマ** | TBD | Pydantic/TypedDict/dataclass検討中 |
| **Executor選択** | TBD | ProcessPool vs ThreadPool |
| **命名規則** | TBD | プレフィックス vs ネームスペース |
| **循環参照処理** | TBD | 検出してエラーが妥当 |
| **ストリーミングIF** | TBD | Generic[T]で型安全に |
| **JSON構造** | TBD | 階層 vs フラット |
| **データ抽出最適化** | TBD | キャッシング/プリフェッチ検討 |
| **CLI引数** | TBD | 基本構造は固まっている |

---

## 実装仕様

### 1. Calculator._filter() メソッド

**シグネチャ**:
```python
def _filter(self, categories: list[Category], item: SLDatasetItem) -> NDArray:
    """カテゴリーに基づいてランドマークデータを抽出・結合"""
```

**責任範囲**:
1. カテゴリー名解決 (`_resolve_categories()`)
2. ランドマークキー取得 (正規表現マッチング)
3. データロード (`item.landmarks[key][()]`)
4. 複数カテゴリーの結合 (`np.concatenate`)

**ヘルパー関数**:
- `_resolve_categories(cats: list[Category]) -> list[str]`
- `_parse_single_category(name: str) -> list[str]`

### 2. Metric インターフェース

**基本インターフェース**:
```python
class Metric(ABC):
    @abstractmethod
    def calculate(self, data: NDArray) -> MetricResult:
        """単一サンプルの計算"""
        pass
```

**ストリーミング集約（オプション）**:
```python
class StreamingMetric(Metric, Generic[T]):
    def init_aggregator(self) -> T:
        """集約状態を初期化"""
        pass
    
    def update_aggregator(self, state: T, result: MetricResult) -> T:
        """集約状態を更新"""
        pass
    
    def finalize_aggregator(self, state: T) -> MetricResult:
        """最終結果を計算"""
        pass
```

**フォールバックチェイン**:
```
1. init/update/finalize があれば使用 (ストリーミング)
2. なければ全結果を collect して aggregate()
3. aggregate() もなければデフォルト統計 (mean/std/min/max)
```

### 3. エラーハンドリング

| 例外タイプ | 対応 | 詳細 |
|-----------|------|------|
| データ不備 (KeyError) | スキップ+警告 | ランドマークキー不存在 |
| 前提条件違反 (ValueError) | スキップ+警告 | フレーム数不足等 |
| 数値計算エラー (ZeroDivisionError) | スキップ+警告 | データ依存、正規の可能性 |
| システムエラー (OSError) | スキップ+警告 | I/Oエラー |

**中間生成物の保存**:
- Aggregator状態を定期的に保存
- エラー時に再実行、既処理サンプルはスキップ
- キャッシュとして再利用

### 4. 並列処理

**方式**: サンプル並列（A）
- カテゴリー並列・メトリクス並列は不採用
- lmpipe Executor相当の仕組み

**Executor選択**:
- ProcessPoolExecutor (真の並列、GIL回避)
- ThreadPoolExecutor (軽量、共有メモリ)
- ユーザー指定 (`--workers N`, `--executor process|thread`)

**順序保証**: 必須
- SLDataset idx順で処理
- 集約統計のみでも順序保持

### 5. 出力フォーマット

**初期実装**: JSON
```json
{
  "metadata": {
    "timestamp": "2025-12-02T10:30:00",
    "dataset": "fs50-lmpipe-v5.2.1.zarr",
    "total_samples": 40214
  },
  "results": {
    "pose": {
      "NaNRateMetric": {"mean": 0.05, "std": 0.02, ...},
      "TemporalConsistencyMetric": {...}
    },
    "left_hand": {...}
  }
}
```

**将来対応**: CSV, JSONL, 複数ファイル出力

### 6. CLI インターフェース

**基本構造**:
```bash
sldataset calculate-metrics \
  --dataset <path> \
  --config <path> \
  --output <path> \
  [--workers N] \
  [--resume]
```

**主要オプション**:
- `--dataset`: SLDatasetパス
- `--config`: メトリクス設定ファイル (YAML/JSON/TOML)
- `--output`: 結果出力先
- `--workers`: 並列ワーカー数
- `--resume`: チェックポイントから再開
- `--verbose/-v`: 詳細ログ
- `--samples N`: サンプル数制限（テスト用）

---

## 設計検討記録

### 検討プロセス
設計は段階的に詰めており、以下の3フェーズで質問・回答を記録。

**フェーズ1 (Q1-Q10)**: 基本設計
- 初期化、インスタンス管理、並列化、設定形式等の方針決定

**フェーズ2 (Q11-Q20)**: 詳細仕様
- セキュリティ、データ抽出、エラーハンドリング、出力形式等

**フェーズ3 (Q21-Q30)**: 実装詳細
- チェックポイント、最適化、CLI設計等（一部未決定）

---

## Q&A 詳細記録

以下、各質問と回答の詳細を記録。

---

## Q1: MetricCalculatorの初期化時の振る舞い
**質問**: `MetricCalculator` は初期化時に、`CalculationSpec` のリストを受け取って、各スペックに対して `Metric` インスタンスを生成しますか？それとも、既に生成された `Metric` インスタンスのリストを受け取りますか？

**選択肢**:
- A: CalculationSpecから自動生成（`method`文字列 → インスタンス化）
- B: 事前に生成されたMetricインスタンスを登録
- C: 両方サポート（柔軟なAPI）

C: yaml / toml での簡単な計算指示を使うほうが楽 + 凝った計算（依存関係がある）に対応

---

## Q2: MetricTargetとメトリクスのマッチング戦略
**質問**: 1つの `Metric` インスタンスは、複数の `MetricTarget`（カテゴリー組み合わせ）に対して再利用されますか？それとも、各 `MetricTarget` ごとに専用のインスタンスを作成しますか？

**例**: `NaNRateMetric` を `pose`, `left_hand`, `all` の3つのターゲットで使用する場合
- A: 1つのインスタンスを3回呼び出す（状態共有）
- B: 3つの独立したインスタンスを作成（完全分離）

**考慮点**: メトリクス内部に状態を持つ場合の影響

作成パラメータが異なる場合にインスタンスを作成できるようにする
（コード上は独立インスタンスも定義できるように自然になりそう）

---

## Q3: カテゴリーの名前解決タイミング
**質問**: 名前付きカテゴリー組み合わせ（`{"all": ["pose", "left_hand", "right_hand"]}`）の前方参照解決は、いつ行われますか？

**例**:
```python
categories = [
    {"hands": ["left_hand", "right_hand"]},  # 定義
    "hands",  # 参照
    {"body_hands": ["pose", "hands"]}  # handsを参照
]
```

**選択肢**:
- A: CalculationSpec作成時に即座に解決（エラー早期検出）
- B: MetricCalculator初期化時に一括解決
- C: 計算実行時に遅延解決（柔軟性重視）

B: 内部状態が変化するようなCalculatorは気持ち悪い

---

## Q4: white_list_methodの実装方法
**質問**: `white_list_method` による安全なメソッド呼び出しは、どのように実装しますか？

**選択肢**:
- A: 許可リストを静的に定義（`ALLOWED_METRICS = ["NaNRateMetric", "TemporalConsistencyMetric", ...]`）
- B: Entry Pointsから動的に取得（プラグインシステム活用）
- C: 設定ファイル（YAML/JSON）で管理
- D: デコレータベース（`@metric_registry.register("nan_rate")`）

**考慮点**: fqn（fully qualified name）使用時のセキュリティ設定

A -> B: 許容度が異なる
（プラグインシステムの柔軟性を考慮するとcslrtools2が考慮できないが、
安全性に疑問　これについてさらなる情報を表示し、必要があれば詳細を質問）

---

## Q5: extrasとコンストラクタ引数の関係
**質問**: `extras: AnyModel` と `<extras>: Any` の使い分けは？

**理解確認**:
```python
class CalculationSpec:
    method: str
    categories: list[Category]
    extras: AnyModel  # ← これは何を想定？
    # <extras>はコンストラクタ引数として展開される？
```

**例**:
```python
spec = CalculationSpec(
    method="AnatomicalConstraintMetric",
    categories=["pose"],
    extras={"bone_pairs": [...], "threshold": 0.5},  # ← Metricのcalculate()に渡す？
    # または
    bone_pairs=[...],  # ← コンストラクタに直接渡す？
    threshold=0.5
)
```

**質問**: `extras` はメトリクス計算時のパラメータですか、それともインスタンス化時のコンストラクタ引数ですか？

直接設定 `*extras` したいが、型安全性が下がるため（別にAnyModelでも同じ）
`args` のに統合したほうが pythonic か

---

## Q6: 並列計算のスコープ
**質問**: 並列化は以下のどのレベルで行われますか？

**選択肢**:
- A: サンプル並列（複数の `SLDatasetItem` を並列処理）
- B: カテゴリー並列（同一サンプルの複数カテゴリーを並列処理）
- C: メトリクス並列（同一カテゴリーの複数メトリクスを並列処理）
- D: 階層的並列（A+B、またはA+C の組み合わせ）

**考慮点**: zarrのI/Oボトルネック vs CPU並列の効果

A

---

## Q7: 計算結果の集約方法
**質問**: 複数サンプルに対するメトリクス計算結果は、どのように集約されますか？

**選択肢**:
- A: MetricCalculatorが自動集約（mean, std, min, max を計算）
- B: Metricインスタンスが集約ロジックを持つ（`aggregate(results: list) -> dict`）
- C: ユーザーコード（CLI）が集約を担当
- D: 両方サポート（Metricにaggregate()があればそれを使用、なければデフォルト統計）

(C), D 
Cを実現するにはCalculateSpecに計算コードを入れる必要があるが、
toml / yaml に記述できないため難しいと予想

---

## Q8: エラーハンドリング戦略
**質問**: 特定のサンプル/カテゴリー/メトリクスの計算が失敗した場合、どう処理しますか？

**シナリオ**: サンプル1000個のうち、サンプル#42で `TemporalConsistencyMetric` が例外を投げた

**選択肢**:
- A: 即座に全体を中断（fail-fast）
- B: 警告を出して続行、失敗したサンプルをスキップ
- C: 失敗したメトリクスのみスキップ、他のメトリクスは継続
- D: リトライ機構を実装（最大N回）

**考慮点**: ログ出力、失敗サンプルのトラッキング

想定される例外情報の不足のため、現時点で回答できない
詳細な状況を再度質問

---

## Q9: SLDatasetItemの責任範囲
**質問**: `Metric` クラスが `SLDatasetItem` を直接受け取る場合、カテゴリーデータの抽出は誰が行いますか？

**例**:
```python
# Metric内で抽出？
class NaNRateMetric:
    def calculate(self, item: SLDatasetItem, category: str) -> dict:
        data = item.landmarks[f"mediapipe.{category}"]
        # ...

# または Calculator内で抽出？
class MetricCalculator:
    def calculate(self, item: SLDatasetItem, spec: CalculationSpec):
        data = self._extract_category_data(item, spec.categories)
        result = spec.metric.calculate(data)  # ← numpy配列を渡す
```

**選択肢**:
- A: Metric内でSLDatasetItemから直接抽出（Metric側の柔軟性高）
- B: Calculator内で抽出してnumpy配列として渡す（Metric側はシンプル）
- C: 両方サポート（`calculate(data: ndarray | SLDatasetItem)`）

フィルターメソッドが担当　（Metricからカテゴリーターゲットの分離）

Calculator
    ->
    Calculator._filter (cats, item) -> NDArray
        ->
        Metric
        <-
    <-


---

## Q10: 設定ファイルフォーマット
**質問**: CLIで使用する設定ファイル（メトリクス計算仕様）のフォーマットは？

**例**:
```yaml
# metrics_config.yaml
calculations:
  - method: completeness.NaNRateMetric
    categories: [pose, left_hand, right_hand]
  
  - method: temporal.TemporalConsistencyMetric
    categories:
      - name: all
        landmarks: [pose, left_hand, right_hand]
    extras:
      window_size: 5

execution:
  parallel_samples: 4
  parallel_categories: false
```

**質問**: このようなYAML/JSON形式で良いですか？それとも別のフォーマット（TOML, Python DSL等）が適切ですか？

json / yaml / toml を想定　（なのでpydantic BaseModelをスキーマに使用）

---

# 回答サマリー

## 決定事項
1. **Q1**: 両方サポート (YAML/TOML簡易指定 + 複雑な依存関係用コード)
2. **Q2**: パラメータ差異時に独立インスタンス作成
3. **Q3**: Calculator初期化時に一括解決
4. **Q4**: A→B (静的リスト → Entry Points) ※安全性要検討
5. **Q5**: `args` に統合の方向 (Pythonic)
6. **Q6**: サンプル並列 (A)
7. **Q7**: (C) または D (Metricのaggregate()優先、なければデフォルト)
8. **Q8**: 保留 (例外情報不足)
9. **Q9**: Calculator._filter()が担当 (責任分離)
10. **Q10**: JSON/YAML/TOML (Pydantic BaseModel)

## 次フェーズの質問

---

## Q11: Entry Pointsベースのプラグインシステムのセキュリティ
**背景**: Q4で Entry Points による動的メトリクス読み込みを選択したが、安全性に懸念がある。

**質問**: プラグインシステムの安全性をどう担保しますか？

**セキュリティリスク**:
```python
# 悪意のあるプラグイン例
[project.entry-points."cslrtools2.metrics"]
malicious = "malware:EvilMetric"  # ← __init__で任意コード実行
```

**対策案**:
- A: サンドボックス実行（制限付きPython環境）
- B: コード署名検証（信頼できる開発者のみ）
- C: 明示的な承認フロー（初回実行時にユーザー確認）
- D: Entry Points + 静的許可リスト併用（デフォルトは組み込みのみ、`--allow-plugins`で拡張）
- E: その他の方法

**考慮点**: 
- 研究用途 vs プロダクション環境
- CLI使用者の技術レベル

`Future` としてマーク　（簡単なのは C か）

---

## Q12: Calculator._filter()の実装詳細
**背景**: Q9でCalculator._filter()がカテゴリーデータ抽出を担当すると決定。

**質問**: `_filter()` メソッドの具体的な責任範囲は？

**実装イメージ**:
```python
class MetricCalculator:
    def _filter(self, categories: list[Category], item: SLDatasetItem) -> NDArray:
        # 1. カテゴリー名解決 (e.g. "hands" → ["left_hand", "right_hand"])
        # 2. ランドマークキー取得 (e.g. "pose" → "mediapipe.pose")
        # 3. データロード (item.landmarks["mediapipe.pose"])
        # 4. 複数カテゴリーの結合 (np.concatenate)
        # 5. その他？
        pass
```

**質問A**: 上記1-4の全てを担当しますか、それとも一部は別の場所に分離しますか？

担当するのは `(cats, item) -> NDArray`
おそらく 3, 4 を含む

`_resolve_categories(cats)`
`_parse_signle_category(name)`

が必要そう



**質問B**: カテゴリー名 → ランドマークキーの変換規則は？
- 固定マッピング (`{"pose": "mediapipe.pose", ...}`)
- プラグイン可能（独自ランドマーク形式対応）
- 設定ファイルで定義

正規表現　`".*\.{?category}"`


**質問C**: データ前処理（正規化、NaN補完等）は `_filter()` の責任ですか？

現時点ではユーザーが責任を持つ　（誰も担当しない）
（前処理としてsldataset.transform使う可能性は無きにしも非ず）

---

## Q13: Metricのaggregate()インターフェース設計
**背景**: Q7でMetricインスタンスが集約ロジックを持つ可能性を示唆。

**質問**: `aggregate()` メソッドのシグネチャは？

**案1**: シンプルな集約
```python
class Metric(ABC):
    @abstractmethod
    def calculate(self, data: NDArray) -> MetricResult:
        pass
    
    def aggregate(self, results: list[MetricResult]) -> MetricResult:
        """デフォルト実装: mean, std, min, max"""
        pass
```

**案2**: ストリーミング集約（メモリ効率）
```python
class Metric(ABC):
    def init_aggregator(self) -> Aggregator:
        """集約状態を初期化"""
        pass
    
    def update_aggregator(self, agg: Aggregator, result: MetricResult) -> Aggregator:
        """1サンプルずつ集約状態を更新"""
        pass
    
    def finalize_aggregator(self, agg: Aggregator) -> MetricResult:
        """最終結果を計算"""
        pass
```

**質問**: どちらのインターフェースが適切ですか？または別の設計がありますか？

メトリックごとに使用できるインターフェースを用意
(Dataset に対して　IterableDatasetがあるように)
フォールバックチェインは
    (no-prefix) -> init, update, reduce -> Error
prefer: Aggregatorに任せるため、おそらく更新インターフェースが有利
AggregatorもMetricResultを返せるとなおよい

**考慮点**: 
- 40,000サンプルのメモリ使用量
- 並列計算時の集約タイミング

---

## Q14: エラーハンドリングの例外分類
**背景**: Q8で例外情報不足により保留。

**質問**: 以下の例外カテゴリーごとに、どう対処すべきですか？

**A. データ不備**:
```python
# サンプル#42のlandmarksにposeキーが存在しない
KeyError: 'mediapipe.pose'
```
→ 対応: 1

**B. 計算前提条件違反**:
```python
# TemporalConsistencyMetricがフレーム数不足を検出
ValueError: "Requires at least 3 frames, got 2"
```
→ 対応: 1

**C. 数値計算エラー**:
```python
# 骨の長さが0でゼロ除算
ZeroDivisionError: division by zero
```
→ 対応: 1

**D. システムエラー**:
```python
# zarr I/Oエラー
OSError: [Errno 5] Input/output error
```
→ 対応: 1

**選択肢** (各カテゴリーごとに):
- 1: スキップして続行（警告ログ）
- 2: スキップして続行（サイレント）
- 3: エラーレポートに記録して続行
- 4: 即座に中断
- 5: リトライ (N回まで)

A, B (形式エラー)は計算へのオプションで変化しそう
(lmpipeのスキップオプションを参照)
C (実行時エラー：データセットによってはそれが正規の可能性),
D(形式エラー)

Aggregatorの中間生成物を保存可能にするか、
全体実行を強制　（エラーログを出しておけば自然と再実行しそう、
そのアイテムだけの問題の場合、中間生成物をキャッシュとして利用）

---

## Q15: CalculationSpecの`args`統合後の型安全性
**背景**: Q5で `extras` を `args` に統合する方向性。

**質問**: Pydantic BaseModelで動的な追加フィールドをどう扱いますか？

**課題**:
```python
# 型安全に記述したいが...
class CalculationSpec(BaseModel):
    method: str
    categories: list[Category]
    # ← ここにMetric固有のパラメータをどう定義？
```

**案1**: Extra allow
```python
class CalculationSpec(BaseModel):
    model_config = ConfigDict(extra='allow')
    method: str
    categories: list[Category]
    # 任意の追加フィールドを許可（型チェックなし）
```

**案2**: Union型ですべてのMetricパラメータを列挙
```python
class CalculationSpec(BaseModel):
    method: str
    categories: list[Category]
    bone_pairs: list[tuple[int, int]] | None = None  # Anatomical用
    window_size: int | None = None  # Temporal用
    # ← 全メトリクスのパラメータを定義（肥大化）
```

**案3**: ネストしたモデル
```python
class AnatomicalParams(BaseModel):
    bone_pairs: list[tuple[int, int]]

class CalculationSpec(BaseModel):
    method: str
    categories: list[Category]
    params: AnatomicalParams | TemporalParams | ...  # Union型
```

**質問**: どの案が適切ですか？または別のアプローチがありますか？

~~こうなると複数のメトリックで同じキーを共有するときに不便なので、*extrasに戻す~~
この場合、より厳格な命名規則が必要
Any で

---

## Q16: 並列サンプル処理時の順序保証
**背景**: Q6でサンプル並列を選択。

**質問**: 並列処理後の結果は、元のサンプル順序を保持する必要がありますか？

**シナリオ**:
```python
samples = [0, 1, 2, ..., 40213]
# 並列処理後
results = [result_0, result_2, result_1, ...]  # ← 順序不定
```

**考慮点**:
- 集約統計のみなら順序不要
- サンプル別レポート（CSV出力等）なら順序必要
- 順序保証のコスト（同期待ち）

**質問**: どちらが要件ですか？
- A: 順序保持必須
- B: 順序不要（集約のみ）
- C: オプション化（デフォルトは順序保持）

A SLDatasetに対して idx 順のほうが処理しやすいため、

---

## Q17: カテゴリー名前付き定義のスコープ
**背景**: Q3でCalculator初期化時に名前解決を決定。

**質問**: 名前付きカテゴリーの定義スコープは？

**例1**: グローバル定義
```yaml
# metrics_config.yaml
category_definitions:
  hands: [left_hand, right_hand]
  all: [pose, hands]

calculations:
  - method: NaNRateMetric
    categories: [hands]  # ← 参照
```

**例2**: ローカル定義
```yaml
calculations:
  - method: NaNRateMetric
    categories:
      - hands: [left_hand, right_hand]  # ← 定義
      - hands  # ← 同一calculation内で参照
```

**例3**: 混在
```yaml
category_definitions:  # グローバル
  hands: [left_hand, right_hand]

calculations:
  - method: NaNRateMetric
    categories:
      - body_hands: [pose, hands]  # ← グローバル参照 + ローカル定義
```

**質問**: どの方式を採用しますか？

3 副作用がなければ

---

## Q18: Metricインスタンスのキャッシュ戦略
**背景**: Q2でパラメータ差異時に独立インスタンス作成を決定。

**質問**: 同一パラメータのMetricインスタンスは再利用しますか？

**シナリオ**:
```yaml
calculations:
  - method: NaNRateMetric
    categories: [pose]
  - method: NaNRateMetric
    categories: [left_hand]
  # ← NaNRateMetricはパラメータなしなので同一インスタンス再利用？
```

**選択肢**:
- A: 常に独立インスタンス作成（シンプル、状態分離）
- B: パラメータハッシュでキャッシュ（メモリ効率、複雑性増）
- C: ユーザー指定（`reuse: true` オプション）

**考慮点**: Metricインスタンスが内部状態を持つ場合の影響

ユーザーがパラメーターで作成可能
```python
spec_a1 = {"method": NaNRateMetric, "categories": ["cat1"], "type": "landmarks" param=p1}
spec_a2 = {"method": NaNRateMetric, "categories": ["cat1"], "type": "landmarks", param=p2}
spec_b1 = {"method": NaNRateMetric, "categories": ["cat1", "cat2"], "type": "landmarks"}
spec_c1 = {"method": NaNRateMetric, "categories": ["named_cats"], "type", "landmarks"}
```

のため考慮なし　むしろ1spec 1instance

---

## Q19: 計算結果の出力フォーマット
**質問**: CLI実行後の結果は、どのような形式で出力されますか？

**要件**:
- カテゴリー別
- メトリクス別
- サンプル別（オプション？）

**案1**: 階層的JSON
```json
{
  "pose": {
    "NaNRateMetric": {"mean": 0.05, "std": 0.02, ...},
    "TemporalConsistencyMetric": {...}
  },
  "left_hand": {...}
}
```

**案2**: フラットCSV
```csv
category,metric,stat,value
pose,NaNRateMetric,mean,0.05
pose,NaNRateMetric,std,0.02
...
```

**案3**: 複数ファイル
```
results/
  summary.json          # 集約統計
  pose_nan_rate.csv     # サンプル別詳細
  pose_temporal.csv
  ...
```

**質問**: どの形式が適切ですか？または複数形式をサポートしますか？

まずはJSON
将来的に複数形式

---

## Q20: 並列実行時のプログレス表示
**背景**: Q6でサンプル並列、40,000サンプルの処理には時間がかかる。

**質問**: 並列実行中のプログレス表示はどうしますか？

**課題**:
- 複数プロセス/スレッドからの出力競合
- プログレスバーの更新頻度

**選択肢**:
- A: tqdm等のプログレスバー（メインプロセスで集約）
- B: シンプルなログ出力（`Processed 100/40214 samples...`）
- C: リッチUI（カテゴリー別・メトリクス別の進捗）
- D: サイレント実行（ログファイルのみ）

**考慮点**: CLI使用シーン（対話的 vs バッチ処理）

このプロジェクト(lmpipe)では `rich.progress` を使用

---

# 回答サマリー (Q11-Q20)

## 決定事項
11. **Q11**: セキュリティは Future (簡単なのは C: 承認フロー)
12. **Q12**: 
    - A: `_filter()`は全て担当、ヘルパー関数で分離
    - B: 正規表現 `".*\.{category}"`
    - C: 前処理はユーザー責任 (将来的にsldataset.transform?)
13. **Q13**: 更新インターフェース優先 (init/update/reduce), Aggregatorも使用可
14. **Q14**: 全て 1 (警告ログでスキップ), Aggregator中間生成物を保存可能に
15. **Q15**: Any で (厳格な命名規則が必要)
16. **Q16**: A (順序保持必須) - SLDataset idx 順で処理しやすい
17. **Q17**: 例3 (混在) - 副作用なければ
18. **Q18**: 1spec = 1instance (ユーザーがパラメータで制御可能)
19. **Q19**: まずJSON, 将来的に複数形式
20. **Q20**: `rich.progress` (lmpipeで使用中)

## 次フェーズの質問

---

## Q21: Aggregatorの中間生成物の保存仕様
**背景**: Q14でエラー時にAggregator中間生成物を保存可能にする方針。

**質問**: 中間生成物の保存・再開メカニズムは？

**ユースケース**:
```python
# 1回目の実行: サンプル0-1000処理後にエラー
# 2回目の実行: 中間結果を読み込んでサンプル1001から再開
```

**実装案**:

**案1**: チェックポイントファイル
```python
# 定期的に保存
checkpoint = {
    "aggregator_state": aggregator.to_dict(),
    "processed_samples": [0, 1, 2, ..., 1000],
    "timestamp": "2025-12-02T10:30:00"
}
# .metrics_checkpoint.json に保存
```

**案2**: 増分結果ファイル
```python
# 各サンプル処理後に追記
results/
  incremental_results.jsonl  # JSON Lines形式
  # 再実行時に既存行を読み込んで集約状態を復元
```

**案3**: データベース（SQLite等）
```python
# サンプルごとの結果をDB保存
# 再実行時はクエリで未処理サンプルを特定
```

**質問A**: どの方式を採用しますか？
    itemでの中間生成物がそこまで大きくないと仮定し、
    ```JSON
        {
            str_idx: {artifacts}
            ...
        }
    ```

**質問B**: チェックポイント頻度は？
- 固定間隔 (100サンプルごと)
- 時間ベース (5分ごと)
- メモリ使用量ベース
- ユーザー指定

逐次 計算結果をバックアップ (Aggregatorの担当？)

**質問C**: 再実行時の動作は？
- 自動検出して続行
- `--resume` フラグ必須
- 毎回確認プロンプト

`--resume`

---

## Q22: カテゴリー名の正規表現マッチング詳細
**背景**: Q12Bでカテゴリー → ランドマークキー変換に正規表現 `".*\.{category}"` を使用。

**質問**: 複数マッチ時の挙動は？

**シナリオ**:
```python
# SLDatasetItemに以下が存在
item.landmarks = {
    "mediapipe.pose": ...,
    "openpose.pose": ...,    # ← 複数の"pose"
    "custom.pose": ...
}
```

**パターンA**: 最初のマッチのみ
```python
# "pose" → "mediapipe.pose" (アルファベット順最初)
```

**パターンB**: 全てマッチして結合
```python
# "pose" → ["mediapipe.pose", "openpose.pose", "custom.pose"]
# 全て連結
```

**パターンC**: プレフィックス優先度
```python
# 設定で優先度定義
preferred_engines = ["mediapipe", "openpose", "custom"]
# "pose" → "mediapipe.pose" (優先度最高)
```

**パターンD**: エラー
```python
# 複数マッチは曖昧性エラー
# ユーザーに明示的な指定を要求 ("mediapipe.pose")
```

**質問**: どのパターンが適切ですか？


fallback: C -> A

---

## Q23: MetricResultの標準スキーマ
**質問**: `MetricResult` の標準的な構造は？

**背景**: 現在のprototype v2では辞書形式だが、型安全性とバリデーションが必要。

**現状**:
```python
result = {
    "metric_name": "nan_rate",
    "values": {"nan_rate": 0.05},
    "metadata": {"total_frames": 104, "shape": (104, 33, 4)}
}
```

**提案案**:

**案1**: Pydantic BaseModel
```python
class MetricResult(BaseModel):
    metric_name: str
    values: dict[str, float]
    metadata: dict[str, Any]
    timestamp: datetime | None = None
    sample_id: str | int | None = None
```

**案2**: TypedDict (軽量)
```python
class MetricResult(TypedDict):
    metric_name: str
    values: dict[str, float]
    metadata: dict[str, Any]
```

**案3**: dataclass
```python
@dataclass
class MetricResult:
    metric_name: str
    values: dict[str, float]
    metadata: dict[str, Any]
    
    def to_dict(self) -> dict: ...
```

**質問A**: どのスキーマが適切ですか？
計算結果の検証が必要ないため（よりむしろMetricの責任）、TypedDict

**質問B**: `values` の型制約は？
- `dict[str, float]` (数値のみ)
- `dict[str, float | int]`
- `dict[str, Any]` (柔軟性重視)

Any (nameによってのみ検証されると困るので、計算スペックそのものを持つ)

**質問C**: 必須フィールド vs オプショナルフィールドは？
- `metric_name`: 必須 <- 識別子であり一意
- `values`: 必須 <- 空でも可能
- `metadata`: オプション？ <- ここまで来たらこれも必須（空でも問題ない）
- 追加フィールド（タイムスタンプ、サンプルID等）？ <- それはmetadata

---

## Q24: 並列処理のExecutor選択
**背景**: Q6でサンプル並列を選択、lmpipeのExecutorと同等の方式。

**質問**: 具体的な並列処理の実装方式は？

**選択肢**:

**A. ProcessPoolExecutor**
```python
from concurrent.futures import ProcessPoolExecutor
# メリット: 真の並列、GIL回避
# デメリット: プロセス起動コスト、メモリ使用量大
```

**B. ThreadPoolExecutor**
```python
from concurrent.futures import ThreadPoolExecutor
# メリット: 軽量、共有メモリ
# デメリット: GIL制約、CPU並列性低い
```

**C. multiprocessing.Pool**
```python
from multiprocessing import Pool
# メリット: プロセス再利用、メモリ効率
# デメリット: Windows互換性に注意
```

**D. Ray (分散実行フレームワーク)**
```python
import ray
# メリット: スケーラビリティ、タスクスケジューリング
# デメリット: 依存関係増加、複雑性
```

**質問A**: どの方式を採用しますか？

**質問B**: ワーカー数の決定方法は？
- CPU数と同じ (`os.cpu_count()`)
- CPU数 - 1 (システムリソース確保)
- ユーザー指定 (`--workers N`)
- 自動調整 (I/O待機時間を監視)

**質問C**: zarr I/Oのスレッドセーフ性は保証されていますか？
- 複数プロセスから同時読み込みは安全？
- ロック機構が必要？

lmpipeの現在の状況を参照　（多分ユーザーのオプション選択）

---

## Q25: CalculationSpecの名前付け規則
**背景**: Q15で `Any` を採用、厳格な命名規則が必要。

**質問**: Metric固有パラメータの命名規則は？

**課題**: 複数メトリクスが同名パラメータを持つ場合の衝突回避

**案1**: プレフィックス付与
```yaml
calculations:
  - method: AnatomicalConstraintMetric
    categories: [pose]
    anatomical_bone_pairs: [[0, 1], [1, 2]]
    anatomical_threshold: 0.5
  
  - method: TemporalConsistencyMetric
    categories: [pose]
    temporal_window_size: 5
```

**案2**: ネームスペース
```yaml
calculations:
  - method: AnatomicalConstraintMetric
    categories: [pose]
    params:
      bone_pairs: [[0, 1], [1, 2]]
      threshold: 0.5
```

**案3**: メトリクス名由来の短縮形
```yaml
calculations:
  - method: AnatomicalConstraintMetric
    categories: [pose]
    ac_bone_pairs: [[0, 1], [1, 2]]  # ac = AnatomicalConstraint
    ac_threshold: 0.5
```

**質問**: どの命名規則が適切ですか？または他の案がありますか？

同名を使うのは、defaultで指定された extra を共通で使用したいため
意図しない衝突を防ぐ
pypi のパッケージ名衝突防止を考えると、
プラグインパッケージをプレフィックスとして
その中の衝突はパッケージの責任

よしなに


---

## Q26: _resolve_categories()の実装戦略
**背景**: Q12で `_resolve_categories(cats)` ヘルパー関数が必要と判明。

**質問**: カテゴリー解決の実装詳細は？

**入力例**:
```python
categories = [
    "pose",
    {"hands": ["left_hand", "right_hand"]},
    "hands",  # 前方参照
    {"body_hands": ["pose", "hands"]}  # 再帰参照
]
```

**実装課題**:

**A. 循環参照の検出**
```python
# 無限ループを防ぐ
{"a": ["b"], "b": ["a"]}  # ← エラーにすべき？
```

**B. 未定義参照**
```python
["undefined_category"]  # ← エラー？警告？無視？
```

**C. 再帰解決の深さ制限**
```python
# 過度にネストした定義
{"a": ["b"], "b": ["c"], "c": ["d"], ...}
# 何階層まで許可？
```

**質問A**: 循環参照時の挙動は？
- 検出してエラー
- 警告して無視
- 検出しない（無限ループリスク）
ユーザーハンドルのエラーを送出　（責任外）

**質問B**: 未定義参照時の挙動は？
- エラー (厳格)
- 警告して空扱い
- そのまま文字列として使用

無視　（すべてのitemで存在しないカテゴリーとして処理される）

**質問C**: 解決アルゴリズムは？
- 深さ優先探索 (DFS)
- 幅優先探索 (BFS)
- トポロジカルソート

よしなに

---

## Q27: Metricのinit/update/reduceインターフェース詳細
**背景**: Q13で更新インターフェース優先を選択。

**質問**: 具体的なメソッドシグネチャは？

**提案**:
```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')  # Aggregator state type

class StreamingMetric(ABC, Generic[T]):
    """ストリーミング集約対応メトリクス"""
    
    @abstractmethod
    def calculate(self, data: NDArray) -> MetricResult:
        """単一サンプルの計算"""
        pass
    
    @abstractmethod
    def init_aggregator(self) -> T:
        """集約状態を初期化"""
        pass
    
    @abstractmethod
    def update_aggregator(self, state: T, result: MetricResult) -> T:
        """集約状態を更新"""
        pass
    
    @abstractmethod
    def finalize_aggregator(self, state: T) -> MetricResult:
        """最終集約結果を計算"""
        pass
```

**質問A**: このインターフェースで十分ですか？

**質問B**: 非ストリーミングメトリクス（全データ必要）はどう扱う？
```python
class BatchMetric(ABC):
    """全サンプル必要なメトリクス"""
    
    def calculate(self, data: NDArray) -> MetricResult:
        """単一サンプルでは計算不可"""
        raise NotImplementedError("Use aggregate() instead")
    
    @abstractmethod
    def aggregate(self, all_data: list[NDArray]) -> MetricResult:
        """全サンプルをまとめて処理"""
        pass
```

**質問C**: Calculator側のフォールバックロジックは？
```python
if hasattr(metric, 'init_aggregator'):
    # ストリーミング集約
    state = metric.init_aggregator()
    for result in results:
        state = metric.update_aggregator(state, result)
    return metric.finalize_aggregator(state)
elif hasattr(metric, 'aggregate'):
    # バッチ集約
    return metric.aggregate(results)
else:
    # デフォルト集約 (mean, std, min, max)
    return default_aggregate(results)
```

Aggregator クラスがそのIF, 処理を受け持つ
設定 / インスタンスは Metric -> Calculator -> Noop
が持つ


---

## Q28: 出力JSONの階層構造詳細
**背景**: Q19でまずJSON形式を採用。

**質問**: 具体的なJSON構造は？

**要素**:
- 複数カテゴリー
- 複数メトリクス
- 集約統計 (mean, std, min, max)
- メタデータ (実行時刻、パラメータ等)

**案1**: カテゴリー → メトリクス階層
```json
{
  "metadata": {
    "timestamp": "2025-12-02T10:30:00",
    "dataset": "fs50-lmpipe-v5.2.1.zarr",
    "total_samples": 40214,
    "processed_samples": 40000
  },
  "results": {
    "pose": {
      "NaNRateMetric": {
        "mean": 0.05, "std": 0.02, "min": 0.0, "max": 0.15,
        "samples": 40000
      },
      "TemporalConsistencyMetric": {...}
    },
    "left_hand": {...}
  }
}
```

**案2**: フラット構造（検索しやすい）
```json
{
  "metadata": {...},
  "results": [
    {
      "category": "pose",
      "metric": "NaNRateMetric",
      "statistics": {"mean": 0.05, "std": 0.02, ...}
    },
    {
      "category": "pose",
      "metric": "TemporalConsistencyMetric",
      "statistics": {...}
    }
  ]
}
```

**案3**: メトリクス → カテゴリー階層（逆）
```json
{
  "metadata": {...},
  "results": {
    "NaNRateMetric": {
      "pose": {"mean": 0.05, ...},
      "left_hand": {"mean": 0.12, ...}
    },
    "TemporalConsistencyMetric": {...}
  }
}
```

**質問**: どの構造が適切ですか？または別の案がありますか？

[spec, ...] に対応させ、案2を採用
{spec_ident: res4spec}
res4spec: 案2

---

## Q29: SLDatasetItemからのデータ抽出最適化
**質問**: `_filter()` でのデータ読み込みを最適化するには？

**背景**: 40,000サンプル × 複数カテゴリーの読み込みは重い。

**最適化案**:

**A. 遅延読み込み**
```python
# 必要なカテゴリーのみ読み込む
if "pose" in categories:
    pose_data = item.landmarks["mediapipe.pose"][()]
```

**B. キャッシング**
```python
# 同一サンプルの複数メトリクス計算時に再利用
@lru_cache(maxsize=100)
def load_category_data(item_id: int, category: str) -> NDArray:
    ...
```

**C. バッチ読み込み**
```python
# 複数サンプルをまとめて読み込み（zarr chunk最適化）
dataset[0:100]  # より効率的
```

**D. 事前読み込み（プリフェッチ）**
```python
# 次のサンプルを非同期で先読み
async def prefetch_next(idx):
    ...
```

**質問A**: どの最適化を実装しますか？（複数選択可）
A+B (sldatasetの実利用時にidxをもとにしたランダムアクセスが存在)

**質問B**: メモリ使用量とのトレードオフは？
- キャッシュサイズの上限
- メモリ使用量の監視
- OOM時のフォールバック

できるだけ 1回の訪問で処理を完了させるため、
OOMフォールバックで行う　（多分ユーザーが メモリのページングに気づく）

---

## Q30: CLI引数の全体設計
**質問**: `sldataset calculate-metrics` コマンドの引数仕様は？

**基本構造**:
```bash
sldataset calculate-metrics \
  --dataset <path> \
  --config <path> \
  --output <path> \
  [OPTIONS]
```

**オプション候補**:

**必須引数**:
- `--dataset`: SLDatasetへのパス
- `--config`: メトリクス設定ファイル (YAML/JSON/TOML)

**出力関連**:
- `--output`: 結果出力先
- `--format`: 出力形式 (json, csv, jsonl)
- `--save-checkpoint`: チェックポイント保存先

**並列化関連**:
- `--workers`: 並列ワーカー数
- `--executor`: 並列実行方式 (process, thread)

**実行制御**:
- `--samples`: サンプル数制限 (テスト用)
- `--resume`: チェックポイントから再開
- `--dry-run`: 実行計画のみ表示

**デバッグ関連**:
- `--verbose`, `-v`: 詳細ログ
- `--quiet`, `-q`: 静音モード
- `--log-file`: ログファイルパス

**質問A**: この引数セットで十分ですか？追加・削除すべき引数は？

**質問B**: 設定ファイル vs CLI引数の優先度は？
- CLI引数が常に優先
- 設定ファイルが優先、CLI引数はオーバーライド
- 明示的な `--override` フラグ必要

**質問C**: サブコマンド構造は必要ですか？
```bash
sldataset metrics calculate ...
sldataset metrics list  # 利用可能なメトリクス一覧
sldataset metrics validate-config  # 設定検証
```

```shell
sldataset metrics <subcommand> ...
```

まずは lmpipe のCLI設定を見てください

---

# 設計サマリー・実装ロードマップ

## 確定した設計

### コア設計原則
1. **責任分離アーキテクチャ**
   - `MetricCalculator`: オーケストレーション、データ抽出
   - `Metric`: 計算ロジックのみ
   - `Aggregator`: 集約処理（ストリーミング対応）

2. **柔軟な設定システム**
   - YAML/JSON/TOML + Pythonコード両対応
   - Pydantic BaseModel でバリデーション
   - Entry Points プラグインシステム

3. **効率的な処理**
   - サンプル並列処理 (Process/Thread選択)
   - ストリーミング集約（メモリ効率）
   - 中間生成物保存（エラー回復）

### カテゴリーシステム
```python
# 3種類のカテゴリー指定
"pose"                                    # 単一
["left_hand", "right_hand"]               # 無名組み合わせ
{"all": ["pose", "left_hand", "right_hand"]}  # 名前付き

# 正規表現マッチング: ".*\.{category}"
# プレフィックス優先度: mediapipe > openpose > custom
# 名前解決: Calculator初期化時（前方参照・混在スコープ対応）
```

### エラーハンドリング方針
- **基本**: 警告ログでスキップ、処理継続
- **中間保存**: Aggregator状態を定期保存
- **再実行**: チェックポイントから再開可能

### 出力形式
- **Phase 1**: JSON (階層構造)
- **Future**: CSV, JSONL, 複数ファイル

## 未決定事項（優先度順）

### 高優先度（実装前に決定必要）
1. **MetricResult スキーマ** (Q23)
   - Pydantic/TypedDict/dataclass の選択
   - フィールド定義（必須 vs オプション）

2. **Executor 選択** (Q24)
   - ProcessPoolExecutor vs ThreadPoolExecutor
   - ワーカー数決定ロジック
   - zarr スレッドセーフ性確認

3. **ストリーミングIF詳細** (Q27)
   - init/update/finalize シグネチャ
   - Generic[T] 型パラメータ設計
   - フォールバックチェイン実装

### 中優先度（初期実装で簡易版、後で改善）
4. **チェックポイント機構** (Q21)
   - 案1: JSONファイル（簡単）
   - 案2: JSONL（増分）
   - 案3: SQLite（高機能）

5. **カテゴリー複数マッチ** (Q22)
   - プレフィックス優先度実装
   - 設定ファイルでの優先度定義

6. **命名規則** (Q25)
   - プレフィックス vs ネームスペース
   - ドキュメント化

### 低優先度（Future）
7. **セキュリティ** (Q11)
   - 承認フロー実装
   - サンドボックス化

8. **データ抽出最適化** (Q29)
   - キャッシング、プリフェッチ
   - バッチ読み込み

9. **複数出力形式** (Q19, Q28)
   - CSV, JSONL サポート
   - カスタムフォーマッター

## 実装ステップ

### Phase 1: MVP（最小限動作するプロトタイプ）
- [ ] `MetricResult` スキーマ定義（Pydantic推奨）
- [ ] `Metric` 基底クラス実装
- [ ] `MetricCalculator` 基本構造
  - [ ] `_filter()` メソッド
  - [ ] `_resolve_categories()` ヘルパー
  - [ ] 単一サンプル計算
- [ ] デフォルト集約（mean/std/min/max）
- [ ] JSON出力
- [ ] 基本的なCLI (`--dataset`, `--config`, `--output`)

### Phase 2: 並列化・エラー処理
- [ ] ProcessPoolExecutor 統合
- [ ] プログレス表示（`rich.progress`）
- [ ] エラーハンドリング（スキップ+警告）
- [ ] 順序保証実装
- [ ] ワーカー数設定（`--workers`）

### Phase 3: 高度な機能
- [ ] ストリーミング集約インターフェース
- [ ] チェックポイント機構
- [ ] `--resume` 対応
- [ ] Entry Points プラグインシステム
- [ ] 設定ファイルバリデーション

### Phase 4: 最適化・拡張
- [ ] データ抽出最適化（キャッシング等）
- [ ] 複数出力形式
- [ ] サブコマンド（`list`, `validate-config`）
- [ ] 詳細なログ・デバッグ機能

## 参考実装

### 既存コード
- `evaluate_real_dataset_v2.py`: 基本的なメトリクス計算ループ実装済み
- `metrics_prototype2/`: Metric実装の参考
- `docs/CALCULATOR_ARCHITECTURE.md`: 初期設計ドキュメント

### 関連モジュール
- `lmpipe`: 並列処理、CLI設計の参考
- `sldataset`: データローダー、transform（将来の前処理）

---

## 次のアクション

1. **Phase 1 実装開始**
   - `MetricResult` Pydantic モデル作成
   - `Metric` 抽象基底クラス定義
   - `MetricCalculator` 骨格実装

2. **Q21-Q30 未回答質問の決定**
   - 特に Q23, Q24, Q27 を優先

3. **プロトタイプテスト**
   - `evaluate_real_dataset_v2.py` をリファクタリング
   - 新設計に基づいて書き直し

---

**最終更新**: 2025-12-02  
**ステータス**: 設計フェーズ完了、実装準備中  
**次のマイルストーン**: Phase 1 MVP実装