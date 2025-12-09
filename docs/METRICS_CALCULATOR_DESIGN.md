# Metrics Calculator 設計概要

## システム全体像

```mermaid
graph TB
    subgraph "ユーザーインターフェース"
        CLI[CLI: sldataset metrics calculate]
        CONFIG[設定ファイル<br/>YAML/JSON/TOML]
    end
    
    subgraph "MetricCalculator"
        CALC[MetricCalculator]
        RESOLVER[_resolve_categories]
        FILTER[_filter]
        EXECUTOR[並列実行エンジン<br/>Process/Thread Pool]
    end
    
    subgraph "データソース"
        DATASET[SLDataset<br/>40K+ samples]
        ITEM[SLDatasetItem]
        LANDMARKS[landmarks dict<br/>mediapipe.pose<br/>mediapipe.left_hand<br/>mediapipe.right_hand]
    end
    
    subgraph "メトリクス"
        METRIC1[NaNRateMetric]
        METRIC2[TemporalConsistencyMetric]
        METRIC3[AnatomicalConstraintMetric]
        METRICN[... 他のメトリクス]
    end
    
    subgraph "集約システム"
        AGG[Aggregator]
        STREAM[ストリーミング集約<br/>init/update/finalize]
        CHECKPOINT[チェックポイント保存]
    end
    
    subgraph "出力"
        JSON[JSON出力]
        RESULT[MetricResult]
    end
    
    CLI --> CONFIG
    CONFIG --> CALC
    CALC --> RESOLVER
    RESOLVER --> FILTER
    DATASET --> ITEM
    ITEM --> FILTER
    FILTER --> LANDMARKS
    LANDMARKS --> |NDArray| METRIC1
    LANDMARKS --> |NDArray| METRIC2
    LANDMARKS --> |NDArray| METRIC3
    LANDMARKS --> |NDArray| METRICN
    
    CALC --> EXECUTOR
    EXECUTOR --> METRIC1
    EXECUTOR --> METRIC2
    EXECUTOR --> METRIC3
    
    METRIC1 --> |MetricResult| AGG
    METRIC2 --> |MetricResult| AGG
    METRIC3 --> |MetricResult| AGG
    
    AGG --> STREAM
    AGG --> CHECKPOINT
    STREAM --> RESULT
    RESULT --> JSON
    
    style CALC fill:#e1f5ff
    style FILTER fill:#fff4e1
    style AGG fill:#e8f5e9
    style EXECUTOR fill:#fce4ec
```

## データフロー詳細

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Calculator
    participant Dataset
    participant Filter
    participant Metric
    participant Aggregator
    participant Output

    User->>CLI: sldataset metrics calculate --config ...
    CLI->>Calculator: 初期化(CalculationSpec[])
    
    Note over Calculator: カテゴリー名前解決<br/>_resolve_categories()
    
    Calculator->>Dataset: 並列サンプル取得
    
    loop 各サンプル (並列処理)
        Dataset->>Filter: SLDatasetItem
        Filter->>Filter: カテゴリーマッチング<br/>(正規表現: .*\.category)
        Filter->>Filter: データ連結<br/>(np.concatenate)
        Filter->>Metric: NDArray
        Metric->>Metric: calculate(data)
        Metric->>Aggregator: MetricResult
        
        alt ストリーミング集約
            Aggregator->>Aggregator: update_aggregator(state, result)
        else バッチ集約
            Aggregator->>Aggregator: results.append(result)
        end
        
        alt チェックポイントタイミング
            Aggregator->>Aggregator: 中間状態を保存
        end
    end
    
    Aggregator->>Aggregator: finalize_aggregator(state)
    Aggregator->>Output: 集約結果
    Output->>User: JSON出力
```

## カテゴリーシステム

```mermaid
graph LR
    subgraph "カテゴリー定義"
        STR[単一カテゴリー<br/>'pose']
        LIST[無名組み合わせ<br/>['left_hand', 'right_hand']]
        DICT[名前付き組み合わせ<br/>{hands: ['left_hand', 'right_hand']}]
    end
    
    subgraph "名前解決"
        GLOBAL[グローバル定義<br/>category_definitions]
        LOCAL[ローカル定義<br/>calculation内]
        RESOLVE[_resolve_categories<br/>前方参照・循環検出]
    end
    
    subgraph "ランドマークマッチング"
        REGEX[正規表現<br/>'.*\.{category}']
        PRIORITY[プレフィックス優先度<br/>mediapipe > openpose]
        KEY[ランドマークキー<br/>'mediapipe.pose']
    end
    
    subgraph "データ抽出"
        ITEM[SLDatasetItem.landmarks]
        LOAD[zarr Array [()]ロード]
        CONCAT[複数カテゴリー結合<br/>np.concatenate]
        NDARRAY[NDArray<br/>(frames, keypoints, coords)]
    end
    
    STR --> RESOLVE
    LIST --> RESOLVE
    DICT --> RESOLVE
    GLOBAL --> RESOLVE
    LOCAL --> RESOLVE
    
    RESOLVE --> REGEX
    REGEX --> PRIORITY
    PRIORITY --> KEY
    
    KEY --> ITEM
    ITEM --> LOAD
    LOAD --> CONCAT
    CONCAT --> NDARRAY
    
    style RESOLVE fill:#fff4e1
    style CONCAT fill:#e8f5e9
```

## Metricインターフェース階層

```mermaid
classDiagram
    class Metric {
        <<abstract>>
        +calculate(data: NDArray) MetricResult
    }
    
    class StreamingMetric {
        <<abstract>>
        +init_aggregator() T
        +update_aggregator(state: T, result: MetricResult) T
        +finalize_aggregator(state: T) MetricResult
    }
    
    class BatchMetric {
        <<abstract>>
        +aggregate(all_data: list[NDArray]) MetricResult
    }
    
    class NaNRateMetric {
        +calculate(data: NDArray) MetricResult
    }
    
    class TemporalConsistencyMetric {
        +calculate(data: NDArray) MetricResult
        +init_aggregator() AggregatorState
        +update_aggregator(state, result) AggregatorState
        +finalize_aggregator(state) MetricResult
    }
    
    class AnatomicalConstraintMetric {
        -bone_pairs: list[tuple]
        +calculate(data: NDArray) MetricResult
    }
    
    class MetricResult {
        +metric_name: str
        +values: dict[str, float]
        +metadata: dict[str, Any]
        +timestamp: datetime
        +sample_id: str | int
    }
    
    Metric <|-- StreamingMetric
    Metric <|-- BatchMetric
    Metric <|-- NaNRateMetric
    StreamingMetric <|-- TemporalConsistencyMetric
    Metric <|-- AnatomicalConstraintMetric
    
    Metric ..> MetricResult : creates
    
    note for StreamingMetric "メモリ効率的な集約\n40K+ サンプルに対応"
    note for BatchMetric "全データ必要な計算\n（将来対応）"
```

## CalculationSpec構造

```mermaid
graph TB
    subgraph "設定ファイル (YAML)"
        YAML["calculations:<br/>  - method: NaNRateMetric<br/>    categories: [pose]<br/>  - method: TemporalConsistencyMetric<br/>    categories:<br/>      - name: all<br/>        landmarks: [pose, left_hand, right_hand]<br/>    window_size: 5"]
    end
    
    subgraph "Pydantic Model"
        SPEC[CalculationSpec]
        METHOD[method: str]
        CATS[categories: list~Category~]
        EXTRA[model_config: extra='allow'<br/>任意フィールド]
    end
    
    subgraph "カテゴリー型"
        CAT_STR["str: 'pose'"]
        CAT_LIST["list: ['left_hand', 'right_hand']"]
        CAT_DICT["dict: {'all': ['pose', 'left_hand']}"]
    end
    
    subgraph "Entry Points / Import"
        EP[entry_points<br/>'cslrtools2.metrics']
        IMPORT[importlib]
        CLASS[Metric Class]
    end
    
    YAML --> SPEC
    SPEC --> METHOD
    SPEC --> CATS
    SPEC --> EXTRA
    
    CATS --> CAT_STR
    CATS --> CAT_LIST
    CATS --> CAT_DICT
    
    METHOD --> EP
    EP --> IMPORT
    IMPORT --> CLASS
    
    style SPEC fill:#e1f5ff
    style CATS fill:#fff4e1
```

## 並列処理アーキテクチャ

```mermaid
graph TB
    subgraph "メインプロセス"
        MAIN[MetricCalculator]
        CONFIG[設定読み込み]
        RESOLVE[カテゴリー解決]
        SCHEDULE[タスクスケジューリング]
        COLLECT[結果収集]
        AGG[集約処理]
    end
    
    subgraph "Worker Pool"
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker 3]
        WN[Worker N]
    end
    
    subgraph "共有データ"
        DATASET[SLDataset<br/>読み取り専用]
        QUEUE[タスクキュー<br/>idx: 0, 1, 2, ...]
        RESULTS[結果キュー<br/>順序保証]
    end
    
    CONFIG --> MAIN
    MAIN --> RESOLVE
    RESOLVE --> SCHEDULE
    
    SCHEDULE --> QUEUE
    QUEUE --> W1
    QUEUE --> W2
    QUEUE --> W3
    QUEUE --> WN
    
    DATASET -.-> W1
    DATASET -.-> W2
    DATASET -.-> W3
    DATASET -.-> WN
    
    W1 --> |MetricResult| RESULTS
    W2 --> |MetricResult| RESULTS
    W3 --> |MetricResult| RESULTS
    WN --> |MetricResult| RESULTS
    
    RESULTS --> COLLECT
    COLLECT --> AGG
    
    style MAIN fill:#e1f5ff
    style QUEUE fill:#fce4ec
    style RESULTS fill:#e8f5e9
```

## エラーハンドリングフロー

```mermaid
flowchart TD
    START[サンプル処理開始]
    LOAD[データロード]
    FILTER[_filter実行]
    CALC[Metric.calculate]
    AGG[Aggregator.update]
    CHECKPOINT{チェックポイント?}
    SAVE[中間状態保存]
    NEXT[次のサンプル]
    
    ERROR_DATA[データ不備<br/>KeyError]
    ERROR_PRECOND[前提条件違反<br/>ValueError]
    ERROR_CALC[数値計算エラー<br/>ZeroDivisionError]
    ERROR_SYS[システムエラー<br/>OSError]
    
    LOG[警告ログ出力]
    SKIP[スキップして続行]
    
    START --> LOAD
    LOAD --> FILTER
    FILTER --> CALC
    CALC --> AGG
    AGG --> CHECKPOINT
    
    CHECKPOINT -->|Yes| SAVE
    CHECKPOINT -->|No| NEXT
    SAVE --> NEXT
    
    LOAD -.->|例外| ERROR_DATA
    FILTER -.->|例外| ERROR_DATA
    CALC -.->|例外| ERROR_PRECOND
    CALC -.->|例外| ERROR_CALC
    LOAD -.->|例外| ERROR_SYS
    
    ERROR_DATA --> LOG
    ERROR_PRECOND --> LOG
    ERROR_CALC --> LOG
    ERROR_SYS --> LOG
    
    LOG --> SKIP
    SKIP --> NEXT
    
    style ERROR_DATA fill:#ffebee
    style ERROR_PRECOND fill:#fff3e0
    style ERROR_CALC fill:#fff9c4
    style ERROR_SYS fill:#f3e5f5
    style LOG fill:#e1f5ff
```

## 集約システム

```mermaid
stateDiagram-v2
    [*] --> 初期化: init_aggregator()
    
    初期化 --> 集約中: 最初のresult
    
    集約中 --> 集約中: update_aggregator(state, result)
    集約中 --> チェックポイント: 定期保存タイミング
    チェックポイント --> 集約中: 保存完了
    
    集約中 --> エラー発生: 例外
    エラー発生 --> チェックポイント復元: --resume
    チェックポイント復元 --> 集約中: 続行
    
    集約中 --> 最終化: 全サンプル処理完了
    最終化 --> [*]: finalize_aggregator(state)
    
    note right of 初期化
        state = {
            'count': 0,
            'sum': 0,
            'sum_sq': 0,
            ...
        }
    end note
    
    note right of 集約中
        メモリ効率的に
        逐次更新
        40K+ サンプル対応
    end note
    
    note right of チェックポイント
        100サンプルごと
        or
        5分ごと
    end note
```

## 出力フォーマット構造

```mermaid
graph TB
    subgraph "JSON出力構造"
        ROOT[results.json]
        META[metadata]
        RESULTS[results]
        
        META --> TS[timestamp]
        META --> DS[dataset]
        META --> TOTAL[total_samples]
        META --> PROC[processed_samples]
        
        RESULTS --> CAT1[pose]
        RESULTS --> CAT2[left_hand]
        RESULTS --> CAT3[right_hand]
        RESULTS --> CAT4[hands]
        RESULTS --> CAT5[all]
        
        CAT1 --> M1_1[NaNRateMetric]
        CAT1 --> M1_2[TemporalConsistencyMetric]
        CAT1 --> M1_3[AnatomicalConstraintMetric]
        
        M1_1 --> STATS1[mean: 0.05<br/>std: 0.02<br/>min: 0.0<br/>max: 0.15<br/>samples: 40000]
        
        CAT2 --> M2_1[NaNRateMetric]
        CAT2 --> M2_2[TemporalConsistencyMetric]
    end
    
    style ROOT fill:#e1f5ff
    style META fill:#fff4e1
    style RESULTS fill:#e8f5e9
```

## CLI実行フロー

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant Validator
    participant Calculator
    participant Progress
    participant Output

    User->>CLI: sldataset metrics calculate<br/>--dataset path.zarr<br/>--config config.yaml<br/>--output results.json<br/>--workers 4
    
    CLI->>Validator: 設定ファイル検証
    
    alt 設定エラー
        Validator-->>User: エラーメッセージ<br/>終了コード: 1
    else 設定OK
        Validator->>Calculator: CalculationSpec[]
    end
    
    Calculator->>Calculator: カテゴリー解決
    Calculator->>Calculator: Metric初期化
    
    Calculator->>Progress: 進捗表示開始<br/>(rich.progress)
    
    loop 各サンプル (並列)
        Calculator->>Calculator: データ抽出・計算
        Calculator->>Progress: 進捗更新<br/>Processed 100/40214
    end
    
    Calculator->>Calculator: 集約完了
    Calculator->>Output: JSON書き込み
    
    Output-->>User: 結果ファイル<br/>終了コード: 0
    
    alt --verbose
        Calculator-->>User: 詳細ログ出力
    end
```

## 実装ロードマップ

```mermaid
gantt
    title Metrics Calculator 実装計画
    dateFormat  YYYY-MM-DD
    section Phase 1: MVP
    MetricResult定義           :done, p1-1, 2025-12-02, 1d
    Metric基底クラス           :active, p1-2, 2025-12-02, 2d
    MetricCalculator骨格        :p1-3, after p1-2, 3d
    _filter()実装              :p1-4, after p1-2, 2d
    _resolve_categories()      :p1-5, after p1-4, 2d
    デフォルト集約             :p1-6, after p1-5, 1d
    JSON出力                   :p1-7, after p1-6, 1d
    基本CLI                    :p1-8, after p1-7, 2d
    
    section Phase 2: 並列化
    ProcessPoolExecutor統合    :p2-1, after p1-8, 3d
    プログレス表示             :p2-2, after p2-1, 1d
    エラーハンドリング         :p2-3, after p2-2, 2d
    順序保証                   :p2-4, after p2-3, 1d
    
    section Phase 3: 高度な機能
    ストリーミング集約         :p3-1, after p2-4, 3d
    チェックポイント機構       :p3-2, after p3-1, 3d
    --resume対応              :p3-3, after p3-2, 2d
    Entry Pointsプラグイン    :p3-4, after p3-3, 2d
    
    section Phase 4: 最適化
    データ抽出最適化           :p4-1, after p3-4, 3d
    複数出力形式               :p4-2, after p4-1, 2d
    サブコマンド               :p4-3, after p4-2, 2d
```

---

## 設計原則まとめ

### 1. 責任分離
- **Calculator**: オーケストレーション、データ抽出
- **Metric**: 計算ロジックのみ
- **Aggregator**: 集約処理

### 2. 柔軟性
- YAML/JSON/TOML + Python両対応
- Entry Pointsプラグインシステム
- ストリーミング/バッチ両対応

### 3. 効率性
- サンプル並列処理（40K+対応）
- ストリーミング集約（メモリ効率）
- チェックポイント機構（エラー回復）

### 4. 拡張性
- プラグインシステム
- 複数出力形式
- カスタムメトリクス追加容易

---

**生成日**: 2025-12-02  
**ステータス**: 設計確定、実装準備中  
**関連ドキュメント**: `metrics_calc_memo.md` (詳細Q&A記録)
