# LMPipe フレームワーク分類評価レポート

**評価日**: 2025年11月14日  
**対象**: cslrtools2.lmpipe  
**評価者**: AI Architecture Analyst

> **関連ドキュメント**: このガイドを使用する際は、以下のドキュメントも併せて参照してください。
> - [コーディングスタイルガイド](CODING_STYLE_GUIDE.md) - プラグイン実装規約
> - [統合テスト戦略](INTEGRATION_TEST_STRATEGY.md) - フレームワークテスト方法
> - [ライブラリ適格性チェック](LIBRARY_CHECKLIST.md) - 品質評価基準

---

## 📋 評価対象フレームワーク分類

1. **Platform** (プラットフォーム)
2. **Extensible Framework** (拡張可能フレームワーク)
3. **Self-hosted Framework** (セルフホスティングフレームワーク)
4. **Executable Framework** (実行可能フレームワーク)
5. **Framework-as-an-App** (アプリケーション型フレームワーク)
6. **Runtime / Runtime Environment** (ランタイム/実行環境)

---

## 🔍 LMPipe アーキテクチャ分析

### 現在の構造

```
lmpipe/
├── estimator.py          # ABC (抽象基底クラス) - Strategy Pattern
├── collector/            # 出力ハンドラー - Collector Pattern
├── interface/            # ユーザーAPI - Facade Pattern
│   ├── __init__.py      # LMPipeInterface
│   └── runner.py        # LMPipeRunner (1,252行)
├── app/                  # CLI アプリケーション
│   ├── cli.py           # Entry point (lmpipe command)
│   └── plugins.py       # Plugin discovery
├── options.py            # 設定管理
├── runspec.py            # ジョブ仕様
└── utils.py              # ユーティリティ
```

### 主要な特徴

#### ✅ プラグインシステム
- **Entry Points**: `pyproject.toml` で定義
- **Runtime Discovery**: 実行時にプラグインを発見
- **Estimator 抽象化**: `Estimator[K]` ABC を継承

```python
[project.entry-points."cslrtools2.lmpipe.plugins"]
"mediapipe.holistic" = "cslrtools2.plugins.mediapipe.lmpipe.holistic_args:holistic_info"
"mediapipe.pose" = "cslrtools2.plugins.mediapipe.lmpipe.pose_args:pose_info"
```

#### ✅ 制御の逆転 (IoC)
- **依存性注入**: Estimator, Collector をコンストラクタで注入
- **テンプレートメソッド**: イベントハンドラー (on_start, on_complete, etc.)
- **フック**: カスタマイズ可能なライフサイクルイベント

```python
interface = LMPipeInterface(
    estimator=my_estimator,      # 注入
    collectors=[my_collector],    # 注入
    runner_type=MyCustomRunner    # カスタマイズ可能
)
```

#### ✅ CLI アプリケーション
- **Console Script**: `lmpipe` コマンド
- **Standalone Execution**: 単独で実行可能
- **Configuration**: コマンドライン引数で設定

```bash
lmpipe mediapipe.holistic input.mp4 -o output.npz --workers 4
```

#### ✅ 実行管理
- **Executor Pool**: マルチプロセス/スレッド管理
- **Progress Tracking**: Rich ライブラリで進行状況表示
- **Resource Management**: コンテキストマネージャーで自動クリーンアップ

#### ❌ セルフホスティング機能
- **No Web Server**: サーバー機能なし
- **No Network Layer**: ネットワーク通信なし
- **No Distributed Runtime**: 分散実行環境なし

#### ❌ 独立したランタイム
- **Python Interpreter依存**: Python 3.12+ が必要
- **No Custom VM**: 独自仮想マシンなし
- **No JIT Compilation**: JIT コンパイラーなし

---

## 📊 各分類への適合性評価

### 1. Platform (プラットフォーム)

**定義**: 他のアプリケーションやサービスを構築・実行するための基盤

**評価**: ⚠️ **部分的に該当 (40%)**

| 基準 | 判定 | 理由 |
|------|------|------|
| 拡張可能性 | ✅ Yes | プラグインシステムあり |
| 独立実行環境 | ❌ No | Python インタープリター依存 |
| エコシステム | ⚠️ Partial | 限定的 (MediaPipe プラグインのみ) |
| マルチアプリサポート | ❌ No | 単一用途 (ランドマーク抽出) |
| API提供 | ✅ Yes | Python API + CLI |
| リソース管理 | ⚠️ Partial | プロセス/スレッドのみ |

**結論**: 
プラットフォームとしては**小規模すぎる**。プラグインで拡張可能だが、汎用的な基盤というより特定ドメイン(ランドマーク抽出)に特化している。

**類似例**: 
- ❌ Not like: AWS, Kubernetes, Android (汎用プラットフォーム)
- ⚠️ Somewhat like: FFmpeg (特定ドメインのプラットフォーム)

---

### 2. Extensible Framework (拡張可能フレームワーク)

**定義**: ユーザーがコードを追加して機能を拡張できる構造化されたフレームワーク

**評価**: ✅ **強く該当 (90%)**

| 基準 | 判定 | 理由 |
|------|------|------|
| 抽象基底クラス | ✅ Yes | `Estimator[K]` ABC |
| プラグイン機構 | ✅ Yes | Entry points システム |
| 制御の逆転 (IoC) | ✅ Yes | 依存性注入、テンプレートメソッド |
| カスタマイズポイント | ✅ Yes | Estimator, Collector, Runner |
| イベントシステム | ✅ Yes | on_start, on_complete, etc. |
| 規約 > 設定 | ⚠️ Partial | 一部の規約あり |
| ドキュメント化された拡張点 | ✅ Yes | docstring で明記 |

**拡張ポイント**:
1. **Estimator**: カスタムランドマーク検出器
2. **Collector**: カスタム出力形式
3. **Runner**: カスタム実行ロジック
4. **Plugin**: Entry points 経由の新機能

**コード例**:
```python
from cslrtools2.lmpipe.estimator import Estimator

class MyEstimator(Estimator):
    def process(self, frame):
        # カスタムロジック
        return ProcessResult(landmarks={...})
    
    def configure_estimator_name(self):
        return "my_custom_estimator"
```

**結論**: 
**これが最も適切な分類**。典型的な拡張可能フレームワークの特徴をすべて満たしている。

**類似例**: 
- ✅ Django (Web フレームワーク)
- ✅ Pytest (テストフレームワーク)
- ✅ Sphinx (ドキュメントフレームワーク)

---

### 3. Self-hosted Framework (セルフホスティングフレームワーク)

**定義**: 自身をホストして他のアプリケーションを実行する環境を提供

**評価**: ❌ **該当しない (5%)**

| 基準 | 判定 | 理由 |
|------|------|------|
| Webサーバー | ❌ No | サーバー機能なし |
| アプリケーションホスティング | ❌ No | 他アプリをホストしない |
| ネットワーク層 | ❌ No | ネットワーク機能なし |
| マルチテナント | ❌ No | 単一ユーザー実行のみ |
| デプロイメント | ❌ No | デプロイメント機能なし |

**結論**: 
完全に該当しない。セルフホスティングの概念は LMPipe には適用不可。

**類似例**: 
- ❌ Not like: Heroku, Vercel, Docker (セルフホスティング環境)

---

### 4. Executable Framework (実行可能フレームワーク)

**定義**: フレームワーク自体が実行可能で、設定やスクリプトで動作をカスタマイズ

**評価**: ✅ **該当する (85%)**

| 基準 | 判定 | 理由 |
|------|------|------|
| CLI コマンド | ✅ Yes | `lmpipe` コマンド |
| Standalone実行 | ✅ Yes | 単独で動作可能 |
| 設定ベース動作 | ✅ Yes | CLI引数、Options オブジェクト |
| コード不要で実行可能 | ✅ Yes | プラグイン選択のみ |
| バッチ処理 | ✅ Yes | 複数ファイルの自動処理 |
| プログレスレポート | ✅ Yes | Rich で進行状況表示 |

**使用例 (コード不要)**:
```bash
# Python コードを書かずに実行可能
lmpipe mediapipe.holistic video.mp4 -o output.npz

# バッチ処理も可能
lmpipe mediapipe.pose videos/ -o results.zarr --workers 8
```

**結論**: 
**強く該当する**。ユーザーがPythonコードを書かずに、コマンドラインだけで利用可能。

**類似例**: 
- ✅ FFmpeg (設定ベース実行)
- ✅ Terraform (宣言的設定)
- ✅ pytest (コマンド実行 + プラグイン)

---

### 5. Framework-as-an-App (アプリケーション型フレームワーク)

**定義**: フレームワークとアプリケーションの境界が曖昧で、両方の側面を持つ

**評価**: ✅ **該当する (80%)**

| 基準 | 判定 | 理由 |
|------|------|------|
| エンドユーザーツール | ✅ Yes | 研究者が直接使用 |
| 開発者ライブラリ | ✅ Yes | Python API 提供 |
| CLI + API 両立 | ✅ Yes | 両方サポート |
| 即座に使える | ✅ Yes | インストール後すぐ動作 |
| カスタマイズ可能 | ✅ Yes | 拡張ポイント多数 |
| 特定ドメイン | ✅ Yes | ランドマーク抽出に特化 |

**二面性**:
1. **アプリケーション側面**:
   - CLI ツールとして動作
   - エンドユーザー向けドキュメント
   - すぐに使える機能

2. **フレームワーク側面**:
   - 拡張可能なアーキテクチャ
   - 抽象基底クラス
   - プラグインシステム

**使用パターン**:
```bash
# アプリケーションとして (コード不要)
lmpipe mediapipe.holistic input.mp4 -o output.npz
```

```python
# フレームワークとして (カスタマイズ)
from cslrtools2.lmpipe import Estimator, LMPipeInterface

class MyEstimator(Estimator):
    ...

interface = LMPipeInterface(MyEstimator())
interface.run("input.mp4", "output/")
```

**結論**: 
**該当する**。アプリケーションとフレームワークの両方の性質を持つハイブリッド設計。

**類似例**: 
- ✅ Jupyter (ノートブック + 拡張システム)
- ✅ VS Code (エディター + 拡張フレームワーク)
- ✅ pytest (テストランナー + プラグインフレームワーク)

---

### 6. Runtime / Runtime Environment (ランタイム/実行環境)

**定義**: プログラムを実行するための環境を提供 (VM、インタープリター、標準ライブラリ等)

**評価**: ❌ **該当しない (10%)**

| 基準 | 判定 | 理由 |
|------|------|------|
| 独自VM/インタープリター | ❌ No | Python 3.12+ に依存 |
| メモリ管理 | ❌ No | Python の GC を使用 |
| 標準ライブラリ | ⚠️ Partial | 限定的 (処理ユーティリティのみ) |
| プラットフォーム抽象化 | ❌ No | OS 依存 |
| プロセス管理 | ⚠️ Partial | Executor で一部管理 |
| ガベージコレクション | ❌ No | Python に依存 |

**結論**: 
該当しない。Python ランタイムの上で動作するライブラリであり、独自のランタイムを提供していない。

**類似例**: 
- ❌ Not like: JVM, Node.js, .NET Runtime (独立したランタイム)

---

## 🎯 総合評価と推奨分類

### 適合度ランキング

| 順位 | 分類 | 適合度 | 評価 |
|------|------|--------|------|
| 🥇 1位 | **Extensible Framework** | 90% | ✅ 最適 |
| 🥈 2位 | **Executable Framework** | 85% | ✅ 適切 |
| 🥉 3位 | **Framework-as-an-App** | 80% | ✅ 適切 |
| 4位 | Platform | 40% | ⚠️ 部分的 |
| 5位 | Runtime | 10% | ❌ 不適切 |
| 6位 | Self-hosted Framework | 5% | ❌ 不適切 |

---

## 📝 推奨される正式分類

### 主分類 (Primary Classification)

**🎯 Extensible Application Framework for Domain-Specific Processing**

**日本語**: **ドメイン特化型拡張可能アプリケーションフレームワーク**

### 副分類 (Secondary Classifications)

1. **Processing Pipeline Framework** (処理パイプラインフレームワーク)
2. **ETL Framework** (Extract-Transform-Load フレームワーク)
3. **CLI-based Extensible Tool** (CLI ベース拡張可能ツール)

---

## 🔬 詳細な特性分析

### LMPipe は何か?

LMPipe は以下の特性を組み合わせた**ハイブリッドアーキテクチャ**です:

#### 1. **Application Framework** (アプリケーションフレームワーク)
- ✅ 特定のタスク (ランドマーク抽出) に特化
- ✅ すぐに使える CLI アプリケーション
- ✅ エンドユーザー向けドキュメント

#### 2. **Extensible Library** (拡張可能ライブラリ)
- ✅ 抽象基底クラス (Estimator, Collector)
- ✅ プラグインシステム (Entry Points)
- ✅ 制御の逆転 (IoC) と依存性注入
- ✅ イベント駆動アーキテクチャ

#### 3. **Command-line Tool** (コマンドラインツール)
- ✅ Console script (`lmpipe` コマンド)
- ✅ コード不要で実行可能
- ✅ バッチ処理機能

#### 4. **Pipeline Engine** (パイプラインエンジン)
- ✅ ETL パターン実装
- ✅ ストリーミング処理
- ✅ 並列実行管理

---

## 💡 提案: より適切な用語

既存の6分類の中にない場合、LMPipeは以下のように分類すべきです:

### 🎯 最も適切な新しい分類

**"Domain-Specific Application Framework with Embedded CLI"**

**日本語**: **CLI組み込み型ドメイン特化アプリケーションフレームワーク**

### 特徴:
1. **Domain-Specific** (ドメイン特化)
   - 手話動画のランドマーク抽出という明確な目的
   - 汎用的すぎず、狭すぎない適切なスコープ

2. **Application Framework** (アプリケーションフレームワーク)
   - アプリケーションとフレームワークの両側面
   - すぐ使える + カスタマイズ可能

3. **Embedded CLI** (CLI組み込み)
   - フレームワークの一部として CLI が統合
   - コマンドラインとPython APIの両立

4. **Extensible** (拡張可能)
   - プラグインシステム
   - 抽象基底クラス
   - イベントフック

### 類似プロジェクト:
- **pytest**: テストフレームワーク + CLI + プラグイン
- **Scrapy**: Webスクレイピングフレームワーク + CLI + 拡張性
- **Airflow**: データパイプラインフレームワーク + CLI + プラグイン
- **FFmpeg**: メディア処理フレームワーク + CLI + フィルター拡張

---

## 📋 実装上の推奨事項

### ドキュメント更新

`src/cslrtools2/lmpipe/__init__.py` の分類を以下に更新することを推奨:

```python
"""LMPipe: Landmark Extraction Pipeline Framework for Sign Language Videos.

**Software Type**: Domain-Specific Application Framework / Processing Pipeline Framework  
**Pattern**: Pipeline Pattern, Plugin Architecture, Command Pattern  
**Architecture**: Extensible Framework with Embedded CLI
**Dependencies**: MediaPipe, OpenCV, NumPy, Rich (for progress bars)
```

### 追加すべき説明

```python
Framework Classification
------------------------

LMPipe is best classified as a **Domain-Specific Application Framework** that combines:

1. **Application Aspect**:
   - Ready-to-use CLI tool (``lmpipe`` command)
   - Immediate functionality for end users
   - No coding required for basic usage

2. **Framework Aspect**:
   - Extensible architecture with abstract base classes
   - Plugin system via entry points
   - Inversion of Control (IoC) and Dependency Injection
   - Event-driven customization hooks

3. **Pipeline Engine**:
   - ETL (Extract-Transform-Load) pattern
   - Parallel processing support
   - Progress tracking and resource management

This hybrid design allows it to function as both:
- **Executable Tool**: ``lmpipe mediapipe.holistic video.mp4 -o output.npz``
- **Developer Library**: Custom ``Estimator`` subclasses and Python API

Similar projects include pytest, Scrapy, and Airflow - tools that provide
immediate functionality while remaining highly extensible.
```

---

## 🎯 最終結論

### 推奨される公式分類

**Primary**: **Extensible Application Framework**  
**Secondary**: **Processing Pipeline Framework with CLI**  
**Tertiary**: **Domain-Specific ETL Framework**

### 根拠

1. ✅ **Extensible Framework** (90%適合)
   - プラグインシステム完備
   - 抽象基底クラス設計
   - IoC/DI パターン実装
   - イベント駆動アーキテクチャ

2. ✅ **Executable Framework** (85%適合)
   - CLI として単独実行可能
   - コード不要で動作
   - 設定ベースのカスタマイズ

3. ✅ **Framework-as-an-App** (80%適合)
   - アプリとフレームワークの二面性
   - エンドユーザーと開発者の両方をサポート

### 不適切な分類

- ❌ **Platform**: 汎用性が不足 (特定ドメインに特化)
- ❌ **Runtime**: 独自実行環境を提供していない
- ❌ **Self-hosted**: ホスティング機能なし

### 一言で表現すると

**"Pytest for video landmark extraction"**

- すぐ使える (CLI)
- 拡張可能 (プラグイン)
- 特定ドメイン (ランドマーク抽出)
- ハイブリッド設計 (ツール + フレームワーク)

---

## 📚 参考文献

- **Framework Patterns**: Gang of Four Design Patterns
- **Plugin Architecture**: OSGi, Eclipse Plugin System
- **CLI Framework**: Click, argparse ecosystem
- **Pipeline Patterns**: Enterprise Integration Patterns (Gregor Hohpe)
- **Similar Projects**: pytest, Scrapy, Airflow, FFmpeg

---

**評価完了日**: 2025年11月14日  
**次のアクション**: `lmpipe/__init__.py` のdocstring更新を推奨
