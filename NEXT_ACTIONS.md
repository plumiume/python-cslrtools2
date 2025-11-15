# cslrtools2 次のアクション提案

**更新日**: 2025年11月14日  
**現在のステータス**: Alpha v0.1.0 (PyPI公開準備完了)  
**直近の成果**: 型システム近代化完了、モジュール分割完了

---

## 🎯 プロジェクト概要

### 各モジュールの役割

#### 📹 **LMPipe** - Landmark Extraction Pipeline
**目的**: 手話動画から姿勢・手・顔のランドマークを自動抽出  
**現状**: 
- ✅ MediaPipe統合完了
- ✅ 並列処理対応
- ✅ 7種類の出力形式サポート
- ✅ CLI完備 (`lmpipe` コマンド)
- ✅ プラグインアーキテクチャ実装済み

**使用例**:
```bash
lmpipe mediapipe.holistic input_video.mp4 -o landmarks.npz --workers 4
```

#### 📊 **SLDataset** - Sign Language Dataset Management  
**目的**: 手話データセットの効率的な管理と読み込み  
**現状**:
- ✅ Zarr形式ストレージ実装
- ✅ PyTorch Dataset互換
- ✅ 柔軟なスキーマ設計
- ✅ CLI完備 (`sldataset2` コマンド)
- ⚠️ ドキュメント不足

**使用例**:
```python
from cslrtools2.sldataset import SLDataset
dataset = SLDataset.from_zarr(zarr_group)
```

#### 🧮 **ConvSize** - PyTorch Convolution Utilities
**目的**: 畳み込み層の出力サイズ計算補助  
**現状**:
- ✅ 基本機能実装完了
- ✅ テスト完備 (5/14テスト)
- ⚠️ ドキュメント不足

---

## 🚀 優先度別アクション

### 🔴 **Priority 1: PyPI公開準備** (推定: 1-2日)

#### Task 1.1: バージョン管理最終確認
- [ ] Git tagでバージョン指定テスト
- [ ] `hatch-vcs` 動作確認
- [ ] バージョン番号取得確認 (`python -c "import cslrtools2; print(cslrtools2.__version__)"`)

**コマンド**:
```bash
git tag -a v0.1.0 -m "Initial alpha release"
git push origin v0.1.0
uv build
```

#### Task 1.2: PyPI Test環境での公開テスト
- [ ] TestPyPIアカウント作成
- [ ] TestPyPIへのアップロード
- [ ] TestPyPIからのインストールテスト

**コマンド**:
```bash
uv publish --repository testpypi
pip install --index-url https://test.pypi.org/simple/ cslrtools2
```

#### Task 1.3: 本番PyPI公開
- [ ] CHANGELOG.md 最終レビュー
- [ ] README.md 最終レビュー
- [ ] PyPIアカウント設定
- [ ] 本番公開

**コマンド**:
```bash
uv publish
```

**期待成果**: 
- `pip install cslrtools2` で誰でもインストール可能
- ライブラリチェック: 100/130 → 110/130 (+10点)

---

### 🟠 **Priority 2: ドキュメント整備** (推定: 2-3日)

#### Task 2.1: Sphinx API ドキュメント自動生成
**現状**: `sphinx/` ディレクトリに環境設定済み

- [ ] Sphinx設定ファイル最終調整
- [ ] APIドキュメント自動生成
- [ ] サンプルコード追加
- [ ] GitHub Pages デプロイ設定

**コマンド**:
```bash
cd sphinx
sphinx-build -b html . _build/html
```

**ファイル構成**:
```
docs/
├── index.md (トップページ)
├── installation.md (インストールガイド)
├── api/
│   ├── index.md
│   ├── lmpipe.md (自動生成)
│   └── sldataset.md (自動生成)
└── examples/
    ├── index.md
    ├── lmpipe_basic.md
    └── sldataset_basic.md
```

#### Task 2.2: チュートリアル作成
- [ ] LMPipe 基本チュートリアル (Jupyter Notebook)
- [ ] SLDataset 基本チュートリアル
- [ ] カスタムEstimator作成ガイド
- [ ] プラグイン開発ガイド

**期待成果**:
- ユーザーが自己学習可能
- ライブラリチェック: +15点 (ドキュメント完備)

---

### 🟡 **Priority 3: 例外処理・ログ改善** (推定: 2-3日)

**現状**: `EXCEPTION_LOGGING_TODO.md` に詳細な実装計画あり

#### Task 3.1: カスタム例外階層実装
- [ ] `src/cslrtools2/exceptions.py` 作成 (既に存在する可能性あり)
- [ ] 例外クラス定義完了
- [ ] 既存コードへの適用

**例外階層**:
```
CSLRToolsError (base)
├── ConfigurationError
├── ValidationError
├── LMPipeError
│   ├── EstimatorError
│   ├── CollectorError
│   └── VideoProcessingError
└── SLDatasetError
    ├── DataLoadError
    └── DataFormatError
```

#### Task 3.2: ロギング改善
- [ ] 統一されたログフォーマット
- [ ] ログレベル適切化
- [ ] コンテキスト情報追加

**期待成果**:
- デバッグ効率向上
- ユーザーフレンドリーなエラーメッセージ
- ライブラリチェック: +5点

---

### 🟢 **Priority 4: 機能拡張** (推定: 1-2週間)

#### Task 4.1: より多くのデータセットプラグイン
- [ ] WLASL (Word-Level American Sign Language) サポート
- [ ] PHOENIX-2014 サポート
- [ ] CSL (Chinese Sign Language) サポート

#### Task 4.2: 追加の出力フォーマット
- [ ] HDF5フォーマットサポート
- [ ] Parquetフォーマットサポート
- [ ] TFRecordフォーマットサポート

#### Task 4.3: 可視化ツール拡張
- [ ] Web UIでのランドマーク可視化
- [ ] アニメーション生成機能
- [ ] 統計分析ダッシュボード

**期待成果**:
- より広範なユースケース対応
- コミュニティ貢献促進

---

### 🔵 **Priority 5: テスト・CI/CD強化** (推定: 1-2日)

#### Task 5.1: テストカバレッジ向上
**現状**: 14テスト (主にインポートテスト)

- [ ] LMPipe 統合テスト追加
- [ ] SLDataset 統合テスト追加
- [ ] エンドツーエンドテスト追加
- [ ] カバレッジ80%以上達成

#### Task 5.2: CI/CD パイプライン構築
- [ ] GitHub Actions設定
  - [ ] 自動テスト実行
  - [ ] 自動型チェック (Pyright)
  - [ ] 自動ビルド
  - [ ] 自動PyPI公開
- [ ] pre-commit hooks設定

**GitHub Actions ワークフロー例**:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov=src
      - run: pyright src/
```

**期待成果**:
- 自動品質保証
- リグレッション防止
- ライブラリチェック: +10点

---

## 📅 推奨スケジュール (2週間計画)

### Week 1: 公開準備とドキュメント

| 日 | タスク | 成果物 |
|----|--------|--------|
| Day 1 | PyPI公開準備 (Task 1.1-1.2) | TestPyPI公開完了 |
| Day 2 | PyPI本番公開 (Task 1.3) | PyPI公開完了 🎉 |
| Day 3 | Sphinx設定 (Task 2.1) | APIドキュメント生成 |
| Day 4 | チュートリアル作成 (Task 2.2) | LMPipe チュートリアル |
| Day 5 | チュートリアル作成 (Task 2.2) | SLDataset チュートリアル |
| Day 6-7 | 例外処理改善 (Task 3.1-3.2) | カスタム例外実装 |

### Week 2: 品質向上と拡張

| 日 | タスク | 成果物 |
|----|--------|--------|
| Day 8-9 | テストカバレッジ向上 (Task 5.1) | 80%カバレッジ達成 |
| Day 10 | CI/CD構築 (Task 5.2) | GitHub Actions稼働 |
| Day 11-14 | 機能拡張 (Task 4.1-4.3) | 新機能追加 |

---

## 🎯 今日・明日の具体的アクション

### ✅ 今日 (Day 1): PyPI公開準備

#### 1. バージョンタグ作成
```bash
cd c:\Users\ikeko\Workspace\1github\python-cslrtools2
git tag -a v0.1.0 -m "Initial alpha release - Type system modernization complete"
git push origin v0.1.0
```

#### 2. ビルドテスト
```bash
# Clean previous builds
rm -rf dist/

# Build wheel and source distribution
uv build

# Verify build artifacts
ls dist/
```

#### 3. TestPyPI アカウント設定
- [ ] https://test.pypi.org/ でアカウント作成
- [ ] API token生成
- [ ] `~/.pypirc` 設定

#### 4. TestPyPI アップロード
```bash
uv publish --repository testpypi
```

#### 5. TestPyPI インストールテスト
```bash
# 新しい環境で
python -m venv test_env
test_env\Scripts\activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cslrtools2
python -c "import cslrtools2; print(cslrtools2.__version__)"
```

---

### ✅ 明日 (Day 2): PyPI本番公開

#### 1. 最終レビュー
- [ ] README.md 読み直し
- [ ] CHANGELOG.md 確認
- [ ] LICENSE 確認
- [ ] pyproject.toml メタデータ確認

#### 2. 本番PyPI公開
```bash
uv publish
```

#### 3. 公開確認
```bash
pip install cslrtools2
python -c "import cslrtools2; print(cslrtools2.__version__)"
```

#### 4. アナウンス準備
- [ ] GitHub Release作成
- [ ] README.mdバッジ更新
- [ ] SNS投稿準備

---

## 📊 期待される成果

### ライブラリチェックスコア推移

| タイミング | スコア | 改善点 |
|-----------|--------|--------|
| **現在** | 100/130 | 基盤完成 |
| PyPI公開後 | 110/130 | +10 (配布) |
| ドキュメント完備 | 125/130 | +15 (文書) |
| CI/CD構築 | 135/130 | +10 (自動化) |

### プロジェクト成熟度

- **現在**: Alpha (機能は動くが、ドキュメント不足)
- **PyPI公開後**: Beta候補 (一般利用可能)
- **ドキュメント完備後**: Beta (実用レベル)
- **CI/CD構築後**: Stable候補 (本番利用可能)

---

## 🤔 判断が必要な事項

### 1. バージョン番号
- **現在**: v0.1.0 (alpha)
- **選択肢**:
  - `v0.1.0` (現状維持)
  - `v0.1.0-alpha.1` (明示的alpha表記)
  - `v0.0.1` (さらに控えめ)

**推奨**: `v0.1.0` (現状維持、PyPIでは `Development Status :: 3 - Alpha` で表現)

### 2. 公開範囲
- **オプション**:
  - TestPyPIのみ (安全策)
  - 本番PyPI公開 (推奨、現状で十分品質高い)

**推奨**: 本番PyPI公開 (Alpha表記で注意喚起済み)

### 3. MediaPipe依存関係
- **現状**: optional dependency (ユーザーが手動インストール)
- **選択肢**:
  - 現状維持 (推奨、ライセンス・サイズ問題回避)
  - extras_require追加 (`pip install cslrtools2[mediapipe]`)

**推奨**: 現状維持 (README.mdで明記済み)

---

## 🎓 学習リソース

今後の開発で参考になるリソース:

1. **Python Packaging**: https://packaging.python.org/
2. **Sphinx Documentation**: https://www.sphinx-doc.org/
3. **GitHub Actions**: https://docs.github.com/actions
4. **PyPI Publishing**: https://pypi.org/help/
5. **Semantic Versioning**: https://semver.org/

---

## 📝 メモ

- Git push完了済み (2025/11/14)
- 型システム近代化完了 (future annotations導入)
- モジュール分割完了 (interface → __init__ + runner)
- すべてのテスト合格 (14/14)
- 型エラーゼロ

**次の大きなマイルストーン**: PyPI公開 🚀
