# 🚀 ワークスペース構成ガイド

## 📍 現在の並列ワークスペース構成

このリポジトリは **Git Worktree** を使用した並列作業環境として構成されています。

### 構成済みワークスペース一覧

```
C:\Users\ikeda\Workspace\1github\
├── python-cslrtools2/           # メインワークスペース
│   ├── ブランチ: dev-ai/fix-sldataset-logger
│   ├── .venv/                   # 独立仮想環境
│   └── 役割: logger修正・メイン開発
│
├── cslrtools2-dataset/          # データセット機能強化
│   ├── ブランチ: dev-ai/dataset-enhancement
│   ├── .venv/                   # 独立仮想環境
│   ├── 役割: SLDataset機能追加・テスト環境構築
│   └── 🎯 取り込むブランチ: dev-ai/merge-integration
│
├── cslrtools2-merge/            # ブランチ統合専用
│   ├── ブランチ: dev-ai/merge-integration
│   ├── .venv/                   # 独立仮想環境
│   ├── 役割: 複数ブランチのマージ・統合テスト
│   └── ✅ 統合済み: utilities-expansion, dependencies-update, gitignore-cleanup
│
├── cslrtools2-dependencies/     # 依存関係管理
│   ├── ブランチ: dev-ai/dependencies-update (detached)
│   ├── .venv/                   # 独立仮想環境
│   └── 役割: 依存関係の更新・互換性テスト
│
├── cslrtools2-gitignore/        # Git・Docker設定
│   ├── ブランチ: dev-ai/gitignore-cleanup (detached)
│   ├── .venv/                   # 独立仮想環境
│   └── 役割: .gitignore最適化・Docker環境構築
│
└── cslrtools2-utilities/        # ユーティリティ拡張
    ├── ブランチ: dev-ai/utilities-expansion (detached)
    ├── .venv/                   # 独立仮想環境
    └── 役割: コアユーティリティ・例外処理追加
```

## ✅ セットアップ済み内容

### 1. Git Worktree構成
- [x] メインワークスペース: `python-cslrtools2`
- [x] データセットワークスペース: `cslrtools2-dataset` ⭐ **推奨作業場所**
- [x] マージワークスペース: `cslrtools2-merge`
- [x] 依存関係ワークスペース: `cslrtools2-dependencies`
- [x] Git設定ワークスペース: `cslrtools2-gitignore`
- [x] ユーティリティワークスペース: `cslrtools2-utilities`

### 2. 各ワークスペースの仮想環境
- [x] 全ワークスペースで `.venv` ディレクトリ作成済み
- [x] Python 3.12.11 を使用
- [x] 依存関係インストール済み（`--index-strategy unsafe-best-match`）

### 3. インストール済みパッケージ
- PyTorch 2.9.0+cu128 (utilities, gitignore) / 2.9.1 (dependencies)
- NumPy 2.3.4
- Zarr 3.1.3
- safetensors 0.6.2
- その他40+パッケージ

## 🔄 ワークスペース切り替え方法

### PowerShellでの切り替え

```powershell
# メインワークスペースに移動
cd C:\Users\ikeda\Workspace\1github\python-cslrtools2

# 依存関係ワークスペースに移動
cd ..\cslrtools2-dependencies

# Git設定ワークスペースに移動
cd ..\cslrtools2-gitignore

# ユーティリティワークスペースに移動
cd ..\cslrtools2-utilities
```

### 一時的な移動（Push/Pop-Location）

```powershell
# 一時的に別ワークスペースで作業
Push-Location ..\cslrtools2-utilities
uv run python script.py
Pop-Location  # 元のディレクトリに戻る
```

## 🛠️ 開発ワークフロー

### 各ワークスペースでの基本操作

```powershell
# ワークスペースに移動
cd ..\cslrtools2-<name>

# 必ずuv run pythonを使用
uv run python -c "import cslrtools2; print('OK')"

# テスト実行
uv run pytest

# コマンドラインツール実行
uv run lmpipe --help
```

### コミット・プッシュ

```powershell
# 変更を確認
git status
git diff

# コミット
git add .
git commit -m "feat: Description"

# プッシュ（detached HEADの場合はブランチ作成が必要）
git checkout -b dev-ai/my-changes
git push origin dev-ai/my-changes
```

## 🔍 ワークスペース管理コマンド

### 現在の構成を確認

```powershell
# worktree一覧表示
git worktree list

# 出力例:
# C:/Users/ikeda/Workspace/1github/python-cslrtools2        e3cbc45 [dev-ai/fix-sldataset-logger]
# C:/Users/ikeda/Workspace/1github/cslrtools2-dependencies  47cf266 (detached HEAD)
# ...
```

### 新しいworktreeを追加

```powershell
# メインワークスペースで実行
cd C:\Users\ikeda\Workspace\1github\python-cslrtools2

# 新規ブランチでworktree作成
git worktree add ..\cslrtools2-<name> -b dev-ai/<task-name>

# 既存ブランチでworktree作成
git worktree add ..\cslrtools2-<name> origin/dev-ai/<branch-name>

# 環境セットアップ（uv sync --all-groups推奨）
cd ..\cslrtools2-<name>
uv sync --all-groups

# 最小構成の場合（MediaPipeなし）
# uv sync
```

### worktreeの削除

```powershell
# 作業完了後、worktreeを削除
git worktree remove ..\cslrtools2-<name>

# 強制削除（変更がある場合）
git worktree remove ..\cslrtools2-<name> --force
```

## ⚠️ 重要な注意事項

### DO（推奨）

✅ **必ず`uv run python`を使用**
```powershell
uv run python script.py  # ✓ 正しい
python script.py         # ✗ 間違い
```

✅ **各ワークスペースで独立した環境を維持**
- `.venv`は共有しない
- パッケージは各ワークスペースで個別にインストール

✅ **頻繁にコミット＆プッシュ**
- AIエージェント間の協調のため小さく頻繁に

✅ **Conventional Commits形式を使用**
```
feat: 新機能
fix: バグ修正
docs: ドキュメント
chore: ビルド・設定変更
```

### DON'T（避けるべき）

❌ **複数ワークスペースで同じブランチを編集しない**
- Worktreeは同じブランチを複数箇所でチェックアウトできません

❌ **`.venv`や`__pycache__`を共有しない**
- Git管理対象外なので、各自で再生成

❌ **bare pythonコマンドを使わない**
- 必ず`uv run python`経由で実行

## 🔧 トラブルシューティング

### 依存関係インストールエラー

```powershell
# エラー: No solution found when resolving dependencies
# 解決: --index-strategy unsafe-best-matchを使用
uv pip install -e . --index-strategy unsafe-best-match
```

### detached HEAD状態からブランチ作成

```powershell
# 現在の状態を確認
git log --oneline -1

# 新しいブランチを作成
git checkout -b dev-ai/my-branch-name

# プッシュ
git push origin dev-ai/my-branch-name
```

### worktree削除時のエラー

```powershell
# エラー: worktree has modifications
# 解決1: 変更をコミット
git add .
git commit -m "chore: Save changes"

# 解決2: 強制削除（注意: 変更が失われます）
git worktree remove ..\cslrtools2-<name> --force
```

## 📊 現在のブランチ状況

### リモートdev-aiブランチ
- `origin/dev-ai/dependencies-update` → `cslrtools2-dependencies`
- `origin/dev-ai/gitignore-cleanup` → `cslrtools2-gitignore`
- `origin/dev-ai/utilities-expansion` → `cslrtools2-utilities`
- `origin/dev-ai/integrate-gitignore-and-docker`
- `origin/dev-ai/mp-constants-refactor`
- `origin/dev-ai/torch-2.3-compatibility-test`

### ローカル作業ブランチ
- `dev-ai/fix-sldataset-logger` → `python-cslrtools2`（メイン）

## 🎯 クイックリファレンス

```powershell
# 構成確認
git worktree list

# 別ワークスペースで一時作業
Push-Location ..\cslrtools2-utilities
uv run python script.py
Pop-Location

# 新worktree作成
git worktree add ..\cslrtools2-newfeature -b dev-ai/new-feature
cd ..\cslrtools2-newfeature
uv sync --all-groups

# 最小構成の場合（MediaPipeなし）
# uv sync

# 状態確認
git status
git log --oneline -5

# コミット＆プッシュ
git add .
git commit -m "feat: New feature"
git push origin dev-ai/new-feature
```

## 🎯 ワークスペース別の推奨作業

### cslrtools2-dataset（dataset機能強化）

**目的**: SLDataset機能の拡充、テスト環境構築、ドキュメント整備

**取り込むべきブランチ**:
```powershell
# dev-ai/merge-integrationから最新の統合成果を取り込む
cd C:\Users\ikeda\Workspace\1github\cslrtools2-dataset
git fetch origin
git merge origin/dev-ai/merge-integration
```

**優先タスク**:
1. ✅ **テスト環境の構築**
   - `pyproject.toml`に`[dependency-groups.test]`追加（pytest, pytest-cov）
   - `tests/test_sldataset.py`作成（基本的なCRUD操作テスト）
   - `tests/test_array_loader.py`作成（複数フォーマット対応テスト）

2. 📝 **sldataset2 CLIの実装**
   - `sldataset2 info <dataset.zarr>`: データセット統計表示
   - `sldataset2 validate <dataset.zarr>`: 整合性チェック
   - `sldataset2 convert <input> <output>`: フォーマット変換

3. 📚 **ドキュメント拡充**
   - `README.md`にFluentSigners50プラグイン使用例追加
   - `sldataset2`コマンドの実用例追加
   - データセット作成チュートリアル作成

4. 🔧 **コード改善**
   - `dataset.py`の型ヒント強化
   - Zarr型スタブ（`typings/zarr/`）の完成
   - エラーハンドリング改善

**開発ワークフロー**:
```powershell
cd C:\Users\ikeda\Workspace\1github\cslrtools2-dataset

# テスト実行
uv run pytest tests/ -v

# 型チェック
uv run pyright src/

# コマンド動作確認
uv run sldataset2 --help
```

### cslrtools2-merge（ブランチ統合専用）

**目的**: 複数dev-aiブランチの統合・テスト・main合流準備

**既に統合済み**:
- ✅ utilities-expansion
- ✅ dependencies-update  
- ✅ gitignore-cleanup
- ✅ tests/ディレクトリ追加

**次のステップ**:
```powershell
cd C:\Users\ikeda\Workspace\1github\cslrtools2-merge

# dataset-enhancementが完成したら統合
git merge dev-ai/dataset-enhancement

# 全テスト実行
uv run pytest tests/ -v --cov=cslrtools2

# mainへのマージ準備
git checkout main
git merge --squash dev-ai/merge-integration
git commit -m "feat: Integrate dataset enhancements, tests, and utilities"
```

---

## 🤖 その他できること

### パフォーマンス最適化
- [ ] Zarr配列読み込みの遅延評価最適化
- [ ] MediaPipeバッチ処理の並列化改善
- [ ] キャッシュ機構の導入（lmpipe結果のキャッシュ）

### 新機能追加
- [ ] データ拡張（augmentation）サポート（SLDatasetに統合）
- [ ] ストリーミング推論モード（リアルタイムランドマーク抽出）
- [ ] TensorBoard統合（学習プロセス可視化）

### ドキュメント
- [ ] アーキテクチャ図の生成（Mermaid/PlantUML）
- [ ] APIリファレンスの自動生成（Sphinx）
- [ ] 貢献ガイドライン（CONTRIBUTING.md）

### CI/CD
- [ ] GitHub Actions: pytestとPyright自動実行
- [ ] GitHub Actions: PyPI自動公開ワークフロー
- [ ] pre-commitフックの設定（型チェック、フォーマット）

---

**最終更新**: 2025年11月14日  
**構成完了日**: 2025年11月14日

この構成により、複数のタスクを効率的に並列実行できます。各ワークスペースは独立しているため、ブランチ切り替えなしで複数の機能開発を同時進行できます。
