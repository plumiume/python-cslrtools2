# Dependency Watch List

このドキュメントでは、プロジェクトが監視すべき外部依存関係の重要な変更や制約について記載します。

## MediaPipe - Protobuf 依存関係

### 現状 (2025-11-21時点)

**問題**: MediaPipe 0.10.14 が protobuf 4.x系に依存しており、Python 3.14との互換性問題が存在

- **MediaPipe バージョン**: 0.10.14
- **Protobuf 要件**: `>=4.25.3,<5` (mainブランチ確認済み)
- **Python サポート**: 3.12, 3.13のみ (3.14は非対応)

### 技術的背景

#### Protobuf Issue #15077
- **問題**: protobuf 4.25.8 が Python 3.14 で DeprecationWarning を発生
- **警告内容**: `Type google._upb._message.MessageMapContainer uses PyType_Spec with a metaclass that has custom tp_new`
- **修正**: protobuf 5.26.0+ (2024-03-06リリース) で解決済み
- **GitHub Issue**: https://github.com/protocolbuffers/protobuf/issues/15077

#### MediaPipe の状況
- MediaPipe はまだ protobuf 5.x系へのアップグレードを行っていない
- API互換性の問題により protobuf 4.x系に留まっている
- `requirements.txt`: `protobuf>=4.25.3,<5`

### 対応状況

#### このプロジェクト
- **Python制約**: `requires-python = ">=3.12,<3.14"` (pyproject.toml)
- **理由**: MediaPipe の protobuf 依存による Python 3.14 非互換性
- **ステータス**: 全テスト正常動作 (613 passed, 15 skipped)

### 監視すべき項目

#### 1. MediaPipe リリース
**監視対象**: https://github.com/google/mediapipe/releases

**チェック項目**:
- [ ] protobuf 5.x系への依存関係更新
- [ ] `requirements.txt` の protobuf バージョン制約変更
- [ ] Python 3.14 サポート追加

**確認コマンド**:
```bash
# MediaPipe mainブランチの requirements.txt を確認
curl -s https://raw.githubusercontent.com/google/mediapipe/master/requirements.txt | grep protobuf
```

**期待される変更**:
```
# 現在: protobuf>=4.25.3,<5
# 将来: protobuf>=5.26.0,<6  (or similar)
```

#### 2. Protobuf バージョン
**監視対象**: https://github.com/protocolbuffers/protobuf/releases

**重要バージョン**:
- **4.25.8**: 現在MediaPipeが使用 (Python 3.14警告あり)
- **5.26.0+**: Python 3.14互換性問題修正済み

#### 3. Python バージョン
**監視対象**: https://www.python.org/downloads/

**重要情報**:
- **Python 3.14.0**: 2025-10-07リリース済み
- **Python 3.15**: 2026-10予定 (PEP 790)

### アップグレード手順

MediaPipeがprotobuf 5.x対応したら以下を実施:

#### 1. 依存関係の確認
```bash
# MediaPipeの最新バージョンを確認
uv pip index versions mediapipe

# 新バージョンの依存関係を確認
pip show mediapipe
```

#### 2. pyproject.toml の更新
```toml
# 変更前
requires-python = ">=3.12,<3.14"

# 変更後 (MediaPipeがPython 3.14対応したら)
requires-python = ">=3.12"
```

#### 3. README.md の更新
```markdown
# 変更前
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)]

# 変更後
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)]
```

```markdown
# 削除対象
> **Note:** Currently supports Python 3.12 and 3.13. Python 3.14 is not supported...
```

#### 4. テストの実施
```bash
# Python 3.14環境でテスト
uv run --python 3.14 pytest tests/

# MediaPipe統合テストを重点的に確認
uv run --python 3.14 pytest tests/integration/test_lmpipe_e2e_basic.py -v
```

### 関連リソース

#### 公式ドキュメント
- [MediaPipe Python API](https://developers.google.com/mediapipe/solutions/guide)
- [Protobuf Python Tutorial](https://protobuf.dev/getting-started/pythontutorial/)
- [Python Release Schedule](https://peps.python.org/pep-0745/)

#### Issue追跡
- [protobuf#15077](https://github.com/protocolbuffers/protobuf/issues/15077) - Python 3.14 DeprecationWarning
- [tensorflow#68194](https://github.com/tensorflow/tensorflow/issues/68194) - 関連する TensorFlow issue
- [googleapis/google-cloud-python#12560](https://github.com/googleapis/google-cloud-python/issues/12560) - Google Cloud での同様の問題

#### プロジェクト内ドキュメント
- `README.md` - Python バージョン制約の説明
- `pyproject.toml` - requires-python 設定
- `tests/integration/test_lmpipe_e2e_basic.py` - MediaPipe 統合テスト

### チェックリスト (定期確認)

月次または四半期ごとに以下を確認:

- [ ] MediaPipe の最新バージョン確認
- [ ] MediaPipe の protobuf 依存関係確認
- [ ] protobuf 最新バージョンのPython互換性確認
- [ ] Python 新バージョンのリリース状況確認
- [ ] このプロジェクトのPython制約が妥当か再評価

### 備考

**現時点での推奨**:
- DeprecationWarning は無視可能 (実動作に影響なし)
- Python 3.14 で利用したい場合は MediaPipe のアップデートを待つ
- 緊急対応が必要な場合は pytest の警告フィルタで抑制可能

**将来的な対応**:
- MediaPipe が protobuf 5.x対応後、速やかにPython 3.14サポートを追加
- この制約は一時的なものであり、upstream修正待ちの状態
