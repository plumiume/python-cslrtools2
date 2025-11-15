# Tests Directory Structure

このディレクトリには、cslrtools2プロジェクトのすべてのテストが含まれています。

## ディレクトリ構造

```
tests/
├── import/              # インポートテスト
│   ├── test_imports.py  # 基本的なインポートテスト
│   ├── test_mediapipe_constants.py  # MediaPipe定数のテスト
│   └── test_sldataset_imports.py    # SLDatasetインポートテスト
│
├── unit/                # ユニットテスト
│   ├── lmpipe/          # lmpipeモジュールのユニットテスト
│   │   ├── collector/   # Collectorのテスト
│   │   │   ├── test_base.py
│   │   │   ├── test_csv_lmsc.py
│   │   │   ├── test_npy_lmsc.py
│   │   │   ├── test_npz_lmsc.py
│   │   │   ├── test_zarr_lmsc.py
│   │   │   ├── test_safetensors_lmsc.py
│   │   │   └── test_torch_lmsc.py
│   │   ├── test_estimator.py  # ✅ 実装済み
│   │   ├── test_options.py    # TODO
│   │   ├── test_runspec.py    # TODO
│   │   └── test_utils.py      # TODO
│   │
│   ├── sldataset/       # sldatasetモジュールのユニットテスト
│   │   ├── test_array_loader.py      # TODO
│   │   ├── test_dataset_core.py      # TODO
│   │   ├── test_dataset_item.py      # TODO
│   │   ├── test_dataset_holder.py    # TODO
│   │   └── test_utils.py             # TODO
│   │
│   └── test_convsize.py  # ✅ 実装済み (要拡張)
│
├── integration/         # 統合テスト
│   └── README.md        # 統合テスト実装予定
│
└── build/               # Docker/ビルド環境用テスト
    └── test_pytorch_cuda.py
```

## テストの種類

### 1. Import Tests (`tests/import/`)
モジュールとクラスの基本的なインポート可能性を検証します。

**目的**:
- 循環インポートの検出
- 必須依存関係の確認
- オプション依存関係の適切なハンドリング

**実行方法**:
```powershell
uv run pytest tests/import/ -v
```

### 2. Unit Tests (`tests/unit/`)
個別の関数やクラスの動作を検証します。

**目的**:
- 各モジュールの独立した機能テスト
- エッジケースとエラーハンドリング
- 高いコードカバレッジ（目標: 80%以上）

**実行方法**:
```powershell
# 全ユニットテスト実行
uv run pytest tests/unit/ -v

# 特定モジュールのみ
uv run pytest tests/unit/lmpipe/ -v
uv run pytest tests/unit/sldataset/ -v
```

### 3. Integration Tests (`tests/integration/`)
複数のコンポーネントを組み合わせた動作を検証します。

**目的**:
- エンドツーエンドのワークフロー検証
- コンポーネント間の相互作用テスト
- 実際の使用シナリオの検証

**実行方法**:
```powershell
uv run pytest tests/integration/ -v
```

## テスト実行

### 全テスト実行
```powershell
uv run pytest tests/ -v
```

### カバレッジ測定
```powershell
# ターミナル出力
uv run pytest --cov=cslrtools2 --cov-report=term-missing tests/

# HTMLレポート生成
uv run pytest --cov=cslrtools2 --cov-report=html tests/
# レポート: htmlcov/index.html
```

### 特定のテストのみ実行
```powershell
# 特定のファイル
uv run pytest tests/unit/lmpipe/test_estimator.py -v

# 特定のクラス
uv run pytest tests/unit/lmpipe/test_estimator.py::TestEstimatorABC -v

# 特定のテスト
uv run pytest tests/unit/lmpipe/test_estimator.py::TestEstimatorABC::test_estimator_abstract -v
```

## カバレッジ目標

### 現在の状態 (2025-11-15)
- **総合カバレッジ**: 41%
- **目標**: 80%以上

### モジュール別目標

| モジュール | 現在 | 目標 | 優先度 |
|-----------|------|------|--------|
| `lmpipe/interface/runner.py` | 0% | 80% | 🔴 最高 |
| `lmpipe/interface/executor.py` | 0% | 80% | 🔴 最高 |
| `sldataset/dataset/core.py` | 43% | 85% | 🔴 最高 |
| `sldataset/dataset/item.py` | 35% | 85% | 🟡 高 |
| `lmpipe/collector/*` | 27-51% | 85% | 🟡 高 |
| `sldataset/array_loader.py` | 58% | 90% | 🟡 高 |
| `lmpipe/utils.py` | 31% | 85% | 🟢 中 |
| `convsize.py` | 51% | 85% | 🟢 中 |
| `lmpipe/estimator.py` | 68% | 85% | 🟢 中 |
| `lmpipe/options.py` | 96% | 100% | 🔵 低 |

## 実装進捗

### ✅ 完了
- [x] テストディレクトリ構造の再編成
- [x] `tests/unit/lmpipe/test_estimator.py` (6テスト)
- [x] `tests/unit/test_convsize.py` (11テスト)
- [x] `tests/import/` 配下のインポートテスト (16テスト)

### 🚧 TODO (優先度順)

#### Priority 1: Critical Modules
1. [ ] `tests/unit/lmpipe/collector/test_csv_lmsc.py`
2. [ ] `tests/unit/lmpipe/collector/test_npy_lmsc.py`
3. [ ] `tests/unit/lmpipe/collector/test_npz_lmsc.py`
4. [ ] `tests/unit/lmpipe/collector/test_zarr_lmsc.py`
5. [ ] `tests/unit/sldataset/test_dataset_core.py`
6. [ ] `tests/unit/sldataset/test_dataset_item.py`

#### Priority 2: High Impact
7. [ ] `tests/unit/sldataset/test_array_loader.py`
8. [ ] `tests/unit/lmpipe/test_utils.py`
9. [ ] `tests/unit/lmpipe/collector/test_safetensors_lmsc.py`
10. [ ] `tests/unit/lmpipe/collector/test_torch_lmsc.py`

#### Priority 3: Coverage Improvement
11. [ ] `tests/unit/lmpipe/test_options.py`
12. [ ] `tests/unit/lmpipe/test_runspec.py`
13. [ ] `tests/unit/sldataset/test_dataset_holder.py`
14. [ ] `tests/unit/sldataset/test_utils.py`
15. [ ] `tests/unit/test_convsize.py` の拡張

#### Priority 4: Integration
16. [ ] `tests/integration/test_lmpipe_e2e.py`
17. [ ] `tests/integration/test_sldataset_workflow.py`

## 開発ガイドライン

### テスト作成時のベストプラクティス

1. **Fixtureの活用**
```python
@pytest.fixture
def temp_path(tmp_path: Path) -> Path:
    """一時ディレクトリを提供"""
    return tmp_path / "test_output"
```

2. **パラメータ化テスト**
```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert input * 2 == expected
```

3. **エラーケースのテスト**
```python
def test_invalid_input_raises():
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_should_raise()
```

4. **MediaPipeのスキップ**
```python
@pytest.mark.skipif(not HAS_MEDIAPIPE, reason="MediaPipe not installed")
def test_with_mediapipe():
    ...
```

## 継続的改善

1. 新機能追加時は、対応するテストも同時に作成
2. バグ修正時は、回帰テストを追加
3. 週次でカバレッジレポートを確認
4. カバレッジ80%以上を維持

## 参考資料

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
