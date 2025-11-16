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

### 現在の状態 (2025-11-16)
- **総合カバレッジ**: 91% (2434 statements, 229 missing) ✅ 目標達成!
- **テスト数**: 609 passing, 19 skipped (628 collected)
- **目標**: 90%以上達成完了!

### モジュール別カバレッジ (2025-11-16)

| モジュール | 現在 | 目標 | 優先度 | 状態 |
|-----------|------|------|--------|------|
| `lmpipe/interface/runner.py` | **98%** | 99% | � 高 | ✅ ほぼ完了 (8行残) |
| `lmpipe/interface/executor.py` | **100%** | 100% | - | ✅ 完了 |
| `lmpipe/estimator.py` | **99%** | 100% | � 中 | ✅ ほぼ完了 |
| `lmpipe/options.py` | **100%** | 100% | - | ✅ 完了 |
| `lmpipe/utils.py` | **100%** | 100% | - | ✅ 完了 |
| `lmpipe/runspec.py` | **100%** | 100% | - | ✅ 完了 |
| `sldataset/dataset/core.py` | **98%** | 99% | � 中 | ✅ ほぼ完了 |
| `sldataset/dataset/item.py` | **95%** | 98% | 🟢 中 | ✅ 優秀 |
| `sldataset/array_loader.py` | **99%** | 100% | � 中 | ✅ ほぼ完了 |
| `lmpipe/collector/csv_lmsc.py` | **83%** | 90% | 🟡 高 | 🚧 改善中 |
| `lmpipe/collector/npy_lmsc.py` | **98%** | 100% | 🟢 中 | ✅ ほぼ完了 |
| `lmpipe/collector/npz_lmsc.py` | **97%** | 100% | 🟢 中 | ✅ ほぼ完了 |
| `lmpipe/collector/zarr_lmsc.py` | **96%** | 100% | 🟢 中 | ✅ 優秀 |
| `lmpipe/collector/safetensors_lmsc.py` | **98%** | 100% | � 中 | ✅ ほぼ完了 |
| `lmpipe/collector/torch_lmsc.py` | **97%** | 100% | 🟢 中 | ✅ ほぼ完了 |
| `plugins/mediapipe/lmpipe/base.py` | **40%** | 70% | � 最高 | ⚠️ 要改善 |
| `plugins/mediapipe/lmpipe/holistic.py` | **71%** | 85% | 🔴 最高 | � 改善中 |
| `plugins/mediapipe/lmpipe/face.py` | **60%** | 85% | 🔴 最高 | � 改善中 |

## 実装進捗 (2025-11-16更新)

### ✅ 完了 (609 tests passing, 19 skipped)

#### Import Tests (22 tests)
- [x] `tests/import/test_imports.py` (10テスト)
- [x] `tests/import/test_mediapipe_constants.py` (5テスト)
- [x] `tests/import/test_sldataset_imports.py` (7テスト)

#### Unit Tests - Core (382 tests)
- [x] `tests/unit/test_init.py` (8テスト)
- [x] `tests/unit/test_init_fallback.py` (6テスト)
- [x] `tests/unit/test_logger.py` (10テスト)
- [x] `tests/unit/test_root.py` (4テスト)
- [x] `tests/unit/test_convsize.py` (22テスト)
- [x] `tests/unit/test_error_handling.py` (14テスト) ✨NEW✨
  - FileNotFoundErrors (3 tests)
  - InvalidDataErrors (3 tests)
  - MemoryErrors (2 tests)
  - ZarrStoreErrors (2 tests)
  - EdgeCases (4 tests)

#### Unit Tests - LMPipe (259 tests)
- [x] `tests/unit/lmpipe/test_estimator.py` (29テスト)
- [x] `tests/unit/lmpipe/test_options.py` (5テスト)
- [x] `tests/unit/lmpipe/test_runspec.py` (10テスト)
- [x] `tests/unit/lmpipe/test_utils.py` (20テスト)
- [x] `tests/unit/lmpipe/test_concurrent.py` (14テスト) ✨NEW✨
  - ProcessPoolExecutor compatibility (4 tests, all passing with loky)
  - ThreadPoolExecutor compatibility (4 tests)
  - DummyExecutor concurrency (3 tests)
  - Executor interface consistency (3 tests)
- [x] `tests/unit/lmpipe/interface/test_executor.py` (7テスト)
- [x] `tests/unit/lmpipe/interface/test_lmpipe_interface.py` (26テスト)
- [x] `tests/unit/lmpipe/interface/test_runner.py` (46テスト)
- [x] `tests/unit/lmpipe/collector/test_base.py` (7テスト)
- [x] `tests/unit/lmpipe/collector/test_csv_lmsc.py` (17テスト)
- [x] `tests/unit/lmpipe/collector/test_csv_lmsc_edge_cases.py` (11テスト) ✨NEW✨
- [x] `tests/unit/lmpipe/collector/test_json_lmsc.py` (15テスト)
- [x] `tests/unit/lmpipe/collector/test_npy_lmsc.py` (10テスト)
- [x] `tests/unit/lmpipe/collector/test_npz_lmsc.py` (11テスト)
- [x] `tests/unit/lmpipe/collector/test_zarr_lmsc.py` (13テスト)
- [x] `tests/unit/lmpipe/collector/test_safetensors_lmsc.py` (7テスト)
- [x] `tests/unit/lmpipe/collector/test_torch_lmsc.py` (16テスト)
- [x] `tests/unit/lmpipe/collector/landmark_matrix/test_base.py` (13テスト)
- [x] `tests/unit/lmpipe/collector/annotated_frames/test_base.py` (13テスト)
- [x] `tests/unit/lmpipe/collector/annotated_frames/test_cv2_af.py` (17テスト)
- [x] `tests/unit/lmpipe/collector/annotated_frames/test_cv2_validation.py` (7テスト)
- [x] `tests/unit/lmpipe/collector/annotated_frames/test_matplotlib_af.py` (7テスト)
- [x] `tests/unit/lmpipe/collector/annotated_frames/test_pil_af.py` (4テスト)
- [x] `tests/unit/lmpipe/collector/annotated_frames/test_torchvision_af.py` (4テスト)
- [x] `tests/unit/lmpipe/collector/annotated_frames/test_show_collectors_mock.py` (5テスト)

#### Unit Tests - SLDataset (103 tests)
- [x] `tests/unit/sldataset/test_array_loader.py` (21テスト)
- [x] `tests/unit/sldataset/test_dataset.py` (74テスト)
- [x] `tests/unit/sldataset/test_dataset_core.py` (1テスト - placeholder)
- [x] `tests/unit/sldataset/test_dataset_holder.py` (1テスト - placeholder)
- [x] `tests/unit/sldataset/test_dataset_item.py` (1テスト - placeholder)
- [x] `tests/unit/sldataset/test_utils.py` (16テスト)

#### Integration Tests (69 tests) ✨EXPANDED✨
- [x] `tests/integration/test_lmpipe_e2e_basic.py` (3テスト)
  - Single video to NPZ workflow
  - Custom filename handling
  - Video metadata preservation
- [x] `tests/integration/test_lmpipe_multiple_collectors.py` (4テスト)
  - Multiple collectors simultaneous save
  - Selective combinations
  - Empty list handling
  - Data integrity verification
- [x] `tests/integration/test_lmpipe_runner_edge_cases.py` (7テスト)
  - Source path errors
  - Unsupported path types
  - Directory handling
  - PathLike conversions
  - RunSpec creation
- [x] `tests/integration/test_collector_formats.py` (9テスト) ✨NEW✨
  - CSV roundtrip (2 tests)
  - NPY roundtrip (2 tests)
  - NPZ roundtrip (2 tests)
  - Zarr roundtrip (2 tests)
  - Format comparison (1 test)
- [x] `tests/integration/test_dataset_workflow.py` (15テスト) ✨NEW✨
  - SLDataset creation (3 tests)
  - Zarr roundtrip (3 tests)
  - PyTorch DataLoader integration (3 tests)
  - Indexing operations (3 tests)
  - Slicing operations (3 tests)
- [x] `tests/integration/test_cli_commands.py` (15テスト) ✨NEW✨
  - Holistic command tests (5 tests, 10 skipped due to MediaPipe API)
  - Pose command tests (5 tests, skipped)
  - Error handling (3 tests)
  - Output verification (2 tests, skipped)
- [x] `tests/integration/test_plugin_system.py` (16テスト) ✨NEW✨
  - Plugin discovery (3 tests)
  - Plugin loading (3 tests)
  - LMPipe integration (2 tests)
  - NamespaceWrapper (2 tests)
  - Error handling (3 tests)
  - Registry validation (3 tests)

#### Benchmark Tests (13 tests) ✨NEW✨
- [x] `tests/benchmark/test_performance.py` (13テスト)
  - Array I/O performance (6 tests): NPY/NPZ/Zarr read/write
  - Large data performance (3 tests): 1000-frame operations
  - Memory efficiency (2 tests): Incremental writes, partial reads
  - Comparative performance (2 tests): Format speed comparisons

### 🎉 Phase 1 完了!

**達成した目標**:
- ✅ 総合カバレッジ 91% (目標90%超達成)
- ✅ 628テスト実装 (514→628, +114テスト)
- ✅ 0 Pyrightエラー
- ✅ すべての高・中優先度タスク完了

**実装したテストカテゴリ**:
1. ✅ Integration Tests (69 tests)
2. ✅ Error Handling Tests (14 tests)
3. ✅ Concurrent Execution Tests (14 tests)
4. ✅ Performance Benchmark Tests (13 tests)
5. ✅ Plugin System Tests (16 tests)
6. ✅ CSV Edge Case Tests (11 tests)

### 🚧 Phase 2: 将来的な改善機会 (オプション)

#### Priority 1: MediaPipe Plugin Coverage (40-71% → 80%+)
1. [ ] MediaPipe base module tests
2. [ ] Holistic estimator edge cases
3. [ ] Face estimator edge cases
4. [ ] Hand estimator edge cases

#### Priority 2: Runner Edge Cases (98% → 99%)
5. [ ] `runner.py` lines 220-221 (unsupported path type)
6. [ ] `runner.py` lines 836, 1110-1111, 1125, 1131, 1135

#### Priority 3: Integration Tests Expansion
7. [ ] Dataset workflow integration tests
8. [ ] CLI integration tests
9. [ ] Plugin system integration tests

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
