# Integration Tests

このディレクトリには、複数のコンポーネントを組み合わせた統合テストを配置します。

## ⚠️ 実行上の注意

### Transform Tests のメモリ問題

`test_transform_dynamic.py` は torchvision を使用するため、並列実行でメモリ不足エラーが発生する場合があります:

```powershell
# ❌ 並列実行 (メモリ不足の可能性)
uv run pytest tests/integration/test_transform_dynamic.py -n auto

# ✅ 推奨: 並列数を制限
uv run pytest tests/integration/test_transform_dynamic.py -n 2

# ✅ または: 並列実行を無効化
uv run pytest tests/integration/test_transform_dynamic.py
```

**エラー例**:
- `OSError: [WinError 1455] ページング ファイルが小さすぎる`
- `ERROR gw* - Different tests were collected between workers`

**原因**: 各ワーカープロセスが独立に torchvision をインポートし、メモリを大量消費

## 実装済みテスト (79 tests)

### ✅ 1. エンドツーエンド基本フロー (`test_lmpipe_e2e_basic.py`)
**3 tests** - LMPipeの基本的なE2Eワークフローを検証

- `test_single_video_to_npz` - ビデオ入力 → ランドマーク抽出 → NPZ保存
- `test_single_video_to_npz_with_custom_filename` - カスタムファイル名での保存
- `test_video_metadata_preservation` - ビデオメタデータの保持確認

**検証内容**:
- MockEstimatorを使用したエンドツーエンド処理
- NPZファイル生成とキー構造の検証
- ビデオ情報（幅、高さ、FPS）の正確な保存

### ✅ 2. 複数Collector同時使用 (`test_lmpipe_multiple_collectors.py`)
**4 tests** - 複数フォーマットへの同時保存を検証

- `test_multiple_collectors_csv_npy_npz` - CSV + NPY + NPZ 3形式同時保存
- `test_multiple_collectors_selective_combination` - 選択的な組み合わせ
- `test_multiple_collectors_with_empty_list` - 空リストでの動作確認
- `test_multiple_collectors_data_integrity` - データ整合性の検証

**検証内容**:
- 複数CollectorFactoryの正常動作
- 全フォーマットでのファイル生成確認
- データ一致性の検証（CSV ≈ NPY ≈ NPZ）

### ✅ 3. Runnerエッジケース (`test_lmpipe_runner_edge_cases.py`)
**7 tests** - Runnerの境界条件とエラーハンドリングを検証

- `test_runner_source_path_not_exist` - 存在しないパスのエラー
- `test_runner_unsupported_path_type_socket` - サポート外パスタイプ
- `test_runner_run_with_directory_empty` - 空ディレクトリの処理
- `test_runner_output_directory_creation` - 出力ディレクトリ自動作成
- `test_runner_with_pathlike_strings` - PathLike文字列の処理
- `test_runner_runspec_creation` - RunSpec生成の検証
- `test_runner_single_vs_batch_detection` - 単一/バッチ自動検出

**検証内容**:
- エラーメッセージの正確性
- 境界条件での動作保証
- パス処理の堅牢性

### ✅ 4. Collectorフォーマット検証 (`test_collector_formats.py`)
**9 tests** - 各Collectorフォーマットの保存/読み込みを検証

**ラウンドトリップテスト** (4 tests):
- `test_csv_roundtrip` - CSV保存/読み込み (float32精度考慮)
- `test_npy_roundtrip` - NPY per-key format検証
- `test_npz_roundtrip` - NPZ container format検証
- `test_zarr_roundtrip` - Zarr階層構造検証

**フォーマット比較** (2 tests):
- `test_npz_vs_npy_consistency` - NPZ/NPY間のデータ一致性
- `test_collector_format_metadata` - Collectorメタデータ属性

**エッジケース** (3 tests):
- `test_empty_landmarks` - 空ランドマーク処理
- `test_single_frame_zarr` - 単一フレームZarr
- `test_csv_with_special_characters_in_keys` - 特殊文字キー処理

### ✅ 5. Dataset Workflow (`test_dataset_workflow.py`)
**15 tests** - SLDatasetの作成からPyTorch統合までを検証

**データセット作成** (3 tests):
- `test_create_empty_dataset` - 空データセット作成
- `test_create_dataset_with_metadata` - メタデータ付きデータセット
- `test_add_items_to_dataset` - アイテム追加

**Zarrラウンドトリップ** (2 tests):
- `test_save_and_load_dataset` - 完全な保存/読み込みサイクル
- `test_dataset_persistence_with_connections` - コネクションと permanence

**PyTorch DataLoader統合** (3 tests):
- `test_dataloader_basic_iteration` - 基本的な反復処理 (カスタムcollate_fn使用)
- `test_dataloader_with_shuffle` - シャッフル機能
- `test_dataloader_batch_size_one` - batch_size=1

**インデックス/スライス** (3 tests):
- `test_positive_indexing` - 正のインデックス
- `test_negative_indexing` - 負のインデックス
- `test_out_of_bounds_indexing` - 範囲外エラー

**エッジケース** (4 tests):
- `test_dataset_with_single_item` - 単一アイテム
- `test_dataset_with_variable_frame_counts` - 可変フレーム数
- `test_dataset_with_multiple_landmark_keys` - 複数ランドマークキー
- `test_dataset_save_load_preserves_structure` - 構造の完全保存

### ✅ 6. CLI Commands (`test_cli_commands.py`)
**15 tests** - コマンドラインインターフェースを検証

**基本実行** (3 tests):
- `test_cli_help_command` - --help フラグ
- `test_cli_version_command` - --version フラグ
- `test_cli_no_arguments` - 引数なし実行

**Holistic Estimator** (3 tests):
- `test_holistic_mediapipe_basic` - 基本的なholistic実行
- `test_holistic_with_csv_collector` - CSV出力
- `test_holistic_with_multiple_collectors` - 複数フォーマット同時出力

**Pose Estimator** (2 tests):
- `test_pose_mediapipe_basic` - 基本的なpose実行
- `test_pose_with_model_complexity` - モデル複雑度オプション

**エラーハンドリング** (3 tests):
- `test_cli_nonexistent_input_file` - 存在しないファイル
- `test_cli_invalid_output_directory` - 無効な出力ディレクトリ
- `test_cli_missing_estimator` - estimator未指定

**出力検証** (2 tests):
- `test_npz_output_structure` - NPZ構造検証
- `test_csv_output_format` - CSVフォーマット検証

**ログ出力** (2 tests):
- `test_cli_with_log_file` - ログファイル出力
- `test_cli_verbose_output` - 詳細出力

**注意**: CLI tests中、一部のテスト（`test_cli_no_arguments`等）が無限待機する問題が確認されています。

### ✅ 7. Transform Dynamic (`test_transform_dynamic.py`)
**26 tests** - Data Augmentation Transformsの統合テストを検証

**UniformSpeedChange** (14 tests):
- `test_uniform_speed_change_reduces_frame_count_when_scale_greater_than_one` - スピードアップ（フレーム削減）
- `test_uniform_speed_change_increases_frame_count_when_scale_less_than_one` - スローダウン（フレーム増加）
- `test_uniform_speed_change_preserves_spatial_dimensions` - 空間次元の保持
- `test_uniform_speed_change_applies_to_multiple_video_keys` - 複数ビデオキー対応
- `test_uniform_speed_change_applies_to_multiple_landmark_keys` - 複数ランドマークキー対応
- Parameter validation tests (3 tests) - 負の値、ゼロ、min > max
- Reproducibility tests (2 tests) - 同一seed、異なるseed
- `test_uniform_speed_change_handles_single_frame_video` - 単一フレーム処理
- `test_uniform_speed_change_with_scale_exactly_one` - scale=1.0（変更なし）
- `test_uniform_speed_change_works_with_various_spatial_sizes` - 様々な解像度対応

**RandomResizePaddingCrop** (10 tests):
- `test_random_resize_padding_crop_preserves_tensor_shape` - テンソル形状不変性
- `test_random_resize_padding_crop_scale_greater_than_one_crops` - Zoom in（クロップ）
- `test_random_resize_padding_crop_scale_less_than_one_pads` - Zoom out（パディング）
- `test_random_resize_padding_crop_transforms_landmark_coordinates` - 座標変換
- `test_random_resize_padding_crop_preserves_landmark_other_dimensions` - 他次元の保持
- Parameter validation tests (2 tests) - 負の値、min > max
- `test_random_resize_padding_crop_is_reproducible_with_same_generator` - 再現性
- `test_random_resize_padding_crop_affine_handles_5d_tensors` - 5Dテンソル対応
- `test_random_resize_padding_crop_center_invariant_transformation` - 中心不変性
- `test_random_resize_padding_crop_works_with_various_coordinate_dimensions` - 可変次元対応

**Error Handling** (2 tests):
- `test_transform_raises_key_error_for_missing_video_key` - 存在しないビデオキー
- `test_transform_raises_key_error_for_missing_landmark_key` - 存在しないランドマークキー

**検証内容**:
- 時間軸/空間軸のデータ拡張動作
- パラメータバリデーション
- 再現性の確保
- エッジケース処理

**カバレッジ**: `transform/dynamic.py` - **100%** (67 statements)

**⚠️ メモリ注意**: torchvision使用により並列実行でメモリ不足の可能性あり。`-n 2`推奨。

## インフラストラクチャ

### Fixtures (`conftest.py`)
- `sample_video_stop` - テストビデオパス (tests/data/videos/hand_gesture_stop.mp4)
- `sample_video_man` - テストビデオパス (tests/data/videos/hand_gesture_man.mp4)
- `integration_tmp_path` - 一時出力ディレクトリ
- `skip_if_no_mediapipe` - MediaPipeインストール確認

### Helpers (`helpers.py`)
- `verify_npz_structure()` - NPZファイル構造検証
- `verify_zarr_structure()` - Zarrディレクトリ構造検証
- `load_and_compare_landmarks()` - ランドマーク比較ヘルパー

## 今後の予定テスト

### Priority 2: Plugin System
- [ ] `test_plugin_integration.py`
  - プラグイン動的ロード
  - カスタムEstimatorの登録
  - entry pointの検証

### Priority 3: Error Handling
- [ ] `test_error_handling.py`
  - ファイル未発見エラー
  - 無効なビデオフォーマット
  - 破損したZarrストア
  - メモリ制限エラー

### Priority 4: MediaPipe Real Execution (BLOCKED)
- [ ] `test_mediapipe_pipeline.py`
  - 実際のMediaPipe estimatorでの実行
  - API互換性の検証
  - ⚠️ 現在MediaPipe API問題により保留中

## 実行方法

```powershell
# 統合テストのみ実行
uv run pytest tests/integration/ -v

# 特定のテストファイルのみ
uv run pytest tests/integration/test_lmpipe_e2e_basic.py -v

# Transform tests (メモリ節約のため並列数制限推奨)
uv run pytest tests/integration/test_transform_dynamic.py -n 2

# すべてのテスト実行
uv run pytest tests/ -v

# カバレッジ付き
uv run pytest tests/integration/ --cov=cslrtools2 --cov-report=term-missing

# MediaPipeテストをスキップ
uv run pytest tests/integration/ -v -m "not mediapipe"

# ビデオ必須テストをスキップ
uv run pytest tests/integration/ -v -m "not requires_video"
```

## テスト戦略

### Mock vs Real
- **Unit Tests**: 全てMock（高速、決定性）
- **Integration Tests**: 部分的にReal（実際のファイルI/O、実パス処理）
- **E2E Tests**: 可能な限りReal（実際のビデオ、実estimator）

### データサイズ制限
- テストビデオ: < 5MB
- 生成ファイル: < 10MB per test
- 実行時間: < 10秒 per test (目標)
