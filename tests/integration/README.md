# Integration Tests

このディレクトリには、複数のコンポーネントを組み合わせた統合テストを配置します。

## テスト対象

### 予定されている統合テスト

1. **エンドツーエンドのランドマーク抽出パイプライン**
   - `test_lmpipe_e2e.py`: ビデオ入力 → 推定 → 複数フォーマットで保存
   
2. **データセット作成から利用までの完全なフロー**
   - `test_sldataset_workflow.py`: Dataset作成 → データ追加 → PyTorchで読み込み

3. **プラグインシステムの統合**
   - `test_plugin_integration.py`: プラグイン読み込み → 実行

## 実装予定

これらの統合テストは、ユニットテストのカバレッジが80%を超えた後に実装します。

## 実行方法

```powershell
# 統合テストのみ実行
uv run pytest tests/integration/ -v

# すべてのテスト実行
uv run pytest tests/ -v
```
