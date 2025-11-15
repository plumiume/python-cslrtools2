# PyTorch環境テスト用Docker設定

このディレクトリには、複数のPyTorch+CUDA環境でプロジェクトをテストするためのDocker設定が含まれています。

## 構成

- **Dockerfile**: マルチステージビルドで5つのテスト環境を定義
- **docker-compose.yaml**: 各環境を簡単に起動・テストするための設定
- **test_pytorch_cuda.py**: 環境検証スクリプト
- **DOCKER_STRATEGY.md**: 詳細な戦略ドキュメント

## テスト環境

| サービス名 | CUDA | PyTorch Index | 用途 |
|-----------|------|--------------|------|
| `pytorch-cu128` | 12.8.0 | cu128 | メイン開発環境 |
| `pytorch-cu126` | 12.6.0 | cu126 | 後方互換性検証 |
| `pytorch-cu130` | 13.0.2 | cu130 | 最新CUDA検証 |
| `pytorch-cpu` | なし | cpu | CI/CD用 |
| `pytorch-2.3-cu128` | 12.8.0 | cu128 | PyTorch 2.3.0検証 |

## 使用方法

### 1. イメージのビルド

```bash
# tests/buildディレクトリに移動
cd tests/build

# すべての環境をビルド
docker compose build

# 特定の環境のみビルド
docker compose build pytorch-cu128
```

### 2. テストの実行

```bash
# 特定環境でコマンド実行
docker compose run --rm pytorch-cu128 uv run python tests/build/test_pytorch_cuda.py

# MediaPipe互換性テスト
docker compose run --rm pytorch-cu128 uv run python tests/build/test_pytorch_cuda.py --test-mediapipe

# 全環境でテスト（PowerShell）
foreach ($env in @('pytorch-cu128', 'pytorch-cu126', 'pytorch-cu130', 'pytorch-cpu')) {
    Write-Host "`n=== Testing $env ===" -ForegroundColor Cyan
    docker compose run --rm $env uv run python tests/build/test_pytorch_cuda.py
}

# 全環境でテスト（Bash）
for env in pytorch-cu128 pytorch-cu126 pytorch-cu130 pytorch-cpu; do
  echo "=== Testing $env ==="
  docker compose run --rm $env uv run python tests/build/test_pytorch_cuda.py
done
```

### 3. バッチ処理例

```bash
# 大量の動画からランドマーク抽出
docker compose run --rm pytorch-cu128 uv run lmpipe mediapipe.holistic \
  videos/ \
  -o landmarks.zarr \
  --workers 4

# CPU環境で単体テスト
docker compose run --rm pytorch-cpu uv run pytest
```

## UVキャッシュ共有

すべての環境が共有の`uv-cache`ボリュームを使用し、パッケージダウンロードを最小化：

- **初回実行**: 約1分（パッケージダウンロード）
- **2回目以降**: 約0.4〜0.5秒（キャッシュのみ使用）
- **キャッシュサイズ**: 約6.9GB

## マルチステージビルドの利点

1. **単一ファイル管理**: 1つのDockerfileで全環境を管理
2. **共通レイヤーのキャッシュ**: ビルド時間の短縮
3. **保守性向上**: 変更が一箇所に集約

## 注意事項

- 仮想環境（`.venv`）は毎回クリーンビルドされます
- UVキャッシュにより高速インストールが実現されています
- `pytorch-2.3-cu128`環境でPyTorch 2.3.0をテストする場合は、`pyproject.toml`の`torch>=2.9.0`制約を一時的に変更してください

## トラブルシューティング

### 外部ネットワークエラー

`pypi-cache-network`が存在しない場合：

```bash
# ネットワークを作成
docker network create pypi-cache-network

# または、docker-compose.yamlからnetworksセクションを削除
```

### キャッシュのクリア

```bash
# UVキャッシュボリュームを削除
docker volume rm tests-build_uv-cache

# または、docker composeのvolumesセクションを一時的にコメントアウト
```
