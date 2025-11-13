# PyTorch+CUDA Docker環境構築戦略

## プロジェクト要件 (2025年11月14日更新)

### テスト対象環境
**CUDA+PyTorchの組み合わせ:**
- **CUDA 12.8 + PyTorch 2.9.0** (メイン開発環境)
- **CUDA 12.6 + PyTorch 2.9.0** (後方互換性検証)
- **CUDA 13.0 + PyTorch 2.9.0** (最新CUDA検証 - 公式サポート済み)
- **CPU版 + PyTorch 2.9.0** (CI/CD用)
- **PyTorch 2.3.0** (下方互換性検証 - 最小サポートバージョン)

### プロジェクトの性質
- **用途**: 研究ツール（Continuous Sign Language Recognition）
- **依存関係の特殊性**:
  - `torchvision==0.24.0`: Windows用パッケージハッシュ未提供のため一時的にpin
  - `mediapipe>=0.10.14`: 下方互換性の検証が必要
- **開発フロー**:
  - Dev Container: Linux互換性検証、プロジェクト更新作業時
  - Dockerイメージ: テスト環境として使用

### 現在のプロジェクト設定
```toml
# pyproject.toml
torch>=2.9.0         # 2.3.0まで下方互換性を持たせる
torchvision==0.24.0  # 一時的なpin（0.24.1のWindows問題回避）

[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu128"
```

## 調査結果サマリー

### ベースイメージのユーザー調査

**nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04の既存ユーザー:**
- `ubuntu` ユーザーが存在 (UID=1000, GID=1000)
- sudoグループに所属（video, adm等の権限も保有）
- ホームディレクトリ: `/home/ubuntu` (作成済み)
- **sudo権限: 未設定** (sudoers.d/にファイルなし)

**結論: 新規ユーザー作成を推奨 ❌ → 既存ubuntuユーザー活用を推奨 ✅**

**理由:**
1. VS Code Dev Containerは任意のユーザー名に対応
2. 既存ubuntuユーザーはUID 1000（一般的なデフォルト）
3. ホームディレクトリ作成済みで設定ファイル(.bashrc等)も完備
4. sudo権限追加のみで使用可能
5. イメージサイズ削減（ユーザー作成処理不要）

### PyTorch CUDAサポート状況
- **CUDA 12.6**: `--index-url https://download.pytorch.org/whl/cu126`
- **CUDA 12.8**: `--index-url https://download.pytorch.org/whl/cu128`
- **CUDA 13.0**: `--index-url https://download.pytorch.org/whl/cu130` ✅ PyTorch 2.9.0で公式サポート
- **CPU版**: `--index-url https://download.pytorch.org/whl/cpu`

## マルチ環境テスト戦略

### docker-compose.yaml構成

プロジェクトルートに以下の5つのテスト環境を定義:

1. **pytorch-cu128** - CUDA 12.8（メイン開発環境）
2. **pytorch-cu126** - CUDA 12.6（後方互換性）
3. **pytorch-cu130** - CUDA 13.0（最新CUDA）
4. **pytorch-cpu** - CPU版（CI/CD用）
5. **pytorch-2.3-cu128** - PyTorch 2.3.0（最小バージョン検証）

各環境は独立した仮想環境（venv）を持ち、同時に複数のPyTorchバージョンをテスト可能。

### 使用方法

```bash
# 特定環境でビルド
docker compose build pytorch-cu128

# 特定環境でコマンド実行
docker compose run --rm pytorch-cu128 uv run python test_pytorch_cuda.py

# MediaPipe互換性テスト
docker compose run --rm pytorch-cu128 uv run python test_pytorch_cuda.py --test-mediapipe

# バッチ処理例（複数動画を並列処理）
docker compose run --rm pytorch-cu128 uv run lmpipe mediapipe.holistic videos/

# CPU版でユニットテスト
docker compose run --rm pytorch-cpu uv run pytest

# 全環境でテスト
for env in pytorch-cu128 pytorch-cu126 pytorch-cu130 pytorch-cpu; do
  echo "Testing $env..."
  docker compose run --rm $env uv run python test_pytorch_cuda.py
done
```

### 各Dockerfileの役割

| ファイル | ベースイメージ | PyTorch Index | 用途 |
|---------|--------------|--------------|------|
| `.devcontainer/Dockerfile` | `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` | cu128 | メイン開発（Dev Container兼用） |
| `Dockerfile.cu126` | `nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04` | cu126 | CUDA 12.6後方互換性検証 |
| `Dockerfile.cu130` | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` | cu130 | CUDA 13.0最新検証 |
| `Dockerfile.cpu` | `ubuntu:24.04` | cpu | CPU専用CI/CD |
| `Dockerfile.torch23` | `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` | cu128 | PyTorch 2.3.0下方互換性 |

**注意**: `pytorch-2.3-cu128`環境では`pyproject.toml`の`torch>=2.9.0`を一時的に`torch>=2.3.0,<2.4.0`に変更してテストする必要がある。

## Dev Container vs docker-compose.yaml

### Dev Container（.devcontainer/）
- **用途**: VS Codeでの開発、Linux互換性検証、プロジェクト更新作業
- **起動方法**: VS Codeの「Reopen in Container」
- **環境**: CUDA 12.8のみ（pyproject.tomlの設定に一致）
- **永続性**: 開発中は常時起動

### docker-compose.yaml
- **用途**: マルチバージョンテスト、バッチ処理、CI/CD
- **起動方法**: `docker compose run` コマンド
- **環境**: 5つの異なる環境を切り替え可能
- **永続性**: 一時的なテスト実行（`--rm`フラグ推奨）

## 推奨Docker戦略

### メイン開発環境（既存ubuntuユーザー活用版）

`.devcontainer/Dockerfile` - CUDA 12.8環境:
```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# 基本パッケージ
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# uvインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 既存ubuntuユーザーにsudo権限を付与
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

USER ubuntu
WORKDIR /workspace
```

**利点:**
- pyproject.tomlのcu128設定と完全一致
- cuDNN含む（最新版）
- Ubuntu 24.04 LTS対応
- 既存ubuntuユーザー活用でシンプル
- UID 1000でホスト環境との親和性が高い



## テストスクリプト

### test_pytorch_cuda.py

環境検証用スクリプト（プロジェクトルートに配置）:

**基本使用:**
```bash
# 環境情報とGPU演算テスト
python test_pytorch_cuda.py

# MediaPipe互換性も検証
python test_pytorch_cuda.py --test-mediapipe
```

**主な機能:**
- PyTorchバージョン、CUDAバージョン、cuDNNバージョンの表示
- GPU情報（デバイス名、メモリ、Compute Capability）
- GPUテンソル演算テスト（1000x1000行列演算）
- MediaPipe初期化テスト（オプション）

**docker-compose.yamlでの使用例:**
```bash
# CUDA 12.8環境でテスト
docker compose run --rm pytorch-cu128 uv run python test_pytorch_cuda.py --test-mediapipe

# 全環境で検証
for env in pytorch-cu128 pytorch-cu126 pytorch-cu130 pytorch-cpu pytorch-2.3-cu128; do
  echo "=== Testing $env ==="
  docker compose run --rm $env uv run python test_pytorch_cuda.py
done
```

## 実行手順

### 1. ローカルでのテスト（Dev Container）

```bash
# VS Codeで開く
code .

# Dev Containerで再起動（コマンドパレット）
# > Dev Containers: Reopen in Container

# コンテナ内でテスト実行
uv run python test_pytorch_cuda.py --test-mediapipe
```

### 2. マルチ環境テスト（docker-compose）

```bash
# 特定環境のビルド
docker compose build pytorch-cu128

# 環境確認
docker compose run --rm pytorch-cu128 uv run python test_pytorch_cuda.py

# MediaPipe互換性確認
docker compose run --rm pytorch-cu128 uv run python test_pytorch_cuda.py --test-mediapipe

# 全CUDA環境でテスト
for cuda_env in pytorch-cu128 pytorch-cu126 pytorch-cu130; do
  docker compose run --rm $cuda_env uv run python test_pytorch_cuda.py
done

# CPU版テスト
docker compose run --rm pytorch-cpu uv run python test_pytorch_cuda.py

# PyTorch 2.3.0テスト（pyproject.toml一時変更が必要）
# torch>=2.9.0 -> torch>=2.3.0,<2.4.0
docker compose run --rm pytorch-2.3-cu128 uv run python test_pytorch_cuda.py
```

### 3. バッチ処理例

```bash
# 大量の動画からランドマーク抽出
docker compose run --rm pytorch-cu128 uv run lmpipe mediapipe.holistic \
  /path/to/videos/ \
  -o landmarks.zarr \
  --workers 4

# CPU環境で単体テスト
docker compose run --rm pytorch-cpu uv run pytest tests/
```

## .gitignore設定

テスト用ファイルの管理:

```gitignore
# Docker関連（ビルド済みイメージ、オーバーライド設定）
docker-compose.override.yaml
.docker/

# テストスクリプト（一時的な検証用）
test_pytorch_cuda.py

# その他のテスト用スクリプト
test_*.py
analyze_*.py
check_*.py
```

**注意**: 
- `docker-compose.yaml`と各`Dockerfile.*`は**トラッキング対象**（リポジトリに含める）
- `test_pytorch_cuda.py`は一時的な検証用スクリプトのため`.gitignore`に追加

## 推奨アクション

### 完了済み ✅
- `.devcontainer/Dockerfile`作成（CUDA 12.8、ubuntuユーザー活用）
- `.devcontainer/devcontainer.json`設定
- `docker-compose.yaml`作成（5つのテスト環境定義）
- 各環境用Dockerfile作成（`Dockerfile.cu126`, `Dockerfile.cu130`, `Dockerfile.cpu`, `Dockerfile.torch23`）
- `test_pytorch_cuda.py`実装（環境検証スクリプト）

### 次のステップ
1. **短期**: 各環境でのビルドとテスト実行
2. **中期**: MediaPipe下方互換性の検証（`mediapipe>=0.10.14`の下限確認）
3. **中期**: CI/CDでCPU版自動テスト実装
4. **長期**: PyTorch 2.3.0互換性の継続的検証

## 参考リンク

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA Images](https://hub.docker.com/r/nvidia/cuda/tags)
- [cuDNN Release Notes](https://docs.nvidia.com/deeplearning/cudnn/release-notes/)
- [uv Documentation](https://docs.astral.sh/uv/)
