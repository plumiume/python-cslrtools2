# PyTorch+CUDA Docker環境構築戦略

## 調査結果サマリー (2025年11月14日)

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

### PyTorch 2.9.0 (最新版) - CUDAサポート
- **CUDA 12.6**: `--index-url https://download.pytorch.org/whl/cu126`
- **CUDA 12.8**: `--index-url https://download.pytorch.org/whl/cu128`
- **CUDA 13.0**: `--index-url https://download.pytorch.org/whl/cu130`
- **CPU版**: `--index-url https://download.pytorch.org/whl/cpu`

### 現在のプロジェクト設定
```toml
# pyproject.toml
torch>=2.9.0
torchvision==0.24.0

[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu128"
```

## 推奨Docker戦略

### 戦略1: CUDA 12.8環境 - 既存ubuntuユーザー活用版 ✅ 推奨

#### Dockerfile
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
- cuDNN 9含む（最新版）
- Ubuntu 24.04 LTS対応
- 既存ubuntuユーザー活用でシンプル
- UID 1000でホスト環境との親和性が高い

#### 戦略1-B: vscodeユーザー作成版（従来方式）

VS Code Dev Container固有の要件がある場合:

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

# vscodeユーザーを作成（UID衝突を避ける）
ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /workspace
```

**利点:**
- VS Code拡張機能の明示的なユーザー名要件に対応
- UID 1001で既存ubuntuユーザー(1000)と衝突回避
- 複数のdev containerプロジェクトで統一可能

**検証コマンド:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
```

### 戦略2: CPU専用テスト環境

#### Dockerfile (CPU)
```dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /workspace
```

#### pyproject.toml (CPU版)
```toml
[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cpu"
```

**利点:**
- GPU不要でテスト可能
- 軽量（CUDAランタイム不要）
- CI/CD環境に最適

### 戦略3: マルチバージョンテスト

#### docker-compose.yml
```yaml
version: '3.8'

services:
  # CUDA 12.8環境
  pytorch-cu128:
    build:
      context: .
      dockerfile: .devcontainer/Dockerfile
      args:
        CUDA_VERSION: "12.8.0"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - UV_PROJECT_ENVIRONMENT=/home/vscode/.venv
      - UV_INDEX_URL=https://download.pytorch.org/whl/cu128
    volumes:
      - .:/workspace:cached
      - venv-cu128:/home/vscode/.venv
    working_dir: /workspace
    command: ["tail", "-f", "/dev/null"]

  # CUDA 12.6環境（互換性テスト用）
  pytorch-cu126:
    build:
      context: .
      dockerfile: .devcontainer/Dockerfile
      args:
        CUDA_VERSION: "12.6.0"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - UV_PROJECT_ENVIRONMENT=/home/vscode/.venv
      - UV_INDEX_URL=https://download.pytorch.org/whl/cu126
    volumes:
      - .:/workspace:cached
      - venv-cu126:/home/vscode/.venv
    working_dir: /workspace
    command: ["tail", "-f", "/dev/null"]

  # CUDA 13.0環境（最新版テスト用）
  pytorch-cu130:
    build:
      context: .
      dockerfile: .devcontainer/Dockerfile
      args:
        CUDA_VERSION: "13.0.2"
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - UV_PROJECT_ENVIRONMENT=/home/vscode/.venv
      - UV_INDEX_URL=https://download.pytorch.org/whl/cu130
    volumes:
      - .:/workspace:cached
      - venv-cu130:/home/vscode/.venv
    working_dir: /workspace
    command: ["tail", "-f", "/dev/null"]

  # CPU専用環境
  pytorch-cpu:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    environment:
      - UV_PROJECT_ENVIRONMENT=/home/vscode/.venv
      - UV_INDEX_URL=https://download.pytorch.org/whl/cpu
    volumes:
      - .:/workspace:cached
      - venv-cpu:/home/vscode/.venv
    working_dir: /workspace
    command: ["tail", "-f", "/dev/null"]

volumes:
  venv-cu128:
  venv-cu126:
  venv-cu130:
  venv-cpu:
```

### 戦略4: 引数付きDockerfile (柔軟性重視)

```dockerfile
ARG CUDA_VERSION=12.8.0
ARG CUDNN_VERSION=9
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /workspace
```

**ビルド例:**
```bash
# CUDA 12.8 (デフォルト)
docker build -t pytorch-cu128 .

# CUDA 12.6
docker build --build-arg CUDA_VERSION=12.6.0 --build-arg CUDNN_VERSION=9 -t pytorch-cu126 .

# CUDA 13.0
docker build --build-arg CUDA_VERSION=13.0.2 --build-arg CUDNN_VERSION=9 -t pytorch-cu130 .
```

## テストスクリプト

### test_pytorch_cuda.py
```python
#!/usr/bin/env python
"""PyTorch CUDA環境検証スクリプト"""
import sys

def test_pytorch_cuda():
    """PyTorchとCUDAの動作確認"""
    try:
        import torch
        import torchvision
        
        print("=" * 60)
        print("PyTorch環境情報")
        print("=" * 60)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Torchvision Version: {torchvision.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            
            # 簡単なテンソル演算
            print("\nGPUテンソル演算テスト:")
            x = torch.rand(5, 3).cuda()
            print(f"  Tensor shape: {x.shape}")
            print(f"  Tensor device: {x.device}")
            print(f"  Tensor dtype: {x.dtype}")
            print("  ✓ GPU計算成功")
        else:
            print("\nCPUモードで動作中")
            x = torch.rand(5, 3)
            print(f"  Tensor shape: {x.shape}")
            print(f"  Tensor device: {x.device}")
            print("  ✓ CPU計算成功")
        
        print("=" * 60)
        print("✓ すべてのテスト完了")
        print("=" * 60)
        return 0
        
    except ImportError as e:
        print(f"エラー: {e}", file=sys.stderr)
        print("PyTorchがインストールされていません", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"予期しないエラー: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(test_pytorch_cuda())
```

## 実行手順

### 1. 現行Dockerfileの修正
```bash
# Dockerfileのベースイメージを修正
sed -i 's/13.0.2/12.8.0/g' .devcontainer/Dockerfile
sed -i 's/cudnn-devel/cudnn9-devel/g' .devcontainer/Dockerfile
```

### 2. テスト実行
```bash
# コンテナビルド
docker compose build pytorch-cu128

# コンテナ起動
docker compose up -d pytorch-cu128

# 環境確認
docker compose exec pytorch-cu128 uv run python test_pytorch_cuda.py

# CPU版テスト
docker compose up -d pytorch-cpu
docker compose exec pytorch-cpu uv run python test_pytorch_cuda.py
```

### 3. 複数バージョンのテスト
```bash
# すべてのバージョンでビルド
docker compose build

# 各環境でテスト
for env in cu128 cu126 cu130 cpu; do
  echo "Testing pytorch-$env..."
  docker compose exec pytorch-$env uv run python test_pytorch_cuda.py
done
```

## .gitignore追加

テスト用Dockerファイルを除外:
```gitignore
# Docker
docker-compose.override.yml
.docker/

# Test outputs
test_pytorch_*.py
```

## 推奨アクション

1. **即座実施**: `.devcontainer/Dockerfile`のCUDAバージョンを12.8.0に修正
2. **短期**: `docker-compose.yml`を作成してマルチ環境テスト
3. **中期**: CI/CDでCPU版自動テスト実装
4. **長期**: CUDA 13.0対応を検討（PyTorch 2.10+待ち）

## 参考リンク

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA Images](https://hub.docker.com/r/nvidia/cuda/tags)
- [cuDNN Release Notes](https://docs.nvidia.com/deeplearning/cudnn/release-notes/)
