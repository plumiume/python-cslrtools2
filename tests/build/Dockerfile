# マルチステージビルド: PyTorch環境テスト用Dockerfile
# 使用方法: docker compose build --build-arg TARGET=<target-name>
# または: docker compose.yaml で target を指定

# ==============================================================================
# ベースステージ: 共通の基本パッケージ
# ==============================================================================
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# 基本パッケージインストール
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# uvインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# rootユーザーで実行（UVキャッシュの権限問題を回避）
WORKDIR /workspace

# ==============================================================================
# CPU専用ステージ
# ==============================================================================
FROM base AS cpu

# PyTorch CPU版のインデックスを設定
ENV UV_INDEX_URL=https://download.pytorch.org/whl/cpu

# ==============================================================================
# CUDA 12.6ステージ
# ==============================================================================
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04 AS cuda-12.6

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# PyTorch CUDA 12.6用のインデックスを設定
ENV UV_INDEX_URL=https://download.pytorch.org/whl/cu126

# ==============================================================================
# CUDA 12.8ステージ (メイン開発環境)
# ==============================================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS cuda-12.8

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# PyTorch CUDA 12.8用のインデックスを設定（デフォルト）
ENV UV_INDEX_URL=https://download.pytorch.org/whl/cu128

# ==============================================================================
# CUDA 13.0ステージ
# ==============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS cuda-13.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# PyTorch CUDA 13.0用のインデックスを設定
ENV UV_INDEX_URL=https://download.pytorch.org/whl/cu130

# ==============================================================================
# PyTorch 2.3.0検証ステージ (CUDA 12.8ベース)
# ==============================================================================
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS pytorch-2.3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# PyTorch 2.3.0用のインデックスを設定 (CUDA 12.8対応)
ENV UV_INDEX_URL=https://download.pytorch.org/whl/cu128

# Note: pyproject.tomlのtorch>=2.9.0を一時的に2.3.0に変更してテストする必要がある
