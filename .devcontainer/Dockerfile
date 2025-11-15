FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 非対話モードに設定
ENV DEBIAN_FRONTEND=noninteractive

# 基本パッケージとPython関連のインストール
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# uvのインストール（マルチステージビルドでコピー）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# rootユーザーで実行（UVキャッシュの権限問題を回避）
WORKDIR /workspace
