#!/bin/bash
# PyTorch特定バージョンのテストスクリプト
# 
# 使用方法:
#   docker compose run --rm --no-deps pytorch-2.3-cu128 bash /workspace/docker/test-torch-version.sh
#
# 環境変数:
#   TORCH_VERSION_CONSTRAINT: torchの制約 (例: ">=2.3.0,<2.4.0")
#   TORCHVISION_VERSION_CONSTRAINT: torchvisionの制約 (例: ">=0.18.0,<0.19.0")
#   PYTORCH_INDEX_URL: PyTorchインデックスURL (デフォルト: https://download.pytorch.org/whl/cu121)

set -e

# デフォルト値
TORCH_VERSION_CONSTRAINT="${TORCH_VERSION_CONSTRAINT:->=2.3.0,<2.4.0}"
TORCHVISION_VERSION_CONSTRAINT="${TORCHVISION_VERSION_CONSTRAINT:->=0.18.0,<0.19.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-http://pypi-nginx/pypi/}"

echo "========================================"
echo "PyTorch Version Test"
echo "========================================"
echo "Torch constraint: ${TORCH_VERSION_CONSTRAINT}"
echo "Torchvision constraint: ${TORCHVISION_VERSION_CONSTRAINT}"
echo "PyTorch index: ${PYTORCH_INDEX_URL}"
echo "PyPI index: ${PYPI_INDEX_URL}"
echo ""

# 一時プロジェクトを初期化
uv init --bare --python cp312 /tmp/test-env
cd /tmp/test-env

# インデックス設定を追加
cat >> pyproject.toml << EOF

[[tool.uv.index]]
name = "pytorch"
url = "${PYTORCH_INDEX_URL}"
explicit = true

[[tool.uv.index]]
name = "pypi"
url = "${PYPI_INDEX_URL}"
default = true

[tool.uv.sources]
torch = { index = "pytorch" }
torchvision = { index = "pytorch" }

# Linux環境のみに限定
[tool.uv]
environments = ["sys_platform == 'linux'"]
EOF

# PyTorchをインストール
echo "Installing torch and torchvision..."
uv add "torch${TORCH_VERSION_CONSTRAINT}" "torchvision${TORCHVISION_CONSTRAINT}"

echo ""
echo "========================================"
echo "Running test..."
echo "========================================"

# テスト実行（PYTHONPATHで/workspaceを追加）
PYTHONPATH=/workspace/src uv run python /workspace/test_pytorch_cuda.py
