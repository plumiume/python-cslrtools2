#!/usr/bin/env python
"""PyTorch CUDA環境検証スクリプト

使い方:
    # 基本的な環境確認
    python test_pytorch_cuda.py

    # MediaPipe互換性テスト（オプション）
    python test_pytorch_cuda.py --test-mediapipe
"""
from typing import Protocol, cast
import sys
import argparse

class CUuuid(Protocol):
    @property
    def bytes(self) -> list[int]: ...

class CudaDeviceProperties(Protocol):
    @property
    def gcnArchName(self) -> str: ...
    @property
    def is_integrated(self) -> int: ...
    @property
    def is_multi_gpu_board(self) -> int: ...
    @property
    def major(self) -> int: ...
    @property
    def minor(self) -> int: ...
    @property
    def max_threads_per_multi_processor(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def pci_bus_id(self) -> int: ...
    @property
    def pci_device_id(self) -> int: ...
    @property
    def pci_domain_id(self) -> int: ...
    @property
    def regs_per_multiprocessor(self) -> int: ...
    @property
    def shared_memory_per_block(self) -> int: ...
    @property
    def shared_memory_per_block_optin(self) -> int: ...
    @property
    def shared_memory_per_multiprocessor(self) -> int: ...
    @property
    def total_memory(self) -> int: ...
    @property
    def uuid(self) -> CUuuid: ...
    @property
    def warp_size(self) -> int: ...


def test_pytorch_cuda():
    """PyTorchとCUDAの動作確認"""
    try:
        import torch
        import torchvision # pyright: ignore[reportMissingTypeStubs]

        print("=" * 70)
        print("PyTorch環境情報")
        print("=" * 70)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Torchvision Version: {torchvision.__version__}")
        print(f"Python Version: {sys.version}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_props = cast(CudaDeviceProperties, torch.cuda.get_device_properties(i)) # pyright: ignore[reportUnknownMemberType]
                print(f"  Device {i}: {device_name}")
                print(f"    Total Memory: {device_props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute Capability: {device_props.major}.{device_props.minor}")

            # GPUテンソル演算テスト
            print("\n" + "=" * 70)
            print("GPUテンソル演算テスト")
            print("=" * 70)
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"  Matrix Shape: {z.shape}")
            print(f"  Matrix Device: {z.device}")
            print(f"  Matrix dtype: {z.dtype}")
            print("  ✓ GPU行列演算成功")
        else:
            print("\n" + "=" * 70)
            print("CPUモード")
            print("=" * 70)
            x = torch.rand(100, 100)
            y = torch.rand(100, 100)
            z = torch.matmul(x, y)
            print(f"  Matrix Shape: {z.shape}")
            print(f"  Matrix Device: {z.device}")
            print("  ✓ CPU行列演算成功")

        return True

    except ImportError as e:
        print(f"エラー: {e}", file=sys.stderr)
        print("PyTorchまたはtorchvisionがインストールされていません", file=sys.stderr)
        return False
    except Exception as e:
        print(f"予期しないエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def test_mediapipe():
    """MediaPipeの動作確認（オプション）"""
    try:
        import mediapipe # pyright: ignore[reportMissingTypeStubs]
        import mediapipe.python.solutions.pose as mp_pose # pyright: ignore[reportMissingTypeStubs]

        print("\n" + "=" * 70)
        print("MediaPipe環境情報")
        print("=" * 70)
        print(f"MediaPipe Version: {mediapipe.__version__}") # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

        # 簡単な初期化テスト
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        print("  ✓ MediaPipe Pose初期化成功")
        pose.close()

        return True

    except ImportError:
        print("\n" + "=" * 70)
        print("MediaPipe: インストールされていません（オプション）")
        print("=" * 70)
        return True  # MediaPipeはオプショナルなので失敗としない
    except Exception as e:
        print(f"MediaPipeエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="PyTorch CUDA環境検証スクリプト"
    )
    parser.add_argument(
        "--test-mediapipe",
        action="store_true",
        help="MediaPipe互換性テストを実行"
    )
    args = parser.parse_args()

    success = test_pytorch_cuda()

    if args.test_mediapipe:
        success = test_mediapipe() and success

    print("\n" + "=" * 70)
    if success:
        print("✓ すべてのテスト完了")
    else:
        print("✗ テスト失敗")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
