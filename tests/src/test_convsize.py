# Copyright 2025 cslrtools2 contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the convsize module.

This module tests the conv_size function and ConvSize class for calculating
convolution output dimensions.
"""

from __future__ import annotations
import pytest # pyright: ignore[reportUnusedImport]
import torch
import torch.nn as nn

from cslrtools2.convsize import conv_size, ConvSize


class TestConvSize:
    """Test the conv_size function."""

    def test_basic_conv2d(self) -> None:
        """Test basic 2D convolution size calculation."""
        size = torch.tensor([224, 224], dtype=torch.int64)
        kernel_size = torch.tensor([3, 3], dtype=torch.int64)
        stride = torch.tensor([1, 1], dtype=torch.int64)
        padding = torch.tensor([1, 1], dtype=torch.int64)
        dilation = torch.tensor([1, 1], dtype=torch.int64)

        output = conv_size(size, kernel_size, stride, padding, dilation)

        # (224 + 2*1 - 1*(3-1) - 1) / 1 + 1 = 224
        assert torch.equal(output, torch.tensor([224, 224], dtype=torch.int64))

    def test_stride_2(self) -> None:
        """Test convolution with stride=2 (downsampling)."""
        size = torch.tensor([224, 224], dtype=torch.int64)
        kernel_size = torch.tensor([3, 3], dtype=torch.int64)
        stride = torch.tensor([2, 2], dtype=torch.int64)
        padding = torch.tensor([1, 1], dtype=torch.int64)
        dilation = torch.tensor([1, 1], dtype=torch.int64)

        output = conv_size(size, kernel_size, stride, padding, dilation)

        # (224 + 2*1 - 1*(3-1) - 1) / 2 + 1 = 112
        assert torch.equal(output, torch.tensor([112, 112], dtype=torch.int64))

    def test_no_padding(self) -> None:
        """Test convolution without padding."""
        size = torch.tensor([28, 28], dtype=torch.int64)
        kernel_size = torch.tensor([5, 5], dtype=torch.int64)
        stride = torch.tensor([1, 1], dtype=torch.int64)
        padding = torch.tensor([0, 0], dtype=torch.int64)
        dilation = torch.tensor([1, 1], dtype=torch.int64)

        output = conv_size(size, kernel_size, stride, padding, dilation)

        # (28 + 0 - 1*(5-1) - 1) / 1 + 1 = 24
        assert torch.equal(output, torch.tensor([24, 24], dtype=torch.int64))

    def test_dilation(self) -> None:
        """Test convolution with dilation."""
        size = torch.tensor([32, 32], dtype=torch.int64)
        kernel_size = torch.tensor([3, 3], dtype=torch.int64)
        stride = torch.tensor([1, 1], dtype=torch.int64)
        padding = torch.tensor([0, 0], dtype=torch.int64)
        dilation = torch.tensor([2, 2], dtype=torch.int64)

        output = conv_size(size, kernel_size, stride, padding, dilation)

        # (32 + 0 - 2*(3-1) - 1) / 1 + 1 = 28
        assert torch.equal(output, torch.tensor([28, 28], dtype=torch.int64))

    def test_asymmetric_sizes(self) -> None:
        """Test with asymmetric input dimensions."""
        size = torch.tensor([64, 32], dtype=torch.int64)
        kernel_size = torch.tensor([3, 3], dtype=torch.int64)
        stride = torch.tensor([2, 2], dtype=torch.int64)
        padding = torch.tensor([1, 1], dtype=torch.int64)
        dilation = torch.tensor([1, 1], dtype=torch.int64)

        output = conv_size(size, kernel_size, stride, padding, dilation)

        # H: (64 + 2*1 - 1*(3-1) - 1) / 2 + 1 = 32
        # W: (32 + 2*1 - 1*(3-1) - 1) / 2 + 1 = 16
        assert torch.equal(output, torch.tensor([32, 16], dtype=torch.int64))


class TestConvSizeClass:
    """Test the ConvSize class."""

    def test_convsize_class_instantiation(self) -> None:
        """Test ConvSize class can be instantiated."""
        conv_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        conv = ConvSize(conv_layer)
        assert conv is not None

    def test_convsize_class_forward(self) -> None:
        """Test ConvSize forward method."""
        conv_layer = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        conv = ConvSize(conv_layer)
        
        input_size = torch.tensor([224, 224], dtype=torch.int64)
        output = conv.forward(input_size)
        
        # Should match manual calculation
        expected = torch.tensor([112, 112], dtype=torch.int64)
        assert torch.equal(output, expected)

    def test_convsize_class_callable(self) -> None:
        """Test ConvSize can be called like a function."""
        conv_layer = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, dilation=1)
        conv = ConvSize(conv_layer)
        
        input_size = torch.tensor([32, 32], dtype=torch.int64)
        output = conv(input_size)
        
        # (32 + 2*2 - 1*(5-1) - 1) / 1 + 1 = 32 (same padding)
        expected = torch.tensor([32, 32], dtype=torch.int64)
        assert torch.equal(output, expected)

    def test_convsize_sequential_operations(self) -> None:
        """Test sequential convolution size calculations."""
        # Simulate two conv layers
        conv1_layer = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        conv1 = ConvSize(conv1_layer)
        
        conv2_layer = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        conv2 = ConvSize(conv2_layer)
        
        input_size = torch.tensor([224, 224], dtype=torch.int64)
        output1 = conv1(input_size)  # 112x112
        output2 = conv2(output1)      # 56x56
        
        assert torch.equal(output1, torch.tensor([112, 112], dtype=torch.int64))
        assert torch.equal(output2, torch.tensor([56, 56], dtype=torch.int64))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_kernel(self) -> None:
        """Test with large kernel size."""
        size = torch.tensor([100, 100], dtype=torch.int64)
        kernel_size = torch.tensor([11, 11], dtype=torch.int64)
        stride = torch.tensor([1, 1], dtype=torch.int64)
        padding = torch.tensor([5, 5], dtype=torch.int64)
        dilation = torch.tensor([1, 1], dtype=torch.int64)

        output = conv_size(size, kernel_size, stride, padding, dilation)
        
        # (100 + 2*5 - 1*(11-1) - 1) / 1 + 1 = 100
        assert torch.equal(output, torch.tensor([100, 100], dtype=torch.int64))

    def test_3d_convolution(self) -> None:
        """Test 3D convolution size calculation."""
        size = torch.tensor([16, 32, 32], dtype=torch.int64)
        kernel_size = torch.tensor([3, 3, 3], dtype=torch.int64)
        stride = torch.tensor([2, 2, 2], dtype=torch.int64)
        padding = torch.tensor([1, 1, 1], dtype=torch.int64)
        dilation = torch.tensor([1, 1, 1], dtype=torch.int64)

        output = conv_size(size, kernel_size, stride, padding, dilation)
        
        # Each dimension: (n + 2*1 - 1*(3-1) - 1) / 2 + 1
        # D: (16 + 2 - 2 - 1) / 2 + 1 = 8
        # H: (32 + 2 - 2 - 1) / 2 + 1 = 16
        # W: (32 + 2 - 2 - 1) / 2 + 1 = 16
        assert torch.equal(output, torch.tensor([8, 16, 16], dtype=torch.int64))
