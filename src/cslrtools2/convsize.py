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

# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from typing import (
    TypeVar, Generic, overload,
    Sequence
)
import torch
import torch.nn as nn
from torch import Tensor, Size

def conv_size(
    size: Tensor,
    kernel_size: Tensor,
    stride: Tensor,
    padding: Tensor,
    dilation: Tensor
    ) -> Tensor:
    """Calculate the output size of a convolution operation.

    Computes the spatial dimensions of the output feature map for a
    convolution layer given the input size and convolution parameters.

    Args:
        size (`Tensor`): Input spatial dimensions tensor. Shape: ``(N,)``
            where N is the number of spatial dimensions (typically 2 for
            height and width).
        kernel_size (`Tensor`): Convolution kernel size tensor. Shape: ``(N,)``.
        stride (`Tensor`): Stride of the convolution. Shape: ``(N,)``.
        padding (`Tensor`): Padding applied to the input. Shape: ``(N,)``.
        dilation (`Tensor`): Spacing between kernel elements. Shape: ``(N,)``.

    Returns:
        :class:`Tensor`: Output spatial dimensions. Shape: ``(N,)`` matching the
            input size tensor shape.

    Note:
        This function implements the standard convolution output size formula used
        by :class:`torch.nn.Conv2d` and :class:`torch.nn.Conv3d`:

        .. math::
            \\text{output} = \\left\\lfloor \\frac{\\text{input} + 2 \\times \\text{padding} - \\text{dilation} \\times (\\text{kernel} - 1) - 1}{\\text{stride}} \\right\\rfloor + 1

    Example:
        Calculate output size for a 2D convolution::

            >>> import torch
            >>> from cslrtools2.convsize import conv_size
            >>> output = conv_size(
            ...     size=torch.tensor([224, 224]),
            ...     kernel_size=torch.tensor([3, 3]),
            ...     stride=torch.tensor([2, 2]),
            ...     padding=torch.tensor([1, 1]),
            ...     dilation=torch.tensor([1, 1])
            ... )
            >>> output
            tensor([112, 112])

        Verify with PyTorch :class:`torch.nn.Conv2d` layer::

            >>> import torch.nn as nn
            >>> conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            >>> x = torch.randn(1, 3, 224, 224)
            >>> y = conv(x)
            >>> y.shape
            torch.Size([1, 64, 112, 112])
    """

    return torch.floor_divide(
        size + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride
    ) + 1

def conv_transpose_size(
    size: Tensor,
    kernel_size: Tensor,
    stride: Tensor,
    padding: Tensor,
    output_padding: Tensor,
    dilation: Tensor
    ) -> Tensor:
    """Calculate the output size of a transposed convolution operation.

    Computes the spatial dimensions of the output feature map for a
    transposed convolution (deconvolution) layer given the input size
    and convolution parameters.

    Args:
        size (`Tensor`): Input spatial dimensions tensor. Shape: ``(N,)``
            where N is the number of spatial dimensions.
        kernel_size (`Tensor`): Convolution kernel size tensor. Shape: ``(N,)``.
        stride (`Tensor`): Stride of the transposed convolution. Shape: ``(N,)``.
        padding (`Tensor`): Padding applied to the input. Shape: ``(N,)``.
        output_padding (`Tensor`): Additional size added to one side of
            the output shape. Shape: ``(N,)``.
        dilation (`Tensor`): Spacing between kernel elements. Shape: ``(N,)``.

    Returns:
        :class:`Tensor`: Output spatial dimensions. Shape: ``(N,)`` matching
            the input size tensor shape.

    Note:
        This function implements the transposed convolution output size formula
        used by :class:`torch.nn.ConvTranspose2d` and :class:`torch.nn.ConvTranspose3d`:

        .. math::
            \\text{output} = (\\text{input} - 1) \\times \\text{stride} - 2 \\times \\text{padding} + \\text{dilation} \\times (\\text{kernel} - 1) + \\text{output\\_padding} + 1

    Example:
        Calculate output size for a 2D transposed convolution::

            >>> import torch
            >>> from cslrtools2.convsize import conv_transpose_size
            >>> output = conv_transpose_size(
            ...     size=torch.tensor([112, 112]),
            ...     kernel_size=torch.tensor([3, 3]),
            ...     stride=torch.tensor([2, 2]),
            ...     padding=torch.tensor([1, 1]),
            ...     output_padding=torch.tensor([1, 1]),
            ...     dilation=torch.tensor([1, 1])
            ... )
            >>> output
            tensor([224, 224])

        Verify with PyTorch :class:`torch.nn.ConvTranspose2d` layer::

            >>> import torch.nn as nn
            >>> deconv = nn.ConvTranspose2d(
            ...     64, 3, kernel_size=3, stride=2,
            ...     padding=1, output_padding=1
            ... )
            >>> x = torch.randn(1, 64, 112, 112)
            >>> y = deconv(x)
            >>> y.shape
            torch.Size([1, 3, 224, 224])
    """

    return (
        (size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding + 1
    )

_ConvNd = TypeVar('_ConvNd', bound=nn.Conv1d | nn.Conv2d | nn.Conv3d)
_ConvTransposeNd = TypeVar('_ConvTransposeNd', bound=nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d)

class ConvSize(nn.Module, Generic[_ConvNd]):
    """Calculate the output size of a convolutional layer.

    A :class:`torch.nn.Module` that computes the spatial output dimensions
    for convolutional layers (:class:`torch.nn.Conv1d`, :class:`torch.nn.Conv2d`,
    :class:`torch.nn.Conv3d`) given their parameters.

    Args:
        conv (_ConvNd): A convolutional layer instance (e.g., :class:`torch.nn.Conv1d`,
            :class:`torch.nn.Conv2d`, :class:`torch.nn.Conv3d`).

    Shape:
        - Input: :class:`torch.Tensor` of shape ``(batch_size, channels, ...)``
          or :class:`torch.Size`.
        - Output: :class:`torch.Tensor` of shape ``(batch_size, out_channels, ...)``
          or :class:`torch.Size`.

    Example:
        Calculate output size for a 2D convolutional layer::

            >>> import torch
            >>> import torch.nn as nn
            >>> from cslrtools2.convsize import ConvSize
            >>> conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            >>> conv_size_calc = ConvSize(conv)
            >>> input_size = torch.tensor([224, 224])
            >>> output_size = conv_size_calc(input_size)
            >>> output_size
            tensor([112, 112])
    """

    def __init__(self, conv: _ConvNd):
        super().__init__() # pyright: ignore[reportUnknownMemberType]
        self.kernel_size = nn.Parameter(torch.tensor(conv.kernel_size), requires_grad=False)
        self.stride = nn.Parameter(torch.tensor(conv.stride), requires_grad=False)
        self.padding = nn.Parameter(torch.tensor(conv.padding), requires_grad=False)
        self.dilation = nn.Parameter(torch.tensor(conv.dilation), requires_grad=False)

    @overload
    def forward(
        self,
        size: Tensor,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor: ...

    @overload
    def forward(
        self,
        shape: Size,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Size: ...

    def forward( # pyright: ignore[reportInconsistentOverload]
        # Reason: Cannot express "arg is Tensor XOR arg is Size" constraint in Python type system.
        # Overloads guarantee correct usage, runtime isinstance checks handle validation.
        self,
        arg: Tensor | Size,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor | Size:

        if isinstance(arg, Tensor):
            return self._forward_tensor(arg, dim)
        elif isinstance(arg, Size):
            return self._forward_size(arg)
        else:
            raise TypeError(f"Expected Tensor or Size, got {type(arg).__name__}")

    def _forward_tensor(
        self,
        size: Tensor,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor:
        return conv_size(
            size,
            self.kernel_size[dim],
            self.stride[dim],
            self.padding[dim],
            self.dilation[dim]
        )

    def _forward_size(
        self,
        shape: Size
        ) -> Size:

        change_ndim = len(self.kernel_size)
        change_shape = shape[-change_ndim:]
        unchange_shape = shape[:-change_ndim]
        changed_shape = Size(
            conv_size(
                # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                torch.tensor(change_shape),
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation
            ).tolist()
        )
        return Size(unchange_shape + changed_shape)

class ConvTransposeSize(nn.Module, Generic[_ConvTransposeNd]):
    """Calculate the output size of a transposed convolutional layer.

    A :class:`torch.nn.Module` that computes the spatial output dimensions
    for transposed convolutional layers (:class:`torch.nn.ConvTranspose1d`,
    :class:`torch.nn.ConvTranspose2d`, :class:`torch.nn.ConvTranspose3d`)
    given their parameters.

    Args:
        conv (_ConvTransposeNd): A transposed convolutional layer instance
            (e.g., :class:`torch.nn.ConvTranspose1d`, :class:`torch.nn.ConvTranspose2d`,
            :class:`torch.nn.ConvTranspose3d`).

    Shape:
        - Input: :class:`torch.Tensor` of shape ``(batch_size, channels, ...)``
          or :class:`torch.Size`.
        - Output: :class:`torch.Tensor` of shape ``(batch_size, out_channels, ...)``
          or :class:`torch.Size`.

    Example:
        Calculate output size for a 2D transposed convolutional layer::

            >>> import torch
            >>> import torch.nn as nn
            >>> from cslrtools2.convsize import ConvTransposeSize
            >>> deconv = nn.ConvTranspose2d(
            ...     64, 3, kernel_size=3, stride=2,
            ...     padding=1, output_padding=1
            ... )
            >>> deconv_size_calc = ConvTransposeSize(deconv)
            >>> input_size = torch.tensor([112, 112])
            >>> output_size = deconv_size_calc(input_size)
            >>> output_size
            tensor([224, 224])
    """

    def __init__(self, conv: _ConvTransposeNd):
        super().__init__() # pyright: ignore[reportUnknownMemberType]
        self.kernel_size = nn.Parameter(torch.tensor(conv.kernel_size), requires_grad=False)
        self.stride = nn.Parameter(torch.tensor(conv.stride), requires_grad=False)
        self.padding = nn.Parameter(torch.tensor(conv.padding), requires_grad=False)
        self.output_padding = nn.Parameter(torch.tensor(conv.output_padding), requires_grad=False)
        self.dilation = nn.Parameter(torch.tensor(conv.dilation), requires_grad=False)

    @overload
    def forward(
        self,
        size: Tensor,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor: ...

    @overload
    def forward(
        self,
        shape: Size,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Size: ...

    def forward( # pyright: ignore[reportInconsistentOverload]
        # Reason: Cannot express "arg is Tensor XOR arg is Size" constraint in Python type system.
        # Overloads guarantee correct usage, runtime isinstance checks handle validation.
        self,
        arg: Tensor | Size,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor | Size:

        if isinstance(arg, Tensor):
            return self._forward_tensor(arg, dim)
        elif isinstance(arg, Size):
            return self._forward_size(arg)
        else:
            raise TypeError(f"Expected Tensor or Size, got {type(arg).__name__}")

    def _forward_tensor(
        self,
        size: Tensor,
        dim: int | Sequence[int] | slice = slice(None)
        ) -> Tensor:
        return conv_transpose_size(
            size,
            self.kernel_size[dim],
            self.stride[dim],
            self.padding[dim],
            self.output_padding[dim],
            self.dilation[dim]
        )

    def _forward_size(
        self,
        shape: Size
        ) -> Size:

        change_ndim = len(self.kernel_size)
        change_shape = shape[-change_ndim:]
        unchange_shape = shape[:-change_ndim]
        changed_shape = Size(
            conv_transpose_size(
                # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                torch.tensor(change_shape),
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation
            ).tolist()
        )
        return Size(unchange_shape + changed_shape)
