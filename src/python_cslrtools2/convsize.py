# pyright: reportUnnecessaryIsInstance=false

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
    """Calculate the output size of a convolution.

    Args:
        size (`Tensor`): Input size tensor.
        kernel_size (`Tensor`): Kernel size tensor.
        stride (`Tensor`): Stride tensor.
        padding (`Tensor`): Padding tensor.
        dilation (`Tensor`): Dilation tensor.

    Returns:
        :class:`Tensor`: Output size tensor.
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
    """Calculate the output size of a transposed convolution.

    Args:
        size (`Tensor`): Input size tensor.
        kernel_size (`Tensor`): Kernel size tensor.
        stride (`Tensor`): Stride tensor.
        padding (`Tensor`): Padding tensor.
        output_padding (`Tensor`): Output padding tensor.
        dilation (`Tensor`): Dilation tensor.

    Returns:
        :class:`Tensor`: Output size tensor.
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

    """
    Calculate the output size of a convolutional layer.

    Parameters:
        conv (`_ConvNd`): A convolutional layer instance
        (e.g., :class:`nn.Conv1d`, :class:`nn.Conv2d`, :class:`nn.Conv3d`).

    Shape:
        - Input: :class:`Tensor` of shape `(batch_size, channels, ...)` or :class:`Size`.
        - Output: :class:`Tensor` of shape `(batch_size, out_channels, ...)` or :class:`Size`.
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

    def forward( # type: ignore[reportInconsistentOverload]
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

    """
    Calculate the output size of a transposed convolutional layer.

    Parameters:
        conv (`_ConvTransposeNd`): A transposed convolutional layer instance
        (e.g., :class:`nn.ConvTranspose1d`, :class:`nn.ConvTranspose2d`, :class:`nn.ConvTranspose3d`).

    Shape:
        - Input: :class:`Tensor` of shape `(batch_size, channels, ...)` or :class:`Size`.
        - Output: :class:`Tensor` of shape `(batch_size, out_channels, ...)` or :class:`Size`.
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

    def forward( # type: ignore[reportInconsistentOverload]
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
