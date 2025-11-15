# ConvSize API Reference

## Overview

The `convsize` module provides utilities for calculating output sizes of convolutional layers in PyTorch models.

## Functions

### conv_size

Calculate the output size of a convolution operation.

```python
from cslrtools2.convsize import conv_size
import torch

output_size = conv_size(
    size: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    dilation: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `size` (Tensor): Input spatial dimensions. Shape: `(N,)` where N is number of spatial dimensions
- `kernel_size` (Tensor): Convolution kernel size. Shape: `(N,)`
- `stride` (Tensor): Stride of the convolution. Shape: `(N,)`
- `padding` (Tensor): Padding applied to the input. Shape: `(N,)`
- `dilation` (Tensor): Spacing between kernel elements. Shape: `(N,)`

**Returns:**
- `Tensor`: Output spatial dimensions. Shape: `(N,)`

**Formula:**

$$
\text{output} = \left\lfloor \frac{\text{input} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel} - 1) - 1}{\text{stride}} \right\rfloor + 1
$$

**Example:**

```python
import torch
from cslrtools2.convsize import conv_size

# 2D convolution: 224x224 input, 3x3 kernel, stride 2, padding 1
output = conv_size(
    size=torch.tensor([224, 224]),
    kernel_size=torch.tensor([3, 3]),
    stride=torch.tensor([2, 2]),
    padding=torch.tensor([1, 1]),
    dilation=torch.tensor([1, 1])
)
print(output)  # tensor([112, 112])
```

---

### conv_transpose_size

Calculate the output size of a transposed convolution (deconvolution) operation.

```python
from cslrtools2.convsize import conv_transpose_size
import torch

output_size = conv_transpose_size(
    size: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    output_padding: torch.Tensor,
    dilation: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `size` (Tensor): Input spatial dimensions. Shape: `(N,)`
- `kernel_size` (Tensor): Convolution kernel size. Shape: `(N,)`
- `stride` (Tensor): Stride of the transposed convolution. Shape: `(N,)`
- `padding` (Tensor): Padding applied to the input. Shape: `(N,)`
- `output_padding` (Tensor): Additional size added to output. Shape: `(N,)`
- `dilation` (Tensor): Spacing between kernel elements. Shape: `(N,)`

**Returns:**
- `Tensor`: Output spatial dimensions. Shape: `(N,)`

**Formula:**

$$
\text{output} = (\text{input} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation} \times (\text{kernel} - 1) + \text{output\_padding} + 1
$$

**Example:**

```python
import torch
from cslrtools2.convsize import conv_transpose_size

# 2D transposed convolution: 112x112 input
output = conv_transpose_size(
    size=torch.tensor([112, 112]),
    kernel_size=torch.tensor([3, 3]),
    stride=torch.tensor([2, 2]),
    padding=torch.tensor([1, 1]),
    output_padding=torch.tensor([1, 1]),
    dilation=torch.tensor([1, 1])
)
print(output)  # tensor([224, 224])
```

---

## Classes

### ConvSize

PyTorch module for calculating convolution output sizes.

```python
from cslrtools2.convsize import ConvSize
import torch.nn as nn

conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
conv_size_calc = ConvSize(conv)
```

**Constructor:**

```python
ConvSize(conv: nn.Conv1d | nn.Conv2d | nn.Conv3d)
```

**Parameters:**
- `conv`: A PyTorch convolutional layer instance

**Methods:**

#### forward

```python
def forward(
    self,
    arg: torch.Tensor | torch.Size,
    dim: int | Sequence[int] | slice = slice(None)
) -> torch.Tensor | torch.Size
```

Calculate output size from input size.

**Parameters:**
- `arg`: Input size (Tensor or Size)
- `dim`: Dimensions to apply convolution (default: all)

**Returns:**
- Output size (same type as input)

**Example:**

```python
import torch
import torch.nn as nn
from cslrtools2.convsize import ConvSize

conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
conv_size_calc = ConvSize(conv)

input_size = torch.Size([224, 224])
output_size = conv_size_calc(input_size)
print(output_size)  # torch.Size([112, 112])
```

---

### ConvTransposeSize

PyTorch module for calculating transposed convolution output sizes.

```python
from cslrtools2.convsize import ConvTransposeSize
import torch.nn as nn

deconv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
deconv_size_calc = ConvTransposeSize(deconv)
```

**Constructor:**

```python
ConvTransposeSize(conv: nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d)
```

**Parameters:**
- `conv`: A PyTorch transposed convolutional layer instance

**Methods:**

#### forward

```python
def forward(
    self,
    arg: torch.Tensor | torch.Size,
    dim: int | Sequence[int] | slice = slice(None)
) -> torch.Tensor | torch.Size
```

**Example:**

```python
import torch
import torch.nn as nn
from cslrtools2.convsize import ConvTransposeSize

deconv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
deconv_size_calc = ConvTransposeSize(deconv)

input_size = torch.Size([112, 112])
output_size = deconv_size_calc(input_size)
print(output_size)  # torch.Size([224, 224])
```

---

## Use Cases

### Track Output Shapes Through Network

```python
import torch.nn as nn
from cslrtools2.convsize import ConvSize

# Define network layers
conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

# Create size calculators
calc1 = ConvSize(conv1)
calc2 = ConvSize(conv2)
calc3 = ConvSize(conv3)

# Track shape changes
input_size = torch.Size([224, 224])
size1 = calc1(input_size)
size2 = calc2(size1)
size3 = calc3(size2)

print(f"Input: {input_size}")
print(f"After conv1: {size1}")
print(f"After conv2: {size2}")
print(f"After conv3: {size3}")
```

### Verify Network Architecture

```python
import torch
import torch.nn as nn
from cslrtools2.convsize import ConvSize

# Define network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Verify output size calculation
net = SimpleNet()
calc1 = ConvSize(net.conv1)
calc2 = ConvSize(net.conv2)

# Expected output
input_size = torch.Size([1, 3, 224, 224])
expected_size = calc2(calc1(torch.Size([224, 224])))

# Actual output
x = torch.randn(input_size)
y = net(x)

print(f"Expected spatial size: {expected_size}")
print(f"Actual output shape: {y.shape}")
assert y.shape[2:] == expected_size
```

---

## Navigation

- **[← API Reference](index.md)**
- **[← LMPipe API](lmpipe.md)**
