# Docstring Style Guide

> **関連ドキュメント**: このガイドを使用する際は、以下のドキュメントも併せて参照してください。
> - [コーディングスタイルガイド](CODING_STYLE_GUIDE.md) - 全体的なコード規約
> - [Pyrightスタイルガイド](PYRIGHT_STYLE_GUIDE.md) - 型ヒント記述規約
> - [例外処理・ログスタイルガイド](EXCEPTION_LOGGING_STYLE_GUIDE.md) - 例外のドキュメント方法

このプロジェクトでは、**Google Style + Sphinx**のドキュメント形式を採用しています。

## 基本原則

1. **Google Style**: セクションはインデントベース
2. **Sphinx互換**: reStructuredText (reST) ロールを使用
3. **バッククォート**: Args内の型名は `` `Type` `` で囲む（VSCode表示用）
4. **本文内のロール**: 説明文中のクラス、メソッド、関数等も適切なロールでマークアップ

## Sphinxロールの適用ルール

### セクション内での使用

すべてのセクション（Args, Returns, Raises, Note, Example等）の本文で、
以下のキーワードが出現した場合は適切なSphinxロールでマークアップすること：

- **クラス名**: `:class:`ClassName``
- **メソッド名**: `:meth:`method_name``
- **関数名**: `:func:`function_name``
- **属性名**: `:attr:`attribute_name``
- **例外名**: `:exc:`ExceptionName``
- **モジュール名**: `:mod:`module.name``
- **汎用オブジェクト**: `:obj:`object_name``
- **定数/リテラル**: ``:obj:`True```, ``:obj:`False```, ``:obj:`None```

### 例

```python
"""
Args:
    param (`Tensor`): Input tensor. See :class:`torch.Tensor` for details.
        Must not be :obj:`None`.

Returns:
    :class:`Tensor`: Output tensor processed by :meth:`forward` method.

Raises:
    :exc:`ValueError`: Raised when :func:`validate_input` fails.

Note:
    This implementation uses :class:`torch.nn.Module` as the base class.
    The :meth:`forward` method must be called with :obj:`training=True`.
"""
```

## テンプレート

### モジュールレベルdocstring

```python
"""Brief module description.

More detailed description of what the module does.
Can span multiple lines.

Attributes:
    module_level_variable (type): Description of module-level variable.

Example:
    Example usage of the module::

        >>> import module
        >>> module.function()
"""
```

### 関数/メソッドdocstring

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief function description (one line).

    More detailed description if needed. Can reference other functions
    like :func:`related_function` or classes like :class:`RelatedClass`.
    Can span multiple lines.

    Args:
        param1 (`Type1`): Description of param1. Use backticks for types
            to enable VSCode hover information. References to other objects
            should use Sphinx roles like :class:`SomeClass`.
        param2 (`Type2`): Description of param2. If :obj:`None`, uses
            default behavior. Multiple lines should be indented to align
            with the description start.

    Returns:
        :class:`ReturnType`: Description of return value. Use Sphinx
            roles like :class:, :obj:, :func:, etc. Returns :obj:`None`
            if no valid result is found.

    Raises:
        :exc:`ValueError`: Raised when :func:`validate_input` returns
            :obj:`False` or param1 is :obj:`None`.
        :exc:`TypeError`: Raised when param2 is not an instance of
            :class:`Type2`.

    Note:
        Additional notes, implementation details, or caveats. This method
        calls :meth:`helper_method` internally and uses :class:`torch.Tensor`
        for computation.

    Warning:
        Do not call this function when :attr:`is_initialized` is :obj:`False`.

    See Also:
        :func:`related_function` : Related function description
        :class:`RelatedClass` : Related class description

    Example:
        Example usage with expected behavior::

            >>> result = function_name(value1, value2)
            >>> print(result)
            expected_output
            >>> # Returns None when value1 is None
            >>> function_name(None, value2)
            None
    """
```

### クラスdocstring

```python
class ClassName:
    """Brief class description.

    More detailed description of the class purpose and usage.
    This class extends :class:`BaseClass` and implements :meth:`interface_method`.

    Attributes:
        attr1 (`Type1`): Description of attribute 1. Set to :obj:`None` by default.
        attr2 (`Type2`): Description of attribute 2. Initialized by :meth:`__init__`.

    Note:
        Instances of this class should be created via :func:`factory_function`
        rather than direct instantiation.

    Example:
        Example usage::

            >>> obj = ClassName(param)
            >>> obj.method()
            >>> # Check if attr1 is None
            >>> if obj.attr1 is None:
            ...     print("Not initialized")
    """

    def __init__(self, param: Type) -> None:
        """Initialize the class.

        Creates a new instance of :class:`ClassName` with the given parameter.

        Args:
            param (`Type`): Description of initialization parameter. Must not
                be :obj:`None`. If a :class:`CustomType`, additional validation
                is performed via :func:`validate_param`.

        Raises:
            :exc:`ValueError`: Raised when param is :obj:`None` or invalid.
        """
```

## Sphinxロールの使用

### 型参照（よく使うロール）

- `:class:`ClassName`` - クラス名（例: :class:`torch.Tensor`, :class:`pathlib.Path`）
- `:func:`function_name`` - 関数名（例: :func:`torch.stack`, :func:`validate_input`）
- `:meth:`method_name`` - メソッド名（例: :meth:`forward`, :meth:`__init__`）
- `:mod:`module.name`` - モジュール名（例: :mod:`torch.nn`, :mod:`numpy`）
- `:obj:`object_name`` - 汎用オブジェクト、定数、ブール値（例: :obj:`None`, :obj:`True`, :obj:`False`）
- `:exc:`Exception`` - 例外クラス（例: :exc:`ValueError`, :exc:`TypeError`）
- `:attr:`attribute`` - 属性（例: :attr:`shape`, :attr:`dtype`）
- `:const:`CONSTANT`` - 定数（例: :const:`MAX_SIZE`）
- `:data:`variable`` - モジュールレベル変数（例: :data:`__version__`）

### 適用のタイミング

以下の場合は**必ず**適切なロールを使用：

1. **クラス名が出現**: `torch.Tensor` → `:class:`torch.Tensor``
2. **関数名が出現**: `validate_input()` → `:func:`validate_input``
3. **メソッド名が出現**: `.forward()` → `:meth:`forward``
4. **None/True/False**: `None` → `:obj:`None``
5. **例外名**: `ValueError` → `:exc:`ValueError``
6. **属性名**: `.shape` → `:attr:`shape``

### 外部参照

外部パッケージへの参照も同様にマークアップ：

```python
"""
This function uses :class:`torch.Tensor` for computation and calls
:func:`numpy.array` for conversion. Results are compatible with
:class:`pandas.DataFrame`.

See :mod:`torch.nn.functional` for related operations.
"""
```

### 数式

```python
"""
.. math::
    y = \\frac{x + 2}{\\sigma}
"""
```

## コードブロック

### インラインコード

```python
"""Use ``variable_name`` for inline code."""
```

### コードブロック

```python
"""
Example::

    >>> import torch
    >>> x = torch.tensor([1, 2, 3])
    >>> print(x)
    tensor([1, 2, 3])
"""
```

## セクションの順序

関数/メソッドdocstringのセクション順序:

1. Brief description (1行)
2. Detailed description (複数行可)
3. `Args:`
4. `Returns:`
5. `Raises:`
6. `Yields:` (ジェネレータの場合)
7. `Note:`
8. `Warning:`
9. `See Also:`
10. `Example:` または `Examples:`

## 実例

### 良い例

```python
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
            where N is the number of spatial dimensions.
        kernel_size (`Tensor`): Convolution kernel size tensor. Shape: ``(N,)``.
        stride (`Tensor`): Stride of the convolution. Shape: ``(N,)``.
        padding (`Tensor`): Padding applied to the input. Shape: ``(N,)``.
        dilation (`Tensor`): Spacing between kernel elements. Shape: ``(N,)``.

    Returns:
        :class:`Tensor`: Output spatial dimensions. Shape: ``(N,)`` matching 
            the input size tensor shape.

    Note:
        This function implements the standard convolution formula:
        
        .. math::
            \\text{output} = \\left\\lfloor \\frac{\\text{input} + 2 \\times \\text{padding} - \\text{dilation} \\times (\\text{kernel} - 1) - 1}{\\text{stride}} \\right\\rfloor + 1

    Example:
        Calculate output size for a 2D convolution::

            >>> import torch
            >>> output = conv_size(
            ...     size=torch.tensor([224, 224]),
            ...     kernel_size=torch.tensor([3, 3]),
            ...     stride=torch.tensor([2, 2]),
            ...     padding=torch.tensor([1, 1]),
            ...     dilation=torch.tensor([1, 1])
            ... )
            >>> output
            tensor([112, 112])
    """
    return torch.floor_divide(
        size + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride
    ) + 1
```

## チェックリスト

- [ ] Brief descriptionは1行に収まっているか？
- [ ] Args内の型は `` `Type` `` でマークされているか？
- [ ] ReturnsとRaisesは `:class:`、`:exc:` などのロールを使用しているか？
- [ ] 本文中のクラス名、メソッド名、関数名に適切なロールを適用しているか？
- [ ] `None`、`True`、`False` は `:obj:` ロールでマークされているか？
- [ ] コード例は `::` で始まり、適切にインデントされているか？
- [ ] 数式は `.. math::` ディレクティブを使用しているか？
- [ ] セクションの順序は正しいか？
- [ ] 外部パッケージの参照（torch, numpy等）も適切にマークアップされているか？

## よくある間違いと修正例

### ❌ 間違い

```python
"""
Args:
    param (Tensor): Input tensor. Returns None if invalid.

Returns:
    Tensor: Processed tensor using torch.nn.Module.

Raises:
    ValueError: When input is None.
"""
```

### ✅ 正しい

```python
"""
Args:
    param (`Tensor`): Input tensor. Returns :obj:`None` if invalid.

Returns:
    :class:`Tensor`: Processed tensor using :class:`torch.nn.Module`.

Raises:
    :exc:`ValueError`: When input is :obj:`None`.
"""
```

## クイックリファレンス

| 対象 | 記述方法 | 例 |
|------|---------|-----|
| クラス名 | `:class:`Name`` | `:class:`torch.Tensor`` |
| 関数名 | `:func:`name`` | `:func:`validate_input`` |
| メソッド名 | `:meth:`name`` | `:meth:`forward`` |
| 例外 | `:exc:`Name`` | `:exc:`ValueError`` |
| None/True/False | `:obj:`value`` | `:obj:`None`` |
| 属性 | `:attr:`name`` | `:attr:`shape`` |
| モジュール | `:mod:`path`` | `:mod:`torch.nn`` |
| Args内の型 | `` `Type` `` | `` `Tensor` `` |

## 参考資料

- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Sphinx Documentation - Domains](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html)
- [Napoleon - Google Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
