# Pyright Style Guide

**Version:** 1.0.0  
**Last Updated:** 2025-11-16  
**Applies to:** All Python files in `src/cslrtools2/`

> **関連ドキュメント**: このガイドを使用する際は、以下のドキュメントも併せて参照してください。
> - [コーディングスタイルガイド](CODING_STYLE_GUIDE.md) - 型ヒント記述規約
> - [Docstringスタイルガイド](DOCSTRING_STYLE_GUIDE.md) - 型情報のドキュメント化
> - [例外処理・ログスタイルガイド](EXCEPTION_LOGGING_STYLE_GUIDE.md) - 型安全なエラーハンドリング

## Table of Contents

1. [Overview](#overview)
2. [Type Ignore Comments](#type-ignore-comments)
3. [Common Diagnostic Rules](#common-diagnostic-rules)
4. [Error Code Reference](#error-code-reference)
5. [Suppression Guidelines](#suppression-guidelines)
6. [Configuration](#configuration)
7. [Resources](#resources)

---

## Overview

This guide establishes standards for **Pyright/Pylance-specific type checking** in the cslrtools2 project. We use Pyright in **strict mode** for maximum type safety, and all type checking suppressions must follow the conventions outlined here.

### Core Principles

1. **Pyright-specific comments**: Always use `# pyright: ignore[ErrorCode]` instead of generic `# type: ignore`
2. **Document suppressions**: Every suppression MUST include a reason comment explaining why it's necessary
3. **Prefer fixing over suppressing**: Only suppress when there's a genuine limitation in the type system
4. **Use specific error codes**: Never use bare `# pyright: ignore` without error codes

---

## Type Ignore Comments

### ✅ Required Format

```python
# pyright: ignore[ErrorCode]
# Reason: Brief explanation of why suppression is necessary
```

### ❌ Forbidden Formats

```python
# type: ignore                           # ❌ Generic, no error code
# pyright: ignore                        # ❌ No error code
# type: ignore[ErrorCode]                # ❌ Not Pyright-specific
```

### Common Patterns

#### Pattern 1: Optional Dependency Attribute Access

```python
if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = None

# Later in code:
# pyright: ignore[reportAttributeAccessIssue]
# Reason: Matplotlib is optional dependency, runtime isinstance check ensures availability
fig.canvas.draw()
```

#### Pattern 2: Overload Implementation Constraints

```python
@overload
def calculate(x: int) -> int: ...
@overload
def calculate(x: float) -> float: ...

def calculate(
    # pyright: ignore[reportInconsistentOverload]
    # Reason: Cannot express "x is int XOR x is float" constraint in Python type system.
    # Overloads guarantee correct usage, runtime isinstance checks handle validation.
    x: int | float
) -> int | float:
    ...
```

#### Pattern 3: Dynamic Type Resolution

```python
# pyright: ignore[reportUnknownMemberType]
# Reason: Type dynamically determined at runtime via plugin system
result = plugin.process(data)
```

---

## Common Diagnostic Rules

### High Priority (Must Fix or Document)

| Error Code | Description | Common Cause |
|------------|-------------|--------------|
| `reportAttributeAccessIssue` | Accessing attribute that may not exist | Missing type stubs, optional dependencies |
| `reportInconsistentOverload` | Overload signatures overlap or mismatch | Implementation doesn't match overload declarations |
| `reportGeneralTypeIssues` | Generic type safety violations | Missing annotations, incorrect types |
| `reportCallIssue` | Function call with wrong arguments | Signature mismatch |
| `reportArgumentType` | Argument type doesn't match parameter | Type incompatibility |

### Medium Priority (Document if Suppressed)

| Error Code | Description | Common Cause |
|------------|-------------|--------------|
| `reportUnknownMemberType` | Member type cannot be determined | Dynamic attributes, missing stubs |
| `reportUnknownVariableType` | Variable type unknown | Inference failure |
| `reportOptionalMemberAccess` | Accessing member on optional value | Need None check first |
| `reportMissingTypeStubs` | Library lacks type stubs | Third-party package without py.typed |

### Low Priority (Consider Fixing)

| Error Code | Description | Common Cause |
|------------|-------------|--------------|
| `reportUnusedImport` | Import not used | Dead code |
| `reportUnusedVariable` | Variable assigned but not used | Dead code |
| `reportUnnecessaryTypeIgnoreComment` | Ignore comment no longer needed | Code was fixed |

---

## Error Code Reference

### reportAttributeAccessIssue

**Description**: Warns when code tries to access an attribute that may not exist or is not recognized by the type checker.

**Common Scenarios**:
- Optional dependencies (matplotlib, cv2)
- Dynamic attributes
- Missing type stubs

**Fix Strategies**:
1. Add explicit type annotations
2. Initialize all instance variables in `__init__`
3. Use type guards or runtime checks
4. Create/improve type stubs for libraries

**Example**:
```python
# Problem
obj.dynamic_attr  # Pyright doesn't know about this

# Solution 1: Type annotation
obj: MyClass  # where MyClass declares dynamic_attr
obj.dynamic_attr

# Solution 2: Suppression (if truly dynamic)
# pyright: ignore[reportAttributeAccessIssue]
# Reason: Attribute added dynamically by plugin system at runtime
obj.dynamic_attr
```

**References**:
- [Pylance Documentation](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportAttributeAccessIssue.md)
- [Pyright Configuration](https://github.com/microsoft/pyright/blob/main/docs/configuration.md#reportAttributeAccessIssue)

---

### reportInconsistentOverload

**Description**: Warns when function overloads are inconsistent—overlapping signatures or implementation mismatches.

**Common Scenarios**:
- Implementation signature doesn't match any overload
- Overload signatures overlap
- Type constraints impossible to express

**Fix Strategies**:
1. Ensure implementation matches at least one overload
2. Make overload signatures mutually exclusive
3. Use precise, non-overlapping type annotations
4. Suppress with detailed reason if type system limitation

**Example**:
```python
from typing import overload

@overload
def process(data: list[int]) -> int: ...
@overload
def process(data: list[str]) -> str: ...

def process(
    # pyright: ignore[reportInconsistentOverload]
    # Reason: Cannot express "T is int XOR T is str" constraint.
    # Overloads guarantee correct usage at call sites.
    data: list[int] | list[str]
) -> int | str:
    if isinstance(data[0], int):
        return sum(data)  # type: ignore
    return "".join(data)
```

**References**:
- [Pylance Documentation](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportInconsistentOverload.md)
- [Pyright Configuration](https://github.com/microsoft/pyright/blob/main/docs/configuration.md#reportInconsistentOverload)

---

### reportUnknownMemberType

**Description**: Warns when the type of a class or object member cannot be determined.

**Common Scenarios**:
- Missing type annotations on members
- Dynamic member assignment
- Third-party libraries without type stubs

**Fix Strategies**:
1. Add explicit type annotations to members
2. Use/create type stubs for libraries
3. Refactor to clarify member types
4. Suppress if intentionally dynamic

**Example**:
```python
# Problem
class MyClass:
    def __init__(self):
        self.member = get_value()  # Type unknown

# Solution 1: Explicit annotation
class MyClass:
    member: int
    
    def __init__(self):
        self.member = get_value()

# Solution 2: Inline annotation
class MyClass:
    def __init__(self):
        self.member: int = get_value()
```

**References**:
- [Pylance Documentation](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportUnknownMemberType.md)
- [Pyright Configuration](https://github.com/microsoft/pyright/blob/main/docs/configuration.md#reportUnknownMemberType)

---

## Suppression Guidelines

### When to Suppress

✅ **Valid Reasons**:
- Type system cannot express the constraint (e.g., XOR type conditions)
- Optional dependency with runtime checks
- Intentional dynamic behavior (plugin systems)
- Third-party library limitation (missing/incomplete stubs)
- Performance-critical code where type checking adds overhead

❌ **Invalid Reasons**:
- "I don't want to fix the type error"
- "Too lazy to add annotations"
- "It works at runtime"
- "Tests pass"

### Suppression Template

```python
# pyright: ignore[ErrorCode]
# Reason: [One of the following]
#   - Cannot express [constraint] in Python type system
#   - [Library] is optional dependency, runtime check ensures availability
#   - Attribute added dynamically by [system] at runtime
#   - [Library] lacks type stubs, issue tracked at [link]
#   - Performance optimization, types verified by [test/other means]
```

### Examples from Codebase

#### convsize.py (Overload Constraint)
```python
def forward(
    # pyright: ignore[reportInconsistentOverload]
    # Reason: Cannot express "arg is Tensor XOR arg is Size" constraint in Python type system.
    # Overloads guarantee correct usage, runtime isinstance checks handle validation.
    self,
    arg: Tensor | Size,
    dim: int | Sequence[int] | slice = slice(None)
):
```

#### matplotlib_af.py (Optional Dependency)
```python
if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = None

# Later:
# pyright: ignore[reportAttributeAccessIssue]
# Reason: Matplotlib is optional dependency, runtime availability checked before use
ax.set_xlim(0, width)
```

---

## Configuration

### pyproject.toml Settings

```toml
[tool.pyright]
# Strict mode enabled
typeCheckingMode = "strict"

# Custom severity overrides (if needed)
reportAttributeAccessIssue = "error"
reportInconsistentOverload = "error"
reportUnknownMemberType = "warning"
reportMissingTypeStubs = "warning"

# Disable specific checks (use sparingly)
# reportUnusedImport = "none"
```

### Per-File Configuration

```python
# pyright: strict, reportAttributeAccessIssue=false

"""Module with custom Pyright settings."""
```

### Inline Configuration

```python
def legacy_function():
    # pyright: reportGeneralTypeIssues=false
    result = some_untyped_call()
    return result
```

---

## Resources

### Official Documentation

- **Pyright GitHub**: https://github.com/microsoft/pyright
- **Pylance Release**: https://github.com/microsoft/pylance-release
- **Diagnostic Rules Directory**: https://github.com/microsoft/pylance-release/tree/main/docs/diagnostics
- **Configuration Reference**: https://github.com/microsoft/pyright/blob/main/docs/configuration.md

### Diagnostic Rules (Selected)

All 62+ diagnostic rules are documented at:  
https://github.com/microsoft/pylance-release/tree/main/docs/diagnostics

**Key Rules**:
- [reportAbstractUsage.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportAbstractUsage.md)
- [reportArgumentType.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportArgumentType.md)
- [reportAttributeAccessIssue.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportAttributeAccessIssue.md) ⭐
- [reportCallIssue.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportCallIssue.md)
- [reportGeneralTypeIssues.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportGeneralTypeIssues.md)
- [reportInconsistentOverload.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportInconsistentOverload.md) ⭐
- [reportMissingImports.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportMissingImports.md)
- [reportMissingTypeStubs.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportMissingTypeStubs.md)
- [reportOptionalMemberAccess.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportOptionalMemberAccess.md)
- [reportUnknownMemberType.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportUnknownMemberType.md) ⭐
- [reportUnusedImport.md](https://github.com/microsoft/pylance-release/blob/main/docs/diagnostics/reportUnusedImport.md)

⭐ = Frequently used in cslrtools2 codebase

### Related Project Guides

- `guides/CODING_STYLE_GUIDE.md` - Overall coding standards
- `guides/DOCSTRING_STYLE_GUIDE.md` - Documentation format
- `guides/EXCEPTION_LOGGING_STYLE_GUIDE.md` - Error handling

### Tools

- **VS Code Pylance Extension**: Provides real-time diagnostics
- **pyright CLI**: `uv run pyright` for batch checking
- **pytest with pyright plugin**: Type checking in CI/CD

---

## Changelog

### 1.0.0 (2025-11-16)
- Initial version based on cslrtools2 codebase analysis
- Documented 3 primary error codes used in project
- Established suppression comment format and reasoning requirements
- Added examples from actual codebase (`convsize.py`, `matplotlib_af.py`)
- Linked official Pylance/Pyright diagnostic documentation (62+ rules)
