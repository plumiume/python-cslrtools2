# Worktree Copilot Instructions — tests-enhancement

Worktree: dev-ai/tests-enhancement

## URGENT: Fix PyTorch Triton Namespace Conflict

### Current Issue (2025-11-17)

**Problem**: Full test suite execution fails with PyTorch Triton namespace error:

```
RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace triton
Previous registration was at torch/__init__.py:2700
```

**Test Results**:
- ✅ Unit tests only: `521 passed, 3 skipped`
- ✅ Integration tests only: `50 passed, 19 skipped`  
- ❌ Full suite together: `4 failed, 567 passed, 22 skipped`

**Failing Tests**:
1. `tests/integration/test_dataset_workflow.py::TestPyTorchDataLoaderIntegration::test_dataloader_basic_iteration`
2. `tests/integration/test_dataset_workflow.py::TestPyTorchDataLoaderIntegration::test_dataloader_with_shuffle`
3. `tests/integration/test_dataset_workflow.py::TestPyTorchDataLoaderIntegration::test_dataloader_batch_size_one`
4. `tests/integration/test_plugin_system.py::TestPluginDiscovery::test_mediapipe_plugins_available`

**Note**: Each test passes when run individually.

### Task: Fix Full Suite Execution

**Goal**: All 571 tests must pass when running:
```powershell
uv run pytest tests/unit/ tests/integration/ --tb=no
```

### Recommended Solution: pytest-xdist

Add test isolation using pytest-xdist:

1. **Add dependency** to `pyproject.toml`:
```toml
[project.optional-dependencies]
test = [
    "pytest-xdist>=3.5.0",
    # ... existing
]
```

2. **Configure pytest** in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = "--dist=loadfile"  # Isolate each test file
```

3. **Verify fix**:
```powershell
uv run pytest tests/ --tb=no
# Expected: 571 passed, 22 skipped, 0 failed
```

### Alternative Solutions (if xdist doesn''t work)

**Option A: Session Fixture**
```python
# tests/conftest.py
@pytest.fixture(scope="session", autouse=True)
def initialize_torch_once():
    import torch
    _ = torch.cuda.is_available()
    yield
```

**Option B: Test Markers**
```python
# Mark problematic tests
@pytest.mark.pytorch_triton
class TestPyTorchDataLoaderIntegration:
    ...
```

### Success Criteria

- ✅ `uv run pytest tests/ --tb=no` → 571+ passed, 0 failed
- ✅ No Triton namespace errors
- ✅ Test time < 20 seconds
- ✅ Works on Windows and Linux

---

## Previous Status (2025-11-16)

✅ **Phase 1 Complete**: Comprehensive test suite validation
- **609 tests passing, 19 skipped** (628 tests collected)
- **Coverage: 91%** (2434 statements, 229 missing)
- **0 Pyright errors or warnings** (38 files analyzed)
