# Merge Integration Workspace - Code Agent Instructions

## 🎯 Workspace Context

**Branch**: `dev-ai/merge-integration`  
**Location**: `C:\Users\ikeda\Workspace\1github\cslrtools2-merge`  
**Purpose**: Integration of multiple dev-ai branches, comprehensive testing, and main merge preparation

## 📋 Current State

### Completed Integrations
- ✅ `dev-ai/utilities-expansion`: +11,229 lines (Apache licenses, exceptions, utilities)
- ✅ `dev-ai/dependencies-update`: Already included in utilities
- ✅ `dev-ai/gitignore-cleanup`: Already included in utilities
- ✅ `tests/` directory: Core import tests (6 passing, 1 skipped)

### Environment
- ✅ Python 3.12.11 + PyTorch 2.9.1+cu128
- ✅ 43 packages installed via `uv sync`
- ✅ Clean test results (6/6 core tests passing)

### Pending Integrations
**Branches to merge**:
1. `dev-ai/dataset-enhancement` (when ready)
2. Any new feature branches from dev-ai/*

## 🚀 Priority Tasks (in order)

### 1. Monitor Dataset Enhancement Progress

**Watch for**: `dev-ai/dataset-enhancement` completion signals
- All tests passing in dataset workspace
- CLI commands functional
- Documentation updated

**Integration command** (when ready):
```powershell
git fetch origin
git merge dev-ai/dataset-enhancement --no-edit
```

### 2. Run Comprehensive Test Suite

After each integration:
```powershell
# Run all tests with verbose output
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=cslrtools2 --cov-report=html --cov-report=term

# Check coverage threshold (target: 80%+)
uv run pytest tests/ --cov=cslrtools2 --cov-fail-under=80
```

**Expected test structure after dataset merge**:
```
tests/
├── __init__.py
├── conftest.py
├── test_core_imports.py (6 tests)
├── test_mediapipe_constants.py (5 tests, skipped if no mediapipe)
├── test_sldataset.py (new, 8+ tests)
├── test_array_loader.py (new, 6+ tests)
└── test_cli.py (optional, CLI integration tests)
```

### 3. Fix Import Issues

**Known issues to resolve**:

1. **SLDataset export** (if not fixed in dataset branch):
```python
# File: src/cslrtools2/sldataset/__init__.py
from .dataset import SLDataset

__all__ = ["SLDataset"]
```

2. **Logger naming** (if exists):
```python
# File: src/cslrtools2/logger.py
# Ensure exported name matches usage
from .lmpipe.logger import root_logger as get_logger
```

### 4. Validate Type Safety

```powershell
# Run Pyright type checker
uv run pyright src/

# Expected: 0 errors in strict mode
```

**Common type issues to fix**:
- Missing return type annotations
- Generic type parameter mismatches
- Optional type handling in Zarr operations

### 5. Update Integration Documentation

**File**: `INTEGRATION_LOG.md` (create if needed)

Content:
```markdown
# Integration Log - dev-ai/merge-integration

## Merged Branches

### utilities-expansion (2025-11-14)
- **Commit**: 39f81f4
- **Changes**: +11,229 lines
- **Features**: Apache licenses, exceptions, core utilities
- **Tests**: All passing

### dataset-enhancement (TBD)
- **Commit**: [hash]
- **Changes**: [lines]
- **Features**: SLDataset tests, CLI commands, documentation
- **Tests**: [status]

## Test Summary
- Total tests: [count]
- Passing: [count]
- Coverage: [percentage]%

## Known Issues
- [list any remaining issues]

## Ready for Main Merge
- [ ] All tests passing
- [ ] Coverage ≥ 80%
- [ ] No type errors
- [ ] Documentation complete
```

### 6. Prepare Main Merge

**Pre-merge checklist**:
```powershell
# 1. Ensure clean working tree
git status

# 2. Run full test suite
uv run pytest tests/ -v --cov=cslrtools2

# 3. Type check
uv run pyright src/

# 4. Review commit history
git log --oneline main..HEAD

# 5. Squash commits if needed
git rebase -i main
```

**Main merge commands**:
```powershell
# Switch to main
git checkout main
git pull origin main

# Squash merge integration branch
git merge --squash dev-ai/merge-integration

# Create consolidated commit
git commit -m "feat: Integrate dataset enhancements, tests, and utilities

- Add comprehensive test suite (tests/ directory)
- Implement SLDataset functionality and CLI commands
- Add Apache 2.0 licenses and exception handling
- Update dependencies and improve type safety
- Add documentation and usage examples

Merged branches:
- dev-ai/utilities-expansion
- dev-ai/dependencies-update
- dev-ai/gitignore-cleanup
- dev-ai/dataset-enhancement

BREAKING CHANGE: None
Closes: #[issue-numbers]"

# Push to main
git push origin main
```

### 7. Cleanup After Merge

```powershell
# Delete merged remote branches
git push origin --delete dev-ai/utilities-expansion
git push origin --delete dev-ai/dependencies-update
git push origin --delete dev-ai/gitignore-cleanup
git push origin --delete dev-ai/dataset-enhancement

# Keep merge-integration as integration workspace
# (don't delete until all future merges are complete)

# Update CHANGELOG.md
# Add entry for the integrated release
```

## 🔧 Development Workflow

### Testing Workflow

```powershell
# Quick test (core only)
uv run pytest tests/test_core_imports.py -v

# Full suite
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=cslrtools2 --cov-report=html

# Specific test
uv run pytest tests/test_sldataset.py::test_create_dataset -v

# Failed tests only (on re-run)
uv run pytest tests/ --lf -v
```

### Integration Testing

After merging a new branch:
```powershell
# 1. Merge
git merge dev-ai/<branch-name> --no-edit

# 2. Install any new dependencies
uv sync --all-groups

# 3. Run tests
uv run pytest tests/ -v

# 4. Check types
uv run pyright src/

# 5. If issues found
git status  # Check conflicts
git diff    # Review changes

# 6. Commit resolution
git add .
git commit -m "fix: Resolve integration issues from <branch-name>"
```

### Rollback Procedure

If integration fails:
```powershell
# Abort merge
git merge --abort

# OR reset to before merge
git reset --hard HEAD~1

# Re-evaluate merge strategy
git log --graph --oneline --all
```

## 📊 Quality Gates

### Before Main Merge

All of these must be **true**:

1. **Tests**
   - ✅ All pytest tests passing
   - ✅ Test coverage ≥ 80%
   - ✅ No skipped tests (except MediaPipe when not installed)

2. **Type Safety**
   - ✅ Pyright reports 0 errors
   - ✅ All public APIs have type annotations

3. **Code Quality**
   - ✅ No TODOs in critical paths
   - ✅ All functions have docstrings
   - ✅ Consistent code style

4. **Documentation**
   - ✅ README.md updated with new features
   - ✅ CHANGELOG.md has release entry
   - ✅ CLI help text accurate

5. **Git Hygiene**
   - ✅ No merge conflicts
   - ✅ Clean commit history (squashed)
   - ✅ Meaningful commit message

## 🚨 Troubleshooting

### Test Failures

**Symptom**: `ImportError: cannot import name 'X'`
```powershell
# Check __init__.py exports
cat src/cslrtools2/<module>/__init__.py

# Verify installation
uv run python -c "from cslrtools2.<module> import X; print(X)"
```

**Symptom**: `ModuleNotFoundError: No module named 'mediapipe'`
```powershell
# MediaPipe tests should be skipped, check marks
uv run pytest tests/test_mediapipe_constants.py -v

# Should show: SKIPPED (MediaPipe not installed)
```

### Type Errors

**Symptom**: Pyright reports type errors
```powershell
# Run with verbose output
uv run pyright src/ --verbose

# Check specific file
uv run pyright src/cslrtools2/sldataset/dataset.py
```

### Merge Conflicts

```powershell
# View conflicts
git status
git diff

# Use merge tool
git mergetool

# Manual resolution
# Edit files, then:
git add <resolved-files>
git commit
```

## 📈 Success Metrics

### Test Coverage Targets

- **Overall**: ≥ 80%
- **sldataset/**: ≥ 85%
- **lmpipe/**: ≥ 75%
- **plugins/**: ≥ 70% (MediaPipe optional)

### Performance Benchmarks

After integration, verify:
```powershell
# Dataset creation speed
uv run python -c "
from cslrtools2.sldataset.dataset import SLDataset
import time
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    start = time.time()
    ds = SLDataset.create(f'{tmpdir}/test.zarr', {})
    print(f'Creation time: {time.time() - start:.3f}s')
"

# Landmark extraction speed (if MediaPipe available)
# uv run lmpipe mediapipe.holistic test_video.mp4 -o output.zarr
```

## 🎯 Next Steps After Main Merge

1. **Tag Release**
```powershell
git tag -a v0.1.0 -m "Release v0.1.0: Dataset enhancements and test infrastructure"
git push origin v0.1.0
```

2. **Update GitHub**
   - Create release notes from CHANGELOG.md
   - Close associated issues
   - Update project board

3. **PyPI Preparation** (future)
   - Verify `pyproject.toml` metadata
   - Test build: `uv build`
   - Publish: `uv publish`

4. **Documentation Deployment** (future)
   - Build Sphinx docs
   - Deploy to GitHub Pages
   - Update API reference

## 🔄 Continuous Integration Workflow

This workspace follows the pattern:
```
dev-ai/* branches → dev-ai/merge-integration → main
```

**Process**:
1. Feature work happens in `dev-ai/<feature>` branches
2. Completed features merge to `dev-ai/merge-integration`
3. Integration testing happens here
4. Squash merge to `main` when stable
5. Delete feature branches, keep integration branch

**Benefits**:
- Isolated feature development
- Comprehensive integration testing
- Clean main branch history
- Easy rollback if needed

---

**Last Updated**: 2025-11-14  
**Agent Mode**: Integration & Quality Assurance  
**Python Version**: 3.12.11  
**Package Manager**: uv (with `uv.lock`)

## 📞 Communication with Other Workspaces

### Dataset Workspace Ready Signal

When `cslrtools2-dataset` is ready:
```powershell
# Dataset workspace will push
git push origin dev-ai/dataset-enhancement

# This workspace should:
git fetch origin
git merge dev-ai/dataset-enhancement
uv sync --all-groups
uv run pytest tests/ -v
```

### Main Branch Sync

Before merging to main:
```powershell
# Sync with latest main
git fetch origin main
git rebase origin/main

# Resolve any conflicts
# Re-test after rebase
```
