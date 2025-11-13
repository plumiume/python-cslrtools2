# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `from __future__ import annotations` to modernize type hint system across the codebase
- Created CHANGELOG.md to track project changes

### Changed
- **Module Refactoring**: Split `lmpipe.interface` module into two files for better organization:
  - `interface/__init__.py`: Contains `LMPipeInterface` class (430 lines)
  - `interface/runner.py`: Contains `LMPipeRunner` class and utilities (1,252 lines)
- **Type System Modernization**:
  - Removed Python 3.14 workaround type redeclarations (`type X = "X"`)
  - Replaced explicit string literal type hints with `from __future__ import annotations`
  - Simplified forward references using automatic annotation evaluation
  - Files updated: `estimator.py`, `runspec.py`, `dataset.py`, `interface/__init__.py`, `interface/runner.py`

### Fixed
- Fixed circular import issues with `ProcessResult` (moved imports from `interface` to `estimator`)
- Corrected imports in collector modules: `matplotlib_af.py`, `pil_af.py`, `torchvision_af.py`
- Resolved `NameError` in runner.py by properly handling `LMPipeInterface` import with `TYPE_CHECKING`

### Technical Details
- Python 3.12 compatibility maintained with PEP 585+ generic syntax
- All 14 pytest tests passing
- Zero type errors from Pyright
- Cleaner, more maintainable code with modern Python type hints

---

## Version History

For older changes, please see the git commit history.
