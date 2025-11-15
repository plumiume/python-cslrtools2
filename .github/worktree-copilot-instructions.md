# Worktree Copilot Instructions — tests-enhancement

Worktree: dev-ai/tests-enhancement

Purpose
- This file describes the tasks the code agent should perform within the `tests-enhancement` worktree.

Goal
- Improve and validate the test suite related to dataset and lmpipe functionality. Ensure tests run reliably across worktrees and CI.

Tasks for the agent
1. Run the full test suite using `uv run pytest -q` and collect failures. Produce a brief summary with failing tests and stack traces.
2. If tests fail, attempt minimal fixes or produce PR-style patches that:
   - Make tests deterministic (mock external dependencies like MediaPipe where possible)
   - Fix small logic bugs surfaced by tests
   - Add or update unit tests that cover the regression
3. Run lint/typecheck (if configured) and report any critical issues.
4. Add `# pyright: ignore[reportUnusedImport]` comments to imports that provide effects by import alone (e.g., `import pytest`). These imports trigger Pyright's `reportUnusedImport` warning but are necessary for test framework registration or side effects. Use this comment to suppress false positives.
5. Prepare a short changelog entry describing what was changed and why (for PR body).

Constraints & Environment
- Use `uv` for all Python execution (e.g., `uv run pytest`, `uv run python`) to respect workspace virtual envs.
- Do not modify release-related files (e.g., `pyproject.toml` versions) unless explicitly required and documented in the PR description.
- Avoid adding secrets or machine-specific paths in commits.

Outputs expected from agent
- A small list of failing tests (if any) with one-line summary and failing assertion/trace snippet.
- A set of proposed git patches (or commits) with clear commit messages following Conventional Commits.
- Updated or new tests (pytest) that reproduce and validate the fix.
- Pyright ignore comments added to imports that are used for side effects only.
- A suggested PR title and body.

Try it (examples)
```powershell
# run tests
uv run pytest -q

# run a single failing test (example)
uv run pytest tests/src/test_dataset.py::test_loader -q

# check for unused import warnings
uv run pyright tests/
```

Notes
- The worktree is configured with an independent `.venv`. Ensure `uv sync` has been run if dependencies are out of date.
- For imports like `import pytest` that register fixtures or enable test discovery, use `# pyright: ignore[reportUnusedImport]` on the same line.
