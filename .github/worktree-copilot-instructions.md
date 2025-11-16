# Worktree Copilot Instructions — dataset-enhancement

Worktree: dev-ai/dataset-enhancement

Purpose
- Describe dataset creation/validation tasks for agents working in this worktree.

Goal
- Produce reproducible landmark dataset slices and validate loader compatibility.

Tasks for the agent
1. Run relevant dataset tests: `uv run pytest tests/src/test_dataset.py -q` and collect failures.
2. Run small sample extraction using included example videos and save outputs to a local `./out/` folder.
3. Validate `SLDataset` read/write compatibility (zarr layout) and report any schema mismatches.
4. Propose small fixes and test patches where failures occur.

Constraints
- Prefer lightweight operations (avoid full MediaPipe heavy models; use stubs/mocks where practical).
- Use `uv` for execution and keep outputs under `./out/` inside the worktree.

Outputs expected
- Test summary and failing test list.
- Small sample output files and a short validation report.

# Worktree Copilot Instructions — main-ai

Worktree: main-ai

Purpose
- Provide integration-level instructions for agents operating in the `main-ai` workspace.

Goal
- Verify integration of AI-driven features into `main` and prepare release-ready checks.

Tasks for the agent
1. Run the integration test suite: `uv run pytest -q` and collect results.
2. Run static checks (type checks / lint) and summarize critical issues.
3. Generate a short report listing merged `dev-ai/*` branches and any outstanding CI failures.
4. If regressions are found, produce PR-ready patches targeting the originating `dev-ai/*` branches.

Constraints
- Use `uv` for all Python execution.
- Do not alter release version numbers without explicit approval.

Outputs expected
- Integration test summary and failing test list (if any).
- A short PR description template when a fix is prepared.
