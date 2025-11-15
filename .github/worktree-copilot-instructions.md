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
