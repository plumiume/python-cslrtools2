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

Worktree: integration-ai (integration for gitignore & docker)

Purpose
- Provide focused instructions for testing gitignore and Docker-related changes in this workspace.

Goal
- Verify Docker-based workflows and ensure .gitignore changes do not omit essential files.

Tasks for the agent
1. Run lightweight container build checks if Docker is available, otherwise verify Dockerfile syntax.
2. Scan `.gitignore` changes and report any risky omissions (e.g., committing `.venv` or build artifacts).
3. Run unit tests affected by Docker-related utilities and report failures.
4. Propose corrective patches or .gitignore fine-tuning entries.

Constraints
- Do not push large binary artifacts as part of fixes.
- If Docker cannot run in the environment, produce reproducible checks that do not require container execution.

Outputs expected
- A short report on Dockerfile health and `.gitignore` recommendations.
- PR-ready patches for small fixes.
