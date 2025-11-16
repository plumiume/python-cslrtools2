# Worktree Copilot Instructions — integrate-gitignore-docker

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
