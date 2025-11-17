# Git WIP Status Checker

## Quick Status Check

```powershell
# Check current WIP status
git config --local --get-regexp 'user.wip'

# Check all worktrees status
git worktree list

# View work status dashboard
cat .work\STATUS.md
```

## Setting WIP Status

```powershell
# Mark current branch as WIP
git config --local user.wip "$(git branch --show-current)"
git config --local user.wip.workspace "$(Split-Path -Leaf (Get-Location))"
git config --local user.wip.started "$(Get-Date -Format 'yyyy-MM-dd HH:mm')"

# Clear WIP status
git config --local --unset user.wip
git config --local --unset user.wip.workspace
git config --local --unset user.wip.started
```

## Automated WIP Commit

```powershell
# Create a WIP commit (can be amended later)
git add .
git commit -m "WIP: Work in progress - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

# Continue work later and amend
git commit --amend
```

## Check Other Workspaces

```powershell
# Check each worktree's WIP status
cd ..\cslrtools2-utilities
git config --local --get-regexp 'user.wip'

cd ..\cslrtools2-dependencies
git config --local --get-regexp 'user.wip'

cd ..\cslrtools2-gitignore
git config --local --get-regexp 'user.wip'
```
