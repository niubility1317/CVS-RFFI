# Git Maintenance

This workspace is configured as a Git repository for `C:\Users\lh594\Desktop\CVS-RFFI`.

## What Is Tracked

Tracked by default:
- root-level source files
- `tests/`
- `scripts/`
- `docs/`

Ignored by default:
- datasets, logs, checkpoints, and model weights
- Python caches and local environments
- child experiment snapshots: `type10-4/`, `type10-6-sat/`, `type10-7/`, `unkown/`

## Auto-Save Scripts

Manual one-shot save:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\git_autosave.ps1
```

Manual one-shot save and push:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\git_autosave.ps1 -Push
```

Start a file watcher for automatic commits:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\start_git_autosave.ps1
```

Start a file watcher that also pushes when an `origin` remote exists:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\start_git_autosave.ps1 -Push
```

## Remote Push Setup

Auto-push needs a configured remote and working credentials. Add a remote with:

```powershell
git remote add origin <your-repository-url>
git push -u origin main
```

After that, `git_autosave.ps1 -Push` and the watcher `-Push` mode will push committed changes automatically.

## Logs

The scripts write logs under:

```text
logs/git_autosave.log
logs/git_autosave_watcher.log
```
