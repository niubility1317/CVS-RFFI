# Local Workspace Cleanup 2026-06-26

## Scope

This note records a local cleanup of `E:\type10-7`. The workspace root is not a Git repository, so destructive local filesystem changes are not directly versioned. The Git-backed record for this cleanup is this Markdown file on branch `codex/cvs-rffi-release-20260626`.

## Deleted

- Python/runtime cache directories outside the Git release working tree: `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, and `.ipynb_checkpoints` where removable.
- Duplicate non-Git release staging directory: `E:\type10-7\github_publish\CVS-RFFI`.
- Temporary VS Code server payload directory: `E:\type10-7\tmp\vscode-server`.
- Zero-byte root temporary files: `E:\type10-7\$tmp` and `E:\type10-7\=25`.
- Ignored cache directories inside the Git release working tree using `git clean -fdX`.

Approximate reclaimed space from the non-Git cleanup pass: `237.05 MB`.

## Preserved

The cleanup intentionally did not delete datasets, checkpoints, logs, metrics, reports, run outputs, remote artifacts, server backups, PPT/DOCX/PDF materials, or analysis artifacts. Those categories require a more explicit deletion scope because project rules require preserving experiment evidence and run outputs by default.

## Remaining

Two empty cache directories remained because Windows returned permission/access errors:

- `E:\type10-7\.pytest_cache`
- `E:\type10-7\code\.pytest_cache`

Large remaining candidates that may be reviewed separately include:

- `E:\type10-7\code\snapshots\...`
- `E:\type10-7\server_log_backups\...`
- `E:\type10-7\automation_reports\...`
- `E:\type10-7\PPT\...`
- `E:\type10-7\RFFI少样本学习\...`
- `E:\type10-7\analysis_tmp\...`

These were not removed in this pass.

## Verification

- `E:\type10-7` root: `git status -sb` reports it is not a Git repository.
- `E:\type10-7\github_publish\CVS-RFFI-repo`: Git working tree clean after cleanup.
- Release staging directory `E:\type10-7\github_publish\CVS-RFFI` no longer exists.
- Temporary directory `E:\type10-7\tmp\vscode-server` no longer exists.
- Cache scan after cleanup found only the two empty permission-blocked `.pytest_cache` directories outside the Git working tree.
