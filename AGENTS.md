# Project Instructions

- Before any CVS-RFFI / CV-SincNet research-scenario, data-protocol, experiment-matrix, optimization-route, paper/report-framing, or Stage2-A/B/C interpretation work, read `docs/PROJECT_PROTOCOL.md`. Treat it as the public source of truth for CVS scientific scenario, data protocol, receiver/TX split, weak-label/semi-supervised DG framing, Stage2-A/B/C boundaries, metric claims, and allowed research narrative.
- Project-related changes must use a Git-backed workflow. Before editing, run `git status -sb`; after editing, inspect `git diff`/`git status -sb`, run the narrowest useful verification, and commit the intended change unless the user explicitly says not to commit.
- Project-relevant Markdown must stay synchronized with code/config/script/protocol changes. Update this file for workflow, Git, collaboration, or safety rules; update `docs/PROJECT_PROTOCOL.md` before changing CVS scientific/data-protocol/Stage2/metric-claim semantics; update README/docs when user-facing usage, reproduction scope, or publication boundaries change.
- Do not commit datasets, trained weights, checkpoints, private runtime details, logs, generated experiment outputs, or local machine state. Keep `.gitignore` aligned with this boundary.
- Do not claim deployment success from clean-view or diagnostic-only evidence. Any reported result must be tied to a concrete run, split, K-shot setting, satellite/LEO view, and same-row metric context.

