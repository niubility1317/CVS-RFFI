# Class-Incremental Wireless Device Identification Reproduction Plan

## Goal

Reproduce `Class-Incremental Learning for Wireless Device Identification in IoT` in a paper-faithful layer, then add a separate CVS-aligned extension only where it does not conflict with `PROJECT_PROTOCOL.md`/`项目.md`.

## Current Status

| Phase | Status | Notes |
|---|---|---|
| 0. Control files and Git state | complete | Read `AGENTS.md` and `项目.md`; root workspace is not Git; release repo has existing unrelated dirty files. |
| 1. PDF evidence extraction | complete | Extracted structured PDF evidence and received paper mechanism subagent summary. |
| 2. Multi-agent cross-check | complete | Four read-only agents covered paper mechanism, repo mapping, protocol design, and paper-only audit checklist. |
| 3. Reproduction design | complete | Created paper-faithful CSIL module boundary, config, checklist, and missing-detail ledger. |
| 4. Implementation | complete | Added isolated CSIL protocol/model/loss/metrics/dry-run code and tests; fixed post-audit bias mask gap. |
| 5. Local verification | complete | Focused CSIL test file passes under `ssr-gpu`; compileall and formal dry-run pass. |
| 6. Paper-vs-work audit | complete | Paper checklist added; post-implementation subagent audit completed with remaining data/experiment gaps documented. |

## Boundaries

- Paper-faithful reproduction must not be claimed as CVS deployment success.
- CVS extension must live separately from the paper-faithful reproduction and carry explicit protocol fields.
- Unknown/open-set rejection is Phase3 backup under the current CVS protocol unless the paper layer explicitly evaluates it as part of its own setting.
- No remote launch in this phase unless the N607 preflight, local report, sync mapping, and occupancy checks are completed.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| PowerShell variable expansion removed `$p`/`$d` in nested commands | Initial local file/Git checks | Re-ran commands with single-quoted `-Command` script blocks. |
| `E:\type10-7` root is not a Git repository | `git status -sb` at workspace root | Use `E:\type10-7\github_publish\CVS-RFFI-repo` as Git-backed working surface. |
| Python PDF extraction printed non-ASCII to GBK console | Initial `pdfplumber` extraction | Re-ran with `PYTHONIOENCODING=utf-8`; extraction artifact written successfully. |
| `conda activate ssr-gpu` did not switch from base Python in nested shell | First test command | Used serial `conda run -n ssr-gpu` for local tests. |
| Parallel `conda run` triggered `__conda_tmp_*.txt` lock noise | Mistaken parallel verification call | Treated as wrapper noise per `AGENTS.md` and reran compile serially; serial command passed. |
