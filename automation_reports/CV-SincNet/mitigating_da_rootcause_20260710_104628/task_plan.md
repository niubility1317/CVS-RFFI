# Task Plan: Mitigating receiver impact DA root-cause analysis

## Goal
Explain the large WiSig reproduction gap with paper-to-code and full-log evidence, implement only paper-aligned fixes, and validate them locally and on N607.

## Current Phase
Phase 4: in_progress

## Phases

### Phase 1: Evidence inventory and traceability
- [x] Extract every reproducibility requirement from the paper.
- [x] Inventory implementation, launchers, tests, reports, logs, checkpoints, and result artifacts.
- [x] Build the paper-to-code traceability matrix.

### Phase 2: Independent diagnosis
- [x] Complete paper-spec audit.
- [x] Complete algorithm/code audit.
- [x] Parse all relevant logs and structured artifacts in full.
- [x] Run a supervisor cross-check of the independent findings.

### Phase 3: Fix design and local implementation
- [x] Separate confirmed paper mismatches from diagnostic extensions and paper ambiguities.
- [x] Add failing focused tests for confirmed defects.
- [x] Implement the smallest paper-aligned changes locally.
- [x] Verify focused tests, syntax, CLI, and launcher dry-runs in `ssr-gpu`.

### Phase 4: N607 validation
- [x] Run direct preflight and inspect live GPU/process state.
- [x] Record report, Git state, sync map, commands, GPUs, PIDs, logs, and expected artifacts before launch.
- [x] Sync locally verified files and launch a bounded reproduction matrix without interfering with unrelated jobs.
- [x] Parse the first repaired matrix logs and metrics in full.
- [ ] Run and parse the architecture/Table III localization matrix.

### Phase 5: Final comparison and delivery
- [x] Compare each first-matrix same-run result with the corresponding paper row.
- [ ] Update traceability statuses and report root causes, residual risks, and strict-vs-diagnostic claim boundaries.
- [ ] Commit intended Git-backed changes without touching unrelated edits.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Full-history forked agents cannot set `agent_type` explicitly | 1 | Respawn with inherited agent type and no explicit model/effort |
| PDF extraction hit GBK ligature encoding error | 1 | Set `PYTHONIOENCODING=utf-8` and re-extract all pages |
| Broad `rg` scan hit an access-denied temp path | 1 | Restrict scans to explicit project/report directories |
| First focused test command stayed in base Conda env | 1 | Load `conda-hook.ps1`, activate `ssr-gpu`, and rerun |
| Initial multi-file patch included a wrong protocol context | 1 | Confirm no partial diff, then apply smaller patches |
| First repaired MINE path referenced removed local names | 1 | Use the refreshed `source_outputs/target_outputs` estimate logits; rerun 36 tests |
