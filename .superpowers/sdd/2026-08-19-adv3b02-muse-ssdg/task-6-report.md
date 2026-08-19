# Task 6 Report — deterministic satellite student, telemetry, and checkpoint recovery

Status: COMPLETE

RED: `test_muse_ssdg_satellite.py` failed for absent stable selection/key APIs; `test_muse_ssdg_checkpoint.py` failed for absent MUSE restore API.
GREEN: the two new files pass (4 tests); stable selection hashes UTF-8 `seed|epoch|rx|day|eq|sig|base_index` with the first 8 SHA-256 bytes.
Coverage: `pytest code/tests/test_muse_ssdg_*.py code/tests/test_post_stage_trainers.py -q` passed 94 tests; `py_compile` and `git diff --check` passed.
Adjacent telemetry: the brief-named `code/tests/test_ssdg_telemetry.py` is absent in this worktree, so its exact command is `FAILED` only for missing path; existing telemetry coverage ran in `test_post_stage_trainers.py`.
Checkpoint: training heads, temporal-memory streaks, classification-prototype counts/state, and schedule round-trip; deployment `model` remains separate from `muse_training_heads`.
Commit: `e847c7d6` (`feat: close MUSE satellite and checkpoint state`).
Push/OID: VERIFIED; local and `origin/codex/adv3b02-muse-ssdg-20260819` both resolve to `e847c7d62506bfb12bed94f313a93096bb2db50f`.
Self-review: only Task 6 implementation/tests changed; no Task 1–5 edits were reverted; no U_s TX truth was introduced.
