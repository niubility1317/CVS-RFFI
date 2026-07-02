# phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024

## Objective

Continue the live ADV3B02 Phase2 goal after `phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004` failed to reach `old_acc>=80%` and `seen_new_acc>=65%`. This follow-up keeps actual unknown TX excluded and tests whether the stronger OLD80_FIRST head48F old-class recovery configuration can be combined with seen-new retention for multi-new-class enrollment.

## Delta From Previous Run

| Change | Reason |
|---|---|
| Added `old80_head_apply_policy=replace_all_except_seen_new_override` | Let OLD80 recover target-old rows while preserving rows already accepted by seen-new registration evidence |
| Reused head48F K10 old-class recovery settings | Historical ADV3B02 OLD80 diagnostic had the strongest old-class row at 77.50%, higher than the first multi-new run |
| Tested 2-new and 3-new scopes | User requires at least 2 new classes; 3-new remains a stress extension |
| Kept actual unknown TX excluded | The live goal explicitly does not evaluate unknown rejection |

## Candidate Plan

| Candidate | GPU | Target-new TX | K-old | K-new | Seen-new threshold mode | Goal |
|---|---:|---|---:|---:|---|---|
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_2NEW_STRICT` | 0 | `1-16,1-18` | 10 | 10 | strict | primary 2-new target |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_2NEW_RELAX` | 1 | `1-16,1-18` | 10 | 10 | relaxed | recover new if strict under-calls |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_3NEW_STRICT` | 2 | `1-16,1-18,1-14` | 10 | 10 | strict | 3-new stress target |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_3NEW_RELAX` | 3 | `1-16,1-18,1-14` | 10 | 10 | relaxed | 3-new relaxed stress target |

## Local Files

| File | Purpose |
|---|---|
| `E:\type10-7\code\cvsrffi\spaceborne_fewshot.py` | Adds the new OLD80 apply policy |
| `E:\type10-7\code\eval_spaceborne_fewshot.py` | Exposes the new policy in CLI choices |
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024.sh` | Four-candidate follow-up launcher |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024\report.md` | This report |

## Verification Plan

| Check | Status |
|---|---|
| Local Python compile | PASS for root and Git-backed mirror |
| Bash syntax | PASS for root and Git-backed mirror launchers |
| Dry-run audit | PASS: 4 candidates, 0 actual `--unknown_tx_ids`, new OLD80 policy present |
| Git-backed mirror | PASS, committed as `9ee545f` |
| N607 preflight/occupancy | PASS before sync/launch |

## Claim Boundary

A row passes the live goal only if the same candidate reaches `old_acc>=80%` and `seen_new_acc>=65%`. A 2-new row satisfies the multi-new requirement; 3-new rows are stronger stress evidence. Unknown FAR remains outside the success claim for this live scoped goal.

## Local Verification Evidence

| Artifact | SHA256 |
|---|---|
| `code/scripts/launch_phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024.sh` | `FE75985F6366B657CFF190B136E05FFAD83751A4BF69CE47A45886ABC9B10427` |
| `automation_reports/CV-SincNet/phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024/local_dry_run.out` | `60C29A742D2AD72EF8985461EE261BB638D669FF6F100ECFAAAE3754DF9B7203` |

Dry-run audit:

| Field | Observed |
|---|---:|
| Candidate rows | 4 |
| Actual `--unknown_tx_ids` occurrences | 0 |
| `replace_all_except_seen_new_override` occurrences | 5 |
| 2-new candidate lines | 2 |
| 3-new candidate lines | 2 |

The Git-backed launcher has path-scoped LF checkout policy in `.gitattributes`.

## N607 Prelaunch Evidence

| Check | Evidence |
|---|---|
| Direct preflight | PASS via `tools\n607_ssh_preflight.ps1`; project root visible; 8 GPUs visible |
| Live occupancy | `tools\n607_training_inventory.py --direct-only --pretty`: no `gpu_compute`, no active training processes, no launcher context |
| Target path conflict | no existing launcher, run root, or log root for this `RUN_ID` |
| Disk | `/home` has 7.7T available, 26% used |
| SSH cleanup | local `ssh_process_count=0`, `n607_or_bridge_established_count=0` after preflight, occupancy, and path checks |

Planned remote sync:

| Local | Remote |
|---|---|
| `code/cvsrffi/spaceborne_fewshot.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/spaceborne_fewshot.py` |
| `code/eval_spaceborne_fewshot.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/eval_spaceborne_fewshot.py` |
| `code/scripts/launch_phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024.sh` |
| `automation_reports/CV-SincNet/phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024/report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024/report.md` |

Planned launch command:

```bash
ssh -F E:\type10-7\tools\n607_ssh_config -o BatchMode=yes N607 'cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024 automation_reports/CV-SincNet/phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024 && nohup env RUN_ID=phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024 bash code/scripts/launch_phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024.sh > logs/phase2_adv3b02_head48f_retain_seen_multinew_no_unknown_20260703_0024/scheduler.out 2>&1 < /dev/null & echo scheduler_pid=$!'
```

## Remote Verification and Launch Evidence

| Check | Evidence |
|---|---|
| Remote compile | PASS for `code/cvsrffi/spaceborne_fewshot.py` and `code/eval_spaceborne_fewshot.py` |
| Remote launcher syntax | PASS via `bash -n` |
| Remote dry-run | PASS: 4 candidates, actual `--unknown_tx_ids` count 0, 2 two-new rows, 2 three-new rows |
| Remote code hash | `spaceborne_fewshot.py=28cab198ad1108bbcff403658700ed7e29b637e53981cbc1bf72028d0803b758`; `eval_spaceborne_fewshot.py=64921019b04a92916e96eaf3fdc422af09f38ac48b3a99c9c3e7b0cbddbc6906` |
| Remote launcher hash | `fe75985f6366b657cff190b136e05ffad83751a4bf69ce47a45886abc9b10427` |
| Launch | `scheduler_pid=880965` |
| Startup health | PASS: 4 eval processes observed, feature files generated, no fatal log lines |
| Final process state | COMPLETE: no remaining eval process for this run |
| SSH cleanup | PASS: local `ssh.exe` and established N607/bridge TCP checks clear after bounded remote tasks |

## Completion Results

All rows failed the live same-row target of `old_acc>=80%` and `seen_new_acc>=65%`.

| Candidate | New-class scope | Threshold mode | K-old | K-new | Old acc | Seen-new acc | Harmonic mean | Coverage | Seen-new to old | Rollback | Loss trend | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_2NEW_RELAX` | 2 new | relaxed | 10 | 10 | 55.00% | 30.00% | 0.3882 | 100.00% | 70.00% | triggered=False, accepted=True | 4.3464->1.3118 | FAIL_TARGET |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_2NEW_STRICT` | 2 new | strict | 10 | 10 | 59.17% | 15.00% | 0.2393 | 100.00% | 85.00% | triggered=False, accepted=True | 5.4984->1.7087 | FAIL_TARGET |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_3NEW_RELAX` | 3 new | relaxed | 10 | 10 | 58.75% | 8.33% | 0.1460 | 100.00% | 60.00% | triggered=False, accepted=True | 4.4362->1.8484 | FAIL_TARGET |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_3NEW_STRICT` | 3 new | strict | 10 | 10 | 61.25% | 14.17% | 0.2301 | 100.00% | 73.33% | triggered=False, accepted=True | 5.5617->2.0031 | FAIL_TARGET |

Per-new-class accuracy shows the old-retention policy preserves old labels by swallowing many target-new queries as old classes.

| Candidate | `1-14` acc | `1-16` acc | `1-18` acc |
|---|---:|---:|---:|
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_2NEW_RELAX` | n/a | 0.00% | 60.00% |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_2NEW_STRICT` | n/a | 0.00% | 30.00% |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_3NEW_RELAX` | 0.00% | 0.00% | 25.00% |
| `ADV3B02_HEAD48F_RETAIN_SEEN_K10_3NEW_STRICT` | 0.00% | 0.00% | 42.50% |

## Post-hoc Diagnostic Checks

These checks reused the completed feature artifacts and exact support/query manifests; they did not create a deployment-success claim.

| Diagnostic | Best observed row | Interpretation |
|---|---|---|
| Target support prototype/kNN on first-run 3-new K10/K20 features | K20 `knn3`: old 69.44%, seen-new 55.56% | Target support evidence improves the balance but remains below both thresholds |
| Target support prototype/kNN on second-run 2-new features | `knn3`: old 70.00%, seen-new 63.75%; `knn1`: old 63.75%, seen-new 67.50% | New-class target can approach 65%, but old target stays far below 80% |
| ADV3B02 structural old logits on second-run 2-new features | old-only target accuracy 65.83%, seen-new 0.00% | The base old-class head is not an 80% old-domain anchor for this target receiver |
| Old-logit plus support-similarity threshold sweep | Best tradeoffs included old 57.08%/seen-new 67.50% and old 53.75%/seen-new 75.00% | Even oracle post-hoc thresholds cannot give same-row old 80% and seen-new 65% on this receiver/class set |

## Interpretation and Next Decision

The `replace_all_except_seen_new_override` route is diagnostic-negative. It raises old accuracy only into the 55-61% range and severely suppresses new-class recognition, especially `1-16` and `1-14`. Combined with the first run, the current evidence says target receiver `20-1` with the tested ADV3B02 features does not expose a same-row 80% old / 65% multi-new solution through terminal thresholding or OLD80 retention.

The next valid route is not another threshold-only tweak on `20-1`; it should be a receiver/new-class sweep under the same no-unknown, at-least-two-new-class scope to test whether another allowed Stage2-C target receiver gives a cleaner old/new separation from `ADV3B02_CORE90_SOFT_E200`.
