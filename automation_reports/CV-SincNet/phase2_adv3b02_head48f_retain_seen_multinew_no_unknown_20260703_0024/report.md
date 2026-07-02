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
| Git-backed mirror | pending commit |
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
