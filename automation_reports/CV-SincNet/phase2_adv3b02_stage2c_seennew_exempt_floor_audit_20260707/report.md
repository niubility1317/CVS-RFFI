# phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707

## Scope

- Operator: Codex.
- Timestamp: 2026-07-07.
- Objective: diagnose why `phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707` produced zero seen-new exemption hits in real Stage2-C evidence.
- Protocol: CVS Stage2-C, K=10 target-domain support, target old plus seen-new support only, target unknown query evaluation only, satellite/LEO target channel view.
- Source feature run: `phase2_adv3b02_stage2c_normsep_protocol_20260707`.
- Case: `PHASE2_STAGE2C_RX7_14`.
- Variants: `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP`.
- Diagnostic status: diagnostic-only event audit. This is not a deployment-success candidate.

## Rationale

The seen-new exempt-veto probe had 18 FAR-feasible rows but `seen_new_acc=0` and `rescue_unknown_veto_seen_new_exemption_count=0` in every row. This audit deliberately lowers exemption floors to identify whether the branch can trigger at all on real evidence.

## Planned Matrix

Total planned diagnostics: 4 runs.

| Axis | Values |
|---|---|
| Variant | `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP` |
| K-shot | `10` |
| Fusion policy | `scorer_cvs` |
| Veto profile | event/label/shell/component=`0.80/0.80/0.85/0.70`, min sources=`2` |
| Exemption profiles | `AUDIT_FLOORLESS_X0P000R000`, `AUDIT_SUPPORT1_X1P000R000` |
| Event detail | `--include_event_results` |

## Local Version State

- Base commit before this audit launcher: `5774b04 Record Stage2-C seen-new exempt veto results`.
- New local file: `code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707.sh`.
- New report: `automation_reports/CV-SincNet/phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707/report.md`.

## Local Verification Plan

| Check | Result |
|---|---|
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707.sh` | PASS |
| `ROOT=/tmp/CV-SincNet-seennew-exempt-floor-audit-test DRY_RUN=1 bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707.sh --dry-run` | PASS via `bash -lc`, expanded 4 planned diagnostics |

## N607 Plan

- Sync launcher to `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707.sh`.
- Reuse already-synced evaluator and wrapper from commit `579baa6`.
- Remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707.sh > logs/phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707.launch.out 2>&1 & echo $!
```

## Output Paths

- Remote run root: `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707`.
- Remote log root: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707`.
- Expected summary JSON: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707/stage2c_seennew_exempt_floor_audit_summary.json`.
- Expected event audit CSV: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707/stage2c_seennew_exempt_floor_audit_summary_event_audit.csv`.

## N607 Execution

Pending.

## Result Interpretation

Pending.
