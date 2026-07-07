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

| Item | Value |
|---|---|
| Remote sync | PASS, launcher only; evaluator and wrapper reused from the previous synced commit |
| Remote hash check | PASS, launcher SHA256 matched local |
| Remote verification | PASS: bash -n and remote dry-run expanded 4 diagnostics |
| Launch PID | 4191310 |
| Launch status | Completed |
| JSON outputs | 4 |
| Summary pulled local | E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707\stage2c_seennew_exempt_floor_audit_summary.json |
| Event JSONs pulled local | E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707\remote_case |
| SSH cleanup | No local ssh.exe or established TCP22 connection after SCP, launch, monitor, and pull tasks |

## Result Interpretation

This audit identifies why the prior seen-new exemption matrix did not trigger, but it is not a promotable solution.

- Floorless and support>=1 exemptions both trigger on real Stage2-C events: NORM_SEP has 772 exempt seen-new candidate events, HEAD_SEP has 722.
- Triggering the exemption restores seen-new accuracy, but unknown false accepts rise sharply: best seen-new row is STAGE2C_NORM_SEP/AUDIT_FLOORLESS_X0P000R000 with seen_new_acc=0.3875 and min_seen_new_class_acc=0.1857, but unknown_FAR=0.5714.
- Old adaptation remains partial and lowest old class remains collapsed: old_acc=0.4810 at best in this audit, min_old_class_acc=0.0000.
- Event distributions show the failure mode: true seen-new and unknown false accepts have nearly identical support_count=10, high p-values, and overlapping receiver reliability. In NORM_SEP floorless, seen-new reliability spans roughly 0.2387-0.3055 while unknown spans 0.2387-0.3055. In HEAD_SEP floorless, seen-new spans roughly 0.2410-0.2809 while unknown spans 0.2410-0.2809.
- Therefore, lowering receiver reliability to 0.25-0.30 can restore seen-new but cannot protect unknown FAR by itself. The next route must add an unknown-specific discriminator after seen-new registration, not another global support/p-value/reliability threshold.

Recommended next route: audit candidate-level geometry for true seen-new vs unknown false accepts after floorless exemption, especially old-contrast delta, label shell risk, event unknown risk, receiver-pair disagreement, and per-class confusion. If no separable signal exists, qKNNV42 needs a new seen-new registration verifier rather than threshold tuning.

## Result Table

| variant | profile | old | min_old | seen | min_seen | FAR | reject | cov | exempt_n | veto_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STAGE2C_NORM_SEP | AUDIT_FLOORLESS_X0P000R000 | 0.4810 | 0.0000 | 0.3875 | 0.1857 | 0.5714 | 0.3929 | 0.7469 | 772 | 460 |
| STAGE2C_NORM_SEP | AUDIT_SUPPORT1_X1P000R000 | 0.4810 | 0.0000 | 0.3875 | 0.1857 | 0.5714 | 0.3929 | 0.7469 | 772 | 460 |
| STAGE2C_HEAD_SEP | AUDIT_FLOORLESS_X0P000R000 | 0.4310 | 0.0000 | 0.3536 | 0.2143 | 0.5268 | 0.4286 | 0.6980 | 722 | 532 |
| STAGE2C_HEAD_SEP | AUDIT_SUPPORT1_X1P000R000 | 0.4310 | 0.0000 | 0.3536 | 0.2143 | 0.5268 | 0.4286 | 0.6980 | 722 | 532 |

## Event Audit Table

| variant | profile | events | seen_new_candidates | rescued_seen | exempt_seen | max_rel | max_p | max_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STAGE2C_HEAD_SEP | AUDIT_FLOORLESS_X0P000R000 | 1540 | 740 | 722 | 722 | 0.2809 | 1.0000 | 10.0000 |
| STAGE2C_HEAD_SEP | AUDIT_SUPPORT1_X1P000R000 | 1540 | 740 | 722 | 722 | 0.2809 | 1.0000 | 10.0000 |
| STAGE2C_NORM_SEP | AUDIT_FLOORLESS_X0P000R000 | 1540 | 790 | 772 | 772 | 0.3055 | 1.0000 | 10.0000 |
| STAGE2C_NORM_SEP | AUDIT_SUPPORT1_X1P000R000 | 1540 | 790 | 772 | 772 | 0.3055 | 1.0000 | 10.0000 |
