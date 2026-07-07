# phase2_adv3b02_stage2c_target_old_upper_bound_20260707

## Scope

- Operator: Codex.
- Timestamp: 2026-07-07.
- Objective: diagnose the target-old-only upper bound after qKNNV42 Stage2-C runs failed OLD80_FIRST.
- Protocol boundary: TARGET_OLD_ONLY_UPPER_BOUND_DIAGNOSTIC, non-deployment. Uses only target-old rows from the target receiver domain for support/query splitting.
- Source feature run: `phase2_adv3b02_stage2c_normsep_protocol_20260707`.
- Case: `PHASE2_STAGE2C_RX7_14`.
- Variants: `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP`.
- K: `5`, `10`.
- Models: frozen feature prototype, support-only ridge linear probe, fixed-epoch support-only MLP adapter.
- Explicit target-old TX IDs: `14-10,14-7,20-15,20-19,6-15,8-20`.

## Rationale

The latest qKNNV42 seen-new exemption work did not solve the main objective. Best FAR-feasible rows still keep old accuracy around `0.48` and minimum old-class accuracy at `0.0`. The floorless seen-new exemption can recover seen-new accuracy, but unknown FAR rises above `0.52`.

Before adding more open-set gates, this diagnostic checks whether the frozen Stage2-C features contain enough target-old class signal under a target-old-only split. If even target-old-only upper-bound models stay far below `old_acc>=0.80` or keep `min_old_class_acc=0`, the bottleneck is feature/target-domain representation rather than only qKNN thresholding.

## Planned Matrix

| Axis | Values |
|---|---|
| Variant | `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP` |
| K | `5`, `10` |
| Prototype upper bound | deterministic target-old prototype classifier |
| Linear upper bound | ridge lambdas `0.001,0.01,0.1,1.0,10.0` |
| MLP upper bound | seeds `1,7,13`, epochs `120`, hidden dim `64`, CPU |

## Local Version State

- Git carrier: `E:\type10-7\github_publish\CVS-RFFI-repo`.
- Base commit before this launcher: `a8228a1 Record Stage2-C seen-new exemption floor audit results`.
- New launcher: `code/scripts/launch_phase2_adv3b02_stage2c_target_old_upper_bound_20260707.sh`.
- Report mirror: `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_target_old_upper_bound_20260707\report.md`.
- Repair note: first remote attempt failed closed because the feature package did not expose `manifest.target_old_tx_ids`; launcher was repaired to pass the project-protocol `Y_old` list explicitly.

## Local Verification Plan

| Check | Result |
|---|---|
| `python -m pytest code/tests/test_target_old_only_upper_bound.py code/tests/test_target_old_linear_probe_upper_bound.py code/tests/test_target_old_mlp_adapter_upper_bound.py -q` | PASS, `7 passed` |
| `python -m py_compile code/scripts/eval_target_old_only_upper_bound.py code/scripts/eval_target_old_linear_probe_upper_bound.py code/scripts/eval_target_old_mlp_adapter_upper_bound.py` | PASS |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_target_old_upper_bound_20260707.sh` | PASS |
| `ROOT=/tmp/CV-SincNet-target-old-upper-test DRY_RUN=1 bash code/scripts/launch_phase2_adv3b02_stage2c_target_old_upper_bound_20260707.sh --dry-run` | PASS via `bash -lc`, expanded 6 mode/variant jobs |

## N607 Plan

- Preflight first: `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`.
- Sync launcher and three target-old upper-bound scripts to `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/`.
- Remote verify hashes, `py_compile`, `bash -n`, and dry-run before launch.
- Remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_target_old_upper_bound_20260707.sh > logs/phase2_adv3b02_stage2c_target_old_upper_bound_20260707.launch.out 2>&1 < /dev/null & echo $!
```

## Output Paths

- Remote run root: `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_target_old_upper_bound_20260707`.
- Remote log root: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_target_old_upper_bound_20260707`.
- Expected summary JSON: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_target_old_upper_bound_20260707/stage2c_target_old_upper_bound_summary.json`.
- Expected summary CSV: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_target_old_upper_bound_20260707/stage2c_target_old_upper_bound_summary.csv`.

## N607 Execution

Pending.

## Result Interpretation

Pending.
