# phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707

## Scope

- Operator: Codex.
- Timestamp: 2026-07-07.
- Objective: continue qKNNV42 Stage2-C optimization after the negative support-floor probe by combining the previously useful seen-new/old contrast relief with candidate-set shell-risk veto thresholds.
- Protocol: CVS Stage2-C, K=5 and K=10 target-domain support, target old plus seen-new support only, target unknown query evaluation only, satellite/LEO target channel view.
- Source feature run: `phase2_adv3b02_stage2c_normsep_protocol_20260707`.
- Case: `PHASE2_STAGE2C_RX7_14`.
- Variants: `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP`.
- Diagnostic status: diagnostic-only. This run must not be reported as deployment success unless it meets protocol metrics and is promoted by a separate governance decision.

## Hypothesis

The prior support-floor route protected unknown rejection too aggressively and collapsed known coverage, especially for K=5. The next narrower route keeps the contrast-relief mechanism but adds shell-risk candidate veto/reject thresholds:

- Preserve old-class K=10 adaptation from the contrast-relief route.
- Reduce unknown FAR that remained above 0.10 in the support-floor probe.
- Avoid zeroing seen-new coverage and inspect whether lowest seen-new class improves when shell veto is less destructive than support-floor gating.

## Local Changes

- `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`: expose `--candidate_set_max_label_shell_risk` and `--candidate_set_shell_reject_risk` in the ManyTx wrapper and pass them to `evaluate_collaborative_open_set_evidence`; include shell-veto count/rate in wrapper summary rows.
- `code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py`: add CLI coverage for the shell-risk knobs and summary-row coverage for shell-veto evidence fields.
- `code/scripts/launch_phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707.sh`: new bounded diagnostic launcher.
- `automation_reports/CV-SincNet/phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707/report.md`: this report.

## Planned Matrix

Total planned diagnostics: 36 runs.

| Axis | Values |
|---|---|
| Variant | `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP` |
| K-shot | `5`, `10` |
| Contrast profile | `U095_W050M02_D008_R035`, `U095_W050M02_D008_R050`, `U095_W050M02_D005_R050` |
| Shell profile | `L065_R085`, `L075_R090`, `L085_R095` |
| Query per class | `70` |
| qKNN K | `8` |
| Fusion policy | `candidate_set_cvs` |
| Support selection | `stable_first` |

## Local Verification Plan

Completed locally under `ssr-gpu` unless noted.

| Check | Result |
|---|---|
| `python -m pytest code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | PASS, `6 passed` |
| `python -m py_compile code\scripts\phase2_frozen_manytx_unknown_diagnostic.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py` | PASS |
| `python -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | PASS, `134 passed` |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707.sh` | PASS |
| `ROOT=/tmp/CV-SincNet-shell-veto-test DRY_RUN=1 bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707.sh --dry-run` | PASS via `bash -lc`, expanded 36 planned diagnostics |

Note: one first Windows-host dry-run attempt set `ROOT` from PowerShell before launching `bash` and failed by trying to create the default remote path. The verified dry-run uses bash-local environment assignment and is the command recorded above.

## N607 Plan

- Preflight: `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`.
- Sync local verified files to:
  - `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`
  - `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707.sh`
- Remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707.sh > logs/phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707.launch.out 2>&1 & echo $!
```

## Output Paths

- Remote run root: `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707`.
- Remote log root: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707`.
- Expected summary JSON: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707/stage2c_contrast_shell_veto_probe_summary.json`.
- Expected summary CSV: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707/stage2c_contrast_shell_veto_probe_summary.csv`.

## N607 Execution

| Item | Value |
|---|---|
| Direct SSH preflight | PASS |
| Remote sync | PASS |
| Remote hash check | PASS, matched local SHA256 for wrapper and launcher |
| Remote verification | PASS: `py_compile`, `bash -n`, remote dry-run |
| Launch PID | `4163824` |
| Launch status | Completed |
| JSON outputs | `36` |
| Summary pulled local | `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707\stage2c_contrast_shell_veto_probe_summary.json` |
| CSV pulled local | `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_contrast_shell_veto_probe_20260707\stage2c_contrast_shell_veto_probe_summary.csv` |
| SSH cleanup | No local `ssh.exe` or established TCP22 connection after SCP/monitor tasks |

## Result Interpretation

This route is diagnostic-negative as a qKNNV42 improvement path.

- FAR feasibility improved in 17/36 rows, with minimum `unknown_FAR=0.0571`.
- All FAR-feasible rows have `old_acc=0.0000` and `min_old_class_acc=0.0000`, so they cannot support old-class adaptation.
- The best old-class rows are K=10 NORM with `old_acc=0.6357`, but their `unknown_FAR` remains at least `0.3143`, well above the 0.10 diagnostic bound.
- `min_seen_new_class_acc` remains `0.0000` in every row, so shell-veto does not solve lowest seen-new class collapse.
- Shell veto is therefore useful only as evidence that shell-risk thresholds can trade FAR down; it does not provide a promotable route under the current Stage2-C objective.

Recommended next route: stop escalating pure unknown-shell veto. Move to a known-class-preserving mechanism, such as class-balanced seen-new quota or receiver-class reliability compensation that protects old support coverage first, then applies unknown confirmation as a second layer.

## Full Result Table

| variant | profile | K | c | old | min_old | seen | min_seen | FAR | reject | cov | shell_n | shell_r | feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L065_R085 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1089 | 0.0000 | 0.0571 | 0.9286 | 0.1041 | 276.0000 | 0.1792 | True |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L065_R085 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1089 | 0.0000 | 0.0571 | 0.9286 | 0.1041 | 276.0000 | 0.1792 | True |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L065_R085 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1286 | 0.0000 | 0.0768 | 0.9036 | 0.1245 | 276.0000 | 0.1792 | True |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L075_R090 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1125 | 0.0000 | 0.0643 | 0.9250 | 0.1092 | 222.0000 | 0.1442 | True |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L075_R090 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1125 | 0.0000 | 0.0643 | 0.9250 | 0.1092 | 222.0000 | 0.1442 | True |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L075_R090 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1321 | 0.0000 | 0.0875 | 0.8982 | 0.1316 | 222.0000 | 0.1442 | True |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L085_R095 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1143 | 0.0000 | 0.0714 | 0.9161 | 0.1153 | 163.0000 | 0.1058 | True |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L085_R095 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1143 | 0.0000 | 0.0714 | 0.9161 | 0.1153 | 163.0000 | 0.1058 | True |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L085_R095 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1339 | 0.0000 | 0.0964 | 0.8857 | 0.1398 | 163.0000 | 0.1058 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L065_R085 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 | 0.0696 | 0.9196 | 0.1051 | 289.0000 | 0.1877 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L065_R085 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 | 0.0696 | 0.9196 | 0.1051 | 289.0000 | 0.1877 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L065_R085 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1196 | 0.0000 | 0.0929 | 0.8893 | 0.1316 | 289.0000 | 0.1877 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L085_R095 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1054 | 0.0000 | 0.0804 | 0.9125 | 0.1153 | 150.0000 | 0.0974 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L085_R095 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1054 | 0.0000 | 0.0804 | 0.9125 | 0.1153 | 150.0000 | 0.0974 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L075_R090 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1018 | 0.0000 | 0.0768 | 0.9179 | 0.1102 | 229.0000 | 0.1487 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L075_R090 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1018 | 0.0000 | 0.0768 | 0.9179 | 0.1102 | 229.0000 | 0.1487 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L075_R090 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1214 | 0.0000 | 0.1000 | 0.8857 | 0.1378 | 229.0000 | 0.1487 | True |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L065_R085 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1357 | 0.0000 | 0.3179 | 0.6714 | 0.4490 | 80.0000 | 0.0519 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L065_R085 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1357 | 0.0000 | 0.3179 | 0.6714 | 0.4490 | 80.0000 | 0.0519 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L075_R090 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1357 | 0.0000 | 0.3268 | 0.6714 | 0.4520 | 58.0000 | 0.0377 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L085_R095 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1357 | 0.0000 | 0.3268 | 0.6714 | 0.4520 | 41.0000 | 0.0266 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L075_R090 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1357 | 0.0000 | 0.3268 | 0.6714 | 0.4520 | 58.0000 | 0.0377 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L085_R095 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1357 | 0.0000 | 0.3268 | 0.6714 | 0.4520 | 41.0000 | 0.0266 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L065_R085 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1196 | 0.0000 | 0.3143 | 0.6714 | 0.4541 | 74.0000 | 0.0481 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L065_R085 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1196 | 0.0000 | 0.3143 | 0.6714 | 0.4541 | 74.0000 | 0.0481 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L075_R090 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1196 | 0.0000 | 0.3214 | 0.6696 | 0.4541 | 59.0000 | 0.0383 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L075_R090 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1196 | 0.0000 | 0.3214 | 0.6696 | 0.4541 | 59.0000 | 0.0383 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R035_SHELL_L085_R095 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1196 | 0.0000 | 0.3232 | 0.6661 | 0.4551 | 44.0000 | 0.0286 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D008_R050_SHELL_L085_R095 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1196 | 0.0000 | 0.3232 | 0.6661 | 0.4551 | 44.0000 | 0.0286 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L065_R085 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1500 | 0.0000 | 0.3768 | 0.6054 | 0.4867 | 80.0000 | 0.0519 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L065_R085 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1357 | 0.0000 | 0.3554 | 0.6268 | 0.4755 | 74.0000 | 0.0481 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L075_R090 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1500 | 0.0000 | 0.3893 | 0.6054 | 0.4898 | 58.0000 | 0.0377 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L085_R095 | 10.0000 | 1.0000 | 0.6048 | 0.1429 | 0.1500 | 0.0000 | 0.3929 | 0.6036 | 0.4898 | 41.0000 | 0.0266 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L075_R090 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1357 | 0.0000 | 0.3661 | 0.6250 | 0.4765 | 59.0000 | 0.0383 | False |
| STAGE2C_NORM_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L085_R095 | 10.0000 | 1.0000 | 0.6357 | 0.0857 | 0.1357 | 0.0000 | 0.3679 | 0.6179 | 0.4776 | 44.0000 | 0.0286 | False |
| STAGE2C_HEAD_SEP | RELIEF_U095_W050M02_D005_R050_SHELL_L085_R095 | 5.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 | 0.1107 | 0.8732 | 0.1469 | 150.0000 | 0.0974 | False |
