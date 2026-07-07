# phase2_adv3b02_stage2c_conformal_veto_probe_20260707

## Scope

- Operator: Codex.
- Timestamp: 2026-07-07.
- Objective: test a qKNNV42 Stage2-C hybrid route that keeps `SCORER_CONFORMAL_SEEN` known/seen-new recovery but adds rescue-scoped unknown veto evidence.
- Protocol: CVS Stage2-C, K=5 and K=10 target-domain support, target old plus seen-new support only, target unknown query evaluation only, satellite/LEO target channel view.
- Source feature run: `phase2_adv3b02_stage2c_normsep_protocol_20260707`.
- Case: `PHASE2_STAGE2C_RX7_14`.
- Variants: `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP`.
- Diagnostic status: diagnostic-only. This run must not be reported as deployment success unless it meets protocol metrics and is promoted by a separate governance decision.

## Rationale

The previous seen-new rescue sweep showed `SCORER_CONFORMAL_SEEN` can recover old-class and seen-new performance (`old_acc` near 0.80 and `seen_new_acc` up to 0.3875), but it accepts unknown too often (`unknown_FAR` near 0.87-0.96). The shell-veto probe showed shell thresholds can reduce FAR but destroy old-class coverage when applied as a candidate-set accept gate.

This run keeps the scorer/conformal rescue path and applies unknown veto only to rescue/conformal accepted known labels. The aim is to reject unknown-like rescue hits while preserving real old and seen-new support-confirmed labels.

## Local Changes

- `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`: include `seen_new_rescue_count`, `seen_new_rescue_rate`, `rescue_unknown_veto_count`, and `rescue_unknown_veto_rate` in summary rows.
- `code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py`: require those summary fields.
- `code/scripts/launch_phase2_adv3b02_stage2c_conformal_veto_probe_20260707.sh`: new bounded diagnostic launcher.
- `automation_reports/CV-SincNet/phase2_adv3b02_stage2c_conformal_veto_probe_20260707/report.md`: this report.

## Planned Matrix

Total planned diagnostics: 20 runs.

| Axis | Values |
|---|---|
| Variant | `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP` |
| K-shot | `5`, `10` |
| Fusion policy | `scorer_cvs` |
| Candidate generator | `seen_new_rescue_enabled` plus `conformal_rescue_enabled` |
| Veto profile | `CONF_VETO_M1_E090_L090_S090_C075`, `CONF_VETO_M2_E090_L090_S090_C075`, `CONF_VETO_M1_E085_L085_S085_C070`, `CONF_VETO_M2_E085_L085_S085_C070`, `CONF_VETO_M2_E080_L080_S085_C070` |
| Query per class | `70` |
| qKNN K | `8` |
| Support selection | `stable_first` |

## Local Verification Plan

Completed locally under `ssr-gpu` unless noted.

| Check | Result |
|---|---|
| `python -m py_compile code\scripts\phase2_frozen_manytx_unknown_diagnostic.py code\evaluation\collaborative_open_set_qknn_eval.py` | PASS |
| `python -m pytest code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py` | PASS, `6 passed` |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_conformal_veto_probe_20260707.sh` | PASS |
| `ROOT=/tmp/CV-SincNet-conformal-veto-test DRY_RUN=1 bash code/scripts/launch_phase2_adv3b02_stage2c_conformal_veto_probe_20260707.sh --dry-run` | PASS via `bash -lc`, expanded 20 planned diagnostics |

## N607 Plan

- Preflight: `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`.
- Sync local verified files to:
  - `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`
  - `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_conformal_veto_probe_20260707.sh`
- Remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_conformal_veto_probe_20260707.sh > logs/phase2_adv3b02_stage2c_conformal_veto_probe_20260707.launch.out 2>&1 & echo $!
```

## Output Paths

- Remote run root: `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_conformal_veto_probe_20260707`.
- Remote log root: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_conformal_veto_probe_20260707`.
- Expected summary JSON: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_conformal_veto_probe_20260707/stage2c_conformal_veto_probe_summary.json`.
- Expected summary CSV: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_conformal_veto_probe_20260707/stage2c_conformal_veto_probe_summary.csv`.

## Pending Results

Not launched yet. Final result table will be appended after N607 completion.
