# phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707

## Scope

- Operator: Codex.
- Timestamp: 2026-07-07.
- Objective: continue qKNNV42 Stage2-C optimization after the conformal-veto diagnostic by allowing rescue/conformal seen-new candidates to bypass rescue unknown veto only when support/conformal evidence is strong.
- Protocol: CVS Stage2-C, K=5 and K=10 target-domain support, target old plus seen-new support only, target unknown query evaluation only, satellite/LEO target channel view.
- Source feature run: `phase2_adv3b02_stage2c_normsep_protocol_20260707`.
- Case: `PHASE2_STAGE2C_RX7_14`.
- Variants: `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP`.
- Diagnostic status: diagnostic-only. This run must not be reported as deployment success unless it meets protocol metrics and is promoted by a separate governance decision.

## Rationale

The prior conformal-veto route reduced unknown FAR but also vetoed all seen-new recovery. Best FAR-feasible row retained only partial old accuracy (`old_acc=0.4810`, `unknown_FAR=0.0857`) and still had `seen_new_acc=0`, `min_seen_new_class_acc=0`, and `min_old_class_acc=0`.

This run keeps the same scorer/conformal candidate generator and rescue unknown veto, but adds a label-set-aware seen-new exemption. The exemption is not based on true query role; it requires the predicted label to be in the registered seen-new label set and to satisfy support count, class conformal p-value, and receiver-class reliability floors. Unknown false accepts with weak support evidence should still be vetoed.

## Local Changes

- `code/evaluation/collaborative_open_set_qknn_eval.py`: add `rescue_unknown_veto_seen_new_exemption_*` controls, event fields, aggregate counts, and internal propagation across collaboration paths.
- `code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`: expose exemption CLI parameters and include exemption count/rate in summary rows.
- `code/scripts/phase2_collaborative_open_set_qknn_eval.py`: expose and pass the same rescue-veto/exemption controls for direct qKNN evaluation.
- `code/tests/test_collaborative_open_set_qknn_eval.py`: add behavior test proving strong seen-new support bypasses rescue veto while weak unknown false accept remains rejected.
- `code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py`: add diagnostic CLI and summary-field coverage.
- `code/tests/test_phase2_collaborative_open_set_qknn_eval.py`: add direct qKNN CLI parser coverage.
- `code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707.sh`: new bounded diagnostic launcher.
- `automation_reports/CV-SincNet/phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707/report.md`: this report.

## Planned Matrix

Total planned diagnostics: 20 runs.

| Axis | Values |
|---|---|
| Variant | `STAGE2C_NORM_SEP`, `STAGE2C_HEAD_SEP` |
| K-shot | `5`, `10` |
| Fusion policy | `scorer_cvs` |
| Candidate generator | `seen_new_rescue_enabled` plus `conformal_rescue_enabled` |
| Veto/exemption profile | `EXEMPT_STRICT_E080_L080_S085_C070_X3P070R070`, `EXEMPT_MID_E085_L085_S085_C070_X3P070R070`, `EXEMPT_OLDPROT_E080_L080_S085_C070_X4P080R080`, `EXEMPT_OPENSEEN_E080_L080_S085_C070_X2P050R050`, `EXEMPT_STRICTVETO_E075_L075_S085_C065_X3P070R070` |
| Query per class | `70` |
| qKNN K | `8` |
| Support selection | `stable_first` |

## Local Verification Plan

Completed locally under `ssr-gpu` unless noted.

| Check | Result |
|---|---|
| `python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_frozen_manytx_unknown_diagnostic.py code/scripts/phase2_collaborative_open_set_qknn_eval.py` | PASS |
| `python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py -k "seen_new_support_exemption"` | PASS |
| `python -m pytest code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py -k "seen_new_rescue_cli_knobs or summary_rows"` | PASS |
| `python -m pytest code/tests/test_phase2_collaborative_open_set_qknn_eval.py -k "rescue_veto_seen_new_exemption_knobs or ospr_ci_pp"` | PASS |
| `python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py code/tests/test_phase2_collaborative_open_set_qknn_eval.py -q` | PASS, `136 passed` |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707.sh` | PASS |
| `ROOT=/tmp/CV-SincNet-seennew-exempt-veto-test DRY_RUN=1 bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707.sh --dry-run` | PASS via `bash -lc`, expanded 20 planned diagnostics |

## N607 Plan

- Preflight: `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`.
- Sync local verified files to:
  - `/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py`
  - `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`
  - `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py`
  - `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707.sh`
- Remote command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && nohup bash code/scripts/launch_phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707.sh > logs/phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707.launch.out 2>&1 & echo $!
```

## Output Paths

- Remote run root: `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707`.
- Remote log root: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707`.
- Expected summary JSON: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707/stage2c_seennew_exempt_veto_probe_summary.json`.
- Expected summary CSV: `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_seennew_exempt_veto_probe_20260707/stage2c_seennew_exempt_veto_probe_summary.csv`.

## N607 Execution

Pending.

## Result Interpretation

Pending.
