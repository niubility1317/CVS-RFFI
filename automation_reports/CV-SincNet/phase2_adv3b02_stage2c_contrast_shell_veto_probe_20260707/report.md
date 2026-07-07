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

## Pending Results

Not launched yet. Final result table will be appended after N607 completion.
