# SATVAL RA Ablation 20260527 103712

## Run Identity

- Experiment group: `SATVAL_RA_ABLATION_20260527_103712`
- Operator/agent: Codex automation
- Timestamp: 2026-05-27 10:37:12 Asia/Hong_Kong
- Objective: run two validations requested by the user after the BEX02 satellite runs exposed a gap versus the receiver-agnostic comparison model.

## Validation Questions

1. Same-evaluator checkpoint validation: compare `BEX02_ratio010_baseline_concat_mixed_gpu0_20260526_223716` and the receiver-agnostic baseline checkpoint under the same WiSig split, same main OOD test subsets, same satellite-channel simulator, same scenario list, and same accuracy-counting loop.
2. Receiver-agnostic objective ablation: re-run the receiver-agnostic RA-Collab baseline at WiSig ratio `0.1` with supervised clean+satellite mixed-orbit view expansion to test whether RA objective plus supervised satellite views explains the stronger satellite metrics.

## Hypothesis And Comparison Target

- Hypothesis: the satellite simulator is not the main difference. The leading difference is objective/architecture: RA-Collab trains TX CE on clean+satellite duplicated samples while applying receiver-adversarial pressure, whereas BEX02 concat keeps the CVS-RFFI DG/MixStyle/Fishr objective stack.
- Comparison target: BEX02 mixed concat best primary checkpoint versus receiver-agnostic baseline `best_by_val.pt`.
- Main success evidence: RA remains materially above BEX02 on satellite strict UDU under the same script; if so, the next BEX02 direction should transfer receiver-adversarial pressure to satellite views instead of only retuning satellite channel parameters.

## Local Files Changed

- `E:\type10-7\automation_reports\CV-SincNet\SATVAL_RA_ABLATION_20260527_103712\same_evaluator_sat_validation.py`
- `E:\type10-7\automation_reports\CV-SincNet\SATVAL_RA_ABLATION_20260527_103712\launch_same_eval.sh`
- `E:\type10-7\automation_reports\CV-SincNet\SATVAL_RA_ABLATION_20260527_103712\launch_ra_ablation.sh`
- `E:\type10-7\automation_reports\CV-SincNet\SATVAL_RA_ABLATION_20260527_103712\report.md`

## Planned Remote Paths

- Remote root: `/home/szu2070436088/2510044040/CV-SincNet`
- Remote launcher dir: `/home/szu2070436088/2510044040/CV-SincNet/automation_launchers/SATVAL_RA_ABLATION_20260527_103712`
- Same-evaluator output JSON: `/home/szu2070436088/2510044040/CV-SincNet/runs/SATVAL_RA_ABLATION_20260527_103712/same_evaluator_results.json`
- Same-evaluator log: `/home/szu2070436088/2510044040/CV-SincNet/logs/SATVAL_RA_ABLATION_20260527_103712/same_eval.log`
- RA ablation script: remote existing `baselines/receiver_agnostic_rffi/train_cvs.py`; local equivalent is `E:\type10-7\baselines\ra_collab\train_cvs.py`
- RA ablation run dir: `/home/szu2070436088/2510044040/CV-SincNet/runs/SATVAL_RA_ABLATION_20260527_103712/RAABL_ratio010_satview_mixed_20260527_103712`
- RA ablation log: `/home/szu2070436088/2510044040/CV-SincNet/logs/SATVAL_RA_ABLATION_20260527_103712/ra_ablation.log`

## Configuration

- Dataset: WiSig compact PKL `/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- Split: train days `0,1`, test days `2,3`, train RX `0..6`, test RX `7..11`
- Train ratio: `0.1`
- Seed: `1337`
- Satellite scenarios: `clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit`
- Satellite eval subsets: main OOD splits, including strict `test_unseen_day_unseen_rx`
- BEX02 checkpoint: `/home/szu2070436088/2510044040/CV-SincNet/runs/BEX02_ratio010_baseline_concat_mixed_gpu0_20260526_223716/best_primary_ood_model.pth`
- RA checkpoint: `/home/szu2070436088/2510044040/CV-SincNet/runs/wisig_baselines_seed1337_ratio010_satview/receiver_agnostic_seed1337/best_by_val.pt`

## Risks And Assumptions

- The same-evaluator script uses one counting/evaluation loop and the same satellite simulator for both models, but model forward functions remain architecture-specific.
- RA clean scores in the same-evaluator script are per-sample scores, not receiver-collaborative group fusion. This is intentional for fairness of the counting loop.
- The RA ablation is a long run. It should be interpreted with its `metrics.json`, full stdout, and checkpoint at completion, not from startup health alone.
- Remote baseline module naming differs from current local naming: remote uses `receiver_agnostic_rffi`, while local uses `ra_collab`; the verified validation script supports both imports.

## Launch Attempt Notes

- First same-evaluator launch attempt failed before evaluation because the standalone script did not copy `train.py`'s runtime fallback from `sample_rate_hz <= 0` to WiSig `25e6`; fixed locally and re-synced before relaunch.
- First RA ablation launch attempt did not start because the shell command grouped background jobs incorrectly and resolved the log path outside the project root; no training process remained active after the failed attempt.

## Verification And Sync

- Local env: `ssr-gpu`
- Local checks:
  - `python -m py_compile automation_reports\CV-SincNet\SATVAL_RA_ABLATION_20260527_103712\same_evaluator_sat_validation.py` -> pass
  - `python baselines\ra_collab\train_cvs.py --help > nul` -> pass
  - `bash -n /mnt/e/type10-7/automation_reports/CV-SincNet/SATVAL_RA_ABLATION_20260527_103712/launch_same_eval.sh` -> pass
  - `bash -n /mnt/e/type10-7/automation_reports/CV-SincNet/SATVAL_RA_ABLATION_20260527_103712/launch_ra_ablation.sh` -> pass
- Snapshot: `E:\type10-7\code\snapshots\SATVAL_RA_ABLATION_20260527_103712`
- Remote checks:
  - `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile automation_launchers/SATVAL_RA_ABLATION_20260527_103712/same_evaluator_sat_validation.py` -> pass
  - `bash -n automation_launchers/SATVAL_RA_ABLATION_20260527_103712/launch_same_eval.sh` -> pass
  - `bash -n automation_launchers/SATVAL_RA_ABLATION_20260527_103712/launch_ra_ablation.sh` -> pass
  - checkpoint existence check -> `CHECKPOINTS_OK`
- Remote SHA256:
  - `same_evaluator_sat_validation.py`: `63e8699a50617aa11ffa9923e9ba59178d7533564baa9c92efaccd9997415bac`
  - `launch_same_eval.sh`: `e7f26cdc63e6195b057c4a1618e8cd4d6bfc4a3655b0d9463fdbbb01ce88d485`
  - `launch_ra_ablation.sh`: `28ce95b859300d339d4d16746bb34478e90de50efecd99687102d6d43ba5dc48`

## Launch State

- Pre-launch SSH gate: `ssh -o BatchMode=yes N607` passed on `dell-DSS8440`.
- Pre-launch GPU state: all GPUs `10 MiB` used and `0%` util; no target train/eval process active.
- Same-evaluator command:
  - CWD: `/home/szu2070436088/2510044040/CV-SincNet`
  - Env: `GPU=1`, `CUDA_VISIBLE_DEVICES=1`, `PYTHONUNBUFFERED=1`
  - PID: `1376001`
  - Log: `/home/szu2070436088/2510044040/CV-SincNet/logs/SATVAL_RA_ABLATION_20260527_103712/same_eval.log`
  - Output JSON: `/home/szu2070436088/2510044040/CV-SincNet/runs/SATVAL_RA_ABLATION_20260527_103712/same_evaluator_results.json`
- RA ablation command:
  - CWD: `/home/szu2070436088/2510044040/CV-SincNet`
  - Env: `GPU=2`, `CUDA_VISIBLE_DEVICES=2`, `PYTHONUNBUFFERED=1`
  - PID: `1376476`
  - Log: `/home/szu2070436088/2510044040/CV-SincNet/logs/SATVAL_RA_ABLATION_20260527_103712/ra_ablation.log`
  - Output dir: `/home/szu2070436088/2510044040/CV-SincNet/runs/SATVAL_RA_ABLATION_20260527_103712/RAABL_ratio010_satview_mixed_20260527_103712`
- Startup health at 2026-05-27 10:45 Asia/Hong_Kong:
  - Same-evaluator loaded both checkpoints with `missing=0 unexpected=0`; running full main-split clean+satellite evaluation.
  - RA ablation started and reached `[Epoch 001/170][START]`.
  - GPU state: PID `1376001` on GPU1 around `564 MiB`; PID `1376476` on GPU2 around `606 MiB`; no Traceback/OOM/NaN marker in the latest startup scan.

## Live Results Snapshot

- Snapshot time: 2026-05-27 10:48 Asia/Hong_Kong.
- Same-evaluator BEX02 side completed and reproduced the known clean/satellite metrics under the new script:
  - Clean main overall `87.50%`; clean strict UDU `81.97%`.
  - Satellite strict UDU: clear `48.21%`, low `47.58%`, rain `45.74%`, storm `41.51%`, mixed `44.43%`.
- Same-evaluator RA side was still running at the snapshot; no result JSON had been written yet.
- RA ablation first tested epoch:
  - E001 train loss `4.0491`, val TX `21.10%`, tested `1`, best-val-test overall `16.81%`.

## Same-Evaluator Completion Snapshot

- Update time: 2026-05-27, read-only SSH check by Codex.
- Evidence file: `/home/szu2070436088/2510044040/CV-SincNet/runs/SATVAL_RA_ABLATION_20260527_103712/same_evaluator_results.json`.
- The same evaluation script used the same WiSig split, the same satellite-channel simulator, the same satellite scenarios, and the same counting loop for both checkpoints.
- BEX02/CVS primary checkpoint:
  - Clean aggregate `87.50%`; clean strict UDU `81.97%`.
  - Satellite strict UDU: clear `48.21%`, low `47.58%`, rain `45.74%`, storm `41.51%`, mixed `44.43%`.
  - Clear-LEO split floors: UDSR `62.48%`, SDUR `51.71%`, UDUR `48.21%`.
- Receiver-agnostic baseline checkpoint:
  - Clean aggregate `68.58%`; clean strict UDU `54.60%`.
  - Satellite strict UDU: clear `53.78%`, low `51.70%`, rain `51.24%`, storm `49.59%`, mixed `52.15%`.
  - Clear-LEO split floors: UDSR `84.30%`, SDUR `60.10%`, UDUR `53.78%`.
- Interpretation: the satellite simulator and evaluator are not the leading explanation for the gap. BEX02/CVS is much stronger on clean classification, but the RA checkpoint is materially more stable under satellite-channel perturbation. This points to objective/architecture/checkpointing differences, especially receiver-adversarial training and the simpler supervised clean+satellite view path.
- RA ablation status at this read: `RAABL_ratio010_satview_mixed_20260527_103712` was still running, so its final training result remains pending.
