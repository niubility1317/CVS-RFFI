# CVS-RFFI SA27 Optimization Validation

## Run

- Experiment ID: `cvs_sa27_optimization_central_20260527_204005`
- Created: 2026-05-27 20:40 Asia/Hong_Kong
- Operator/agent: Codex
- Objective: validate optimization directions on top of `SA27_domain_dsq_ch2_leo3_ce1_r010` using idle GPUs only.
- Base route: centralized `lite_d + no_dac + baseline_view`, WiSig train ratio `0.1`, LEO-only satellite samples, concat satellite CE-only.
- Base candidate: `SA27_domain_dsq_ch2_leo3_ce1_r010`.
- Fair comparison set: `SA26-SA30`, organized under `E:\type10-7\automation_reports\CV-SincNet\sat_log_organization_20260527_191519\report.md`.
- Reference thresholds:
  - `SA16`: primary `84.45`, strict UDU `82.78`, LEO-subset avg `45.64`.
  - `SA27`: primary `84.68`, strict UDU `83.34`, LEO avg/min `46.10 / 44.69`, worst RX `74.78`.

## Hypotheses

- Multi-seed confirmation (`SA31`, `SA32`): `domain DSQ ch2 + LEO3 CE-only` should remain stable beyond seed `1337`.
- CE pressure scan (`SA33`, `SA34`): fixed CE `0.7` may protect clean/worst-RX, while fixed CE `1.2` may improve LEO SAT; the result will guide whether a future CE schedule is worth implementing.
- View probability scan (`SA35`): `sat_view_prob=0.75` may reduce early satellite perturbation and improve primary/worst-RX while preserving most LEO gain.
- Domain enhancer strength (`SA36`): increasing `domain_enhancer_strength` from `0.35` to `0.45` may improve weak receiver tails, especially worst-RX, while keeping the ID backbone untouched.

## Planned Branches

| Branch | GPU | Seed | Difference vs SA27 | Purpose |
| --- | ---: | ---: | --- | --- |
| `SA31_sa27_seed2027_leo3_ce1_r010` | 2 | 2027 | same as SA27 | seed confirmation |
| `SA32_sa27_seed3407_leo3_ce1_r010` | 3 | 3407 | same as SA27 | seed confirmation |
| `SA33_sa27_ch2_leo3_ce0p7_r010` | 4 | 1337 | `concat_sat_ce_weight=0.7` | lighter CE pressure |
| `SA34_sa27_ch2_leo3_ce1p2_r010` | 5 | 1337 | `concat_sat_ce_weight=1.2` | stronger CE pressure |
| `SA35_sa27_ch2_leo3_viewp075_r010` | 6 | 1337 | `sat_view_prob=0.75` | reduce satellite view frequency |
| `SA36_sa27_ch2_enh0p45_leo3_ce1_r010` | 7 | 1337 | `domain_enhancer_strength=0.45` | worst-RX/domain tail repair |

## Environment And Paths

- Remote root: `/home/szu2070436088/2510044040/CV-SincNet`
- Remote Python: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- Run root: `/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_sa27_optimization_central_20260527_204005`
- Log root: `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_sa27_optimization_central_20260527_204005`

## Local Files Changed

- None. This validation uses existing `train.py` CLI options only.
- No `scp` sync required.
- Local version state: `E:\type10-7\code` is `NOT_GIT`; no code/config/script file was changed, so no snapshot or sync manifest entry was needed.

## Pre-Launch Verification

- Skill gate: `cv-sincnet-n607-automation` instructions read.
- Project gate: `E:\type10-7\AGENTS.md` read.
- SSH gate: `ssh -o BatchMode=yes N607` succeeded.
- GPU check at 2026-05-27 20:38 Asia/Hong_Kong:
  - Busy: GPU0 (`5252 / 24576 MiB`) and GPU1 (`4377 / 24576 MiB`) have active baseline/federated jobs.
  - Idle: GPUs `2,3,4,5,6,7` each show `10 / 24576 MiB`, `0%` util.
- Launch policy: use GPUs `2-7` only; do not stack on GPU0/1.

## Common Command Contract

All branches use:

```bash
--dataset wisig
--wisig_domain rx_day
--wisig_train_ratio 0.1
--primary_udu_weight 0.65
--epochs 170
--eval_sat_channel
--eval_sat_on test_unseen_day_unseen_rx
--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo
--sat_eval_max_batches -1
--slim_group none
--model_variant lite_d
--branch_ablation no_dac
--domain_branch_ablation no_stats
--domain_enhancer rcn_stats
--exp_group s3_rxrobust_no_dac
--use_mixstyle
--mixstyle_layers time_down,t1
--mixstyle_mix same_tx_crossdomain
--mixstyle_fallback skip
--mixstyle_strength 0.70
--mixstyle_p 0.18
--mixstyle_late_start 110
--mixstyle_late_ramp_epochs 40
--mixstyle_late_min_p 0.05
--mixstyle_late_min_strength 0.32
--lambda_fishr 0.02
--fishr_min_domains 4
--sat_train_scenarios clear_leo,low_elev_leo,rain_leo
--use_concat_sat_channel_aug
--concat_sat_ce_only
--concat_sat_start_epoch 1
--lambda_sat_cls 0.00
--lambda_sat_cons 0.00
--domain_freq_stability_mode dsq
--freq_stability_channels 2
```

## Launch Record

- Launch timestamp: 2026-05-27 20:41 Asia/Hong_Kong
- Remote launch stamp: `20260527_204104`
- Working directory: `/home/szu2070436088/2510044040/CV-SincNet`
- Conda/Python environment: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- Launch method: `nohup`, one centralized `train.py` process per GPU.
- Exact command form:

```bash
CUDA_VISIBLE_DEVICES=<GPU> nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u train.py \
  --batch_size 256 --eval_batch_size 256 \
  <Common Command Contract above> \
  --seed <SEED> \
  --use_mixstyle \
  --run_name <BRANCH> \
  --latest_save_path /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_sa27_optimization_central_20260527_204005/<BRANCH>/latest_model.pth \
  --best_save_path /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_sa27_optimization_central_20260527_204005/<BRANCH>/best_val_model.pth \
  --best_primary_save_path /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_sa27_optimization_central_20260527_204005/<BRANCH>/best_primary_ood_model.pth \
  --best_unseen_day_unseen_rx_save_path /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_sa27_optimization_central_20260527_204005/<BRANCH>/best_strict_udu_model.pth \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --use_concat_sat_channel_aug \
  --concat_sat_ce_only \
  --concat_sat_ce_weight <CE_WEIGHT> \
  --concat_sat_start_epoch 1 \
  --sat_view_prob <VIEW_PROB> \
  --lambda_sat_cls 0.00 \
  --lambda_sat_cons 0.00 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  > /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_sa27_optimization_central_20260527_204005/<BRANCH>_20260527_204104.log 2>&1 &
```

| Branch | GPU | PID | Extra values | Log |
| --- | ---: | ---: | --- | --- |
| `SA31_sa27_seed2027_leo3_ce1_r010` | 2 | 1699851 | `seed=2027`, `CE=1.0`, `view_prob=1.00`, `enhancer_strength=0.35` | `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_sa27_optimization_central_20260527_204005/SA31_sa27_seed2027_leo3_ce1_r010_20260527_204104.log` |
| `SA32_sa27_seed3407_leo3_ce1_r010` | 3 | 1699917 | `seed=3407`, `CE=1.0`, `view_prob=1.00`, `enhancer_strength=0.35` | `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_sa27_optimization_central_20260527_204005/SA32_sa27_seed3407_leo3_ce1_r010_20260527_204104.log` |
| `SA33_sa27_ch2_leo3_ce0p7_r010` | 4 | 1699987 | `seed=1337`, `CE=0.7`, `view_prob=1.00`, `enhancer_strength=0.35` | `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_sa27_optimization_central_20260527_204005/SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104.log` |
| `SA34_sa27_ch2_leo3_ce1p2_r010` | 5 | 1700057 | `seed=1337`, `CE=1.2`, `view_prob=1.00`, `enhancer_strength=0.35` | `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_sa27_optimization_central_20260527_204005/SA34_sa27_ch2_leo3_ce1p2_r010_20260527_204104.log` |
| `SA35_sa27_ch2_leo3_viewp075_r010` | 6 | 1700456 | `seed=1337`, `CE=1.0`, `view_prob=0.75`, `enhancer_strength=0.35` | `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_sa27_optimization_central_20260527_204005/SA35_sa27_ch2_leo3_viewp075_r010_20260527_204104.log` |
| `SA36_sa27_ch2_enh0p45_leo3_ce1_r010` | 7 | 1700855 | `seed=1337`, `CE=1.0`, `view_prob=1.00`, `enhancer_strength=0.45` | `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_sa27_optimization_central_20260527_204005/SA36_sa27_ch2_enh0p45_leo3_ce1_r010_20260527_204104.log` |

## Startup Health Check

- Health check timestamp: 2026-05-27 20:49-20:50 Asia/Hong_Kong.
- GPU/process status:
  - GPU2-7 are active with roughly `2721-2935 / 24576 MiB` and `20-29%` utilization.
  - All six PIDs are alive with parent PID `1`.
  - Existing jobs on GPU0 and GPU1 were left untouched.
- Config activation confirmed in all six logs:
  - `lite_d`, `no_dac`, `domain_branch_ablation=no_stats`;
  - `domain_freq_stability_mode=dsq`, `freq_stability_channels=2`;
  - `domain_enhancer=rcn_stats`, with `enhancer_strength=0.35` except SA36 at `0.45`;
  - `eval_sat_scenarios=clear_leo,low_elev_leo,rain_leo`;
  - `[CONCAT-SAT-AUG]` shows `ce_only=1`, LEO3 scenario cycle, and the intended CE/view values.
- `SAT-TRAIN` / `CONCAT-SAT-TRAIN` startup lines are absent because classic satellite training losses are deliberately disabled (`lambda_sat_cls=0`, `lambda_sat_cons=0`) and this route uses concat satellite CE-only augmentation.
- Progress at check:

| Branch | Latest parsed epoch begin | Hard errors | Unsafe skip lines | Early note |
| --- | ---: | ---: | ---: | --- |
| `SA31_sa27_seed2027_leo3_ce1_r010` | 5 / 170 | 0 | 2 | best primary `55.42` by E005 |
| `SA32_sa27_seed3407_leo3_ce1_r010` | 4 / 170 | 0 | 1 | best primary `49.60` by E004 |
| `SA33_sa27_ch2_leo3_ce0p7_r010` | 4 / 170 | 0 | 1 | best primary `63.83` by E004 |
| `SA34_sa27_ch2_leo3_ce1p2_r010` | 7 / 170 | 0 | 3 | best primary `61.40` by E005 |
| `SA35_sa27_ch2_leo3_viewp075_r010` | 5 / 170 | 0 | 3 | best primary `65.58` by E004 |
| `SA36_sa27_ch2_enh0p45_leo3_ce1_r010` | 4 / 170 | 0 | 1 | best primary `57.08` by E004 |

- Hard-error scan looked for `Traceback`, `RuntimeError`, `ValueError`, `unrecognized`, `Killed`, `out of memory`, `CUDA out`, and `CUDA error`; none were found.
- Early metrics are not used for route selection yet; select only from full 170-epoch results and final/best checkpoint summaries.

## Success Criteria

- Multi-seed route confirmation:
  - mean primary score `>= 84.45`;
  - no seed below `SA16` by more than `0.3`;
  - mean strict UDU `>= 82.78`;
  - mean LEO avg `> 45.64`.
- CE/view/enhancer variant acceptance:
  - preferred if LEO avg reaches `>= 47.0` with primary `>= 84.0` and strict UDU `>= 82.5`;
  - or if worst-RX improves over `SA27` by at least `+1.0` while LEO avg does not drop below `45.64`.
- Reject if hard errors occur, unsafe skip count is materially higher than SA27 (`8`), or primary falls below `84.0` without a compensating LEO SAT gain above `47.0`.

## Known Risks

- This is still centralized validation; do not extrapolate directly to federated training until the route is stable.
- Fixed CE scans are proxies for a future schedule; if `0.7` and `1.2` pull in opposite directions, implement a schedule rather than choosing one fixed value blindly.
- `domain_enhancer_strength=0.45` may over-emphasize domain features and hurt ID invariance; it is a tail-repair probe only.

## Completion Analysis

- Final status: all six runs finished 170 epochs with no hard errors.
- Remote logs copied to:
  - `E:\type10-7\automation_reports\CV-SincNet\cvs_sa27_optimization_central_20260527_204005\remote_logs_snapshot_20260527_complete\`
- Full combined parser:
  - `E:\type10-7\automation_reports\CV-SincNet\cvs_sa27_optimization_central_20260527_204005\analyze_related_central_logs.py`
- Full combined outputs:
  - `E:\type10-7\automation_reports\CV-SincNet\cvs_sa27_optimization_central_20260527_204005\related_central_full_log_analysis.md`
  - `E:\type10-7\automation_reports\CV-SincNet\cvs_sa27_optimization_central_20260527_204005\related_central_full_log_analysis.json`

| Branch | Primary | Overall | Strict UDU | LEO avg/min | Worst RX | Skips | Interpretation |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| `SA31_sa27_seed2027_leo3_ce1_r010` | 81.12 | 84.85 | 79.11 | 44.44/43.22 | 70.26 `test_rx_8` | 10 | seed collapse; base SA27 not stable |
| `SA32_sa27_seed3407_leo3_ce1_r010` | 80.39 | 84.20 | 78.34 | 46.41/45.09 | 69.64 `test_rx_7` | 10 | seed collapse despite acceptable LEO |
| `SA33_sa27_ch2_leo3_ce0p7_r010` | 83.98 | 86.80 | 82.46 | 44.61/43.37 | 74.82 `test_rx_8` | 8 | lighter CE protects clean partly but loses LEO |
| `SA34_sa27_ch2_leo3_ce1p2_r010` | 83.91 | 86.71 | 82.41 | 47.49/46.22 | 73.98 `test_rx_7` | 10 | strongest LEO repair, clean/strict slightly low |
| `SA35_sa27_ch2_leo3_viewp075_r010` | 83.62 | 86.78 | 81.92 | 43.65/42.53 | 73.03 `test_rx_8` | 9 | view probability reduction hurts both balance and LEO |
| `SA36_sa27_ch2_enh0p45_leo3_ce1_r010` | 84.39 | 87.00 | 82.98 | 46.26/44.74 | 74.24 `test_rx_7` | 9 | best balanced follow-up, but still below SA27 clean and not seed-verified |

### Route Interpretation

- `SA27` is not promotable as-is because the two extra seeds fell to primary `81.12` and `80.39`; the apparent best seed was real but fragile.
- `SA36` is the best continuation route because it keeps primary and strict closest to `SA27/SA16` while slightly improving LEO over `SA27`.
- `SA34` shows the most useful LEO knob: stronger concat SAT CE can lift LEO above `47`, but it needs a clean/strict stabilizer.
- `SA35` rejects lower view probability as a main direction.
- The next experiment should combine `SA36` domain enhancer strength with CE `1.1-1.2`, and verify whether `enhancer=0.45` repairs the failed SA27 seeds.
