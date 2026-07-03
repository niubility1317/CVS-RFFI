# ADV3B02 closed-set collaborative satellite-channel full evaluation

## Run metadata

| Field | Value |
|---|---|
| run_id | `phaseA_adv3b02_closed_collab_sat_full_20260703` |
| timestamp | 2026-07-03 |
| operator | Codex |
| objective | Evaluate the frozen `ADV3B02_CORE90_SOFT_E200` checkpoint with receiver collaborative inference under clean and satellite-channel views, using `collab_counts=all` so inference count spans 1 to all observed receivers. |
| boundary | Closed-set old-class collaborative receiver fusion only. This is not Stage2-C seen-new enrollment and not unknown rejection evidence. |

## Protocol and hypothesis

Checkpoint:

`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`

Hypothesis: receiver-aligned collaborative probability fusion may rescue some target receiver/day errors under clean and satellite-channel perturbations. The evaluation must report `K=1..N` receiver counts and satellite scenario results, but it cannot support open-set old/new/unknown claims.

## Changed/synced files used

| File | Role |
|---|---|
| `code/evaluation/collaborative_inference_eval.py` | Existing closed-set checkpoint collaborative evaluator; synced to N607 for this run. |
| `code/tests/test_collaborative_inference_eval.py` | Regression test already passed remotely in the Phase A evaluator report. |
| `code/evaluation/__init__.py` | Ensures `code/evaluation` package wins over root-level `evaluation` on N607. |

## Preflight and occupancy

Direct N607 preflight passed earlier on 2026-07-03. Latest post-test GPU probe showed all GPUs at approximately 10 MiB and no remaining Phase A test process. User explicitly allowed launching even when other low-memory processes exist; current GPU state is lower than that threshold.

Selected GPU: `CUDA_VISIBLE_DEVICES=4`.

## Remote command

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=phaseA_adv3b02_closed_collab_sat_full_20260703
mkdir -p logs/$RUN_ID runs/$RUN_ID
nohup env PYTHONPATH=$PWD/code:$PWD CUDA_VISIBLE_DEVICES=4 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/evaluation/collaborative_inference_eval.py \
  --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
  --expect_run_name ADV3B02_CORE90_SOFT_E200 \
  --device cuda:0 \
  --eval_on test_unseen_day_unseen_rx \
  --collab_counts all \
  --collab_fusion adaptive \
  --eval_sat_channel \
  --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --eval_sat_on test_unseen_day_unseen_rx \
  --sat_eval_max_batches 0 \
  --eval_batch_size 256 \
  --num_workers 0 \
  --output runs/$RUN_ID/collab_sat_full.json \
  > logs/$RUN_ID/collab_sat_full.out 2>&1 &
echo $!
```

Expected output files:

- `/home/szu2070436088/2510044040/CV-SincNet/logs/phaseA_adv3b02_closed_collab_sat_full_20260703/collab_sat_full.out`
- `/home/szu2070436088/2510044040/CV-SincNet/runs/phaseA_adv3b02_closed_collab_sat_full_20260703/collab_sat_full.json`

## Metrics to inspect

| Metric family | Fields |
|---|---|
| Clean closed-set collaboration | `K=1..receiver_count`, base/fused accuracy, rescue, harm, net gain |
| Satellite closed-set collaboration | scenario, split, `K=1..receiver_count`, base/fused accuracy, rescue, harm, net gain |
| Resource context | GPU allocation, log runtime, absence of lingering process |

## Status

Pending launch.
