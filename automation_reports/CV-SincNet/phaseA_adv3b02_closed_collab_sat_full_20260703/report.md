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

## Final status

Completed on N607.

| Field | Value |
|---|---|
| launch status | landed after initial SSH timeout; remote Python PID observed as `1528181` |
| final status | completed; PID exited |
| GPU | `CUDA_VISIBLE_DEVICES=4` |
| peak observed GPU memory during poll | 573 MiB on GPU4 |
| final GPU memory | 10 MiB on all GPUs |
| checkpoint SHA256 | `2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98` |
| remote log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phaseA_adv3b02_closed_collab_sat_full_20260703/collab_sat_full.out` |
| remote JSON | `/home/szu2070436088/2510044040/CV-SincNet/runs/phaseA_adv3b02_closed_collab_sat_full_20260703/collab_sat_full.json` |
| local log artifact | `E:\type10-7\automation_reports\CV-SincNet\phaseA_adv3b02_closed_collab_sat_full_20260703\artifacts\collab_sat_full.out` |
| local JSON artifact | `E:\type10-7\automation_reports\CV-SincNet\phaseA_adv3b02_closed_collab_sat_full_20260703\artifacts\collab_sat_full.json` |
| artifact hashes | `out=1570B0220612F5433368BDA5C6AAFB5A23EDBEEC42810BAA72EB71945274ABDC`; `json=098C2254DEB12DCAA3BA56BB098640AE1F768662717D6ACDA7F1960AAEAB2344` |

## Result table

All rows use `test_unseen_day_unseen_rx`, `receiver_count=5`, `eligible_groups=12000`, `excluded_incomplete=0`, and `collab_fusion=adaptive`.

| View | K | Base acc % | Fused acc % | Rescue | Harm | Net gain |
|---|---:|---:|---:|---:|---:|---:|
| clean | 1 | 86.32 | 86.32 | 0 | 0 | 0 |
| clean | 2 | 86.32 | 89.10 | 640 | 306 | 334 |
| clean | 3 | 86.32 | 95.38 | 1212 | 125 | 1087 |
| clean | 4 | 86.32 | 97.44 | 1420 | 85 | 1335 |
| clean | 5 | 86.32 | 98.63 | 1542 | 64 | 1478 |
| `leo_clear_weak` | 1 | 67.22 | 67.22 | 0 | 0 | 0 |
| `leo_clear_weak` | 2 | 67.22 | 72.01 | 891 | 316 | 575 |
| `leo_clear_weak` | 3 | 67.22 | 82.34 | 2054 | 239 | 1815 |
| `leo_clear_weak` | 4 | 67.22 | 89.18 | 2783 | 147 | 2636 |
| `leo_clear_weak` | 5 | 67.22 | 91.28 | 3041 | 153 | 2888 |
| `leo_low_elev_weak` | 1 | 65.04 | 65.04 | 0 | 0 | 0 |
| `leo_low_elev_weak` | 2 | 65.04 | 69.28 | 878 | 370 | 508 |
| `leo_low_elev_weak` | 3 | 65.04 | 80.38 | 2111 | 270 | 1841 |
| `leo_low_elev_weak` | 4 | 65.04 | 86.95 | 2821 | 192 | 2629 |
| `leo_low_elev_weak` | 5 | 65.04 | 89.53 | 3108 | 170 | 2938 |
| `leo_rain_weak` | 1 | 64.21 | 64.21 | 0 | 0 | 0 |
| `leo_rain_weak` | 2 | 64.21 | 68.57 | 876 | 353 | 523 |
| `leo_rain_weak` | 3 | 64.21 | 79.73 | 2143 | 280 | 1863 |
| `leo_rain_weak` | 4 | 64.21 | 85.90 | 2827 | 224 | 2603 |
| `leo_rain_weak` | 5 | 64.21 | 88.67 | 3134 | 199 | 2935 |

## Interpretation

The clean closed-set collaborative result reaches 98.63% at `K=5`, close to but still below the user's 99% old-class target. The satellite-channel closed-set results show strong positive collaboration gains from `K=1` to `K=5`, but remain below 99%: 91.28% for `leo_clear_weak`, 89.53% for `leo_low_elev_weak`, and 88.67% for `leo_rain_weak`.

This result supports the engineering value of multi-receiver collaborative inference under star-ground channel stress, but only for closed-set old-class recognition. It does not validate qknn8 open-set unknown rejection, seen-new enrollment, per-class old floors, or the requested unknown 99% rejection target. Those require the Phase A open-set evidence interface to be connected to real ADV3B02/qknn8 per-receiver old/seen-new/unknown evidence.

## Cleanup

The initial launch SSH command timed out but landed the background job. A stale local `ssh.exe` process with an established connection to N607 was identified and closed without killing the remote Python job. After completion, a final check showed no remote evaluation process, GPU memory back to 10 MiB on all GPUs, and no local ESTABLISHED SSH connection to N607 or the bridge.
