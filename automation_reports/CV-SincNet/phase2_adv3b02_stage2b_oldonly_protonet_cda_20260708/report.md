# phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708

## Scope

| Field | Value |
|---|---|
| Date | 2026-07-08 |
| Operator | Codex |
| Base model | `ADV3B02_CORE90_SOFT_E200` |
| Checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| Protocol | Stage2-B target-old-only ProtoNet-CDA |
| Boundary | No target-new registration, no unknown rejection, no Stage2-C/deployment claim |
| LEO view | Support and query both use simplified LEO star-ground channel views |
| K values | `5,10` |

## Local And Remote Versioning

| Item | Value |
|---|---|
| Launcher | `code/scripts/launch_phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708.sh` |
| Launcher commit | `1fe755f Add ADV3B02 old-only ProtoNet CDA launcher` |
| Launcher sha256 | `e6a0c07600acbff3da1751b26ce3e4eeb05f23afd0c7dbe6f41fc597ed45622e` |
| Remote destination | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708.sh` |
| Remote command | `cd /home/szu2070436088/2510044040/CV-SincNet && RUN_ID=phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708 GPU=0 bash code/scripts/launch_phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708.sh` |
| Launch marker | `[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA-LAUNCHED] pid=535954 gpu=0` |
| Done marker | `[ADV3B02-STAGE2B-OLDONLY-PROTONET-CDA-DONE]` |

## Remote Artifacts

| Artifact | Size |
|---|---:|
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708/ADV3B02_CORE90_SOFT_E200_STAGE2B_OLDONLY_PROTONET_CDA/features_target_old_leo.npz` | 20849832 |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708/ADV3B02_CORE90_SOFT_E200_STAGE2B_OLDONLY_PROTONET_CDA/target_old_protonet_cda_detail.csv` | 428744 |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708/ADV3B02_CORE90_SOFT_E200_STAGE2B_OLDONLY_PROTONET_CDA/target_old_protonet_cda_metrics.json` | 1725 |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2b_oldonly_protonet_cda_20260708/ADV3B02_CORE90_SOFT_E200_STAGE2B_OLDONLY_PROTONET_CDA/target_old_protonet_cda_summary.csv` | 154 |

## Results

| K | support_count | query_count | support_query_overlap_count | old_acc | min_old_class_acc | weakest_tx | 14-10 | 14-7 | 20-15 | 20-19 | 6-15 | 8-20 |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 30 | 9570 | 0 | 0.726228 | 0.489028 | `20-19` | 0.607524 | 0.700313 | 0.743574 | 0.489028 | 0.888401 | 0.928527 |
| 10 | 60 | 9540 | 0 | 0.725262 | 0.457233 | `20-19` | 0.613836 | 0.722642 | 0.738365 | 0.457233 | 0.890566 | 0.928931 |

## Interpretation

`K=5` is slightly stronger on overall old-class accuracy and also keeps a higher worst-class floor than `K=10`. Both runs are target-old support/query evaluations with zero support/query overlap. These rows are evidence for Stage2-B old-class target-domain adaptation only; they do not provide target-new registration, unknown rejection, Stage2-C, or deployment-success evidence.
