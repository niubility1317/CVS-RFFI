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

## No-Adaptation Star-Ground Baseline

This baseline uses `tx_logits` from the same `features_target_old_leo.npz` package. It audits the frozen `ADV3B02_CORE90_SOFT_E200` source classifier on target-old LEO rows without target support, prototype update, threshold calibration, or adapter update. The logit mapping is `0=14-10`, `1=14-7`, `2=20-15`, `3=20-19`, `4=6-15`, `5=8-20`.

| Scope | count | overall_acc | min_class_acc | weakest_tx | 14-10 | 14-7 | 20-15 | 20-19 | 6-15 | 8-20 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| All target-old LEO | 9600 | 0.725729 | 0.524375 | `20-19` | 0.613750 | 0.660000 | 0.730625 | 0.524375 | 0.894375 | 0.931250 |
| K=5 matched query split | 9570 | 0.725183 | 0.522884 | `20-19` | 0.612539 | 0.659561 | 0.730408 | 0.522884 | 0.894671 | 0.931034 |
| K=10 matched query split | 9540 | 0.724948 | 0.522013 | `20-19` | 0.611950 | 0.661006 | 0.729560 | 0.522013 | 0.894340 | 0.930818 |

| LEO scenario | count | overall_acc | 14-10 | 14-7 | 20-15 | 20-19 | 6-15 | 8-20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `leo_clear_weak` | 3456 | 0.684317 | 0.512153 | 0.597222 | 0.560764 | 0.519097 | 0.947917 | 0.968750 |
| `leo_low_elev_weak` | 3072 | 0.718099 | 0.732422 | 0.646484 | 0.947266 | 0.404297 | 0.753906 | 0.824219 |
| `leo_rain_weak` | 3072 | 0.779948 | 0.609375 | 0.744141 | 0.705078 | 0.650391 | 0.974609 | 0.996094 |

## Results

| K | support_count | query_count | support_query_overlap_count | old_acc | min_old_class_acc | weakest_tx | 14-10 | 14-7 | 20-15 | 20-19 | 6-15 | 8-20 |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 30 | 9570 | 0 | 0.726228 | 0.489028 | `20-19` | 0.607524 | 0.700313 | 0.743574 | 0.489028 | 0.888401 | 0.928527 |
| 10 | 60 | 9540 | 0 | 0.725262 | 0.457233 | `20-19` | 0.613836 | 0.722642 | 0.738365 | 0.457233 | 0.890566 | 0.928931 |

## Delta Versus No Adaptation

| K | ProtoNet-CDA old_acc | matched no-adapt old_acc | delta_pp | ProtoNet-CDA min_class_acc | matched no-adapt min_class_acc | floor_delta_pp | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.726228 | 0.725183 | +0.104 | 0.489028 | 0.522884 | -3.386 | Overall is nearly flat; `14-7` and `20-15` improve, but `20-19` floor drops |
| 10 | 0.725262 | 0.724948 | +0.031 | 0.457233 | 0.522013 | -6.478 | Overall is nearly flat; `14-7` improves, but `20-19` floor drops more |

## Interpretation

`K=5` is slightly stronger on overall old-class accuracy and also keeps a higher worst-class floor than `K=10`. Against the frozen source-classifier no-adaptation baseline, ProtoNet-CDA is nearly flat on overall old-class accuracy and worse on the `20-19` floor, so this route is not a clear old-class adaptation gain. Both runs are target-old support/query evaluations with zero support/query overlap. These rows are evidence for Stage2-B old-class target-domain adaptation only; they do not provide target-new registration, unknown rejection, Stage2-C, or deployment-success evidence.
