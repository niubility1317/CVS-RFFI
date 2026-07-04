# ADV3B02 proxy-unknown calibrated collaborative inference

## Run metadata

| Field | Value |
|---|---|
| experiment_id | phase2_adv3b02_proxy_unknown_ci_20260704 |
| timestamp_local | 2026-07-04 |
| operator | Codex |
| objective | 在ADV3B02_CORE90_SOFT_E200/qknn8 Stage2-C上加入source-side真实`proxy_unknown`校准,检验是否能在保持旧类/新类性能的同时提高unknown拒识 |
| scenario | CVS Stage2-C,target receiver domain/deployment proxy,LEO satellite stress |
| base_checkpoint_remote | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| remote_project_root | `/home/szu2070436088/2510044040/CV-SincNet` |
| local_test_env | ssr-gpu |
| remote_env | CVS-RFFI |
| in_orbit_method | qknn8 |

## Hypothesis

PCET/SOVC/ENPC/SLEV和support-only virtual negative诊断均显示:只用target old/seen-new support几何或logit-energy后处理不足以建立真实unknown边界。当前主ADV3B02 Phase2 feature NPZ不包含`proxy_unknown`。本实验验证一个尚未完整跑过的组合:

```text
ADV3B02 frozen z_id
+ source-receiver真实proxy_unknown rows
+ Stage2-C target-old/seen-new support qknn8 enrollment
+ proxy/source校准的collaborative open-set evidence
+ ENPC/SLEV old-protected fusion
```

`proxy_unknown`来自source receiver side,TX与`Y_old/Y_new/Y_unknown`互斥,不使用target_unknown query调阈值。

## Protocol

| Item | Value |
|---|---|
| source_tx_ids / Y_old | `14-10,14-7,20-15,20-19,6-15,8-20` |
| target_new_tx_ids / Y_new | `19-3,3-8` |
| target_unknown_tx_ids / Y_unknown | `10-1,10-10` |
| proxy_unknown_tx_ids | `1-1,1-10,1-11,1-12` |
| source_receiver_ids | `1-1,1-19,14-7,18-2,19-2,2-1` |
| target_receiver_ids | `20-1,3-19,7-14,7-7,8-8` |
| proxy_receiver_ids | `1-1,1-19,14-7,18-2,19-2,2-1` |
| target_channel_view | satellite/LEO |
| target_scenarios | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| proxy_unknown_channel_view | satellite/LEO |
| proxy_unknown_scenarios | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| k_shot | 8 |
| qknn_k | 8 |
| query_per_class | 20 |
| collab_counts | `1..5` |
| unknown query use | final evaluation only |

Protocol guard expected from `phase2_collaborative_open_set_qknn_eval.py`: reject if proxy receiver overlaps target receivers or proxy TX overlaps old/new/unknown TX.

## Local verification before N607

| Command | Result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\export_spaceborne_features.py code\scripts\phase2_collaborative_open_set_qknn_eval.py code\scripts\phase2_orbit_enpc_ci_eval.py code\scripts\phase2_orbit_slev_ci_eval.py` | PASS |

## Remote plan

Create run dir:

```bash
RUN=runs/phase2_adv3b02_proxy_unknown_ci_20260704
mkdir -p "$RUN"
```

Export feature NPZ:

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/export_spaceborne_features.py \
  --ckpt runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
  --wisig_pkl Dataset_WigSig/ManySig.pkl \
  --new_wisig_pkl Dataset_WigSig/ManyTx.pkl \
  --out_npz "$RUN/features_proxy_unknown.npz" \
  --feature_name z_id \
  --source_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --source_rxs 1-1,1-19,14-7,18-2,19-2,2-1 \
  --target_old_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --target_old_rxs 20-1,3-19,7-14,7-7,8-8 \
  --target_old_channel_view satellite \
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --new_tx_ids 19-3,3-8 \
  --new_rxs 20-1,3-19,7-14,7-7,8-8 \
  --unknown_tx_ids 10-1,10-10 \
  --target_new_channel_view satellite \
  --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --proxy_unknown_tx_ids 1-1,1-10,1-11,1-12 \
  --proxy_unknown_rxs 1-1,1-19,14-7,18-2,19-2,2-1 \
  --proxy_unknown_channel_view satellite \
  --proxy_unknown_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --star_ground_channel_impl simplified_leo_residual \
  --wisig_equalized 1 \
  --wisig_domain rx_day \
  --wisig_out_len 256 \
  --max_samples_per_combo 0 \
  --max_samples_per_tx 400 \
  --batch_size 512 \
  --device cuda:0 \
  --seed 4070404
```

Run ENPC and SLEV:

```bash
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_orbit_enpc_ci_eval.py ...
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_orbit_slev_ci_eval.py ...
```

Expected outputs:

```text
runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz
runs/phase2_adv3b02_proxy_unknown_ci_20260704/enpc_proxy_summary.json
runs/phase2_adv3b02_proxy_unknown_ci_20260704/enpc_proxy_summary.csv
runs/phase2_adv3b02_proxy_unknown_ci_20260704/slev_proxy_summary.json
runs/phase2_adv3b02_proxy_unknown_ci_20260704/slev_proxy_summary.csv
```

## Decision rule

This experiment is only promotable if same-row metrics improve under the old/seen-new constraints. A row with highunknown_reject but old/seen-new collapse remains diagnostic-only.

Target remains:

| Gate | Target |
|---|---:|
| old_acc | 0.99 |
| min_old | 0.95 |
| seen_new_acc | 0.97 |
| min_seen | 0.93 |
| unknown_reject | 0.99 |

OLD80-first diagnostic gate:

| Gate | Minimum |
|---|---:|
| old_acc | 0.80 |
| unknown query calibration | forbidden |

## Current status

Remote run complete. Results pulled to:

```text
E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\
```

## N607 execution

N607 preflight and process context:

| Item | Result |
|---|---|
| direct target | N607 |
| remote host | dell-DSS8440 |
| remote project root | `/home/szu2070436088/2510044040/CV-SincNet` |
| remote time | 2026-07-04 12:15:34 CST |
| visible GPUs | 8 x RTX 3090 |
| selected GPU | GPU0 |
| before export VRAM | all GPUs 10 MiB / 24576 MiB |
| after evaluation VRAM | all GPUs 10 MiB / 24576 MiB |
| active user Python experiments before launch | none observed |
| lingering local ssh after tasks | none |

The first remote process check using`ps -u $USER`failed because this server's`ps`rejected the empty/quoted user list in that shell context. It was rerun with`u=$(whoami)`and returned no active matching Python experiment processes. This is a command quoting issue, not experiment evidence.

Feature export:

| Field | Value |
|---|---|
| remote feature path | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz` |
| sha256 | `7f5c2956ce78f0a2b44c6f41fee453613eede5cf916be0ff6899365fac7a3297` |
| source rows | 2400 |
| target_old rows | 2400 |
| target_new rows | 800 |
| target_unknown rows | 800 |
| proxy_unknown rows | 1600 |
| proxy_overlap_audit | empty for source,target_unknown,target_new |
| proxy_target_receiver_overlap | none |

Remote feature role audit:

| role | rows |
|---|---:|
| source | 2400 |
| target_old | 2400 |
| target_new | 800 |
| target_unknown | 800 |
| proxy_unknown | 1600 |

Proxy TX and receiver audit:

| Field | Value |
|---|---|
| proxy_unknown_tx_ids | `1-1,1-10,1-11,1-12` |
| proxy_unknown_receiver_ids | `1-1,1-19,14-7,18-2,19-2,2-1` |
| target_receiver_ids | `20-1,3-19,7-14,7-7,8-8` |

Remote verification:

| Command | Result |
|---|---|
| `py_compile code/scripts/phase2_orbit_enpc_ci_eval.py code/scripts/phase2_orbit_slev_ci_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py` | PASS |
| `phase2_orbit_enpc_ci_eval.py ... --feature_npz features_proxy_unknown.npz --profiles all --collab_counts all` | PASS |
| `phase2_orbit_slev_ci_eval.py ... --feature_npz features_proxy_unknown.npz --profiles all --collab_counts all` | PASS |

One attempted command passed unsupported flags through theENPC wrapper:

```text
--support_calibration_mode
--class_score_threshold_enabled
--class_conformal_enabled
--receiver_class_reliability_policy
--class_verifier_policy
```

The wrapper rejected them at argparse before evaluation. The experiment was rerun with supported wrapper parameters. This failed attempt did not produce result rows and is not used for metrics.

Remote output hashes:

| File | SHA256 |
|---|---|
| `enpc_proxy_summary.csv` | `2853e379e2b23e9a1c176d3926e072e98ba92fc226ff2259a6bdf3a443b67604` |
| `slev_proxy_summary.csv` | `83e8a0b7f8fd2b1abd87e300de2af2366bd57fec1ff5f2fab8e2934454a58b02` |

Remote outputs pulled:

```text
E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\enpc_proxy_summary.json
E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\enpc_proxy_summary.csv
E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\enpc_proxy_evidence.csv
E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\slev_proxy_summary.json
E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\slev_proxy_summary.csv
E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\slev_proxy_evidence.csv
```

## N607 results

ENPC-proxy aggregate:

| Metric | Value |
|---|---:|
| summary_rows | 20 |
| old80_rows | 5 |
| target_pass_rows | 0 |
| evidence_rows | 1000 |
| qknn_threshold_scope | source_only |
| unknown_query_eval_only | true |

ENPC-proxy same-row ranking:

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| enpc_balanced | 5 | 0.8254 | 0.6000 | 0.3333 | 0.0500 | 0.1000 | 0.9000 | 0.0000 | 0.0000 | OLD80保持,seen-new和unknown不足 |
| enpc_balanced | 4 | 0.8360 | 0.6500 | 0.3167 | 0.0500 | 0.0667 | 0.9333 | 0.0080 | 0.0000 | OLD80保持,seen-new和unknown不足 |
| enpc_known_anchor | 4 | 0.8466 | 0.6923 | 0.3833 | 0.1250 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | known锚点,无拒识 |
| enpc_old80_unknown_probe | 5 | 0.5608 | 0.3000 | 0.0833 | 0.0000 | 0.7667 | 0.2333 | 0.0000 | 0.0000 | unknown提升但old/seen-new崩溃 |
| enpc_unknown_strict | 5 | 0.5026 | 0.1282 | 0.2000 | 0.0000 | 0.6667 | 0.2333 | 0.1767 | 0.1000 | unknown提升但old/seen-new崩溃 |

SLEV-proxy aggregate:

| Metric | Value |
|---|---:|
| summary_rows | 20 |
| old80_rows | 3 |
| target_pass_rows | 0 |
| evidence_rows | 1000 |
| qknn_threshold_scope | source_only in base qknn metadata |
| SLEV energy threshold_scope | target_old_and_seen_new_support_only |
| unknown_query_eval_only | true |

SLEV-proxy same-row ranking:

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| slev_known_anchor | 4 | 0.8466 | 0.6923 | 0.3833 | 0.1250 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | known锚点,无拒识 |
| slev_known_anchor | 5 | 0.8307 | 0.6000 | 0.3833 | 0.1250 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | known锚点,无拒识 |
| slev_energy_strict | 5 | 0.4233 | 0.0769 | 0.1000 | 0.0000 | 0.8000 | 0.1333 | 0.1004 | 0.0667 | unknown提升但old/seen-new崩溃 |
| slev_old80_energy_probe | 5 | 0.5979 | 0.3846 | 0.1500 | 0.0500 | 0.7500 | 0.2500 | 0.0000 | 0.0000 | unknown提升但old/seen-new崩溃 |

SLEV energy calibration:

| Field | Value |
|---|---:|
| support_count | 320 |
| global_threshold | -6.282681877185347 |
| support_median | -13.736764022180607 |
| support_min | -15.710171707989277 |
| support_max | -1.8186365511741038 |

## Interpretation

The previously untested real`proxy_unknown`Stage2-C route does not solve the target. It confirms the same tradeoff observed in SOVC/PCET/SLEV/oracle diagnostics:

1. When OLD80 is preserved,unknown rejection remains poor.
2. When unknown rejection is increased,old and especially seen-new collapse.
3. Real source-side proxy unknown improves the existence of a calibration source, but the proxy direction does not match the target_unknown boundary under LEO target distortion strongly enough.

This route remains useful negative evidence: it rules out the claim that the earlier failures were only caused by missing realproxy_unknown rows in the Phase2 feature NPZ. It does not rule out training-stage open-set representation learning, but it does rule out proxy-only qknn8 calibration as sufficient.

## Next route

The next aligned route should move from post-hoc calibration to representation or adapter training with an explicit old/seen-new preservation objective:

```text
source/proxy_unknown open-set energy or reciprocal-point loss
+ old/seen-new replay preservation
+ qknn8 Stage2-C enrollment
+ collaborative selective fusion
```

Any such route must still report same-row`old_acc,min_old,seen_new_acc,min_seen,unknown_reject,unknown_FAR,coverage,defer,latency,bytes`and must keep`target_unknown`query evaluation-only.
