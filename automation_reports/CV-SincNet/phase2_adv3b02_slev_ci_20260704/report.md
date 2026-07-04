# SLEV-CI support-only logit-energy collaborative inference

## Run metadata

| Field | Value |
|---|---|
| experiment_id | phase2_adv3b02_slev_ci_20260704 |
| timestamp_local | 2026-07-04 |
| operator | Codex |
| objective | 在ADV3B02_CORE90_SOFT_E200/qknn8上实现support-only logit-energy verifier协同推理,优先在不破坏旧类OLD80的条件下提高unknown拒识 |
| scenario | CVS Stage2-C,target receiver domain/deployment proxy,LEO satellite stress |
| feature_npz | `E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz` |
| in_orbit_method | qknn8 |
| conda_local | ssr-gpu |
| conda_remote | CVS-RFFI |

## Protocol boundary

SLEV-CI不改变`项目.md`定义的数据协议。`R_t`与`R_s`保持不相交,`Y_old/Y_new/Y_unknown`保持互斥,target-old/seen-new/unknown query均来自target receiver domain和LEO satellite stress视图。unknown query只用于最终评估,不参与能量阈值、profile或超参选择。

阈值选择口径:

| Item | Value |
|---|---|
| threshold_selection_label_scope | target_old_and_seen_new_support_only |
| unknown_query_eval_only | true |
| labeled_unknown_support_used_for_boundary_fit | false |
| support_energy_quantile | 0.90 |
| support_energy_count | 320 |
| global_support_energy_threshold | -6.0259079951233465 |
| support_energy_median | -13.829085397021537 |
| support_energy_min/max | -15.815929440562918 / -2.7134463012848036 |

## Algorithm

SLEV-CI在ENPC-CI的support verifier/prototype evidence基础上增加source classifier logit-energy异常压力:

```text
E(x)=-T*logsum_c exp(logit_c(x)/T)
r_energy=sigmoid((E(x)-tau_support+margin)/temperature)
p_slev=max(p_enpc,(1-w)*p_enpc+w*r_energy)
```

每个receiver只需上传已有qknn8候选证据、support verifier标量和一个logit-energy风险标量。协同数量通过`--collab_counts all`覆盖`1..|R_t|`,本次为1到5个target receivers。该方法是后处理/证据层候选,不更新backbone,不使用unknown query拟合阈值。

## Local files changed

| File | Purpose |
|---|---|
| `E:\type10-7\code\scripts\phase2_orbit_slev_ci_eval.py` | 新增SLEV-CI评估脚本 |
| `E:\type10-7\code\tests\test_phase2_orbit_slev_ci_eval.py` | 新增support-only energy校准单元测试 |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_slev_ci_20260704\report.md` | 本报告 |

根目录`E:\type10-7`不是Git仓库。同步前已创建本地代码快照:

```text
E:\type10-7\code\snapshots\phase2_adv3b02_slev_ci_20260704\
```

## Local verification

| Command | Result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\phase2_orbit_slev_ci_eval.py code\tests\test_phase2_orbit_slev_ci_eval.py` | PASS |
| `conda run -n ssr-gpu python code\tests\test_phase2_orbit_slev_ci_eval.py` | PASS,3 tests |
| `conda run -n ssr-gpu python code\scripts\phase2_orbit_slev_ci_eval.py ... --profiles all --collab_counts all` | PASS,20 summary rows |

Local output:

```text
E:\type10-7\local_artifacts\phase2_adv3b02_slev_ci_20260704\local\slev_ci_summary.json
E:\type10-7\local_artifacts\phase2_adv3b02_slev_ci_20260704\local\slev_ci_summary.csv
E:\type10-7\local_artifacts\phase2_adv3b02_slev_ci_20260704\local\slev_ci_evidence.csv
```

## Local result table

Same-row local ranking under the OLD80-first constraint:

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | resource_pass | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| slev_old80_energy_probe | 5 | 0.8201 | 0.5500 | 0.5000 | 0.3250 | 0.3833 | 0.6167 | 0.0040 | 0.0000 | true | OLD80保持,unknown不足 |
| slev_old80_energy_probe | 4 | 0.8201 | 0.5500 | 0.4833 | 0.3250 | 0.3000 | 0.6500 | 0.0080 | 0.0500 | true | OLD80保持,unknown不足 |
| slev_balanced | 5 | 0.8201 | 0.5500 | 0.5000 | 0.3250 | 0.1333 | 0.8000 | 0.0120 | 0.0667 | true | OLD80保持,unknown不足 |
| slev_known_anchor | 5 | 0.8307 | 0.5500 | 0.5500 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | true | known锚点,无拒识 |
| slev_energy_strict | 5 | 0.5503 | 0.1667 | 0.3667 | 0.2500 | 0.6000 | 0.2833 | 0.2048 | 0.1167 | true | unknown提升但old崩,不合格 |

Quantile sensitivity summary:

| support_quantile | best_old80_profile | collab_count | old_acc | seen_new_acc | unknown_reject | unknown_FAR | verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.70 | slev_old80_energy_probe | 5 | 0.8148 | 0.5167 | 0.3833 | 0.6167 | 未达unknown目标 |
| 0.80 | slev_old80_energy_probe | 5 | 0.8148 | 0.5000 | 0.3833 | 0.6167 | 未达unknown目标 |
| 0.90 | slev_old80_energy_probe | 5 | 0.8201 | 0.5000 | 0.3833 | 0.6167 | 未达unknown目标 |
| 0.95 | slev_old80_energy_probe | 5 | 0.8201 | 0.5000 | 0.3833 | 0.6167 | 未达unknown目标 |

## Interpretation

SLEV-CI没有解决最终目标。它保持了ENPC-CI的OLD80水平,但在相同old约束下unknown拒识没有超过ENPC最好点。更强的energy-strict拒识可以把unknown_reject提升到0.60或0.6667,但old_acc降到0.52到0.56,违反“旧类准确性不能下降”和OLD80_FIRST边界。

因此本候选只能登记为`NON_LAUNCH_DIAGNOSTIC`或`OLD80-constrained unknown rejection diagnostic`,不能声明Stage2-C成功、不能声明部署成功、不能作为论文主结论。

## N607 plan

Before remote access:

```text
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
```

Sync mapping:

| Local | Remote |
|---|---|
| `E:\type10-7\code\scripts\phase2_orbit_slev_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_orbit_slev_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_orbit_slev_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_orbit_slev_ci_eval.py` |

Remote command:

```text
cd /home/szu2070436088/2510044040/CV-SincNet && CUDA_VISIBLE_DEVICES=<low_vram_gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_orbit_slev_ci_eval.py --feature_npz remote_artifacts/phase2_adv3b02_features/features.npz --output_json runs/phase2_adv3b02_slev_ci_20260704/slev_ci_summary.json --output_summary_csv runs/phase2_adv3b02_slev_ci_20260704/slev_ci_summary.csv --output_evidence_csv runs/phase2_adv3b02_slev_ci_20260704/slev_ci_evidence.csv --profiles all --collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 1 --k_shot 8 --query_per_class 20 --qknn_k 8 --seed 4070404 --event_alignment_policy receiver_domain_ranked --support_selection_policy scenario_diverse --slev_energy_support_quantile 0.90 --slev_logit_temperature 1.0 --slev_energy_risk_temperature 0.75 --max_event_bytes 1152 --max_event_latency_ms 20
```

Expected remote outputs:

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_slev_ci_20260704/slev_ci_summary.json
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_slev_ci_20260704/slev_ci_summary.csv
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_slev_ci_20260704/slev_ci_evidence.csv
```

## N607 execution result

N607 preflight:

| Item | Result |
|---|---|
| direct target | N607 |
| remote host | dell-DSS8440 |
| project_root | `/home/szu2070436088/2510044040/CV-SincNet` |
| remote time | 2026-07-04 12:06:49 CST |
| GPU visibility | 8 x RTX 3090 |
| selected GPU | GPU0 |
| pre-run VRAM | all GPUs 10 MiB / 24576 MiB |
| post-run VRAM | all GPUs 10 MiB / 24576 MiB |
| lingering local ssh after tasks | none |

Remote sync:

| Local | Remote |
|---|---|
| `E:\type10-7\code\scripts\phase2_orbit_slev_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_orbit_slev_ci_eval.py` |
| `E:\type10-7\code\tests\test_phase2_orbit_slev_ci_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_orbit_slev_ci_eval.py` |

Remote verification:

| Command | Result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_orbit_slev_ci_eval.py code/tests/test_phase2_orbit_slev_ci_eval.py` | PASS |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_orbit_slev_ci_eval.py` | PASS,3 tests |
| `CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_orbit_slev_ci_eval.py ... --profiles all --collab_counts all` | PASS,20 summary rows |

Remote feature path used:

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_enpc_ci_20260704/features.npz
```

Remote outputs:

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_slev_ci_20260704/slev_ci_summary.json
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_slev_ci_20260704/slev_ci_summary.csv
/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_slev_ci_20260704/slev_ci_evidence.csv
```

Pulled local copies:

```text
E:\type10-7\local_artifacts\phase2_adv3b02_slev_ci_20260704\remote\slev_ci_summary.json
E:\type10-7\local_artifacts\phase2_adv3b02_slev_ci_20260704\remote\slev_ci_summary.csv
E:\type10-7\local_artifacts\phase2_adv3b02_slev_ci_20260704\remote\slev_ci_evidence.csv
```

N607 same-row result table:

| profile | collab_count | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_defer | unknown_defer | resource_pass | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| slev_old80_energy_probe | 5 | 0.8201 | 0.5500 | 0.5000 | 0.3250 | 0.3833 | 0.6167 | 0.0040 | 0.0000 | true | OLD80保持,unknown不足 |
| slev_old80_energy_probe | 4 | 0.8201 | 0.5500 | 0.4833 | 0.3250 | 0.3000 | 0.6500 | 0.0080 | 0.0500 | true | OLD80保持,unknown不足 |
| slev_balanced | 5 | 0.8201 | 0.5500 | 0.5000 | 0.3250 | 0.1333 | 0.8000 | 0.0120 | 0.0667 | true | OLD80保持,unknown不足 |
| slev_known_anchor | 5 | 0.8307 | 0.5500 | 0.5500 | 0.3500 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | true | known锚点,无拒识 |
| slev_energy_strict | 5 | 0.5503 | 0.1667 | 0.3667 | 0.2500 | 0.6000 | 0.2833 | 0.2048 | 0.1167 | true | unknown提升但old崩,不合格 |

Remote aggregate:

| Metric | Value |
|---|---:|
| summary_rows | 20 |
| old80_rows | 7 |
| target_pass_rows | 0 |
| receiver_count | 5 |
| collab_counts | 1,2,3,4,5 |

## Subagent review

| Role | Finding |
|---|---|
| literature/method review | 推荐old-protected selective open-set inference,多证据门控,prototype/qkNN主分类,energy/logit作为异常压力项 |
| reasonableness review | 阻断unknown query调阈值,要求old约束下优化unknown |
| completion monitor | 旧候选PCET/SOVC/ENPC未完成目标,要求SLEV仍需本地/N607/报告/Git闭环 |

## Next route

SLEV说明单独logit-energy后处理不足以同时满足old和unknown。下一步应把unknown边界前移到representation/verifier训练:source-side leave-class-out open-set episode verifier、class-conditional conformal/EVT shrinkage、old replay约束下的小adapter/temperature更新,并保留`unknown_query_eval_only=true`硬门。
