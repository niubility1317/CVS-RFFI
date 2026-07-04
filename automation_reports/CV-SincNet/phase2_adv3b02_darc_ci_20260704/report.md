# phase2_adv3b02_darc_ci_20260704

## 基本信息

| 字段 | 值 |
|---|---|
| 实验ID | phase2_adv3b02_darc_ci_20260704 |
| 时间 | 2026-07-04 |
| 操作者 | Codex |
| 目标 | 在ADV3B02_CORE90_SOFT_E200+qknn8证据基础上，验证接收机分歧确认是否能提升未知类拒识，同时保持旧类不下降 |
| 场景 | CVS Stage2-C；target receiver domain；satellite/LEO view；`Y_old/Y_new/Y_unknown`互斥 |
| 底座证据 | `local_artifacts/phase2_adv3b02_opv_ci_20260704/local_all_clean/opv_base_dual_route_evidence.csv` |
| metadata | `local_artifacts/phase2_adv3b02_feature_bank_opu_sweep_20260704/ra_phase2_adv3b02_features_features_npz/opu_ci_summary.json` |

## 算法设计

DARC-CI使用同一事件内多接收机的base qKNN证据构造拒识确认信号：

| 组件 | 规则 |
|---|---|
| 标签来源 | 只使用base qKNN标签；DARC-CI不改写`predicted_label` |
| 分歧证据 | 按`event_id`统计多接收机top-label agreement/disagreement |
| 弱已知门控 | 当score、margin、support density、receiver reliability不足时，提高辅助未知风险 |
| 强旧类保护 | 旧类top-label且score/margin/support/agreement足够时，对unknown risk限幅 |
| 阈值边界 | `unknown_query_used_for_threshold=false`；unknown query只用于评估 |
| 资源字段 | 继续报告`collab_count`、`bytes_per_event`、`latency_ms_p95`、`participating_receivers_p95` |

设计假设：叠加LEO信道下未知TX在不同接收机视角中更容易造成top-label不一致；旧类强已知样本应表现为较高一致性。因此DARC-CI只在“分歧+弱已知”同时出现时提高未知风险。

## 本地改动

| 文件 | 目的 |
|---|---|
| `github_publish/CVS-RFFI-repo/code/scripts/phase2_disagreement_confirm_ci_eval.py` | 新增DARC-CI接收机分歧确认评估脚本 |
| `github_publish/CVS-RFFI-repo/code/tests/test_phase2_disagreement_confirm_ci_eval.py` | 覆盖base label authority、强旧类限幅、分歧事件风险提升 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_disagreement_confirm_ci_eval.py -q` | PASS，3 passed |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_disagreement_confirm_ci_eval.py` | PASS |
| DARC-CI local all profiles，policies=`opu_old_preserve,opu_old_guarded` | PASS，`summary_rows=40`，`candidate_count=0` |

## 本地结果

| profile | policy | collab_count | old_acc | seen_new_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_unknown_reject_rate | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| darc_light | opu_old_preserve | 4 | 0.8021390374 | 0.7833333333 | 0.1833333333 | 0.7500000000 | 0.0000000000 | 0.0000000000 | diagnostic_only |
| darc_balanced | opu_old_preserve | 4 | 0.8021390374 | 0.7833333333 | 0.1833333333 | 0.7500000000 | 0.0000000000 | 0.0000000000 | diagnostic_only |
| darc_unknown_push | opu_old_preserve | 4 | 0.7967914439 | 0.7833333333 | 0.1833333333 | 0.7666666667 | -0.0053475936 | 0.0000000000 | diagnostic_only |
| darc_light | opu_old_guarded | 4 | 0.7754010695 | 0.7333333333 | 0.2500000000 | 0.5833333333 | 0.0053475936 | 0.0000000000 | diagnostic_only |

结论：接收机分歧确认没有解决未知拒识。light/balanced基本等于base；unknown_push没有带来有效拒识收益，还会提高FAR或伤旧类。因此DARC-CI不能作为成功候选。

## 资源约束说明文档状态

用户指定的`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`在当前工作区可访问扫描中未定位。当前实验仍按已有评估器输出记录基本资源字段：`collab_count`、`bytes_per_event`、`latency_ms_p95`、`participating_receivers_p95`。若该文档后续可定位，需要把其中更细的链路预算、包格式、星间通信约束同步到脚本报告字段。

## N607计划

| 项 | 值 |
|---|---|
| Conda环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| GPU选择 | 先preflight读取显存，选择低占用GPU；DARC-CI为CPU证据评估，GPU仅记录部署环境 |
| 远端脚本 | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_disagreement_confirm_ci_eval.py` |
| 远端输出 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_darc_ci_20260704/` |
| 远端日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_darc_ci_20260704/` |

## N607执行记录

| 项 | 结果 |
|---|---|
| SSH preflight | PASS；direct `N607`；project root可见；8张RTX3090均为10MiB/24576MiB |
| GPU选择 | GPU0；`CUDA_VISIBLE_DEVICES=0` |
| Conda环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 同步文件 | DARC脚本、DARC测试、本报告、base evidence、metadata |
| 远端单元验证 | `py_compile` PASS；无pytest内联测试PASS，输出`remote_darc_unit_tests=PASS` |
| 远端运行命令 | `CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_disagreement_confirm_ci_eval.py --base_evidence_csv runs/phase2_adv3b02_darc_ci_20260704/inputs/base_evidence.csv --metadata_json runs/phase2_adv3b02_darc_ci_20260704/inputs/base_metadata.json --output_dir runs/phase2_adv3b02_darc_ci_20260704/remote_all_profiles --policies opu_old_preserve,opu_old_guarded` |
| 远端输出 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_darc_ci_20260704/remote_all_profiles/` |
| 远端日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_darc_ci_20260704/darc_all_profiles.log` |
| 拉回artifact | `remote_artifacts/phase2_adv3b02_darc_ci_20260704/remote_all_profiles/` |
| SSH清理 | 每次SSH/SCP后检查本地`ssh.exe`和到`172.31.111.215:22`的ESTABLISHED连接；最终无残留 |

## N607远端结果

远端summary共41行，即表头+`base_paired`10行+`darc_ci`30行，覆盖2个policy、3个profile、`collab_count=1..5`。`candidate_count=0`。

| profile | policy | collab_count | old_acc | seen_new_acc | unknown_reject_rate | unknown_FAR | delta_old_acc | delta_unknown_reject_rate | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| darc_light | opu_old_preserve | 4 | 0.8021390374 | 0.7833333333 | 0.1833333333 | 0.7500000000 | 0.0000000000 | 0.0000000000 | diagnostic_only |
| darc_balanced | opu_old_preserve | 4 | 0.8021390374 | 0.7833333333 | 0.1833333333 | 0.7500000000 | 0.0000000000 | 0.0000000000 | diagnostic_only |
| darc_unknown_push | opu_old_preserve | 4 | 0.7967914439 | 0.7833333333 | 0.1833333333 | 0.7666666667 | -0.0053475936 | 0.0000000000 | diagnostic_only |
| darc_light | opu_old_guarded | 4 | 0.7754010695 | 0.7333333333 | 0.2500000000 | 0.5833333333 | 0.0053475936 | 0.0000000000 | diagnostic_only |

远端结论：DARC-CI没有形成有效未知拒识提升。接收机top-label分歧在当前证据上不足以区分unknown与星地信道下的旧类/seen-new困难样本。继续推进时应转向更直接的support-only距离/能量/EVT校准，而不是只依赖接收机分歧。

## 字段级可分性诊断

本地追加运行`phase2_evidence_field_separability_diag.py`，输入为同一base evidence。该诊断会使用query标签做oracle sweep，因此只能作为上界分析，不能作为可部署阈值选择。

| 约束 | 最佳字段 | old_acc | seen_new_acc | min_old | min_seen | unknown_reject_rate | unknown_FAR | known_coverage | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| FAR<=0.05 | `unknown_risk` | 0.1550000000 | 0.0750000000 | 0.0000000000 | 0.0000000000 | 0.9500000000 | 0.0500000000 | 0.1350000000 | 强拒识会严重误拒已知 |
| FAR<=0.10 | `unknown_risk` | 0.2383333333 | 0.1600000000 | 0.0000000000 | 0.1200000000 | 0.9100000000 | 0.0900000000 | 0.2237500000 | 不可用 |
| FAR<=0.20 | `score_risk` | 0.3633333333 | 0.4400000000 | 0.1100000000 | 0.3800000000 | 0.8000000000 | 0.2000000000 | 0.4512500000 | 不可用 |
| known floor>=0.80 | 无 | - | - | - | - | - | - | - | oracle也找不到可行门 |
| 目标99/97/99 | 无 | - | - | - | - | - | - | - | 当前evidence风险字段上界不足 |

诊断结论：现有`unknown_risk/score_risk/radius/mahalanobis/EVT/class_shell/class_negative`等字段在当前base evidence中不具备足够可分性。继续在这些字段上做阈值或接收机分歧融合无法达到目标；下一条路线必须改变证据本身，例如从feature/logit层重新构造support-only energy、quantized KNN distance、prototype shell和per-receiver calibration，而不是只重组已有risk字段。
