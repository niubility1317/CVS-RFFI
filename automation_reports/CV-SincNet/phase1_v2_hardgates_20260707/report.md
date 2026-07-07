# phase1_v2_hardgates_20260707

## 基本信息

- 时间：2026-07-07
- 操作员：Codex
- 目标：把Phase1地面训练优化修改方案v2落到本地代码与启动脚本，形成endpoint硬边界、tail safety、OS有效预算、U_s三态利用、source episode density gate和可达性审计的闭环控制面。
- 协议边界：仍为Phase1 source-only地面训练；不使用目标接收机样本、不使用真实unknown训练或阈值校准、不声明unknown_FAR/FPR95/Stage2成功/Phase2成功/部署成功。open-set相关结论只作为Phase1几何风险和Phase3备选unknown拒识诊断证据。

## 假设与对照

- 对照：phase1_dgleo_directmetric16_20260706、phase1_dgleo_uopt24_20260707、phase1_dgleo_osfix16_20260707原launcher只传递direct metric和U_s损失参数，但缺少最终endpoint契约、tail后期扩张拦截、有效open-set梯度预算、U_s活跃度fail-closed、source episode密度证据和目标可达性审计。
- 假设：新增v2控制面后，实验即使闭集/星地指标提升，也不能在proxy_vaccept、p95/p99/tail_cvar、OS有效预算或U_s直接优化空转时被自动视为可推进候选。

## 本地版本状态

- `E:\type10-7`不是Git仓库：`git -C E:\type10-7 status -sb`返回`fatal: not a git repository`。
- `E:\type10-7\code`不是Git仓库：`git -C E:\type10-7\code status -sb`返回`fatal: not a git repository`。
- 初版v2预编辑快照：`E:\type10-7\code\snapshots\phase1_v2_hardgates_20260707_160634`。
- 本次P0补齐生产代码预编辑快照：`E:\type10-7\code\snapshots\phase1_v2_osfix_supplement_20260707_164438`。
- P0/P1闭环补丁生产代码预编辑快照：`E:\type10-7\code\snapshots\phase1_v2_p0p1_closure_20260707_170911`。

## 改动文件

| 文件 | 改动目的 |
|---|---|
| `E:\type10-7\code\cvsrffi\phase1_v2_control.py` | 新增v2控制模块：endpoint契约、tail safety状态机、OS有效预算、U_s三态、可达性审计。 |
| `E:\type10-7\code\cvsrffi\losses.py` | source episode loss输出receiver/local component、core/tail/outside ready和density gate活跃证据。 |
| `E:\type10-7\code\SSDG\train_ssdg.py` | 接入v2参数、启动日志、训练期guard、best checkpoint拦截和final prototype export fail-closed。 |
| `E:\type10-7\code\scripts\launch_phase1_dgleo_directmetric16_20260706.sh` | directmetric16接入`PHASE1_V2_FLAGS`，dry-run和候选声明输出v2证据。 |
| `E:\type10-7\code\scripts\launch_phase1_dgleo_uopt24_20260707.sh` | uopt24接入`PHASE1_V2_FLAGS`，强制U_s三态门控。 |
| `E:\type10-7\code\scripts\launch_phase1_dgleo_osfix16_20260707.sh` | osfix16接入`PHASE1_V2_FLAGS`，强制U_s三态门控。 |
| `E:\type10-7\code\tests\test_phase1_v2_control.py` | 新增v2控制模块单元测试和parser暴露测试。 |
| `E:\type10-7\code\tests\test_open_world_feature_space_loss.py` | 增加source episode真实local component、core/tail/outside和分位数证据测试。 |
| `E:\type10-7\code\tests\test_phase1_dgleo_directmetric16_launcher.py` | launcher dry-run新增v2参数断言。 |
| `E:\type10-7\code\tests\test_phase1_unlabeled_direct_training.py` | uopt24 dry-run新增v2和U_s三态断言。 |
| `E:\type10-7\code\tests\test_phase1_dgleo_osfix16_launcher.py` | osfix16 dry-run新增v2和U_s三态断言。 |

## v2闭环内容

| 问题 | 落地措施 | 失败时动作 |
|---|---|---|
| 动态软门控不等于最终拒识边界 | `assess_endpoint_contract`要求保留`endpoint_accept_v1`，并硬校验`endpoint_threshold_source=source_val_only`和`endpoint_calibration_split=source_val`；禁止把dynamic dm/loss gate导出为最终边界，禁止Phase1声明真实unknown/Stage2成功。 | `phase1_v2_guard_fired=1`，禁止best更新；关键P0原因同时触发final export fail-closed。 |
| known域太散/source-only几何矛盾 | tail safety状态机直接监控p95、p99、tail_cvar、proxy_vaccept；新增best p99和tail_cvar到当前/final扩张量。`p99_delta>2.0`或`tail_cvar_delta>4.0`阻断final export，`p99_delta>3.5`或`tail_cvar_delta>6.0`阻断promotion/best。 | 标记`tail_expansion_blocks_*`或`tail_cvar_expansion_blocks_*`；final export跳过并声明`NON_PROMOTABLE_DIAGNOSTIC`。 |
| 闭集/KD/sat压过open-set几何 | `assess_open_set_effective_budget`计算OS加权loss相对CE/KD/sat/domain的有效预算，launcher默认`--os_eff_min_budget 0.15`，并启用`--phase1_v2_os_eff_all_phases true`。 | 任一阶段预算不足则禁止best更新并阻断final export。 |
| 无标签分支空转 | `assess_unlabeled_tri_state`检查U_s direct loss、active、selected和三态计数；三组launcher统一`--u_tri_state_required true`。 | U_s direct idle、缺少selected或缺少三态证据时禁止best更新并阻断final export。 |
| 训练后期扩大tail | tail safety状态机跨epoch累计unsafe计数，WARNING/ROLLBACK/STOP逐级fail-closed。 | 后期tail恶化不再允许事后挑best或继续导出final。 |
| source_episode_overflow长期约0.97 | `source_episode_three_sigma_loss`改为输出TX×held-out-domain local component数、core/tail/outside计数/比例、p50/p95/p99/tail_CVaR；`assess_source_episode_density_gate`要求overflow、local component、三态ready、density active和分位数证据同时存在。 | `SOURCE_EPISODE_OVERFLOW_HIGH`、`RECEIVER_AWARE_LOCAL_COMPONENT_MISSING`、`CORE_TAIL_OUTSIDE_NOT_READY`、`SOURCE_EPISODE_DENSITY_GATE_INACTIVE`或`SOURCE_EPISODE_QUANTILES_MISSING`禁止best更新并阻断final export。 |
| 目标阈值差距过大导致罚很多但推不动 | `assess_feasibility_gate`接入P0 fail-closed路径；三组launcher启用`--feasibility_gate true --feasibility_stage full --feasibility_relaxed_pass false --feasibility_local_pass false`，默认把不可达全目标标为诊断负例。 | 不可达时禁止best更新并阻断final export。 |

## 启动脚本关键配置

三组launcher统一注入：

```bash
--phase1_v2_hard_gates true
--endpoint_accept_policy_id endpoint_accept_v1
--endpoint_threshold_source source_val_only
--endpoint_calibration_split source_val
--loss_gate_exported false
--tail_safety_state_machine true
--tail_stop_blocks_final true
--tail_safety_p95_target_deg 54
--tail_safety_p99_target_deg 70
--tail_safety_cvar_target_deg 56
--tail_safety_proxy_vaccept_target 0.35
--tail_safety_p99_expansion_block_final_delta 2.0
--tail_safety_p99_expansion_block_best_delta 3.5
--tail_safety_cvar_expansion_block_final_delta 4.0
--tail_safety_cvar_expansion_block_best_delta 6.0
--os_eff_min_budget 0.15
--phase1_v2_os_eff_all_phases true
--phase1_v2_guard_blocks_final true
--source_episode_density_gate true
--source_episode_overflow_warn 0.90
--source_episode_min_local_components 4
--u_direct_idle_blocks_promotion true
--u_tri_state_required true
--feasibility_gate true
--feasibility_stage full
--feasibility_relaxed_pass false
--feasibility_local_pass false
```

## 验证记录

| 命令 | 结果 |
|---|---|
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code\cvsrffi\phase1_v2_control.py code\cvsrffi\losses.py code\SSDG\train_ssdg.py` | 通过 |
| `bash -n code/scripts/launch_phase1_dgleo_directmetric16_20260706.sh; bash -n code/scripts/launch_phase1_dgleo_uopt24_20260707.sh; bash -n code/scripts/launch_phase1_dgleo_osfix16_20260707.sh` | 通过 |
| `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_dgleo_directmetric16_launcher.py code\tests\test_phase1_unlabeled_direct_training.py code\tests\test_phase1_dgleo_osfix16_launcher.py -q` | 10 passed |
| `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_v2_control.py -q` | 10 passed |
| `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_v2_control.py code\tests\test_ssdg_guard.py code\tests\test_direct_metric_acceptance_loss.py code\tests\test_unlabeled_quarantine_acceptance_loss.py code\tests\test_phase1_unlabeled_direct_training.py code\tests\test_phase1_dgleo_directmetric16_launcher.py code\tests\test_phase1_dgleo_osfix16_launcher.py -q` | 30 passed |
| `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_v2_control.py code\tests\test_open_world_feature_space_loss.py code\tests\test_phase1_dgleo_directmetric16_launcher.py code\tests\test_phase1_unlabeled_direct_training.py code\tests\test_phase1_dgleo_osfix16_launcher.py -q` | 34 passed |
| `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_v2_control.py code\tests\test_open_world_feature_space_loss.py code\tests\test_ssdg_guard.py code\tests\test_direct_metric_acceptance_loss.py code\tests\test_unlabeled_quarantine_acceptance_loss.py code\tests\test_phase1_unlabeled_direct_training.py code\tests\test_phase1_dgleo_directmetric16_launcher.py code\tests\test_phase1_dgleo_osfix16_launcher.py -q` | 44 passed |

备注：pytest有本地`.pytest_cache`写权限告警，不影响测试断言。

## N607状态

- 本次只完成本地代码、测试、launcher dry-run闭环。
- 未执行N607 preflight。
- 未同步文件到N607。
- 未启动新实验。

## 后续建议

1. 若要跑v2验证，先执行N607 direct preflight，再同步上述改动文件。
2. 第一轮只建议每GPU一到两个短候选dry-run/正式run，观察`[CONFIG-PHASE1-V2]`、`phase1_v2_guard_fired`、`tail_safety_state`、`B_os_eff`和`US_DIRECT_LOSS_IDLE`是否按预期写入metrics。
3. 若大量候选被OS预算或U_s三态门控阻断，应先调loss权重和U_s采样/selected策略，再扩大全矩阵。
