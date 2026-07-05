# phase2_adv3b02_frozen_manytx_unknown_diag_20260706

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | `phase2_adv3b02_frozen_manytx_unknown_diag_20260706` |
| 时间 | 2026-07-06 |
| 操作者 | Codex |
| 目标 | 基于`ADV3B02_CORE90_SOFT_E200`冻结特征，导出完整Stage2-C多接收机LEO特征包，并用`qknn8`评估协同推理数量`M=1..R_t`下的旧类、新类和未知类拒识表现。 |
| 性质 | `NON_DEPLOYMENT_DIAGNOSTIC`，除非后续`strict_event_key`与完整资源证据通过。 |

## 假设与对照

假设：现有ADV3B02冻结特征加`qknn8`协同推理能给出可解释的多接收机趋势；如果仍无法同时保旧类、新类和未知拒识，则该结果用于判断是否进入更底层的地面模型再训练或蒸馏路线。

对照：前一轮兼容旧Stage2-C特征包的诊断只覆盖`receiver_count=1`，且出现`unknown_reject_rate=1.0`但`old_acc=0.0`、`seen_new_acc=0.0`、`known_coverage=0.0`，只能作为负证据，不能证明当前目标。

## 协议配置

| 项 | 值 |
|---|---|
| 底座模型 | `ADV3B02_CORE90_SOFT_E200` |
| checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| 地面/source pkl | `/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl` |
| 新类/未知类pkl | `/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl` |
| `Y_old` | `14-10,14-7,20-15,20-19,6-15,8-20` |
| `R_s` | `1-1,1-19,14-7,18-2,19-2,2-1,2-19` |
| `R_t` | `20-1,3-19,7-14,7-7,8-8` |
| `Y_new` | `1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4` |
| `Y_unknown` | `10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20` |
| proxy unknown | `12-7,13-14,13-19,13-3,13-7,14-11,14-12,14-13` |
| proxy unknown receivers | `1-1,1-19,14-7,18-2,19-2,2-1`；`ManyTx.pkl`缺少`2-19`，因此不能把完整`R_s`直接传给`proxy_unknown_rxs`。 |
| LEO视图 | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| 星地信道实现 | `simplified_leo_residual` |
| K-shot | `8` |
| qkNN | `8` |
| query per class | `20` |
| 协同数量 | `collab_counts=all`，即`M=1..target_receiver_count` |
| 协同策略 | `collab_group_policy=available_up_to_k`，`partial_collab_min_receivers=1` |
| 事件对齐 | `receiver_domain_ranked` |
| 资源代理字段 | `max_event_bytes=1152`，`max_event_latency_ms=20` |

## 安全边界

| 检查项 | 约束 |
|---|---|
| 地面训练是否接触真实未知类 | 否。launcher只加载ADV3B02既有checkpoint，不做地面训练。 |
| `target_unknown`是否参与support | 否。 |
| `target_unknown`是否参与阈值拟合 | 否。冻结诊断器记录`uses_unknown_query_for_threshold=false`和`unknown_query_eval_only=true`。 |
| 是否可声明部署成功 | 否。本run为诊断证据。 |
| 是否满足资源约束说明文档 | 未验证。当前本地未找到`卫星协同射频指纹识别（RFFI）系统资源约束设计说明.md`，只报告代理字段。 |

## 本地变更

| 文件 | 用途 | SHA256 |
|---|---|---|
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh` | 新增ADV3B02冻结特征Stage2-C导出+qknn8协同诊断launcher。 | `16B769CA05B9A50F724A555CDC038B8FA9211CDCD3CE579EE0B31EA658F4AF6C` |
| `E:\type10-7\code\tests\test_phase2_adv3b02_frozen_manytx_unknown_diag_launcher.py` | 新增launcher dry-run协议断言。 | `7D39D3B09E8957261ABD8FB2622533C178D7C46CBDBD6119C36D4B870C581C89` |

本地`E:\type10-7`和`E:\type10-7\code`不是Git仓库；已创建快照：

`E:\type10-7\code\snapshots\phase2_adv3b02_frozen_manytx_unknown_diag_20260706`

Git镜像仓库为`E:\type10-7\github_publish\CVS-RFFI-repo`，当前存在非本轮未归属改动，后续只提交本轮新增文件。

## 本地验证

| 命令 | 结果 |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\tests\test_phase2_adv3b02_frozen_manytx_unknown_diag_launcher.py` | PASS |
| `conda run -n ssr-gpu python -m pytest -q code\tests\test_phase2_adv3b02_frozen_manytx_unknown_diag_launcher.py` | PASS，`1 passed`；仅`.pytest_cache`权限警告。 |
| `bash -n code/scripts/launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh` | PASS |
| `bash code/scripts/launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh --dry-run` | PASS，输出包含ADV3B02、Stage2-C、qknn8、`collab_counts=all`、LEO三场景、未知类eval-only、目标阈值字段和`proxy_unknown_rxs`修复值。 |

## 远端计划

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`

同步目标：

| 本地 | 远端 |
|---|---|
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh` |
| `E:\type10-7\code\tests\test_phase2_adv3b02_frozen_manytx_unknown_diag_launcher.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_adv3b02_frozen_manytx_unknown_diag_launcher.py` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_frozen_manytx_unknown_diag_20260706\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/report.md` |

远端启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash code/scripts/launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh
```

预期输出：

| 文件 | 说明 |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz` | 完整Stage2-C LEO多接收机冻结特征包。 |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5/frozen_manytx_diag.json` | 完整诊断JSON。 |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5/frozen_manytx_summary.csv` | `M=1..R_t`同row摘要表。 |
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5.out` | 导出和诊断日志。 |

## 成功判据与观察指标

目标阈值：`old_acc>=0.99`、`min_old_class_acc>=0.95`、`seen_new_acc>=0.97`、`min_seen_new_class_acc>=0.93`、`unknown_reject_rate>=0.99`且`unknown_FAR<=0.01`。

必须按`summary_rows`同一行判断，不得把不同`M`或不同候选的单项最大值拼成结论。若`goal_satisfied_counts`为空，则当前协同推理未达目标，需进入更底层路线，例如ADV3B02指导蒸馏、source-only虚拟未知边界、或旧类tail隔离再训练。

## 远端执行与验证

| 项 | 结果 |
|---|---|
| N607预检 | PASS。直连`N607`可用，项目根目录可见，GPU可见。 |
| GPU状态 | 启动前GPU0-3有训练进程且显存约2.4-2.8GB；GPU4-7显存约10MB。 |
| 标签覆盖 | PASS。`ManySig.pkl`对本轮`Y_old/R_s/R_t`无缺失；`ManyTx.pkl`对本轮`Y_new/Y_unknown/proxy_unknown/target R_t/proxy RXS`无缺失。 |
| 远端哈希 | launcher、测试、报告、`SYNC_MANIFEST.txt`与本地一致。 |
| 远端语法 | `bash -n`PASS。 |
| 远端dry-run | PASS，关键字段匹配。 |
| 远端focused测试 | `CVS-RFFI`环境缺少`pytest`；改用同一Python直接调用测试函数，`direct_launcher_test=PASS`。 |
| 启动命令 | `cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_phase2_adv3b02_frozen_manytx_unknown_diag_20260706.sh` |
| PID/GPU | PID`3096666`，GPU4。 |
| 日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5.out` |
| 结束状态 | 完成，日志出现`[ADV3B02-FROZEN-MANYTX-DIAG-DONE]`。 |

远端输出：

| 文件 | 大小 |
|---|---:|
| `features_stage2c_leo_multirx.npz` | 44572224 bytes |
| `frozen_manytx_diag.json` | 62408 bytes |
| `frozen_manytx_summary.csv` | 1061 bytes |

小型结果已拉回本地：

| 本地文件 | 说明 |
|---|---|
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_frozen_manytx_unknown_diag_20260706\artifacts\frozen_manytx_diag.json` | 完整诊断JSON。 |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_frozen_manytx_unknown_diag_20260706\artifacts\frozen_manytx_summary.csv` | `M=1..5`摘要CSV。 |

## 结果表

| M | old_acc | min_old_class_acc | seen_new_acc | min_seen_new_class_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | latency_ms_p95 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.096866 | 0.000000 | 0.000000 | 0.000000 | 0.974359 | 0.025641 | 0.051701 | 0.000000 | 2.103000 | FAIL |
| 2 | 0.136752 | 0.000000 | 0.000000 | 0.000000 | 0.958974 | 0.012821 | 0.066667 | 0.016889 | 2.103000 | FAIL |
| 3 | 0.133903 | 0.000000 | 0.000000 | 0.000000 | 0.969231 | 0.012821 | 0.065306 | 0.010667 | 2.103000 | FAIL |
| 4 | 0.133903 | 0.000000 | 0.000000 | 0.000000 | 0.969231 | 0.012821 | 0.065306 | 0.010667 | 2.103000 | FAIL |
| 5 | 0.133903 | 0.000000 | 0.000000 | 0.000000 | 0.969231 | 0.012821 | 0.065306 | 0.010667 | 2.103000 | FAIL |

`goal_satisfied_counts=[]`。没有任何协同数量同时满足旧类、新类和未知类目标。

协议安全字段：

| 字段 | 值 |
|---|---|
| `receiver_count` | `5` |
| `target_unknown_training_count` | `0` |
| `target_unknown_calibration_count` | `0` |
| `target_unknown_query_count` | `8000` |
| `uses_unknown_query_for_threshold` | `false` |
| `unknown_query_eval_only` | `true` |
| `target_channel_view` | `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| `role_counts` | `source=13440,target_old=9600,target_new=8000,target_unknown=8000,proxy_unknown=9350` |
| `evidence_row_count` | `2200` |

## 解释

本轮补齐了目标要求中此前缺失的当前ADV3B02、完整Stage2-C、LEO、多接收机、`M=1..5`证据链，但结果是负证据。协同数量从1增加到5后，未知拒识接近但仍未达到99%，旧类准确率仅约9.7%-13.7%，新类准确率为0。这说明当前冻结特征+qknn8+保守拒识门控主要把known样本拒绝或吸收到错误区域，不能解决目标问题。

当前结论：单纯决策层协同推理不足以达成目标。下一步应进入更底层路线，但仍不能接触真实未知类训练数据。优先方向是用ADV3B02作为教师，在source-only地面阶段蒸馏/再训练一个更适合LEO开集边界的学生模型，使用源域旧类、source-heldout/虚拟负样本、LEO强视图、一致性和core/tail/overflow隔离，使叠加星地信道后的未知类特征远离已知类自动接受区，同时维持旧类和新类support识别。

## 当前状态

本轮实验已完成，结果已拉回并写入报告。目标未完成，下一步需要设计并启动source-only蒸馏/再训练路线或更强的Stage2-C轻量adapter路线。
