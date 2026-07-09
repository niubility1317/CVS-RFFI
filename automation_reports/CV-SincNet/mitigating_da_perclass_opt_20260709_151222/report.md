# Mitigating Receiver Impact DA per-class gap optimization

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | `mitigating_da_perclass_opt_20260709_151222` |
| 时间 | 2026-07-09 15:12 Asia/Hong_Kong |
| 操作者 | Codex |
| 目标 | 针对与论文Proposed差距大的个别类别/任务，定位per-class塌缩原因，做最小优化修改，并在N607复现实验验证 |
| 前序主结果 | `14-7->3-19` selected66.59%、curve max76.58%，论文92.42%；`1-1->1-19` selected47.90%、curve max61.43%，论文95.44%；`7-7->8-8` selected61.47%，论文99.74% |
| 本轮边界 | 仍为论文WiSig ManySig DA复现，不作为CVS Stage2/LEO部署证据 |

## 已读规则与版本状态

| 项目 | 结果 |
|---|---|
| `AGENTS.md` | 已读 |
| `项目.md` | 已读 |
| 根目录Git | `E:\type10-7`不是Git仓库，报告需镜像到Git承载面 |
| 代码Git承载面 | `E:\type10-7\github_publish\CVS-RFFI-repo`，分支`codex/cvs-rffi-release-20260626` |
| 初始Git状态 | 存在与本任务无关的`code/scripts/phase2_qknn_active_support_select.py`、`code/tests/test_phase2_active_support_selection_policy.py`改动及`local_artifacts/...`未跟踪目录；本轮不触碰 |

## 调试假设

本轮不先猜测单个超参，而是先补齐per-class证据。前序artifact只包含整体`target_pred_acc`、`target_pseudo_selected_acc`、`class_weight_max`和target loss/accuracy曲线，无法直接判断哪些TX类别被错分、漏选或过度伪标。因此第一步先用已有checkpoint和target_eval数据做只读per-class评估，再决定最小修改。

## N607只读预检

15:12 CST执行`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`，直连`N607`通过；远端项目根目录`/home/szu2070436088/2510044040/CV-SincNet`可见，8张RTX3090可见且空闲。该步骤不改变远端状态。

## 已有checkpoint per-class诊断

对前序主复现中差距最大的3个cross-receiver任务，仅加载已有E/C checkpoint并在`target_eval`上评估。checkpoint缺少`estimate_network.*`键是预期行为，因为保存的是推理用E/C权重。

| 任务 | overall | 主要塌缩类别 | 高置信错分方向 | 解释 |
|---|---:|---|---|---|
| `14-7->3-19` | 66.59% | `14-10`34.43%、`14-7`5.33% | `14-10->20-19`2441/4000；`14-7->20-19`3604/4000 | 预测分布严重偏向`20-19`，该类总预测9633/24000；不是均匀低分 |
| `1-1->1-19` | 47.90% | `20-15`0.00%、`20-19`0.83%、`8-20`0.20% | `20-15->8-20`3997/4000；`20-19->14-7`3959/4000；`8-20->20-15`3992/4000 | 形成近乎置换式高置信错分，多个类别互相映射错误 |
| `7-7->8-8` | 61.47% | `20-19`0.48%、`8-20`0.00%、`20-15`69.20% | `20-19->14-7`3979/4000；`8-20->20-19`3995/4000；`20-15->20-19`988/4000 | 类0/1/4接近满分，但类3/5完全塌缩，overall被少数类拉低 |

结论：差距大的“个别类别”不是评估统计噪声，而是目标域伪标签自训练在部分TX上形成高置信错误吸引子。仅看overall accuracy和`class_weight_max`无法定位；必须在训练history和最终row中写入per-class伪标签、预测直方图、混淆矩阵和类别权重向量。

## 本地代码修改

| 文件 | 修改 | 目的 |
|---|---|---|
| `paper_reproduction/mitigating_receiver_impact_da/algorithm.py` | `_select_pseudo_labels`新增默认关闭的`pseudo_threshold_floor`和`pseudo_quota_mode=balanced_topk`/`pseudo_quota_per_class`；`gada_batch_step`输出`class_weight_vector`、`pseudo_threshold_vector`、`target_pred_hist`、`target_pseudo_selected_hist`及按真类/预测类的伪标签正确计数 | 用最小开关验证“低阈值/多数类伪标签挤占”是否造成类别塌缩，并保留默认paper路径 |
| `paper_reproduction/mitigating_receiver_impact_da/train.py` | 新增`_evaluate_target_metrics`，输出per-class accuracy、prediction histogram、confusion matrix；训练history聚合per-class pseudo/pred/weight/threshold字段；CLI新增`--pseudo-threshold-floor`、`--pseudo-quota-mode`、`--pseudo-quota-per-class` | 让复现实验能直接定位具体TX类别差距，并能跑可控诊断 |
| `tests/test_mitigating_receiver_impact_da.py` | 增加per-class评估、伪标签floor/quota、Table II JSON透传和history字段测试 | 防止后续复现实验再次只有overall指标 |

本地验证：`conda run -n ssr-gpu python -m pytest tests/test_mitigating_receiver_impact_da.py -q`通过，33项。

## 同步记录

本地提交：`cdd2b79 Add per-class diagnostics for receiver impact DA`。同步到N607：

| 本地文件 | 远端路径 | SHA256 |
|---|---|---|
| `paper_reproduction/mitigating_receiver_impact_da/algorithm.py` | `/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/mitigating_receiver_impact_da/algorithm.py` | `6e0eb3f39b808cf0a785f6dad9cbec9c3103fbafa8a67292740b61822465ab76` |
| `paper_reproduction/mitigating_receiver_impact_da/train.py` | `/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/mitigating_receiver_impact_da/train.py` | `ce33396e2c4909e30607dfdacb0d774323b4f3b051ff53b17b84a9ca180b071d` |

远端CLI帮助已确认新增`--pseudo-threshold-floor`、`--pseudo-quota-mode`、`--pseudo-quota-per-class`。同步后本地检查：未发现残留`ssh.exe`或到N607的`TCP:22`连接。

## 优化验证矩阵

共同设置：只跑`Proposed`；`epochs=10`、`batch=128`、`seed=20260710`、`source_pretrain_epochs=0`、`adapt_start_epoch=0`、`base_tau=0.7`、`estimate_steps=7`、`class_prior_mode=source`、`kl_estimator_mode=mine_ma`、`mine_update_scale=0.5`、`pseudo_threshold_mode=paper`、`pseudo_score_mode=probability`、`class_weight_timing=current`、`target_model_selection=target_loss_best`。

| 候选 | 额外参数 | 假设 |
|---|---|---|
| `floor07` | `--pseudo-threshold-floor 0.7` | 防止CPL少数类阈值降到过低，减少低质伪标签进入target CE |
| `quota16_floor05` | `--pseudo-threshold-floor 0.5 --pseudo-quota-mode balanced_topk --pseudo-quota-per-class 16` | 每个batch每个预测类最多16个target伪标签，抑制多数预测类挤占 |
| `quota32_floor05` | `--pseudo-threshold-floor 0.5 --pseudo-quota-mode balanced_topk --pseudo-quota-per-class 32` | 保留更多target信号，检查quota16是否过强 |

任务：`14-7->3-19`、`1-1->1-19`、`7-7->8-8`。成功判据：任一候选在对应任务selected或curve max超过前序主结果，并且per-class最差类别准确率提升，不以单一overall最高掩盖类别塌缩。

## 启动记录

15:18 CST启动9个实验。远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`；Python环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

| run id | 任务 | 候选 | GPU | PID | 结果 |
|---|---|---|---:|---:|---|
| `mitigating_da_perclass_floor07_14-7_to_3-19_b128_s20260710_20260709_1518` | `14-7->3-19` | `floor07` | 0 | `1340942` | `paper_reproduction/runs/mitigating_da_perclass_floor07_14-7_to_3-19_b128_s20260710_20260709_1518/results.json` |
| `mitigating_da_perclass_quota16_floor05_14-7_to_3-19_b128_s20260710_20260709_1518` | `14-7->3-19` | `quota16_floor05` | 1 | `1340947` | `paper_reproduction/runs/mitigating_da_perclass_quota16_floor05_14-7_to_3-19_b128_s20260710_20260709_1518/results.json` |
| `mitigating_da_perclass_quota32_floor05_14-7_to_3-19_b128_s20260710_20260709_1518` | `14-7->3-19` | `quota32_floor05` | 2 | `1340952` | `paper_reproduction/runs/mitigating_da_perclass_quota32_floor05_14-7_to_3-19_b128_s20260710_20260709_1518/results.json` |
| `mitigating_da_perclass_floor07_1-1_to_1-19_b128_s20260710_20260709_1518` | `1-1->1-19` | `floor07` | 3 | `1340957` | `paper_reproduction/runs/mitigating_da_perclass_floor07_1-1_to_1-19_b128_s20260710_20260709_1518/results.json` |
| `mitigating_da_perclass_quota16_floor05_1-1_to_1-19_b128_s20260710_20260709_1518` | `1-1->1-19` | `quota16_floor05` | 4 | `1340962` | `paper_reproduction/runs/mitigating_da_perclass_quota16_floor05_1-1_to_1-19_b128_s20260710_20260709_1518/results.json` |
| `mitigating_da_perclass_quota32_floor05_1-1_to_1-19_b128_s20260710_20260709_1518` | `1-1->1-19` | `quota32_floor05` | 5 | `1340967` | `paper_reproduction/runs/mitigating_da_perclass_quota32_floor05_1-1_to_1-19_b128_s20260710_20260709_1518/results.json` |
| `mitigating_da_perclass_floor07_7-7_to_8-8_b128_s20260710_20260709_1518` | `7-7->8-8` | `floor07` | 6 | `1340972` | `paper_reproduction/runs/mitigating_da_perclass_floor07_7-7_to_8-8_b128_s20260710_20260709_1518/results.json` |
| `mitigating_da_perclass_quota16_floor05_7-7_to_8-8_b128_s20260710_20260709_1518` | `7-7->8-8` | `quota16_floor05` | 7 | `1340977` | `paper_reproduction/runs/mitigating_da_perclass_quota16_floor05_7-7_to_8-8_b128_s20260710_20260709_1518/results.json` |
| `mitigating_da_perclass_quota32_floor05_7-7_to_8-8_b128_s20260710_20260709_1518` | `7-7->8-8` | `quota32_floor05` | 0 | `1340982` | `paper_reproduction/runs/mitigating_da_perclass_quota32_floor05_7-7_to_8-8_b128_s20260710_20260709_1518/results.json` |

启动健康检查：9个PID均在运行，GPU0承载2个本轮任务，其余GPU各1个；未超过每GPU2个训练实验约束。启动后本地检查：未发现残留`ssh.exe`或到N607的`TCP:22`连接。

## 第一轮优化结果

结果文件已拉取到`E:\type10-7\automation_reports\CV-SincNet\mitigating_da_perclass_opt_20260709_151222\remote_artifacts\`。期间发现一个本地旧`ssh.exe`连接仍挂在`logs/dadda_targeted_gapfix_20260709_1526/*.out`的tail命令上，已识别并关闭；该连接不是本轮训练进程。

| 任务 | 候选 | selected | curve max | 最后epoch | 最差类acc | 相对前序selected | 相对论文 | per-class target acc |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `1-1->1-19` | `floor07` | 42.50% | 48.67%@7 | 46.02% | 0.03% | -5.40pp | -52.94pp | `[93.05,4.10,0.43,57.38,100.00,0.03]` |
| `1-1->1-19` | `quota16_floor05` | 83.47% | 83.47%@4 | 76.22% | 30.93% | +35.57pp | -11.97pp | `[99.58,70.70,99.92,30.93,99.98,99.75]` |
| `1-1->1-19` | `quota32_floor05` | 66.16% | 66.16%@1 | 38.27% | 0.05% | +18.26pp | -29.28pp | `[99.45,94.35,99.92,3.23,99.95,0.05]` |
| `14-7->3-19` | `floor07` | 40.64% | 47.92%@7 | 40.69% | 0.00% | -25.95pp | -51.78pp | `[68.85,27.98,0.83,47.70,98.47,0.00]` |
| `14-7->3-19` | `quota16_floor05` | 63.76% | 67.10%@5 | 61.22% | 26.77% | -2.83pp | -28.66pp | `[52.92,60.12,26.77,54.47,89.90,98.35]` |
| `14-7->3-19` | `quota32_floor05` | 30.43% | 44.73%@5 | 12.47% | 0.80% | -36.16pp | -61.99pp | `[36.00,25.90,0.80,35.45,14.50,69.92]` |
| `7-7->8-8` | `floor07` | 86.61% | 86.61%@6 | 78.89% | 25.47% | +25.14pp | -13.13pp | `[95.05,25.47,99.85,99.40,99.90,100.00]` |
| `7-7->8-8` | `quota16_floor05` | 78.34% | 81.55%@2 | 80.05% | 49.43% | +16.87pp | -21.40pp | `[77.10,68.90,74.70,49.43,99.90,100.00]` |
| `7-7->8-8` | `quota32_floor05` | 85.27% | 88.59%@5 | 76.06% | 62.48% | +23.80pp | -14.47pp | `[86.17,62.48,99.45,63.75,99.92,99.85]` |

结论：`quota16_floor05`显著修复`1-1->1-19`的类别置换式塌缩，selected从47.90%升至83.47%；`floor07`和`quota32_floor05`显著修复`7-7->8-8`，selected最高86.61%、曲线最高88.59%。`14-7->3-19`仍未提升overall，但`quota16_floor05`把最差类从约5%提升到26.77%，说明按类配额改善了类别覆盖，却牺牲了原本高分类，仍需与类别权重稳定化或pretrain组合。

## 第二轮组合验证计划

基于第一轮结果，第二轮只跑有证据支持的组合：

| 任务 | 候选 | 参数 | 目的 |
|---|---|---|---|
| `1-1->1-19` | `pre10_quota16_floor05` | `--source-pretrain-epochs 10 --pseudo-threshold-floor 0.5 --pseudo-quota-mode balanced_topk --pseudo-quota-per-class 16` | 结合历史有效`source_pretrain=10`和本轮有效quota，尝试接近论文95.44% |
| `1-1->1-19` | `pre10_noquota` | `--source-pretrain-epochs 10` | 用新per-class日志复核历史92.35%路径 |
| `7-7->8-8` | `pre10_floor07` | `--source-pretrain-epochs 10 --pseudo-threshold-floor 0.7` | 检查pretrain是否把86.61%继续推高 |
| `7-7->8-8` | `pre10_quota32_floor05` | `--source-pretrain-epochs 10 --pseudo-threshold-floor 0.5 --pseudo-quota-mode balanced_topk --pseudo-quota-per-class 32` | 结合曲线最高88.59%的quota32路径 |
| `14-7->3-19` | `cw05_2_quota16_floor05` | `--class-weight-smoothing 1.0 --class-weight-clip-min 0.5 --class-weight-clip-max 2.0 --class-weight-mean-normalize --pseudo-threshold-floor 0.5 --pseudo-quota-mode balanced_topk --pseudo-quota-per-class 16` | 结合此前70.54%的权重稳定化与本轮类别覆盖改善 |
| `14-7->3-19` | `cw05_2_noquota` | `--class-weight-smoothing 1.0 --class-weight-clip-min 0.5 --class-weight-clip-max 2.0 --class-weight-mean-normalize` | 用新per-class日志复核当前最佳权重稳定化候选 |

## 第二轮启动记录

15:30 CST启动。`cw05_2_noquota`初次PID`1351093`因参数解析错误退出，日志报`unrecognized arguments: --class-weight-mean-normalize`；随后远端`--help`确认该参数存在，并用简化单条命令重启为`mitigating_da_perclass2_cw05_2_noquota_rerun_14-7_to_3-19_b128_s20260710_20260709_1530`，PID`1352927`。其它5个PID正常运行。

| run id | 任务 | 候选 | GPU | PID | 结果 |
|---|---|---|---:|---:|---|
| `mitigating_da_perclass2_pre10_quota16_floor05_1-1_to_1-19_b128_s20260710_20260709_1530` | `1-1->1-19` | `pre10_quota16_floor05` | 0 | `1351068` | `paper_reproduction/runs/mitigating_da_perclass2_pre10_quota16_floor05_1-1_to_1-19_b128_s20260710_20260709_1530/results.json` |
| `mitigating_da_perclass2_pre10_noquota_1-1_to_1-19_b128_s20260710_20260709_1530` | `1-1->1-19` | `pre10_noquota` | 1 | `1351073` | `paper_reproduction/runs/mitigating_da_perclass2_pre10_noquota_1-1_to_1-19_b128_s20260710_20260709_1530/results.json` |
| `mitigating_da_perclass2_pre10_floor07_7-7_to_8-8_b128_s20260710_20260709_1530` | `7-7->8-8` | `pre10_floor07` | 2 | `1351078` | `paper_reproduction/runs/mitigating_da_perclass2_pre10_floor07_7-7_to_8-8_b128_s20260710_20260709_1530/results.json` |
| `mitigating_da_perclass2_pre10_quota32_floor05_7-7_to_8-8_b128_s20260710_20260709_1530` | `7-7->8-8` | `pre10_quota32_floor05` | 3 | `1351083` | `paper_reproduction/runs/mitigating_da_perclass2_pre10_quota32_floor05_7-7_to_8-8_b128_s20260710_20260709_1530/results.json` |
| `mitigating_da_perclass2_cw05_2_quota16_floor05_14-7_to_3-19_b128_s20260710_20260709_1530` | `14-7->3-19` | `cw05_2_quota16_floor05` | 4 | `1351088` | `paper_reproduction/runs/mitigating_da_perclass2_cw05_2_quota16_floor05_14-7_to_3-19_b128_s20260710_20260709_1530/results.json` |
| `mitigating_da_perclass2_cw05_2_noquota_rerun_14-7_to_3-19_b128_s20260710_20260709_1530` | `14-7->3-19` | `cw05_2_noquota_rerun` | 5 | `1352927` | `paper_reproduction/runs/mitigating_da_perclass2_cw05_2_noquota_rerun_14-7_to_3-19_b128_s20260710_20260709_1530/results.json` |

启动后本地检查：未发现残留`ssh.exe`或到N607的`TCP:22`连接。

## 第二轮结果

有效6个结果均已完成并拉取到本地artifact目录。`cw05_2_noquota`初次失败run仅保留日志，不纳入结果表；有效结果使用`cw05_2_noquota_rerun`。

| 任务 | 候选 | selected | curve max | 最后epoch | 最差类acc | 相对前序selected | 相对论文 | per-class target acc |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `1-1->1-19` | `pre10_noquota` | 68.87% | 68.87%@10 | 68.87% | 3.43% | +20.97pp | -26.57pp | `[99.70,3.43,75.00,35.23,99.98,99.90]` |
| `1-1->1-19` | `pre10_quota16_floor05` | 67.75% | 73.78%@7 | 66.20% | 9.03% | +19.85pp | -27.69pp | `[99.55,9.03,99.08,36.95,99.98,61.93]` |
| `7-7->8-8` | `pre10_floor07` | 60.61% | 66.68%@4 | 50.62% | 0.18% | -0.86pp | -39.13pp | `[63.80,95.40,4.65,0.18,99.93,99.73]` |
| `7-7->8-8` | `pre10_quota32_floor05` | 56.46% | 63.22%@7 | 45.49% | 0.08% | -5.01pp | -43.28pp | `[92.53,60.95,0.08,85.15,99.95,0.13]` |
| `14-7->3-19` | `cw05_2_noquota_rerun` | 56.57% | 56.57%@4 | 43.57% | 22.38% | -10.02pp | -35.85pp | `[22.70,22.38,80.28,22.45,93.95,97.68]` |
| `14-7->3-19` | `cw05_2_quota16_floor05` | 63.98% | 66.11%@9 | 59.60% | 8.88% | -2.62pp | -28.45pp | `[91.53,62.60,8.88,40.23,85.75,94.88]` |

第二轮结论：source pretrain没有复现此前摘要中的`1-1->1-19=92.35%`，反而低于第一轮`quota16_floor05=83.47%`；对`7-7->8-8`也明显有害。`14-7->3-19`的类别权重稳定化与quota组合没有超过前序主结果，说明该pair不是单纯由“低阈值/多数类伪标签挤占/权重爆炸”三者之一独立造成。

## 当前最佳可复现结果

| 任务 | 论文Proposed | 本轮/前序最佳selected | 差值 | 最佳curve max | curve差值 | 最佳候选 |
|---|---:|---:|---:|---:|---:|---|
| `14-7->3-19` | 92.42% | 66.59% | -25.83pp | 76.58% | -15.84pp | 前序独立run，`no quota/no floor`；本轮未超过 |
| `1-1->1-19` | 95.44% | 83.47% | -11.97pp | 83.47% | -11.97pp | `quota16_floor05` |
| `7-7->8-8` | 99.74% | 86.61% | -13.13pp | 88.59% | -11.15pp | selected为`floor07`，curve max为`quota32_floor05` |

## 定位结论

1. 大差距任务不是所有类别均匀变差，而是少数TX类高置信塌缩。例如`1-1->1-19`原始run中`20-15->8-20`为3997/4000、`20-19->14-7`为3959/4000、`8-20->20-15`为3992/4000；`7-7->8-8`中`8-20->20-19`为3995/4000。
2. `quota16_floor05`能显著修复`1-1->1-19`类别塌缩，说明伪标签多数类挤占是该任务主因之一；但仍差论文11.97pp。
3. `floor07`能显著修复`7-7->8-8`，说明CPL低阈值引入低质伪标签是该任务主因之一；但仍差论文13.13pp。
4. `14-7->3-19`的难点更深：提高阈值、按类quota、类别权重裁剪/平滑及其组合都没有超过前序66.59% selected/76.58% curve max。该pair更可能依赖论文未公开的初始化、batch顺序、训练轮数/选择准则或数据处理细节。
5. `target_model_selection=target_loss_best`仍是诊断/Oracle边界：它使用目标标签计算loss，不应当作为纯paper-faithful UDA正式声明。它帮助证明模型选择敏感，但不能替代论文未公开的验证策略。

## 第三轮`14-7->3-19`针对性计划

新增本地分析脚本`analyze_14_7_to_3_19.py`解析本轮artifact。关键证据：`quota16_floor05`的训练history在epoch6达到70.96%，但`target_loss_best`选择epoch4并得到63.76%；全量`target_eval_history`最高为67.10%@epoch5。这说明`14-7->3-19`同时存在类别塌缩和模型选择口径问题。`quota16_floor05`在峰值epoch的伪标签预测直方图接近平衡，但`20-15`和`20-19`伪标签精度仍低，仅约41.65%和28.59%，因此下一轮只验证最小因素：延长训练、降低quota提高伪标签精度、使用paper/official兼容的`zip_min+epoch`状态、以及seed敏感性。

15:50 CST重新执行`tools\n607_ssh_preflight.ps1`，直连`N607`通过；远端项目根目录可见，8张RTX3090空闲。第三轮仍只跑论文`Proposed`方法，不作为CVS Stage2/LEO部署证据。

| 候选 | 参数 | 目的 |
|---|---|---|
| `q16_final_e20` | `epochs=20 --target-model-selection final --pseudo-threshold-floor 0.5 --pseudo-quota-mode balanced_topk --pseudo-quota-per-class 16` | 检查10epoch是否截断了`quota16`上升路径，并避免`target_loss_best`选错epoch |
| `q8_final_e20` | `epochs=20 --target-model-selection final --pseudo-threshold-floor 0.5 --pseudo-quota-mode balanced_topk --pseudo-quota-per-class 8` | 降低每类伪标签数量，检验少数类低精度伪标签是否是后续塌缩主因 |
| `q16_zip_epoch_e20` | `epochs=20 --target-model-selection final --batch-pairing zip_min --pseudo-state-scope epoch --pseudo-threshold-floor 0.5 --pseudo-quota-mode balanced_topk --pseudo-quota-per-class 16` | 对齐`official_compat`中的batch配对和按epoch伪标签状态，但保留paper概率伪标签 |
| `q16_final_e20_seed42` | 同`q16_final_e20`但`seed=42` | 检查`14-7->3-19`是否对初始化和shuffle高度敏感 |

## 第三轮启动记录

15:55 CST第一次启动`mitigating_da_perclass3_*_20260709_1551`四个任务，均快速退出；启动健康检查显示`train.py: error: the following arguments are required: --config`。根因是新写launcher误用了旧的直接参数入口。该失败属于launcher参数错误，不纳入结果比较；失败日志保留在远端`paper_reproduction/logs/mitigating_da_perclass3_*_20260709_1551.out`。

已修复本地launcher`paper_reproduction/mitigating_receiver_impact_da/launch_perclass3_20260709_1551.sh`：补齐`--config paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json`、`--run-table2`、`--manysig-pkl`、`--checkpoint-dir`和`--output`。本地验证：

| 验证 | 结果 |
|---|---|
| `bash -n ./paper_reproduction/mitigating_receiver_impact_da/launch_perclass3_20260709_1551.sh` | 通过 |
| `conda run -n ssr-gpu python -m paper_reproduction.mitigating_receiver_impact_da.train --config paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json --dry-run --output ...\dryrun_perclass3_config.json` | 通过 |
| 修复版launcher SHA256 | `22f15806dfbcf1dbd77022e7b133f4983f18c27cb7a77734ef2b38d0530f61c0` |

16:00 CST同步修复版到N607并远端验证SHA/`bash -n`通过后，重启四个`perclass3b`任务：

| run id | 任务 | 候选 | GPU | PID | 远端结果路径 |
|---|---|---|---:|---:|---|
| `mitigating_da_perclass3b_q16_final_e20_14-7_to_3-19_b128_s20260710_20260709_1600` | `14-7->3-19` | `q16_final_e20` | 0 | `1365894` | `paper_reproduction/runs/mitigating_da_perclass3b_q16_final_e20_14-7_to_3-19_b128_s20260710_20260709_1600/results.json` |
| `mitigating_da_perclass3b_q8_final_e20_14-7_to_3-19_b128_s20260710_20260709_1600` | `14-7->3-19` | `q8_final_e20` | 1 | `1365896` | `paper_reproduction/runs/mitigating_da_perclass3b_q8_final_e20_14-7_to_3-19_b128_s20260710_20260709_1600/results.json` |
| `mitigating_da_perclass3b_q16_zip_epoch_e20_14-7_to_3-19_b128_s20260710_20260709_1600` | `14-7->3-19` | `q16_zip_epoch_e20` | 2 | `1365898` | `paper_reproduction/runs/mitigating_da_perclass3b_q16_zip_epoch_e20_14-7_to_3-19_b128_s20260710_20260709_1600/results.json` |
| `mitigating_da_perclass3b_q16_final_e20_seed42_14-7_to_3-19_b128_s42_20260709_1600` | `14-7->3-19` | `q16_final_e20_seed42` | 3 | `1365900` | `paper_reproduction/runs/mitigating_da_perclass3b_q16_final_e20_seed42_14-7_to_3-19_b128_s42_20260709_1600/results.json` |

启动健康检查：四个PID均在运行，命令行参数与计划一致；GPU0-3有训练负载。本地SSH检查已确认断开。

## 第三轮结果

四个`perclass3b`结果均已完成并拉取到`remote_artifacts`。第三轮使用`target_model_selection=final`，因此`selected`就是最终epoch后全量target eval，不再使用`target_loss_best` oracle选点。

| 候选 | selected | 相对前序最佳selected66.59% | 相对论文92.42% | history curve max | 最差类acc | per-class target acc |
|---|---:|---:|---:|---:|---:|---|
| `q16_final_e20` | 70.97% | +4.38pp | -21.45pp | 70.47%@6 | 41.05% | `[74.78,71.33,41.05,44.00,98.25,96.40]` |
| `q8_final_e20` | 53.86% | -12.73pp | -38.56pp | 54.70%@16 | 8.38% | `[49.53,30.60,8.38,42.62,93.73,98.32]` |
| `q16_zip_epoch_e20` | 64.52% | -2.07pp | -27.90pp | 70.64%@13 | 36.73% | `[62.68,43.18,48.30,36.73,98.00,98.22]` |
| `q16_final_e20_seed42` | 48.77% | -17.82pp | -43.65pp | 53.01%@13 | 0.12% | `[67.33,67.27,12.62,55.67,89.58,0.12]` |

第三轮结论：

1. 延长`quota16_floor05`到20epoch并改用`final`口径有效，`14-7->3-19`从前序selected66.59%提升到70.97%，但仍未超过前序独立run的curve max76.58%，距离论文92.42%仍差21.45pp。
2. `q8_final_e20`明显变差，说明把每类伪标签从16降到8会让`20-15`几乎学不起来；不是简单“少选更准”。
3. `q16_zip_epoch_e20`训练中峰值约70.64%，但最终回落到64.52%；`zip_min+epoch`不能独立解决差距。
4. `seed42`显著塌缩，`8-20`仅0.12%，说明该任务对初始化/loader顺序高度敏感。单seed高低不能解释论文差距，但必须报告方差。
5. 低分仍集中在`20-15`/`20-19`：最佳`q16_final_e20`中两类只有41.05%和44.00%。下一轮优先验证子agent指出的BN/批大小敏感性，以及`quota16+floor0.7`是否能提高这两类伪标签质量。

## 第四轮计划

只围绕第三轮最佳`q16_final_e20`继续，不扩展任务：

| 候选 | 参数 | 假设 |
|---|---|---|
| `q16_b64_final_e20` | `batch_size=64, quota16, floor0.5, final` | 若BN/批统计或batch级伪标签排序影响类对塌缩，小batch会改变`20-15/20-19`轨迹 |
| `q16_b256_final_e20` | `batch_size=256, quota16, floor0.5, final` | 大batch使BN统计和每类top-k更稳定，可能减少类对吸附 |
| `q16_floor07_final_e20` | `batch_size=128, quota16, floor0.7, final` | 保留每类quota覆盖，同时提高置信底线，测试能否过滤`20-15/20-19`低精伪标签 |

本轮launcher：`paper_reproduction/mitigating_receiver_impact_da/launch_perclass4_20260709_1620.sh`。

启动前N607检查发现GPU0-5已有无关`dadda_cross_receiver`实验运行，PID`1363543`-`1363548`，每卡1个训练进程；本轮不干预。第四轮改用GPU6/7：`q16_b64_final_e20`和`q16_floor07_final_e20`放GPU6，`q16_b256_final_e20`放GPU7，未超过每GPU最多两个训练任务的默认约束。

本地验证：`bash -n ./paper_reproduction/mitigating_receiver_impact_da/launch_perclass4_20260709_1620.sh`通过；SHA256为`b132186e223d607d74689b1fae409d8cf11d78afac34d0fc36868299a805acc1`。远端同步后SHA一致且`bash -n`通过。

16:20 CST启动：

| run id | 候选 | GPU | PID | 远端结果路径 |
|---|---|---:|---:|---|
| `mitigating_da_perclass4_q16_b64_final_e20_14-7_to_3-19_s20260710_20260709_1620` | `q16_b64_final_e20` | 6 | `1374624` | `paper_reproduction/runs/mitigating_da_perclass4_q16_b64_final_e20_14-7_to_3-19_s20260710_20260709_1620/results.json` |
| `mitigating_da_perclass4_q16_b256_final_e20_14-7_to_3-19_s20260710_20260709_1620` | `q16_b256_final_e20` | 7 | `1374626` | `paper_reproduction/runs/mitigating_da_perclass4_q16_b256_final_e20_14-7_to_3-19_s20260710_20260709_1620/results.json` |
| `mitigating_da_perclass4_q16_floor07_final_e20_14-7_to_3-19_b128_s20260710_20260709_1620` | `q16_floor07_final_e20` | 6 | `1374628` | `paper_reproduction/runs/mitigating_da_perclass4_q16_floor07_final_e20_14-7_to_3-19_b128_s20260710_20260709_1620/results.json` |

启动健康检查：3个PID均在运行；GPU6承载2个训练进程，GPU7承载1个训练进程；本地SSH检查已确认断开。

## 第四轮结果

三项均完成并拉取到`remote_artifacts`。结果如下：

| 候选 | selected | 相对第三轮最佳70.97% | 相对论文92.42% | history curve max | 最差类acc | per-class target acc |
|---|---:|---:|---:|---:|---:|---|
| `q16_b64_final_e20` | 15.83% | -55.14pp | -76.59pp | 33.70%@5 | 0.15% | `[0.15,4.42,9.93,40.62,38.35,1.50]` |
| `q16_b256_final_e20` | 54.72% | -16.25pp | -37.70pp | 56.43%@16 | 0.90% | `[92.07,68.67,0.90,7.17,60.48,99.00]` |
| `q16_floor07_final_e20` | 63.00% | -7.97pp | -29.42pp | 73.57%@14 | 21.02% | `[51.73,70.88,21.02,42.80,95.60,95.95]` |

第四轮结论：

1. batch size不是修复方向。`batch_size=64`直接塌缩，`batch_size=256`把`20-15`和`20-19`压到0.90%/7.17%；说明BN/批统计确实敏感，但简单调batch不能接近论文。
2. `quota16+floor0.7`在训练中达到73.57%@epoch14，且epoch14的类级别为`[88.85,70.38,52.15,39.10,95.20,95.75]`，这是当前所有诊断曲线中对`20-15`最好的结果；但final回落到63.00%，说明该设置需要无标签停止准则或保存中间checkpoint，否则不能作为paper-faithful final结果。
3. 当前最佳可直接按final口径报告的`14-7->3-19`仍是`q16_final_e20`的70.97%，比本轮前序selected66.59%提升4.38pp，但仍比论文92.42%低21.45pp。
4. 若允许diagnostic curve max，`q16_floor07_final_e20`给出73.57%，仍低论文18.85pp，并且不能作为严格UDA最终选点，除非实现论文可接受的无标签选择准则。

## 最终验证

| 验证项 | 结果 |
|---|---|
| 聚焦单测 | `conda run -n ssr-gpu python -m pytest tests/test_mitigating_receiver_impact_da.py -q`通过，33项 |
| N607本轮进程 | `mitigating_receiver_impact_da.train`相关进程已退出；GPU6/7空闲 |
| 未干预进程 | GPU0-5仍有无关`dadda_cross_receiver`实验进程，本轮未杀停、未覆盖 |
| SSH清理 | 本地未发现残留`ssh.exe`或到N607:22的ESTABLISHED连接 |
