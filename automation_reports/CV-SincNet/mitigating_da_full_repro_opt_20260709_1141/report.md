# Mitigating Receiver Impact DA full reproduction optimization

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | `mitigating_da_full_repro_opt_20260709_1141` |
| 时间 | 2026-07-09 11:41 Asia/Hong_Kong |
| 操作者 | Codex |
| 目标 | 继续定位`14-7->3-19`与论文Proposed 92.42%的剩余差距，做最小优化补丁，然后跑完整Table II Proposed复现实验 |
| 上轮最佳 | `14-7->3-19=71.96%`，仍低论文`-20.46pp`；`1-1->1-19=92.35%`，低论文`-3.09pp` |
| 论文显式超参 | `lr=0.0006`、`lambda=0.005`、`mu=0.5`、`m=7`、`tau=0.7` |

## 已读规则和状态

| 项目 | 结果 |
|---|---|
| `AGENTS.md` | 已读 |
| `项目.md` | 已读 |
| N607 preflight | 通过，2026-07-09 11:40 CST |
| 根目录Git | `E:\type10-7`不是Git仓库 |
| 代码仓库Git | `E:\type10-7\github_publish\CVS-RFFI-repo`，后续修改在该Git承载面完成 |

## 初始假设

上轮完成结果显示：

- raw-logit伪标签阈值是`1-1->1-19`主因，paper CPL概率路径能显著恢复。
- `14-7->3-19`仍有20pp缺口；长训30/50轮坍塌，batch/seed影响大，target loss best和target accuracy在hard task上不稳定相关。
- 公开trainer包含每batch `lr_scheduler.step()`，而当前复现训练循环使用常数学习率；论文未公开scheduler、epoch、batch size、seed和完整模型/数据配置。

本轮先完整解析已完成曲线，再验证“学习率调度/早停记录缺失”是否是剩余缺口的重要原因。

## 曲线复核与根因收敛

已完成结果的逐epoch曲线显示，`14-7->3-19`并不是显式论文超参错误导致的单调低分，而是hard target receiver下伪标签自举早期轨迹极不稳定：

| 证据 | 观察 | 解释 |
|---|---|---|
| 早期类别权重 | 当前最佳`batch=128,seed=20260710`在epoch1出现`class_weight_max=533.33`，epoch2后才回落到约`10.67`和`1.7`附近 | paper Eq.9 / official `get_class_weight`在少数类早期预测计数很小但非零时会放大到极端权重，hard task更容易被放大错误梯度带偏 |
| 伪标签质量 | 最佳完成行epoch3目标精度`71.96%`，selected pseudo acc约`69.71%`；部分长训epoch的伪标签精度更高但target accuracy不随之稳定提升 | 自训练可形成阶段性高精度，但目标loss、伪标签精度和最终target accuracy不同步 |
| 训练长度 | 30/50轮没有修复，反而坍塌；15轮曲线最高到`72.62%`但target loss误选低精度epoch | 问题不是“跑得不够久”，而是早期权重/伪标签动态和模型选择口径 |
| 任务差异 | `1-1->1-19`用paper CPL概率路径已到`92.35%`，距论文仅`-3.09pp` | 共同实现路径已基本可行，剩余大缺口集中在`14-7->3-19`这个hard pair |

本轮优化原则：保持论文显式超参`tau=0.7,m=7,lambda=0.005,mu=0.5,lr=0.0006`不变；不改默认paper-faithful路径；只验证类别权重稳定化是否能抑制首轮极端权重和坏驻点。

## 第六轮类别权重稳定化验证矩阵

共同设置：`Proposed`、`14-7->3-19`、`epochs=10`、`batch=128`、`seed=20260710`、`source_pretrain_epochs=0`、`adapt_start_epoch=0`、`base_tau=0.7`、`estimate_steps=7`、`kl_estimator_mode=mine_ma`、`mine_update_scale=0.5`、`class_prior_mode=source`、`pseudo_threshold_mode=paper`、`pseudo_score_mode=probability`、`target_model_selection=target_loss_best`。

| 变体 | 额外参数 | 目的 |
|---|---|---|
| `cw_guard_025_4_current` | `--class-weight-timing current --class-weight-smoothing 1.0 --class-weight-clip-min 0.25 --class-weight-clip-max 4.0 --class-weight-mean-normalize` | 抑制`533x`极端权重，保留类不均衡方向 |
| `cw_guard_05_2_current` | `--class-weight-timing current --class-weight-smoothing 1.0 --class-weight-clip-min 0.5 --class-weight-clip-max 2.0 --class-weight-mean-normalize` | 更强稳定化，检查是否牺牲必要类别补偿 |
| `cw_guard_025_4_previous` | `--class-weight-timing previous --class-weight-smoothing 1.0 --class-weight-clip-min 0.25 --class-weight-clip-max 4.0 --class-weight-mean-normalize` | 避免当前batch预测立即影响同batch权重 |
| `cw_guard_01_10_smooth10_current` | `--class-weight-timing current --class-weight-smoothing 10.0 --class-weight-clip-min 0.1 --class-weight-clip-max 10.0 --class-weight-mean-normalize` | 更接近平滑先验，保留更宽类别补偿范围 |

成功判据：若完成行超过`71.96%`且曲线不出现早期权重爆炸，则把该组合落为显式优化profile，保持默认复现口径不变，再跑完整Table II Proposed。

## 第六轮启动记录

11:48 CST启动，远端工作目录`/home/szu2070436088/2510044040/CV-SincNet`，Python环境`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。启动前只读检查显示GPU3、GPU7空闲；GPU0/1/2/4/5/6已有DADDA进程，本轮未干预。

| run id | GPU | PID | 日志 | 结果 |
|---|---:|---:|---|---|
| `mitigating_da_cw_guard_025_4_current_e10_b128_s20260710_14-7_to_3-19_20260709_1148` | 3 | `1245705` | `paper_reproduction/logs/mitigating_da_cw_guard_025_4_current_e10_b128_s20260710_14-7_to_3-19_20260709_1148/run.out` | `paper_reproduction/runs/mitigating_da_cw_guard_025_4_current_e10_b128_s20260710_14-7_to_3-19_20260709_1148/results.json` |
| `mitigating_da_cw_guard_05_2_current_e10_b128_s20260710_14-7_to_3-19_20260709_1148` | 7 | `1245707` | `paper_reproduction/logs/mitigating_da_cw_guard_05_2_current_e10_b128_s20260710_14-7_to_3-19_20260709_1148/run.out` | `paper_reproduction/runs/mitigating_da_cw_guard_05_2_current_e10_b128_s20260710_14-7_to_3-19_20260709_1148/results.json` |
| `mitigating_da_cw_guard_025_4_previous_e10_b128_s20260710_14-7_to_3-19_20260709_1148` | 3 | `1245709` | `paper_reproduction/logs/mitigating_da_cw_guard_025_4_previous_e10_b128_s20260710_14-7_to_3-19_20260709_1148/run.out` | `paper_reproduction/runs/mitigating_da_cw_guard_025_4_previous_e10_b128_s20260710_14-7_to_3-19_20260709_1148/results.json` |
| `mitigating_da_cw_guard_01_10_smooth10_current_e10_b128_s20260710_14-7_to_3-19_20260709_1151` | 7 | `1247633` | `paper_reproduction/logs/mitigating_da_cw_guard_01_10_smooth10_current_e10_b128_s20260710_14-7_to_3-19_20260709_1151/run.out` | `paper_reproduction/runs/mitigating_da_cw_guard_01_10_smooth10_current_e10_b128_s20260710_14-7_to_3-19_20260709_1151/results.json` |

启动后本地检查：未发现残留`ssh.exe`或到N607的`TCP:22`连接。

## 完整Proposed复现最终结果

13:20 CST重新执行N607只读预检，通过：直连`N607`、项目根目录`/home/szu2070436088/2510044040/CV-SincNet`、GPU可见。13:21 CST将完整复现的6个`results.json`拉取到本地：

`E:\type10-7\automation_reports\CV-SincNet\mitigating_da_full_repro_opt_20260709_1141\remote_artifacts\`

拉取后本地检查：未发现残留`ssh.exe`或到N607的`TCP:22`连接。

### 每任务独立并行run（主结果）

该结果用于主对比，因为每个任务独立启动、独立初始化，避免串行all-task run中后续任务继承前序任务消耗后的RNG状态。

| 任务 | 论文Proposed | 复现selected | 差值 | 曲线最高 | 曲线最高差值 | best-loss epoch | max epoch | 最后epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d01->d23` | 93.34% | 93.69% | +0.35pp | 94.99% | +1.65pp | 1 | 2 | 94.94% |
| `14-7->3-19` | 92.42% | 66.59% | -25.83pp | 76.58% | -15.84pp | 5 | 8 | 61.05% |
| `1-1->1-19` | 95.44% | 47.90% | -47.54pp | 61.43% | -34.01pp | 2 | 9 | 50.34% |
| `1-1->8-8` | 99.78% | 89.91% | -9.87pp | 89.91% | -9.87pp | 1 | 1 | 79.68% |
| `7-7->8-8` | 99.74% | 61.47% | -38.27pp | 61.47% | -38.27pp | 5 | 5 | 58.70% |
| 平均 | 96.14% | 71.91% | -24.23pp | 76.87% | -19.27pp | - | - | 68.94% |

主结论：`d01->d23`已对齐并略高于论文；`1-1->8-8`明显改善但仍差9.87pp；`14-7->3-19`仍是核心缺口，selected为66.59%，曲线最高76.58%，说明`target_loss_best`没有选中最高准确率epoch；`1-1->1-19`和`7-7->8-8`在统一配置下出现坏轨迹，表明当前复现仍有强任务/初始化敏感性。

### 串行all-task run（辅助结果）

串行结果保留作完整run证据，但不作为主结论，因为同一个Python进程顺序训练5个任务会使后续任务受到任务顺序和RNG消耗影响。

| 任务 | 论文Proposed | 复现selected | 差值 | 曲线最高 | 曲线最高差值 | best-loss epoch | max epoch | 最后epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d01->d23` | 93.34% | 96.98% | +3.64pp | 96.98% | +3.64pp | 10 | 10 | 96.98% |
| `14-7->3-19` | 92.42% | 54.45% | -37.97pp | 54.45% | -37.97pp | 4 | 4 | 29.55% |
| `1-1->1-19` | 95.44% | 82.84% | -12.60pp | 82.85% | -12.59pp | 10 | 1 | 82.84% |
| `1-1->8-8` | 99.78% | 51.05% | -48.73pp | 66.62% | -33.16pp | 3 | 8 | 50.11% |
| `7-7->8-8` | 99.74% | 97.35% | -2.39pp | 97.35% | -2.39pp | 10 | 10 | 97.35% |
| 平均 | 96.14% | 76.54% | -19.61pp | 79.65% | -16.49pp | - | - | 71.37% |

### 本轮问题定位结论

| 检查项 | 证据 | 结论 |
|---|---|---|
| 论文显式超参 | 当前全量run使用`tau=0.7,m=7,lambda=0.005,mu=0.5`对应路径；`pseudo_score_mode=probability`、`class_prior_mode=source`、`kl_estimator_mode=mine_ma`、`mine_update_scale=0.5` | 已回到论文显式设置和公共代码可确认路径 |
| 类别权重稳定化 | `cw_guard_05_2_current`最高仅70.54%，其它裁剪/平滑更低 | 类别权重爆炸是放大器，不是足以补齐论文差距的单点原因 |
| 每batch scheduler | `sched600/sched2500`均未超过无scheduler当前最佳，部分轨迹显著恶化 | 公共trainer的StepLR路径已实现并验证，但不是缺口主因 |
| 模型选择 | `14-7->3-19`独立run selected66.59%、曲线最高76.58%；`1-1->1-19` selected47.90%、曲线最高61.43% | `target_loss_best`与目标准确率不同步，会漏选高准确率epoch |
| 任务敏感性 | `7-7->8-8`串行97.35%、独立61.47%；`1-1->1-19`历史最优92.35%依赖`source_pretrain=10`，统一配置下47.90% | 单一统一配置无法稳定复现所有跨接收机pair，剩余差距更像未公开训练细节/初始化策略/任务特定日程造成 |

### artifact哈希

| 本地文件 | SHA256 |
|---|---|
| `parallel_d01_to_d23_results.json` | `c7b8bcbe07bbb5fbdc944305406b4c4a21dad50292836a836335b95611134acf` |
| `parallel_14-7_to_3-19_results.json` | `5c53f05c613b681b4b9e83038e5f28bf86b63bc608683a47e783ae9c09314b3a` |
| `parallel_1-1_to_1-19_results.json` | `ca19d54324c3fe63406af9ea387110d08f0d944b90ffdc03d3dfcdfc9894d3f4` |
| `parallel_1-1_to_8-8_results.json` | `79d00fb6b7393b89eadfea55eadd84fb4c27615fd2143d655175cb83a5a6648b` |
| `parallel_7-7_to_8-8_results.json` | `91b6dd35c1500c3c323502a423489209a55602e1f4116a521b358ce6582cb20b` |
| `serial_all_tasks_results.json` | `b00081ff79e0a7f88c8384eea0ec7261a2116c5c0148dc757a44caafb1e673eb` |

## 并行run中间结果

已完成4个cross-receiver任务，`d01->d23`仍在运行。

| 任务 | 完成行 | 曲线最高 | 最高epoch | best loss epoch | 论文Proposed | 完成行差值 | 曲线最高差值 | 关键诊断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `14-7->3-19` | 66.59% | 76.58% | 8 | 5 | 92.42% | -25.83pp | -15.85pp | 训练轨迹超过此前72%，但target loss未选中最高acc |
| `1-1->1-19` | 47.90% | 61.43% | 9 | 2 | 95.44% | -47.54pp | -34.01pp | 统一hard-pair配置不适合该任务；此前`source_pretrain=10`可到92.35% |
| `1-1->8-8` | 89.91% | 89.91% | 1 | 1 | 99.78% | -9.87pp | -9.87pp | 该任务显著改善，但仍未达论文 |
| `7-7->8-8` | 61.47% | 61.47% | 5 | 5 | 99.74% | -38.27pp | -38.27pp | class weight峰值仍很高，伪标签精度不足 |

12:40 CST监控：串行全任务run仍在运行且GPU3有训练负载，但尚未完成第一个大任务checkpoint。由于N607其余GPU空闲，为缩短全任务结果等待时间，追加同配置的每任务独立并行run；串行run保留，不中断。

12:41 CST并行启动记录：

| run id | 任务 | GPU | PID | 结果 |
|---|---|---:|---:|---|
| `mitigating_da_table2_proposed_opt_parallel_d01_to_d23_b128_s20260710_20260709_1241` | `d01->d23` | 0 | `1271930` | `paper_reproduction/runs/mitigating_da_table2_proposed_opt_parallel_d01_to_d23_b128_s20260710_20260709_1241/results.json` |
| `mitigating_da_table2_proposed_opt_parallel_14-7_to_3-19_b128_s20260710_20260709_1241` | `14-7->3-19` | 1 | `1271932` | `paper_reproduction/runs/mitigating_da_table2_proposed_opt_parallel_14-7_to_3-19_b128_s20260710_20260709_1241/results.json` |
| `mitigating_da_table2_proposed_opt_parallel_1-1_to_1-19_b128_s20260710_20260709_1241` | `1-1->1-19` | 2 | `1271934` | `paper_reproduction/runs/mitigating_da_table2_proposed_opt_parallel_1-1_to_1-19_b128_s20260710_20260709_1241/results.json` |
| `mitigating_da_table2_proposed_opt_parallel_1-1_to_8-8_b128_s20260710_20260709_1241` | `1-1->8-8` | 4 | `1271936` | `paper_reproduction/runs/mitigating_da_table2_proposed_opt_parallel_1-1_to_8-8_b128_s20260710_20260709_1241/results.json` |
| `mitigating_da_table2_proposed_opt_parallel_7-7_to_8-8_b128_s20260710_20260709_1241` | `7-7->8-8` | 5 | `1271938` | `paper_reproduction/runs/mitigating_da_table2_proposed_opt_parallel_7-7_to_8-8_b128_s20260710_20260709_1241/results.json` |

启动后本地检查：未发现残留`ssh.exe`或到N607的`TCP:22`连接。

## 第七轮结果

| 变体 | 完成行 | 曲线最高 | 最高epoch | best loss epoch | 最后epoch | 最后LR | 相对当前最佳71.96% | 相对论文92.42% | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sched600_e10_s20260710` | 23.63% | 31.52% | 3 | 2 | 18.10% | 0.0001296 | -48.33pp | -68.79pp | StepLR600破坏当前好seed轨迹 |
| `sched600_e10_s20260712` | 64.93% | 70.63% | 7 | 3 | 50.93% | 0.0001296 | -7.04pp | -27.50pp | 仍低于无scheduler同seed70.94% |
| `sched600_e15_s20260710` | 48.53% | 59.18% | 15 | 3 | 59.18% | 0.00007776 | -23.43pp | -43.89pp | 15轮不恢复 |
| `sched2500_e20_s20260710` | 50.69% | 67.30% | 16 | 1 | 47.07% | 0.00036 | -21.27pp | -41.73pp | 公共模板常见StepLR也未修复 |

结论：公共trainer“每batch scheduler step”路径已实现并验证，但`14-7->3-19`没有提升；`StepLR`反而加重坏轨迹或无法阻止坍塌。当前全任务复现采用仍然最强的统一配置：`batch=128,seed=20260710,source_pretrain=0,adapt_start=0,paper CPL probability,class_prior=source,class_weight_timing=current,no scheduler,no class-weight guard`。

## 完整Table II Proposed复现计划

共同设置：只跑论文方法`Proposed`；任务为`d01->d23,14-7->3-19,1-1->1-19,1-1->8-8,7-7->8-8`；`epochs=10`、`batch=128`、`seed=20260710`、`source_pretrain_epochs=0`、`adapt_start_epoch=0`、`base_tau=0.7`、`estimate_steps=7`、`kl_estimator_mode=mine_ma`、`mine_update_scale=0.5`、`class_prior_mode=source`、`class_weight_timing=current`、`pseudo_threshold_mode=paper`、`pseudo_score_mode=probability`、`target_model_selection=target_loss_best`、`lr_scheduler_mode=none`。

12:18 CST启动记录：

| run id | GPU | PID | 日志 | 结果 |
|---|---:|---:|---|---|
| `mitigating_da_table2_proposed_opt_b128_s20260710_20260709_1218` | 3 | `1260580` | `paper_reproduction/logs/mitigating_da_table2_proposed_opt_b128_s20260710_20260709_1218/run.out` | `paper_reproduction/runs/mitigating_da_table2_proposed_opt_b128_s20260710_20260709_1218/results.json` |

启动后本地检查：未发现残留`ssh.exe`或到N607的`TCP:22`连接。

## 第六轮结果

| 变体 | 完成行 | 曲线最高 | best loss epoch | `class_weight_max`峰值 | 相对当前最佳71.96% | 相对论文92.42% | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `cw_guard_025_4_current` | 40.09% | 43.66% | 1 | 2.43 | -31.87pp | -52.33pp | 权重爆炸被压制，但伪标签质量不足，明显退化 |
| `cw_guard_05_2_current` | 70.54% | 70.54% | 6 | 1.86 | -1.42pp | -21.88pp | 最好稳定化候选，但仍低于原71.96% |
| `cw_guard_025_4_previous` | 33.43% | 39.51% | 5 | 2.57 | -38.54pp | -58.99pp | previous权重时序不适合该pair |
| `cw_guard_01_10_smooth10_current` | 48.65% | 48.65% | 1 | 2.72 | -23.31pp | -43.77pp | 更强平滑没有改善 |

结论：类别权重极端值是坏轨迹的放大器，但不是足以解释`71.96% -> 92.42%`缺口的主因；单纯平滑/裁剪会降低类别补偿，未超过当前最佳。下一步转向公共trainer已确认存在而本地尚未实现的每batch learning-rate scheduler路径。该路径仍属于未公开配置诊断，不改变默认paper-faithful设置。

## 本地scheduler补丁

| 文件 | 修改 | 验证 |
|---|---|---|
| `github_publish/CVS-RFFI-repo/paper_reproduction/mitigating_receiver_impact_da/train.py` | 新增默认关闭的`--lr-scheduler-mode none|step`、`--lr-step-size`、`--lr-gamma`；启用`step`时按公共trainer路径在每个batch后对E/C和T optimizer各执行一次scheduler step；记录`lr_scheduler_*`与epoch级`lr_ec/lr_t` | `conda run -n ssr-gpu python -m pytest tests/test_mitigating_receiver_impact_da.py -q`通过，29项 |
| `github_publish/CVS-RFFI-repo/tests/test_mitigating_receiver_impact_da.py` | 增加scheduler每batch步进和Table II runner记录测试 | 同上 |

本地`train.py` SHA256：`DF39ACB304FF525176FECC5FCB9660AE323124A1F282445DCC2432817B1B5DAE`。默认仍为`lr_scheduler_mode=none`，既有paper-faithful/official兼容路径不自动改变。

Git提交：`172eb03 Add scheduler option for receiver impact DA reproduction`。

同步记录：已用`scp -F tools\n607_ssh_config`同步到`N607:/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/mitigating_receiver_impact_da/train.py`。远端SHA256为`df39acb304ff525176fecc5fcb9660ae323124a1f282445dcc2432817b1b5dae`，并确认`--lr-scheduler-mode`、`--lr-step-size`、`--lr-gamma`出现在CLI帮助中。同步后未发现残留`ssh.exe`或到N607的`TCP:22`连接。

## 第七轮scheduler验证矩阵

共同设置：`Proposed`、`14-7->3-19`、`batch=128`、`source_pretrain_epochs=0`、`adapt_start_epoch=0`、`base_tau=0.7`、`estimate_steps=7`、`kl_estimator_mode=mine_ma`、`mine_update_scale=0.5`、`class_prior_mode=source`、`pseudo_threshold_mode=paper`、`pseudo_score_mode=probability`、`class_weight_timing=current`、`target_model_selection=target_loss_best`。

| 变体 | epochs | seed | scheduler | 目的 |
|---|---:|---:|---|---|
| `sched600_e10_s20260710` | 10 | 20260710 | `StepLR(step_size=600,gamma=0.6)` | 当前最佳主路径最小scheduler A/B |
| `sched600_e15_s20260710` | 15 | 20260710 | `StepLR(step_size=600,gamma=0.6)` | 检查scheduler是否缓解15轮坍塌和target loss误选 |
| `sched2500_e20_s20260710` | 20 | 20260710 | `StepLR(step_size=2500,gamma=0.6)` | 对齐公共模板常见`step_size=2500`，更长训练才触发衰减 |
| `sched600_e10_s20260712` | 10 | 20260712 | `StepLR(step_size=600,gamma=0.6)` | 验证scheduler是否降低seed敏感性 |

## 第七轮启动记录

12:06 CST启动。启动前GPU3和GPU7空闲，GPU0/1/2/4/5/6仍有其他DADDA进程，本轮未干预；每个GPU最多两个本轮mitigating任务。

| run id | GPU | PID | 日志 | 结果 |
|---|---:|---:|---|---|
| `mitigating_da_sched600_e10_b128_s20260710_14-7_to_3-19_20260709_1206` | 3 | `1254704` | `paper_reproduction/logs/mitigating_da_sched600_e10_b128_s20260710_14-7_to_3-19_20260709_1206/run.out` | `paper_reproduction/runs/mitigating_da_sched600_e10_b128_s20260710_14-7_to_3-19_20260709_1206/results.json` |
| `mitigating_da_sched600_e15_b128_s20260710_14-7_to_3-19_20260709_1206` | 7 | `1254706` | `paper_reproduction/logs/mitigating_da_sched600_e15_b128_s20260710_14-7_to_3-19_20260709_1206/run.out` | `paper_reproduction/runs/mitigating_da_sched600_e15_b128_s20260710_14-7_to_3-19_20260709_1206/results.json` |
| `mitigating_da_sched2500_e20_b128_s20260710_14-7_to_3-19_20260709_1206` | 3 | `1254708` | `paper_reproduction/logs/mitigating_da_sched2500_e20_b128_s20260710_14-7_to_3-19_20260709_1206/run.out` | `paper_reproduction/runs/mitigating_da_sched2500_e20_b128_s20260710_14-7_to_3-19_20260709_1206/results.json` |
| `mitigating_da_sched600_e10_b128_s20260712_14-7_to_3-19_20260709_1206` | 7 | `1254710` | `paper_reproduction/logs/mitigating_da_sched600_e10_b128_s20260712_14-7_to_3-19_20260709_1206/run.out` | `paper_reproduction/runs/mitigating_da_sched600_e10_b128_s20260712_14-7_to_3-19_20260709_1206/results.json` |

启动后本地检查：未发现残留`ssh.exe`或到N607的`TCP:22`连接。

## 最终复现结论（索引）

完整结果已拉取到`E:\type10-7\automation_reports\CV-SincNet\mitigating_da_full_repro_opt_20260709_1141\remote_artifacts\`。主结果采用每任务独立并行run；串行all-task run仅作辅助，因为任务顺序会改变RNG状态。

| 任务 | 论文Proposed | 主复现selected | 差值 | 曲线最高 | 曲线最高差值 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `d01->d23` | 93.34% | 93.69% | +0.35pp | 94.99% | +1.65pp | 已对齐 |
| `14-7->3-19` | 92.42% | 66.59% | -25.83pp | 76.58% | -15.84pp | 仍是核心缺口，`target_loss_best`漏选最高准确率epoch |
| `1-1->1-19` | 95.44% | 47.90% | -47.54pp | 61.43% | -34.01pp | 统一配置出现坏轨迹；此前任务特定`source_pretrain=10`可到92.35% |
| `1-1->8-8` | 99.78% | 89.91% | -9.87pp | 89.91% | -9.87pp | 明显改善但未达论文 |
| `7-7->8-8` | 99.74% | 61.47% | -38.27pp | 61.47% | -38.27pp | 对初始化/任务顺序高度敏感，串行辅助run为97.35% |
| 平均 | 96.14% | 71.91% | -24.23pp | 76.87% | -19.27pp | 仍未完整复现论文Proposed |

最终判断：已修正并验证论文显式设置路径（`tau=0.7,m=7,lambda=0.005,mu=0.5`）、source类别先验、CPL概率阈值、MINE移动平均KL路径、类别权重时序和公共trainer scheduler路径；类别权重裁剪/平滑与StepLR均未补齐差距。剩余差距主要来自硬跨接收机pair的伪标签自训练坏轨迹、`target_loss_best`与准确率不一致、以及论文未公开的训练日程/初始化/任务特定设置。
