# IB-RFF 作者代码复现结果

状态：首折 VERIFIED；四折及论文表验收未完成。

## 固定方案与范围

作者代码为 https://github.com/BeechburgPieStar/IB-RFF ，固定提交 `5a03a04da86c4bbf5cf90f41a55c268bf2152203`。本项目实现与配置见 `paper_reproduction/ibrff_20261019/`，运行代码提交为 `aaebd70187598165c7d937bfdf7de575a2723c1c`。

本次为 ManySig clean CRD 第 0 折，6 TX，作者接收机顺序前 3 RX 为目标、后 9 RX 为源。源数据为 day1/2，目标为 day3/4；每 TX/RX/day 使用非均衡 IQ 前 100 包并按样本均功率归一化。源训练/验证为 7560/3240，目标为 3600。模型和划分 seed 均为 2023。只使用源验证 NLL 选模；不加载作者权重，不另训分类头。

两阶段各最多 200 轮，batch128，Adam lr0.001、weight_decay0；源验证 plateau factor0.1/patience5/min_lr1e-6，早停 patience10。第一阶段 alpha0，第二阶段 alpha0.04，第二阶段恢复同 run 源域最佳父权重并重置优化器。原模型、HSIC、初始化 RNG 和两阶段单步更新与作者训练函数核对一致。原任务为全监督 clean CRD，不代表 CVS 弱标签或星地部署效果。

首次目标预测前固定的工程标准为作者 round0 日志 95.89% 允许低 1 个百分点，即 94.89%。该标准不等于论文表或通用复现标准。预测不读取真值；全部预测固定后，独立 scorer 才读取真值。精确执行参数、环境/CWD、GPU、输出路径、技术停止规则与预登记保存在同 run 本地实验记录中。

## 实际结果

run_id：`20261019-phase1-ibrff-author-manysig-s2023-r02`。N607 普通账户，GPU2，Python3.8、torch1.11.0+cu113、NumPy1.24.4。真实 source checkpoint 无 query 烟测通过，自动训练、冻结、预测及独立评分均完成。

| 指标 | 本次 | 作者公开日志 | 差值 |
|---|---:|---:|---:|
| 准确率 | 95.2778% | 95.89% | −0.6122 个百分点 |
| Macro-F1 | 95.3074% | 未给出 | 不适用 |

达到预先固定的 94.89% 下限。本次只完成 1 折 1 seed，不能据此宣称完整达到论文效果。

| 目标 RX | day3 | day4 | 每格样本数 |
|---|---:|---:|---:|
| 1-1 | 95.0000% | 88.1667% | 600 |
| 1-19 | 99.0000% | 97.3333% | 600 |
| 2-1 | 96.5000% | 95.6667% | 600 |

第一阶段实际 45 轮，源验证选择第 35 轮；第二阶段实际 59 轮，选择第 49 轮。累计 104 轮，未用目标结果选轮次。完整 epoch 的训练/验证累计计时分别为 46.1460 s、65.0801 s，不含启动、数据读取、烟测或预测开销。

远端 `completion.json` 为 `SCORED`。3600 条冻结预测及 truth-side 标签已取回，并独立重算准确率、Macro-F1 和混淆矩阵，与远端结果一致；所选 epoch49 与源验证最小 NLL 一致。私有证据位于 `local_artifacts/rffi_dg_research_20261019/20261019-phase1-ibrff-author-manysig-s2023-r02/verified_summary.json` 和 `retrieved/`；不提交数据、权重、预测或原始日志。

## 技术失败保留

前序 `20261019-phase1-ibrff-author-manysig-s2023-r01` 在正式训练和目标访问前失败：NumPy2 创建的原有 pickle 引用 `numpy._core`，作者 NumPy1.24 环境无法解析。r01 产物保留。r02 仅将读取时模块前缀兼容映射为 `numpy.core`；6 TX 原数组的 dtype、形状、顺序和数值逐项一致，科学配置不变。一次定点独立 P0/P1 复审未发现阻断问题。

完整目标仍包含其余 3 折、第二篇论文的实际复现，以及历史方法的论文效果核验。当前 `paper_effect_verified=false`；不把作者代码日志当作已核准的论文表。
