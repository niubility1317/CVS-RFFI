# 已有三个冻结模型的目标评估

本次补测 VERIFIED：`20261019-phase1-existing-original-eval-m3-r01`。固定评估父实验 `20260930-phase1-three-original-manysig-m3-r01` 的三个源验证选中模型，没有新增训练、重训分类头或目标选模。实际代码提交 `00b167673ba16d986f54aec184e2e36b1ff27fe0`；使用父环境 torch2.1+cu121、隔离 NumPy1.26.4，在 GPU2 依次完成三个真实 source checkpoint 烟测、预测和独立 truth-last 评分。

| 方法 | 本次目标任务 | 样本数 | 准确率 | Macro-F1 | 论文效果判断 |
|---|---|---:|---:|---:|---|
| ASKNet | ManySig round0，raw 前100包，3个未见RX的day3/4，model/split seed2023 | 3600 | 93.8333% | 93.9307% | 接近作者本折公开日志；完整论文矩阵未验收 |
| WaveMLP | 同上 | 3600 | 90.4444% | 90.4943% | 仅完成本折，不能代替论文四折均值 |
| DIFL | equalized day1，每TX/RX用seed300选200包，3个未见RX；model seed300/split seed32 | 3600 | 82.2778% | 82.5758% | 仍不能认定达到论文效果，特征位置及同条件矩阵有缺口 |

数据任务不同，不跨这三行混排方法优劣。ASKNet、WaveMLP 是结构基线；DIFL 是域泛化方法。所有预测均使用 checkpoint 原头，模型处于 eval/no-grad 状态；没有使用测试标签、类别配额或目标数据拟合。

## 同一行的接收机与日期结果

ASKNet/WaveMLP 每格600包，原 round0 目标 RX 为1-1、1-19、2-1。其匿名 IQ/ID 复用已冻结 IB-RFF round0 导出，物理包和均功率归一化相同。本地重新核对真值与物理 ID，未换数据或另选目标子集。

| 目标RX/day | ASKNet准确率 | WaveMLP准确率 |
|---|---:|---:|
| 1-1/day3 | 86.3333% | 88.8333% |
| 1-1/day4 | 97.0000% | 95.3333% |
| 1-19/day3 | 92.5000% | 81.1667% |
| 1-19/day4 | 92.6667% | 85.6667% |
| 2-1/day3 | 98.3333% | 94.3333% |
| 2-1/day4 | 96.1667% | 97.3333% |

DIFL 使用独立的均衡 day1 导出，保持原包选择，每 RX1200包；没有套用上表的 raw/day3-4 IQ。

| 目标RX/day | DIFL准确率 |
|---|---:|
| 1-1/day1 | 70.7500% |
| 1-19/day1 | 77.5833% |
| 2-1/day1 | 98.5000% |

接收机间差异明显，整体82.28%不能掩盖1-1和1-19上的错误。此观察只描述固定模型的当前结果，不用于调参、重排候选或选择性重跑。

## 与原论文或作者日志的边界

ASKNet 固定作者提交的 ManySig CRD round0 日志为93.47%，本次高0.3633个百分点。这是描述性差值，不在看到结果后新增容差或追认完整论文成功。作者日志路径为 `WiSig/logs/ASKNet/logs/ManySig/ManySig_CRD_date1_2_round0_seed2023.txt`，来源为[作者仓库](https://github.com/BeechburgPieStar/ASKNet-RFF)。

[WaveMLP 作者仓库](https://github.com/BeechburgPieStar/WaveMLP-RFF)报告 ManySig 平均93.13%；本次只有round0，不能把90.44%与完整均值直接相减作为同条件复现差距。其余折和ManyRx没有本次结果。

[DIFL 作者仓库](https://github.com/Edith-xx/Receiver-agnostic-RFFI-CL-)对应 IEEE Communications Letters 2025，DOI `10.1109/LCOMM.2025.3598034`。旧修复在保留分类logits和参数形状的情况下将返回特征移至已有512维fc1，用于分配256维蒸馏和256维CORAL；该特征位置尚未与全文核准。当前单个9:3物理RX划分也不能替代其他RX组合的论文结果。因此不能仅凭模型跑通、源验证成绩或本次单行分数宣称论文效果已达到。

ASKNet/WaveMLP 的既有 exporter 在归一化前已转 float32，DIFL 的均衡导出也为 float32；未宣称从原始pickle到所有训练张量全流程逐位相同。各方法原 source 协议保持，不冒充 CVS 7%标签或星地部署任务。

## 验证与保存

三个 source-only 父记录、完整 resolved config、实际checkpoint、DIFL同run教师来源和实际RX/day/包选择已在加载前核对。各行全部3600条预测固定后才由独立scorer读取对应真值；全部三行最终completion均为SCORED。

已取回10800条冻结预测、truth-side标签/物理ID、scores和完成记录；独立重算accuracy、Macro-F1、完整混淆矩阵和逐RX/day指标，逐值一致。每行3600个物理目标ID唯一，日期集合与注册任务一致。私有证据为 `local_artifacts/rffi_dg_research_20261019/20261019-phase1-existing-original-eval-m3-r01/verified_summary.json` 与 `retrieved/`，不提交权重、数据、预测或原始日志。

目前可给出的结论是：三个历史模型已有真实目标结果，不能再将它们标为“只有源训练”。但完整论文效果仍未获得匹配证据；RIEI-FD、DRIFT 的历史 CVS 结果与论文原协议不同，结论仍见[总核查报告](RFFI_DG_REPRODUCTION_AUDIT_20261019.md)。
