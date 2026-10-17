# 共轭 IQ 必要身份保持检验

状态：ANALYZED；实际 source-only 结果见下文。

研究问题：固定完整 clean 监督的 F＋跨 RX 同 TX 对比学习参照，只检验 I+jQ→I−jQ 是否满足必要的源身份保持条件。它不产生训练收益证据，也不是星地性能验证。

代码提交：f7c9bba3da8863d7fbf278f9e0e3b7ac8fd3ec6b；已推送并独立核对远端相同。6 项相关算子与协议负例通过；一次独立 P0/P1 审查 PASS。真实 checkpoint 无 query smoke 是 launcher 首步，2 个合成样本 PASS 后立即继续，不另设审批。

唯一行、六类 seed、实际命令、环境/CWD、checkpoint 与数据路径、GPU、唯一 owner、输出/log 路径和预期产物见 experiment.json。physical GPU0，普通账户；一次预检：所有 GPU 无 compute process，GPU0 空闲约 24 GB；本 run 的 release、output、log 均不存在。不得修改其他任务。

固定输入：P41 预登记 ORD128 的 own20＋own10、固定 E30 终轮、原默认 cosine30/joint160；检查 own stage20、own EMA 与完整原源域契约。它是冻结诊断来源，不是训练初始化。该参照在其正式测试前即已固定，当前不按正式成绩排序选取。全部 5 个原源 RX，day1/2/3、6TX；既有原 V27000 内平衡 V1800。L/U 新消费为 0，U TX 真值与 target 访问均为 0；不生成 As，不改正式 IQ。

预先固定失败条件：全局准确率下降大于 0.5 个百分点，或任一 RX 下降大于 1.5 个百分点，或原 raw 正确集合保留率小于 99%。任一失败就关闭固定共轭路线，不靠角度、轴、概率、seed 或预算补救。通过也只是必要准入检查，不自动启动 E30/扩种子。

新成本：2 synthetic＋1800 raw＋1800 conjugate＝3602 默认头 NN；0 epoch、0 update、0 teacher；父训练 E30/6660 updates 另列。精确 double-conjugate 逆变换、逐样本功率相等、状态冻结与计数在实际 IQ 上检查；可逆和保能量不能证明身份保持。

技术停止：来源/角色不符、形状非有限、状态变化、NN 超预算、输出冲突或运行错误，保留原产物并停止所属新 run，不自动重试。科学 FAIL 不是技术失败。预测固定后独立重算源域混淆矩阵与配对结果；0 新 NN。

先前的跨视图 SupCon P41 在固定源条件 0/4 达标，已关闭；不将其形式换名重开。原文献亦提醒翻转可能改变硬件指纹：Semi-Supervised RF Fingerprinting with Consistency-Based Regularization，https://arxiv.org/pdf/2304.14795 。本实验因此只做零更新可证伪检验。

## 实际执行与结论

状态：VERIFIED_ANALYZED_SOURCE_NEGATIVE。唯一行 POLARITY-s2026092701 自然完成；不是训练失败。固定共轭候选未通过任何一项必要源检验，关闭该候选，不启动训练或扩种子。全部模型参数/缓冲不变，0 梯度更新；正式测试 IQ 与信道保持原状。

| 同一源 V1800、冻结默认头 | raw | conjugate |
|---|---:|---:|
| 准确率 | 98.111111% | 34.166667% |
| 六类宏平均 F1 | 98.106544% | 28.194455% |
| 真实类对最强错误类间隔／30 | 0.233772039 | −0.022314979 |

总体下降 63.944444 个百分点；原正确集合 1766 包中保留 612 包（34.654587%），修复 3 包，新增错误 1154 包。5 个源 RX 的准确率差分别为 RX1 −69.722222、RX3 −62.500000、RX4 −56.666667、RX6 −66.388889、RX8 −64.444444 个百分点；每个 RX 360 包。逐 TX/RX/day 20 包覆盖核对通过。

预先固定的全局下降≤0.5 个百分点、逐 RX 下降≤1.5 个百分点、原正确保留率≥99% 均 FAIL。预测固定后用独立实现重建 30 个混淆矩阵、全部分层指标与配对关系，最大数值差 0，0 新 NN。

实际共轭变换与逆变换严格对应：I 不变、Q 取负，逐样本幅度平方完全相同，双次变换精确回到原 IQ；无新 FIR、边界、噪声或归一化。相对 IQ L2 均值 1.399431825，不能称为小扰动。

必须限定解释：冻结模型在共轭输入上失败，既可能反映其未学到该不变性，也可能涉及指纹非对称性。它不证明真实 TX 标签改变，更不证明训练永远无法学习该变换。本轮只否定固定判据下直接采用这个算子的准入假设，不宣称普遍物理因果结论。

成本：父模型 E30／6660 次更新属于既有 P41 ORD，不计为本轮新训练。本轮 2 合成＋3600 源前向＝3602 NN，0 epoch／0 update／0 teacher／0 As／0 U 真值／0 target；诊断墙钟 2.161144 s，独立复算与绘图 1.756851 s，传输与等待未纳入。无训练因此没有本轮训练曲线。

执行限制：诊断在 2.16 s 内结束，第一次独立 post-launch 读回时 PID 已消失，未捕获运行中的 PID/CWD/GPU 独立快照。启动 action 保存了 PID/路径，独立读回核实了自然终态、日志增长、smoke、完整产物与结果；不把启动返回码写成 live 验证，也不为补快照重复运行。

验证：6 项聚焦算子/协议负例 PASS，一次独立 P0/P1 PASS，真实 checkpoint 无 query smoke 2 NN PASS，单归档 SHA 本远端匹配，远端编译 PASS，source-only 30 CM 独立复算 PASS。两个本地 Python 环境未装 Matplotlib，未安装或修改环境；使用现有 N607 Matplotlib3.10.8 完成图表。

交付：[逐视图/逐层指标](source_metrics_recount.csv)、[逐 RX/day/TX 配对表](paired_source.csv)、[源 RX 与原正确保留率图](source_RX_guard.png)、[混淆矩阵](source_confusion.png)，两图均附同名 PDF。源预测与真实 IQ/权重/原始日志保持私有，未提交 Git。完整命令及逐行配置见 experiment.json；代码为 experiments/cvs_polarity_guard42，独立复算为 analysis/cvs_polarity_source_results_20261018.py。

本轮没有 formal urban 预测，因而不能回答新的 urban 退化改善幅度或 clean 训练代价。既有 P41 的 all7 正式结果与训练曲线见 ../20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01/report.md：SAT192 相对 ORD128 clean −1.427381、low urban −0.147619、urban 均值 −0.256548 个百分点；单 seed，未证明跨 seed 重现。此处仅报告已经冻结的终态，不用于调参/选模/选择重跑。

主目标保持 ACTIVE：完整 clean 监督份额路线、E30、全部 5 个源 RX、F＋跨 RX 同 TX 对比学习、默认头与正式 practical 条件继续固定；新的有效 IQ Ag、B00/B10/B01/B11 因子证据、low urban>60% 和跨 seed 净收益仍未完成。下一步须先找到有独立源证据的观测因素，不能把已关闭算子换名或再堆额外损失。
