# 浅中层条件交互、Fishr 与 MixStyle：九行固定实验

本轮遵循源证据报告 `experiments/cvs_source_conditional_geometry/SOURCE_INTERPRETATION.md`。该报告在 energy6 目标成绩采集前已提交于 `bd7b725cd09bc827c6217f27d5e8f112f7b1d765`。目标仍为完整三种子 clean 均值至少 84%，不据目标成绩选择配置或重跑。

## 固定矩阵

|方案|敏感度|辅助约束|三个 model seed|
|---|---|---|---|
|MARGIN_INTERACTION|真 margin 导数／固定层 RMS|TX×RX 条件交互|2026092701、2026092702、2026092703|
|CE_INTERACTION|真 CE 导数／固定层 RMS|TX×RX 条件交互|同上|
|CE_MOMENT_INTERACTION（主方案）|真 CE 导数／固定层 RMS|混合方差＋条件交互|同上|

九行学生均从随机原生旧门控身份网络训练 20 epoch、4440 次更新；cos 指 cosine 学习率。使用另外两个原始 E20 的冻结源概率教师，因此整个方法仍含教师预训练，不能称为完全无预训练。学生不加载 E20 身份或 optimizer。教师完整数据角色、seed、原始 final20 来源由既有 strict verifier 核实。

对照固定为 energy6 的 MARGIN_ENERGY／CE_ENERGY 同 seed，且报告原纯 CE E20 P0。三项跨批比较全部保留，不能选择最高单种子拼接；GPU 数值非确定性使同 seed 不等于完全相同预热权重。

## 多层机制与梯度

在浅层 `time_down` 和中层 `t2.pool` 保留原虚拟逐样本仿射坐标，无新增可训练头。CE 使用未约减损失直接求导，置信度保留在混合二阶图中；margin 为正确 logit 减其余 logits 的 logsumexp。归一化只除以本行学生源 L 标定的正有限层 RMS，不按每样本归一化。

每个辅助更新读取同 day 的 A/B 两个独立 L60，每批为 6 TX×5 RX×2 样本，批内唯一 PID，批间可重合。设 M[y,r] 为每单元敏感度均值，B=M−mean_RX(M)−mean_TX(M)+mean_TX,RX(M)，交互损失为 mean(B_A*B_B)。减去 TX 与 RX 主效应，仅约束其非加性交互。保留有符号估计，不截断负数，不声称已获得物理可辨识的发射机与信道分离。

组合方案同时在相同两次前向上求原有限 L 修正混合方差交叉项。混合方差包含类内方差与类均值散布；它不是逐身份对应约束，故与交互项检验不同性质。

## 标定与训练预算

与 energy6 一致，在完成 444 次身份更新后执行一次标定：第一遍固定 12 对 A/B（24 前向、1440 次 L 读出）计算层 RMS；第二遍复用相同 12 对（另24前向、1440L）测量各项参数梯度。每层每项系数为 `min(10000, 0.2 * mean(CE gradient norm) / (mean(component gradient norm)+eps))`。逐分项测量不可用组合梯度替代。

单项方案使用其标定项；组合方案将两个已标定项等权平均。外层系数固定 1，层间固定除以 2，再乘 ramp，最后统一使用浅／中／其余参数组总上限 0.2。没有两个独立上限，也没有多乘一遍分项系数。后续不重新标定。48次前向／2880L相同，但系数求导为单项48次／组合72次；额外计算和真实耗时单列，不宣称等计算预算。

AdamW、峰值学习率 2e-4 至 1e-6、weight decay 1e-4、L128、warm444／ramp888、每4身份更新一次辅助，继承 energy6。每行999次 A/B 辅助、1998次辅助前向、119880次辅助 L 读出；3996次额外 L 风格或 clean replay 前向，19980次候选 eval 前向。没有辅助 optimizer，没有暖头训练。

MixStyle 在同 TX/day 跨 RX 的合法 L 上搜索四个凸混合候选，包含 clean 作为第五候选，选择本批 CE 最大者；再以与 clean 相同的 Dropout RNG 回放训练。是否实际选中风格逐次记录，不能把3996次路径调用全称为3996次有效扰动。

## 数据、评分、资源与失败处理

L6300 的真标签合法可见；U56700 隐藏 TX，只用固定另外两个原 E20 概率均值和 ALL soft KL；V27000 只在 E5/10/15/20 健康验证。没有 EMA、星地增强、U 风格变换或 target 拟合。既有共享数据验证可复用，本方法变化不重验数据。

固定 final20，无 V/target 选模。每行训练完成即自动冻结、对相同168000条 clean query 预测、独立 truth-last 评分；九行合计1512000次决策。登记全部9行、完整seed、RX/TX/day、Macro-F1、NLL、混淆矩阵、资源及缺项。六种 practical 不在本轮明确 clean-only 范围内。

N607 每卡最多3个总CUDA进程，包含其他任务；先验证可用容量，九行独占输出与日志、唯一 launch owner。技术故障保留产物并停止本批待启动行，不自动覆盖、选择性重跑或干预其他健康任务。

本地检查覆盖：条件主效应零空间、独立 A/B 有限总体期望、混合二阶数值导数、分项系数／共享上限、严格状态保存／拒绝、CPU及原生CUDA训练路径、完整评分入口。远端预检验证实际源角色、教师及三seed原生有限前向/反向；登记检查不读取目标。

固定尺度、分项系数、组成与 outer系数显式保存于 final checkpoint/auxiliary state；预测入口与校准记录严格比对，旧 energy 状态不兼容。保留详细文本、完整逐步gzip、紧凑epoch JSONL/CSV及所有负结果。


发布 VERIFIED：执行提交 `dc7afae5025bc4b4abef053c3f84f81b2863de75`，dispatcher PID 485284。已启动 9/9 行，9 行的实际配置、无学生权重继承、进程和首轮日志核实通过，待运行 0 行；每卡总进程不超过 3。训练完成后逐行自动 clean 测试。当前无本批最终测试成绩，84% 目标待验证。


九行实际固定标定已完成；当前已完成2至3轮，已有6行记录55次辅助更新，其余3行当前完整epoch日志仍为预热结束，后续进度以实时记录为准。工程方法后续六方案×三seed已写入 [engineering_next.md](evidence/engineering_next.md)，尚未实现或发布，不包含在本run九行之内。


**VERIFIED / ANALYZED：9/9训练与clean评分完成。** 三方案各三种子，39960源更新／1512000clean预测；完整源日志及全部校准／组合诊断核验。MARGIN_INTERACTION79.9917%，CE_INTERACTION79.9331%，CE_MOMENT_INTERACTION79.8752%，均未达到84%。完整每seed和RX/TX/day见 [final_analysis.md](final_analysis.md)。分析器分项均值变量覆盖梯度汇总字典的错误已修复，未影响训练/预测/评分；修复后全部39960条源记录完整通过。
