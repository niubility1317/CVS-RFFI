# 完整 practical 内的等平均 N0 源反证

状态：ANALYZED／VERIFIED_SOURCE_ONLY；固定 N0 范围失败关闭，0 新训练。唯一问题是在原完整传播与接收处理中，等概率选择 0.5N0／1.5N0 的单视图观测是否比固定 N0 提供更多可用身份信息。它是 A_s 噪声先验的源输入检验，不是 A_g、训练收益或最终联合结果。

源模型固定为原完整源普通 CONCAT 的 source-final20（seed2026092701，split392005），与此前源 MMSE 输入检验相同。L6300／源 RX1/3/4/6/8／day1/2/3 全部保留，无训练域留出。U 真值隐藏，U 与 V 统计均不参与本次反证，不访问目标。既有 MMSE E30 负结果关闭，不回调其参数。

## 来源与范围

[Receiver Thermal Noise](https://www.mathworks.com/help/comm/ref/receiverthermalnoise.html) 给出接收噪声功率与系统噪声温度、带宽的关系 N=kTB；[Satellite Link Budget](https://www.mathworks.com/help/satcom/gs/satellite-link-budget.html) 将系统温度分为天线与接收机贡献。这仅支持考虑接收噪声功率变化，不证明本因素范围是实测温度分布，也不证明 CVS 性能收益。0.5／1.5 是本次在读取新 IQ 前固定的工程敏感性范围，不由目标或源准确率选择；期望线性 N0 不变，但平均 SNR、dB 值和接收状态分布不保证不变。

现有代码 residual.py 在接收质量之前设置 N0=10^(-reference_snr_db/10)，再依次确定锁定、残余频偏／相位跟踪、噪声与 AGC。新的两种观测只在这里替换一次 N0，仍由原 ResidualChannel 完成一遍原传播与接收处理。H、状态、几何、阴影、散射、RX/day 硬件、标定与标准高斯噪声 draw 共享。没有新增 AWGN 链、连续两次全信道、MMSE、幅度归一化、裁窗、部署参数、损失、U 规则或分类头。

名义噪声仍对应 reference_snr_db=30；候选两级对应约 33.0103 dB／28.2391 dB。它们只用于源反证，未来若成立也只能用于训练；原正式测试 IQ／SNR／信道保持。继承原 source probe seed20261017931 与其命名随机流以固定已有 H/noise realization，MMSE 一直关闭；这不是反号噪声配对或另一个对话的周期重排。

## 固定最小矩阵与放弃条件

原始 L6300，以及 practical_low_suburban／practical_mid_urban／practical_low_urban 各 5 条处理：名义 N0；0.5N0／1.5N0 且依原规则重算质量；另两条同倍率但固定名义质量的因果控制。固定质量路径只用来区分噪声直接效应与接收处理效应，不能晋级为训练方案。共 100800 个源视图决策、另有真实 CP 的 2 个合成 smoke 前向，1 CP/cache、18900 个潜在传播 realization、94500 次完整残差处理、0 更新／教师／头拟合／目标。

主要指标为两条重算质量视图的等权单次决策风险，不平均 logits，不作推理 ensemble。low 与 mid urban 各至少 +0.5 个百分点；low suburban 至少 −0.25 个百分点；每个源 RX 的 low 增量至少 −1 个百分点。任一失败关闭固定范围，不改倍率／概率／SNR／scene／seed／阈值救跑。源通过也只支持一项另行固定的匹配 E30 训练，不自动训练或扩种子。

## 阶段组织的排查

完整原源日志包括父 20 轮／4440 step 和名义续训 10 轮／2220 step。30 轮始终保留 L 监督，只有 U 叠加阶段为 17 轮，不是只训练了 13 轮有标签数据。V 准确率 E13=97.848148%、E20=97.729630%、E30=98.003704%；U 的实际总目标增量比例约 0.46% 至 0.66%。这些日志不足以将问题归因于 U 阶段；小标量损失也不能证明无梯度冲突。本次不改训练阶段、教师或门控。完整源字段重计见 cvs_source_stage_organization_20261017.json，0 新 IQ／CP／NN／更新／目标读取。

具体行、代码、命令、不可变输出、停止规则和工件以本 run 的 experiment.json 为准。全部真实源结果与独立复算已完成：mid 增益 +0.492063492 pp 未达 +0.5 pp 固定条件，low +0.507936508 pp，suburban low −0.111111111 pp。该固定范围关闭，不放宽临界条件、不启动训练。源输入收益不等于 As 训练或目标收益。

完整100800源预测／306CM／96TX召回复算一致，名义路径对原源对照logits最大差0；100802实际NN／1CP／0更新／0target，墙钟208.848413 s。详见本run报告。
