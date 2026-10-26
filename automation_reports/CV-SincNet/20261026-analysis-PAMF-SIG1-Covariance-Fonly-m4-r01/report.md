# 用户指定 F-only 的协方差源分析

状态：PLANNED_SOURCE_ANALYSIS。用户2026-10-26明确撤下DURA配合路线，只用clean及同包星地衍生IQ训练。对应协议和当前目标已更新；本独立分析不读取恢复IQ/权重，不用G性能列晋级，也不改变旧run的原判定。

四个现有模型都保留完整PAMF、单视图SIG_TIME、原L/T1跨RX、联合默认头和普通PL.95/weight1。它们实际训练已经只用clean+H，符合新输入范围；无需重训。对照复用已完成的同架构SIG1-PL，不重训、不重新预测旧已完成目标单元。

只读冻结sourceV27000的raw、lowSuburban、midUrban、lowUrban计数以及实际全U曝光。新范围的必要条件为相对PL的rawclean至多少27条、raw三景总正确数不减，并核实全56700U各30新增访问、1701000曝光和权限边界。逐行独立判断，G列不能影响结果。缺少实际final源结果的行只标等待；源通过不等于正式目标提升。

此范围在部分源结果已知后由用户明确变更，不能伪称原八视图run的预登记条件；旧G-inclusive判定保留。新covariance模型尚无任何正式target访问。后续直接F测试clean及六practical，保留原rawclean≥85.981746%、raw六景>72.096197%、严格同SIG1PL/Sup及原三seed等目标，不再使用DURA指标验收。

全部命令、输入、方法、输出、条件与声明边界见experiment.json。本项是CPU缓存计数分析，不训练或运行新模型、不读新IQ/隐藏U真值/target，不影响健康旧F训练或其他owner。
