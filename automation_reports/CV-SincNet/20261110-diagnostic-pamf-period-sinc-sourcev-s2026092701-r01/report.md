# 冻结 PAMF 逐周期参考与共享 Sinc 源域诊断

状态：REGISTERED。单行、单模型种子、source-only，无训练、无目标、无 U。运行条件与矩阵见 [experiment.json](experiment.json)。

## 问题与范围

使用当前完整 Tri-SIG-TIME SSL 三视图固定时间重排、seed 2026092701、源域固定 final E50（从零自身 20+10+20；父 run `20261027-phase1-pamf-repl-shuffled-views-m16-r02`），复用同一 Main split source contract/物理 ID。冻结加载器核验每段同契约、自身父链、最终状态和不访问目标；本次不拟合任何参数或分类头。

SourceDataset 载入 V 全部 27000 IQ，按六 TX × 五 RX × 三 day 的原顺序每格前 20 包，共 1800 条进入真实冻结模型。未消费 L 或 U，未读取目标。单次无查询真实 checkpoint 合成 IQ smoke 是入口首步。所有分层、相同序号均为描述统计，不是同步同包或独立 seed。

原 80:160 四个周期使用模型实际公共 CFO/相对时移模板/复增益，提取 12 子载波逐周期相对响应并减去四周期自身均值。共享 Sinc 捕获同一次实际前向 [B,48,256]，先完整滤波再取 80:160；比较对原相同输入先截 80 点再滤波，逐点报告能量与边界差。比较 joint160、原参考输出、周期差、Sinc 时序与 Sinc 周期差的同 TX 跨 RX 与异 TX 参考 RX 的固定 36000 组三元余弦关系，报告按 anchor RX/日期/TX 分层；joint 失败而候选正确及反向比例仅是关系互补代理，不证明融合收益或标签因果。无训练与目标分数。

CPU 两线程；失败保留产物，停止且不干预其他任务。代码、路径、固定模型身份及命令见 experiment.json。实验系统例行检查仅为协议、单 Git、负例/smoke、一次独立 P0/P1 审查、预登记、一次传输校验及终态独立读回，不引入额外 gate。
