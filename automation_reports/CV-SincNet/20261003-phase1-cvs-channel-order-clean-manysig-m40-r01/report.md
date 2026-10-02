# CVS信道结构：冻结候选clean独立测试

状态PLANNED。源域16个新模型已完成E200，完整日志与20行源选择均已独立审计。固定规则选择`channel_dual`，本次仅预测该结构的4个seed，与36份已冻结控制预测构成同40行矩阵。所有预测固定后独立truth-last评分；不回流训练、重排候选或选择性重跑。

每模型使用相同168000个物理query、6个注册TX及7个目标RX。只测clean；无LEO、support/SFT或新类。完整复算320条ALL/RX评分、80组汇总、72组配对，并核对旧288条不变。已审查条件流程的实际参数与模型checkpoint contract；launcher将对这次真实冻结checkpoint执行无query smoke后直接继续预测。

[源报告](../20261003-phase1-cvs-channel-order-identity-manysig-m16-r01/report.md) · [本次配置](experiment.json) · [结构与判定](../../../docs/CVS_CHANNEL_ORDER_EXPERIMENT_20261003.md)。当前尚无新测试结果，不宣称性能改善。
