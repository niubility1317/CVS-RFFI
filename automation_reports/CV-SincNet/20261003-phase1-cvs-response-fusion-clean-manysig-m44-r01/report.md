# CVS约束响应融合：冻结候选clean独立测试

状态PLANNED。8个新模型已完成E200，80000步日志和28行源选择已独立核对。固定规则选择`response_anchor_mean`，只预测该结构的4个seed，与40份既有冻结预测构成44行矩阵。所有预测固定后独立truth-last评分；不回流训练、重排候选或选择性重跑。

每模型使用相同168000个物理query、6个注册TX及7个目标RX。只测clean；无LEO、support/SFT或新类。完整复算352条ALL/RX评分、88组汇总与80组配对，并核对旧320条不变。launcher对实际冻结checkpoint执行无query smoke后继续预测。

[源报告](../20261003-phase1-cvs-response-fusion-identity-manysig-m8-r01/report.md) · [实际矩阵](experiment.json) · [结构](../../../docs/CVS_RESPONSE_FUSION_DESIGN_20261003.md) · [论文证据边界](../../../docs/CVS_CHANNEL_RESPONSE_NOVELTY_20261003.md)。当前尚无新测试结果，不宣称性能改善。
