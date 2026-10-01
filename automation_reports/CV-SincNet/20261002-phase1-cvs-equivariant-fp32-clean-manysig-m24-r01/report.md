# CVS 全精度 FP32：选中候选 clean 确认

状态 LOCAL_VERIFIED，尚未访问本轮新 query。固定源规则依据 4 个新 scratch E200 与 4 原残差源控制选择 `equivariant_memory`。4 个新冻结模型＋20份原控制预测，共24行、168000原物理query/6TX/7RX，全部固定后独立truth-last评分。

训练与预测同一固定FP32策略，关闭cuDNN TF32，其他实际flags必须符合源payload；旧控制保留历史精度，只读复用。92项检查、唯一矩阵路径P1修复的定点检查及独立P0/P1审查PASS。完整40000步/800轮日志已核实；所有源checkpoint在新预测前再次核对完整物理角色、预算、scratch和payload。

只测试源选中的4seed，无增强、仅身份骨干、clean-only，目标分数不反馈结构/超参数/选模/重排/重跑。公共相位约束不等同于任意RX解耦或TX参数恢复，尚无本轮clean性能结论。

[完整源报告](../20261002-phase1-cvs-equivariant-fp32-manysig-m4-r01/report.md) · [冻结源选择](evidence/performance_selection.json) · [核实](evidence/local_validation.json)。
