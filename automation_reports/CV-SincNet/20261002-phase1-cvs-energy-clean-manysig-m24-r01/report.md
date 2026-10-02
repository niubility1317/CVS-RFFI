# CVS 相对滤波能量保留：选中候选 clean 确认

状态 LOCAL_VERIFIED，尚未访问本轮新 query。固定源规则依据 8 个新 scratch E200 与 4 原残差源控制选择 `energy_equivariant`。4 个新冻结模型＋20份原控制预测，共24行、168000原物理query/6TX/7RX，全部固定后独立truth-last评分。

训练与预测同一固定FP32策略，关闭cuDNN TF32，其他实际flags必须符合源payload；旧控制保留历史精度，只读复用。98项相关检查及独立P0/P1审查PASS。完整80000步/1600轮日志已核实；所有源checkpoint在新预测前再次核对完整物理角色、预算、scratch和payload。

只测试源选中的4seed，无增强、仅身份骨干、clean-only，目标分数不反馈结构/超参数/选模/重排/重跑。公共常相位性质及相对滤波能量保留不等同于整体仿射相位不变、任意RX解耦或TX参数恢复，尚无本轮clean性能结论。

[完整源报告](../20261002-phase1-cvs-energy-identity-manysig-m8-r01/report.md) · [冻结源选择](evidence/performance_selection.json) · [核实](evidence/local_validation.json)。
