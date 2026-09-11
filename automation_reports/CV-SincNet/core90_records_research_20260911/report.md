# CORE90实验记录扩展检索与性能差异分析

本目录是只读检索报告，不是新训练run。

已核对6次CORE90名义基线E200训练的1200条epoch记录、对应完整stdout、checkpoint配置和冻结release代码。新增两次B0训练、D0/P0源LORO对照、RIEI/DRIFT完整批次，并补录ECRS R0与未完成fold记录。

完整结果、26条分类记录及新增评分附件见[扩展检索报告](E:/type10-7/docs/CORE90_REPRODUCTION_INDEX_20260911/expanded_report.md)。首轮3份目标基线与测试记录目录见[原始汇总](E:/type10-7/docs/CORE90_REPRODUCTION_INDEX_20260911/report.md)。

主要结论：统一168000条目标范围后，clean跨度1.8625pp、LEO均值跨度2.7264pp。测试总体构成、有效几何损失、原型梯度语义、AMP、U采样及LEO随机输入均存在实际差异；RIEI/DRIFT还使用不同训练日期。差异因素已确认，尚不能逐项归因其准确率贡献。FCR与同权重C0复测四场景正确数完全一致。

H0在本次取证时已进入目标预测阶段，完整四场景独立评分仍未闭合。未启动新训练或推理，未改动任何远端任务。
