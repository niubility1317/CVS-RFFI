# V2全部21模型目标测试完成

状态：TARGET_ARTIFACTS_COMPLETE / VERIFIED。11模型补测于2026-09-13 01:09:02 +08:00完成，预测及评分2286.748秒（38.11分钟），进程3258296已退出。原单卡partial不计入结果。

合并此前target10及七卡target11，模型清单互斥且与21行矩阵一致，共14112000条预测。此前10模型全量核验已完成，本次对新增7392000条预测逐条重算总体混淆矩阵、Accuracy、Macro-F1、Macro-Recall及最弱TX，最大误差0；RX/day计数与加权准确率闭合。两批各自完成全部prediction后才评分，不声称21模型跨批次同时封存。跨卡同型号、同冻结输入及增强规则，不将不同批次记为逐位预测一致性证明。

单模型最高：STRONG_SOURCE_LR_HIGH_seed392005，clean79.725595%、LEO67.599206%、LEO Macro-F1 67.095749%。相对同seed BASE的LEO增加3.848214个百分点，但只有单seed。

三seed：F clean76.851±1.204%、LEO63.745±2.182%；E clean74.700±4.228%、LEO63.378±2.291%。因此此前仅依据source数据偏好E的判断需修正：目标平均指标F更高，E392007 clean仅70.055%。F−E的LEO均值+0.367pp并不证明显著优势。E−A平均+3.232pp、F−B平均+2.579pp，均为三个seed正提升；D−B平均−2.748pp，三个seed全负。

详细报告：target_report.md；target_summary.csv为21行总体成绩；target_detailed_metrics.csv含7056行RX/day/TX及组内TX指标；target_confusion_matrices.csv含3024格；target_seed_aggregates.csv为跨seed统计；target_all_paired_metric_deltas.csv为18组同seed指标差值。target_paired_predictions.csv仅保留此前已核验的逐样本配对子集，不能视为全部18组的逐样本核验。

数据为168000物理IQ的四种视图，每模型672000决策；6类闭集，无support适配。previously_exposed_benchmark_recheck，不以测试反馈调参、重训或晋级，不作新盲测结论。
