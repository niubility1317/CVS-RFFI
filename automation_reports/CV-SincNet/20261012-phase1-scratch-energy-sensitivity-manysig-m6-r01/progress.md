# 完整六行结果

状态 VERIFIED / ANALYZED。6/6 行训练与 clean 独立评分完成，两方案各三种子；26,640 次源更新、120 个 epoch、1,008,000 次 clean 决策。逐行与汇总评分一致，完整源日志无错误。

预登记主方案 CE_ENERGY：79.9806%；固定 E20 参考 P0：80.0504%。MARGIN_ENERGY 均值 80.1010%，相对 P0 +0.0506 个百分点；CE_ENERGY 相对 P0 -0.0698 个百分点。所有完整方案均未达到 84%。不据此改变已固定的后续源驱动设计或选择性重跑。

完整每种子、配对差、RX/TX/day、Macro-F1、NLL、成本和来源边界见 [final_analysis.md](final_analysis.md)。逐步机制核验见 [source_trajectory.md](source_trajectory.md)，两项固定跨批对照见 [reference_comparisons.md](reference_comparisons.md)。
