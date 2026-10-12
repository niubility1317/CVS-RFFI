# 完整18行结果

状态 VERIFIED / ANALYZED。18/18行训练与clean独立评分完成，6方案各3种子；79,920次源更新、360个epoch、3,024,000次clean决策。全部逐行与汇总评分一致，完整源日志无错误。

预登记主方案DELAYCAP_BIDIR：79.1216%；固定E20参考P0：80.0504%。本批描述性最高均值CAP_POS：80.0202%，相对P0为-0.0302个百分点。所有完整方案均未达到84%。这不是从测试选择后续候选的依据。

完整每种子、配对差、RX/TX/day、Macro-F1、NLL、成本和来源边界见[final_analysis.md](final_analysis.md)。逐步机制核验见[source_trajectory.md](source_trajectory.md)，6项固定跨批对照见[reference_comparisons.md](reference_comparisons.md)。健康在训批次保持原配置。
