# ADVB02 NTRS-V4实现与实验计划

1.为v4 context、条件低秩算子、反事实损失和launcher profile编写失败测试。
2.实现`NTRSNormalizedMetadataContext`与`NTRSConditionalLowRankOperator`，保持成熟raw路径detach和共享头冻结。
3.扩展loss bundle，加入`q_distill`、`pair_shift`、`pair_cosine`、`harm`、`rescue`和`clean_tail`。
4.实现B0 paired-shift/PCA/连续oracle诊断脚本，输出JSON与CSV。
5.扩展模型checkpoint重建、训练参数、遥测和launcher，发布B0-C/B0-S/B0-R/B1-M/B1-N/B2-A/B2-O/B3-R。
6.运行聚焦单元测试、协议负测和成熟checkpoint无query smoke。
7.完成一次独立P0/P1正确性审查；仅修复会直接导致真实实验跑错、越权、覆盖输出或无法闭合prediction的问题。
8.显式stage、commit、push并核对远端OID。
9.完成N607只读preflight、单归档SHA传输校验、远端编译和不可覆盖run发布。
10.逐row保存E200 checkpoint、clean及三种LEO_WEAK测试，形成同行数据报告并再次提交推送。
