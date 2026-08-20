# ADVB02 NTRS-V4反事实风险优化实验矩阵预登记

正式run ID为`phase1_advb02_ntrs_v4_counterfactual_matrix_20260821_r1`。本轮在Phase1 source-only边界内，以seed=`392034`和`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`运行；训练和独立测试均只使用`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，不使用历史`mixed_orbit`。

矩阵包括`b0_constant`、`b0_shuffled`、`b0_random_feature`、`b1_metadata`、`b1_normalized`、`b2_additive`、`b2_operator`和`b3_risk`。所有行从成熟D1 checkpoint初始化并冻结raw路径；除B2-A读取B0产物并固定PCA公共shift外，其余训练IQ上下文和有界低秩算子。完成训练后必须对E200最终checkpoint执行clean和三个LEO_WEAK场景的全量独立测试，并报告同checkpoint raw→fused差分、Strict UDU与rescue/harm。

不训练的B0-PCA诊断只读取source clean/LEO配对导出，报告rank4/8/16/32、全shift oracle、连续gate oracle及场景、TX、TX×场景方差分解。性能低不触发技术停止。
