# RX1实验与历史证据

[返回总索引](../README.md)

路径标签/旧目录字段仅用于查找；历史记录和备份不等于独立实验，状态未实时核实。

|名称|记录类型|证据入口|
|---|---|---|
|提供代码CORAL/MMMD/CNN/RIEI及PL六方法十种子|managed_run|[打开](../../automation_reports/CV-SincNet/baselines_rx1_s10_e100_20260921_r01/report.md)|
|旧PairSNN：RX1九种子八阈值|managed_run|[打开](../../automation_reports/CV-SincNet/pairsnn_rx1_s9_t8_e100_20260922_r01/report.md)|
|真实标签CE下限0.05＋梯度裁剪10：八种子六阈值|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_anchor05_pl9_rx1_s8_t6_e100_20260924_r01/report.md)|
|源校准严格伪标签：3seed × 6阈值|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_calibrated_plainpl_rx1_s3_t6_e100_20260924_r01/report.md)|
|冻结E5教师：探索性比较，不是同方法阈值敏感性|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_frozen5_pl9_rx1_s8_t5_e100_20260925_r01/report.md)|
|冻结门控自伪标签：统一 CE 下限 0.10|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_frozengate_anchor10_rx1_s8_t5_e100_20260925_r01/report.md)|
|冻结门控自伪标签：有标签CE下限0.20|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_frozengate_anchor20_rx1_s8_t5_e100_20260925_r01/report.md)|
|最终状态：ANALYZED，40/40完成，期望趋势未达到|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_frozengate_selfpl9_rx1_s8_t5_e100_20260925_r01/report.md)|
|类别支持联合置信度，统一 CE 下限 0.10|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_jointconf_anchor10_rx1_s8_t5_e100_20260926_r01/report.md)|
|Smooth E STAR＋普通置信度伪标签|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_plainpl_rx1_s299_t8_e100_20260923_r01/report.md)|
|Smooth E普通伪标签移除延迟渐增与SG投影：RX1单种子八阈值|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_plainpl_unprotected_rx1_s299_t8_e100_20260923_r01/report.md)|
|Smooth E unprotected RX1新增四种子六阈值|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_plainpl_unprotected_rx1_s4_t6_e100_20260923_r01/report.md)|
|无校准＋弱真实标签锚定：PL9压力实验|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_weakanchor_pl9_rx1_s3_t6_e100_20260924_r01/report.md)|
|弱真实标签PL9：新增八种子六阈值|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_weakanchor_pl9_rx1_s8_t6_e100_20260924_r01/report.md)|
|零有标签后期监督＋无梯度裁剪PL9压力实验|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_e_zeroanchor_noclip_pl9_rx1_s8_t6_e100_20260924_r01/report.md)|
|Comment3 RX1 w45单seed六阈值敏感性|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_comment3_s299_e100_20260921_r01/report.md)|
|Comment3 RX1 w45单seed六阈值敏感性|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_comment3_s299_e100_20260921_r02/report.md)|
|原Smooth E：原seed299加8种子的六阈值实验|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_comment3_s9_t6_e100_20260921_r01/report.md)|
|RX1伪标签噪声压力实验：取消渐增与监督增权|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_plstress_s299_e100_20260921_r01/report.md)|
|RX1 Smooth E SG增权与推理EMA精调9项|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_refine_s3_e100_20260921_r01/report.md)|
|Smooth STAR E RX1目标域三种子100轮|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_s3_e100_20260920_r01/report.md)|
|RX1严格阈值与提前伪标签实验|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_strict_early_s299_e100_20260921_r01/report.md)|
|RX1 Smooth E SG权重0.45十种子验证|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_w45_s10_e100_20260921_r01/report.md)|
|RX1 Smooth E SG伪标签权重两档三种子|managed_run|[打开](../../automation_reports/CV-SincNet/smooth_star_e_rx1_wsg_s3_e100_20260920_r01/report.md)|
