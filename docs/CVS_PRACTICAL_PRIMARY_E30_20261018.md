# 固定 CVS：practical 优先研究


## P67：物理样本跨轮确认，匹配 NoAs／As E30

20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01 新增两条 own20＋10 尾段，固定 F＋CROSS_RX／默认头／全5源RX和 L6300／hidden U56700／V27000。唯一变化是 native batch_neighbor→epoch_bank，min_streak2／min_conf.8 与原 confidence/domain/strong 规则保持，缓存 E21 重置。教师原本直接读取 raw U，不存在旁路弱教师增强的变化。无新 IQ 增强、损失、教师或前向，也不把该时间规则称为 A_g。旧 epoch_bank 的其他架构 E200／无As历史保持。

源晋级要求 AS_BANK fullV clean 分别对 matched NOAS_BANK、原 NOAS30 均≥−.1pp；midlow 对原 AS30≥＋.5pp且low≥0。任一FAIL关闭固定方式，不能调窗口/阈值/seed或按formal扩预算；源合格后才可总计E50。科学失败仍逐row完成clean＋六固定practical独立truth-last。四条件已完成全部七视图评分与分析，完整日志、配对误差及分层结果已保留。源四判据均 FAIL，不扩 E50。正式单 seed AS_BANK 对匹配 NOAS_BANK：clean ＋0.041071pp、mid urban ＋27.739286pp、low urban ＋21.054762pp、urban 均值 ＋28.405159pp，满足正式 clean 下降≤0.1pp 的阶段要求。新增时间机制对原 AS30：clean −0.086310pp、mid ＋0.023214pp、low −0.017262pp、urban 均值 ＋0.012103pp；不归因既有 As 收益，不宣称可重复双提升，整体目标 ACTIVE。

预登记：[P67 report](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01/report.md)。


用户允许表现好的方案延长到总计 E50；本轮按预登记合法源域条件判断，不以正式测试分数晋级。P67 未通过，未启动 E50，也未扩种子或扫时间窗口。后续候选先与另一对话去重。

最终分析：[P67 final](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01/final_analysis.md)。
