# 固定 CVS：practical 优先研究


## P67：物理样本跨轮确认，匹配 NoAs／As E30

20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01 新增两条 own20＋10 尾段，固定 F＋CROSS_RX／默认头／全5源RX和 L6300／hidden U56700／V27000。唯一变化是 native batch_neighbor→epoch_bank，min_streak2／min_conf.8 与原 confidence/domain/strong 规则保持，缓存 E21 重置。教师原本直接读取 raw U，不存在旁路弱教师增强的变化。无新 IQ 增强、损失、教师或前向，也不把该时间规则称为 A_g。旧 epoch_bank 的其他架构 E200／无As历史保持。

源晋级要求 AS_BANK fullV clean 分别对 matched NOAS_BANK、原 NOAS30 均≥−.1pp；midlow 对原 AS30≥＋.5pp且low≥0。任一FAIL关闭固定方式，不能调窗口/阈值/seed或按formal扩预算；源合格后才可总计E50。科学失败仍逐row完成clean＋六固定practical独立truth-last。四条件报告交互项、全源分层、默认头收益、clean代价与实际成本；新结果尚未产生，整体目标ACTIVE。

预登记：[P67 report](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01/report.md)。
