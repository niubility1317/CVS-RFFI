# 2026-10-24 对比实验整理与隔离

VERIFIED：旧外部训练增强不匹配当前 practical residual 要求；新论文统一采用 `20261023-phase1-practical-residual-full2x2-m600-r01`。全部 627 行、11 契约、三类泛化及四组合保留。

新论文包：`exports/unified_figures/20261024-practical-residual-paper-r03`。22 张 Times New Roman 独立图，PDF/SVG 可编辑矢量、4800 dpi PNG/TIFF；25 页中文逐图说明；6 页 IEEE 实验部分。9 张有效 PAMF 历史机制/结构/时延图保留原口径。

原目录已回收 2046 个文件，逻辑大小 13980418082 字节（约 13.98 GB，不等于实测磁盘释放量）。归档位于 `F:\AI\exports\STRA-RFFI\quarantine\20261024-obsolete-comparisons-r01\obsolete_local_comparisons.zip`，恢复映射 `F:\AI\exports\STRA-RFFI\quarantine\20261024-obsolete-comparisons-r01\restore_map.json.gz`。删除前完整读回 2046 个成员并核验 CRC，删除后逐目标确认原路径不存在。

旧混合图包同时包含正确的 PAMF 证据，因此完整归档，不能整包声称科学无效。保留模型、IQ、所有 Git 工作树、远端数据/进程、实验登记及恢复事件。旧外部注册记录增加 usage_notice.json，禁止作为当前比较；并未把历史 ANALYZED 改成虚假失败或删除必要追溯信息。新结果 r14 原包保留；其历史附件已有明确 historical_evidence 目录。

当前论文、当前方法和冻结输入的逐路径引用检查为 0；本地相关运行进程引用为 0。只对检查范围内的本地导出和旧外部聚合副本操作，没有执行远端删除。

恢复：解压恢复映射，按原相对路径从归档提取到新目录；核对成员大小和 CRC 后再恢复所需路径，禁止覆盖现有新成果。详见 cleanup_manifest.json 和隔离目录内说明。
