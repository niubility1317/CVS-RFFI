# SatIQ 原型监督 320 的 E5 与 E10 分类头全量诊断

[配置](experiment.json) · [事件](events.jsonl) · [同期四组 K16 配对](../20261120-diagnostic-satiq-native-paired-satellite-e5-e10-s2026111907-r01/report.md)

冻结原型监督 320 的第 5 轮和第 10 轮，以与原版 160、融合 256 相同的分类程序读取全部源 L/V。报告 3,103 类封闭集微 Top-1、宏 Top-1、CE、预测过的类别和最高频输出占比。独立测试留出。

N607 控制器 PID 2187677 已读回，注册的原型监督 320 仍在完成第 10 轮。检查点出现后按固定路径在物理 GPU2 启动分类头评估。

- 控制器日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-proto320-head-e5-e10-s2026111907-r01/controller.log`
- 结果路径：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-proto320-head-e5-e10-s2026111907-r01/source_scores.json`
