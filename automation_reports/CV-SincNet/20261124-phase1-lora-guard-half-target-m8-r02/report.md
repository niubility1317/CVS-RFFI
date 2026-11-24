# LoRa 源域最佳 guard_half 的三种子目标测试

[预登记](experiment.json) · [运行事件](events.jsonl) · [选定候选及原无目标启动失败](../20261124-phase1-lora-guard-half-target-m8-r01/report.md)

修复无目标检查的历史包导入顺序。恢复使用原秩四模型的冻结终步及 `guard_half` 原规则。两路线与冻结 Tri-set 三种子共用恢复权重、检测门限和固定 0.5 剂量。预先登记的 15 台未见接收机、30,000 个同序物理包和七景覆盖不变。完整固定预测后使用独立本机 truth-last，并复核与既有直通、DURA 半量和 Full 复响应半量的配对结果。
