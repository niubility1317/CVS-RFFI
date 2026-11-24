# LoRa 扩容 DURA 正式测试 r02

[预登记](experiment.json) · [运行事件](events.jsonl) · [r01 技术失败](../20261124-phase1-lora-dura-capacity-target-m8-r01/report.md)

r01 在无目标检查中缺少历史扰动模块，留下零个预测。r02 在发布入口本地计算同一固定幅度扰动。依然使用同一 `expanded` 终点、源 U 门限和 `expanded_half` 推理规则。九项聚焦测试覆盖无历史训练依赖的确定性扰动、冻结校验、矩阵及配对统计。 [技术修复入口](../../../experiments/pamf_lora_main/weak_guard_target.py)

固定 15 台未见接收机、30,000 个同序物理包和 Tri-set 三种子。采用两条已登记路线，各含 clean 和六景。逐包对照直通、原 DURA、`guard_half`，Full 另对照冻结复响应半量。在 N607 先进行真实源 U 权重检查，分路线与接收机盲预测。全部 1,260 个数组闭合后，独立本机核对原直通、匹配基线和物理包顺序，再读取目标真值。 [盲预测](../../../experiments/pamf_lora_main/dura_capacity_target.py) · [独立评分](../../../tools/lora_weak_guard_target_score.py)

正式验收为 Full 六景至少 67.0%、低城区至少 53.0%、clean 至少 64,490/90,000，两条路线均相对原 DURA 提高六景并保持 clean。报告逐景、逐接收机和逐种子的纠错、改错、净变化和 95% 物理包配对区间，以及 Full 六景距本方案 clean 是否在 3 个百分点内。
