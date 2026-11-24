# WiSig 平滑实收前导时标冻结识别对照

[预登记](experiment.json) · [运行事件](events.jsonl)

固定 WiSig Main split Tri-SIG-TIME seed 2026092701 和原 R69，用 1,800 个已登记源 V 包验证 Full、Residual 各 clean 和六景。八个读出包括直通、局部过渡检测、固定三抽头和其反射边界对照、平滑复增益全窗、平滑复增益留出和其反射边界对照，以及原 R69。两条路线用同一套门限和推理规则。 [前导时标与接收信息](../../../docs/analysis/WISIG_HELD_PREAMBLE_20261124.md)

门限分别由同一 4,096 个原始源 U 最大分数加 2% 绝对值和 `1e-6` 固定，零网络训练，定标进程只读取源 U。前端使用当前窗前导与实收推得的估计频偏，实际 IQ 重叠区逐点保留。平滑或固定响应只用于确定时标及外推边界，反射对照保留相同时标以隔离边界处理。真实无查询检查在门限固定后执行，包含逐值 clean 旁路核对和固定人工移位的有效预测。全矩阵固定后，独立 CPU 读取源 V 标签，报告逐场景、逐接收机纠错、改错和配对区间。

当前状态为已登记。此处源 V 识别用于机制对照，新的恢复网络仍按正式目标测试验收。 [前端代码](../../../experiments/pamf_unrolled_restore/wisig_smooth_preamble.py) · [冻结评测入口](../../../experiments/pamf_unrolled_restore/wisig_smooth_preamble_source.py)
