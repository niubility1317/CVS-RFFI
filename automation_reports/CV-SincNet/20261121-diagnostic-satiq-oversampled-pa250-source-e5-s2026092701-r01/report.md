# SatIQ PAMF-SUP 250 kHz 过采样非线性分支第 5 轮

[配置](experiment.json) · [事件](events.jsonl) · [PA 首轮](../20261120-diagnostic-satiq-oversampled-pa-bandwidth-source-e1-s2026092701-r01/report.md) · [原实现第 5 轮](../20261120-diagnostic-satiq-full-header-grid256-grid512-source-e5-s2026092701-r01/report.md)

匹配首轮 250 kHz 组比原实现多命中 19 条，主 CE 从 7.5832 降到 7.4981，远时段仍只命中 1 条。只用本组第 28,225 步权重继续到第 141,125 步，恢复 Adam 和随机状态。使用首轮原版发布，独立部署续训配置。原实现的 GRID256 第 5 轮已完成，命中 363/49,814、主 CE 7.0429。双方解析配置的源 L/V、固定标签、11,000 点输入、256 维身份表示与选模口径已核对。独立 T 留出。

GPU6 有正在运行的辅助头实验。本试验等待该调度退出后再由原 paper_extend 启动，会在进入训练前核对 GPU 余量。

**VERIFIED 已排队。** 2026-11-21T00:54:48+08:00 核对续训配置包、首轮原发布工作目录与 N607 调度 PID 3005505。当前等待本组的辅助头调度 PID 2989650，之后由原续训器检查 GPU6。独立 P0/P1 审查无阻断。
