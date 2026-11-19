# SatIQ 同报头 PAMF 512 点内部时间网格对照

[配置](experiment.json) · [事件](events.jsonl) · [同步模板已对齐的 256 点对照](../20261120-diagnostic-satiq-full-header-uw-time-source-e1-s2026092701-r01/report.md)

## 改动和验证

正式同报头实验给两种方法同一条 25 MS/s 的 11,000 点 IQ 信号。PAMF 在前处理后将信号用 256 点网格送入原有五条分支。这次将内部网格提高到 512 点，覆盖时长仍是 440 µs，保留 187.5 至 250 kHz 的低通过渡带。傅里叶零填充在同一带限信号上增加时间采样。每隔一点取样可复现 256 点输入，误差在浮点舍入量级。

将同步模板、参考分支的时延坐标和 Sinc 分支的采样率参数按新网格换算。固定顺序的序列池化前合并相邻时间单元，保留原有位置排列、五分支顺序和参数尺寸。五个分支仍各自输出 160 维，融合仍为 256 维，主头和分支头的候选均为 3,103 个波束。真实源首批 32 条包已验证两组同种子初始化的可训练参数逐元素相等，确认两个网格的输入嵌套关系以及前向结果有限。训练批次、物理包、损失、标签、学习率和评估样本共用，以区分增加内部网格采样的影响。

## 比较和状态

两组从零训练，固定比较第 5,000、20,000 和 28,225 步。两组完整首轮均选取相同的 903,200 个物理训练包，源 V 使用同一批 49,814 个标签合格的包。主指标是每条信号面对全部 3,103 个波束的 Top-1，附主分类交叉熵。独立 T 保持留出。

**VERIFIED 已启动。** 2026-11-20T03:24:27+08:00 从 N607 读回真实全流程两步短检查退出码 0、GPU5 上训练 PID 2275013、监督进程 PID 2274790，以及第 2,400 步训练记录。完整首轮和源验证待出。

- 原始日志 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-full-header-grid512-source-e1-s2026092701-r01/train.log`
- 真实执行配置 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-full-header-grid512-source-e1-s2026092701-r01/PAMF_SUP/resolved_config.json`
- 完整首轮预期产物 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-full-header-grid512-source-e1-s2026092701-r01/PAMF_SUP/s0028225.pt` 与 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-full-header-grid512-source-e1-s2026092701-r01/PAMF_SUP/complete.json`
