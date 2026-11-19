# Sinc 频带适配的首轮源验证

[配置](experiment.json) · [事件](events.jsonl) · [固定包先验诊断](../20261120-diagnostic-satiq-native-sinc-fit-s2026092701-r01/report.md) · [原版 160 训练](../20261119-phase1-satiq-ccs2023-native160-sup100-s2026092701-r01/report.md)

只更改原有 Sinc 的通带初始化。24 个 79 抽头 FIR 核的名义频带均为 20 kHz 宽，下截止频率相距 8 kHz。保留 8,000 点同步字段至 256 点的处理、原五分支及 160 维、完整监督损失、源 L/V 和原版种子。每轮 28,225 个批次、903,200 个独立训练包。记录 5,000、20,000 和 28,225 步的源验证，与原版 160 的相同步数逐项对照。独立 T 留出。

N607 GPU0，PID 2166873。2026-11-20T00:42:40+08:00 读回工作目录、命令、实际配置及第 200 步日志，**VERIFIED 已开始训练**。首轮源验证待出。

- 生效配置及进度：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-native-sinc-source-e1-s2026092701-r01/PAMF_SUP/resolved_config.json`、`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-native-sinc-source-e1-s2026092701-r01/PAMF_SUP/steps.jsonl`
- 运行日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-native-sinc-source-e1-s2026092701-r01/train.log`
