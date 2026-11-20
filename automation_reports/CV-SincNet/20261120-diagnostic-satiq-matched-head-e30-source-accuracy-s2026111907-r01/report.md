# 同报头 PAMF 第 30 轮全量训练集和验证集准确率

[配置](experiment.json) · [事件](events.jsonl) · [固定第 20 轮同口径结果](../20261120-diagnostic-satiq-matched-head-e20-source-accuracy-s2026111907-r01/report.md)

对固定 846,750 步的正式同报头 PAMF-SUP 做一次全量源 L/V 评分。每包仍直接面对 3,103 个波束。相同分母是 907,716 条已知 L 包和 49,814 条已知 V 包。对照第 20 轮的训练和验证 Top-1、主 CE、宏平均以及输出集中度，比较训练拟合和验证性能是否同步变化。

沿用第 20 轮的固定评分程序和原训练发布。仅使用这一正式源契约训练的第 30 轮权重，没有更新任何模型状态。独立 T 留出。本次不占用 GPU。

- 原始全量结果 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-matched-head-e30-source-accuracy-s2026111907-r01/pamf_source_e30.json`
- 评分日志 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-matched-head-e30-source-accuracy-s2026111907-r01/score.log`

## 评分进度

训练日志固定第 30 轮验证命中 715/49,814（1.4353%），主 CE 为 6.7478。第 20 轮在同批 V 上命中 675 条，增加 0.0803 个百分点。

**VERIFIED 已启动。** 2026-11-20T12:37:01+08:00 独立读回 N607 CPU 评分 PID 2546266、调度 PID 2546168、命令和工作目录。第 30 轮真实权重的 32 条源 L 检查退出 0。全量 L/V 同口径准确率从正式评分产物读回。
