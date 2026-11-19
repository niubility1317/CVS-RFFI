# SatIQ 和原生 PAMF-SUP 第 5 至第 10 轮配对验证

[配置](experiment.json) · [事件](events.jsonl) · [第 5 轮报告](../20261119-diagnostic-satiq-native-paired-satellite-e5-s2026111907-r01/report.md)

把 SatIQ、原版 160、融合 256 和原型监督 320 放在完全相同的注册、查询包、日期切分和检查点步数上比较。固定 E5 为 141,125 步，E10 为 282,250 步。两组以相同的 8,192 个源训练包分别拟合每模型的部分白化，并使用相同的 1,000 次按卫星重采样。这样可报告方法间差值和同一个方法随训练变化的差值区间。随机与跨日同星 K16 均同时报告原始、部分白化 AUC 和 EER。独立测试留出。

N607 控制器 PID 2183561 已读回，当前等待 SatIQ 和原型监督 320 的第 10 轮检查点，其余两组已齐。所有注册的第 10 轮模型可读后，控制器检查 GPU4 资源并启动评分。正式训练进程保持运行。

- 控制器日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-native-paired-satellite-e5-e10-s2026111907-r01/controller.log`
- 完整输出将写入：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-native-paired-satellite-e5-e10-s2026111907-r01/source_scores.json`
