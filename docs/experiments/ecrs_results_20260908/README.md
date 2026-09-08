# 数据附件与复算入口

本目录属于ECRS_FULL_RESULTS_20260908.md报告。训练曲线截至2026-09-08约09:05的完整下载快照；live_readback.json记录09:52的独立进程与产物状态。目标测试全部完成。

- analysis.json：65个原始文件解析清单、训练进度、时间与实际激活计数。
- epochs.csv、source_validation.csv、stdout_losses.csv、telemetry_epochs.csv：全部已记录epoch及分量，不是尾部样本。
- source_final_scores.json：完成组源域四场景、全部路径和分组。
- source_paired_vs_B0.json：同ID的raw跨模型描述性差异；不是融合收益。
- checkpoint_audit.json：14个best/latest的配置、split、scaler和optimizer步骤元数据。
- target_test_summary.json：独立scorer完整目标评分、混淆矩阵、分组和推理资源。
- target_groups.csv：上述目标分组的可筛选表格；各分组总数和正确数已与整体、混淆矩阵互相核对。

在原仓库布局内复算源域下载快照：

```text
python code/scripts/analyze_ecrs_snapshot_20260908.py <snapshot目录> docs/experiments/ecrs_results_20260908 --checkpoint-summary <checkpoint_summary.json>
python code/scripts/render_ecrs_report_20260908.py
```

原始完整源日志与预测：本机E:/type10-7/local_artifacts/ecrs_analysis_20260908_0904/complete_logs.zip。解压后snapshot内容对应runs/与logs/。checkpoint_summary.json位于同一本机目录。

ZIP交付包为报告和分析附件；附带脚本是仓库入口，需在原仓库及相应依赖环境中使用，并非独立模型发布。数据集、.pth权重及大体积目标逐样本logits未打入Git/报告ZIP，N607原始run保持完整。目标结果已冻结，不授权根据本次目标分数调参、换head或选择性重跑。
