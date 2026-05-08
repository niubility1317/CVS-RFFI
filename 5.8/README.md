# 5.8 实验归档

本文件夹保存由 `run_final_best_sgc_queue.sh` 启动的 seed=1337 后续实验组。

## 内容

- `logs/`：原始逐实验训练日志，以及总启动器 nohup 日志。
- `metrics/experiment_metrics.csv`：解析后的扁平指标表，适合用表格软件对比。
- `metrics/experiment_metrics.json`：完整解析指标，适合后续脚本复用。
- `reports/5_8_training_analysis_20260508.md`：5.8 训练日志分析报告和下一步路线建议。

## 解读提醒

日志中包含基于 test 指标的 checkpoint 选择，例如 `best_primary_ood_score`、`best_test_overall`、`best_strict_udu`，Phase-E 的 source 选择也读取了 test-derived primary 指标。因此这些结果适合作为开发集分析证据；如果要写论文或做最终汇报，需要使用 validation-only 选择流程重新跑 shortlist，或在全新 final holdout 上只评估一次。
