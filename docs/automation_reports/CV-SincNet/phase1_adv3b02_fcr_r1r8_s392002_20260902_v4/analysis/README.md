# v4独立分析数据说明

- `row_summary.csv`：8个实验row的同row准确率、数值稳定性、诊断与资源汇总。
- `scenario_results.csv`：8个row×4个场景共32行，包含总体准确率、receiver floor、样本数及预测直方图。
- `per_class_results.csv`：8个row×4个场景×6个TX共192行，每行包含该类4,200个样本的正确数与准确率。
- `analysis.json`：完整结构化结果，包括混淆矩阵、预测直方图、全日志曲线摘要、FCR诊断与artifact完整性检查。
- `analyze_fcr_v4.py`：独立分析器，不导入训练代码；输入为本地下载的原始日志、diagnostics和逐样本prediction。

本次prediction的样本ID可逆包含TX标签，因此结果属于独立进程重算，但不能声明严格不透明ID的truth-last隔离。原始逐样本prediction约247MB，保存在本地artifact根和N607不可覆盖run root，不提交到Git。
