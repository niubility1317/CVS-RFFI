# E50 原 practical 匹配预算扩展

固定 B00/B01、seed 2026092701，从各自 P43 FINAL30 student 和 EMA 追加 20 轮，总轨迹 20＋10＋20。固定 FINAL50；不按源域或正式分数改变轮数、参数或种子。默认头和完整 practical 的 CE、KL、dropout 保持。

本地检查：python -m experiments.cvs_practical_budget50.checks。

发布：python -m experiments.cvs_practical_budget50.publish。主代理在 Git 提交登记、一次独立 P0/P1 审查后执行。私有运行配置按 runtime.example.json 的键写到 local_artifacts/F_As_budget_E50_20261018/runtime.json，不能提交该私有文件。远端发布根 runtime.json 仅含 run_id/project/reference_release/reference_commit，由发布器写入；它固定引用 P43 commit，不复制改写旧发布。

远端入口：python -u experiments/cvs_practical_budget50/runner.py。第一步自动真实 FINAL30 无 query smoke，通过后连续分派训练。内部 segment=30 只用于现有 backend 的 parent-load 语义；实际新段 epochs=20、offset=30/6660、累计 50/11100。EMA 新段计数 0 至 4440，累计加 6660；恢复的是各自真实 EMA 权重。

每 row 自动 source 读出、clean＋六 practical 预测全部固定、独立 CPU truth-last 评分。source_matrix_decision 仅做固定预算报告，无 source 科学失败跳过正式评分，无自动扩种。automatic_summary 状态保留为 SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS；独立全日志及逐包错误分析由主代理接续，不能以训练或评分完成冒充 ANALYZED。
