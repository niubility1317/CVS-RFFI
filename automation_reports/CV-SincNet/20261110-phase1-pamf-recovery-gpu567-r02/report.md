# PAMF 恢复队列第二次发布

前次 r01 的无目标访问 smoke 已通过，旧结构 190 模型及 36 个分支对照已完成独立汇总。新结构 E200 训练已完成的两个模型在补充资源统计前，被恢复代码错误地按 JSON 完全相等检查源合同；训练器实际保留原物理 ID、源角色和数据合同，并额外写入实际数据路径、类别及表示字段。六个 checkpoint 的保存合同、训练种子、200 轮/10,000 步、初始化和最终权重均已独立核对。

r02 在新 run `20261110-phase1-pamf-recovery-gpu567-r02` 延续 r01：校验原合同每项内容一致，并校验派生数据路径、等化、归一化、长度和类别数；checkpoint 内合同和训练初始化仍须与保存文件一致。此前已完成和 r01 收尾完成的阶段按合法来源引用，其他阶段在 r02 重新配置输出及依赖；六个既有 E200 不重训。仅使用 GPU 5、6、7，原资源上限、无 query smoke、独立 truth-last、clean 和六 practical 场景、DURA 读出、stop rule 均沿用 [r01 预登记](../20261110-phase1-pamf-recovery-gpu567-r01/report.md)。

代码身份为本报告所在 Git 提交，发布命令仍为 `python -m experiments.pamf_split_spectrum_paper.publish_recover_567 --repo <workspace> --output E:/type10-7/local_artifacts/pamf_split_spectrum_paper_20261109`；仅在 r01 活动预测自然完成、调度器结束后启动。发布后核验实际 owner、GPU、日志、六个 checkpoint 恢复和评分依赖。
