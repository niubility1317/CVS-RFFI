# DURA 星地性能与训练效率优化

本页为检索入口，实时状态以各run产物/进程为准，目标仍ACTIVE。

|实验|已核实证据|入口|
|---|---|---|
|源逆问题r01|旧Torch/NumPy2 C桥首次smoke故障，FAILED，保留产物|[报告](../../automation_reports/CV-SincNet/20261030-diagnostic-DURA-source-inverse-s2026092701-r01/report.md)|
|源逆问题r02|512位置×两epoch，0TX/F/V/target和模型更新；16CG未改善|[报告](../../automation_reports/CV-SincNet/20261030-diagnostic-DURA-source-inverse-s2026092701-r02/report.md)|
|先验监督/常规续训源训练|两行E20完成；原评估模式故障，固定权重保留并由r02补齐|[报告](../../automation_reports/CV-SincNet/20261030-phase1-DURA-prior-anchor-s2026092701-m2-r01/report.md)|
|先验监督/常规续训零训练恢复|两行各168000×7完成，252分组独立复算；lowurban67.86%/67.48%，未优于默认|[报告](../../automation_reports/CV-SincNet/20261030-phase1-DURA-prior-anchor-s2026092701-m2-r02/report.md)|
|单阶段成本诊断|CPU2 B256在线教师1.535s/随机联合1.186s；仅单步CPU成本|[报告](../../automation_reports/CV-SincNet/20261030-diagnostic-DURA-single-stage-cost-s2026092701-r01/report.md)|
|从零单阶段E20|799.978s、0祖先/教师；七景完成，clean85.40%/lowurban56.04%，质量未保持|[报告](../../automation_reports/CV-SincNet/20261030-phase1-DURA-direct-joint-s2026092701-r01/report.md)|
|从零匹配学习率E20|已按source完整曲线固定下一试验，仅峰值LR2e-4→1e-3；发布状态见原记录|[报告](../../automation_reports/CV-SincNet/20261030-phase1-DURA-direct-fastlr-s2026092701-r01/report.md)|

目标与验收见[当前目标](../../docs/goals/PAMF_DURA_PERFORMANCE_20261030.md)。默认DURA v1.0保持；六星地逐景3pp与clean保持尚未达到。
