# SF-TAPFT S15+优化落地计划

## 目标

严格按用户报告，将S15的300步快速工作点改造成短程校准路线，并发布最小实验矩阵验证；保持`p2_min_v1`、K=10×6、support-only选择和query truth-last边界。

## 阶段

1. `completed`：审计现有runner、配置、bundle、测试和N607资源状态。
2. `completed`：按TDD实现短程schedule、OOF温度、head预训练、稀疏validation、向量化LOO和训练期delta snapshot；部署bundle瘦身保持后续独立项。
3. `completed`：冻结F1–F3与S16-A/B首发矩阵；F0复用历史S15，F4/Q1/Q3按前置证据保持deferred。
4. `completed`：聚焦测试、编译/CLI回读、一次独立P0/P1审查及定点复核；真实checkpoint no-query smoke作为远端launcher第一步。
5. `completed`：提交并推送，建立最小预登记，执行N607 preflight、发布、远端编译、真实checkpoint smoke和启动回读。
6. `completed`：5/5行selection、clean-single bundle、GNU time与support OOF分析闭合；Q2A/Q2B通过门槛，F1–F3淘汰。

## 错误记录

|错误|定位|处理|
|---|---|---|
|过宽文本检索产生截断输出|本地代码定位|改为函数/行段窄检索，无状态影响|
|delta snapshot首次未恢复forward漂移buffer|聚焦回归测试|以完整不可变anchor重建，再叠加许可delta；回归转绿|
|全仓测试无法收集|Python3.10缺`tomllib`；`tests/`与`code/tests/`同名模块|记录为既有非阻断环境问题；本次76项聚焦测试、编译和CLI回读通过|

## 不扩大事项

- 不重跑已完成S00–S15。
- 不复用rank10–19 query truth选择新候选。
- 不把Q3提前到Q1独立query通过之前。
- 不在同一首轮候选同时改变schedule、norm scope、蒸馏和head anchor。
