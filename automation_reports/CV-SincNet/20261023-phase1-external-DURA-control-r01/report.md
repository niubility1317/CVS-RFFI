# 四方法接DURA：冻结读出及健康队列接管

发布状态：**VERIFIED**；本报告对应阶段为 **RUNNING**。准确率与配对性能尚未完成，不以发布或源侧检查代替结果。

覆盖四方法已冻结516模型，每模型raw、DURA-L、DURA-LU；原始预测复用，34个同契约共享恢复任务、516预测、516独立评分、1汇总。没有新增F/G训练或五源数量扫描。

实际执行提交 `117fbaecc1b2eda8080fb6503a16ce4006cc45ca`。新owner PID1336081，真实checkpoint源侧检查PASS；接管8条当时健康worker，7条同PID日志跨124秒增长，随后8卡各3条。原owner正常退休。

两个旧core源诊断类型失败保留，分别映射到新recovery-m2的预测/评分，原核心汇总由core-factorial-recovery接续；只迁移未启动的同缺陷执行消费者。原统一计时已完成r02替代保持，不重测。

源训练尚余104个F，估算约144共享worker小时；按14至18个F等效槽约8至10.3小时。新增读出、5条完整G依赖、独占时延、CPU分析、论文和导出后，**整体暂估18至30小时**。这是新增范围后的初估；首批缓存和分类实际吞吐回来后校准，不能按24并发线性缩时，也不逐小时重置倒计时。

详细设置见 [固定方案](../../../docs/EXTERNAL_DURA_FROZEN_COMPARISON_20261023.md)，行级权限、原生预算引用和输出见 experiment.json，独立发布证据见 launch_verification.json。监督F+G-L与监督F+G-LU分列。所有七景固定后truth-last，保留负收益、实际三种子、每RX/TX/day与净纠错。


2026-10-23接管更新：本资源controller已正常退休，当前owner为20261023-phase1-PAMF-PL-Gfactor-control-r02，PID1558142。原外部516模型读出/34缓存/分析任务全部原配置保留继续；不是科学范围完成或调度故障，禁止重启旧owner。
