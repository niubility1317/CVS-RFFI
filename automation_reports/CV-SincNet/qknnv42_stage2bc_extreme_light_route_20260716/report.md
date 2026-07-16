# qKNNv4.2正式Stage2-B/C极轻路线研发报告

此文件镜像`E:\type10-7\automation_reports\CV-SincNet\qknnv42_stage2bc_extreme_light_route_20260716\report.md`，用于Git版本管理。根目录报告是自动化交接面，二者必须同步更新。

当前状态：`STRICT_V2_125_COMPLETE_PERFORMANCE_FAIL_ROUTE_EXPLORATION_ACTIVE`。

核心边界：Stage2-B域适应与Stage2-C真实seen-new注册同等优先；必须报告注册前与注册后同row结果。Phase2全链路只允许预先叠加`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`的密封输入，clean样本及任何clean派生信号物理不可达。query逐样本面对全部注册类，禁止角色Oracle、真实批次数/类别quota、query标签拟合、global assignment和dense query图。

包含K=1的严格v2 125任务已完成：125/125 job、8/8 shard PASS、0技术失败；75/75个K1/K10 support-query嵌套审计通过`pipeline→COMMIT→execution receipt→opened support/query SHA`闭环。v1/v2的125个score逐job精确一致。严格汇总状态为`INCOMPLETE_DIRECT_BASELINE_PERFORMANCE_FAIL`：K10/new20 seen-new为86.29%，但注册后old、逐旧类floor、K5下降和K1旧类增益均失败，direct ADV3B02 K1仍为`MISSING_NOT_RUN`。代码修复提交为`eaabeed`、`7c97720`，验证记录提交为`446b16d`，组合回归28项PASS。

完整明细、逐receiver/逐类表、资源审计、远端路径、文件SHA和本地回收路径以根目录报告第42–45节为准；本目录`traceability.md`保存逐项Git追踪。
