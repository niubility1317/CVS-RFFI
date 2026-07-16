# qKNNv4.2正式Stage2-B/C极轻路线研发报告

此文件镜像`E:\type10-7\automation_reports\CV-SincNet\qknnv42_stage2bc_extreme_light_route_20260716\report.md`，用于Git版本管理。根目录报告是自动化交接面，二者必须同步更新。

当前状态：`SINGLE_OBSERVATION_PROTOCOL_REPAIR_ACTIVE`。

核心边界：Stage2-B域适应与Stage2-C真实seen-new注册同等优先；必须报告注册前与注册后同row结果。Phase2全链路只允许预先叠加`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`的密封输入，clean样本及任何clean派生信号物理不可达。query逐样本面对全部注册类，禁止角色Oracle、真实批次数/类别quota、query标签拟合、global assignment和dense query图。

包含K=1的旧v2 125任务曾完成125/125 job、8/8 shard PASS、0技术失败，且v1/v2的125个score逐job一致。但2026-07-17项目协议新增单物理样本单LEO接收观测硬约束：一个clean/raw物理样本进入Phase2前只能叠加一种LEO状态；K个support必须是K个独立物理样本；Phase2可对固定接收IQ做均衡、增强、变换或多表征适配，但不得生成另一LEO状态或增加K。

旧D1开发、D1-B0-Cap、D1 v1/v2 125和D3均使用同一物理样本跨三个LEO场景的平行观测，现统一标记`PROTOCOL_INVALID_FOR_PHASE2_SINGLE_OBSERVATION`。旧v1/v2逐receiver、逐类、loss和资源数据继续保留供查看，但不得用于候选选择、正式Pareto、K10/K5/K1门槛或部署声明。

当前最小修复是：cache builder先给独立物理样本分配唯一scenario并仅叠加一次；三场景support/query物理ID集合两两互斥；pre-open validator由跨场景ID一致改为跨场景ID无交集；删除跨LEO场景同物理ID的support augmentation，改为固定接收IQ上的接收后均衡/增强/变换view。资源硬上限同步为80,000参数、30epoch、256KB状态和无dense query图。

完整旧v1/v2明细、逐receiver/逐类表、资源审计、远端路径、文件SHA和本地回收路径以根目录报告第42–45节为准；最新协议降级与修复顺序见根目录报告第49节，新增追踪表为`analysis/phase2_single_observation_traceability_20260717.md`。
