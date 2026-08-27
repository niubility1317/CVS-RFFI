# SF-TAPFT H6 P0原位适配与轻量部署设计追踪

来源：用户提供的《SF-TAPFT从研究型4-fold选择流程转为full-support星上适配后的继续优化报告》，2026-08-28。用户已确认后续按推荐顺序执行。当前状态：`LOCAL_VERIFIED/N607_PENDING`。

|ID|设计要求|状态|落地与证据计划|
|---|---|---|---|
|P0-01|部署训练不再无条件深拷贝完整checkpoint模型|local_verified|`fit_sf_tapft_inplace`复用调用方独占模型；默认研究入口仍复制|
|P0-02|只保留许可参数训练前锚点，不复制完整初始`state_dict`|local_verified|许可参数CPU锚点、优化器可达性检查和冻结buffer等值检查通过|
|P0-03|把缓存后缀收敛为清晰、稳定、可测试的部署接口|local_verified|`encode_h6_prefix`、`forward_h6_suffix`和引用型`H6SuffixTrainer`已实现|
|P0-04|输出checkpoint绑定的紧凑delta，避免把完整适配模型当作部署状态|local_verified|delta v2从许可锚点计算；严格loader及query closure支持delta-only；历史v1继续兼容|
|P0-05|报告常驻推理内存、适配额外峰值、cache字节和delta字节|local_verified_measurement_pending|无新增依赖的Windows RSS采样、CUDA峰值、预热3次/正式10次汇总已实现；真实N607数值待测|
|P0-06|扩展缓存等价与原位训练回归测试|local_verified|相关5文件共99项测试通过；真实checkpoint parity待N607 smoke|
|P0-07|FP16训练结束后执行一次FP32 full-path support安全复核|local_verified|有限性、argmax、margin和逐类recall已检查；失败自动恢复许可锚点并FP32重训|
|P0-08|原位模式不得改变H6的目标函数、训练日程和许可参数集合|local_verified_remote_parity_pending|矩阵仅改变所有权/cache精度/输出形态；同row真实prediction待闭合|
|P0-09|不得新增逐成员hash、签名或额外发布门|rejected_extra_gate|依照`Exclusive Minimal Experiment Workflow`，使用Git提交、单一release归档SHA和既有状态审计|
|P1-01|在新的未暴露合法capsule上比较D0–D4|deferred_after_p0|D0 H6、D1 Q2A、D2 Q2B、D3 R1-T、D4 head-only class-CVaR；P0闭合后另行预登记|
|P1-02|不继续当前HardPair、Adapter、完整`t3`、frequency或EMA路线|frozen|从后续最小矩阵排除，不补跑已证伪路线|
|P2-01|固定晋级结构后扩展receiver、三scene、K=10/5/2和多seed|deferred_after_p1|只对P1晋级候选执行，不作为P0/P1前置门|

## 已解决的设计歧义

- “原位适配”只用于显式部署入口；既有研究入口继续默认复制checkpoint，避免调用方意外持久化模型变更。
- “delta-only”指部署状态只需基础checkpoint加紧凑delta。为保持历史实验可重放，完整clean-single bundle在迁移期仍可显式生成，但P0部署行默认关闭，并用delta materialize后的真实prediction闭合证明可用性。
- FP16安全复核不读取query。它只在训练结束后使用同一合法support做一次FP32完整路径前向；任何fallback在query prediction之前完成并记录。
- 报告提出的额外非许可参数hash与项目穷尽式白名单冲突，记为`REJECTED_EXTRA_GATE`，不实现、不阻塞实验。

## 阶段顺序

1. P0代码与聚焦测试。
2. P0真实checkpoint无query smoke、独立P0/P1审查、N607隔离资源与同row预测闭合。
3. P0结果分析和发布。
4. 新未暴露capsule上的P1 D0–D4。
5. 仅对晋级候选执行P2推广确认。

## 本地验证

- `ssr-gpu`：`test_target_only_progressive_adapt.py`、`test_target_only_progressive_deploy.py`、`test_target_only_progressive_runner.py`、`test_stage2_sf_tapft_query_closure.py`、`test_sf_tapft_deployment_benchmark.py`共99项通过。
- 修改模块Python编译通过；三行P0矩阵严格解析通过。
- 一次独立P0/P1正确性审查发现delta-only尚未接入query closure，已定点修复并用delta v2路由测试闭合；复审未发现会导致真实实验跑错、越权、覆盖输出或不能产生合法prediction的剩余P0/P1。
