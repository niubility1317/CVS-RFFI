# SF-TAPFT H6 P0原位适配与轻量部署设计追踪

来源：用户提供的《SF-TAPFT从研究型4-fold选择流程转为full-support星上适配后的继续优化报告》，2026-08-28。用户已确认后续按推荐顺序执行。当前状态：`DESIGN_FROZEN/IMPLEMENTATION_PENDING`。

|ID|设计要求|状态|落地与证据计划|
|---|---|---|---|
|P0-01|部署训练不再无条件深拷贝完整checkpoint模型|pending|新增显式原位适配入口；保留研究入口默认复制语义，防止旧调用方被隐式改变|
|P0-02|只保留许可参数训练前锚点，不复制完整初始`state_dict`|pending|许可参数逐张量锚点、优化器可达性审计和冻结buffer等值检查|
|P0-03|把缓存后缀收敛为清晰、稳定、可测试的部署接口|pending|公开`encode_h6_prefix`和`forward_h6_suffix`，由`H6SuffixTrainer`持有引用而非复制模型|
|P0-04|输出checkpoint绑定的紧凑delta，避免把完整适配模型当作部署状态|pending|delta导出改用训练前许可参数锚点；增加严格materialize/load闭合；完整bundle仅保留为兼容性可选项|
|P0-05|报告常驻推理内存、适配额外峰值、cache字节和delta字节|pending|独立单进程基准，预热3次、正式10次，报告median/P90/max、CPU RSS、CUDA allocated/reserved|
|P0-06|扩展缓存等价与原位训练回归测试|pending|随机norm/head扰动、1/10/100步、不同batch/K；核对logit、梯度、最终许可参数和非许可状态|
|P0-07|FP16训练结束后执行一次FP32 full-path support安全复核|pending|仅用support核对有限性、argmax、margin和逐类召回；失败时恢复锚点并以FP32缓存重训|
|P0-08|原位模式不得改变H6的目标函数、训练日程和许可参数集合|pending|同support固定seed与复制模式做许可参数delta、预测和资源同row对照|
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
