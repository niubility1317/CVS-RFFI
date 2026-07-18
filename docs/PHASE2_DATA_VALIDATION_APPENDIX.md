# Phase2数据一次验证实现附录

本附录服务于`protocol_schema=p2_min_v1`，只规定数据builder/validator如何生成可复用的`VALIDATED_ONCE`句柄。它不是每个方法研发任务的前置工作清单。

## Builder一次性责任

Builder在Phase2边界外完成：

1. 为原始记录生成稳定、不可重命名的`physical_sample_id`；
2. 在跨场景物理ID互斥后，为每个物理样本随机选择一个允许的LEO弱场景和一个seed，只生成一份接收IQ；
3. 建立receiver/TX/scenario/K/support-query split，并验证物理ID互斥；
4. 删除clean/raw/source路径、构建入口、truth sidecar和禁止成员；
5. 计算payload、成员表、split和协议schema的内容摘要；
6. 输出data capsule及最小运行句柄。

Phase1 bundle使用独立的`bundle_id`和独立validator。该validator检查checkpoint与可选只读int8聚合知识是否共同封存、成员是否合法。更换checkpoint或int8组件只重验`bundle_id`，不得使已经通过的数据`capsule_id/split_id`失效；更换data capsule同样不要求重建合法bundle。

## Validator自动检查

- schema、`capsule_id`、`split_id`和`VALIDATED_ONCE`状态一致；
- 每个物理ID恰有一个场景、一个overlay seed和一个接收IQ payload；
- 三场景物理ID两两不交，support/query物理ID不交；
- K等于每类独立support物理ID数，不把计算view计入K；
- capsule成员allowlist不含clean/raw/source dataset、source sample feature、cache build spec或truth sidecar；
- query predictor schema不含query标签、角色、真实batch类数或quota；
- 独立bundle validator确认可选int8组件与checkpoint共同封存、只读、不可独立替换且没有样本级成员；
- runtime访问账本只触及允许的bundle、capsule、split和输出路径。

## 复用与失效

研发runner的数据面只消费`capsule_id + split_id + protocol_schema + phase2_data_status`，模型面另消费已验证的`bundle_id`。以下变化不触发数据重验：方法名、method lock、adapter结构、超参数、epoch、优化器、原型融合/量化策略、注册规则、资源预算、日志、报告格式和`bundle_id`变化。

只有以下变化使句柄失效：接收IQ payload、物理ID、receiver/TX集合、scenario分配、K、support/query划分或协议schema发生变化。

验证失败时返回具体失败项并只重建受影响的capsule/split；不得要求其他`VALIDATED_ONCE`数据随方法迭代重复追溯Phase1/source/clean历史。
