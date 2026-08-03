# Stage2联合方法探索回顾

## 1.状态

|字段|值|
|---|---|
|报告ID|`stage2_joint6_retrospective_20260803`|
|状态|`RETROSPECTIVE_COMPLETE / JOINT6_DESIGN_FROZEN / IMPLEMENTATION_PENDING / NO_NEW_PERFORMANCE_RESULT`|
|范围|D127 r1/r2/r3、D128 r1及下一轮完整联合目标|
|协议|`p2_min_v1`|

## 2.已完成轮次与真实性能边界

|run|最后到达阶段|停止原因|prediction/score|性能结论|
|---|---|---|---:|---|
|D127 r1|prepare前/入口|namespace、Python/NumPy/Torch运行时兼容链|0/0|`NO_PERFORMANCE_RESULT`|
|D127 r2|Phase1输入读取|历史D106 receipt与当前checkout执行闭合漂移|0/0|`NO_PERFORMANCE_RESULT`|
|D127 r3|Phase1 outer audit|冻结asset被要求保留可微caller graph|0/0|`NO_PERFORMANCE_RESULT`|
|D128 r1|Phase1 outer audit|isolation/equivariance/nonzero/query-change closure失败|0/0|`NO_PERFORMANCE_RESULT`|

四次运行均不能评价候选强弱；无正收益、无负收益、无可晋级结果。

## 3.回顾结论

1.D127/D128把方法训练、真实checkpoint hook、source outer audit、量化bundle、三候选merge和Target scorer一次性耦合，导致本地单测通过但真实source资产路径持续在prediction前失败。
2.三轮后仍修复同一release体系，已经偏离“快速获得真实功能证据”；D128虽缩为单A，但仍继承相同Phase1 outer-audit链，因此没有真正缩小失败面。
3.旧设计只形成`base/adapted×qKNN/D92-Lite`四臂，没有满足持久目标要求的`2种表示×3种头`六臂，也不能严格回答精简D92相对历史D92的同表示贡献。
4.下一轮必须保留旧类适应和新类注册同等优先级，完整报告before/after、`seen_new_acc`、`H_old_new`、逐类旧类准确率、floor和forgetting；只看内部loss、support或feature变化不完整。
5.`LEO_weak-only`、无clean/source运行时访问、无query truth/role/quota、逐query全类竞争继续保持；既有`VALIDATED_ONCE`数据不得因候选变化重验。

## 4.已拒绝路线

|路线|结论|原因|
|---|---|---|
|D127/D128 Phase1 autograd＋checkpoint replacement＋outer audit|关闭实现路线|连续四次prediction前技术停止，未形成真实性能|
|只运行D129单候选四臂并把它当最终目标|拒绝作为终点|过度收缩，未覆盖完整持久目标的候选与六臂|
|重复D62/D92/SVRN 125矩阵|禁止|已有历史证据且不直接研发新功能|
|用更多gate、数据复验或通用发布平台解决失败|禁止|不直接证明下一真实实验正确运行|

## 5.冻结决策与下一步

- 冻结两条而不强凑三条：`CSPAR-2`是rank2 sealed nuisance-axis PSD度量；`SRDH-2`是跨类support summary驱动的rank2非线性残差。`RDCE-r3`与前者同族，本轮关闭。
- 每个候选共享`R0/R1`两种160维缓存和`qKNN/D92-Full160/D92-Lite160`三头，形成完整六臂。formal 288维D92只作外部同row系统参考。
- K5仅设三条主比较：`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`；都要求池化H及总正确数严格增加，`A_old/N/F_old`非劣。K1只归因DA，F/L严格alias Q。
- Phase1审计保留7receiver×6class=42个receiver-held×class-LOCO fold；它只验证隔离、等变、非恒等及负迁移，不冒充Target性能。
- 下一步实现科学核心、两份资源receipt、聚焦协议负测和真实checkpoint no-truth smoke。必要门闭合后直接发布小矩阵，不增加新控制面或重复数据验证。

## 6.当前资源状态

无N607运行；D128 PID已退出，GPU0-7为空闲，SSH/TCP22已清理。下一run尚未创建；目标冻结不代表方法已有成效。
