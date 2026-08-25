# Slow-Fast P0.5地面校准与独立目标验证设计

## 1.目标与证据边界

P0.5解决两个同时存在的问题：固定绝对trust尺度不能反映目标support的实际几何，当前support-only风险也不能稳定预测独立query收益。该阶段不重训完整Phase1，不把旧target query truth反馈给门控，也不进入P1因子化慢基或P2中间层Adapter。

地面校准只使用Phase1.5的`L_s` source feature cache。每个source receiver-held-out episode把K=10 support作为唯一适配输入，held-out query只用于训练和选择地面gate。部署时gate只消费目标support统计。任何目标query及其truth、role和全局类别计数均不能更新或选择状态。

## 2.候选收缩

- `FAST_FILM_R8`是唯一主候选，执行完整地面校准。
- `FAST_LOWRANK_R8`只保留一个轻强度辅助消融。
- `COMMON_SHIFT_R4`只保留一个轻强度负对照，不扩大网格。
- `DA0_REG0`始终是安全回退。

旧V2的80状态只用于truth-last诊断，不得据此在receiver20-1、seed392002上挑选状态重跑。

## 3.支持集几何与fold统计

每个候选状态记录逐fold风险增益，而不是只记录拼接后的平均值：

```text
g_f(lambda)=R_f(0)-R_f(lambda)
```

门控同时使用正增益fold数、均值、标准差和90%下置信界。默认要求至少5/6个fold风险改善且LCB大于0。

移动约束由单一`max_feature_move<=0.15`改为三部分：

```text
Q90(move)<=tau90
max(move)<=tau_hard
Q90(move/(boundary_distance+eps))<=kappa
```

先计算全强度状态在support上的`Q90(move)`，再生成row-specific归一化系数：

```text
c_t=min(1,tau90/(Q90(move_lambda1)+eps))
lambda_eff=c_t*lambda_nominal
```

所有阈值和gate参数只能由source receiver-held-out episode冻结。target support可计算统计和`lambda_eff`，但不能重新学习阈值。

## 4.地面receiver-held-out校准

从现有`GroundFeatureCache`按receiver、physical sample ID、view和class构造episode。episode support/query物理ID严格互斥，每类support固定K=10；source held-out query仅承担地面监督角色。

比较四种预注册规则：

1. V2固定max trust对照；
2. `Q90+hard max`分位数trust；
3. margin-normalized相对trust；
4. support-normalized强度加fold LCB。

地面校准输出一个仅供Phase1.5分析与预登记使用的JSON，其中包含冻结阈值、gate参数和source-held-out聚合验证摘要，不包含source样本、特征或样本级派生物。正式Phase2不得打开该JSON；发布前只把最终冻结的纯deployment数值/布尔参数抄入预登记row config。Phase2 runner严格拒绝`calibration_path`和任何source统计字段。

## 5.交叉验证公平性与审计语义

- runner显式传入row seed作为`crossfit_seed`，同scene的候选共享相同划分。
- 重复2-fold划分去除相同或互补重复；每次repeat内每个物理样本只进入一个validation fold。
- selector接收physical IDs并直接检查train/validation集合互斥；不创建逐support-token哈希。
- 新门控记录`selection_protocol=repeated_stratified_2fold`、`crossfit_fit_count`和`loo_fit_count=0`。
- 计算量拆分为deployment候选、cross-fit、legacy诊断、shadow网格和总诊断更新量。

指导报告提出的fold physical-ID hash与逐样本哈希属于项目规则禁止的额外hash门，记为`REJECTED_EXTRA_GATE`。用内存中的ID集合互斥检查和计数替代。

## 6.truth-last scorer与旧V2响应面

scorer必须在全部prediction校验后首次读取truth，并分别输出：

- 旧类decision changes、positive flips和negative flips；
- 新类query decision changes；
- true-class cosine、top1-top2 margin和score-vector L2变化；
- 每类margin变化；
- 新类样本最大旧类cosine变化，即新类侵入指标；
- 每场景FAST响应面、support/query Spearman相关性和移动—收益表。

REG0的新类准确率继续为`N/A`。新类分数只用于侵入风险，不解释为新类识别能力。

## 7.独立目标验证

地面校准冻结后，只允许一次新的receiver／seed capsule验证：

```text
DA0_REG0 vs DA1_REG0
```

必须使用新的`capsule_id`、`split_id`和此前未用于调参的target query。如果本地/远端没有合法新capsule，本轮可以完成地面校准和旧V2诊断，但目标性能状态保持`MISSING_INDEPENDENT_TARGET_CAPSULE`，不得重用receiver20-1、seed392002补结果。

晋级同时要求：

- 三场景聚合mean至少`+1.0pp`；
- 聚合floor至少`+0.5pp`；
- worst-scene mean不低于`-0.5pp`；
- 任一scene/class不低于`-5pp`；
- 新类侵入不超过source-held-out冻结阈值；
- McNemar或配对bootstrap不与“无改善”相矛盾。

若独立验证仍只在一个轻场景有效，或support/query Spearman低于0.2，则P0正式结束并触发P1。P1、P2在本设计中只记录条件状态，不与P0.5同时实现。

## 8.实现边界与验证

新增模块分别负责support统计、source校准和诊断汇总；现有runner只负责Phase2 target support适配和query只读prediction。所有行为变更按TDD先写失败测试。发布遵循项目八项最小流程，只计算一次release归档传输SHA，进行一次独立P0/P1正确性审查。
