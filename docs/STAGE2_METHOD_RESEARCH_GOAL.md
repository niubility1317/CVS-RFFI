# Stage2域适应＋qKNN＋互补机制联合研发目标

版本：2026-07-22
修订：域适应＋qKNN＋互补机制主路线、候选自适应因果证据、正式发布完整125优先
状态：可直接作为新`/goal`目标Prompt
协议：`protocol_schema=p2_min_v1`
初始化文档：`docs/STAGE2_RESEARCH_AGENT_INIT.md`

## 1. 单一目标

在`E:\type10-7`中，严格遵循当前`AGENTS.md`、`项目.md`和`p2_min_v1`，持续研发、实现并验证可逐样本部署的极轻型Phase2方法。主路线固定为“域适应＋qKNN＋一个互补机制”：ADV3B02 final checkpoint是必须保留的matched anchor，不是唯一允许的Phase1 backbone；需要新Phase1 encoder或bundle的候选，只要完全使用合法Phase1数据、在target访问前冻结并接受matched比较，也可作为域适应路线进入研发。每个候选必须同时包含：

1. 域适应功能：只用合法Phase1 bundle和当前row target support降低目标接收机、LEO场景、信道或表征偏移；可以作用于encoder、feature、normalization、metric、receiver nuisance、概率先验或support-conditioned adapter，不把RDA/SRDA规定为唯一方法；
2. qKNN统一分类主干：每个query独立面对全部实际注册旧类和新类，保留sample-level局部多峰邻域、INT8 support bank、按类归一化证据和identity fallback，不允许角色、类别配额、全局重分配或跨query图；
3. 一个互补机制：只针对旧类遗忘、新类注册、old/new尺度失衡、hubness、support噪声、coverage、类级不确定性或INT8 margin中的一个明确剩余误差；不得以第二个分类头替换qKNN的统一全类竞争；
4. 联合协同：域适应与互补机制都要有可归因的独立价值，联合方法必须证明超过两者简单叠加的协同收益，即`1+1>2`。

工作重心放在一个冻结机制delta、必要本地测试、独立review、Git提交、完整125正式发布和真实N607性能证据。不得把大部分时间消耗在重复数据验证、authority/hash重建、跨run原始SHA对齐、控制面扩展、报告格式重构或无关文献扩展上。普通负结果只否定已检验的具体实例，不自动封禁整个方法族；只有新增数据权限、改变科学场景、干预用户现有任务或其他高影响动作才请求用户授权。

## 2. 当前研发起点

优先复用经过审查的代码、runner和报告资产，但不把历史代码链写成新方法必须经过的顺序。现有资产包括：

- Patch A的`z_id160`identity Student-t single-qKNN，可作为轻量reference；
- Patch B的qKNN＋Shrinkage RDA/SRDA，可作为域适应reference；
- Patch C-id的低秩nuisance metric，可作为support可辨识性诊断；
- coverage-controlled cross-branch transport、压缩地面原型和共享domain basis，可作为域适应候选组件；
- 已有int8、runner、matched report和资源审计代码，应在语义匹配时复用。

已有证据用于约束假设和设计证伪实验，不构成算法白名单或永久黑名单：

- D92表明该RDA实例改善old/floor但损伤new；后续域适应与互补机制必须消除old/new交换，而不是被限定为继续使用SRDA；
- D93/D94表明在ground coverage仅约`0.144–0.227`时强制搬动整体坐标会负迁移。全局transport若重新进入，必须说明改变了哪个失败假设，并预注册coverage、identity回退和决策有效性检验；
- D99表明Phase1 LODO正信号不能替代真实target泛化，且一次混改metric、kernel和fusion会破坏归因；
- D100表明增加第二个头不自动产生互补；任何单头、双头或多头方法都必须先证明差异性和救援价值；
- Role-Oracle只用于刻画上限，不得作为协议合法方法、参数选择信号或晋级证据。

## 3. 开放域适应/互补机制空间与落地前可行性门

### 3.1 方法探索自由

qKNN固定承担统一全注册类逐query决策，但不固定域适应结构、rank、表征或互补机制。满足协议并可证伪的方法均可进入候选池，包括但不限于：

- Phase1域不变或域因子化encoder、监督/半监督meta-DG、receiver nuisance分解；
- support-conditioned normalization、adapter、FiLM/LoRA、低秩metric、transport、Riemannian或概率域适应；
- 低秩shrinkage RDA、receiver nuisance correction、support-conditioned adapter、normalization、metric learning或合法Phase1新表征；
- class-balanced calibration、旧类锚定、局部密度校正、anti-hubness、Bayesian shrinkage、不确定性校准或轻量混合专家；
- ground压缩知识与target support的共享先验、后验更新、coverage控制或连续收缩。

上述列表是例子，不是白名单。候选可以不用`z_id/z_dom`、SRDA或地面原型，也可以提出新的Phase1 bundle或encoder；但最终分类必须仍由qKNN完成全部注册类的统一逐query竞争。只要Phase2读取的任何ground状态符合`项目.md`的共同封存、多样本聚合、int8和不可替换sidecar要求，即可评审。

每个候选先提交一张方法卡，不得先写代码再补设计：

```text
candidate_id / revision
domain_adaptation_mechanism
qknn_decision_rule
additional_complementary_mechanism
why_da_qknn_other_are_complementary
legal_inputs_and_persistent_state
K1/K5/K10_identifiability
decision_geometry_change
old_new_conflict_handling
coverage_noise_fallback
resource_estimate
minimal_falsifier
files_and_interfaces
```

### 3.2 可行性讨论与设计冻结

候选按`DESIGN_DRAFT -> FEASIBILITY_REVIEW -> DESIGN_FROZEN -> IMPLEMENTING`推进。只有`DESIGN_FROZEN`可进入代码落地。可行性讨论必须在一个设计波次内回答：

1. 协议：所有训练、support、bundle、query和状态读写是否合法；
2. 可辨识性：给定K和注册类数，参数自由度、统计量和更新是否可由现有support确定；
3. 决策有效性：方法是否真正改变可观测decision geometry，是否会被共同可逆变换、完整重估或其他不变性抵消；
4. 互补性：域适应处理的偏移与OTHER处理的剩余误差是否不同，联合时为何可能产生正交救援；
5. 负迁移：coverage不足、support噪声、old/new竞争或量化误差出现时如何收缩、回退或拒绝更新；
6. 资源：参数、state、MAC、时延、显存、optimizer step和int8生命周期是否可能过门；
7. 工程：需要修改哪些文件、依赖和接口，能否形成一个主要delta及完整发布闭包；
8. 证伪：最小实验在什么结果下立即拒绝该实例，避免落地后反复补丁。

评审只输出`MERGE/REVISE/REJECT`和证据缺口。主agent在修改方法代码前向用户发布一份简短可行性摘要，说明候选机制、为什么可行、主要风险、证伪条件和冻结改动范围；这不是新增授权门，只有扩大数据权限、科学场景或高影响操作时才等待用户决定。`REVISE/REJECT`不得进入正式实现；若某个关键事实只能用代码确认，可以批准一次不读取target query、不可晋级的`FEASIBILITY_SPIKE`，其代码不得自动成为候选实现。设计冻结后若要改变核心机制、输入、loss、head或适应规则，必须增加revision并重新审查；仅修复与冻结设计一致的接口错误不重开方法讨论。

### 3.3 当前候选只作参考

RDA/SRDA、receiver nuisance correction、support-conditioned adapter、normalization、metric learning和合法Phase1新表征均保留为域适应候选；BCRR、class-balanced calibration、旧类锚定、局部密度校正、anti-hubness、Bayesian shrinkage和不确定性校准均可作为OTHER。历史qKNN＋SRDA只是普通reference，不具有优先晋级权；不得因为某条路线已有代码就压制更可行的方法，也不得因一个具体版本失败就否定整个方法族。

### 3.4 通用协同判据

qKNN作为共同分类底座。对于可拆分候选，定义matched reference`M0=qKNN`、`M_DA=候选域适应＋qKNN`、`M_OTHER=qKNN＋互补机制`和`M_JOINT=候选域适应＋qKNN＋互补机制`。主协同指标固定为：

```text
I_syn = H(M_JOINT) - H(M_DA) - H(M_OTHER) + H(M0)
```

`M_DA`和`M_OTHER`都必须相对`M0`产生独立净正确决策收益，并保护old/new/floor/min-class/forgetting；仅有loss下降、support fit提高、metric非identity或logit变化不构成成功。`M_JOINT`必须严格优于两个单组件，且不能靠牺牲任一保护指标制造表面贡献。首次完整125正式实验要求聚合`I_syn>0`、正协同覆盖至少188/375个scene slice且至少2/3个scene的均值为正，并同时报告全部125个job及receiver/scene/K/seed/new-count分层，不得用有利子集替代总体；最终确认要求paired mean `I_syn>0`且95% CI下界>0。

对参数共享、端到端或天然不可拆分的方法，不强制伪造四个独立模块。设计冻结前必须预注册等价的组件干预，例如parameter-block freeze、stop-gradient、loss masking、matched surrogate或Shapley/ANOVA interaction，并由独立监督员确认它能区分域适应贡献、分类贡献和联合贡献。无法完成贡献辨识的候选仍可作为探索性诊断，但不得声明`1+1>2`或正式晋级。

## 4. 候选自适应因果证据包

不再规定固定六臂、固定名称或固定算法。对照数量由候选机制决定，只保留能回答因果问题的最小证据。可拆分候选默认采用四状态factorial core：

|功能状态|域适应|分类机制|用途|
|---|---|---|---|
|`M0`|matched reference|qKNN only|同row基准|
|`M_DA`|candidate on|qKNN only|隔离域适应贡献|
|`M_OTHER`|identity/off|qKNN＋OTHER|隔离互补机制贡献|
|`M_JOINT`|candidate on|qKNN＋OTHER|检验联合与交互|

这四个状态不是方法白名单，也不要求候选使用四份独立代码。若方法天然耦合，可按第3.4节采用经过审查的等价干预。任何额外臂必须对应明确风险或证伪问题；不得为了满足固定数量制造无信息实验。

可拆分候选的machine receipt应在适用处证明：

```text
M_DA.qknn_other_state == M0.qknn_other_state
M_OTHER.adaptation_state == M0.adaptation_state
M_JOINT.da_component == M_DA.da_component
M_JOINT.other_component == M_OTHER.other_component
same capsule / split / row / seed / query policy
```

共享参数方法无法逐值相等时，receipt必须记录冻结参数块、loss mask、stop-gradient、随机性、compute/state差异和干预实现，避免把第三个变化混入协同项。

负对照按候选风险选择，而非预设统一清单。可能包括identity/no-adaptation、ground知识关闭、support标签或特征置换、随机同rank子空间、coverage回退、共同变换不变性、head logit置换、prior关闭或量化精度对照。每个候选至少需要一个协议安全的null和一个直接攻击其核心机制的falsifier；共同可逆变换后若决策不变，不得把“完成对齐”写成性能机制。

## 5. 数据协议硬边界

Phase2只能读取：immutable Phase1 deployment bundle、匹配`VALIDATED_ONCE`的固定单LEO弱观测capsule、当前row合法target-old/target-new K-shot support与标签、query访问前锁定的数据无关配置。

必须保持：

- 一个物理IQ仅有一次随机允许的LEO弱信道观测；K-shot是K个独立物理support；
- support/query物理ID互斥，三个场景的物理ID集合互斥；
- 不访问clean/raw/source样本、sample-level source feature、source replay或可替换sidecar；
- query逐样本面对全部注册类，只前向和一次评分；不得更新任何状态；
- 禁止query伪标签、熵最小化、图、OT/Hungarian、quota、角色Oracle和batch reassignment；
- ground知识只能来自target访问前与checkpoint共同封存的int8多样本聚合组件。候选可以将其用于共享先验、class-conditional旧类知识、metric、likelihood或其他合法机制，但必须保持类标签置换等价，不得预置新类、读取成员级状态、按真实old/new角色路由或替代Stage2-B/C的target support；
- target-old和target-new正式状态均采用int8，无FP32 sidecar。

匹配的`capsule_id/split_id/schema=p2_min_v1/VALIDATED_ONCE`只核对一次。只有received IQ字节、physical ID、receiver/TX集合、scenario、K、support-query划分或schema改变时重验；方法、adapter、head、超参数、checkpoint推理状态、bundle、资源或报告变化不得触发数据重建。

## 6. 高效研发顺序

1. 并行提出域适应、分类机制和联合优化候选方法卡；允许不同方法族竞争，不预先指定胜者。
2. 在任何正式代码修改前完成可行性讨论。监督员依据协议、可辨识性、决策有效性、互补性、负迁移、资源、工程闭包和证伪条件裁决；只有`DESIGN_FROZEN`候选进入实现。
3. 主agent从通过者中选择信息增益/成本最优的1–2个候选，冻结revision、matched reference、候选自适应因果证据包、指标、停止条件、文件owner和最小diff。不得因代码已有或个人偏好选择方法。
4. Terra代码owner按冻结设计完成最小实现和专项测试。核心机制需要改变时停止编码、增加revision并回到可行性审查，不用连续补丁掩盖设计问题。
5. 根据候选机制使用合法Phase1 LODO/LOCO、source validation或support-only cross-fit完成开发证据；这些代理不能替代target held评价，也不得读取query选参。
6. 冻结候选完成本地专项测试、协议负例、无query smoke、独立`P0=0/P1=0`review和Git提交后，第一次正式N607性能发布即运行既有完整`125/125`矩阵；禁止发布只覆盖单一K、单一receiver、单一scene、少量seed或其他有利条件的窄性能run。K5/18-slice等小矩阵只可作为本地实现验证，不形成独立N607性能发布。
7. 当前完整125定义固定为`5 target receivers×5 independent seeds×5 registration slices=125 jobs`；5个切片为`(K10,new5)`、`(K10,new10)`、`(K10,new20)`、`(K5,new20)`、`(K1,new20)`，每个job均覆盖`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。四臂方法必须在同一commit、同一冻结状态和同row因果对照下，一次性产出`375 prediction slices`和`1500 score rows`，并报告逐类及receiver/scene/K/seed/new-count分层。不得以旧的`5 receivers×5 seeds×K{1,2,5,10,20}`matched-history bundle冒充本目标的完整125。
8. 完成全部prediction但未达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；没有完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。125结果不得用于回调候选结构、rank、loss、融合、量化格式、阈值或fallback；任何机制变化必须增加revision并重新审查。125通过后，以同一commit运行`5 receivers×5 seeds×3 scenes×K{1,5,10,20}×new{2,5,10,20}=1200`评价单元完整确认。

## 7. 性能与资源门

每个完成候选必须同时报告注册前old、注册后old、old adaptation gain、seen-new、H、BA、全部注册类floor、min-old、min-new、forgetting、old→new/new→old和完整逐类/receiver/scene/K/seed结果；不得只说明缺陷或拼接不同row极值。

K10完整确认硬门：

- `old_acc_after_increment≥92%`；
- `min_old_class_acc≥88%`；
- `seen_new_acc(new5)≥92%`；
- `seen_new_acc(new10)≥90%`；
- `seen_new_acc(new20)≥86%`。

同时满足：K5核心指标相对matched K10下降≤3pp；K1总体及每receiver old adaptation gain≥0；K1相对direct ADV3B02至少+2pp且paired 95% CI下界>0；K5/K10/K20遗忘不高于matched reference及当前最强合法基线；`M_JOINT`相对`M_DA`和`M_OTHER`的old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new均不降，forgetting、old→new和new→old均不增；`I_syn`通过。

资源硬门：trainable parameters≤80000、adaptation epochs≤30、optimizer steps≤50、persistent incremental state≤256KB、dense query graph=false、query-dependent batch optimization=false。正式int8要求top1一致率≥99.5%、大margin flip=0，并报告实际wire、MAC、平均/P95时延、峰值显存和前向次数。

## 8. 研发自由与无效重复边界

永久禁止的是协议违规、证据污染、因果失真和重复工程，不是某个算法名称。qKNN、RDA、transport、adapter、地面原型、多头、深模型或任何其他方法族都可以探索；历史负结果只否定相同输入、相同机制和相同失败条件下的具体实例。

失败路线重新进入候选池时需提交`REENTRY_CARD`：

```text
failed_instance_and_evidence
changed_assumption_or_mechanism
why_previous_failure_no_longer_applies
minimal_differentiating_test
falsifier_and_stop_condition
additional_cost
```

默认不做以下无效工作：

- 重复建设或人工追溯已经`VALIDATED_ONCE`的数据、hash、allowlist、authority和准入系统；
- 把数据句柄、方法、candidate revision、Git commit、run和result混称为一个“版本”，或进行非matched比较；
- 要求跨run row-specific opaque handle或封装artifact原始SHA bit-exact；跨run只比较稳定语义，raw SHA仅审计；
- 用125、Role-Oracle、development query或confirmation query回调候选、结构、量化格式、阈值或fallback，或只摘取125中的有利单元宣称成功；
- 用support accuracy、prototype重构、LODO单点、代码测试、进程启动或资源达标替代held性能；
- 未完成可行性讨论和`DESIGN_FROZEN`就正式落地代码，再靠反复修改寻找方法；
- 把某个reference candidate、固定rank、固定域适应方法或固定OTHER提升为所有方法必须遵守的架构；
- 不提交`REENTRY_CARD`便重复D93/D94式低coverage强制transport、D99式多机制混改、无互补证据的多头叠加或其他已证伪实例；
- 多个agent重复读取全量历史、修改同一文件、独立启动同一run ID、自我审查或让主agent线性等待N607；
- 为满足形式增加无法改变决策、无法证伪假设或不影响晋级判断的对照、报告和控制面。

## 9. 完成条件

只有方法卡、可行性审查、`DESIGN_FROZEN`、候选自适应因果证据包、合法开发证据、首次完整125正式实验、1200单元完整确认、协议证据、int8生命周期、资源审计、matched baselines、完整日志、报告、复现命令和Git提交全部存在且性能门全部通过，才能标记完成。

完成实验但性能未达标时，必须记录`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并返回下一单一机制假设。无prediction的运行只能记录`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得混入算法结论。
