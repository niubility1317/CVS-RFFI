# Stage2域适应＋qKNN＋互补机制研发目标

版本：2026-07-23
修订：开放域适应、`z_id/z_dom`双qKNN、统一旧类/新类决策、单一OTHER、完整125与N607八卡并行发布
状态：可直接作为新`/goal`目标Prompt
协议：`protocol_schema=p2_min_v1`
初始化文档：`docs/STAGE2_RESEARCH_AGENT_INIT.md`

## 1.单一目标

在`E:\type10-7`中，严格遵循实时`AGENTS.md`、`项目.md`和`p2_min_v1`，研发可在星上稀缺计算资源下运行的Stage2域适应方法。候选至少使用Phase1不可变deployment bundle和当前row合法target support；地面压缩知识、模型adapter、typed metric、domain-factorized表示及概率先验均为可选资产，不得把任一资产写成所有revision的必经路线。

主路线固定为：

```text
真正降低目标receiver、LEO场景、信道或表征偏移的域适应
＋qKNN统一旧类/新类分类
＋一个针对剩余误差的互补机制
```

域适应可以改变encoder、输入前端、normalization、轻量adapter、support-conditioned表示、metric、邻域权重或概率状态。它必须具有明确的域偏移机制，并在held query上产生可观测的邻居贡献、margin、argmax或净正确决策变化；只有loss下降、support fit提高、metric非identity或logit数值变化不构成DA成功。RDA/SRDA、receiver nuisance correction、support-conditioned adapter、normalization、metric learning和合法Phase1新表征均可公平进入候选。

当前下一优先revision固定为`ADV3B02-TS-DRQKNN-BCRR/r2-affine`：保留ADV3B02的`z_id`与`z_dom`双分支；`z_dom`只在每个候选类内部条件化`z_id`Student-t qKNN证据，最终跨类决策始终由`z_id`qKNN完成；BCRR是唯一OTHER。r2把四臂共同support codec冻结为逐向量仿射INT8，并修复完整FP32 teacher审计与系统故障增量停派。该冻结只约束本revision，不把双qKNN、固定rank、仿射codec或BCRR升级为后续所有方法的全局必经路线。

最终目标是：

- Stage2-B的K5/K10域适应单组件相对同row`M0`产生真实净正确决策增益，并在receiver与scene分层上稳定；
- K1在不可辨识时精确identity或强收缩，不伪造单shot DA收益；
- Stage2-C加入新类后同时保护旧类、提高seen-new、H、floor和最差类，降低forgetting及双向混淆；
- `M_JOINT`严格优于DA单组件和OTHER单组件，且mean`I_syn>0`，证明`1+1>2`；
- 在参数、optimizer step、持久state、MAC、时延、显存和INT8生命周期上满足星上部署约束；
- 持续以新revision实验推进，直到达到本文件性能与资源门，或得到足以证明当前Phase1 bundle/encoder信息不足的可复现实验证据。

工作重心是方法、实现和真实N607性能。不得把大部分时间消耗在已经`VALIDATED_ONCE`的数据重验、authority/hash重建、跨run opaque SHA对齐、控制面扩展、报告格式重构或无关文献综述上。

## 2.三个模块的严格定义

### 2.1域适应

域适应必须满足以下定义：

- 只用不可变Phase1 bundle和当前row合法support产生冻结的表示、模型、metric、邻域或概率状态；
- query只能逐样本读取该状态，不能继续fit、选择、回滚或更新；
- 必须说明目标域偏移假设、可辨识自由度、连续收缩或identity fallback；
- 必须报告feature、neighbor、身份邻居贡献、margin、argmax及wrong→correct/correct→wrong变化；
- 只有正式held query净正确决策和同row指标收益才能证明DA有效。

域适应的信息面保持开放。任何由`项目.md`允许、能在Phase2合法读取或由当前固定received IQ合法计算的信息都应进入候选审查范围，包括：

- checkpoint内部identity/domain分支、`z_id`、`z_dom`及合法中间表征；
- 与checkpoint共同封存的domain classifier、domain basis、normalization统计和adapter/meta-learning先验；
- INT8 domain×class聚合中心、地面压缩多原型、半径、聚合方差、低秩残差和量化尺度；
- Stage2-B的target-old support IQ、标签、注册表和由同一received IQ计算的FFT、均衡或其他数学表征；
- Stage2-C新增的target-new support IQ、标签、注册表和同received-IQ合法表征；
- K、scene及不包含query真值的数据无关冻结配置；receiver/TX标签不得作为target拟合或决策输入。

候选不要求机械叠加所有资产，但方法卡必须列出`legal_asset_inventory`、`assets_used`、`assets_not_used_and_reason`，防止在已有domain branch、地面压缩知识或适配先验可用时无依据地退化为只调分类头。任何新Phase1 encoder、domain-factorized branch或bundle只要使用合法Phase1数据、在target访问前冻结并接受matched比较，也可进入研发。

允许探索但不限于：

- MRIOR目标的轻量化版本；
- ADV3B02末端block或关键卷积的LoRA/低秩adapter；
- support-conditioned FiLM、normalization scale/bias或轻量hypernetwork系数；
- 微型IQ residual frontend或receiver-response correction；
- Phase1预训练的domain-factorized adapter，由地面压缩知识和target support只估计少量系数；
- 在合法Phase1数据上重新训练、target访问前冻结的新encoder/bundle。

还允许低秩RDA/SRDA、receiver nuisance basis correction、support-conditioned metric、双qKNN类内条件化、核方法、闭式概率状态和混合专家，只要最终仍满足统一全类逐query竞争并证明真实DA收益。

完整解冻基底模型不是默认路线。任何大范围更新必须证明其资源和K-shot可辨识性优于轻量替代方案。

### 2.2qKNN统一分类主干

模型适应完成后，必须用适应后的encoder重新提取当前row old/new support表征并建立qKNN状态。qKNN负责：

- sample-level局部多峰邻域；
- 全部实际注册旧类和新类的同式逐query竞争；
- 按类归一化证据，避免support数或原型数产生裸优势；
- INT8 support bank、量化尺度、identity/fallback和逐样本argmax；
- 禁止query角色、真实batch类数、类别quota、Hungarian/OT全局重排和跨query图。

OTHER不得以第二个分类头替代qKNN。若使用线性、概率或ridge输出，只能作为qKNN残差、校准、先验或可靠度证据，且qKNN在最终决策中的权重和作用必须可审计。

### 2.3互补机制OTHER

每个revision最多增加一个主要OTHER，且必须只针对一个明确剩余误差：

- 旧类反遗忘；
- 新类注册竞争；
- old/new分数尺度失衡；
- prototype hubness或support局部密度；
- support噪声与K不足；
- coverage或类级不确定性；
- INT8 margin与量化风险。

可选OTHER包括RDA/SRDA、typed metric、BCRR、class-balanced calibration、旧类锚定、局部密度校正、Bayesian shrinkage和不确定性收缩。OTHER必须使用类标签置换等价的统一公式，不能按具体TX/class ID设置分支、阈值或权重。

## 3.Stage2-B/C状态机

同一row必须显式形成三个不可混淆的适应状态：

```text
S0 = immutable Phase1 deployment bundle
S_B = FitOrAdapt(S0, optional_ground_state, target_old_support)
S_C = AppendOrContinue(S_B, target_old_support, target_new_support)
```

### 3.1Stage2-B

Stage2-B只允许读取：Phase1 bundle、候选明确使用的共同封存知识和`Y_old`的K-shot目标域support。任务是适应未见目标receiver并评估旧类。

必须使用同一旧类query和同一qKNN规则报告：

- `S0＋qKNN`的direct/identity结果；
- `S_B＋qKNN`的适应结果；
- old adaptation gain；
- 相对MRIOR-SDA、JG_R8_LR020和当前最强合法Stage2-B基线的matched差值；
- K1/K5/K10及receiver/scene/seed/逐类结果。

### 3.2Stage2-C

Stage2-C新增`Y_new`的K-shot有标签support。候选必须在设计冻结前选择并锁定一种状态更新方式：

1.冻结`S_B`的DA状态，只编码并append新类support到统一qKNN；或
2.以old/new每类等权support执行一次预注册的有界继续适应，形成`S_C`后重建统一qKNN。

若选择继续适应，必须使用全部注册类相同的loss和更新规则；可以知道注册状态，但query预测不得读取真实old/new角色。新类不能获得地面预置身份原型，Phase1知识只能通过共享domain先验、旧类锚定或类无关适应结构帮助新类。

Stage2-C必须在同一row报告注册前old、注册后old、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new和new→old；不得用Stage2-B旧类提升替代Stage2-C成功。

### 3.3K1/K5/K10可辨识性

K1不得估计类内散度、类专属高维协方差或无约束模型参数。若候选的DA自由度在K1下不可辨识，必须精确identity或按Phase1预锁规则强收缩；K1用于验证安全回退，不要求伪造DA收益。

K5是当前双qKNN候选的首个正式DA falsifier；K10用于确认相同机制，不得增加rank、改变`alpha`、重新选择kernel或放宽fallback。Stage2-C的新增类只能按冻结规则append，不得重拟合Stage2-B的旧类域basis、旧bank前缀或收缩强度。

## 4.Phase1压缩知识的可选作用与边界

允许的ground状态只能是target访问前由多个独立Phase1物理样本聚合、INT8量化并与checkpoint共同封存的模型知识。它可以包含旧类多中心、domain×class聚合中心、半径、低秩domain basis、聚合方差或adapter先验，但不得包含raw/clean IQ、单样本feature、成员ID、可逆索引、source replay或独立可替换sidecar。

若正式候选使用地面旧类压缩多原型，必须由Phase1 bundle与method lock共同声明并锁定`bundle_id`、聚合规则、INT8格式、尺度、权重和半径。缺少该组件时不得在Phase2回读地面样本或临时重建；不使用ground的合法target-support DA仍可作为独立候选，不再强制增加`M_DA_NG`臂。

地面知识优先用于：

- 预训练并限制星上adapter/LoRA的更新子空间；
- 为K1提供类无关receiver/domain先验；
- 约束旧类表征漂移和关系保持；
- 估计support可靠度、收缩强度或模型更新信赖度；
- 为新类提供共享域校正，而不是预置新类身份知识。

默认禁止把ground旧类原型直接作为额外旧类query score，因为D19及后续实验表明该做法容易压制新类。若重新探索直接评分，必须提交`REENTRY_CARD`并以ground-off、target-only和old/new交换证据直接证伪。

现有84-cell地面组件只具有约`D_eff=2.139–4.302`的有效域信息，历史target偏移coverage约`0.144–0.227`。候选不得再用低coverage ground span强制搬动完整坐标系；应把ground知识放入低维模型更新先验、coverage收缩或可靠度控制，并保留target support主导权。

## 5.当前证据与研发起点

### 5.1可复用正信号

- `MRIOR-SDA`和`JG_R8_LR020`证明模型级适应可改善目标域old，但都缺少完整统一旧类/新类协同证据，保留为matched reference而非唯一主路线。
- D92证明RDA实例可改善old、floor和forgetting，但会损害seen-new；新DA必须避免old/new交换。
- BCRR在K5 held四臂中相对M0取得old-after`+0.012098`、seen-new`+0.011408`、H`+0.017067`、floor`+0.056410`并使forgetting降低`0.009093`；96次wrong→correct、18次correct→wrong，净`+78`。它是当前可复用OTHER正信号，不构成DA或联合成功。
- 真实ADV3B02 support-only探针确认同SHA checkpoint可在head-bypass路径输出有限`z_id/z_dom[*,160]`；对`z_dom`做类中心残差与TX抑制后，K5/K10可形成rank≤2非均匀域邻域，K1精确identity。该证据只证明可实现性，不是性能结果。

### 5.2已证伪边界

- R2A/RCHM虽改变metric或logit，但可能几乎不改变邻居、argmax或净正确决策；数值变化不能冒充DA成功。
- C-id只有净正确`+1`，同时损害floor和forgetting；极小metric收益不能晋级。
- D93/D94的ground→target全坐标transport在coverage不足时全面负迁移；不能继续提高rank或变换强度。
- D81真实读取84个ground cell，但K1严格恒等；地面知识只做support中心可靠度不足以实现单shot模型适应。
- D62证明hard gate和大面积fallback会让方法实际失效；优先采用连续收缩、条件数限制和渐进identity回退。
- 原始`z_dom`具有明显TX泄漏，禁止直接双余弦跨类融合、第二domain分类头或按TX/receiver专属规则决策。
- support accuracy、重构RMSE、LODO正信号、模型参数变化、代码测试或进程exit0都不是held性能成功。

### 5.3当前唯一下一候选

`ADV3B02-TS-DRQKNN-BCRR/r1`在实现终审和真实checkpoint support-only检查中暴露两项P0：after INT8审计用decoded-old代替完整FP32 teacher，且125调度一次性提交全部row、不能在系统性技术故障时立即停派；共享对称INT8 qKNN还在seed713104 after clear/low-elev触门。r1没有N607 prediction或性能结果，现为`SUPERSEDED_TECHNICAL_REVISION / NO_PERFORMANCE_RESULT`。

当前`ADV3B02-TS-DRQKNN-BCRR/r2-affine`已完成一个新设计波次并由独立监督裁定`MERGE / P0=0 / P1=0`，状态为`DESIGN_FROZEN`。它保持r1的DA、双注册、qKNN、BCRR和四臂因果结构，只把四臂共同support codec升级为固定逐向量仿射INT8，并把量化门实现纠正为本文件既有的`top1>=99.5%`且large-margin flip为0；同时冻结完整FP32 after-teacher和有界增量派发健康退出。

冻结的域状态使用target-old support构造`S_W-S_B`的固定2槽可靠方向；support与query对每个候选类都减同一`mu_c`，不得混用全局中心；`alpha_K=0.5*(K-1)/K*(rho_1+rho_2)/2`且`0<=alpha<0.5`。`z_dom`只形成类内权重，最终score必须复用基础`z_id`Student-t qKNN的同一INT8 bank、`h_c`、`nu`和kernel。K1或数值异常时逐值回到M0。

BCRR是唯一OTHER：raw与dual branch分别用自身同步physical-ID support-LOO logits按同一冻结规则拟合`omega`，但共享同一`z_id`BCR状态；不得读取query或直接读取`z_dom`。仿射codec只读support，固定保存INT8 codes、FP16 scale和FP16 offset，每条support相对单scale增加2B；不得使用query、truth、角色、quota或scene专属codec。现有DSSC、RDA/SRDA、RBSC、C-id、MRIOR和JG保留为普通matched reference或后续候选资产，不与本revision混塞。

## 6.方法卡与可行性门

任何正式方法代码修改前必须按顺序完成：

```text
DESIGN_DRAFT -> FEASIBILITY_REVIEW -> DESIGN_FROZEN -> IMPLEMENTING
```

每个候选方法卡必须包含：

```text
candidate_id / revision
domain_adaptation_mechanism
base_model_and_optional_trainable_blocks
legal_asset_inventory
assets_used
assets_not_used_and_reason
stage2b_adaptation_state_transition
stage2c_adaptation_state_transition
target_old_and_new_support_usage
qknn_decision_rule
additional_complementary_mechanism
why_da_qknn_other_are_complementary
K1/K5/K10_identifiability
decision_geometry_change
old_new_forgetting_protection
resource_and_int8_lifecycle
minimal_falsifier_and_fallback
files_interfaces_and_dependencies
```

可行性讨论必须在一个设计波次内回答：

1.数据、domain branch、可选ground、support、query和状态读写是否合法；
2.K1/K5/K10下拟合自由度是否可辨识；
3.适应是否真正改变表示、邻居贡献、margin或argmax，而非被共同变换或完整重估抵消；
4.Stage2-B适应如何迁移到Stage2-C，新增类是否导致旧类漂移；
5.DA解决的域偏移与OTHER解决的剩余分类误差是否互补；
6.coverage不足、support噪声、old/new冲突和量化误差时如何连续收缩或回退；
7.训练参数、step、state、MAC、时延、显存和optimizer清理是否可部署；
8.需要修改的文件、接口和依赖是否闭合；
9.什么最小结果会立即证伪该revision。

监督只输出`MERGE/REVISE/REJECT`。代码修改前向用户报告不超过20行的可行性摘要；除新增数据权限、科学场景或高影响操作外，不把摘要变成等待确认的新阻塞门。

## 7.最小因果证据包

qKNN作为共同分类底座。可拆分候选默认使用以下状态：

|状态|域适应|qKNN|OTHER|用途|
|---|---|---|---|---|
|`M0`|关闭|基础`z_id`qKNN|关闭|共同基准|
|`M_DA`|开启|候选DA＋统一qKNN|关闭|证明真实域适应贡献|
|`M_OTHER`|关闭|基础qKNN|开启|隔离剩余误差机制|
|`M_JOINT`|开启|与M_DA逐字节共享DA/qKNN state|开启|检验联合与交互|

天然耦合候选可使用parameter freeze、stop-gradient、loss masking或matched surrogate，但不得为凑臂数制造无信息实验。当前双qKNN revision固定四臂，不增加`M_DA_NG`或第二分类头。

主协同量固定为：

```text
I_syn = H(M_JOINT) - H(M_DA) - H(M_OTHER) + H(M0)
```

其中：

- `M_DA>M0`证明候选域适应本身有效；
- `M_OTHER>M0`证明互补机制独立有效；
- `M_JOINT>max(M_DA,M_OTHER)`且`I_syn>0`证明联合协同。

machine receipt必须绑定DA state、support physical ID、qKNN bank、OTHER状态、随机性、capsule/split/row/seed和query policy；若存在训练，还必须绑定trainable block和optimizer schedule。M_DA与M_JOINT必须复用相同DA/qKNN state，不能分别拟合后挑选有利版本。

## 8.数据协议硬边界

Phase2只能读取：immutable Phase1 deployment bundle、匹配`VALIDATED_ONCE`的固定单LEO弱观测capsule、当前row合法target-old/target-new K-shot support及标签、与数据无关的冻结配置。

必须保持：

- 一个物理IQ只有一次允许的LEO弱信道观测；K-shot是K个独立物理support；
- 三个场景物理ID互斥，单场景support/query物理ID互斥；
- 不访问clean/raw/source样本、sample-level source feature、source replay或可替换sidecar；
- query不参与训练、loss、early stop、模型选择、温度、回退或状态更新；
- 禁止query伪标签、熵最小化、图、OT/Hungarian、quota、角色Oracle和batch reassignment；
- 若使用ground知识，它只能来自target访问前共同封存的INT8多样本聚合组件，不增加K；
- Stage2-C新类没有ground身份原型，只能通过共享DA规则和自身support注册；
- target生成的DA state、qKNN bank和OTHER状态必须进入正式INT8/FP16-scale生命周期，无常驻FP32 sidecar；
- optimizer、gradient、momentum和训练临时量在适应完成后删除，不计入query持久state但必须计入训练峰值资源。

匹配的`capsule_id/split_id/schema=p2_min_v1/VALIDATED_ONCE`只核对一次。只有received IQ字节、physical ID、receiver/TX集合、scenario、K、support/query划分或schema改变时重验；方法、adapter、loss、head、超参数、checkpoint推理状态、bundle、资源或报告变化不得触发数据重建。

## 9.性能晋级门

### 9.1Stage2-B域适应门

- K5/K10的`M_DA`相对M0必须产生净正确决策正收益，old/new净变化均不得为负；
- 必须产生可观测的domain neighbor、identity contribution、margin或argmax变化，不能只有logit漂移；
- 必须报告相对direct ADV3B02、identity qKNN、DSSC和其他可用matched reference的同row差值；
- receiver和scene分层不得由单一有利slice主导；
- K1按冻结合同精确identity，`M_DA=M0`且`M_JOINT=M_OTHER`，只验证安全回退。

### 9.2Stage2-C联合门

K10完整确认硬门：

- `old_acc_after_increment>=92%`；
- `min_old_class_acc>=88%`；
- `seen_new_acc(new5)>=92%`；
- `seen_new_acc(new10)>=90%`；
- `seen_new_acc(new20)>=86%`。

同时要求：

- `M_DA`与`M_OTHER`相对`M0`均有净正确决策正收益，old/new净变化各自不负；
- `M_JOINT`的old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old和min-new均不低于两个单组件；
- `M_JOINT`的forgetting、old→new和new→old均不高于两个单组件；
- mean`I_syn>0`，首次完整125中正协同至少覆盖188/375个scene slice，至少2/3个scene均值为正；
- 最终确认中paired mean`I_syn>0`且95% CI下界大于0。

### 9.3完整报告

每个完成候选必须报告同一row的：注册前old、注册后old、old adaptation gain、seen-new、H、BA、全部注册类floor、min-old、min-new、forgetting、old→new、新→old、逐类、receiver、scene、K和seed结果；同时报告DA coverage、feature/domain-neighbor/identity-contribution变化、support fit、held query、量化margin、MAC、时延、显存、state bytes和optimizer step。若候选使用ground或训练，再报告ground-on/off和模型参数变化。

不得只报平均值、只说明缺陷、拼接不同run极值，或把support fit、重构误差、代码测试和进程启动当成性能成功。

## 10.星上资源与量化门

默认硬门：

- trainable parameters`<=80000`；
- adaptation epochs`<=30`；
- optimizer steps`<=50`；
- optimizer清理后的persistent incremental state`<=256KB`；
- dense query graph=`false`；
- query-dependent batch optimization=`false`；
- 正式INT8 top1一致率`>=99.5%`；
- large-margin flip=`0`。

每个DA还必须报告build/fit wall time、计算MAC、峰值显存、可训练参数、optimizer step和持久state；相对可用matched reference给出同口径资源差值。闭式0参数候选也必须报告实际双分支forward增量，不能把参数为0等同于计算免费。

每个适应run必须分别记录训练期峰值资源和部署期持久资源。INT8/FP16-scale adapter从序列化bytes反解后必须复算模型输出、qKNN top1和margin；不能以未量化teacher结果替代部署结果。

## 11.高效研发与实验顺序

1.完整读取初始化栈、当前唯一活动报告和Git状态，输出不超过20行上下文卡。
2.根据候选结构并行形成2–4张互不重复的方法卡和1份联合监督结论；模型、metric、概率或双qKNN候选按同一DA证据门公平审查。
3.一个设计波次后只冻结一个优先候选。设计冻结后只实现一个主要机制delta。
4.本地在`ssr-gpu`完成专项测试、协议负例、真实checkpoint无query smoke、INT8部署等价、diff review和Git提交；只在机制需要时增加ground-off或parameter-freeze消融。
5.允许使用Phase1 LODO/LOCO、source validation和合法held proxy冻结模型结构与超参数；这些代理不得替代target性能，也不得读取target query选参。
6.每个冻结候选、每个revision的任何正式N607性能发布都必须直接运行完整125：`5 receivers×5 seeds×{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每job覆盖3个LEO弱场景。当前双qKNN候选同一commit输出`M0/M_DA/M_OTHER/M_JOINT`，闭合`125 jobs/375 scene slices/1500 score rows/1000 arm-state prediction artifacts`。不得先发布单receiver、单seed、单K、单scene或其他有利子集；局部入口只能用于本地专项、协议负例和真实checkpoint无query smoke。
7.每次完整125只能验证一个已冻结revision，不得用于选择层、rank、loss、step、ground格式、OTHER、量化、阈值或fallback。任何机制变化必须创建新revision、重新审查并以新的不可覆盖run ID重新执行完整125；不同revision不得拼接结果。
8.首次完整125取得联合正收益后，只能以预注册的新seed和全新run ID运行另一份完整125确认；不得用第一次125选择结构、rank、`alpha`、量化、阈值或fallback。
9.每次正式性能发布都必须把冻结矩阵按不可变row/job ID确定性分片，并通过共享动态队列调度到N607的GPU0–7；在8张GPU均可安全使用时，每张卡至少分配一个worker，尽量让8卡同时工作，并通过最长任务优先或等价负载均衡减少尾部空转。
10.发布前记录8张GPU的已有进程、显存和可用slot。默认每卡最多2个训练进程；不得杀死、暂停或迁移无关任务。若部分GPU已满，只使用其余安全slot并排队等待，不得为追求8卡占用而超配，也不得因此缩窄正式矩阵。
11.同一run ID只有一个实验runner负责preflight、精确同步、远端校验、启动、实验健康检查、短连接监控和artifact回收。runner必须在报告中记录逐GPU的job分配、并发上限、启动/结束时间、利用率或可观测替代量、失败重排和尾部空闲原因；单job失败只按冻结retry规则重入队列，不能改变方法或参数。
12.完整125是性能证据矩阵，不是要求技术故障自然跑完。正式启动后必须先执行首波健康检查：只读取PID/parent-child/CWD/cmdline、GPU利用率与显存、row exit、异常指纹、prediction/score数量和artifact闭合，不得读取准确率或据性能早停。若任一P0协议/安全错误发生，或至少2个不同row在没有prediction时出现相同确定性异常指纹，runner必须立即停止继续派发，核对进程归属后终止且仅终止本run进程，确认GPU释放与SSH残留为0，并回收partial日志及失败row；不得等待其余row自然失败。技术修复必须经过本地专项测试、独立P0/P1 review、Git提交和全新不可覆盖run ID，禁止原run续跑或覆盖。
13.服务器runner执行上一revision期间，主agent继续下一DA候选的只读设计、实现准备和历史复盘，不线性等待N607。

没有完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。因系统性技术故障触发健康止损时同时标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`，记录停止阈值、异常指纹、已启动/完成/失败row和prediction/score数量；该状态不是性能结果。完成prediction但性能未达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，记录被证伪假设后立即进入下一revision；不得在同一revision上根据query结果补丁式调参。

## 12.研发自由、重入与停止边界

历史负结果只否定相同输入、机制和失败条件下的具体实例，不封禁MRIOR、LoRA、FiLM、adapter、IQ前端、meta-learning、ground先验、qKNN或其他方法族。失败路线重新进入必须提交：

```text
REENTRY_CARD
failed_instance_and_evidence
changed_assumption_or_mechanism
why_previous_failure_no_longer_applies
minimal_differentiating_test
falsifier_and_stop_condition
additional_cost
```

默认停止以下无效工作：

- 只有metric、协方差、prototype或score数值变化，却没有邻居贡献、argmax或净正确决策变化；
- 用旧类Stage2-B正收益替代Stage2-C新类注册和反遗忘；
- K1整体identity fallback后仍声明单shot适应；
- ground旧类原型直接加分并压制新类；
- 在低coverage下强制高rank ground→target全坐标transport；
- 用support loss、100% support accuracy、LODO单点或重构RMSE晋级；
- 用125、Role-Oracle、development query或confirmation query反向选参；
- 重复建设已`VALIDATED_ONCE`的数据、hash、allowlist、authority或准入系统；
- 多个agent修改同一方法文件、启动同一run ID或由方法作者自我认证。

如果连续3个完成revision都未使DA单组件产生正净正确决策，必须做一次记录化复盘：区分表示信息不足、domain branch泄漏、K-shot不可辨识、适应机制与held泛化不一致或资源限制过强，再决定修改DA族、训练目标或Phase1 bundle。该复盘不授权访问新数据或用query调参。

## 13.完成条件

只有以下证据全部存在并通过，才能标记目标完成：

- 域适应方法卡、可行性审查和`DESIGN_FROZEN`；
- Stage2-B和Stage2-C状态机及联合因果证据；
- K1安全回退、K5首证伪与K10确认；
- DA单组件在receiver/scene分层上产生真实净正确决策正收益；
- Stage2-C绝对性能门、floor、forgetting和双向混淆门；
- `M_JOINT`严格优于单组件且协同CI通过；
- 至少一份完整125性能验证及预注册完整125确认；
- DA state、qKNN和OTHER的INT8生命周期；
- fit/build期与部署期资源审计；
- 完整日志、报告、复现命令和Git提交。

未达到上述条件时，研发目标保持开放。每个负结果必须形成可证伪结论和下一单一机制假设，继续实验推进，而不是以代码完成、资源达标或局部平均值结束研发。
