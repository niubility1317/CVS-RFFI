# Stage2地面压缩知识驱动的星上快速模型域适应＋qKNN＋互补机制研发目标

版本：2026-07-23
修订：模型级快速域适应、Stage2-B/C状态机、地面压缩原型、K1正收益、MRIOR超越、联合协同与N607八卡并行发布
状态：可直接作为新`/goal`目标Prompt
协议：`protocol_schema=p2_min_v1`
初始化文档：`docs/STAGE2_RESEARCH_AGENT_INIT.md`

## 1.单一目标

在`E:\type10-7`中，严格遵循实时`AGENTS.md`、`项目.md`和`p2_min_v1`，研发可在星上稀缺计算资源下运行的Stage2快速模型域适应方法。候选必须同时利用：

1.在任何target访问前与Phase1 checkpoint共同封存的INT8多样本聚合地面旧类知识；
2.Stage2-B提供的少量有标签目标域旧类support；
3.Stage2-C新增的少量有标签目标域新类support。

主路线固定为：

```text
地面压缩知识驱动的快速模型域适应
＋qKNN统一旧类/新类分类
＋一个针对剩余误差的互补机制
```

域适应必须训练或调节基底模型的实际推理路径，使encoder、输入前端、normalization或轻量adapter适应Phase1未见目标接收机。只修改距离metric、协方差、分类头、prototype、logit或score，不单独计作本目标所称的“模型域适应”；RDA/SRDA、C-id、RCHM、SVRN、RBSC和BCRR可以作为OTHER、正则、先验或对照，但不能替代快速模型适应主线。

最终目标是：

- Stage2-B的K10旧类目标域适应显著优于同row matched`MRIOR-SDA`；
- K1总体产生正适应收益，且每个目标receiver不出现负适应；
- Stage2-C加入新类后同时保护旧类、提高seen-new、H、floor和最差类，降低forgetting及双向混淆；
- `M_JOINT`严格优于模型DA单组件和OTHER单组件，证明`1+1>2`；
- 在模型参数、optimizer step、持久state、MAC、时延、显存和INT8生命周期上满足星上部署约束；
- 持续以新revision实验推进，直到达到本文件性能与资源门，或得到足以证明当前Phase1 bundle/encoder信息不足的可复现实验证据。

工作重心是方法、实现和真实N607性能。不得把大部分时间消耗在已经`VALIDATED_ONCE`的数据重验、authority/hash重建、跨run opaque SHA对齐、控制面扩展、报告格式重构或无关文献综述上。

## 2.三个模块的严格定义

### 2.1快速模型域适应

模型DA必须满足以下定义：

- 使用当前row合法support产生新的模型参数、低秩delta、条件化参数或模型内部统计；
- 该状态进入query的encoder或输入推理路径，而不是只进入最终分类score；
- 适应后模型冻结，query只能逐样本前向，不能继续训练、选择、回滚或更新；
- 必须报告更新层、冻结层、trainable parameters、loss、step、适配时延、参数delta、表征变化和邻居变化；
- 参数发生变化、support loss下降或support accuracy提高都不等于成功，必须转化为held query净正确决策和正式指标收益。

模型DA的信息面保持开放。任何由`项目.md`允许、能在Phase2合法读取或由当前固定received IQ合法计算的信息都应进入候选审查范围，包括：

- checkpoint内部identity/domain分支、`z_id`、`z_dom`及合法中间表征；
- 与checkpoint共同封存的domain classifier、domain basis、normalization统计和adapter/meta-learning先验；
- INT8 domain×class聚合中心、地面压缩多原型、半径、聚合方差、低秩残差和量化尺度；
- Stage2-B的target-old support IQ、标签、注册表和由同一received IQ计算的FFT、均衡或其他数学表征；
- Stage2-C新增的target-new support IQ、标签、注册表和同received-IQ合法表征；
- receiver/domain handle、K、scene及不包含query真值的数据无关配置，但不得据具体receiver/TX建立专属规则。

候选不要求机械叠加所有资产，但方法卡必须列出`legal_asset_inventory`、`assets_used`、`assets_not_used_and_reason`，防止在已有domain branch、地面压缩知识或适配先验可用时无依据地退化为只调分类头。任何新Phase1 encoder、domain-factorized branch或bundle只要使用合法Phase1数据、在target访问前冻结并接受matched比较，也可进入研发。

允许探索但不限于：

- MRIOR目标的轻量化版本；
- ADV3B02末端block或关键卷积的LoRA/低秩adapter；
- support-conditioned FiLM、normalization scale/bias或轻量hypernetwork系数；
- 微型IQ residual frontend或receiver-response correction；
- Phase1预训练的domain-factorized adapter，由地面压缩知识和target support只估计少量系数；
- 在合法Phase1数据上重新训练、target访问前冻结的新encoder/bundle。

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

同一row必须显式形成三个不可混淆的模型状态：

```text
S0 = immutable Phase1 deployment bundle
S_B = Adapt(S0, ground_aggregate, target_old_support)
S_C = ContinueOrRefit(S_B, ground_aggregate, target_old_support, target_new_support)
```

### 3.1Stage2-B

Stage2-B只允许读取：Phase1 bundle、共同封存的地面压缩旧类知识和`Y_old`的K-shot目标域support。任务是快速适应未见目标接收机并评估旧类。

必须使用同一旧类query和同一qKNN规则报告：

- `S0＋qKNN`的direct/identity结果；
- `S_B＋qKNN`的适应结果；
- old adaptation gain；
- 相对MRIOR-SDA、JG_R8_LR020和当前最强合法Stage2-B基线的matched差值；
- K1/K5/K10及receiver/scene/seed/逐类结果。

### 3.2Stage2-C

Stage2-C新增`Y_new`的K-shot有标签support。候选必须在设计冻结前选择并锁定一种状态更新方式：

1.冻结`S_B`，只用适应后的encoder编码新类support并注册qKNN；或
2.以old/new每类等权的support执行一次有界继续适应，形成`S_C`，随后冻结模型并重建统一qKNN。

若选择继续适应，必须使用全部注册类相同的loss和更新规则；可以知道注册状态，但query预测不得读取真实old/new角色。新类不能获得地面预置身份原型，地面知识只能通过共享domain先验、旧类锚定或类无关适应结构帮助新类。

Stage2-C必须在同一row报告注册前old、注册后old、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new和new→old；不得用Stage2-B旧类提升替代Stage2-C成功。

### 3.3K1可辨识性

K1不能默认回退identity。方法必须说明地面压缩知识如何提供单shot无法估计的先验，例如：

- receiver/domain nuisance basis；
- adapter参数低维子空间；
- normalization或FiLM先验；
- 旧类关系、半径或协方差收缩；
- meta-learned的一步更新方向。

K1不得估计类专属高维协方差或无约束全模型参数。若某个参数在K1下不可辨识，必须由Phase1预锁规则确定、强收缩或删除，而不是把整个模型DA关闭。

## 4.地面压缩知识的作用与边界

允许的ground状态只能是target访问前由多个独立Phase1物理样本聚合、INT8量化并与checkpoint共同封存的模型知识。它可以包含旧类多中心、domain×class聚合中心、半径、低秩domain basis、聚合方差或adapter先验，但不得包含raw/clean IQ、单样本feature、成员ID、可逆索引、source replay或独立可替换sidecar。

使用地面旧类压缩多原型的正式候选必须由新的Phase1 bundle与method lock同时声明`ground_old_multiprototype_enabled=true`，并锁定`bundle_id`、每类原型上限、聚合规则、INT8格式、尺度、权重和半径。缺少该合规组件时不得在Phase2回读地面样本或临时重建；该候选应标记为bundle依赖未满足并转入新bundle的Phase1冻结流程，现有bundle的target-only路线只能作为`M_DA_NG`对照，不能冒充完成了本目标的ground驱动模型DA。

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

- `MRIOR-SDA`在25个严格K10 Stage2-B matched row上的old均值为`84.5000%`，高于direct ADV3B02的`75.2111%`和identity qKNN的`77.6333%`。它是模型DA性能reference，但尚无新类联合证据，且计算开销高。
- `JG_R8_LR020`只训练6400个低秩参数、5epoch、最多50step，适配约1.3385s/row；old为`78.8222%`，相对identity qKNN提升`1.1889pp`，95% CI为`[0.6218,1.7560]pp`。它证明轻量星上模型适应能够产生真实收益，但floor仅`55.0667%`且没有Stage2-C证据。
- D92在K10/new20上相对D81提高注册后old`2.622pp`、min-old`4.600pp`、H`0.964pp`并降低forgetting`2.622pp`，但seen-new下降`0.653pp`。该结果只作为old/new竞争的OTHER经验，不计模型DA成功。
- BCRR在K5 held四臂中取得净正确`+78`，但SVRN模型分支18/18行identity、`I_syn=0`。BCRR可复用为OTHER，不能替代模型适应。

### 5.2已证伪边界

- RCHM虽改变logit但6630个argmax零变化；共同metric变化不能冒充模型DA。
- C-id只有净正确`+1`，同时损害floor和forgetting；极小metric收益不能晋级。
- D93/D94的ground→target全坐标transport在coverage不足时全面负迁移；不能继续提高rank或变换强度。
- D81真实读取84个ground cell，但K1严格恒等；地面知识只做support中心可靠度不足以实现单shot模型适应。
- support accuracy、重构RMSE、LODO正信号、模型参数变化、代码测试或进程exit0都不是held性能成功。

### 5.3当前候选重分类

`RBSC-TM-qKNN-BCRR`、RDA/SRDA、C-id、RCHM和SVRN归入metric/OTHER/reference线。它们可以完成已冻结的诊断或作为联合组件，但不得阻塞下一波快速模型DA候选，也不得被报告为满足本目标的DA主线。

下一设计波次优先比较：

1.`Ground-MRIOR-Lite-qKNN`：保留MRIOR有效目标，将更新限制在末端LoRA/adapter，并用ground关系或domain basis约束；
2.`Ground-JG-qKNN/r1`：复用JG轻量骨架，改为class-balanced/worst-group support目标并加入旧类表征锚定；
3.`Ground-FiLM-qKNN`：由ground domain basis和target support估计少量FiLM/normalization参数；
4.`Ground-IQAdapter-qKNN`：冻结ADV3B02，只训练微型接收机响应residual frontend。

至少形成2张方法卡，经一次可行性波次只冻结一个优先模型DA候选。

## 6.方法卡与可行性门

任何正式方法代码修改前必须按顺序完成：

```text
DESIGN_DRAFT -> FEASIBILITY_REVIEW -> DESIGN_FROZEN -> IMPLEMENTING
```

每个候选方法卡必须包含：

```text
candidate_id / revision
base_model_and_exact_trainable_blocks
legal_asset_inventory
assets_used
assets_not_used_and_reason
stage2b_adaptation_state_transition
stage2c_adaptation_state_transition
ground_aggregate_usage
target_old_and_new_support_usage
adaptation_loss_and_update_schedule
qknn_decision_rule
additional_complementary_mechanism
why_model_da_qknn_other_are_complementary
K1/K5/K10_identifiability
old_new_forgetting_protection
resource_and_int8_lifecycle
matched_mrior_comparison
minimal_falsifier_and_fallback
files_interfaces_and_dependencies
```

可行性讨论必须在一个设计波次内回答：

1.数据、domain branch、ground、support、query和状态读写是否合法，是否遗漏了可产生互补证据的合法资产；
2.K1/K5/K10下可训练自由度是否可辨识；
3.适应是否真正改变encoder表征、邻居顺序、margin或argmax，而非被qKNN共同变换不变性抵消；
4.Stage2-B适应如何迁移到Stage2-C，新增类是否导致旧类漂移；
5.ground先验解决的误差、模型DA解决的误差和OTHER解决的误差是否互补；
6.coverage不足、support噪声、old/new冲突和量化误差时如何连续收缩或回退；
7.训练参数、step、state、MAC、时延、显存和optimizer清理是否可部署；
8.需要修改的文件、接口和依赖是否闭合；
9.什么最小结果会立即证伪该revision。

监督只输出`MERGE/REVISE/REJECT`。代码修改前向用户报告不超过20行的可行性摘要；除新增数据权限、科学场景或高影响操作外，不把摘要变成等待确认的新阻塞门。

## 7.最小因果证据包

qKNN作为共同分类底座。可拆分候选默认使用以下状态：

|状态|模型适应|ground知识|qKNN|OTHER|用途|
|---|---|---|---|---|---|
|`M0`|关闭|不进入target更新|基础qKNN|关闭|冻结encoder基准|
|`M_DA_NG`|开启|关闭|同一qKNN|关闭|目标support-only模型适应|
|`M_DA`|开启|开启|同一qKNN|关闭|证明ground＋support模型适应贡献|
|`M_OTHER`|关闭|不进入target更新|同一qKNN|开启|隔离互补机制|
|`M_JOINT`|开启|开启|同一qKNN|开启|检验联合与交互|

若ground已内生于Phase1 adapter而不能在不改变模型的情况下关闭，必须预注册matched surrogate、parameter freeze、loss masking或stop-gradient干预。不得伪造无意义对照。

主协同量固定为：

```text
I_syn = H(M_JOINT) - H(M_DA) - H(M_OTHER) + H(M0)
```

其中：

- `M_DA_NG>M0`证明target support快速模型适应本身有效；
- `M_DA>M_DA_NG`证明地面压缩知识提供额外价值；
- `M_OTHER>M0`证明互补机制独立有效；
- `M_JOINT>max(M_DA,M_OTHER)`且`I_syn>0`证明联合协同。

machine receipt必须绑定适应前后模型state、trainable block、ground component、support physical ID、optimizer schedule、qKNN bank、OTHER状态、随机性、capsule/split/row/seed和query policy。模型DA与JOINT必须复用相同适应state，不能分别训练后挑选有利版本。

## 8.数据协议硬边界

Phase2只能读取：immutable Phase1 deployment bundle、匹配`VALIDATED_ONCE`的固定单LEO弱观测capsule、当前row合法target-old/target-new K-shot support及标签、与数据无关的冻结配置。

必须保持：

- 一个物理IQ只有一次允许的LEO弱信道观测；K-shot是K个独立物理support；
- 三个场景物理ID互斥，单场景support/query物理ID互斥；
- 不访问clean/raw/source样本、sample-level source feature、source replay或可替换sidecar；
- query不参与训练、loss、early stop、模型选择、温度、回退或状态更新；
- 禁止query伪标签、熵最小化、图、OT/Hungarian、quota、角色Oracle和batch reassignment；
- ground知识只来自target访问前共同封存的INT8多样本聚合组件，不增加K；
- Stage2-C新类没有ground身份原型，只能通过共享模型适应和自身support注册；
- target生成的adapter delta、qKNN bank和OTHER状态必须进入正式INT8/FP16-scale生命周期，无常驻FP32 sidecar；
- optimizer、gradient、momentum和训练临时量在适应完成后删除，不计入query持久state但必须计入训练峰值资源。

匹配的`capsule_id/split_id/schema=p2_min_v1/VALIDATED_ONCE`只核对一次。只有received IQ字节、physical ID、receiver/TX集合、scenario、K、support/query划分或schema改变时重验；方法、adapter、loss、head、超参数、checkpoint推理状态、bundle、资源或报告变化不得触发数据重建。

## 9.性能晋级门

### 9.1Stage2-B模型DA门

- K10`M_DA`或`M_JOINT`的old accuracy必须严格高于matched MRIOR-SDA；
- paired mean差值必须大于0且95% CI下界大于0；
- 相对direct ADV3B02、identity qKNN和JG_R8_LR020均报告同row差值；
- K1总体old adaptation gain必须大于0，每个receiver均不小于0；
- K1相对direct ADV3B02至少`+2pp`且paired 95% CI下界大于0；
- K5核心指标相对matched K10下降不得超过`3pp`。

### 9.2Stage2-C联合门

K10完整确认硬门：

- `old_acc_after_increment>=92%`；
- `min_old_class_acc>=88%`；
- `seen_new_acc(new5)>=92%`；
- `seen_new_acc(new10)>=90%`；
- `seen_new_acc(new20)>=86%`。

同时要求：

- ground开启相对`M_DA_NG`不损害old、new、floor和forgetting，并在H或floor至少一项严格提高；
- `M_DA`与`M_OTHER`相对`M0`均有净正确决策正收益，old/new净变化各自不负；
- `M_JOINT`的old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old和min-new均不低于两个单组件；
- `M_JOINT`的forgetting、old→new和new→old均不高于两个单组件；
- mean`I_syn>0`，首次完整125中正协同至少覆盖188/375个scene slice，至少2/3个scene均值为正；
- 最终确认中paired mean`I_syn>0`且95% CI下界大于0。

### 9.3完整报告

每个完成候选必须报告同一row的：注册前old、注册后old、old adaptation gain、seen-new、H、BA、全部注册类floor、min-old、min-new、forgetting、old→new、新→old、逐类、receiver、scene、K和seed结果；同时报告ground-on/off、模型参数变化、feature drift、邻居变化、support fit、held query、量化margin、MAC、时延、显存、state bytes和optimizer step。

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

模型DA还必须相对MRIOR报告：适配wall time、训练MAC、训练峰值显存、可训练参数和持久delta。优先目标是在K10 Stage2-B性能显著高于MRIOR的同时，将适配wall time控制在matched MRIOR的25%以内；若硬件或历史artifact无法形成同口径比值，必须报告绝对时间并说明缺口，不能省略资源比较。

每个适应run必须分别记录训练期峰值资源和部署期持久资源。INT8/FP16-scale adapter从序列化bytes反解后必须复算模型输出、qKNN top1和margin；不能以未量化teacher结果替代部署结果。

## 11.高效研发与实验顺序

1.完整读取初始化栈、当前唯一活动报告和Git状态，输出不超过20行上下文卡。
2.并行形成至少2张快速模型DA方法卡、1张qKNN/OTHER卡和1份联合监督结论；metric-only候选不占用模型DA名额。
3.一个设计波次后只冻结一个模型DA主候选。设计冻结后只实现一个主要机制delta。
4.本地在`ssr-gpu`完成专项测试、协议负例、真实checkpoint无query smoke、ground-off消融、INT8部署等价、diff review和Git提交。
5.允许使用Phase1 LODO/LOCO、source validation和合法held proxy冻结模型结构与超参数；这些代理不得替代target性能，也不得读取target query选参。
6.每个冻结候选、每个revision的任何正式N607性能发布都必须直接运行既有完整125稳定性screen：`5 receivers×5 seeds×{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每job覆盖3个LEO弱场景，并在同一commit输出`M0/M_DA_NG/M_DA/M_OTHER/M_JOINT`。不得先发布单receiver、单seed、单K、单scene或其他有利子集；这些局部入口只能用于本地专项测试、协议负例和真实checkpoint无query smoke，不能形成独立N607性能run或方法裁决。
7.每次完整125只能验证一个已冻结revision，不得用于选择层、rank、loss、step、ground格式、OTHER、量化、阈值或fallback。任何机制变化必须创建新revision、重新审查并以新的不可覆盖run ID重新执行完整125；不同revision不得拼接结果。
8.125通过后，以同一commit运行`5 receivers×5 seeds×3 scenes×K{1,5,10,20}×new{2,5,10,20}=1200`评价单元完整确认。
9.每次正式性能发布都必须把冻结矩阵按不可变row/job ID确定性分片，并通过共享动态队列调度到N607的GPU0–7；在8张GPU均可安全使用时，每张卡至少分配一个worker，尽量让8卡同时工作，并通过最长任务优先或等价负载均衡减少尾部空转。
10.发布前记录8张GPU的已有进程、显存和可用slot。默认每卡最多2个训练进程；不得杀死、暂停或迁移无关任务。若部分GPU已满，只使用其余安全slot并排队等待，不得为追求8卡占用而超配，也不得因此缩窄正式矩阵。
11.同一run ID只有一个实验runner负责preflight、精确同步、远端校验、启动、短连接监控和artifact回收。runner必须在报告中记录逐GPU的job分配、并发上限、启动/结束时间、利用率或可观测替代量、失败重排和尾部空闲原因；单job失败只按冻结retry规则重入队列，不能改变方法或参数。
12.服务器runner执行上一revision期间，主agent继续下一模型DA候选的只读设计和历史复盘，不线性等待N607。

没有完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。完成prediction但性能未达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，记录被证伪假设后立即进入下一revision；不得在同一revision上根据query结果补丁式调参。

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

- 把metric、协方差、prototype或score变化重新命名为模型DA；
- 用旧类Stage2-B正收益替代Stage2-C新类注册和反遗忘；
- K1整体identity fallback后仍声明单shot适应；
- ground旧类原型直接加分并压制新类；
- 在低coverage下强制高rank ground→target全坐标transport；
- 用support loss、100% support accuracy、LODO单点或重构RMSE晋级；
- 用125、Role-Oracle、development query或confirmation query反向选参；
- 重复建设已`VALIDATED_ONCE`的数据、hash、allowlist、authority或准入系统；
- 多个agent修改同一方法文件、启动同一run ID或由方法作者自我认证。

如果连续3个完成revision都未使模型DA单组件产生正净正确决策，必须做一次记录化复盘：区分encoder不可适应、ground信息不足、K-shot不可辨识、loss与held泛化不一致或资源限制过强，再决定修改模型DA族、训练目标或Phase1 bundle。该复盘不授权访问新数据或用query调参。

## 13.完成条件

只有以下证据全部存在并通过，才能标记目标完成：

- 快速模型DA方法卡、可行性审查和`DESIGN_FROZEN`；
- Stage2-B和Stage2-C状态机、ground-on/off及联合因果证据；
- K1正收益与逐receiver不负；
- K10 Stage2-B显著优于matched MRIOR-SDA；
- Stage2-C绝对性能门、floor、forgetting和双向混淆门；
- `M_JOINT`严格优于单组件且协同CI通过；
- 完整125稳定性screen和1200单元确认；
- ground、adapter、qKNN和OTHER的INT8生命周期；
- 训练期与部署期资源审计；
- 完整日志、报告、复现命令和Git提交。

未达到上述条件时，研发目标保持开放。每个负结果必须形成可证伪结论和下一单一机制假设，继续实验推进，而不是以代码完成、资源达标或局部平均值结束研发。
