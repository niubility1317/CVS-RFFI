# Stage2轻型域适应与新类注册研发目标

版本：2026-07-21
状态：active goal定义
数据协议引用：`protocol_schema=p2_min_v1`

## 1. 单一总目标

基于ADV3B02 final checkpoint，研发并验证可逐样本部署的极轻型Phase2方法，使Stage2-B旧类目标域适应与Stage2-C新类注册同等有效，并同时解决多新类条件下的旧类遗忘、old/new混淆和持续floor类。开发阶段只用注册support选择方法与统一超参数；query等同测试集，只用于最终评分。

数据合法性只通过`p2_min_v1`的`VALIDATED_ONCE capsule_id/split_id`引用，不在本目标重复建设准入、hash或authority系统。固定接收IQ、物理ID和split未变化时，直接进行方法研发与实验。

## 2. 成功判据

### 2.1 K10主门槛

在锁定候选的独立确认矩阵上同时达到：

- target-old注册后总体准确率`old_acc_after_increment ≥ 92%`；
- 每个旧类确认集准确率`min_old_class_acc ≥ 88%`；
- `seen_new_acc ≥ 92%`，当真实seen-new TX数为5；
- `seen_new_acc ≥ 90%`，当真实seen-new TX数为10；
- `seen_new_acc ≥ 86%`，当真实seen-new TX数为20；
- 同row报告`old_acc_before_increment`、`old_acc_after_increment`、`seen_new_acc`、`H_old_new`和逐类混淆，不能跨row拼接最好值。

真实seen-new TX数2也必须评估和报告，但不另造用户未指定的绝对门槛；它用于小规模注册机制诊断。

### 2.2 K5与K1/K20

- K5在每种新类规模和每个核心指标上相对matched K10下降不超过3pp；
- K1、K5、K10、K20全部执行同row注册前/后遗忘评估；
- K1总体与每个receiver的`old_adaptation_gain=old_acc_after_increment-old_acc_before_increment ≥ 0`；
- K1在相同旧类query上明显优于直接ADV3B02旧类头：总体paired差值至少+2pp、matched paired 95% CI下界大于0、每receiver不为负；
- K5/K10/K20的平均遗忘不得高于matched identity-only单qKNN；
- K20用于检查support增加后是否饱和或反向遗忘，不能参与开发选参。

### 2.3 floor与混淆

必须同时优化总体均值和全部实际注册类中的最低类，禁止用均值掩盖floor，也禁止把floor目标收缩成若干历史难类。开发代理、loss和正式评价必须覆盖当前row的全部旧类与全部已注册新类，并使用类别身份无关的共享规则。

算法可以依据每类合法support，通过同一公式估计prototype半径、不确定度、校准强度或更新幅度；不得读取TX/class ID来选择机制，不得为具体类别设置白名单、专属分支、专属loss权重、专属阈值或专属超参数。历史报告中的难类只能用于解释失败和检验通用方法是否改善下尾，不能作为定向调参集合。

候选必须降低两类失败：新类不可达和新类侵入旧类。不得再以support拟合100%、support侵入为0或LOO安全单独证明held query安全。

## 3. 开发与确认设计

### 3.1 开发锁定

- 只在预登记development seed、K10工作点，使用注册support内部预登记的leave-one-physical-sample-out或nested support-held代理选择一套候选、表征组合、adapter结构、loss、epoch和所有超参数；
- support内部开发代理同时覆盖Stage2-B旧类、模拟Stage2-C后的旧类/新类、调和均值、最低类和遗忘风险；真实development query与confirmation query都必须在候选和超参数完全锁定后只预测、评分一次，不得据结果继续调参；
- K1/K5/K20及独立确认seed不能反向调参；
- target query标签、角色、类别数量或指标不能参与拟合、早停、rollback、路由或候选选择。

### 3.2 独立确认矩阵

锁定后覆盖：

```text
target receivers: 5
confirmation seeds: at least 5, independent from development
LEO scenes: leo_clear_weak, leo_low_elev_weak, leo_rain_weak
K: 1, 5, 10, 20
real seen-new TX counts: 2, 5, 10, 20
```

若合法目标接收机或真实新TX覆盖不足，可使用未进入Phase1的其他WiSig/ManySig接收机或TX子集；仍须引用合法`p2_min_v1` capsule，且不得使用clean样本。

历史“125任务”保留为候选稳定性screen：运行`5 receivers × 5 independent confirmation seeds × 5 evaluation slices = 125 jobs`。五个slice固定为`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`和`K1/new20`；每个job内部都评估三个LEO场景。比较方法作为同一job/同一数据上的matched候选列或配套结果，不构成125的第五轴。125 screen不能替代完整K1/5/10/20×new2/5/10/20正式矩阵。

## 4. 研发路线优先级

### 4.0 强制路线纠偏：域适应不是分类头改名

本目标中的“域适应”必须包含对星地/接收机偏移的显式估计与表示校正。仅修改qKNN距离、prototype打分、协方差、分类温度、融合权重、类别bias、RDA/SRDA头或old/new校准，不得单独作为下一轮主候选，也不得据此宣称已解决P0-2星地偏移。分类头可以作为联合方法的第二模块和matched ablation，但不能持续替代表示层域适应研发。

本目标锁定问题、协议、资源和证据标准，不锁死最终算法路线。`z_id`和`z_dom`是同等级的快速域适应空间与后续识别组件，不预设其中任何一个优先、固定或仅作辅助。`z_id/z_dom`单支适配、联合适配、交叉条件化、规范化融合、轻量adapter/LoRA/FiLM、support-only统计对齐、受约束metric/covariance adaptation、Phase1 meta-DG先验、Bayesian transport等均可作为候选；允许根据合法Phase1 LODO、support-held代理和冻结窄验证证据修改、组合或淘汰。任何路线仍必须具有可独立消融的域偏移估计或表示/metric适配机制，不能退化为只调分类分数。

各根因、机制边界与晋级证据固定如下。主agent必须在候选预登记和完成报告中标明本轮实际处理了哪些行；仅改分类头最多能直接处理P0-3、P0-5和P1-1，不能据此声称处理了P0-2：

| 优先级/根因 | 已知失败表现 | 主解决机制 | 禁止替代或常见误区 | 必须观察的晋级证据 |
|---|---|---|---|---|
| P0-1：注册前target-old判别margin不足 | before-old本身约86%，部分旧类远低于总体均值；新类加入前已存在弱类 | target-old support参与的`z_dom`条件化表示校正；全类class-balanced LOO；弱coverage时identity收缩 | 只在注册后给旧类加bias或ground logit；只优化总体old均值 | 同row注册前old、逐旧类margin、最低旧类和每receiver均不恶化；改善不是由old专属加分产生 |
| P0-2：星地/接收机偏移未被识别和消除 | D93/D94的ground中心transport可support拟合100%但query下降；ground nuisance coverage低 | 同时探索`z_id-only`、`z_dom-only`和joint `z_id+z_dom`快速域适应，也允许其他support-only表示/metric路线；使用全部合法target support进行低自由度、可回退校正 | 把仅调分类分数、共同正交不变换或完整重估后不改变几何的操作称为域适应；使用query更新 | 域适应相对原始基线独立正收益；三种表示路线matched消融、对应置乱/identity负对照、适配幅度、receiver/scene增益和回退完整 |
| P0-3：新类注册改写全部类别竞争边界 | old保护常伴随new下降，new释放又造成old侵入 | 域适应后的target-support qKNN局部头+SRDA全局头；所有类统一均值、温度和先验；ground仅作共享协方差/关系正则 | ground旧类原型直接投票；old/new角色专属bias、阈值或quota | 同rowafter-old、seen-new、`H_old_new`、old→new/new→old混淆同时改善，不能跨row拼最好值 |
| P0-4：K-shot下高维适配不可辨识、support-query失配 | 高维变换support准确率极高但held query负迁移；K1无法估计类内散度 | rank≤4/6的解析ridge或少步小参数更新；K≥2 support cross-fit；K1使用Phase1冻结映射和support域上下文 | K1训练深adapter；全矩阵仿射；用development/confirmation query早停或选rank | support-held与锁定query方向一致；报告fit/held gap、rank、参数norm、步数；K1非identity且每receiver不为负 |
| P0-5：旧类下尾和逐类不均匀严重 | 同一row旧类可从约40%到接近97%，总体均值掩盖floor | 覆盖全部注册类的class-balanced LOO、soft-CVaR/下尾风险、support半径校准；局部头保留弱类多峰，SRDA稳定全局边界 | 按TX ID定向保护历史难类；只提升平均old；用新类损失换旧类floor | 报告全部逐类、最低旧/新类、下尾分位数、CVaR及每类混淆；floor提高且seen-new/H不下降 |
| P1-1：类内多峰被单中心抹平 | 单均值对局部mode和异常shot敏感，部分qKNN类仍有可救援局部证据 | target-support qKNN/Student-t归一化局部混合；合法ground多原型只作弱关系/协方差先验 | 对多原型裸max或不除以数量；ground多原型覆盖target-new；用单clean样本伪装原型 | qKNN与SRDA的disagreement、双向rescue、oracle-union、逐类误差重叠和prototype-count公平性 |
| P1-2：receiver/scenario异质性 | 某些receiver及rain/low-elevation显著更差，单一row级transport外推失败 | `z_dom`逐样本条件化，按类平衡聚合row域上下文；Phase1 LODO/meta-DG提高可外推方向 | receiver ID专属参数；同一clean IQ多信道重放；强制所有场景使用同一非收缩偏移 | 五receiver×三scene分解、coverage-收益关系、各receiver old gain≥0及最差scene不恶化 |
| P1-3：K1现有方法退化为identity | 无类内方差、无法LOO，现有适配器不更新或过拟合 | Phase1预锁解析域映射和可靠度；K1仅估计全类域上下文/coverage并执行一次低秩校正 | 伪造K1类内协方差；从query估计温度、BN或熵；K1多步深微调 | 相对direct ADV3B02总体≥+2pp、paired CI下界>0、每receiver≥0，并报告适配幅度 |
| P1-4：地面聚合知识冗余且任务信息错配 | 84个domain×class中心重构余弦很高但预测不变，`D_eff`远低于名义数量 | 密度反权重、共享domain nuisance basis、类均值+切空间局部残差、coverage证书；rank随`D_eff`自适应 | 逐个重构84中心并把高余弦当成功；强行固定rank；保存成员或单样本feature | `D_eff`、stable rank、LODO transport误差、margin flip、逻辑/序列化字节和相对现有bundle的任务收益 |
| P2：量化与资源误差 | INT8/INT4可能造成小margin样本翻转，但不是当前星地偏移主因 | margin-aware量化、量化前后同row预测一致性、INT8正式状态；仅在表示/分类机制正收益后压缩 | 用量化调参替代域适应；只报告重构余弦；FP32结果直接作为正式候选 | top1一致率、margin sign flip、old/new/H/floor差值、状态字节、MAC、平均/P95时延 |

每轮联合研发至少包含四个正交matched候选：

1. `A=原始合法baseline`；
2. `B=仅分类头改进`；
3. `C=仅显式域适应，分类头保持A不变`；
4. `D=显式域适应+分类头联合`。

只有C相对A在support-held代理及锁定窄验证中表现出可复核正信号，才能把D相对B的增益归因于域适应。若C失败，必须报告表示覆盖、参数可辨识性、逐receiver/scene/类结果和负对照，修改域适应机制；不能跳过C而继续只迭代B。125 screen不得用于选择域适应rank、loss、coverage公式或分类头超参数。

### 4.1 第一优先：同时探索`z_id`、`z_dom`及其联合快速域适应

ADV3B02提供`z_id=feat_joint`和`z_dom=feat_imp`。虽然两支训练职责不同，但二者都可能同时包含身份与域信息，因此都必须作为快速域适应和最终识别的候选组件，而不是预设`z_id`只负责识别、`z_dom`只负责条件化。研发必须在matched条件下至少比较：`DA-id`仅适配/识别`z_id`、`DA-dom`仅适配/识别`z_dom`、`DA-joint`联合适配并融合两支。允许concat、gated fusion、cross-conditioning、product-of-experts、局部—全局双头或其他受约束融合，但必须做分块归一化、能量/温度控制和单支/联合消融。不得把`dom_head`的argmax当作目标域真值，也不得按receiver ID建立专属分支。

Phase1应分别及联合审计`z_id/z_dom`，并按候选所需生成与checkpoint共同封存的bundle状态：

- 在合法地面LODO上分别报告`z_id/z_dom`的TX可分性、domain敏感度、交叉泄漏、有效rank、`D_eff`和跨域稳定性；
- 从多物理样本聚合的地面`z_id/z_dom`域×类统计学习`z_dom→z_id`低秩污染映射、奇异值、半径和coverage证书；不得保存样本级feature、成员ID、可逆归属或独立sidecar；
- 低秩rank只能由Phase1 LODO固定，建议上限4或6，并满足`r <= floor(D_eff)-1`；84个名义domain×class中心不得直接当作84个独立域方向；
- 必须包含random/permuted `z_id`、random/permuted `z_dom`及关闭cross-branch交互的负对照，证明收益来自对应表示与联合机制而不是增加参数。

若已选择的候选确实需要而当前Phase1 bundle缺少聚合`z_id`、`z_dom`或二者交叉统计，应按`项目.md`构建合法、不可替换、共同封存的新bundle并更新`bundle_id`；固定received-IQ capsule和split不变时不得触发数据重验。也允许选择不需要新增bundle状态的其他合法域适应，但必须证明其域适应机制独立于分类头。禁止因为现有bundle只方便其中一支或分类头，就跳过另一支及联合适配消融。

Phase2域适应只能使用当前row合法target support；旧类和新类support均可按类平衡参与估计共享目标域状态。候选可联合使用`z_id/z_dom`、同一固定received IQ的合法数学view和sealed ground聚合知识，但没有ground新类原型，ground知识不得直接给旧类增加logit。任何路线都必须估计适配可信度、限制自由度并提供identity/target-support-only回退；不能在低可辨识或域覆盖不足时强制外推。

参数更新范围由正式总预算和support可辨识性约束，不预先锁死到固定层表。正式候选全部可训练参数合计仍须`<=80,000`、optimizer steps`<=50`；在此预算内可按预登记候选联合快速更新，例如：

- `z_dom`私有后段：末端normalization affine、`DomainFeatureEnhancer`的轻量融合gate/residual、time/frequency/PA末端投影和fuse层的rank-2/4 LoRA或等价低秩adapter；
- `z_id`私有后段：`z_id`后的低秩残差、time/frequency/PA末端投影及`id_proj/joint_proj/id_gate`的低秩adapter；
- `z_dom→z_id`跨分支低秩污染映射、coverage gate，以及轻量分类头/融合参数。

更新`z_dom`或其他与sealed ground坐标关联的模块时，必须保留冻结参考路径或等价的坐标兼容、coverage审计和回退机制；推荐但不强制采用`z_dom^0`与快速适配`z_dom^*`双路径。Phase2不得改写sealed ground状态。适配参数应受L2-SP/参数位移、接收后数学view一致性、TX泄漏抑制或其他可验证正则约束，避免把身份特征误吸收到域分支或使坐标任意漂移。

Sinc/HF共享stem、source分类头和早期大卷积块默认从冻结对照开始，因为它们自由度高；但目标不永久禁止更新任何具体层。可在`<=80,000`总参数和`<=50`步内，根据support cross-fit、梯度冲突、欠拟合及matched ablation证据选择更新`z_dom`、`z_id`、末端私有block、轻量归一化/门控或其他子集。任何扩大更新范围都必须预登记、保留更小更新范围对照并报告参数/梯度/更新norm，不得无消融地全主干微调。

K1也允许快速更新共享的极小`z_id`、`z_dom`或joint适配参数：利用当前row全部注册类support及同一固定received IQ的合法数学view，进行闭式更新或最多5步强收缩更新；不得估计高维类内协方差。K>=2允许support cross-fit下闭式ridge或最多30步联合更新。所有K仍受总计`<=50` optimizer steps约束，并必须分别报告两支及联合模块的参数量、梯度norm、更新norm和冻结参考回退结果。

### 4.2 第二优先：提高Stage2-B旧域表示与通用floor

当前可复核最强比较器B3约为注册前old86.67%、注册后old73.33%、new73.33%，说明域适应和注册均未解决。下一路线必须先在support-held old proxy和旧类floor代理上超过B3，再扩展注册机制；冻结后才在真实held query评估，不再以继续叠加hard visibility gate替代旧域适应研发。

同一固定received IQ上的`z_id160 + FFT96 + RF32`只能作为接收后辅助数学表征和matched ablation，必须分块L2归一化并控制能量。不得把维度增加、共同正交变换或完整重估后几何不变的变换写成域适应成功。域适应的可观察机制必须来自`z_dom`条件化的逐样本非正交残差、受限metric、coverage-controlled ground先验或其他能实际改变support-query判别几何的合法机制。

### 4.3 第三优先：连续联合适配、局部—全局分类与注册

在一个全注册类空间内联合训练target-old和target-new support：

- ground old int8聚合知识只作只读身份先验、正则或不确定度参考，不直接覆盖target原型；
- target-old原型负责域校正，target-new原型独立注册；最终部署的target-old和target-new原型均须量化，优先int8；FP16/FP32只作为matched精度/速度/状态ablation，按Pareto证明最终格式；
- 域适应后再采用qKNN/Student-t局部头与SRDA全局头的matched对比和support-only融合；qKNN保留类内多峰，SRDA提供类平衡共享协方差边界，ground只提供coverage-controlled共享先验；
- 所有旧类和新类的分类均值都来自当前target support，所有类使用同一公式、先验和温度；不得按old/new角色直接增加分数；
- loss同时包含old support分类、new support分类、ground-anchor弱正则、类内半径收缩、类间`margin > radius_i + radius_j`、old/new collision惩罚及adapter幅度正则；
- 为解决旧类严重不均匀，开发代理必须加入覆盖全部已注册类的class-balanced LOO和soft-CVaR/下尾风险项，并报告全部逐类准确率、最低旧类、最低新类和下尾分位数；
- 只允许support梯度，使用快速闭式或少步梯度更新；不用binary visibility/hard release作为主学习机制。

### 4.4 遗忘保护

- 注册前冻结一份旧类决策状态作为teacher/anchor，仅用于support侧蒸馏与参数位移约束；
- 对每个旧类按support不确定度决定target校正与ground int8先验的融合强度；
- 新类追加必须是append-only class state，不重写ground int8组件；
- 用support-held/leave-one-physical-sample-out风险约束连续margin，但其只作开发代理，不能替代query确认；
- 若发生“新类不可达”或“旧类侵入”，优先修正连续几何、校准和loss，不再增加多层hard gate。

## 5. 资源与部署约束

首选Pareto目标：adapter可训练参数不超过50k、适配不超过20epoch、持久化增量状态尽量低于256KB。

正式硬上限：

```text
trainable adapter parameters ≤ 80,000
adaptation epochs ≤ 30
optimizer steps ≤ 50
persistent incremental state ≤ 256 KB
dense query graph = false
query-dependent batch optimization = false
```

为机制探索允许单独使用150%档：不超过120k参数、45epoch、75 optimizer steps和384KB。探索档不能进入正式确认或部署Pareto；正路线必须压缩回正式档再重验。

相对identity-only单qKNN报告增量MAC、平均/P95时延、峰值显存、backbone/FFT前向次数和状态字节。三种强制matched对比方法固定为`identity-only single-qKNN`、`ProtoNet CDA`和最强合法target-support-only轻适应基线；direct ADV3B02另作0-support性能/资源锚。目标是在性能达标后，使新增adapter/注册状态和适配计算低于三种对比方法；不虚构“比direct ADV3B02的0-support状态更少”的要求。

## 6. 执行节奏与止损

1. 直接复用`VALIDATED_ONCE`数据，先做最窄matched开发实验；不因算法变化重做数据封装。
2. 每个新机制必须回答它修复的是旧域不足、floor、新类不可达、旧类侵入还是量化/资源；没有机制假设不启动大矩阵。
3. 连续三个完成的探索轮后做一次技术复盘，审查完整日志、逐类/逐receiver结果和同row注册前后性能，再决定第四轮。
4. 只在开发seed显著优于B3并改善floor后启动125 screen；通过screen后才进入完整独立确认矩阵。
5. 兼容性、loader或报告问题只做最小修复，不把外围工程包装成研发完成。

开发与确认实验以N607为主要计算承载面，但所有修改必须本地先行并按`AGENTS.md`同步；N607不改变`p2_min_v1`输入权限、query只测试和无Oracle边界。

## 7. 完成证据

完成必须同时包含：

- 合法TX/receiver/scenario/K/support-query清单及`capsule_id/split_id`；
- 每个run的锁定candidate、seed、配置、checkpoint/bundle ID和完整训练/闭式求解日志；
- 同row注册前/后old、seen-new、`H_old_new`、遗忘、所有逐类和逐receiver结果；
- 注册前Stage2-B对target-new只报告`not-yet-enrolled reference`，例如被旧类吸收率或score margin，不得称为`seen_new_acc`；注册后Stage2-C才报告`seen_new_acc`，并同时保留before-reference与after结果；
- 5 receivers×至少5 seeds×3 scenes的独立确认矩阵，覆盖K1/5/10/20及真实新类2/5/10/20；
- target-old与target-new原型量化格式、量化误差、状态字节和append-only生命周期；
- adapter参数、epoch/step、MAC、平均/P95时延、峰值显存、前向次数、持久状态的资源审计；
- 相对上述三种强制matched对比方法及direct ADV3B02锚的同row Pareto表；
- 自动化报告、异常/失败说明、可复现命令和Git提交。

只有技术证据齐全且上述性能门槛全部通过，才能标记目标完成；完成实验矩阵但性能未达标，应明确记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不能改写为成功。
