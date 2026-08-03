# Stage2功能研发目标与证据门

状态：`ACTIVE / JOINT_DA_D92_CODESIGN / SMALL_MATRIX_FIRST / NO_NEW_PERFORMANCE_RESULT`

## 1.最终目标

本轮研发必须在`p2_min_v1`下形成一个同时包含轻型共享域适应和精简D92分类头的Stage2-C方法。D92不是冻结的下游基线，而是必须与DA共同删改、共同归因和共同优化的组成部分。最终候选在单seed的25个Target job上必须满足：

|slice|注册后旧类准确率|最低旧类准确率|新类准确率|
|---|---:|---:|---:|
|K10/new5|≥92%|≥85%|≥92%|
|K10/new10|≥92%|≥85%|≥90%|
|K10/new20|≥92%|≥85%|≥86%|

K5/new20相对matched K10/new20的注册后旧类、最低旧类、新类和`H_old_new`下降均不得超过5pp。K1/new20必须相对同row冻结D92基线产生真实提升，不能依靠identity、整臂fallback或未改变prediction通过。

联合方法还必须同时满足下列效率目标。基线固定为完成Target125的正式288维D92，不得用160维source-held代理状态替代：

- 精简D92固定采用单平面对称INT8系数`W_q[C,160]`、每类FP16 scale和FP16 intercept，不保留FP32 sidecar。正式D92核心数组为`B_formal=1152+590C`字节，D92-Lite为`B_lite=164C`字节；在C=26时由16,492B降至4,264B，减少74.1%。计入rank2 DA资产后，联合新增数值状态仍至少减少50%；
- K5头部拟合MAC-equivalent至少减少90%，同机同线程实测墙钟中位数至少减少50%，且不增加backbone forward次数；
- K5的query端head MAC至少减少40%；K1严格alias到qKNN，其资源按qKNN正式口径单列，不得写成D92-Lite零成本；
- 正式D92的old/new稠密协方差属于拟合期瞬时量，部署态本来就是单仿射头。精简目标是删除重复拟合、288维辅助块和无效计算，不得把source-held代理额外保存的两块160×160矩阵冒充正式D92部署开销。

中间门只筛掉没有功能作用或协议不合法的实现，不降低、替换或重新解释上述最终指标。

## 2.证据起点

|方法|已验证结论|对新研发的约束|
|---|---|---|
|D62|完整125；注册后仅24/375个场景状态实际激活，K1整体fallback|禁止用离散安全门把大多数row退回基线|
|D91|仅K10/new5 development；与D62的15/15 outer prediction相同|support目标下降或内部几何变化不等于分类功能|
|D92|K10/new20旧类+2.622pp、floor+4.600pp，但新类−0.653pp；K1逐值不变。正式实现作用于288维D62/D81管线，在大量full/block/crossfit组件内分别拟合old/new收缩协方差，最终只部署单个紧凑仿射头；另一个160维source-held代理会额外保存两块协方差，不能混为正式资源证据|D92必须与DA联合重构；删除role分裂、重复稠密拟合、D62行拼接和无独立贡献的FFT96/RF32块；K1不得伪造不可辨识的类内方差|
|SVRN/r4.2|完整125且相对D62全面劣化|不再使用会放大注册后旧类崩塌的分支状态|
|D104|量化机制和release代码闭合，无Target性能|ANGQ只能作为实现组件，不能预设为最终分类头|

## 3.指标与同row口径

原子scenario-row固定为：

```text
(receiver, seed, K, new_count, scenario,
 capsule_id, split_id, query_id_root, method_lock)
```

同一矩阵的全部臂必须共享全部字段、同一old query、同一new query和同一独立scorer。每个正式slice由同seed的5个receiver×3个互斥LEO场景组成15个scenario-row，并等权宏平均。

- `A_old`：注册后旧类准确率；
- `N`：已注册新类准确率；
- `H`：同一row的`A_old`与`N`调和均值；
- `F_old`：对每个旧类先聚合该slice全部15个scenario-row，再取最低类准确率；
- `forgetting=B_old-A_old`；
- 辅助报告：mean row-floor、worst row-floor、逐类、逐receiver、逐scene和正确数。

不得用不同row的边际最大值拼接结论。`F_old`是全部旧类的通用下界，不是预选弱类清单。

## 4.联合候选小筛选与内联功能门

正式大矩阵前最多保留3条原理不同的DA候选，并只冻结一个由三者共享的精简D92。禁止固定历史D92后只开发DA，也禁止为每条DA再开发一个不同head扩大方法空间。候选可以覆盖早层、中层和晚层干预，但层位只是机制差异的一部分，不构成层扫描。

三候选先使用source receiver-held×class/TX-LOCO完成Phase1方向学习和资产审计；该数据只验证物理ID隔离、outer元目标、非零功能与跨receiver可迁移性，不模拟formal D92性能，也不承担候选排序。随后三候选在不同GPU上并行运行同一份真实checkpoint、Target development received-IQ、K1/K5 support/query划分和评分器。公共臂只计算一次；每个候选只允许一次冻结实现，不扫描层、rank、步数、view、seed或门限。

Phase1资产构建使用单一确定性日程：canonical receiver-mean SVD初始化，模型checkpoint冻结，float32资产前向配合float64统计，K1/K5、receiver和class等权的预冻结全批episode，full-batch L-BFGS固定`max_iter=128`与`strong_wolfe`，单初始化、无early stop、无学习率/epoch/正则扫描。有效秩不足2或前两方向近重根时直接关闭候选；冻结清单内全部support物理ID与全部outer-query物理ID跨K、跨episode全局互斥。A/B只学习`U/V`并封存二维`D_F`及仅由Phase1 inner tap确定的`rho/a_max`，Phase2不得由target support重估预算；C以固定`Q=[I_2,0]`、`b=0`初始化后只学习`U/V/Q/b`并封存5维summary的`m_P1/d_P1`，Phase2固定使用`(s-m_P1)/d_P1`。所有数值下限仅由float64机器精度解析确定。最终只允许量化资产、必要FP16尺度和上述小统计进入Phase2，不保留source样本、样本feature、实体键或FP32 sidecar。该步骤是必要资产构建，不是source性能筛选；outer只审计隔离、置换、非零和真实功能变化，唯一候选排序仍发生在全部S0 prediction封存后的一次truth评分。

outer审计闭合后，每个候选按同一固定日程在全部Phase1 source receiver上重建一次最终资产，不得按outer分数选择checkpoint、fold资产或迭代。C的Phase1独立query loss必须可直接反传到`U/V/Q/b`，但Phase2仍无optimizer且query零梯度。最终资产只绑定全source训练清单及物理ID根，不携带fold专属样本状态。

DA资产量化固定为：`U`按rank列、`V/Q`按rank行、`b`按整向量做对称INT8，每组一个FP16 scale；小统计保存FP16，正统计若下溢为零或非有限则关闭候选，不得静默抬升数值下限；解码后的float32只读运行时视图不得作为sidecar落盘。A/B payload为`4d+14B`，C为`1328B`；C与C=26的D92-Lite合计`5592B`，相对formal D92减少66.09%。量化parity只用Phase1固定fixture核验，不读取Target truth。

三候选的support监督固定使用一次forward的两个同IQ最终表示视图：`z_A=L2(ReLU(pre_relu))`与`z_B=L2(pre_relu)`；仅当ReLU零范数时，`z_A`确定性总化为`z_B`。对另一视图的class mean prototype做stop-gradient，以冻结qKNN温度0.85计算双向全类cosine CE并等权平均。两个视图不增加K、不重跑LEO观测；该损失只生成support-conditioned状态，query不参加。

Phase1 outer元目标必须沿真实checkpoint下游执行。adapted source support按正式qKNN路径量化并解码成stop-gradient的INT8/FP16 bank，独立source query保持可微；identity-metric Student-t logits以float64计算，在部署logit输出点闭合为float32，再转回float64除以冻结temperature计算全类CE。support量化分支不回传梯度，但query分支必须直接对A/B的`U/V`或C的`U/V/Q/b`产生非零梯度；禁止用tap空间代理loss、未量化float support或手工raw asset替代。

预测生成和真实性能评分分离。打开本轮候选的truth评分前先完成以下内联功能检查；检查通过后在同一不可覆盖run中评分，不另设588条G0或新的控制面：

1.状态只读取不可变Phase1 bundle、当前row合法support和冻结配置；
2.query为零fit、零selection、零update，每条query独立面对全部注册类；
3.算法对类别标签置换等价，不含TX/class ID专用参数；
4.DA使用同一共享规则处理所有类；HEAD不读取class-specific ground prototype，ground与target关系全部归入DA；
5.K1自由度必须来自跨类共享低维结构、同一physical IQ的合法数学view或Phase1预锁共享先验，不能估计单类方差或类专属adapter；若精简D92在K1不可辨识，必须显式使用qKNN边界，K1联合收益只归因于DA；
6.support physical-LOO/OOF只用于构造support-only状态，support准确率不能作为晋级依据；
7.no-query真实特征smoke必须改变feature、邻居、margin或argmax中的至少一项；
8.INT8、state、MAC、拟合时延、临时空间和序列化资源闭合；
9.若DA只是共同平移、正交变换或全局正尺度，且在变换后重建欧氏/余弦qKNN导致邻居排序不变，则直接证伪。

协议错误记为`INVALID / NO_PERFORMANCE_RESULT`；机制合法但没有可观测决策作用，记为`REJECT_REVISION_NO_FUNCTION`。

## 5.S0/S1最小联合因果筛选

formal D92使用288维`z_id160+FFT96+RF32`完整D62/D81管线，D92-Lite使用160维`z_id`。二者之间的差值同时包含表示、注册器和head变化，不能解释为纯head效应。联合因果识别因此只在相同160维表示空间中使用干净`2×2`：

|臂|表示|分类头|回答的问题|
|---|---|---|---|
|M0|基础`z_id160`|qKNN|共同基线|
|M_DA|同一候选适应后的`z_id160`|qKNN|DA单独是否提升|
|M_L92|基础`z_id160`|共享D92-Lite|精简头相对qKNN是否提升|
|M_JOINT|同一候选适应后的`z_id160`|同一D92-Lite|联合方法是否提升|
|R_D92_FORMAL|正式288维表示|冻结formal D92|同row全管线历史参照，不进入`2×2`交互|

`M_DA/M_JOINT`必须复用同一DA state和adapted feature缓存；`M_L92/M_JOINT`使用同一公式，但必须分别从各自表示的合法support拟合head。K1没有类内残差自由度，固定`M_L92=M0`、`M_JOINT=M_DA`，以等价receipt避免重复计算；K1只归因DA。`R_D92_FORMAL`优先绑定完全同row key的历史artifact，否则每行最多计算一次。`M_DA_D92`只回答DA与旧formal管线的兼容性，不是联合筛选必要臂；仅允许在S1对唯一胜者作一次非阻塞诊断。

对指标`y`定义：

```text
DA_Q          = M_DA-M0
DA_L92        = M_JOINT-M_L92
L92_BASE      = M_L92-M0
L92_AFTER_DA  = M_JOINT-M_DA
I_DA_X_L92    = M_JOINT-M_DA-M_L92+M0
FORMAL_REPLACE= M_L92-R_D92_FORMAL  # 只表示全管线替换差
```

S0固定为按receiver ID词典序预冻结的3个development receiver×`{K1/new20,K5/new20}`×3个互斥scene，共18个原子row。公共`M0/M_L92/R_D92_FORMAL`只生成一次；三候选各生成`M_DA/M_JOINT`，形成9个逻辑输出，K1别名不重复计算。三候选全部prediction完成后才一次性评分，不查看中间性能。

S0只作方向排序，不设置0.5pp级多指标硬门。候选进入排序只要求：

- `M_DA-M0`的池化`H`严格增加；
- K5下`M_JOINT-M_DA`的池化`H`严格增加；
- `M_JOINT-M0`的池化`H`和old+new总正确数都严格增加。

`A_old/N/F_old`、逐receiver、逐scene、四个简单效应和`I_DA_X_L92`全部报告，但不再新增小样本release gate。满足上列三项的候选按`min(DA_Q_H,L92_AFTER_DA_H)`、最差receiver联合增益、联合总正确数和端到端资源作词典序排序，只选一个胜者。

S1固定为剩余2个receiver×`{K1/new20,K5/new20,K10/new20}`×3scene，共18个原子row，只运行S0胜者的核心四臂和公共formal参照。S0/S1均标为`TARGET_DEVELOPMENT`；S1失败后关闭本轮，不把S0第二名递补。完整负收益候选立即关闭，不根据已打开结果修改层、rank、step、view、shrinkage、量化或阈值。效率目标不阻塞S0启动，但不满足§1的候选不得成为最终胜者。

## 6.G2单seed Target25门

候选、method lock和全部超参数在Target访问前冻结。screen seed必须从已完成D92 Target125中选择一个未参与本轮S0/S1候选评分且具备全部同键D92 artifact的seed；选择规则在打开该seed的本轮候选prediction前冻结。它是`METHOD_UNSEEN_SCREEN`，不是全项目从未读取truth的盲测。矩阵固定：

```text
5 receivers × 1 seed ×
{K10/new5, K10/new10, K10/new20, K5/new20, K1/new20}
= 25 jobs
```

每个job覆盖3个物理ID互斥的`leo_*_weak`场景。一次Target25只评估一个revision，不得从25行中选择receiver、scene、class或slice重跑。

S0/S1按§5完成方向筛选。Target25只运行胜者的必要4臂`M0/M_DA/M_L92/M_JOINT`；历史D92只从已完成125中按完全相同的`capsule_id/split_id/query_id_root/receiver/seed/K/new_count/scenario`键连接，不重跑D92。键不全同则该seed不得用于本轮K1 paired目标，应在prediction前改选另一个具备完整同键D92 artifact且本方法未见的seed。

Target25完整性单位保持：

```text
25 jobs × 3 scenarios × 4 arms
= 300个scenario-arm pair
= 600个state prediction surface（before/after各300）
```

每个pair必须同时封存Stage2-B的before旧类预测和Stage2-C的after旧类/新类预测。任一state或arm缺失、预测可覆盖、truth先于全部600个state prediction surface开放、键不唯一、哈希不匹配或只完成联合臂，均不得进入性能分析。`forgetting`只能由同一pair的before/after旧类预测计算，不能从其他方法或其他run补入。

K10执行§1全部硬门。

K5以同receiver、同scene的K10/new20为matched基线。预登记时必须锁定`K5 support physical IDs⊂K10 support physical IDs`，且两者的`query_id_root`逐scene完全相同；只有满足该嵌套关系时，`A_old/F_old/N/H`下降≤5pp才属于paired结论。若不满足，只能报告非配对差值，不能用于通过本门。单scenario-row退化完整报告，但不再另设边际灾难阈值。

K1以同row冻结D92为基线：

```text
ΔH >= +2pp
ΔF_old >= +2pp
ΔA_old >= 0
ΔN >= 0
old+new总正确数严格增加
```

单seed通过记为`TARGET25_SCREEN_PASS`，证明本轮研发目标在该预注册seed上达到；不能据此宣称多seed稳定。

## 7.G3顺序确认

需要形成可推广的`PROMOTABLE`声明时，保持同一method lock，再运行一个预注册fresh confirm seed的25-job实验，不合并成125大矩阵。screen seed和confirm seed必须各自单独重过G2，并在两seed池化后再次重算。只有当两次结论冲突或置信区间极不稳定时，才单独决定是否增加第三seed；第三seed不是默认硬流程。

确认门：

- K1的paired`ΔH/ΔF_old`95%CI下界>0；
- K5四项下降的95%CI上界≤5pp；
- confirm seed失败记为`SCREEN_POSITIVE_NOT_CONFIRMED`。

修改结构、rank、loss、step、量化、阈值或fallback会清零确认资格，并以新revision回到Phase1资产审计和S0。

## 8.研发工作包与模型分工

|工作包|职责|执行模型|
|---|---|---|
|WP-DA|轻型共享DA、层位/状态可辨识性、K1/K5 support更新与资源|`gpt-5.6-terra/max`|
|WP-D92|历史D92计算图删改、类置换对称共享统计、K1边界及部署状态压缩|`gpt-5.6-terra/max`|
|WP-CODESIGN|DA与精简D92共享view/统计/缓存、4臂因果接口和formal参照边界|`gpt-5.6-terra/max`|
|WP-DATA|目标、矩阵、同row指标、结果分析、拒绝语义和交叉审查|`gpt-5.6-sol/high`|
|WP-INTEGRATE|协议解释、方案冻结、代码整合、独立复审和最终决策|主agent，`gpt-5.6-sol/high`|
|WP-IMPLEMENT|复杂科学核心实现、改变或实现新机制、需要科学判断的复杂缺陷修复|`gpt-5.6-terra/max`|
|WP-RUNNER|唯一N607落地、同步、启动、健康检查、监控和artifact回收；不得修改方法或矩阵|`gpt-5.6-terra/max`|
|WP-MECH|固定清单、hash、manifest、报告骨架、字段完整性及执行已冻结的本地测试命令|`Luna/max`|

方法agent不得自我认证。WP-DATA必须审查K-shot可辨识性、common-transform cancellation、support proxy过拟合、旧/新任务平衡、类置换、资源和query/role/quota禁区。

每个功能包由不同agent拥有非重叠文件面。服务器实验必须另设唯一`gpt-5.6-terra/max`runner；Luna不得执行SSH/SCP、实验启动/停止/重启、科学代码编辑、性能分析或晋级判断。runner只负责冻结run的落地、启动、健康检查、完整日志与artifact回收，不得改方法、调参、按性能重跑或与主agent重复启动。主agent和WP-DATA使用sol-high读取完整25-job/300-pair/600-state预测与评分证据后再作晋级决定。

## 9.拒绝语义

|状态|含义|
|---|---|
|`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|预测前系统性技术失败|
|`PARTIAL_DIAGNOSTIC_BIASED_NOT_PROMOTABLE`|只有partial prediction或score|
|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|完整矩阵完成但性能门失败|
|`TARGET25_SCREEN_PASS`|单seed25达到本轮硬目标|
|`SCREEN_POSITIVE_NOT_CONFIRMED`|单seed通过但确认seed失败|
|`PROMOTABLE`|Phase1资产审计、S0、S1、Target25和fresh confirm全部通过|

禁止按receiver、scene、class、seed挑选结果，禁止跨run拼接极值，禁止用source-held替代Target。

## 10.当前执行优先级

1.`D106-RCMR`真实588条功能面及source-held结果只作为历史非晋级证据；`D121`、`D122`组合项、`D123`已关闭，旧run不重跑、不沿用其G0/G1流程、不修旧通用release链；
2.`D106` Target25 r7仅完成46/600 state后技术退出，严格为`NO_PERFORMANCE_RESULT`；当前没有新的Target性能；
3.历史formal D92保留为固定参照，但不再视为最终头：K1整臂fallback、288维D62/D81管线、old/new重复稠密拟合和row-splice计算是本轮明确删改对象；160维held代理的额外协方差状态只作独立工程诊断；
4.当前最多并行3条DA候选并共享一个D92-Lite。浅层梯度残差只是候选之一；中层残差和晚层无反传超适配可以作为原理不同候选，但最终由同一160维`2×2`证据选择，不预设浅层获胜；
5.每条路线必须在设计时说明DA的唯一干预点/状态；三条路线共享同一个精简D92 K1/K5公式、view/cache契约、formal D92删除项、预测状态和实测资源口径；
6.先完成一次Git承载的设计冻结和追踪映射，再由不同Terra Max agent实现非重叠科学核心和独立复审；Luna Max只承担固定清单、hash、manifest、报告骨架、字段检查及执行已冻结的本地测试命令；N607唯一runner使用Terra Max；
7.本地最小负测、真实checkpoint无query smoke、独立`P0=0、P1=0`和Git提交后立即发布三候选并行小筛选，不新增数据复验、authority或通用控制面；
8.S0只选一个胜者进入S1；S1失败不递补runner-up，完整负收益方法立即关闭并研发新原理。S1正向后直接进入方法未见seed的Target25，不运行588条G0、fresh63或D62/D92/SVRN重复125矩阵。
