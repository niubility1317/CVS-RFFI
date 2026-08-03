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

联合方法还必须同时满足下列效率目标：

- 精简D92部署态数值状态相对历史D92至少减少90%；
- K5头部拟合实测墙钟时间相对历史D92至少减少50%，且不增加backbone forward次数；
- K5的query端head MAC不得高于历史D92；K1若启用精简D92，其新增head计算必须单独报告并保持在一次backbone forward计算量的1%以内；
- 不得在部署态保存query评分不读取的完整协方差、重复old/new协方差或仅供审计的浮点矩阵；审计统计进入receipt，不进入预测状态。

中间门只筛掉没有功能作用或协议不合法的实现，不降低、替换或重新解释上述最终指标。

## 2.证据起点

|方法|已验证结论|对新研发的约束|
|---|---|---|
|D62|完整125；注册后仅24/375个场景状态实际激活，K1整体fallback|禁止用离散安全门把大多数row退回基线|
|D91|仅K10/new5 development；与D62的15/15 outer prediction相同|support目标下降或内部几何变化不等于分类功能|
|D92|K10/new20旧类+2.622pp、floor+4.600pp，但新类−0.653pp；K1逐值不变；实现会分别拟合old/new两套160×160 Ledoit-Wolf协方差，并在评分阶段不再读取这些矩阵|D92必须与DA联合重构；删除role分裂、重复全协方差和部署态无用矩阵；K1不得伪造不可辨识的类内方差|
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

## 4.联合候选小筛选与G0功能机制门

正式大矩阵前最多保留3条原理不同的联合候选。每条候选必须同时定义一个DA机制和一个精简D92机制，禁止只换层位却复用同一未审计头，也禁止固定D92后只开发DA。候选可以覆盖早层、中层和晚层干预，但层位只是机制差异的一部分，不构成层扫描。

三候选在不同GPU上并行运行同一份真实checkpoint、received-IQ、receiver-held×class/TX-LOCO折、K1/K5 support/query划分和评分器。全部6臂共享可复用的backbone、数学view、support索引及基础特征缓存；不得为每个head重复forward。每个候选只允许一次冻结实现，不扫描层、rank、步数、view、seed或门限。

预测生成和真实性能评分分离。打开truth前先完成以下G0功能检查；检查通过后在同一不可覆盖run中评分，不另建一套控制面：

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

## 5.G1 source-held最小6臂联合因果门

为同时识别DA收益、D92删改收益和联合交互，每个候选冻结`2种表示×3种head`的最小6臂：

|臂|表示|分类头|回答的问题|
|---|---|---|---|
|M0|基础表示|qKNN|共同基线|
|M_DA|新DA表示|qKNN|DA单独是否提升|
|M_D92|基础表示|历史D92|历史D92参照|
|M_DA_D92|新DA表示|历史D92|DA与历史D92是否相容|
|M_L92|基础表示|候选精简D92|D92删改本身是否提升|
|M_JOINT|新DA表示|同一候选精简D92|联合共设计是否提升|

`M_DA/M_DA_D92/M_JOINT`必须复用同一DA state；`M_L92/M_JOINT`必须复用同一精简D92公式；`M_D92/M_DA_D92`必须保持历史D92公式不变。缺少任一臂都会丢失至少一个必要反事实，不得用跨run结果补齐。

对指标`y`定义核心对比：

```text
DA_Q         = M_DA-M0
DA_D92       = M_DA_D92-M_D92
DA_L92       = M_JOINT-M_L92
L92_BASE     = M_L92-M_D92
L92_AFTER_DA = M_JOINT-M_DA_D92
I_CODESIGN   = L92_AFTER_DA-L92_BASE
```

小筛选是快速方向筛选，不设置Target级严苛门。候选进入排序只要求：

- `DA_Q`与`DA_L92`的池化`H`和old+new总正确数都严格增加；`A_old/N/F_old`任一项不得低于−0.5pp；这保证DA单独有效且接入精简D92后仍有效；
- `L92_BASE`与`L92_AFTER_DA`在K5的池化`H`和old+new总正确数都严格增加；`A_old/N/F_old`任一项不得低于−0.5pp；这保证D92删改本身有效且DA后仍有效；
- K5是历史D92真实激活和联合交互的主判据；K1主要检验极少样本DA，精简D92若启用则必须单独证明可辨识且相对历史fallback严格增益；
- `M_JOINT`相对`M0`的池化`H`和总正确数必须严格增加，任一receiver聚合`H`不得下降超过2pp；其余指标完整报告而不作为新的release gate；
- 候选先过性能门，再过§1效率门；排名按`min(DA_L92_H,L92_AFTER_DA_H)`最大、最差receiver退化最小、资源最低的词典序进行，不使用跨指标加权调参；
- `I_CODESIGN`必须报告；只要上述两个条件效应都为正，不额外强制其为正。

完整矩阵为负即关闭该候选，不从已打开结果修改层、rank、step、view、shrinkage或阈值。效率目标不阻塞小筛选启动，但未达到§1效率目标的候选不得成为最终胜者。若三候选均失败，返回原理研发并生成新方法，不在原矩阵上盲调参。

若某revision根据已经打开的source-held结果修改结构或超参数，该held立即降级为development。胜出候选冻结后必须使用尚未打开的source-held split完成一次fresh G1；不得把三候选小筛选的最好row冒充fresh证据。

## 6.G2单seed Target25门

候选、method lock和全部超参数在Target访问前冻结。矩阵固定：

```text
5 receivers × 1 seed ×
{K10/new5, K10/new10, K10/new20, K5/new20, K1/new20}
= 25 jobs
```

每个job覆盖3个物理ID互斥的`leo_*_weak`场景。一次Target25只评估一个revision，不得从25行中选择receiver、scene、class或slice重跑。

三候选小筛选和fresh G1使用§5的6臂完成归因。Target25只运行胜者的必要4臂`M0/M_DA/M_L92/M_JOINT`，不再重复`M_D92/M_DA_D92`；历史D92只从已完成125中按完全相同的`capsule_id/split_id/query_id_root/receiver/seed/K/new_count/scenario`键连接，键不全同则该paired比较无效且不得补跑同一D92 revision。

Target25完整性单位保持：

```text
25 jobs × 3 scenarios × 4 arms
= 300个scenario-arm pair
= 600个state prediction surface（before/after各300）
```

每个pair必须同时封存Stage2-B的before旧类预测和Stage2-C的after旧类/新类预测。任一state或arm缺失、预测可覆盖、truth先于全部600个state prediction surface开放、键不唯一、哈希不匹配或只完成联合臂，均不得进入性能分析。`forgetting`只能由同一pair的before/after旧类预测计算，不能从其他方法或其他run补入。

K10执行§1全部硬门。

K5以同receiver、同scene的K10/new20为matched基线。预登记时必须锁定`K5 support physical IDs⊂K10 support physical IDs`，且两者的`query_id_root`逐scene完全相同；只有满足该嵌套关系时，`A_old/F_old/N/H`下降≤5pp才属于paired结论。若不满足，只能报告非配对差值，不能用于通过本门。任一核心指标出现>15pp的单scenario-row灾难退化则不晋级。

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

修改结构、rank、loss、step、量化、阈值或fallback会清零确认资格并回到G0。

## 8.研发工作包与模型分工

|工作包|职责|执行模型|
|---|---|---|
|WP-DA|轻型共享DA、层位/状态可辨识性、K1/K5 support更新与资源|`gpt-5.6-terra/max`|
|WP-D92|历史D92计算图删改、类置换对称共享统计、K1边界及部署状态压缩|`gpt-5.6-terra/max`|
|WP-CODESIGN|DA与精简D92共享view/统计/缓存、6臂因果接口和候选内一致性|`gpt-5.6-terra/max`|
|WP-DATA|目标、矩阵、同row指标、结果分析、拒绝语义和交叉审查|`gpt-5.6-sol/high`|
|WP-INTEGRATE|协议解释、方案冻结、代码整合、独立复审和最终决策|主agent，`gpt-5.6-sol/high`|
|WP-IMPLEMENT|复杂科学核心实现、改变或实现新机制、需要科学判断的复杂缺陷修复|`gpt-5.6-terra/max`|
|WP-EXEC|冻结规格helper/test、Git/report、manifest、sync、固定命令启动/监控和artifact回收|`Luna/max`|

方法agent不得自我认证。WP-DATA必须审查K-shot可辨识性、common-transform cancellation、support proxy过拟合、旧/新任务平衡、类置换、资源和query/role/quota禁区。

每个功能包由不同agent拥有非重叠文件面。服务器实验另设唯一`Luna/max`runner；当commit、matrix、command、paths和health-stop完全冻结时默认使用`Luna/max`，仅在落地或调试需要复杂P0/P1判断或科学判断时例外使用`gpt-5.6-terra/max`。runner只负责落地、启动、健康检查、完整日志与artifact回收，不得改方法、调参、按性能重跑或与主agent重复启动。主agent和WP-DATA使用sol-high读取完整25-job/300-pair/600-state预测与评分证据后再作晋级决定。

## 9.拒绝语义

|状态|含义|
|---|---|
|`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|预测前系统性技术失败|
|`PARTIAL_DIAGNOSTIC_BIASED_NOT_PROMOTABLE`|只有partial prediction或score|
|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|完整矩阵完成但性能门失败|
|`TARGET25_SCREEN_PASS`|单seed25达到本轮硬目标|
|`SCREEN_POSITIVE_NOT_CONFIRMED`|单seed通过但确认seed失败|
|`PROMOTABLE`|G0–G3全部通过|

禁止按receiver、scene、class、seed挑选结果，禁止跨run拼接极值，禁止用source-held替代Target。

## 10.当前执行优先级

1.`D106-RCMR`真实588条K1/K5/K10 G0及source-held G1已有证据；`D121`、`D122`组合项、`D123`已关闭，旧run不重跑、不修旧通用release链；
2.`D106` Target25 r7仅完成46/600 state后技术退出，严格为`NO_PERFORMANCE_RESULT`；当前没有新的Target性能；
3.历史D92保留为固定参照，但不再视为最终头：K1整臂fallback、old/new双协方差、全矩阵求解和部署态无用矩阵是本轮明确删改对象；
4.当前最多并行3条“DA+精简D92”联合候选。浅层梯度残差只是候选之一；中层残差和晚层无反传超适配可以作为原理不同候选，但最终由联合6臂证据选择，不预设浅层获胜；
5.每条路线必须在设计时说明DA的唯一干预点/状态、精简D92的K1/K5共享统计、两者共享的view与缓存、删除的历史D92计算、预测状态与实测资源；
6.先完成一次Git承载的设计冻结和追踪映射，再由不同Terra Max agent实现非重叠科学核心；Luna Max只承担冻结helper/test、报告、manifest和唯一runner机械工作；
7.本地最小负测、真实checkpoint无query smoke、独立`P0=0、P1=0`和Git提交后立即发布三候选并行小筛选，不新增数据复验、authority或通用控制面；
8.完整小筛选为负的候选立即关闭。只允许一个胜者进入fresh63行G1；G1正向后直接进入单seed Target25，不启动D62/D92/SVRN重复125矩阵。
