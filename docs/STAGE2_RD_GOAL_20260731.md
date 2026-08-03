# Stage2功能研发目标与证据门

状态：`ACTIVE / JOINT6_DESIGN_FROZEN / IMPLEMENTATION_PENDING / NO_NEW_PERFORMANCE_RESULT`

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

## 4.冻结候选与原理边界

本轮冻结2条而不是强凑3条候选；二者共享相同160维输入、相同support/query、相同六臂头部和评分器，不为各候选另造head。

|候选|机制|K5状态|K1边界|冻结理由|
|---|---|---|---|---|
|C1=`CSPAR-2`|Phase1联合封存rank2 nuisance轴`B`，Phase2用全类等权类内散度估计轴向收缩，形成非标量PSD度量|2个共享收缩系数|使用Phase1封存`alpha0`；只能称sealed metric benefit，不能称target support DA|闭式、低状态、无encoder梯度|
|C2=`SRDH-2`|Phase1封存rank2非线性响应字典`P/Q`及summary标准化统计，Phase2从全类support共享响应生成低秩残差|2个共享响应系数|用跨类共享summary，允许形成可辨识状态；不得含类专属参数|与PSD度量原理不同，可改变非线性邻域|

`RDCE-r3`与C1同属“Phase1轴＋target scatter PSD”族，历史D106 source-held小幅正收益不足以构成独立原理，本轮关闭。暂不增加第三条SCPM或浅层梯度候选；新增候选必须先证明不复用D127/D128 checkpoint replacement链，且确有独立可辨识机制，否则属于非必要扩张。

C1在Phase1只保存`B∈R^(160×2)`及冻结`alpha0/alpha_max/eps`。K5按全部注册类等权估计：

```text
S_w = [C(K-1)]^-1 Σ_c Σ_i (u_ci-μ_c)(u_ci-μ_c)^T
v_j = b_j^T S_w b_j
v_perp = [tr(S_w)-Σ_j v_j]/158
α_j = clip(1-(v_perp+eps)/(v_j+eps), 0, alpha_max)
φ_C1(u) = normalize((I-B diag(α) B^T)^(1/2)u)
```

C2固定：

```text
a_j = a_max tanh(([C^-1 Σ_c K^-1 Σ_k tanh(q_j^T u_ck)]-m_j)/d_j)
φ_C2(u) = normalize(u + P[a ⊙ tanh(Q^T u)])
```

`B/P/Q/m/d/a_max`只能来自与checkpoint联合封存的允许Phase1聚合知识；Phase2不读取source/clean样本或FP32 sidecar。C1/C2都不得读取query、truth、role、class quota或全局批次计数，也不得包含TX/class ID专用参数。

Phase1科学审计固定为7个receiver×6个class的receiver-held×class-LOCO共42折；每折同时排除held receiver与held class，K1严格为K5 support前缀，outer只验证物理ID隔离、类置换等价、非零功能和跨receiver/class可迁移性，不用outer分数选资产或超参数。最终资产按同一冻结公式重建一次。不得复用D127/D128的Phase1 autograd、checkpoint replacement或outer-audit发布链。

预测生成和truth评分分离。发布前只保留下列必要检查：

1.状态只读取不可变Phase1 bundle、当前row合法support和冻结配置；
2.query零fit、零selection、零update，每条query独立面对全部注册类；
3.类别标签置换等价，old/new使用同一规则；
4.真实checkpoint无truth smoke至少改变feature、neighbor、margin或argmax之一；
5.C1若退化为共同平移、正交或全局正缩放并保持邻居排序，直接拒绝；C2若共享summary为零或残差无功能，直接拒绝；
6.序列化字节、拟合MAC、同机同线程时延、瞬时工作集和backbone forward次数形成receipt。

协议错误记为`INVALID / NO_PERFORMANCE_RESULT`；机制合法但无可观测决策作用记为`REJECT_REVISION_NO_FUNCTION`。

## 5.共享缓存六臂联合筛选

每个候选只缓存两种表示：`R0=normalize(z_id160)`与`R1=normalize(phi_Ci(R0; support-only state))`。每种表示只做一次support/query特征缓存，再供三个头复用：

|头|定义|目的|
|---|---|---|
|Q|冻结Phase1-lock qKNN|DA因果基线|
|F=`D92-Full160`|在160维输入上严格复现历史D92的old/new两组自动收缩full covariance、0.5/0.5平均、等先验与同一仿射中心化|同表示历史D92机制对照|
|L=`D92-Lite160`|相同old/new对称语义，但只拟合diagonal OAS并直接编译为仿射头|验证删减稠密拟合是否增益且更快|

因此每个候选的最小完整矩阵固定为：

|臂|表示|头|
|---|---|---|
|R0Q|基础160维|qKNN|
|R0F|基础160维|D92-Full160|
|R0L|基础160维|D92-Lite160|
|R1Q|适应后160维|qKNN|
|R1F|适应后160维|D92-Full160|
|R1L|适应后160维|D92-Lite160|

F/L必须部署为同一wire：`W_q[C,160]+FP16 scale[C]+FP16 intercept[C]`，序列化均为`164C`字节，query端均为`160C`MAC；Lite的效率主张仅来自拟合时延、峰值瞬时工作集、dense matrix/solve次数，不虚构部署态差异。正式288维`D92-Formal`继续作为同row外部全管线参考和资源参考，不进入六臂head因果结论。

K1中F/L严格alias Q并保存等价receipt，不重复计算，也不提出head改进声明；K1只比较`R1Q-R0Q`。K5使用全部六臂，并用以下三个预注册主效应判断：

```text
DA_EFFECT       = R1Q-R0Q
LITE_BASE       = R0L-R0F
JOINT_REPLACE   = R1L-R1F
```

三个效应都必须在同row池化后满足`ΔH>0`且old＋new总正确数严格增加；`ΔA_old>=0`、`ΔN>=0`、`ΔF_old>=0`作为“不牺牲”条件。逐receiver、逐scene和逐类完整报告，但不添加0.5pp级小样本门。`R1L-R0L`、`R1F-R0F`和六臂交互项只作解释性结果，不增加发布gate。

### 5.1最小实验矩阵与停止规则

科学审计先执行42个receiver-held×class-LOCO fold×`{K1,K5}`，只验证功能、隔离和可迁移性；C1与C2可分配到不同GPU，但必须共享checkpoint、物理ID清单、K前缀、代码commit和评分定义。随后Target development one-shot固定为seed`713102`、receiver`{20-1,3-19,7-14}`×`{K1/new20,K5/new20}`×3个互斥LEO弱场景，共18个原子row；公共`R0Q/R0F/R0L`只计算一次，每个候选补`R1Q/R1F/R1L`。

所有prediction完成并封存后一次性打开truth。候选若任一K5主效应失败，立即记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并关闭，不调层、rank、step、view、seed、shrinkage或阈值。若两候选都通过，只按`min(DA_EFFECT_H,LITE_BASE_H,JOINT_REPLACE_H)`、最差receiver联合增益、总正确数和端到端资源作冻结排序，选择一个胜者。

### 5.2发布硬门

发布前只要求：实际Git方法入口；query零fit/update/selection及禁止clean/source/query-truth/role/quota/global-reassignment的聚焦负测；真实checkpoint no-query smoke；独立复核`P0=0/P1=0`；不可覆盖run ID/output；本地Git提交；N607预检与资源记录。既有`VALIDATED_ONCE`数据不因方法变化重验，不要求新签名层、通用执行平台、完整论文叙事或重复D62/D92/SVRN矩阵。

## 6.胜者后的G0→G1→Target25

小矩阵只负责选出一个联合胜者。胜者保持同一method lock，按以下顺序扩展；失败即停止，不回头从已打开性能中修改候选：

1.`G0`：在既有588条Phase1功能面闭合非恒等、量化parity和两份资源receipt；不读取Target truth，不把功能变化写成性能收益；
2.`G1`：只运行一次未参与本轮设计的fresh63 source-held六臂矩阵；它检查跨receiver/class方向与负迁移，不替代Target Stage2-C；
3.`G2`：G0/G1均闭合后，只运行一次单seed Target25，执行本节K10/K5/K1最终门。

不得为落地G0/G1重建通用发布平台或重复历史D62/D92/SVRN矩阵。两份资源receipt分别为：

- `head_causal_resource_receipt`：同160维、同INT8 affine wire下比较Full160/Lite160的K5拟合墙钟、峰值工作集、dense elements及factorization/solve次数；
- `system_formal_replacement_resource_receipt`：比较formal288 D92的`1152+590C`与Lite160的`164C`字节，并显式标记`representation_pipeline_changed / not_head_causal`。

### 6.1单seed Target25门

候选、method lock和全部超参数在Target访问前冻结。screen seed必须排除development seed`713102`，并从已完成D92 Target125中按数值升序选择第一个未参与本轮小矩阵/fresh63候选评分且具备全部同键D92 artifact的seed；完整性在打开该seed的本轮候选prediction前只读核验，若缺失则顺延到下一个完整seed，不得读取本轮性能后改选。它是`METHOD_UNSEEN_SCREEN`，不是全项目从未读取truth的盲测。矩阵固定：

```text
5 receivers × 1 seed ×
{K10/new5, K10/new10, K10/new20, K5/new20, K1/new20}
= 25 jobs
```

每个job覆盖3个物理ID互斥的`leo_*_weak`场景。一次Target25只评估一个revision，不得从25行中选择receiver、scene、class或slice重跑。

§5小矩阵及§6的G0/fresh63完成方向筛选。Target25运行胜者的六个逻辑臂`R0Q/R0F/R0L/R1Q/R1F/R1L`；K1的F/L按等价receipt alias Q，不重复计算。历史formal D92只从已完成125中按完全相同的`capsule_id/split_id/query_id_root/receiver/seed/K/new_count/scenario`键连接，不重跑D92。键不全同则该seed不得用于本轮K1 paired目标，应在prediction前改选另一个具备完整同键D92 artifact且本方法未见的seed。

Target25完整性单位保持：

```text
25 jobs × 3 scenarios × 6 logical arms
= 450个scenario-arm pair
= 900个state prediction surface（before/after各450）
```

每个pair必须同时封存Stage2-B的before旧类预测和Stage2-C的after旧类/新类预测。任一state或arm缺失、预测可覆盖、truth先于全部900个state prediction surface开放、键不唯一、哈希不匹配或只完成联合臂，均不得进入性能分析。K1 alias必须有逐logit等价receipt；`forgetting`只能由同一pair的before/after旧类预测计算，不能从其他方法或其他run补入。

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

## 7.本轮完成边界

本轮默认终点是一个完整、预注册、单seed的`TARGET25_SCREEN_PASS`，不是再跑125或自动追加第二个Target seed。需要对外形成多seed`PROMOTABLE`声明时，另行预注册confirm seed；它不属于当前发布硬门，也不能延迟G0、fresh63或首个Target25。任何结构、rank、统计、量化、阈值或fallback修改都产生新revision，不得借用旧prediction。

## 8.研发工作包与模型分工

|工作包|职责|执行模型|
|---|---|---|
|WP-DA|轻型共享DA、层位/状态可辨识性、K1/K5 support更新与资源|`gpt-5.6-terra/max`|
|WP-D92|历史D92计算图删改、类置换对称共享统计、K1边界及部署状态压缩|`gpt-5.6-terra/max`|
|WP-CODESIGN|DA与精简D92共享view/统计/缓存、六臂因果接口和formal参照边界|`gpt-5.6-terra/max`|
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
|`PROMOTABLE`|另行授权的多seed确认完成；不属于本轮默认硬门|

禁止按receiver、scene、class、seed挑选结果，禁止跨run拼接极值，禁止用source-held替代Target。

## 10.当前执行优先级

1.`D106-RCMR`真实588条功能面及source-held结果只作为历史非晋级证据；`D121`、`D122`组合项、`D123`已关闭，旧run不重跑、不沿用其G0/G1流程、不修旧通用release链；
2.`D106` Target25 r7仅完成46/600 state后技术退出，严格为`NO_PERFORMANCE_RESULT`；当前没有新的Target性能；
3.历史formal D92保留为固定参照，但不再视为最终头：K1整臂fallback、288维D62/D81管线、old/new重复稠密拟合和row-splice计算是本轮明确删改对象；160维held代理的额外协方差状态只作独立工程诊断；
4.D127/D128全部是prediction前技术停止，没有性能结论；其Phase1 autograd/checkpoint-replacement实现路线已按预注册规则关闭，不再修复、不创建新run；
5.当前方法目标是§4的CSPAR-2与SRDH-2，不强凑第三条；两者禁止encoder梯度、checkpoint replacement、`β`logit融合或类专属状态；
6.先实现两种共享表示、`Q/Full160/Lite160`三头和完整六臂；不得复制D127三候选merge或Phase1 outer-audit发布链；
7.真实checkpoint no-truth smoke必须证明Gram、neighbor或margin发生非数值噪声变化；通过聚焦测试、42-fold覆盖、独立`P0=0/P1=0`、Git提交后立即发布小矩阵，不重验数据；
8.任一候选完整小矩阵中一条K5主比较失败即关闭，不调参复活；胜者才顺序进入588 G0、一次fresh63和单seed Target25，不运行125或重复D62/D92/SVRN矩阵。
