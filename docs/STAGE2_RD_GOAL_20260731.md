# Stage2功能研发目标与证据门

状态：`ACTIVE / D130_COMPLETE_NEGATIVE / NEXT_R1_DESIGN_DRAFT / NO_TARGET_PERFORMANCE_RESULT`

## 0.2026-08-03闭环与目标重置

`D130-JOINT6-AFFINE-SCALE-R1`已在N607完成168/168条source-held LOCO prediction和独立score。CSPAR-2的K5 DA效应为`ΔH=-0.556pp、Δ总正确数=-9`；SRDH-2的K5 DA效应为0；两者均失败。D92-Lite160相对Full160的拟合解析MAC减少99.754%、显式峰值工作集减少90.607%，但`A_held_proxy`下降0.529pp、`F_retained`下降1.270pp，因此只有效率收益，没有满足联合目标的性能收益。两候选均关闭，不进入G0、fresh63、Target25或125，也不得调layer/rank/step/view/seed/shrinkage/阈值复活。

下一研发轮只允许一条原理不同的候选，工作名为`NEXT-R1 Fisher-Anchored Block Residual + Tail-Safe Lite`，当前仅处于`DESIGN_DRAFT`，不是冻结方法。它必须同时解决：

1.从特征空间共享变换转向参数空间局部残差，不更新全部checkpoint参数；
2.层选择不能预设为浅层。先用允许的Phase1数据计算各block的receiver敏感度、TX判别保持率、Fisher曲率和跨receiver方向一致性，预注册选择一个block或至多两个不重叠block；不得用Target query、truth或局部性能选层；
3.Phase1只联合封存类置换对称的低秩梯度/Fisher基和量化统计。Phase2只用当前K-shot support拟合至多4个共享残差系数，并用Fisher二次项约束更新；禁止encoder全参反向、checkpoint replacement、逐类adapter和query更新；
4.D92-Lite必须同步改为尾部安全的单仿射头：保留160维INT8/FP16 wire和低成本对角统计，以类对称的公共trust region约束support拟合相对冻结参考头的偏移；不得恢复old/new role分裂、两套稠密协方差、D62 row splice或按类门控；
5.设计必须先证明K1/K5可辨识性、公共变换对所有注册类的合法性、Fisher锚定与头部trust region不会把identity当收益，再进入实现。Fishr只作为Fisher重要性/曲率来源之一，不把“浅层Fishr”预设为答案。

NEXT-R1的最小性能矩阵固定为一个候选、42个receiver-held×seen-class-LOCO fold、`K∈{1,5}`，共84个candidate-row；每row只保留`R0Q/R0L/R1Q/R1L`四臂，删除D130已经证明不必重复的Full160代理臂。公共R0只计算一次。完整负结果立即关闭；完整正结果直接进入一个预注册单seed Target25 screen，不运行G0 588、fresh63或125。正式Target仍以完全同键的历史formal D92作外部基线。

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
|D62|历史target-capsule完整125诊断；K10/new20同row`A_old=68.68%、N=68.78%、H=68.56%、forgetting=17.34pp`，K1/new20为`44.03%/27.15%/33.41%/24.11pp`；注册后仅24/375个场景状态实际激活，K1整体fallback|禁止用离散安全门把大多数row退回基线；这些历史125不是当前定义的Target25|
|D91|仅历史K10/new5 development；15/15 outer prediction与D62逐值相同：`A_old=82.22%、N=84.67%、H=82.62%、forgetting=10.56pp`|support目标下降或内部几何变化不等于分类功能；不得外推到125或Target25|
|D92|历史target-capsule完整125诊断；K10/new20为`A_old=71.333%、N=68.150%、H=69.555%、forgetting=14.778pp`，相对D62主要改善旧类与遗忘、轻微降低新类；K1逐值不变。正式实现作用于288维D62/D81管线，在大量full/block/crossfit组件内分别拟合old/new收缩协方差，最终只部署单个紧凑仿射头；另一个160维source-held代理会额外保存两块协方差，不能混为正式资源证据|D92必须与DA联合重构；删除role分裂、重复稠密拟合、D62行拼接和无独立贡献的FFT96/RF32块；K1不得伪造不可辨识的类内方差|
|SVRN/r4.2|完整125且相对D62全面劣化|不再使用会放大注册后旧类崩塌的分支状态|
|D104|量化机制和release代码闭合，无Target性能|ANGQ只能作为实现组件，不能预设为最终分类头|
|D130|完整168条source-held LOCO方向性代理；CSPAR-2的K5 DA`ΔH=-0.556pp/正确数-9`，SRDH-2为零效应；Lite160拟合MAC减少99.754%、工作集减少90.607%，但held-proxy与最低类下降|两候选关闭；只复用数值缩放和低成本实现，不复用失败表示变换，不把效率收益写成性能收益|

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

## 4.D130已完成候选与原理边界

以下2条候选是D130的历史冻结设计，已完成且关闭，不再是当前研发入口。二者共享相同160维输入、相同support/query、相同六臂头部和评分器，不为各候选另造head。

|候选|机制|K5状态|K1边界|冻结理由|
|---|---|---|---|---|
|C1=`CSPAR-2`|Phase1联合封存rank2 nuisance轴`B`，Phase2用全类等权类内散度估计轴向收缩，形成非标量PSD度量|2个共享收缩系数|使用Phase1封存`alpha0`；只能称sealed metric benefit，不能称target support DA|闭式、低状态、无encoder梯度|
|C2=`SRDH-2`|Phase1封存rank2非线性响应字典`P/Q`及summary标准化统计，Phase2从全类support共享响应生成低秩残差|2个共享响应系数|用跨类共享summary，允许形成可辨识状态；不得含类专属参数|与PSD度量原理不同，可改变非线性邻域|

`RDCE-r3`与C1同属“Phase1轴＋target scatter PSD”族，历史D106 source-held小幅正收益不足以构成独立原理，继续关闭。CSPAR-2与SRDH-2也因D130完整负结果关闭。NEXT-R1必须按§0证明参数空间局部残差与尾部安全头的独立可辨识机制，不复用D127/D128 checkpoint replacement链，也不把浅层更新写死为唯一选择。

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

首轮候选矩阵固定为7个receiver×6个class的receiver-held×seen-class-LOCO共42折；每折构造新增Phase1资产时同时排除held receiver与held class，随后在held receiver的固定received-IQ上，把其余5个Phase1已见类记为`retained`组、held class记为`held-proxy`组，分别执行K1/K5六臂。checkpoint已经见过全部6个TX，因此held class绝不能重命名为注册新类；42折只产生方向性代理证据，不输出正式`N/H_old_new`，也不代替Stage2-C。K1严格为K5 support前缀，support/query物理ID互斥，且每折Phase1资产seal必须同时绑定held receiver、held class和420条Phase1-fit物理ID根。42折结果可以在全部prediction封存后按§5的代理主比较关闭明显负收益候选；不得据此修改资产、超参数或fold。最终资产按同一冻结公式重建一次。不得复用D127/D128的Phase1 autograd、checkpoint replacement或outer-audit发布链。

预测生成和truth评分分离。发布前只保留下列必要检查：

1.状态只读取不可变Phase1 bundle、当前row合法support和冻结配置；
2.query零fit、零selection、零update，每条query独立面对全部注册类；
3.类别标签置换等价，两个任务组使用同一规则；source-held代理不得产生正式old/new声明；
4.真实checkpoint无truth smoke至少改变feature、neighbor、margin或argmax之一；
5.C1若退化为共同平移、正交或全局正缩放并保持邻居排序，直接拒绝；C2若共享summary为零或残差无功能，直接拒绝；
6.序列化字节、拟合MAC、同机同线程时延、瞬时工作集和backbone forward次数形成receipt。

协议错误记为`INVALID / NO_PERFORMANCE_RESULT`；机制合法但无可观测决策作用记为`REJECT_REVISION_NO_FUNCTION`。

## 5.D130历史共享缓存六臂联合筛选（已完成）

本节记录D130已执行的冻结因果矩阵，供结果追溯；它不再定义NEXT-R1的活动矩阵。NEXT-R1按§0删除Full160代理臂，只保留一个候选和四臂。

每个候选只缓存两种表示：`R0=normalize(z_id160)`与`R1=normalize(phi_Ci(R0; support-only state))`。每种表示只做一次support/query特征缓存，再供三个头复用：

|头|定义|目的|
|---|---|---|
|Q|冻结Phase1-lock qKNN|DA因果基线|
|F=`D92-Full160`|当两组都含多个类时，在160维输入上复现历史D92的两组自动收缩full covariance、0.5/0.5平均、等先验与同一仿射中心化；5-retained/1-held的source-held矩阵只能使用明确标注的`single-class proxy extension`，不是历史D92严格复现|同表示机制对照；正式D92比较推迟到Target25|
|L=`D92-Lite160`|相同双组对称语义，但只拟合diagonal OAS并直接编译为仿射头；source-held结果只称proxy|验证删减稠密拟合的方向与成本|

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

仿射编译若遇到有限但超过FP16范围的截距，只允许对同一head的全部类别共同乘一个正2次幂，并同步缩放权重与截距；这在量化前严格保持argmax和类别置换等价，且不增加wire字节或query MAC。不得逐类clip/scale、fallback到Q或把它宣称为INT8/FP16量化后对任意query严格等价。若共同缩放会使任一非零权重行的逐类scale低于FP16最小正规数，必须确定性失败并关闭该数值实现。receipt需记录指数、缩放前后峰值、截距归零/子正规计数和明确的等价范围。

K1中F/L严格alias Q并保存等价receipt，不重复计算，也不提出head改进声明；K1只比较`R1Q-R0Q`。K5使用全部六臂，并用以下三个预注册主效应判断：

```text
DA_EFFECT       = R1Q-R0Q
LITE_BASE       = R0L-R0F
JOINT_REPLACE   = R1L-R1F
```

source-held矩阵的三个效应必须在同row池化后满足`ΔH_retained_held_proxy>0`且retained＋held-proxy总正确数严格增加；`ΔA_retained>=0`、`ΔA_held_proxy>=0`、`ΔF_retained>=0`作为方向性“不牺牲”条件。这些字段不得写成`A_old/N/H_old_new`。逐receiver和逐类完整报告，但不添加0.5pp级小样本门。真正的`A_old/N/H_old_new/F_old`以及相对历史formal D92的联合收益只在Target25同键评分。`R1L-R0L`、`R1F-R0F`和六臂交互项只作解释性结果，不增加发布gate。

### 5.1D130历史最小实验矩阵与停止规则

首轮只执行42个receiver-held×seen-class-LOCO fold×`{K1,K5}`，每候选84个原子row。C1与C2可分配到不同GPU，但必须共享真实checkpoint、固定received-IQ、物理ID清单、K前缀、代码commit和代理评分定义。每个fold中公共`R0Q/R0F/R0L`只计算一次并由两个候选引用同一cache/receipt，每个候选只补`R1Q/R1F/R1L`；不得增加18row Target development或其他中间矩阵。

所有prediction完成并封存后一次性打开source-held truth。候选若任一K5代理主效应失败，立即记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并关闭，不调层、rank、step、view、seed、shrinkage或阈值。若两候选都通过，只按`min(DA_EFFECT_H_proxy,LITE_BASE_H_proxy,JOINT_REPLACE_H_proxy)`、最差receiver代理联合增益、总正确数和端到端资源作冻结排序，选择一个进入G0/G1的方向性胜者；这不是Stage2-C晋级或正式正收益结论。若两候选都失败，本revision立即以完整负结果结束，不进入G0/G1/Target；下一研发轮最多补1条原理不同的新候选，必须先给出可辨识性与协议合法性推导，不得修改或重跑本轮失败候选。

### 5.2D130历史发布硬门

发布前只要求：实际Git方法入口；query零fit/update/selection及禁止clean/source/query-truth/role/quota/global-reassignment的聚焦负测；真实checkpoint-derived received-IQ archive no-query smoke；独立复核`P0=0/P1=0`；不可覆盖run ID/output；本地Git提交；N607预检与资源记录。既有`VALIDATED_ONCE`数据不因方法变化重验，不要求新签名层、通用执行平台、完整论文叙事或重复D62/D92/SVRN矩阵。source-held proxy发布只需保存解析字节/MAC和原始时延/工作集receipt，不以正式90%/50%/40%阈值阻塞；这些阈值必须由Target25同机同线程重复测量的中位数和实际峰值判定。

## 6.D130原定G0→G1→Target25路径（已取消）

D130没有胜者，因此本节全部后续步骤均未触发并已取消。内容仅保留为预注册历史，不得据此发布G0、fresh63或D130 Target25。

小矩阵只负责选出一个联合胜者。胜者保持同一method lock，按以下顺序扩展；失败即停止，不回头从已打开性能中修改候选：

1.`G0`：在既有588条Phase1功能面闭合非恒等、量化parity和两份资源receipt；不读取Target truth，不把功能变化写成性能收益；
2.`G1`：只运行一次未参与本轮设计的fresh63 source-held六臂矩阵；它仍是Phase1已见类代理，只检查跨receiver/class方向与负迁移，不输出正式new-registration指标，也不替代Target Stage2-C；
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

当前NEXT-R1轮的第一终点是完成理论设计、独立复核并冻结一个候选；第二终点是一次完整84-row四臂source-held矩阵。若该矩阵失败，本轮结束；若通过，默认最终终点是一个完整、预注册、单seed的`TARGET25_SCREEN_PASS`，不再插入G0 588、fresh63、125或自动追加第二个Target seed。需要对外形成多seed`PROMOTABLE`声明时另行预注册confirm seed；它不属于当前发布硬门，也不能延迟首个Target25。任何结构、block、rank、Fisher统计、trust region、量化、阈值或fallback修改都产生新revision，不得借用旧prediction。

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
5.D130的CSPAR-2与SRDH-2已完整失败并关闭；共同正2次幂FP16修复和Lite160低计算实现可复用，但不得据此宣称性能正收益；
6.当前方法目标是§0的NEXT-R1设计推导：先完成block选择准则、低秩Fisher残差、K1/K5可辨识性和Tail-Safe Lite公共trust region；在`DESIGN_FROZEN`前不启动实验；
7.冻结后只实现一个候选和`R0Q/R0L/R1Q/R1L`四臂。真实checkpoint no-truth smoke、聚焦协议负测、独立`P0=0/P1=0`、Git提交完成后立即发布84-row必要矩阵，不重验数据、不建通用平台；
8.NEXT-R1完整小矩阵任一预注册联合主比较失败即关闭，不调参复活；通过才直接进入一个单seed Target25 screen，不运行G0 588、fresh63、125或重复D62/D91/D92/SVRN矩阵。
