# D92 CCOC严格Pareto设计

状态：`DESIGN_FROZEN / USER_APPROVED_FOR_AUTOMATIC_EXECUTION`

候选ID：`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`

candidate：`d92_e0_full_cross_class_offblock_consensus`

registered D mode：`ccoc_full`

## 1.目标与证据起点

CCOC只在`DA1_REG1`且`K>2`时激活。它保留当前D92已经验证的类均值、old/new任务等权、group-local auto-shrinkage FULL协方差、equal-prior单头和D42发布链，只收缩缺少跨类一致性证据的`160/96/32`块间协方差。`DA1_REG0`保持原D81/E0路径；K1和K2保持冻结D92 FULL精确alias。

设计依据不是“公式保证八项提升”，而是同排反证。D92相对E0已有七个方向改善，唯一反向为`new→old +0.058333pp`；TCRA只改变3/8方向且registration wall P90为`336.968ms`；CSOAS虽提高old floor和降低forgetting，却使`H`下降`0.4233pp`、seen-new下降`4.6667pp`、new→old增加`3.3241pp`。因此本轮不重新估计中心或类内尺度，不做注册后head补丁，而在D92 FULL与其block-diagonal端点之间作一次无参、old/new对称的统计收缩。

## 2.合法输入与禁止项

输入仅为同一outer合法K-shot support、标签、现有D81变换及D92已经构造的充分统计量。下文`x_ci`严格指与`Sigma_g^auto`处于完全相同坐标系的、D81变换后的288维support向量，禁止误用D81变换前feature。query及其任何view不得进入fit、更新、系数选择、回退、停止或G0机制判定；禁止clean/source样本、query truth、role Oracle、class quota和global reassignment。

禁止第二次FULL或BLOCK fit、LOO、Fisher、rank-one task contrast、旧类bias、候选/强度/步长扫描、逐边或逐prefix原子搜索、重复D42回缩，以及receiver、scene、seed、K、new-count或类别ID特判。P*型rank-one precision更新永久删除，不得以其他名称恢复。

## 3.冻结数学

维度`p=288`，块固定为`[0:160]`、`[160:256]`、`[256:288]`。先按每类float32行字节的词典序稳定排序，再以float64完成确定性归约。对第`c`类support行`x_ci`，使用现有D92非加权类均值`mu_c`，定义：

`r_ci=x_ci-mu_c`，

`S_c=(K-1)^-1 sum_i r_ci r_ci^T`，

`Q_c=S_c-blockdiag_160_96_32(S_c)`，

`u_c=Q_c/||Q_c||_F`。

`S_c`只估计off-block共识，绝不能作为协方差端点直接平均或求逆。每个注册类的`Q_c`都必须有限且`||Q_c||_F>0`；不设epsilon、不丢弃弱类、不按有效类子集重算。任一类失败时，K>2候选精确回退`E0_FULL_ONLY`。

对`g∈{old,new}`，令`I_g`为该组全部类别，`m_g=|I_g|`。有效公开矩阵中`m_old=6`、`m_new∈{5,10,20}`。冻结共识系数为：

`rho_g=clip((||sum_{c∈I_g}u_c||_F^2-m_g)/(m_g(m_g-1)),0,1)`。

该式等于组内所有不同类`u_c`的平均pairwise Frobenius cosine。内部通用helper在`m_g<2`时返回`rho_g=0`；公开CCOC路径仍要求合法注册矩阵。不得增加阈值、温度、人工权重或经验Bayes/SNR别名。

保留现有D92 group auto-shrinkage SPD端点`Sigma_g^auto`，定义：

`B_g=blockdiag_160_96_32(Sigma_g^auto)`，

`Sigma_g*=rho_g Sigma_g^auto+(1-rho_g)B_g`，

`Sigma*=0.5 Sigma_old*+0.5 Sigma_new*`。

最终使用现有D92类均值和equal prior执行一次288维FULL solve，随后沿用现有class-common居中和一次D42发布。`rho_g=1`回到D92 FULL端点，`rho_g=0`回到同一D92协方差的BLOCK端点。

## 4.SPD、对称性与确定性

`Sigma_g^auto`严格SPD时，其三个主子矩阵严格SPD，故`B_g`严格SPD；二者的凸组合`Sigma_g*`严格SPD，old/new固定`0.5/0.5`平均仍严格SPD。实现必须执行真实Cholesky检查；失败即回退，不得用伪逆、额外jitter或放宽容差掩盖。

方法必须满足：old组内与new组内label permutation等变、old/new任务整体交换不变、support行置换不变。类别标识只用于注册表绑定；任何tie或稳定排序只使用语义class handle与canonical row bytes，不能使用当前数组位置作为科学决策。

## 5.生命周期、回退与D42

K≤2：`active=false`、`fallback=false`、reason=`K1_K2_EXACT_D92_FULL_ALIAS`，输出与冻结D92 FULL byte-exact。REG0：不激活CCOC，不读取new support。

K>2正常路径：candidate FULL实际fit恰为1，额外BLOCK/LOO/Fisher fit均为0，dense solve恰为1，D42正式发布恰为1。任一off-block、rho、SPD、solve或codec数值失败都发布byte-exact `E0_FULL_ONLY`并标记`fallback=true`；异常回退允许额外执行E0 reference，但必须记录真实fit/codec库存且`G0_eligible=false`。结构、schema、registry或seal漂移继续抛错，不得伪装为数值回退。

G0必须从最终D42 state复核：candidate state与同outer、同scene的immutable E0 state不同；至少一个组满足`0<rho_g<1`；最终部署头相对配对E0在合法support上的跨组margin变化不小于一个可执行的真实D42量化量子。对每个support行`j`，以最终解码头定义`M_j=s(y_j)-max_{k in opposite_group(y_j)}s(k)`，并记录`max_j|M_j^CCOC-M_j^E0|`。对冻结三块`b`，令`A_b=max_j max_{d in b}|x_jd|`，令`q_b=A_b max(scale1_E0[:,b],scale2_E0[:,b],scale1_CCOC[:,b],scale2_CCOC[:,b])`，取所有非空块的`q=max_b q_b`。冻结门为`max_j|Delta M_j|>=q>0`。该定义与现有`_cross_group_margin_change_max_abs`和`_cross_group_margin_quantum`语义一致；不得改成仅要求非零。

为保持candidate实际FULL fit=1，G0用两个隔离的truth-free support-only技术执行分别产生E0 reference与CCOC state；reference执行不进入candidate fit库存或candidate wall/peak。validator只在两份state及support identity完全匹配后计算上述margin和量子，且不得读取query truth或用query预测选择候选。

## 6.资源实现

实现按old组、new组顺序流式处理并复用buffer，不保存`C×288×288`的`S_c/Q_c/u_c`。只保留三个上三角cross-block区域：`160×96`、`160×32`、`96×32`。在float64下，upper accumulators为`188,416B`，最大复用Q buffer为`122,880B`，K10 residual buffer为`23,040B`，冻结瞬时上界为`334,336B`，相对`512KiB`硬门保留`189,952B`余量。

query仍是单一F0线性头；query MAC、序列化字段和永久state bytes必须与E0逐row精确相同。support统计MAC必须按真实矩阵乘法、归一化、Cholesky和一次FULL solve给出保守上界；不得用estimated component-fit proxy替代实际wall/peak收据。

## 7.最小receipt契约

core、probe、slim和query至少传播以下语义，query层镜像为`d92_e0d_ccoc_*`：

- `active/fallback_active/fallback_reason/formula_revision`；
- `old_rho/new_rho`、old/new类数、每类off-block范数最小/最大值；
- `row_canonicalization`、`task_swap_invariant`、`within_group_label_permutation_equivariant`；
- `full_endpoint_reused`、`additional_covariance_fit_count=0`、`block/loo/fisher/scan_count=0`；
- `dense_full_solve_count=1`、`cholesky_pass`、`cholesky_diag_min`、trace delta；
- `support_optimization_macs_upper_bound`、`support_transient_bytes_upper_bound`、`persistent_state_bytes_delta=0`、`query_macs_delta=0`；
- `query_rows_used=0`及query fit/update/selection/truth/role/quota/global-reassignment全部为false；
- G0的paired E0 state SHA、candidate state SHA、部署跨组margin变化、真实D42量子和`quantum_pass`。

active receipt中两个rho必须有限并处于`[0,1]`；G0额外要求至少一个严格处于`(0,1)`。K≤2/REG0的rho与部署量子字段为`None`。数值fallback保留可得诊断，但`quantum_pass=false`且不得计作成功。

## 8.TDD与独立审查门

RED必须先证明新模块/arm/receipt不存在或行为错误，再写GREEN。核心行为测试使用手算fixture，覆盖pairwise-cosine公式、`rho=0/1`端点、SPD凸组合、零Q/nonfinite/Cholesky失败回退、row和label permutation、task swap、K1/K2 alias、单FULL库存、无额外fit、资源上界。集成测试覆盖真实D81→D92→D42路径、query零访问、最终state/MAC闭合、receipt篡改拒绝和codec数值回退；禁止只grep源码或断言mock存在。

本地发布门为聚焦测试与邻接D92/E0回归通过、`py_compile`和diff-check通过、真实checkpoint no-query smoke成功、独立审查`P0=0/P1=0`。P2只记录，不延迟G0。

## 9.G0、Hard9与Target125

G0固定使用`rx_7_7__seed_713106__k_10__new_5`的三个`leo_*_weak`场景，只执行truth-free prediction，不运行scorer。三场景都必须active、无fallback、SPD/receipt/量子门闭合、actual FULL fit=1、query/state/MAC精确；registration wall P90目标`≤120ms`且paired ratio`≤1.25×`，硬门`≤150ms`且`≤1.50×`，peak增量`≤512KiB`。

G0通过后，自动进入与G0不重叠的Hard9+K1。九个performance outer与E0逐row配对；`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new accuracy必须严格升高，average forgetting、new→old、old→new必须严格降低。任一tie、反向、fallback、证据不完整或资源硬门失败即`REJECT_ROUTE`，不扫描rho或任何替代权重。

Hard9全部通过后，按用户本轮授权自动创建新的不可覆盖Target125发布，不再等待同类流程性确认；仍须重新完成Git/report/独立审查/sole runner发布门。任何公式、协议、数据、矩阵或阈值的实质变化不属于自动批准范围。

## 10.声明边界

CCOC目前是冻结假设，不是性能结论。合成公式、单元测试、G0激活和资源通过都不能证明八项改善；只有Hard9同排truth-last分析能决定是否进入Target125，只有完整Target125能支持更广的性能陈述。
