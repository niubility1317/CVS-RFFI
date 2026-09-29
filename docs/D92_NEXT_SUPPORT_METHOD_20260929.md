# 下一项support机制研究：D92-BranchMetric-v1

日期：2026-09-29。状态：**设计已定稿，获准进入核心与入口实现；未运行真实实验**。本设计保持Phase1及现有五块特征不变，唯一新增机制是由当前support的类内残差确定共享度量，再按该度量下的最近类中心判决。所有类、所有K使用同一公式，不增加source信息或query反馈。

## 1. 依据与备选路线

本次仅读取冻结方法源码、既有设计和获准support-only结果；未读取正式benchmark报告、query成绩、handoff或方法排名，也未接收任何query结果反馈。支持证据为`20260929-phase2-d92-branch-interaction-support-m4-r01/results/support_summary/summary.json`：4800个episode，包括1200个K1数值诊断和3600个完整物理OOF。含新类任务中，交互核相对能量对照的支持均值差如下，单位为百分点：

| parent K | 旧类差 | 新类差 | H差 |
|---:|---:|---:|---:|
| 5 | +0.358 | +0.867 | +0.757 |
| 10 | +0.375 | +1.102 | +0.888 |
| 20 | +0.683 | +1.614 | +1.368 |

这支持继续沿用已预设的交互表示，并不证明本设计的度量假设，更不提供正式K1性能证据。重复draw复用物理样本，不是独立重复。

| 路线 | 真正改变的机制 | 主要限制 | 决定 |
|---|---|---|---|
| 同一交互核上的类内度量与最近类中心 | 从总散度onehot回归改为仅按类内变化压缩方向，并使用类中心范数截距 | 共享类内结构未必适合所有类；小样本残差可能是偶然变化 | **唯一下一候选** |
| 同一核上的multiclass logistic或margin头 | 从平方损失改为竞争分类损失 | 需固定迭代/停止规则，K1容易分离但未必泛化；当前证据未指向平方损失就是瓶颈 | 本次不实现 |
| 支持验证驱动的分支可靠性权重或核权重 | 学习五块特征的相对权重 | 真K1没有类内held样本；多个权重增加选择自由度，容易把小support偶然性当可靠性 | 本次不实现 |

选定机制针对源码可确认的差异，而不是假定现有方法已经出现某类query失败。共享度量和最近中心是已知判别思想；通常LDA的共享协方差与Mahalanobis最近中心解释可见[scikit-learn官方说明](https://scikit-learn.org/stable/modules/lda_qda.html#lda)。本设计使用自己的固定RKHS类内散度正则化，不声称采用该库的自动shrinkage估计器、符合高斯分布，或发明新的判别理论。

## 2. 唯一冻结表示与散度口径

沿用[BranchInteraction设计](D92_BRANCH_INTERACTION_DESIGN_20260929.md)：

\[
B=u([u(z),4u(FFT96)]),\qquad
A=[u(t),u(f),u(p)]/\sqrt3,
\]

\[
k(x,x')=B_x^TB_{x'}+A_x^TA_{x'}+
(B_x^TB_{x'})(A_x^TA_{x'}).
\]

这里\(u(v)=v/\max(\|v\|_2,10^{-12})\)，FFT96计算保持原实现。隐式特征为\(\phi(x)=[B,A,\mathrm{vec}(BA^T)]\)，维数123616。通常每条特征能量为3，零块或norm floor会降低实际能量；不整体重新归一化，不按维数、trace或每类方差再缩放。

每row有C个实际注册类、每类K个互异物理support，N=CK。只用本次fit的support均值\(\bar\phi\)中心化，令\(x_i=\phi_i-\bar\phi\)、\(\mu_c=K^{-1}\sum_{i:y_i=c}x_i\)。定义

\[
S_W=\sum_{c=1}^C\sum_{i:y_i=c}(x_i-\mu_c)(x_i-\mu_c)^T,
\qquad M=(I+S_W)^{-1}.
\]

**S_W是物理残差外积之和，λ固定为1。**不除以N、K、N−C或特征维数，不在换成均值散度后偷偷保留同一λ。因为特征已无量纲且范数受限，λ=1是在该固定特征单位下的各向同性正则；其数值继承既有ridge1的固定尺度，但不声称这使两种算法具有相同的统计正则强度。

这不是无偏协方差估计，也不是证明真实类内噪声服从某分布。它是一项明确的收缩度量假设：合法support中观察到更大的共享类内变化时，沿相应方向降低距离权重；未观察到的方向保留单位度量。K增大时残差证据按实际物理数量累积，公式不切换。

候选分数为

\[
s_c(q)=x_q^TM\mu_c-\tfrac12\mu_c^TM\mu_c,
\qquad x_q=\phi(q)-\bar\phi.
\]

它等价于最小化\((x_q-\mu_c)^TM(x_q-\mu_c)\)。省略的query二次项对所有类相同。所有类等先验，不使用old/new身份、具体TX ID、类配额、真实query类别集合、类专属超参数或分数温度。类中心范数项必须保留，不能只用内积后宣称最近中心。

## 3. 与现有ridge的机制区别

令X为中心化support特征矩阵，U为以\(\mu_c\)为列的矩阵。均衡K下，\(S_T=X^TX=S_W+KUU^T\)。现有onehot-minus-1/C核岭等价于

\[
W_{ridge}=K(S_W+KUU^T+I)^{-1}U
=K M U(I+K U^TMU)^{-1}.
\]

候选则使用\(MU\)和\(-\tfrac12\mathrm{diag}(U^TMU)\)截距。因此候选去掉了ridge中由所有类中心共同构成的右侧竞争变换，并引入完整最近中心距离的范数项。它不等价于重复特征、统一分数放大或只换一个ridge值。

这种变化未必更好：ridge的类间耦合可能正有利于区分邻近类别，去掉它可能降低判别性。当前support证据仅证明交互表示值得保留，尚未验证“类内度量优于总散度回归”。下一诊断必须直接检验这一点。

## 4. 严格dual公式、K1与数值边界

不显式展开123616维特征。训练Gram为G，\(H=I-11^T/N\)，\(G_c=HGH\)。类别指示矩阵\(Z\in\{0,1\}^{N\times C}\)，每行一个1；定义

\[
P=Z/K,\qquad R=I-ZZ^T/K.
\]

R对每个类内部减去该类support均值，满足\(R=R^T=R^2\)、\(R1=0\)；它**只作用于训练support**。于是\(U=X^TP\)、\(S_W=X^TRX\)。Woodbury恒等式给出

\[
Q=(I+RG_cR)^{-1}(RG_cP),\qquad V=P-RQ,
\]

\[
MU=X^TV,\qquad
d=\mathrm{diag}(P^TG_cV),\qquad b=-\tfrac12d.
\]

所有中间矩阵仅由本次train support、注册类和标签生成。对单条待判别样本q，

\[
k_c(q,S)=k(q,S)-\overline{k(q,S)}1^T
-\tfrac1N1^TG+\tfrac1{N^2}1^TG1\,1^T,
\qquad s(q)=k_c(q,S)V+b.
\]

\(\overline{k(q,S)}\)只对这一条q与固定support的核值取均值，是预先固定的中心化映射，完全不依赖其他query。query没有真实类别，不能对其执行“按类中心化”；R只在训练状态中出现，不允许由预测标签或query角色构造query端R。每条query独立面对所有C类，状态只读。

实现应沿用reference-difference形式的全局核中心化。RG_cR可通过训练标签分组均值计算，不必做两次稠密N阶矩阵乘法；但数学定义必须完全相同。求解使用float64 Cholesky，单位阵保证\(I+RG_cR\)最小特征值至少为1；上界\(1+\mathrm{tr}(RG_cR)\le1+3N\)。不需要特征值截断、伪逆阈值、自动λ或数值失败后的静默回退。

K1时，每类恰好一条support，\(ZZ^T=I\)、R=0，因而V=P=Z，M=I：候选**自然退化为同一交互核的最近单样本原型**。普通单位块下所有原始自核为3，argmax进一步等价于最大原始交互核相似度；存在零块时，原型范数项仍必须保留。此退化不同于现有核岭头，故K1存在真实机制变化，但没有任何类内方差被估计。

C=1及所有support特征完全相同均合法；后者的中心化类中心严格为0，分数精确为0。仅可对数学退化做精确相等处理，不设经验阈值改变一般样本。物理class ID字典序统一打破平局，support顺序按physical ID规范化。

## 5. 最小对照与两类support诊断

固定三臂，表示完全相同：

| 臂 | 公式 | 区分的贡献 |
|---|---|---|
| `interaction_ridge` | 已冻结BranchInteraction，λ=1 | 当前直接对照 |
| `kernel_ncm` | M=I，V=P，保留\(-\tfrac12\mathrm{diag}(P^TG_cP)\) | 最近中心判决本身 |
| `within_metric` | 本文固定类内度量与范数截距 | 相比NCM增加类内残差度量 |

不增加核系数/λ网格，不按row选择臂，也不混合三臂分数。K1时后两臂必须逐值或按严格数值容差等价，这是实现正确性要求，不能要求其性能严格彼此提升。

**A. 原support任务的完整物理OOF。**沿用既有按类physical ID排序位置模\(\min(K,3)\)的分配。K=5/10/20每折三臂使用相同train/held physical。所有全局中心、类中心、R、Gram、V/b均从train重新计算。实际train K与parent K都记录。K1保持`oof=null`、`K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT`，不把同一记录的view当独立持出。

**B. 原任务support内部的1-shot训练代理诊断。**只对parent K≥2的当前row合法support做拆分。对每个类，将本row已有physical ID稳定排序；依次令anchor_index=0…parent_K−1，各类选相同排序位置的一条记录作为训练support，其他parent_K−1条作为独立的`held_support`。所有anchor预先穷尽，不按表现挑选。每个proxy fit只有C条记录，实际`train_k=1`；held记录的标签只在预测固定后计算support指标，绝不进入中心、度量、参数或选择。

这不修改capsule，不新增received观测，也不让真实K1任务借用更大support池。明确保存proxy专属scope、`parent_k`、`proxy_train_k=1`、`train_k=1`、`held_k=parent_k-1`、`trial`、train/held physical IDs；入口在外层保留本次`parent_split_id`。它回答“这些K≥2任务已有support中，单条注册时能否预测独立support”，不等于原正式K1划分或query效果。

在当前4800个episode矩阵上，B只用K5/10/20的3600个parent任务，最多42000个固定anchor子任务。每个子任务的核岭求解维数仅C≤26；candidate与NCM在K1退化下不需Cholesky。不同anchor和不同support draw重用物理记录，应先在parent内汇总，再按既定receiver/model/scene/new_count口径汇总，不能将重复held次数当独立样本量。保存紧凑的support-only预测或等价计数以及每anchor、每parent指标，避免重复大数组日志。

诊断A逐parent K报告旧类、新类、H及candidate相对两对照差；B逐parent K单独报告1-shot代理指标，不与A混合或改标为正式K1。NLL仅作为固定原分数softmax的未校准描述，不用NLL选择温度或方法。

建议在诊断前登记同一支持筛选口径：A中各parent K=5/10/20，candidate相对两对照的新类及H均为正、旧类退化不超过1个百分点；B中各parent K的candidate相对interaction_ridge的新类/H为正且旧类满足同一护栏，candidate与NCM相等。新类/H只在new_count>0任务定义，old-only另报。以上是候选研究的预设解释标准，不是额外数据审批或正式query泛化证明；若未满足，如实归因，不能从同一结果继续扫λ或选择性重跑后隐去失败。

若candidate胜ridge但不胜NCM，证据只支持最近中心判决，不能归因于学习类内度量。若A改善而B不改善，则不能声称覆盖1-shot。若B改善，也仍不能代替真实K1完整benchmark；任何正式query成绩都不得反馈本方法研发。

## 6. 失败解释、成本与验证边界

最重要的可反驳假设是：不同类别在当前接收条件下有足够共享的类内变化，而且这些变化可以用有限support的残差方向概括。它可能因以下原因失败：类内散度夹带身份判别方向；旧类与新类的噪声方向不共享；K较小时残差方向偶然；未观测方向保持单位度量却包含真实噪声；移除ridge类间耦合损害相邻类竞争；交互核放大共同信道变化；类中心不能代表多模态类别。等先验和每类同公式只能避免人为角色偏置，不能保证两类群体同等获益。

每row约\(O(N^2\cdot736+N^3+N^2C)\)训练和\(O(N\cdot736+NC)\)单条推理，与现有交互头同阶。candidate一般一次N阶Cholesky；NCM零次，ridge对照一次。只保留归一化support B/A、V[N,C]、reference-kernel[N]、center-mean[N]、b[C]及两个中心化float64标量，不保留R、Q、Cholesky、类内散度大矩阵或query。

数值状态口径为

\[
8\{N(736+C+2)+C+2\}\ \mathrm{bytes}.
\]

C=26、K=20时为3178464 B，与现有交互头同一数值布局量级；不是原736维线性头的153296 B。类表、audit、Python对象、原cache和模型包开销另记；此处尺寸已由合成nbytes检查核对，没有真实数据实测速度。既有合法support缓存足够，无新checkpoint加载、native forward或source payload。

实现后的必要验证应包括：小维显式展开与dual一致；R的类内中心化及S_T分解；最近中心距离与带范数截距分数等价；K1=NCM、零/恒定/C1退化；所有状态只来自train fold；parent/actual train K与physical互斥；类别置换和old/new角色不影响拟合；query逐条/分块/重排相同；状态不可写；实际nbytes、有限值、配置和错误输入。只运行这些必要的合成正确性检查，不因文档本身启动真实训练或实验。

## 7. API与冻结配置

root已分配实现，核心位于`code/cvsrffi/d92_branch_metric.py`。主API为：

```python
fit_branch_metric(*, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes=(),
    arm='within_metric') -> BranchMetricState

probe_branch_metric(*, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes=()) -> dict

state.score(*, z_id, fft, t_emb, f_emb, pa_local)
state.predict(**features)
state.audit_dict()
```

`fit`的arm仅接受`within_metric`和固定消融`kernel_ncm`；probe固定调用既有BranchInteraction实现作为`interaction_ridge`对照。probe一次返回标准物理OOF与独立`oneshot_proxy`，两者分开汇总，不根据结果切换；真实K1的两者均为null。probe结果不返回可部署状态；正式fit永远使用一次全support的`within_metric`。

冻结配置的关键字段摘录如下，完整内容以算法JSON与core.FROZEN_CONFIG的一致性检查为准：

```json
{
  "method": "D92-BranchMetric-v1",
  "schema": "d92_branch_metric_v1",
  "views": "original_received_only",
  "kernel": "KB+KA+KB*KA",
  "background": "unit(concat(unit(z_id),4*unit(fft96)))",
  "auxiliary": "concat(unit(t_emb),unit(f_emb),unit(pa_local))/sqrt(3)",
  "norm_floor": 1e-12,
  "within_scatter": "sum_physical_class_centered_outer_products",
  "within_scatter_denominator": 1,
  "ridge_coefficient": 1.0,
  "metric": "inverse(I+within_scatter)",
  "score": "centered_query_metric_class_mean_minus_half_class_mean_metric_norm",
  "class_prior": "uniform_all_registered_classes",
  "class_norm_intercept": true,
  "centering": "train_support_only_reference_difference_then_mean",
  "solver": "float64_Cholesky_I_plus_R_Gc_R",
  "selected": "within_metric",
  "selection": "fixed_no_selection",
  "arms": ["interaction_ridge", "kernel_ncm", "within_metric"],
  "max_folds": 3,
  "physical_folds": "per_class_physical_id_sort_position_mod_min_K_3",
  "oneshot_proxy_anchors": "all_per_class_sorted_positions_0_to_parent_K_minus_1",
  "oneshot_proxy_ncm_equivalence_atol": 0.0,
  "k1": "exact_kernel_ncm_no_within_variance_estimate",
  "query_decision_policy": "per_sample_all_registered_classes",
  "tie_break": "physical_class_id_lexicographic",
  "optimizer_steps": 0,
  "source_inputs": false,
  "summary_inputs": false,
  "query_fit": false,
  "phase1_frozen": true
}
```

设计定稿后，root已分配核心、测试、冻结配置与入口的独立实现责任。算法冻结配置为`configs/d92_branch_metric_frozen_20260929.json`。实现及合成验证不等于真实support性能验证；本文件不授权发布、数据重验证或实验启动，继续保持query反馈隔离。

输出接口补充：标准OOF保留逐physical records；`oneshot_proxy.trials[trial]`保留精确training_ids、held_ids、parent_k、proxy_train_k=1与held_k。每臂输出confusion[C,C]、class_order、classwise_nll_sum、classwise_count、record_count及metrics，配对差保存both_correct/left_only_correct/right_only_correct/both_wrong四格计数和mean_correct_delta，不重复全部held逐行记录。每trial记录`ncm_equivalence={max_abs_score_difference,tolerance:0.0,equivalent,predicted_classes_equal}`；两臂同一K1算术路径应精确相等。真实K1的标准OOF与oneshot_proxy均为null。通常完整矩阵分解上界为63600次，即标准21600次与proxy42000次；恒定support的metric退化可减少次数，日志记录实际值。
