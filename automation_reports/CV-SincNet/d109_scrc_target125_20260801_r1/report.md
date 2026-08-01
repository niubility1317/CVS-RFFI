# D109-SCRC/r1方法研发与完整125预登记

状态：`LOCAL_VERIFIED / IMPLEMENTATION_COMPLETE_NOT_RELEASED`

## 目标与证据边界

|字段|值|
|---|---|
|候选|`D109-SCRC/r1`|
|新贡献|support-confusion reciprocal calibration，仅为classification head|
|DA对照|复用冻结D108 CB-RRC；不宣称D109提出新DA|
|强基线|D92 288维表示＋equal-prior LDA|
|完整矩阵|与D62/D92/SVRN及D108相同的5receiver×5seed×5slice×3scene×4arm×2phase|
|协议|`p2_min_v1`；support-only、逐query全注册类竞争、无clean/query truth/role/quota/fit/update/global reassignment|
|发布条件|仅当D108完整125表现弱或SCRC被明确要求独立验证时发布；本地研发不得延迟D108|

D107完整125已经证伪signed/centering/simplex-KRR；D108-SMME是support margin产生的固定零和logit bias。SCRC改用support混淆矩阵对每条query的完整后验进行query相关、但状态冻结的全类互惠校正，因此机制不同。历史D80 ground covariance、D83 precision loading和D93/D94 transport无提升或负向，D109不再派生新ground DA；CB-RRC只作为既有正交DA对照。

## 冻结公式与四臂

对当前phase全部注册类的合法support logits`g_i∈R^C`，令`p_i=softmax(g_i)`，按每类相同K-shot构造`Q_ab=K^{-1}Σ_{i:y_i=a}p_{i,b}`。定义等先验Bayes反向响应`R_ba=Q_ab/(Σ_l Q_lb)`，以及无可调参数强度`ρ=1-tr(Q)/C`；冻结`T=(1-ρ)I+ρR`。对单条query，先得`p(q)=softmax(g(q))`，再算`p̃(q)=p(q)T`，返回`h_c(q)=g_c(q)+log p̃_c(q)-log p_c(q)`。K1仍活动；只有数学上`Q=I`时自然退化恒等。

|arm|DA|head|归因|
|---|---|---|---|
|M0|精确D92|原D92 logits|强基线|
|M_DA|冻结CB-RRC→D92|原D92 logits|既有DA效应|
|M_HEAD|精确D92|SCRC|D109新head效应|
|M_JOINT|冻结CB-RRC→D92|SCRC|交互效应|

before与after分别只用各自合法support构建对应D92/SCRC状态；base与DA各自构建SCRC，禁止跨臂复用。old/new类公式完全相同，类名置换使`Q,R,T,h`同步置换。最大C=26时`Q＋T＋ρ`约5412B，fit为`O(C²K)`，单query为`O(C²)`；无扫描、温度、阈值、router或fallback。

## 可行性复核（16行）

1. SCRC只读取当前phase的support logits、support标签和registry。
2. query仅逐条读取冻结`T`，不进入fit或状态更新。
3. 所有注册类使用同一公式，不读取old/new role。
4. `Q`每类等K-shot平均，保持类置换等价。
5. `ρ`由support自响应唯一确定，不是超参数。
6. K1构造完整`Q,T`，无代码fallback。
7. M0必须与D92逐值一致。
8. M_DA必须与D108 CB-RRC-only逐值一致。
9. M_HEAD与M_JOINT分别隔离head主效应和交互。
10. 不使用D106 RDCE或任何新Phase1 bundle。
11. 不改received-IQ、physical ID、split、K或scenario，不重验数据。
12. 风险一是support过拟合使`Q≈I`而无收益。
13. 风险二是K1混淆噪声导致错误互惠转移和old floor下降。
14. 完整125后只按same-row四臂比较，不选择slice/receiver。
15. 若M_HEAD−M0与M_JOINT−M_DA均未使after floor、seen-new、H形成联合收益，直接淘汰。
16. 判定：`FEASIBILITY_REVIEW_PASS / DESIGN_FROZEN`，允许并行实现核心与D92 pair，不增加发布gate。

## 实现拆分

SCRC核心负责typed冻结状态、stable softmax、`Q/R/T/ρ`、query应用、K1/置换/负例和资源审计。D92 pair负责复用D108 base/CB-RRC正式int8状态，分别从support logits冻结before/after的SCRC状态，并固定四臂评分；不得改变D108 runner、N607运行或D108源码。主agent统一审查、提交和决定是否在D108结果后发布D109。

## 实现与独立复核

SCRC输出采用与冻结`h=g+log(pT)-logp`分类等价的canonical`log(pT)-max(log(pT))`；只在固定float32概率下限`2^-149`处截断，避免极端logit下加回巨大行常数抹掉稠密`pT`差异。数学identity T保持query logits逐字节直通。独立审查先后发现并复现两个P1：概率域floor破坏极端非identity方向、巨大行常数破坏稠密差异；最终加入循环置换零T与均匀Q稠密极限测试后，期望`[0.5555555820,0.2222222090,0.2222222090]`与实际最大误差为`4.82e-9`，复审`P0=0,P1=0 / GO`。

|实现面|文件|验证|
|---|---|---|
|SCRC typed核心|`code/cvsrffi/stage2_d109_scrc.py`；SHA256=`71fa5f6a31333ee53fa928cf2b15cc791ffecd12a6a9d825104a49768b0202ef`|14项；Q/R/T方向、K1、置换、identity、极端稠密/零T、wire/resource|
|D92四臂pair|`code/cvsrffi/stage2_d109_d92_core.py`；SHA256=`b829d59f9bfa9a30e0063c247e4b402eb8c3c79317f487fb624e2262d1229905`|4项；M0/M_DA逐值对齐D108、4个SCRC状态、无持久SMME、异常恢复|
|联合|两组测试及D108依赖|18 passed；仅既有PyTorch只读buffer警告|
|独立复核|SCRC与pair分别审查|SCRC`P0=0,P1=0`；pair`P0=0,P1=0`|

当前只完成方法核心，不含D109 runner、truth发布或N607性能证据；不得把实现完成写成性能成功。若D108完整125已达到目标，D109是否发布由主agent重新排序；若D108性能弱，直接复用其成熟Target125执行/证据面接入D109，不再增加数据或方法gate。
