# D92-BranchLocalMargin：固定局部核上的多类平方间隔头

2026-09-29；support-only 设计及已授权实现规范。唯一候选为 `D92-BranchLocalMargin-v1`。主任务已接受实现；独立 core、冻结配置和合成测试已写入，数值验证由唯一执行负责人串行运行。现有 LocalRidge 代码、配置和产物保持不变。本设计未读取 LocalRidge 的 repeated query benchmark、query 分数、相关报告、索引或交接，也不从用户继续优化的请求反推任何 query 表现。

**保留 LocalRidge 已固定的训练折内径向核与 trace 匹配，只把多输出平方回归改成多类“最强错误类”平方间隔目标。** Phase1 冻结；每个 row 独立拟合；只用当前合法物理 support 的原始 received 五块缓存。margin=1、物理求和权重=1、RKHS 正则系数=1，无可搜索带宽、类别权重、温度或 K 专属参数。该方向是否改善旧类与新类仍须被完整 support 诊断证伪，不能由数学形式保证。

## 1. 证据与假设分开

唯一性能证据来自 [完整 LocalRidge support 汇总](E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-support-m4-r01/results/support_summary/summary.json)及同目录 CSV；解释和成本口径见 [support 结果报告](D92_BRANCH_LOCAL_RIDGE_SUPPORT_RESULT_20260929.md)。本设计不需要再读取旧 Metric/Orbit 的其他文件，也不扩展地面资料访问。

已观测事实如下。

- 4,800 个 parent 全覆盖，其中真实 K1 的 1,200 个仅作数值诊断；标准 OOF 和 proxy 各 3,600 个 parent，proxy 遍历 42,000 个 anchor。三臂总计 158,400 次 Cholesky 完成。
- 标准 OOF 的全部 12 个 K×新增类数格，相对 BranchRidge 和 InteractionRidge 的旧/新/H 都为正。LocalRidge 的局部核因此有合法物理 held 的支持证据，下一候选不更换这部分表示与带宽。
- proxy 在每个 parent K 下相对两基线的新类与 H 都提高，但旧类均值略降；最大 parent K 汇总下降为 0.159 个百分点。全部分层的新类均提高，但 K10、新增 20 类相对 BranchRidge 的 H 下降 0.011 个百分点。通过旧类容忍规则不等于严格共同改善。
- 候选 proxy 训练准确率为 100%，而 stage 等权的训练减 held 准确率差约 60.180 个百分点；其 held“正确类最大相似度减最近错误类最大相似度”的 stage 均值为 −0.027079。该均值包含全部 anchor，不是独立样本统计，也不是 parent 等权性能指标。它说明单个合法同类锚点常不是 held 最近的竞争支持，不能说明这个问题可被任何训练损失修复。
- LocalRidge 相对 InteractionRidge 的未校准 macro NLL 在标准 OOF/proxy 增加约 0.007980/0.016832，而准确率改善。概率损失与判决边界并不一致；本轮不靠 softmax 温度或 NLL 选择头。

由此提出的假设是：**固定局部核下，惩罚每个错误输出列的回归残差，可能把部分更新分配给不影响 argmax 的分数；把物理样本的损失集中到最强错误类，可能改善有限 support 的共同竞争边界。** 这是待检验解释，不是从上述证据证明的机制。已有近饱和训练准确率也意味着单纯加大训练 margin 未必有益，甚至可能加重支持记忆。

本候选不声称新的最大间隔理论。它是固定 kernel ridge 对照上的明确损失干预。相较只做 class norm 归一化，这一目标直接约束正确类与竞争类的分数差；但也更容易被异常锚点、错误标签和不可分物理样本牵引。用户最新明确要求“在 BranchLocalRidge 的性能上继续优化改善”，因此当前冻结的 LocalRidge 是本轮直接推进基线；这一基线变更来自用户要求，不来自 query 指标。

## 2. 输入、核与当前 row 边界

输入仍为每物理样本一行的 `z_id160 / FFT96 / t_emb160 / f_emb160 / pa_local160`，共 736 个缓存数值。沿用 [LocalRidge 冻结设计](D92_NEXT_AFTER_ORBIT_20260929.md)及 [当前实现](../code/cvsrffi/d92_branch_local_ridge.py)的全部归一化、零块和距离语义：

\[
B(x)=\operatorname{unit}([\operatorname{unit}(z),4\operatorname{unit}(FFT)]),\quad
A(x)=[\operatorname{unit}(t),\operatorname{unit}(f),\operatorname{unit}(pa)]/\sqrt3,
\]
\[
\phi(x)=[B(x),A(x),\operatorname{vec}(B(x)A(x)^\top)],\quad
K_0=K_B+K_A+K_BK_A.
\]

训练集共有 N=CK 个物理样本，各注册类的 K 相同。用既有 rank-2 reduced QR 的非负平方和计算完整 interaction 距离 d²；每个训练样本取最近异类 d²，零值保留，τ 为全部 N 个值的中位数。τ>0 时 R=exp(−d²/τ)，τ=0 时使用严格相同完整特征的等价类核。H=I−11ᵀ/N，G=γHRH，γ=tr(HK₀H)/tr(HRH)。数值距离、`expm1`、参考样本差分中心化及 trace 检查完全沿用原定义。

这不把 query 当局部带宽输入。训练后的 τ、γ、参考核及中心化均值不可变；单个待识别物理样本只形成与固定训练 support 的一行中心化核。完整 support row、每个 OOF 训练折及每个 proxy anchor 的所有状态独立估计，不继承别的 row、另一折、旧头或前次优化器状态。

old/new 角色仅供独立诊断汇总，不进入核、损失、约束、先验或停止规则。当前允许地面原型/小统计量并不要求使用；本候选不用它们，也不新增 source 样本、逐样本源 feature、地面摘要或不明来源 checkpoint。只复用已经核实的冻结 Phase1 特征来源，不读取源数据补证。

## 3. 唯一分类目标

记中心化 RKHS 特征为 ψᵢ，内积为 Gᵢⱼ；每一类有向量 w_c。分数为 fᵢc=⟨ψᵢ,w_c⟩，**没有额外 intercept**。对 C≥2，定义最强错误类平方 hinge：

\[
m_i(W)=\max\{0,\;1+\max_{c\ne y_i}f_{ic}-f_{i y_i}\},
\]
\[
\boxed{\min_W P(W)=\frac12\sum_{c=1}^C\|w_c\|_{\mathcal H}^2+
\frac12\sum_{i=1}^N m_i(W)^2.}
\]

每个物理样本只贡献一个最强错误类损失，不按 C−1 个错误类再次求和，也不按 N、K 或 C 平均。间隔常数为 1，与 ridge 目标中真/假类期望分数差 1 对齐；正则系数为 1。损失形状已经改变，不能声称它与 ridge 的有效正则强度完全相同，更不能把不同目标的 loss 数值直接比较为优劣。

等价约束形式为 ξᵢ≥0、fᵢy−fᵢc≥1−ξᵢ（全部 c≠yᵢ），最小化 ½∑c‖w_c‖²+½∑iξᵢ²。平方 slack 使冲突标签也有有限可行解；不要求训练集硬间隔可分。

该目标对 W 严格凸，最优 RKHS 判决函数唯一。共同平移所有类权重不改变间隔，却增加正则项，因此最优解满足 ∑c w_c=0；这不相当于额外学习一个偏置。相同类间 margin 下无需对所有错误输出维持指定绝对回归值。

## 4. K1 能学到什么

K1 时每类只有一个真实物理锚点。核尺度仍来自不同注册类之间的合法距离；margin 约束只表达“该已知锚点的标签应战胜其他注册类”。它没有估计类内方差、构造同类正对或新增 shot，也没有借用 held/view 的类别信息。

与 ridge 不同，最优对偶质量可分配给当前最紧的错误类约束。C≥3 时不同竞争方向可获得不同系数，因而可能改变未见点的竞争边界。这个可识别机制来自观测到的跨类锚点关系；它不能恢复尚未观测的同类变化。两个锚点的 C2K1 情形只有一个中心化方向，两个对称头的边界应与 LocalRidge 一致；该特例必须作为实现检查，不包装为所有 K1 都能改变预测。

K1 的一个错标或偏离类中心的锚点会同时影响带宽和最强对手约束，平方 hinge 还会较重惩罚大违例。这可能牺牲某些容易类别去满足不具代表性的困难锚点，且 old/new 组上的影响可能不同。不能因为没有 old/new 专用系数就宣称没有组间权衡。

## 5. 可核验的二次对偶

为每个 i 和 c≠yᵢ定义 βᵢc≥0，令

\[
s_i=\sum_{c\ne y_i}\beta_{ic},\qquad
\alpha_{iy_i}=s_i,\quad\alpha_{ic}=-\beta_{ic}\;(c\ne y_i),
\]
\[
W=\Psi^\top\alpha,\qquad F=G\alpha.
\]

对偶是非负约束凸二次问题的最大化形式：

\[
\boxed{\max_{\beta\ge0}D(\beta)=\sum_i s_i-
\frac12\sum_i s_i^2-\frac12\operatorname{tr}(\alpha^\top G\alpha).}
\]

采用最小化 −D 时，对偶梯度为

\[
g_{ic}=s_i+F_{iy_i}-F_{ic}-1.
\]

KKT 条件是 βᵢc≥0、gᵢc≥0、βᵢc gᵢc=0。任意非负 β 都是对偶可行点；由它形成 W 并计算实际 mᵢ(W)，就得到合法 primal/dual 目标及 gap。不存在“把训练 loss 不下降误当成已求解”的判据，也不从 optimizer 的单一 success 标记推断收敛。

不形成 N(C−1)×N(C−1) Hessian。若仅用于理论界，行映射 βᵢ↦αᵢ的算子平方范数为 C，s 的行映射平方范数为 C−1，所以对偶 Hessian 的谱上界为 C·s0+C−1，其中 s0=tr(G)。该上界不是新的 ridge、带宽或步长搜索参数。

## 6. 确定性行块求解与停止证书

使用按物理 ID 字典序的循环行块坐标法，β=0、α=0、F=0 初始化。每个完整 sweep 依次更新全部 N 个物理样本；每次只对该样本的 C−1 个 β 联合求最优值。注册类亦按物理 class ID 排序，最后恢复调用者请求的输出列顺序。相同输入得到固定运算顺序；不随机换序、不多起点、不选最佳迭代。

### 6.1 行块闭式解

当前行 i 的 d=Gᵢᵢ，去掉本行贡献后

\[
F_i^{(-i)}=F_i-d\alpha_i,\qquad
a_c=1-F_{iy_i}^{(-i)}+F_{ic}^{(-i)}.
\]

去掉与该行无关的常数，子问题为

\[
\min_{\beta_i\ge0}\frac d2\|\beta_i\|_2^2+
\frac{d+1}{2}(\mathbf1^\top\beta_i)^2-a^\top\beta_i.
\]

非退化径向核下 d>0。若 max a≤0，则 βᵢ=0。否则将 a 降序排列（相等时按物理 class ID），寻找唯一有效 active size m，令

\[
s_m=\frac{\sum_{j=1}^m a_{(j)}}{d+m(d+1)},\qquad
\theta_m=(d+1)s_m,
\]

满足 a_(m)>θ_m，且 m=C−1 或 a_(m+1)≤θ_m。解为 active 内 β_(j)=(a_(j)−θ_m)/d，其余为 0。边界 β=0 归为 inactive。求累计和使用确定顺序的 float64 补偿求和或 `math.fsum`，不使用 float32。

实现时避免在 d 很小时直接相减两个约为 1 的数。令 active 均值 ā=(∑active a)/m，采用等价形式

\[
\beta_{(j)}=\frac{a_{(j)}-\bar a}{d}+\frac{s_m}{m}.
\]

active 判定使用 (a_(j)−ā)/d+s_m/m 与 0 的比较，减少公共大常数的相减，并避免极小 d 使 d·s_m/m 下溢；比较中的除法溢出仅用于确定符号，最终系数仍必须有限。非负投影 `max(0, beta)` 只实现本来的非负约束，不裁剪核谱、不修改 G；若子问题解或 active 判定在有限精度下不能满足冻结数值检查，记录技术失败，不能改成另一种损失或悄悄退回 ridge。

由新 β 用确定顺序重新求 sᵢ并形成 αᵢ。令 Δαᵢ为新旧之差，更新

\[
F\leftarrow F+G_{:,i}\,\Delta\alpha_i.
\]

该操作是 N×C 的外积更新。每个完整 sweep 后用 Gα 重新计算 F，再计算证书，避免把长串增量更新的浮点漂移当成真实优化进展。

### 6.2 冻结的数值停止规则

以下数值契约写入独立 frozen config。统一 float64；ε=eps64；η=√ε≈1.490116×10⁻⁸。每个完整 sweep 计算

\[
q=\operatorname{tr}(\alpha^\top F),\quad
P=\tfrac12q+\tfrac12\sum_i m_i^2,\quad
D=\sum_i s_i-\tfrac12\sum_i s_i^2-\tfrac12q.
\]

记录原始 gap=P−D，尺度 M=max(1,|P|,|D|)，归一化 gap=max(0,P−D)/M。负 gap 只允许浮点舍入范围 `(P−D)≥−128·eps64·max(N,C)·M`；保留原始负值和界，不能把明显负 gap 截成 0 宣称收敛。

KKT 自然残差用不含相消的等价公式 rᵢc=min(βᵢc,gᵢc)，即 β−max(0,β−g)。报告 raw max|r| 和 `max|r|/max(1,max_i s_i)`；不用可能很大的 inactive 正梯度作分母。β 非负、目标及分数有限、归一化 gap≤η、归一化 KKT 残差≤η 同时满足才结束。对偶证书适用于训练优化，不是 held/query 性能保证；决策边界附近的标签仍可能对有限精度敏感。

上限固定为 **1,000 个完整 sweep**，每个 sweep 至多 N 次行块更新；不设“准确率达到某值”或 wall-clock 早停，不放宽容差。达到上限、出现非有限值、非退化状态 d≤0、无法构造合法 active set、目标/证书超出数值一致性界时，记 `TECHNICAL_FAILURE`，保留失败 arm、scope、parent/train K、fold/trial、训练/held 物理 ID、已完成 stages、当前 β/α、最后 F 与逐 sweep 轨迹。不自动回退、改参或重跑。失败不授权影响其他健康任务；后续修复必须是明示的技术修复，不能把低 held 准确率称作技术故障。

每个 sweep 的精确块更新应使 −D 不增加；记录相邻完整 sweep 的变化，并按同一浮点舍入界检查。若出现超界逆向变化，先保留证据并报数值失败，不靠“再跑几轮看起来下降”覆盖问题。

## 7. 退化、对称性与独立推理

1. **单注册类 C1。**没有错误类，定义 margin loss=0、W=0、P=D=0，不运行优化器、不虚构 bandwidth，输出 0 并预测唯一类。
2. **全同完整特征、G=0。**所有中心化特征为 0，最优 W=0，所有类分数精确为 0；C≥2 时 P=N/2。解析对偶取每行 βᵢc=1/(C−1)，sᵢ=1，D=N/2，是零 gap 证书，无需迭代。可存等价零判决状态；证书必须说明解析 β 与零 RKHS 函数的等价关系，不能从零 α 错算出 D=0。
3. **τ=0、s0>0。**仍用精确等价类核，平方 slack 允许同一特征的冲突标签。优化目标不改。所有未与训练特征精确匹配的 held/query 有相同 raw 核行，从而得到相同固定分数行；报告此退化，不加 ε 带宽。
4. **非退化 G 的零/负对角。**径向 RKHS 所有原始特征等范数，中心化后的某个对角精确为 0 只可能对应所有该训练特征相同；全同路径已单列。非退化 G 出现 d≤0 是数值不一致，不能除以人为对角下限修复。
5. **类名与物理排列。**算法不使用 old/new 或类名语义；canonical 排序用于确定运算顺序和并列分数的 class ID 字典序。类列重标和物理排列的等变性在最终收敛证书误差范围内检查；平分输出仍按原物理 class ID 规则处理，不读取 held 标签破平局。

最终 query 分数仅为 `g(q)^T alpha`，g(q) 是原 LocalRidge 定义的固定中心化、trace 缩放核行。没有学习偏置、温度、类 norm 校正、query norm 校准或跨 query 后处理。批量大小、顺序与其他 query 的存在不得改变该行分数。

## 8. 完整诊断、同一筛选与可反驳预测

矩阵继续完整覆盖 8 个冻结模型/cohort、K=1/5/10/20、新增类数=0/2/5/10/20、既定 receiver、scenario、support seed。不选择某个 K 或某个新类规模测试。K≥2 的标准 OOF 与全部 anchor proxy 完全复用既有物理映射；每次候选拟合从零 β 开始。真实 K1 probe 仍只做数值诊断，没有独立 held，不因为它可以优化 margin 就多造一份“泛化证据”。

固定四臂为 `branch_ridge / interaction_ridge / local_ridge / local_margin`。直接基线 LocalRidge 必须在与候选完全相同的每个物理 OOF 训练折及 proxy anchor 上调用当前冻结原方法重新拟合，不能用完整 support 的头冒充折内控制，也不以旧汇总均值代替配对预测。BranchRidge、InteractionRidge 保留完整描述性对照，可以重算原方法，或严格复用已完成且与该 episode 特征、类表、全部训练/held 物理 ID 和固定配置相同的原始 artifact；复用情况、原始成本与新增成本分开披露。候选不能读取控制臂预测或 held 指标来训练。

**保留 A/B support 筛选结构，直接基线按用户要求改为当前冻结 LocalRidge**：标准 OOF、1-shot proxy 分别在每个 parent K=5/10/20 下，相对同折拟合的 LocalRidge 满足新类与 H 严格提高、旧类退化不超过 1 个百分点，才支持继续推进。只超过旧 BranchRidge/InteractionRidge 而未超过 LocalRidge，不满足本轮推进条件。old-only、全部 K×新增类数组格、模型/cohort、receiver×scene 和 support seed 一律报告，不增加全格正向门槛；两个旧对照不再各自附加推进门槛。每 K 旧/新/H 是否均严格为正单独报告，不能把允许轻微旧类下降的规则通过写成严格共同改善。

预先可反驳预测如下。

- H1：最强错误类平方间隔头能在同一固定核、相同物理持出条件下改善旧新共同竞争。若仅训练 margin 增大而 held 旧/新/H 下降，则不支持这一假设。
- H2：K1 proxy 的新类收益不以更大的旧类损失换取。若旧类仍降，只能如实报告容忍规则是否通过，不能声称实现了旧新严格共同提升。
- H3：若收益主要出现在较多新类任务，可能与错误类竞争有关；但仅凭分层相关性不能证明机制。对各类数都保留完整差值，不据结果改 loss、margin、λ 或拼接原头。

科学性能不进入每次拟合的停止或故障判断。代理诊断通过也不等于未知独立数据的泛化保证。本次用户已明确延后独立数据验证，本提案只推进已授权的 support 方法研发；不因独立数据暂缺增加新门槛，也不把后续重复基准重新标为独立验证。

## 9. 成本、状态与实际日志

核几何成本与 LocalRidge 相同：分块 rank-2 距离 O(N²·736)，固定点对块上限 256；保持 float64 的 G，空间 O(N²)。margin 优化不做额外 Cholesky，不形成完整 interaction 张量，也不形成全对偶 Hessian。一次 sweep 的外积更新及 F 重算为 O(N²C)，active 排序为 O(NC log C)；当前实现逐 active size 重新 `math.fsum`，该扫描额外为 O(NC²)，在 C≤26 下换取简单、确定的求和行为。额外工作数组为 β（N×(C−1)）、α/F（各 N×C）与行级临时向量。

标准 OOF 有 10,800 次候选拟合，proxy 有 42,000 次，合计 52,800 次新优化；同折 LocalRidge 直接基线另有 52,800 次 Cholesky，两个旧控制若也重算则再增加 105,600 次。候选的 1,000 sweep 是每次的数值预算上限，不是预计迭代数。按已观测最大 OOF 训练形状 N=364、C=26，一次 sweep 的主量级约 N²C=3,444,896 个乘加/外积元素；若计算与更新分别计入，其常数倍应另计。不能把这个量级当作实测 FLOP、秒数或全矩阵时长承诺。当前没有运行新 solver，因此不能断言它比 ridge 快或已能在某个 wall-time 内完成。

实现后的合成检查应覆盖 N=26/C=26、N=364/C=26 和完整支持形状 N=520/C=26，记录实际 sweep 分布、行块次数、证书、峰值分配及耗时，再据冻结预算披露完整运行的资源风险；不以合成或真实 held 准确率调整数值容差。实现若出现具体数值/性能失败，应先解释失败层及修订范围，不静默改变本提案来凑结果。

部署常驻状态仍只需 B、A、α、参考核行、中心化均值及 τ/γ/两个中心化标量；非退化时为 `8N(738+C)+32` bytes，β、F、slack 和优化轨迹不属于部署头。N=520、C=26 时为 3,178,272 bytes；这不含类表、Python 审计对象、临时核或 RSS。单个 query 的计算仍为 O(N·736+NC)。既有五块 float32 cache 每物理记录 2,944 bytes，无新 view、冻结网络前向、source 载荷或模型权重更新。

日志必须区分“未发生”与“未测量”：启动时打印全部生效公式、样本数、类数、τ/γ、trace、dtype、求解器、容差和最大 sweep。每个 sweep 的结构化记录包括 margin data loss、RKHS penalty、P/D、原始/相对 gap、KKT 原始/归一化残差、active β 数、active 物理样本数、slack 分布、raw 对偶梯度范数、已完成行块数、耗时与算法状态；提供同粒度的紧凑 JSONL/CSV 和 CVS 风格文本摘要。这里 `optimizer_steps` 是实际行块更新次数，不能沿用 ridge 的 0；学习率为 N/A（精确块最小化，无梯度学习率），epoch 用 sweep 明示。有效自由度不套用 ridge 的线性 hat-matrix 公式，写 null 并说明 margin 头是分段非线性估计。

训练准确率和间隔只作记录，不触发停止。最终状态固定后才由 support 诊断器连接 held 标签，记录旧/新/H、macro accuracy、class floor、原始 confusion/NLL、最强错误类及 margin 分布、tau=0 无精确匹配率。NLL 仍是未校准固定分数的描述性指标，不据此调整温度或选择候选。source validation 为 N/A，原因是没有 source 样本权限。

## 10. 实现前的有限验证范围

后续若主任务接受实施，先用合成数据检查对偶推导、行块 active 解与小型独立凸优化解的一致性，逐步核实 P≥D 和 KKT；覆盖 C1、全零核、C2K1、C≥3K1、冲突标签/等价类核、极近重复、类别/物理排列、query 单行与批次不变性、无隐藏 bias、失败上下文保存及预算上限。核几何与 LocalRidge 必须精确复用并回归，不能顺带变更 τ、trace、距离或正则。

这些是实现正确性和既有输入权限要求，不是新增科学审批或逐 episode 性能门槛。独立实现为 [核心模块](../code/cvsrffi/d92_branch_local_margin.py)、[冻结配置](../configs/d92_branch_local_margin_frozen_20260929.json)及[合成测试](../tests/test_d92_branch_local_margin.py)。失败审计保留 β/α/F；非有限值明确编码为 JSON 字符串 `NaN`、`Infinity`、`-Infinity`，避免严格 JSON 写入时丢失失败证据。完整 support 运行与任何后续评价均由主任务按既有权限执行，本设计不预告数值或性能验证结果。
