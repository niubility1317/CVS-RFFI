# D92 FCR8 后续联合方法：固定 B 函数先验的残差 LocalRidge

日期：2026-10-01。状态：`RUNNING_SINGLE_STRUCTURE`。候选：`D92-AnchorJointLocalRidge-v1`，简称 AJLR。本文只固定一条结构及其可证伪假设，不启动实验，不报告尚未测量的收益。

**保留 BranchLocalRidge 主线，改变 C 阶段分类函数的继承方式：以本行实际 B 分类函数为固定先验，闭式拟合全注册类标签残差，再通过这个最终 LocalRidge 头联合微调。** B/C 均从合规 source-only Phase1 开始；不继承历史目标适配状态。C 只继承本行新训练的实际 B，不新增 reset 候选或参数组合搜索。

## 1. 当前问题、证据和假设

唯一性能证据是[完整本轮合法 support 结果](D92_FCR8_SUPPORT_RESULT_20261001.md)，唯一训练解释是[完整训练机制发现](D92_FCR8_TRAINING_FINDINGS_20261001.md)及其引用的完整派生流。它们来自 `20260930-phase2-d92-fcr8-support-m2-r01`，训练代码 `33673fe02a857d790aa69a1ace17fdb9638099d0`。没有读取 query、历史目标 ABC/选模摘要、总实验索引或根交接。

| 已核实证据 | 能支持的判断 | 尚不能支持的判断 |
|---|---|---|
| 936 个有信息阶段全部更新，648 次实际 SVD 全部保留 r=8，首步梯度均非零 | 优化、字典坐标和前向改变实际执行 | 字典已足够表达目标域，或可以排除表征泛化不足 |
| 内层 B 准确率均值提高 6.2292 个百分点，C_seq 提高 2.5205 个百分点；连续 task/total/margin 均改善 | 当前监督目标能改变训练判决 | 该变化一定改善独立持出样本 |
| 96 个可测 new-present 配置：FCR8_seq 的 B 相对 R0 下降 0.1736 个百分点，C 旧/新/H 提高 0.3212/0.0703/0.1115 个百分点，平均绝对新旧差增加 0.5530 个百分点 | 当前完整结构没有形成明确共同改善；B 的训练改善与独立持出结果存在缺口 | 唯一根因是 hinge、teacher、字典、继承或 head 重拟合 |
| C_seq 旧类列内诊断为 71.1719%，全部注册类竞争为 64.0885%，差 7.0833 个百分点 | 新类竞争是必须解释的实际误差部分 | 可以用旧/新 oracle 路由消除这部分错误 |
| guard 激活和 keep 拒绝均为 0；806/936 个信息阶段在三试探预算下停止，末 task 梯度仍非零 | 本轮不是 keep 硬约束挡住更新；有限试探不能当成收敛 | 删除 keep、增加试探或更小步必定提高外层准确率 |
| 当前 C 会重新拟合全部类的头，带宽、中心和 trace 统计随当前训练集合/表征改变 | 参数继承不等于分类函数继承；有必要分开观察注册 head 与表示改变 | 已有数据证明上述统计变化就是性能瓶颈 |

完整训练发现另测得同 stage/fold 初末 τ 相对变化均值 B/seq/reset 为 +3.5768%/+0.9431%/+1.4819%，γ 为 +0.9245%/+0.0569%/+0.1091%；原始 s0 变化折数为 0。中心参考向量没有独立机制测量或消融。τ/γ 确实变化，但没有“变化导致持出下降”的因果证据；本文固定它们是待验证的结构选择。

AJLR 检验一个联合假设：**把 B 分类函数显式放入 C 的先验项，使新注册学习成为对该函数的受正则残差修正；同时固定旧 support 参考下的核尺度和中心测度，用全类平滑监督信号联合更新表示，可能比从零重新拟合 C 分类函数更好地协调新旧竞争。** 这是结构假设，当前证据没有证明它成立。多项结构同时改变后，只能评价整个候选；没有额外消融就不能把收益归于其中某一项。

R0 是 BranchLocalRidge，绝不是未适应 A。本轮 A=N/A，因此当前资料不能检验 B−A≥10 个百分点。注册旧类下降≤1 个百分点、绝对新旧差≤3 个百分点和旧类适应提升≥10 个百分点继续是软目标，不成为每轮必须全部达到的硬门槛。

## 2. 输入、表示和唯一 adapter

Phase1 固定。输入仅来自 practical residual/post_sync/noeq、fs=25MHz 的已固定 received IQ。每个物理 support 对应原有五个特征块，维度为 160/96/160/160/160，总维度 d=736。按原 BranchLocalRidge 定义：

\[
b=\operatorname{unit}([\operatorname{unit}(z_{id}),4\operatorname{unit}(fft96)]),\qquad
a=[\operatorname{unit}(t),\operatorname{unit}(f),\operatorname{unit}(pa)]/\sqrt3,
\]

\[
\Phi_0(x)=[b,a,\operatorname{vec}(ba^T)],\qquad \dim\Phi_0=123616.
\]

继续使用 FCR8 的固定 DCT 字典与切向范数保持 adapter；不扫描 rank、字典、κ 或原始/适配距离混合系数。将原始五块写为 \(\rho_l w_l\)，非零块 \(\|w_l\|=1\)，令

\[
x_d=[w_1,\ldots,w_5]/\sqrt5,\quad h(x)=\operatorname{GELU}(V_0x_d),\quad v=Uh(x),
\]

\[
t_l=(I-w_lw_l^T)v_l,\quad \delta_l={\kappa t_l\over\sqrt{\kappa^2+\|t_l\|^2}},\quad
T_U(x)_l=\rho_l{w_l+\delta_l\over\|w_l+\delta_l\|},\quad \kappa=1/4.
\]

零原始块保持零。\(V_0\in\mathbb R^{8\times736}\) 为固定前 8 个正交 DCT 行，\(U\in\mathbb R^{736\times8}\)。不反传 encoder，不训练 V。\(\Phi_U\) 按适配后的 b/a 构造相同 interaction 映射。

每个 B/C preparation 只在本阶段当前合法 outer-train 的 H 上做一次原有薄 SVD：\(H/\sqrt N=P\Sigma R^T\)，按原数值 rank 规则保留 r≤8，\(W=R_r\Sigma_r^{-1}\)。训练坐标

\[
U=U_a+ZW^T,\qquad U_a=0\text{（B）},\qquad U_a=U_B\text{（C）}.
\]

Z=0 必须精确复制实际 anchor_U，不投影或重建 U_B。C 的 W 可由当前新旧 support 重新估计，但 U_B 未被新坐标覆盖的分量仍保留。该 SVD 只定义当前训练的函数近端坐标，不使用 outer-held、query 或地面逐样本特征。它不是任何 inner head 的统计来源。

\[
\frac1N\sum_{i\in S}\|(U-U_a)h_i\|^2=\|Z\|_F^2.
\]

这只约束当前 support 上投影、饱和、归一化之前的功能增量；不等于核、score 或旧 query 准确率保持。字典宽度 8、κ=1/4 和该坐标是提前固定的表达/预算选择，未被本轮结果证明最优；实际保留秩 r 按本阶段的数值规则测量，不固定为 8。

## 3. 按折固定的核尺度和中心参考

### 3.1 nuisance 参数的来源

每个 inner fold 的旧 inner-train 集合记为 \(O_r\)。必须在 **该集合的原始特征** 上按原 R0 的规则计算 \(\tau_{0,r}\)、\(s_{0,r}\)、\(\gamma_{0,r}\)。不得把完整 outer-old 的参数广播给 inner 头。最终完整 B/C 头使用当前 final old support 集合 O 的原始 R0 参数。

B 阶段从其 R0 初始化开始就固定这两个数值 \(\tau_{B,r}=\tau_{0,r}\)、\(\gamma_{B,r}=\gamma_{0,r}\)，各 trial 不重新估计带宽或 trace scale。实际 B 最终保存的 kernel 参数就是这两个固定值。C 继承 **实际 B 保存值**，不是先训练普通 FCR8 B 再偷偷用另一个 R0 尺度替换。改变 U 后核的实际 trace 一般变化，不能继续声称 trace 恰好匹配 s0；需报告实际 trace。

固定参数正值时，raw kernel 为

\[
D_U(x,x')=\tfrac12\|\Phi_0(x)-\Phi_0(x')\|^2+
\tfrac12\|\Phi_U(x)-\Phi_U(x')\|^2,\qquad
R_U(x,x')=\exp[-D_U(x,x')/\tau_B].
\]

它是对拼接欧氏表示的 Gaussian kernel，所以 PSD。距离仍使用现有 rank-2 QR 的非负平方和计算，不物化 123616 维特征，不加距离 floor、带宽 floor 或 Cholesky jitter。

### 3.2 只固定测度，参考点随 U 重新映射

中心参考是本折固定旧物理 support 集合 \(O_r\)，等权测度 \(\mu_r=M^{-1}\sum_{o\in O_r}\delta_o\)。**没有冻结 RKHS 均值向量。** 每次 U 改变，参考点也通过当前 U 重新映射，所以

\[
k_U(x,x')=\gamma_B\big[R_U(x,x')-\mathbb E_{o\sim\mu_r}R_U(x,o)
-\mathbb E_{o\sim\mu_r}R_U(o,x')+\mathbb E_{o,o'\sim\mu_r}R_U(o,o')\big].
\]

这是 \(\sqrt{\gamma_B}(\varphi_U(x)-\mathbb E_{\mu_r}\varphi_U)\) 的内积，仍 PSD。需反传 train、held 和参考点的全部 current-U 贡献。只 detach 参考 embedding 或只更新第一项距离，都会优化不同的目标。

C 的 inner-train S 含该折 O_r，令 q 为 S 上的固定向量：旧参考记录位置为 1/M，其余为 0；\(q^T\mathbf1=1\)，\(P_q=I-\mathbf1q^T\)。若 R 是 raw train Gram、Q 是 raw held-to-train Gram，则

\[
K=\gamma_B P_qRP_q^T,\qquad
L=\gamma_B(Q-\mathbf1_hq^TR)P_q^T.
\]

B 时 O_r=S、q=1/N，退化为 R0 的普通训练中心化。参考差分、`expm1(-D/tau)` 和 rank-one 广播可以按同一代数稳定计算，不需物化 P_q 或执行额外 N³ 的矩阵乘法。单个 query 用同一固定 q、当前已冻结 U 和已保存训练中心量；其他 query 不进入计算。

U 保持 U_B 时，加入新 support 不改变任何旧/旧 raw kernel、旧测度或旧/旧 centered kernel。U_C 改变后旧几何会改变；本候选没有声称将 B 几何冻结。固定尺度/测度仅去掉由注册集合变化引起的重估，不约束其适配后的值保持原样。

## 4. 无自由 intercept 的非零函数先验闭式头

### 4.1 精确目标与闭式解

当前注册类数 C。标签目标明确为

\[
Y_{ij}=\mathbf1[y_i=j]-1/C.
\]

这是 **每行在类别维度的中心化**，不是对每列再减当前 train 样本均值。没有自由 intercept，没有额外 `Ymean`，也不减 \(Y-M\) 的经验行均值。即使固定旧测度与完整训练集合均值不同，上述约定仍完全不变。

B 的先验 \(m=0\)。C 的先验 \(m(x)=\operatorname{pad}_{C}(f_B(x))\)：本行真实 B 分类函数放在对应旧类列，新类列为零。不是冻结地面原型拟合的头，也不是人工旧/新偏置；没有可扫描的 prior 系数、温度或 group offset。

在当前 k_U 的乘积 RKHS 中求残差函数 g：

\[
g_U^*=\arg\min_{g\in\mathcal H_{k_U}^C}
\frac12\sum_{i\in S}\|m(x_i)+g(x_i)-Y_i\|^2+
\frac12\sum_{j=1}^C\|g_j\|_{\mathcal H_{k_U}}^2.
\]

令 \(M=m(S)\)、\(A=K+I_N\)、\(E=Y-M\)，则

\[
\alpha=A^{-1}E,\qquad f_U(x)=m(x)+k_U(x,S)\alpha.
\]

实现用一次 float64 Cholesky 和两次三角求解，不形成逆矩阵。PSD 给出 A 的最小特征值≥1，即使旧/新记录重复或 K 秩亏也有唯一预测函数。数值系数用该规范解，不能因为 K 有零空间另选一个更好看的 α。

E 不再次中心化。每个 M/Y/E 行的类别和均为零，所以 α 每行的类别和及 f_U(x) 的类别和也为零（浮点容差内）。固定旧测度下残差函数在旧参考记录上的均值为零；这是中心化约定，不是每个旧样本 score 不变的保证。

**无 intercept 是有代价的模型约束。** 当旧参考集合含 M 条平衡记录、旧类数 C_B、新类数 C_N、C=C_B+C_N 时，f_B 与 g 各列的旧参考均值均为零；目标 Y 的对应均值却为旧列 1/C_B−1/C、新列 −1/C。中心核无法拟合这部分组均值偏移。由 Jensen 不等式，任何 U 的旧参考半平方数据误差都满足

\[
\frac12\sum_{i\in O_r}\|f_U(x_i)-Y_i\|^2
\ge\frac M2\left[C_B(1/C_B-1/C)^2+C_N/C^2\right]
=\frac{MC_N}{2C_BC}.
\]

这不妨碍正规解存在或个别记录被正确分类，但可能损害新旧校准和新类学习。本文选择保留零旧参考均值以显式限制注册造成的整体平移，同时接受该误差下界；不声称标签可完全拟合。后续需报告均值残差及全类混淆，失败后不能在当前候选里补扫旧/新偏置。非平衡 reference 时，用其真实 class frequency 向量 π_Q 替代 1/C_B，下界为 M·||π_Q−1/C||²/2；同样没有隐藏 intercept。

最终预测只有

\[
\widehat y(x)=\arg\max_{j\in\mathcal C_{registered}} f_{U^*}(x)_j,
\]

精确并列按物理 class ID 字典序。原型路由、旧/新真值路由、query 类别配额、global 分配及 logit 偏置扫描均不进入算法。

### 4.2 B/C 的真实继承与 fold 绑定

B 的 adapter U_B 由当前合法 outer-train 监督训练得到。C inner fold 的先验 B 头在 **冻结 U_B** 后，仅由该 fold 的旧 inner-train O_r 拟合；nuisance 与中心也仅来自 O_r。不使用完整 B 头给 C inner-held 做先验，不把 outer-held 记录加入 B prior/kernel。这里无需在每个 C inner fold 再嵌套训练一个新的 B adapter。

C inner-held 标签继续参与 C 的训练监督，所以它不是独立验证；U_B 已经过合法 B 的 outer-train 监督，也不能称独立无偏教师。最终完整 C 直接使用本行实际 final B 的 U_B、完整旧 support 头、尺度和中心测度，而不是随便选一个折头。Nnew=0 直接复用对应实际 B，不新增 C 拟合。

C 的分类函数包含固定 B 函数加全类残差，不是把 B 的 α 拼到普通 C α，也不是宣称一次 full refit 自动延续旧函数。在零残差的思想边界，旧列恢复 B；但实际 C 在 Z=0 时也会闭式拟合新的 α，注册即可改变旧 score。新列 padding 为零，在完整 argmax 中仍竞争；不能把该边界推广成“旧类预测保证不变”。

这是 **empirical functional anchoring / 正则延续**。同一旧 support 既用于 B，又进入 C 残差目标；没有继承 B 的后验协方差，故不能称 exact sequential Bayesian posterior。m 固定时，m 未必属于随 U 改变的当前 RKHS；本方法直接在 g 的 RKHS 上正则，不写未经证明的 \(\|f-m\|\) 同空间推论。

## 5. 直接作用于最终全类 score 的联合目标与梯度

在所有 inner folds 上将当前训练 held 记录按类别汇总，每类 mean cross-entropy 为

\[
\ell_c(Z)=\frac1{n_c}\sum_{i:y_i=c}\big[\operatorname{LSE}_{j\in\mathcal C}f_Z(x_i)_j-f_Z(x_i)_c\big].
\]

全类 RMS 风险及唯一外层训练目标为

\[
R_{CE}(Z)=\sqrt{C^{-1}\sum_c\ell_c(Z)^2},\qquad
J(Z)=R_{CE}(Z)+\tfrac12\|Z\|_F^2.
\]

使用稳定 log-sum-exp，温度固定 1，不训练输出校准系数。RMS 使所有类用同一风险形式，不设置具体 TX 或旧/新组专属权重。不再使用 clipped teacher-margin 的 soft keep、keep slack 或 guard；B 函数先验与残差 RKHS 范数是新的保持偏好，不能宣称 keep 已被证明有害或无需保持。

CE 对每一个 wrong-class 列都给出竞争梯度，\(\partial\ell/\partial f=p-e_y\)，无需 max-wrong tie 分支。相较当前 max-margin deficit，改变的是监督风险的形状与信息权重，不是已证实的根因修复。固定温度也可能使 score 梯度较弱，且 RMS/近端仍有偏好；后续必须测量。

对于同一监督 held 集合，若预测错则 CE≥log2，因此 macro 训练错误率≤\(\operatorname{mean}_c\ell_c/\log2\le R_{CE}/\log2\)。这是可能很松的训练上界，不是 held/query 泛化保证，也不是逐类改进保证。保存单独旧/新 CE、margin 和正确率才能判断共享改善。

每个 fold 的 m/M 与标签固定于本阶段，只有 K/L 随 U 变化。令训练 held 的 score 为 F=M_h+Lα，则

\[
d\alpha=-A^{-1}(dK)\alpha,\qquad
dF=(dL)\alpha-LA^{-1}(dK)\alpha.
\]

设 G=∂R_CE/∂F。每条类别 c held 记录的权重为 \(\ell_c/(C R_{CE}n_c)\)；先跨折形成每类风险，再对每折使用这个权重，不能平均不等长 fold risk。R_CE=0 时明确返回零 CE 梯度，不能除零。

\[
\bar L=G\alpha^T,\quad B=A^{-T}L^TG,\quad
\bar K=-\operatorname{sym}(B\alpha^T).
\]

一份已缓存 Cholesky 的两次伴随三角求解即可传播该风险。中心化的全部 VJP 为

\[
\bar Q=\gamma_B\bar L P_q,\qquad
\bar R=\operatorname{sym}\{\gamma_BP_q^T\bar KP_q-
\gamma_Bq(\mathbf1_h^T\bar LP_q)\}.
\]

它包含参考点随 U 改变的贡献。随后 raw Gaussian 的 \(\partial R/\partial D=-R/\tau_B\)，adapted distance 在 D 中的系数为 1/2；对 train/held 两端、interaction 映射和 adapter 全部反传，最终

\[
\nabla_ZJ=(\nabla_UR_{CE})W+Z.
\]

τ/γ/q 无梯度是 **新算法明确固定这些量**，不是漏掉原 FCR8 的动态统计导数。m 用冻结 U_B 计算，所以其 C 梯度严格为零；当前 residual 核仍对 U_C 有梯度。只求 adapter 梯度而不求 closed solve 与 reference VJP 将直接违反本目标。

## 6. 一个有界优化算法，不构造参数网格

每个阶段从 Z=0 开始，最多执行 4 次梯度更新。非零 g=∇J 时取 d=−g/||g||；每次试探 t=(1/8)·2^(−j)，j=0,…,11，最多 12 次。对实际 Δ=td 重算所有 inner 头与完整 J，接受第一个同时满足

\[
J(Z+\Delta)\le J(Z)+10^{-4}g^T\Delta+tol,
\qquad J(Z+\Delta)\le J(Z)+tol
\]

的 trial；tol 沿用 \(128\epsilon_{64}\max(1,|比较标量|)\)。不以 held accuracy 选步，不挑最佳试探，拒绝 cache 不覆盖最后接受 cache。g 真零时停止；未通过 12 次时记 `TRIAL_BUDGET_EXHAUSTED`，保存最后接受状态，结束本阶段，不改步长、增试探或反馈重跑。

4 次更新、初始步 1/8 与最多 1/2 的 Z 路径上界保留已有预算尺度。12 次是 bounded Armijo 的实施上限：当前完整证据证明三次试探经常耗尽，**没有证明 12 最优，也没有证明更多试探改善外层结果**。在带宽正、范数非零且前向可微的区域，正确下降方向足够小步具有通常的 Armijo 下降依据；零带宽、数值不可表示或非光滑边界不能套用该局部论据。

若有 F≤3 个 inner folds，每个信息阶段最多 1+4×12=49 次 inner forward objective，即最多 49F 次 student head factorization；最多 4F 次伴随线性求解、8F 次三角求解，另一次完整 final head。prepaid initial/cache reuse 按实际扣重，缓存展示不是新增拟合。C 每折 prior B 头最多单独拟合一次、prior scores 缓存一次；可复用合法已有 head 时不重复收费。该上限不包含 B 的训练及准备阶段，二者必须另列。

去掉第二个 keep 伴随可能减少单次 backward 工作，12 次试探却可能增加 forward 工作；总效率需要实际测量，不能由参数少或公式短宣布节省。

## 7. 退化、数值失败与可声明边界

| 条件 | 固定语义 |
|---|---|
| Nnew=0 | 精确复用实际 B；不新增 C head/优化计数 |
| 原始 K=1 或阶段合法 train K=1 | 无独立 inner held，不监督更新 adapter；仍登记并执行本阶段合法完整 support 的闭式 head（B 普通 ridge、C 非零先验残差 ridge）。outer oof 为 N/A，proxy 不冒充真实 K1 验证 |
| H 全零或 retained r=0 | Z 空、U=anchor；不进行 adapter 更新。正带宽下 C 残差 head 仍可能非零，应登记其真实拟合，不能称完整方法没有运行 |
| τ0=0、s0>0 | 用原始 interaction 完全特征等价关系的 PSD kernel 及原 R0 γ；不对 equivalence 边界求导、不训练 adapter，B/C score 仍由精确闭式头决定。C 可能有非零 score，不等于 adapter 可学。不得合成小正带宽 |
| τ0=None（例如仅一个旧类）或原 R0 trace=0/γ=None | 该旧参考未提供可定义尺度，明确使用零 residual kernel与零 B prior；g=0，A=I 的规范 α=Y−M，但所有核预测为零，全部类按既定 tie rule。标记 `NO_OLD_KERNEL_INFORMATION`，不静默切到新集合带宽、另一方法或地面统计。C=1 可得到平凡单类预测，C>1 不声称成功注册 |
| 旧/新同特征、不同标签 | 保留相同 kernel 行；K+I 仍可解。确定性单样本算法不能同时区分完全相同输入，记录冲突，不改标签/配额或加 jitter |
| 近重复、下溢、不可表示 W/U 或非有限值 | 使用已有稳定距离/中心化和 float64；真正非同特征正距离若数值下溢到零，或 Cholesky/残差检验失败，记技术失败并保留产物，不放宽公式或当成性能退化 |
| 块范数为零、并列 logits | 零块保持零及零 adapter VJP；全部类并列由统一 class-ID 规则处理，没有 old 优先 tie rule |

这些退化是同一公式的明确分支，不是为改善结果选择 fallback。当前正式候选覆盖 6 个旧类；单旧类或 trace0 分支说明该方法在缺少旧域 kernel 信息时的明确限制。

K+I 的残差检验使用本次实际 K：\(\|A\alpha-E\|/[(1+\operatorname{tr}K)\|\alpha\|+\|E\|]\)。不能沿用动态 trace 已变化后的 1+s0 作为未经核实的谱界。数值检查针对实现正确性，不新增数据重验证。

## 8. 信息权限与真实资源计量

允许当前 received support、标签、注册类表、冻结 Phase1 和本行真实 B。本文不需要地面原型或聚合摘要，新增地面逐样本数据/统计请求均为零；如实现额外引入任何摘要，必须另作结构修改并核对协议，不能默认为本文已授权输入。无 source/clean 样本、逐样本源特征、BN 状态、query 拟合、query 选步或评分反馈。

B/C outer-held 不进入 preparation、字典、kernel、prior、超参数或优化；只在冻结状态下逐样本评分。真实 query 只在后续明确评分任务中访问，先固定 predictions，独立 scorer 才连接 truth。当前文档与 support 实施不启动 query 评分。class-ID 置换时 old/new 注册映射一同置换，所有数学公式和生成规则保持同一形式。

必须把实际优化参数、闭式拟合系数、继承状态和固定数值字典分别报告：

| 数值对象 | 数量/float64 数值字节 | 计量含义 |
|---|---|---|
| 当阶段梯度更新坐标 Z | 736r；8×736r B，r≤8 | 实际 gradient-trainable parameters；上限 5888，不把空 rank 算已训练 |
| 当前部署 U_C | 736×8=5888；47104 B | 全原坐标 adapter 数值状态，不把 active r 与存储混淆 |
| 固定 V0 | 8×736；物化 47104 B | 不训练但占实际 RAM；只有代码实际重建且未序列化时，部署文件才可不含该数组 |
| 冻结 U_B | 736×8；47104 B | C prior 的真实继承 adapter；若和 U_C 不共享实际 buffer，必须另计 |
| C residual α / B prior α_B | N_C C / N_B C_B；各乘 8 B | support 拟合的决策参数，虽无梯度优化仍是 learned state |
| original/adapted support geometry、中心量、类别映射 | 按实际 dtype/shape/nbytes 与共享 buffer 计 | prior 与 residual 都需要真实 feature/kernel 参考；不能只报 U/V |
| H/W/谱、梯度和 trial cache | 训练瞬态与证据存储分列 | 不把这些排除出训练峰值；不把 NPZ 证据大小当部署包 |

常规 dense 部分为每次 kernel O(N²d) 工作量、Cholesky O(N³)、primal/adjoint triangular solve O(N²C)，adapter O(Nd×8)。C 先验在 N 个当前 support 上评价约增加 O(NN_Bd) 的 kernel 工作，在所有 trial 之前缓存；单 query 最终还需评价实际 B prior 与当前 residual 两个 kernel 函数。固定字典与少参数不使这部分自动消失。

需要实际记录：硬件/线程/dtype、IQ feature extraction、准备/prior/head/forward/backward/final fit、单样本与固定批大小推理时间、所有成功及拒绝 trial、峰值 RSS/显存、唯一 buffer 常驻数值字节、实际序列化部署包字节、增量星地上传字节，以及完整/紧凑日志与 CSV/JSONL 大小。序列化压缩字节与 RAM nbytes 分开；共享或嵌套计时不相加。未测项写 N/A。

FCR8 的当前基准是 CPU float64/BLAS 线程 2；run wall 3551.958 s、最大 lane RSS 907350016 B、最大数值常驻状态 6608384 B、U+V0 为 94208 B；部署包/增量传输/显存为 N/A。这些只定位当前工作量，不能作为 AJLR 或星载端收益。本文没有执行新资源测量，不能填“已节省”。

## 9. 最小合成正确性验证

实现前后使用有限的确定性合成构造，不搜索超参数或挑 seed：

1. **closed head 与目标**：直接乘积 RKHS/有限特征 primal oracle 对照 α=(K+I)^−1(Y−M)，包括不平衡 class counts、非零 prior、固定旧测度和非零 residual 样本均值；确认没有隐藏 intercept 或二次 residual-centering。验证 sum-class=0、正规方程和 PSD/SPD。
2. **完整梯度**：在正带宽、非零块的非退化点，分别检查 B/C 的 Z 全链 finite differences，包括 train、held、参考子集、α 隐式求导和 proximal Z；再检查改变参考点 embedding 的显式扰动。构造仅反传 query/train 而 detach ref 会失败的反例。对固定 τ/γ 不沿用原动态统计梯度。
3. **继承与统计边界**：U_C=U_B 时新增注册集合不改变旧旧 kernel；Z=0 复制原坐标 U_B；C prior 是实际 B 函数与类映射 padding；同 fold 只用 old inner-train，final 使用 final B。outer-held/query 突变、输入额外 scorer 字段不改变训练状态，或被接口拒绝。
4. **注册不等于保持**：构造旧列保持但新列赢得 argmax 的样本，确认全部类竞争；构造 residual 修正导致旧类错误，确认没有以 anchor norm 或 CE 下降宣称准确率保证。initial C residual 可改变旧 score，日志必须可分离其变化与后续 adapter 变化。
5. **退化及稳定数值**：K1、单旧类、trace0、τ0=0 的非零 C head、rank0、零块、同特征异类、近重复正距离；不放宽容差、不用 jitter 或未知历史状态补救。类/行置换和单样本/批形状保持同一预测。
6. **优化与真实计数**：合成 Armijo 接受/拒绝/12 次耗尽、零梯度、最后接受 cache 保留；不选最佳 trial、不用 accuracy 接受。校验 head、factorization、伴随 solve、trial attempt/completed、共享 preparation 和字节计费，不将重复 cache 展示相加。

这些检查验证实现和权限，不验证实数据性能。此文只给验证设计，未声称上述测试已运行通过。

## 10. support 验证与可证伪报告

使用已登记的当前合法 capsule/split 与确定性物理 folds；不新增数据重验。由主 Agent 将唯一候选、实际矩阵、单个固定 seed 角色及有限预算登记到新 run。不能从本轮较好的局部 K/新增类/receiver 挑选下一轮样本，也不能通过 seed 选择改善报告。support 范围沿已有授权；具体首轮规模由主 Agent 在运行前登记，不由本文扩成新硬门槛。

完整原有 K×新增类表继续保留：K=1/5/10/20，旧类 6，新增 0/2/5/10/20；无法独立持出的 K1 记 N/A，不能用 proxy 补齐。若首轮只运行预登记子矩阵，未运行项明确 N/A，后续仅按既有科学阈值扩展，不制造完整矩阵早期许可。

比较固定 R0 与唯一 AJLR_seq，不加入 reset 或若干损失权重候选。B 与 C 使用同一 row 的旧物理 support 和旧 outer-held 配对。外层 held 从不参与选步或训练；冻结候选的结果只报告，不反馈更改当前候选参数。记录 A（未测则 N/A）、B、C 旧、C 新、H、B−A、B−R0、B−C_old、逐配置绝对新旧差；H 与差先逐配置计算再汇总，保留 receiver/scenario/K/新增数分层。

在既有头/score cache 上额外记录机制量，避免为机制日志增加拟合：

\[
f_{C,final}-\operatorname{pad}(f_B)
=\underbrace{f_{C,Z=0}-\operatorname{pad}(f_B)}_{\text{固定 U_B 下的注册 residual head 变化}}
+\underbrace{f_{C,final}-f_{C,Z=0}}_{\text{本阶段 adapter 与闭式 head 联合变化}}.
\]

记录对应旧/新 train-held score RMS、全类 margin、old→new/new→old winner 转移、正确/错误配对计数、旧列内诊断与全部类竞争的差，以及实际 trace、fixed τ/γ、kernel block 变化。只在训练 held 上形成监督 diagnostics；外层这些量在状态冻结后独立生成，不用于优化。old-only argmax 只解释误差，不作为部署规则。

候选假设被支持需要完整同 row 外层结果显示 B 与 C 新旧任务能共同改善，且注册下降/绝对差朝软目标推进；不以一个 mean H、某个较好 K 或内层 CE 下降代替。若训练目标与 head inheritance 正确执行，而 B 持出继续下降、注册旧类损失无改善或新类受 prior 压制，则记录该结构假设未获得支持。若只是 trial 耗尽或数值失败，区分算法预算/技术失败与科学结果；不把扩大 grid 或重跑到好结果当修复。

## 11. 亲自核读的数学来源及适用边界

- [Rasmussen 与 Williams，Gaussian Processes for Machine Learning，官方第 2 章 §2.7、式 2.37–2.38，印刷页 27](https://gaussianprocess.org/gpml/chapters/RW2.pdf)：固定确定性均值 m 的残差代数为 \(m(X_*)+K_* (K+\sigma_n^2I)^{-1}(y-m(X))\)。这直接说明非零函数先验下的 residual solve，而不是只初始化参数。本文没有 GP posterior uncertainty、Gaussian 分类似然或顺序 Bayesian 更新的有效性声明。
- [Schölkopf、Herbrich 与 Smola，A Generalized Representer Theorem，作者原文 §2 Theorem 1/式 15–16；印刷页 422 Remark 1](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)：正定核、基于有限训练评价的风险和单调 RKHS 范数正则使解位于训练核展开；Remark 1 讨论 \(\frac12\|f-f_0\|^2\) 的偏好。本文对每个残差输出 g_j 使用 Theorem 1；m 可不在当前 RKHS，故不借 Remark 1 宣称任意变核时存在同空间距离或旧准确率保证。kernel 改变后的 adapter 目标仍非凸。
- [Bertinetto 等，ICLR 2019，Meta-learning with differentiable closed-form solvers，§3.2、式 3–5](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)：ridge 的正规解与样本空间 Woodbury 形式支持把解析头放进可微学习链。本文使用 centered Gaussian RKHS、fixed B prior 和合法目标 support PEFT，不复制论文的跨 episode 地面 meta-training、learned λ/temperature 或任何目标测试集监督；论文 benchmark 效果不能迁移成 AJLR 成功证据。

固定旧测度中心核、RMS CE 和有限 Armijo 是本文提出的单一组合假设，不是这些论文已经验证的方法。有限差分、PSD 证明和优化下降只能证明相应数学/实现性质，不能证明旧 query 提升、新旧差达标或星载资源减少。
