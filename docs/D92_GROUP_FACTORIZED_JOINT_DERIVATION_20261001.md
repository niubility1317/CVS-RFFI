# D92 旧新组概率分解、联合新头与 gate 的严格推导

状态：**仅源码与数学结构设计；本文未实现、未做数值验证、未选择实验矩阵、未启动实验。** 本文没有读取实际数据、特征 cache、checkpoint、prediction、truth、真实评分、结果报告或实验索引。固定合规 Phase1、practical residual、合法目标 support 与逐样本全注册类判决边界保持不变。本文不授权修改、停止或重启任何健康任务。

关键判断：原 hard-constraint 组分解问题可行，在两组训练 support 均非空时有唯一有限的 RKHS 函数及自由截距。它对所有输入精确保留实际 B 的旧类条件排序，但只在指定旧 train 点保护旧—新 margin。它不能保证旧 query 零遗忘、新类准确率或理想目标。原问题的普通隐式梯度只在正则区域成立；将现有 MarginQP 换成另一个依赖严格互补的 hard solver，不能消除退化失败。已确认的独立实现候选为 **D92-GroupBarrierJointLocalRidge-v1**：对全部旧点—新类约束逐条采用固定正 log-barrier。该修正保留严格可行的 margin，去掉 active-set 和 max-tie 的求导依赖，并有显式原目标差界；有限 barrier 不能冒称原 hard 最优解。第 5 节 hard solver 仅为原问题的推导背景，不是此 v1 的生产路径。

## 1. 范围、阶段与不可混用的 prior

设旧类集合为 \(\mathcal O\)，新增类集合为 \(\mathcal N\)，旧类数为 \(c_o\)，新增类数为 \(q\)。一个 head 的合法训练物理行集合为 \(T=O\cup N\)，其中 \(|O|=m\ge1\)、\(|N|=p\ge1\)、\(n=m+p\)。\(H\) 是该 inner fold 的合法 held support。每个物理行保留一次原损失权重；重复输入不等于可以删除物理观测。

B 使用原 CE-only 外层、解析 ridge、固定 8 字典和白化小 adapter。C 的 adapter 起点是该 row、scope、fold 的实际 \(U_B\)，不是零、旧 checkpoint 或另一条路径。C 使用同一字典与白化硬球，写成

\[
U=U_B+ZW^\top,\qquad \|Z\|_F\le\tfrac12.
\tag{1}
\]

保留未被白化坐标覆盖的 anchor 分量。\(W\)、原几何的 \(\tau,\gamma\) 与实际 B 固定；C 不反向更新 B、Phase1、字典或尺度。小 adapter 的监督只能来自该次合法 support 分割；不加入源样本、源逐样本特征、query 拟合或评分反馈。

**Inner prior** 是在冻结的实际 \(U_B\) 下，仅用该次 old inner-train 重新拟合的旧类解析头。尺度也只来自相应 old inner-train。它可以在当前 train/held 特征上预测，但不能把含 held 标签的完整 B head 用于 inner 目标、下界或选模。**Final prior** 才是当次实际完整 B 对象。两个 prior 的列 ID、物理行、scope 和继承来源必须明确；不能混用 final 与 inner 的分数。

记对应 prior 的旧类函数为 \(f_B(x)\in\mathbb R^{c_o}\)。C 不改变它。所有下文的 \(f_B,d\) 都对应当前 head 的合法 prior，而非历史评分选中的头。

## 2. 全注册类概率与排序保证

令

\[
\ell^B_y(x)=\log\operatorname{softmax}(f_B(x))_y,
\qquad
\ell^N_j(x)=\log\operatorname{softmax}(h(x))_j.
\]

gate \(g(x)\in\mathbb R\) 的正方向表示旧组概率较大。统一 class log-score 为

\[
s_y(x)=\log\sigma(g(x))+\ell^B_y(x),\quad y\in\mathcal O,
\qquad
s_j(x)=\log\sigma(-g(x))+\ell^N_j(x),\quad j\in\mathcal N.
\tag{2}
\]

因此

\[
P(y\mid x)=\sigma(g)p_B(y\mid x),\quad
P(j\mid x)=\sigma(-g)p_N(j\mid x),\quad
\sum_{c\in\mathcal O\cup\mathcal N}P(c\mid x)=1.
\tag{3}
\]

有限实数 logits 下，所有条件概率严格为正，log-score 有限且小于等于零。组概率分别是 \(\sigma(g)\) 和 \(\sigma(-g)\)。它们是模型输出，不是 query 的真实 role 或类别数量。

对任意输入与任意两旧类，

\[
s_y-s_k=\ell^B_y-\ell^B_k=f_{B,y}-f_{B,k}.
\tag{4}
\]

这精确保留 B 的旧类排序、成对 gap 与并列集合，覆盖全部输入，并非只覆盖旧 support。条件旧类 argmax 与 B 相同。**C 因而也不能修复 B 的旧—旧排序错误。** 旧类适应提升必须由 B 的合法适应承担；C 的 gate 只能改变旧组与新组的相对竞争。

对旧类 \(y\) 与新类 \(j\)，

\[
s_y-s_j=g+\ell^B_y-\ell^N_j.
\tag{5}
\]

全部注册类统一 argmax 必须计算 B、新头和 gate。可以分别求两组最大条件分数，再比较式 (5) 中两组最大候选，作为全类 argmax 的代数实现；这不是先按 \(g\ge0\) 排除一组。两个组都必须实际参与，tie-break 使用同一固定 class ID 规则。不得使用 query role、真实类数、配额、跨样本统计或全局重排。

数值保证要分开表述。直接把极大的共同 \(\log\sigma(g)\) 加入每个旧列，float64 舍入可能吞掉小的旧类差值。预测比较应保留式 (4)、(5) 的代数结构，并使用稳定 log-sigmoid/log-softmax；不先形成概率再取 log。数学上的排序等式不自动等于逐位浮点等式。跨组 gap 接近数值误差或真实并列时只能保证既定 tie-break 与已记录容差；不得伪称严格判决保证。

## 3. 只用新 train 的完整解析 ridge 头

当前 C 的 raw PSD kernel 为 \(k_U\)。正 \(\tau\) 时沿用现有几何：

\[
k_U(x,z)=\gamma\exp\!\left[-\frac{d_0(x,z)+d_U(x,z)}{2\tau}\right].
\tag{6}
\]

\(\tau,\gamma,d_0\) 固定，不为新类重新估计 bandwidth 或 trace scale。\(K_N=k_U(N,N)\)、\(L_{AN}=k_U(A,N)\)，\(A\) 可同时包含 old train 与 held。新头只用 \(N\) 标签拟合：

\[
Y_{ij}=\mathbf1[y_i=j]-1/q,
\quad
\min_{v_j,b_j}\ \tfrac12\|v(N)+\mathbf1_p b-Y\|_F^2
+\tfrac12\sum_j\|v_j\|_{\mathcal H_{k_U}}^2.
\tag{7}
\]

\(b\in\mathbb R^q\) 是完整、未惩罚自由截距。解析系数为 1，没有额外类别权重或温度。等价的 raw-kernel Schur 表示为

\[
\begin{bmatrix}I_p+K_N&\mathbf1_p\\\mathbf1_p^\top&0\end{bmatrix}
\begin{bmatrix}\alpha_N\\b_N\end{bmatrix}
=\begin{bmatrix}Y\\0\end{bmatrix},
\qquad h_A=L_{AN}\alpha_N+\mathbf1_A b_N.
\tag{8}
\]

\(I+K_N\succeq I\)，因此无需逆 \(K_N\)、伪逆或 jitter。实施用一次 SPD 分解和真实 RHS solves，不显式构造逆。

为核对 BranchLocalRidge 的 centered 实现，令 \(C_p=I-\mathbf1\mathbf1^\top/p\)。它等价于

\[
K_c=C_pK_NC_p,
\quad L_c=(L_{AN}-\mathbf1_A\mathbf1_p^\top K_N/p)C_p,
\quad \alpha_N=(I+K_c)^{-1}C_pY,
\quad b_c=\mathbf1_p^\top Y/p.
\tag{9}
\]

\(h_A=L_c\alpha_N+\mathbf1_A b_c\)，raw 截距为 \(b_N=b_c-\mathbf1_p^\top K_N\alpha_N/p\)。式 (8)、(9) 表示同一函数。中心化不允许删掉自由常数，也不允许 detach 移动的 train 均值；raw 表示可以直接给出等价完整 VJP。新头不拟合 old 标签，但必须在 old train 特征上真实求值，因为其分数进入 gate 下界。

## 4. 原 hard gate：下界、凸性与唯一函数

将当前 B 补零到全部注册列：\(M=\operatorname{pad}(f_B,0_{\mathcal N})\)。对旧 train 的真实类定义

\[
d_i=\min_{c\ne y_i}\{M_{i,y_i}-M_{i,c}\}
=f_{B,y_i}(x_i)-\max\{\max_{k\in\mathcal O\setminus\{y_i\}}f_{B,k}(x_i),0\}.
\tag{10}
\]

保留原负值与零，不 clamp。由式 (5)，全部旧—新约束等价于

\[
g(x_i)\ge a_{ij}:=d_i-\ell^B_{y_i}(x_i)+\ell^N_j(x_i),\quad j\in\mathcal N,
\]
\[
g(x_i)\ge a_i:=d_i-\ell^B_{y_i}(x_i)+\max_j\ell^N_j(x_i).
\tag{11}
\]

原 gate 使用全体合法 \(T\) 的物理 Bernoulli loss，\(r_i=1\) 表示旧 train，\(r_i=0\) 表示新 train：

\[
\min_{w\in\mathcal H_{k_U},b\in\mathbb R}
J(w,b)=\tfrac12\|w\|_{\mathcal H_{k_U}}^2
+\sum_{i=1}^n[\operatorname{softplus}(t_i)-r_it_i],
\quad t_i=w(x_i)+b,
\quad t_O\ge a.
\tag{12}
\]

这是固定新头、固定当前 kernel 的凸问题。自由截距不正则化。取 \(w=0,b>\max_i a_i\) 即严格可行，所以 Slater 成立。两组物理训练行均非空时，bounded \(J\) 先限制 \(\|w\|\)；有限 \(k(x_i,x_i)\) 随后限制 \(w(x_i)\)。\(b\to+\infty\) 的新组 loss 与 \(b\to-\infty\) 的旧组 loss 都趋于无穷，故存在有限最优解。

\(\|w\|^2/2\) 对 \(w\) 严格凸；固定 \(w\) 后 Bernoulli loss 对有限 \(b\) 严格凸。因此 \((w,b)\) 唯一。训练 representer 张成空间外的分量只增加范数，最优时为零。PSD 或奇异 kernel 不破坏函数唯一性；系数与约束乘子仍可能非唯一。

所有可行解在旧 train 满足完整真实类最小 log-margin 不低于 \(d_i\)。旧—旧 gap 已由式 (4) 保留，旧—新 gap 由式 (11) 保证：\(d_i>0\) 时真实类严格胜过所有列，\(d_i=0\) 时只能保证并列，\(d_i<0\) 时不推出正确分类。数值可行残差为 \(\varepsilon\) 时只保证 \(d_i-\varepsilon\)。这些结论不延伸到独立 held/query。

式 (11) 可用 \(\operatorname{LSE}(f_B)-\max\{\max_{k\ne y_i}f_{B,k},0\}+\ell^N_j\) 计算同一 \(a_{ij}\)，避免先算 \(d_i\) 再与 \(f_{B,y_i}\) 做巨大相消。它只是等价数值表达；原 \(d_i\) 的定义与记录不变。

## 5. 原 hard gate 的 kernel-only 求解与正则 VJP

### 5.1 KKT 与无逆 kernel Newton

令 \(K=k_U(T,T)\succeq0\)，\(E\) 选择旧 train 行，\(\mu\ge0\) 为式 (11) 的乘子。canonical representer 与截距满足

\[
\alpha=r-\sigma(t)+E^\top\mu,
\quad t=K\alpha+\mathbf1 b,
\quad \mathbf1^\top\alpha=0,
\quad Et-a\ge0,
\quad \mu\odot(Et-a)=0.
\tag{13}
\]

最后一个截距方程必须保留。\(K\alpha\) 相同不等于可以随意改变 cross-kernel 展开；式 (13) 给出与函数 stationarity 一致的 canonical 系数。

以下是真实 Newton 线性代数，不是将非线性 logistic 当成解析 ridge。令 \(q_0=\sigma(t)-r\)、\(D=\operatorname{diag}(\sigma(t_i)\sigma(-t_i))\)。有限实数下 \(D\succ0\)。定义

\[
H_D=I+\sqrt D K\sqrt D\succ0,
\quad V=K-K\sqrt D H_D^{-1}\sqrt D K,
\]
\[
c=\mathbf1-VD\mathbf1,
\qquad s=\mathbf1^\top D\mathbf1-(D\mathbf1)^\top V(D\mathbf1)
=(\sqrt D\mathbf1)^\top H_D^{-1}(\sqrt D\mathbf1)>0.
\tag{14}
\]

这里 \(V=\Phi(I+\Phi^\top D\Phi)^{-1}\Phi^\top\)，\(K=\Phi\Phi^\top\) 只用于证明，不要求求 feature、逆 Gram 或做 kernel SVD。\(s\) 是未正则化截距的 Schur complement。实现从最后一个 SPD solve 内积计算 \(s\)，避免两个大数相减；\(c\) 也可写为 \(\mathbf1-K\sqrt D H_D^{-1}\sqrt D\mathbf1\)。每一步 \(D\) 改变，分解不能伪称始终复用。

给定独立的 working set \(\mathcal A\)，令 \(h_0=\alpha+q_0\)。无约束 Newton 的 train 响应与截距为

\[
\Delta b_0={-\mathbf1^\top q_0+(D\mathbf1)^\top Vh_0\over s},
\qquad \Delta t_0=-Vh_0+c\Delta b_0.
\tag{15}
\]

置 \(S_D=V+cc^\top/s\)、\(Q_{\mathcal A}=E_{\mathcal A}S_DE_{\mathcal A}^\top\)。独立的 augmented evaluation 行使 \(Q_{\mathcal A}\succ0\)。等式 Newton 子问题解为

\[
Q_{\mathcal A}\nu=a_{\mathcal A}-t_{\mathcal A}-E_{\mathcal A}\Delta t_0,
\quad \Delta b=\Delta b_0+c_{\mathcal A}^\top\nu/s,
\]
\[
R=h_0+D\mathbf1\Delta b-E_{\mathcal A}^\top\nu,
\qquad \Delta\alpha=-R+\sqrt D H_D^{-1}\sqrt D KR.
\tag{16}
\]

\(\nu\) 是 Newton 二次子问题乘子；只有在终点方向为零且 KKT 成立时才等于原 \(\mu\)。它不能在中途直接充当已收敛的原乘子。

从 \(\alpha=0,b>\max a\) 的严格可行状态起步。每个方向用所有 inactive 物理约束的 slack 截断最大步长，再对原 \(J\) 做 feasible Armijo 回溯；完整扫描确认接受点可行。到达新边界要重新核查 working set，负乘子对应约束需要释放或交换。相同分数的并列进入按固定物理 ID 规则处理，同时仍检查全部约束。独立性检查只服务当前线性求解；不能删除物理 loss 或偷偷删除约束。冗余 tight 行可能需要 conic basis exchange，简单“挑一组独立行并忽略其余”不足以证明完整 normal cone。未提供处理退化与循环的证据时，这条路线不能被称为通用稳定 solver。

### 5.2 完整普通隐式 VJP 的成立条件

固定正确 active 集 \(\mathcal A\)，要求 augmented active evaluation 行独立、active 乘子严格正、其余 slack 严格正，并且 \(a(U)\) 局部可微。设 \(a\) 的 max 为唯一赢家，或另有证明表明并列分支的导数相同。在该区域，定义

\[
F=\begin{bmatrix}
\alpha+\sigma(K\alpha+\mathbf1b)-r-E_{\mathcal A}^\top\mu\\
\mathbf1^\top\alpha\\
E_{\mathcal A}(K\alpha+\mathbf1b)-a_{\mathcal A}
\end{bmatrix}=0.
\]

其 Jacobian 是

\[
\mathcal J=\begin{bmatrix}
I+DK&D\mathbf1&-E_{\mathcal A}^\top\\
\mathbf1^\top&0&0\\
E_{\mathcal A}K&\mathbf1_{\mathcal A}&0
\end{bmatrix}.
\tag{17}
\]

正则条件使其可逆，即使 \(K\) 奇异；可用有主元的 LU/QR solves，不对 \(K\) 求逆。由 finite-feature 正定 primal Hessian 与独立 active 行可证明：齐次 KKT 方向先给出零函数方向与零截距，随后零乘子，最后 canonical \(\Delta\alpha=0\)。这与假称所有 PSD kernel 的 active 系统自动正定不同。

gate 在 held 上 \(g_H=L\alpha+\mathbf1_Hb\)，上游为 \(v=\partial\mathcal L/\partial g_H\)。解完整伴随

\[
\mathcal J^\top\begin{bmatrix}\lambda_\alpha\\\lambda_b\\\lambda_a\end{bmatrix}
=\begin{bmatrix}L^\top v\\\mathbf1_H^\top v\\0\end{bmatrix}.
\tag{18}
\]

以全 Gram Frobenius 约定，

\[
\boxed{\bar L=v\alpha^\top,\quad
\bar K=-\operatorname{sym}[(D\lambda_\alpha+E_{\mathcal A}^\top\lambda_a)\alpha^\top],
\quad \bar a_{\mathcal A}=\lambda_a.}
\tag{19}
\]

\(\mathbf1_H^\top v\) 一般不为零；删掉它即漏自由截距链。式 (19) 同时包含 loss、乘子响应、kernel 改变与活动下界响应，不可 detach 乘子或把约束 RHS 当常数。

唯一 max 赢家 \(j_i^*\) 时，\(\bar h_{O,i}=\bar a_i(e_{j_i^*}-p_{N,i})\)。若多个新类并列，真实方向导数为

\[
D\max_j\ell^N_j[\dot h]
=\max_{j\in\operatorname{Argmax}h}\dot h_j-p_N^\top\dot h.
\tag{20}
\]

这一般不是单一线性映射。用固定均分或任选一个赢家是一个选定 subgradient，不是唯一真实 Jacobian。全部 \(j\) 绑定与 max 的**可行域**完全相同；但同一旧行的 active 左端重复，可能有非唯一乘子。约束伴随 \(\sum_j\bar a_{ij}(e_j-p_N)\) 不能因此被省略，也不能声称重复 KKT 行自然消除了非光滑性。

## 6. 推荐的最小可微修正：全部 pair 的正 barrier

仅对 \(g_i-a_i\) 加 barrier 仍保留式 (20) 的 max-tie 问题。为避免任意 active/tie 分支，采用全部 \(mq\) 个平滑 pair 下界 \(a_{ij}\)；令 \(\zeta>0\) 在当前 head 的整个外层过程中固定：

\[
J_\zeta(w,b)=J(w,b)-\zeta\sum_{i\in O}\sum_{j\in\mathcal N}
\log z_{ij},\qquad z_{ij}=t_i-a_{ij}>0.
\tag{21}
\]

这是一项明确的方法修正，不能藏在求解器容差里。它改变原目标，仍使用相同 RKHS 正则、全物理 Bernoulli loss 与 free b；没有 softening 约束。因为任一 \(z_{ij}\downarrow0\) 都令目标趋于无穷，最优点严格满足全部原 margin。两组非空的线性尾部 loss 压过 barrier 的对数尾部，所以有限唯一解仍存在。

每个 \(a_{ij}\) 都是 smooth log-softmax 的单列；并列新类不会导致 max 求导或冗余 active KKT。定义

\[
\mu_{ij}=\zeta/z_{ij},\quad
\mu_i=\sum_j\mu_{ij},\quad
w_{ij}=\zeta/z_{ij}^2,\quad \omega_i=\sum_jw_{ij},
\]
\[
\alpha=r-\sigma(t)+E^\top\mu,
\quad \mathbf1^\top\alpha=0,
\quad \widetilde D=D+E^\top\operatorname{diag}(\omega)E.
\tag{22}
\]

严格可行有限点下，\(\widetilde D\succ0\)。finite-feature Hessian 为

\[
\begin{bmatrix}I&0\\0&0\end{bmatrix}
+[\Phi,\mathbf1]^\top\widetilde D[\Phi,\mathbf1]\succ0.
\tag{23}
\]

截距仍未正则化。约束梯度之间是否独立、乘子是否严格互补、kernel 是否满秩都不是正定性的必要条件。\(\tau>0\) 的 smooth kernel 与有限 logit 域内，唯一解对 K、cross kernel、全部 \(a_{ij}\) 平滑；其 Jacobian 不要求 hard active 集。数值上的强条件或 curvature underflow 仍需真实处理，数学正定不能冒称任意 float64 都可稳定求解。

### 6.1 原 hard 目标差界与固定 barrier 法则

由 \(\mu_{ij}=\zeta/z_{ij}\)，barrier 点对原 hard 问题满足 stationarity、dual 非负与严格 primal 可行。其原问题 primal-dual gap 恰为

\[
\sum_{ij}\mu_{ij}z_{ij}=mq\zeta,
\qquad 0\le J(w_\zeta,b_\zeta)-J^*\le mq\zeta.
\tag{24}
\]

这不是 CE、准确率、query 风险或 parameter-distance 界。全类 log-margin 严格大于 \(d_i\)，但当 \(d_i<0\) 时仍不推出正确分类。

式 (24) 的等式要求精确 stationarity。有限数值求解不能只输出预设的 \(mq\zeta\)，就把实际原问题 gap 认定为该数。原 hard 问题可用以下 Fenchel dual 作为独立 certificate。令 \(u_i\in[-r_i,1-r_i]\)、\(\mu_{ij}\ge0\)、\(\alpha=E^\top[\sum_j\mu_{ij}]-u\)、\(\mathbf1^\top\alpha=0\)，则合法 dual 值为

\[
\mathcal D(u,\mu)=-\tfrac12\alpha^\top K\alpha
-\sum_i\{(u_i+r_i)\log(u_i+r_i)+(1-u_i-r_i)\log(1-u_i-r_i)\}
+\sum_{ij}\mu_{ij}a_{ij},
\tag{24a}
\]

采用 \(0\log0=0\)。精确 barrier 解令 \(u=\sigma(t)-r\)、\(\mu=\zeta/z\) 后得到式 (24)。实际 certificate 还要核对 dual domain、free-b 平衡、function stationarity 与计算的 \(J-\mathcal D\)；未满足 dual 可行性的数组不能给出下界。仅有一个很小的 Euclidean residual、solver success 或预设 budget 均不足以冒称已证 gap。数值误差与原目标差合同分开记录。

本候选已确认固定每物理输入的原 head 目标差预算为 \(\varepsilon_{\rm bar}=10^{-4}\)。令

\[
\delta=n\varepsilon_{\rm bar},\qquad \zeta=\delta/(mq).
\tag{25}
\]

于是 \([J(w_\zeta,b_\zeta)-J^*]/n\le10^{-4}\)，这是 sum Bernoulli loss、RKHS 系数 1 下的数值近似合同，不是准确率阈值或 barrier grid。\(n,m,q\) 是该 head 固定的物理 support/注册元数据；在第一次 outer 更新前固定、登记，所以 C 的 \(\mathrm d\zeta=0\)，不产生漏掉的 adapter 导数。Inner 与 final 因训练物理行数不同而各自计算固定 \(\zeta\)，法则和每物理输入 gap 上界相同。该数值不来自源/目标成绩、held CE 的改善、失败后更高分或 query 结果，不允许试探后换值。

**仅知道 eps64 无法推出普适稳定的正 \(\zeta\)。** 任意 logits/Gram 条件下都存在比给定机器尺度更小的可表达 slack 或 curvature。\(10^{-4}\) 由显式原目标差合同定义，不声称由 eps64 推出普适稳定性。当前缺项是独立合成求解与完整梯度证据；实施须明确登记独立 schema/version、式 (21)、(25)、实际 \(\zeta\) 与残差尺度。它不是原 hard 方法的一次静默重跑。

### 6.2 Barrier 的稳定 Newton 步与完整 VJP

在严格可行点，gradient 用 \(q_\zeta=\sigma(t)-r-E^\top\mu\)。将式 (14)、(15) 中 \(D\) 替换为 \(\widetilde D\)、\(q_0\) 替换为 \(q_\zeta\)，得到 unconstrained Newton 方向。其 SPD factor 为 \(I+\sqrt{\widetilde D}K\sqrt{\widetilde D}\)，没有 active solve。令

\[
\Delta t=K\Delta\alpha+\mathbf1\Delta b,
\quad \beta_{\max}=\min_{\Delta t_i<0,j}z_{ij}/(-\Delta t_i).
\tag{26}
\]

步长严格小于这个边界，再对式 (21) 做 Armijo 回溯。每个接受点都重新检查全部 pair slack 严格正、finite loss、kernel 与 free-b stationarity；不因 barrier 正号就省掉实际可行扫描。可由下一可表达的内侧数选严格可行上限；若没有可表达的内侧步长，保留已接受状态并报告数值失败。最终认证是式 (22) 的 function stationarity、free-b stationarity、严格 primal slack、原 gap 与 Newton/残差尺度；不是 outer 的 CE 好坏。

barrier 的 canonical residual 只有 \(\alpha,b\) 两块：

\[
\mathcal J_\zeta=
\begin{bmatrix}I+\widetilde DK&\widetilde D\mathbf1\\\mathbf1^\top&0\end{bmatrix}.
\tag{27}
\]

解

\[
\mathcal J_\zeta^\top
\begin{bmatrix}\lambda_\alpha\\\lambda_b\end{bmatrix}
=\begin{bmatrix}L^\top v\\\mathbf1_H^\top v\end{bmatrix}.
\]

完整 kernel 与 RHS VJP 为

\[
\boxed{\bar K=-\operatorname{sym}[(\widetilde D\lambda_\alpha)\alpha^\top],
\quad \bar L=v\alpha^\top,
\quad \bar a_{ij}=w_{ij}(\lambda_\alpha)_i.}
\tag{28}
\]

每个 tie 列都参与 \(\bar h_{O,i}=\sum_j\bar a_{ij}(e_j-p_{N,i})\)，没有任选赢家、均分一个 max 梯度或 detach RHS。若将 \(\zeta\) 当独立连续输入，另有 \(\bar\zeta=\lambda_\alpha^\top E^\top[\sum_j1/z_{ij}]\)；本文固定 \(\zeta\)，其 C 更新导数为零。这个项说明动态变更 barrier 后不补链会改变梯度。

式 (27) 不要求直接解病态 nonsymmetric 矩阵。令 \(A=I+\sqrt{\widetilde D}K\sqrt{\widetilde D}\)、\(c_D=\sqrt{\widetilde D}\mathbf1\)、\(r_\alpha=L^\top v\)、\(r_b=\mathbf1_H^\top v\)、\(s=c_D^\top A^{-1}c_D\)，完整伴随可从 SPD solves 得到：

\[
\lambda_b={c_D^\top A^{-1}\sqrt{\widetilde D}r_\alpha-r_b\over s},
\quad y=A^{-1}(\sqrt{\widetilde D}r_\alpha-c_D\lambda_b),
\]
\[
\widetilde D\lambda_\alpha=\sqrt{\widetilde D}y,
\qquad \lambda_\alpha=r_\alpha-K\sqrt{\widetilde D}y-\mathbf1\lambda_b.
\tag{28a}
\]

最后一个式子避免除以很小的 \(\sqrt{\widetilde D}\)。它与式 (23) 的 primal SPD/free-b Schur solve 等价，可以复用已分解的 A。必须完整保存截距 RHS、真实 solves 与 residual；不能为了规避数值困难加截距正则或 epsilon floor。

## 7. 全类 RMS CE、新头与几何的完整联合链

在所有 inner folds 聚合每类 CE sums/counts。令该类均值为 \(\ell_c\)，物理 held 数为 \(n_c\)，注册类数 \(C=c_o+q\)，

\[
\mathcal R=\sqrt{C^{-1}\sum_c\ell_c^2},
\qquad \rho_i={\ell_{y_i}\over C\mathcal R n_{y_i}}.
\tag{29}
\]

\(\mathcal R>0\) 时，统一 score 上游是 \(G_i=\rho_i(P_i-e_{y_i})\)。由于式 (3) 已归一化，CE 也可稳定拆成

\[
\mathrm{CE}_i=\begin{cases}
\operatorname{softplus}(-g_i)-\ell^B_{y_i},&y_i\in\mathcal O,\\
\operatorname{softplus}(g_i)-\ell^N_{y_i},&y_i\in\mathcal N.
\end{cases}
\tag{30}
\]

gate 的 held 上游为 \(v_i=\rho_i(\sigma(g_i)-r_i)\)。新头直接 held 上游仅在新标签行非零：\(F^H_i=\rho_i(p_{N,i}-e_{y_i})\)；旧标签行直接新头上游为零，但其 gate 与约束链并不为零。B 的条件上游可记录，C 中冻结，不更新。

将式 (19) 或 (28) 的 RHS 上游传到新头的 old train logits，再与 \(F^H\) 合并。对式 (8) 的解析头解

\[
\begin{bmatrix}I+K_N&\mathbf1\\\mathbf1^\top&0\end{bmatrix}
\begin{bmatrix}T_N\\t_N\end{bmatrix}
=\begin{bmatrix}L_{AN}^\top F^A\\\mathbf1_A^\top F^A\end{bmatrix}.
\]

其完整伴随为

\[
\boxed{\bar K_N=-\operatorname{sym}(T_N\alpha_N^\top),
\quad \bar L_{AN}=F^A\alpha_N^\top.}
\tag{31}
\]

\(t_N\) 与完整 free-intercept RHS 保留。\(a_{ij}=d_i-\ell^B_{y_i}+\ell^N_j\) 的 \(\bar d_i=\sum_j\bar a_{ij}\)、\(\bar\ell^B_{y_i}=-\sum_j\bar a_{ij}\) 可以作为独立 oracle 输入记录，但真实 C 中 \(\mathrm dd=\mathrm d f_B=0\)。不能把它们重新接入 B 重训，也不能反过来声称 \(a\) 的全部导数为零。

将 gate 的全 \(K_{TT}\)、\(L_{HT}\) 上游，以及新头的 \(K_{NN}\)、\(K_{ON}\)、\(L_{HN}\) 上游累加到同一 raw kernel。\(K_{ON}\) 是全 train Gram 的块，不是独立的新核。全 Gram Frobenius 约定下，独立 cross-block 上游应以两侧各一半 scatter 到对称矩阵；不能再重复乘 2。

正 \(\tau\) 的 kernel 元素链为

\[
\bar d_U=-(\bar k\odot k_U)/(2\tau).
\tag{32}
\]

保留 train Gram 两个端点、held-to-train 两个端点；old-old、old-new、new-new 与 held-to-old 块都存在。随后通过实际 interaction distance 和 norm-preserving residual adapter VJP 累加 \(\bar U\)，最后 \(\bar Z=\bar U W\)。冻结的是 B 函数和旧尺度，**不是** 当前 C 几何中的 old endpoint。新头均值、gate free b、RHS 的所有 new 列也不能 detach。

Outer 只优化式 (29) 的 CE，没有 \(+Z\)、keep loss、guard 或新 penalty。沿现行固定 4 次更新、每次最多 12 个 trial、初始步长 0.125、逐次减半；先投影到式 (1) 的硬球，以实际投影位移做既有 Armijo 接受，保留最后 accepted 状态。Barrier 属于显式内层 head 定义，不作为 outer regularizer，也不根据 held CE 调 \(\zeta\)。任何失败不能触发查询评分、选 checkpoint、换方法或重跑健康实验。

## 8. 必须覆盖的边界与数值状态

| 边界 | 明确行为与限制 |
|---|---|
| new0 | 在任何新头或 gate 前直接返回同一实际 B 对象；不重拟合、不改 log-score 语义、不新增 stage。 |
| new_count=1 | 新类 centered target 恒为零，完整解析头 \(\alpha_N=b_N=0\)，\(\ell^N=0\)。仍定义该头与 free constant，gate 才承担旧新竞争。 |
| new_count=2/5/10/20 | 统一式 (7)、(11)、(21)，无 class-count 特例。hard 归约是 \(m\) 条；all-pair barrier 是 \(mq\) 条且 gap 归一法则明确。 |
| K1 / no-held | 不更新 adapter，不虚构 RMS CE、held supervision 或梯度；final 仍用完整合法 support 拟合新 ridge 与选定 gate。 |
| rank0 | \(Z\) 无有效更新坐标，保留实际 \(U_B\)；仍计算真实 kernel 与完整头。rank0 不等于零 kernel。 |
| tau0 | 使用现行 original complete-input exact equivalence PSD kernel，保留物理 loss 与全部约束。没有 Gaussian 式 (32) 的连续梯度，adapter 不更新；不能给 tau 加小常数。 |
| tau=None / gamma=None / gamma=0 | 明确定义 raw kernel 为零，不估计新尺度。新头是完整自由常数，gate 也有非零 free b；不返回 residual=0。 |
| 只有一组 train | free b 的无约束 Bernoulli optimum 可能在无穷处，不满足上述有限解定理。new0 按首行跳过；new>0 且 old 或 new inner-train 空缺属于输入错误，不能伪造 group 行。 |
| PSD 奇异核、重复输入 | 函数唯一性和 \(I+\sqrt D K\sqrt D\) 表示成立。所有物理 loss 权重保留；hard 冗余 active 普通 Jacobian 不自动成立，all-pair 正 barrier 不依赖它。 |
| hard max ties / active 变更 / 非严格互补 | 不提供不存在的单一普通 Jacobian；完整方向/critical-cone solver 尚未由本文实现。不得 pinv、任选 subgradient、固定 active、零梯度或切换无约束头。 |
| 极端 logits | 稳定 softplus、log-softmax、label-aware sigmoid residual。\(D_i=e^{-|t_i|}/(1+e^{-|t_i|})^2\)，避免用已舍入为 1 的 \(p(1-p)\)。若真实 curvature/slack 不能表达，保留失败而不是 epsilon floor。 |
| RMSCE=0 / 数值 underflow | RMS norm 在零处没有上述普通公式。先区分精确零与浮点 loss 消失，记录缺失或不支持，不能虚构 CE 梯度。 |
| factor/condition/line-search/resource 失败 | 保留最后 accepted U/Z、失败 head 与所有已完成 folds、K/L、RHS、free b、slack、multipliers/weights、残差和已耗成本。停止所属调用；不换来源或重启其他 run。 |

零 kernel 的 hard gate 可独立精确化简为

\[
b^*=\max\{\log(m/p),\max_i a_i\}.
\tag{33}
\]

新头常数 \(b_c=\bar Y\)，所有物理 Bernoulli loss 仍保留。若两个最大项并列，作为一般 \(a\) 输入的映射可能非光滑；在 \(\gamma=0\) 的固定几何分支，所有头对 U 独立，因此 U 梯度为零是结构事实，不是遇困难后的 fallback。零核 all-pair barrier 则是一维严格凸函数

\[
p\sigma(b)-m\sigma(-b)-\zeta\sum_{ij}(b-a_{ij})^{-1}=0,
\quad b>\max_{ij}a_{ij}.
\tag{34}
\]

左边严格递增，从负无穷变到正数 \(p\)，存在唯一根；可以 bracket+有界 Newton 并认证同一 free-b 方程，不给 b 加正则。tau0、rank0 没有连续 adapter 信息时，final 头依旧完整求解，而非空头。

## 9. 与 26 类 MarginQP 的结构成本比较

仅作为 shape 分析，旧 6 类、新 20 类、每类 20 条时，\(m=120,p=400,n=520,C=26\)。这些是结构示例，不是当前数据或任何已运行设备的测量。

| 工作 | 现有 26 类 MarginQP | 组分解候选 |
|---|---|---|
| 解析/优化输出 | \(nC\) residual 系数、\(C\) free 常数 | frozen B + \(pq\) 新 ridge 系数、\(q\) free 常数 + \(n\) gate 系数、1 个 free b |
| 旧 margin 条件 | \(m(C-1)=3000\) 个 pair | hard 归约 \(m=120\) 个；正 barrier \(mq=2400\) 个 scalar slack |
| 约束二阶表示 | 可能是稠密 pair Hessian/active factor | hard active 维度至多独立 augmented evaluation rank；barrier 只把每旧行 pair 权重求和进入 \(n\) 维 diagonal |
| 主要 head factor | affine \(n\) SPD + 约束系统 | 新 ridge \(p\) SPD；gate 每个 Newton 步更新一个 \(n\) SPD/free-b Schur |
| 梯度 | 全 \(C\) RHS + active 伴随 | 新 ridge \(q\) RHS + gate scalar 伴随；新头在 old train 的 RHS 链仍需要计算 |
| 训练与推理 | 真实 QP 和全类展开 | gate 非线性 Newton 并非一次 ridge；推理 B、新头、gate 全部参与 |

Barrier 不需要构造 \(mq\times mq\) Hessian，也不应保存没有必要的 Kronecker；所有 pair slack/权重扫描仍收费。新 ridge 与 gate 共享合法 raw geometry，可复用同一次 U 的距离/kernel，但不能复用改变 U 后的数值。每个 rejected outer trial、所有 gate Newton、line search、factor、RHS、prior 准备和 final fit 都计费。

保留实际可训练 adapter 参数、analytic 系数/free constants、训练与逐样本推理耗时、peak RSS/GPU、驻留状态、实际部署与新增传输字节，注明硬件与 dtype。参数减少、约束更少或 RHS 更少都不能推出设备实测收益。尚未测量的项目记 N/A；不能从 shape 表猜秒数、显存或星载可用性。

## 10. 实现判断、直接验证与资料依据

原 hard 结构定义正确、可行且有唯一函数，但**“所有合法输入都返回单一精确普通梯度”不成立**。普通 active KKT 推导不能充当通用退化 solver。若沿 hard 实施，至少需要完整 directional/critical-cone 与可验证非光滑 outer 更新，不能重复逐案例 UnsupportedJacobian 后宣称结构问题已经解决。

已确认的独立候选 D92-GroupBarrierJointLocalRidge-v1 使用式 (21) 的 all-pair fixed positive barrier，明确登记 finite barrier 目标、式 (25) 的 \(\varepsilon_{\rm bar}=10^{-4}\) 法则与 gap 限制。该方案消除了 max 和 active-set 的结构非光滑，严格保留旧 train margin，但不等于原 hard optimum，也没有普适 float64 稳定性或性能保证。**当前不能声称 solver 已稳定实现**：本文没有数值运行，尚无独立 solver/VJP 证据。这些缺项应在合成、query-blind correctness 范围解决，不由目标评分调参解决。

直接正确性验证应覆盖：独立 finite-feature primal 对照；完整 gate KKT/free b/original gap 与 Fenchel dual 可行性；奇异与零核；物理重复权重；正、零、负 \(d\)；全部新 count；hard 的方向边界；barrier 的全部 ties 列与 RHS 导数；新 ridge+gate 的完整 U/Z 两端方向差分；漏 free b、detach RHS/old endpoint 的错误反例；同 B 继承、inner-held 隔离；new0 原对象、K1/no-held/rank0/tau0；极端数值失败保留与真实资源账。数学验证不代替冻结后独立 truth-last 测试，也不授权新实验。

源码对应仅用于结构核对：[现有 Margin 联合实现](../code/cvsrffi/d92_margin_joint_local_ridge.py)、[Affine 解析 head](../code/cvsrffi/d92_affine_joint_local_ridge.py)、[BranchLocalRidge](../code/cvsrffi/d92_branch_local_ridge.py)、[现有 Margin 结构推导](D92_MARGIN_JOINT_STRUCTURAL_DERIVATION_20261001.md)。本文不引用它们的真实成绩。

通用凸性、Slater、KKT 与 log-barrier duality gap 依据 [Boyd–Vandenberghe, Convex Optimization, §§5.2、5.5、11.2](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf)；KKT 层微分背景见 [Amos–Kolter, OptNet, §3](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf)。式 (1) 至 (34) 是此项目指定结构的推导，资料没有证明本项目准确率或设备收益。BiC 的 [原论文 §5](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf) 说明旧新输出偏差可以单独校准；其 exemplar/validation 权限、两参数 affine 形式与实验结果不移用到此项目。这里仅使用当前合法目标 train support 拟合 gate。
