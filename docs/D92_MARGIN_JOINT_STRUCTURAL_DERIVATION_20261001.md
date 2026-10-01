# 旧类最小 margin 约束的联合 residual affine ridge 推导

状态：**数学推导，尚未实现；独立小矩阵数值证书已通过**。本文只讨论一条候选，不修改已冻结 Conditional、入口、配置或预算；没有读取真实 summary、cache、权重、trace、历史结果或评分。对照仅限 [前序设计的数学章节](D92_POST_AFFINE_JOINT_MATH_DESIGN_20261001.md)，不使用其中运行记录。当前用户授权的合法 support SFT、固定 source-only Phase1 和逐样本全注册类推理边界保持不变。

结论是：此候选没有固有的不可行性缺陷。它将 Conditional 的全列旧点等式保护放宽为同一实际 B 的最小真实类 margin 下界，允许旧分数变化；固定核下仍是有唯一 primal 函数解的凸问题。但一般需要求解带非负约束的 QP，dual 可能退化，梯度只能在明确的正则区域按普通隐式微分计算。本文没有证明准确率、独立 query 风险、用户理想目标或星载成本改善。

## 1. 唯一候选及输入边界

一个 C head 的合法训练集合为 \(T=O\cup N\)，其中旧类训练点数为 \(m\ge1\)，新增类训练点数为 \(p\ge1\)，\(n=m+p\)。注册类数为 \(C\ge2\)。\(H\) 表示当前 inner-held 集合，不属于 head 的拟合或 margin 约束集合。Final head 使用对应路径全部合法 train；不得把独立 outer-held 或 query 加入 \(O\)。

冻结当次同 run、row、parent、fold 的实际 B：

\[
M(x)=\operatorname{pad}_{\mathrm{old}\to\mathrm{all}} f_B(x).
\tag{1}
\]

新增列补零，旧列保持实际 B 的函数、参数和列映射。\(M\) 使用冻结的实际 \(U_B\)；C 的当前 \(U\) 从 \(U_B\) 起步，但改变 \(U\) 不改变 \(M\)。Inner prior 仍只能用该次 old inner-train 在实际 \(U_B\) 下重新拟合，不能复用包含 inner-held 的完整 B head，也不能跨路径取 B。

每行训练标签使用同一 \(C\) 列 one-hot 减 \(1/C\)：\(Y_{ic}=\mathbf1[y_i=c]-1/C\)。令 \(R=Y_T-M_T\)。对每个旧训练点定义

\[
d_i=\min_{j\ne y_i}\{M_{i,y_i}-M_{i,j}\},\qquad i\in O,
\tag{2}
\]

其中最小值面对**全部当前注册列**，包含补零新列。保留 \(d_i\) 原值，不 clamp 到零，不加入新的 margin、keep 系数、温度或类专属规则。\(y_i\) 仅来自合法 old train。C 更新期间 \(M,d,Y\) 固定；计算约束不读取 held 标签，held 标签仅按既有外层 RMS CE 训练 adapter。

核 \(k_U\) 沿用现有 raw Gaussian/equivalence 核和冻结的原 τ、γ、尺度，解析 ridge 系数固定为 1。残差 \(r_c=g_c+b_c\)，\(g_c\in\mathcal H_{k_U}\)，\(b_c\) 是不惩罚自由截距。唯一候选为

\[
\begin{aligned}
\min_{g,b}\quad &J_U(g,b)
=\tfrac12\|M_T+g(T)+\mathbf1_n b-Y_T\|_F^2
+\tfrac12\sum_{c=1}^C\|g_c\|_{\mathcal H_{k_U}}^2,\\
\text{s.t.}\quad &M_{i,y_i}+r_{i,y_i}-M_{i,j}-r_{i,j}\ge d_i,
\quad i\in O,\ j\ne y_i.
\end{aligned}
\tag{3}
\]

旧、新 support 都在同一个 SSE 中，没有把旧误差改成额外 keep loss。外层仍是现有 CE-only、小 adapter、硬球 \(\|Z\|_F\le1/2\)、4 次更新与每次至多 12 次试探；本文不更改这些数值，也不为 QP 发明未执行的迭代预算。

## 2. 可行性、保护能力与严格边界

### 2.1 零 residual 及 Conditional 子集

由式 (2)，任意旧点与任意竞争列都有 \(M_{i,y_i}-M_{i,j}\ge d_i\)。因此 \(g=0,b=0\) 对每个当前 \(U\) 始终满足式 (3)，即使 B 错误、\(d_i<0\)、核为零或存在重复输入，也不会破坏这项可行性证明。

设同一核、prior、标签及 λ=1 下的 Conditional 可行集合为

\[
\mathcal F_{\mathrm{eq}}=\{(g,b):g(O)+\mathbf1_m b=0\},
\quad \mathcal F_{\mathrm{margin}}=\{(g,b):\text{式 (3) 全部约束成立}\}.
\]

在 \(\mathcal F_{\mathrm{eq}}\) 上 \(f(O)=M_O\)，故

\[
\mathcal F_{\mathrm{eq}}\subseteq\mathcal F_{\mathrm{margin}},
\qquad \min_{\mathcal F_{\mathrm{margin}}}J_U\le
\min_{\mathcal F_{\mathrm{eq}}}J_U.
\tag{4}
\]

这只是**同一固定 \(U\) 的正则化 head 目标**比较，不是 SSE、CE、准确率或不同外层优化路径的单调比较。一般会释放分数方向，但不能在退化核或等效判决空间中保证严格增加有效自由度。比如所有列加同一常数不会改变 margin，却也不增加分类能力；若另看零 class-sum 子空间，这个方向本来就消失。

### 2.2 旧 train 的真实类最小 margin

任意可行解满足

\[
\min_{j\ne y_i}(f_{i,y_i}-f_{i,j})\ge d_i.
\tag{5}
\]

- \(d_i>0\)：真实类仍严格胜过所有注册列，旧锚点决策得到保护；无需维持原分数。
- \(d_i=0\)：真实类至少并列最大，不能保证按既有列序 argmax 仍选真实类。不得把非严格不等式写成严格正确率定理。
- \(d_i<0\)：最差真实类 gap 不低于原最差 gap，但真实类仍可能输，竞争赢家也可能改变。原先某一对较大的 margin 可以下降到 \(d_i\)；本约束不是逐对保留原 margin。

与等式方案不同，B 的错误不被原分数锁死。例如在训练 Gram 正定时，可插值一个旧点真实类严格胜出的分数表，令每点所有 true-vs-other 差值大于 \(\max(d_i,0)\)，所以修复错误的函数方向属于可行空间。但式 (3) 的 ridge 最优解是否选择该方向仍由整个目标决定，不保证修复。重复输入、共享特征和正则代价还可能限制同时修复多个错误。

Padding 后的全注册 margin 不能用旧列内部 margin 替代。新增列存在时，式 (2) 同时受 \(M_{i,y_i}-0\) 约束；只有在额外满足零 class sum 且旧列严格正确等条件时，才能从旧列胜出推出 padding 后严格胜出。

若数值解的最大 primal 违反量为 \(\varepsilon_{\mathrm{feas}}\)，实际只能宣称 \(m_{f,y_i}\ge d_i-\varepsilon_{\mathrm{feas}}\)。\(d_i>\varepsilon_{\mathrm{feas}}\) 才足以推出严格正确；必须记录真实残差，不能把理论精确可行写成实测零。

## 3. 稀疏 pair-difference 算子与唯一 primal

按列堆叠 \(\operatorname{vec}(F)\in\mathbb R^{nC}\)。对每个 \((i,j)\)，\(i\in O,j\ne y_i\)，定义一行差分算子

\[
\mathsf D\in\mathbb R^{q\times nC},\quad q=m(C-1),
\qquad (\mathsf D\operatorname{vec}F)_{(i,j)}=F_{i,y_i}-F_{i,j}.
\tag{6}
\]

每行仅两个非零：\(+1,-1\)。令 \(\delta_{(i,j)}=d_i\)，\(h=\delta-\mathsf D\operatorname{vec}M_T\le0\)。约束为 \(\mathsf D\operatorname{vec}r_T\ge h\)。算子只覆盖旧 train 行，但 class 轴是全部注册列，不能仅保留当前最大竞争者而未经证明丢弃其余约束。

为证明性质，任选训练核的有限 feature 表示 \(K=\Phi\Phi^\top\succeq0\)，\(\Phi\in\mathbb R^{n\times t}\)。代表函数 \(g(T)=\Phi W\)，定义

\[
X=[\Phi,\mathbf1_n],\quad \Theta=\begin{bmatrix}W\\b\end{bmatrix},
\quad H_0=X^\top X+\operatorname{diag}(I_t,0).
\tag{7}
\]

对任意非零 \((v,c)\)，

\[
\begin{bmatrix}v\\c\end{bmatrix}^{\!\top}H_0
\begin{bmatrix}v\\c\end{bmatrix}
=\|\Phi v+\mathbf1_n c\|_2^2+\|v\|_2^2>0.
\tag{8}
\]

因为 \(v\ne0\) 时第二项为正，\(v=0,c\ne0\) 时 \(n>0\) 使第一项为正。因此即使 \(K\) 奇异或为零，primal 的 \(W,b\) 仍严格凸且 coercive，可行闭多面体上存在唯一最优解。RKHS 中任何与训练 representer 张成空间正交的成分只增加范数，最优时为零，故完整 \(g,b\) 函数也唯一。训练 kernel 系数则可能不唯一，不能将其与函数唯一性混淆。

此处 feature 分解只用于证明和未来独立 primal oracle，不要求生产代码为每个 head 新做 SVD。凸 QP、KKT 和对偶的一般背景见 [Boyd–Vandenberghe，§4.4、§5.5、§10.1](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf)。以下项目特定公式由式 (3) 推出，并非该书关于本项目效果的结论。

## 4. 自由截距消元与非负对偶

### 4.1 Affine 响应算子

记 \(K=k_U(T,T)\succeq0\)，\(L=k_U(H,T)\)，并定义

\[
A=I_n+K,\quad z=A^{-1}\mathbf1_n,\quad s=\mathbf1_n^\top z>0,
\quad J=A^{-1}-\frac{zz^\top}{s},\quad P=I_n-J.
\tag{9}
\]

这里 \(A\) 是 \(I+K\)，**不是** Conditional 投影中的旧 raw Gram。\(A\succeq I\)；不需要对 \(K\) 求逆或添加 jitter。实现应用已有分解求解，公式中的 inverse 不是显式构造逆矩阵的要求。\(J\mathbf1=0\)，\(P\mathbf1=\mathbf1\)，\(P\succeq0\)。有限 feature 形式给出 \(P=XH_0^{-1}X^\top\)。

取不等式乘子 \(\mu\in\mathbb R^q_{\ge0}\)，采用 Lagrangian

\[
\mathcal L=J_U(g,b)-\mu^\top(\mathsf D\operatorname{vec}r_T-h),
\quad V=\operatorname{unvec}_{n\times C}(\mathsf D^\top\mu),\quad E=R+V.
\tag{10}
\]

将线性项配方，只改变 SSE 的 target 为 \(E\)。其最优函数系数与截距为

\[
\begin{bmatrix}A&\mathbf1\\\mathbf1^\top&0\end{bmatrix}
\begin{bmatrix}\alpha\\b\end{bmatrix}
=\begin{bmatrix}E\\0\end{bmatrix},
\qquad \alpha=JE,\quad b=\frac{z^\top E}{s}.
\tag{11}
\]

因此

\[
r_T=K\alpha+\mathbf1 b=PE,\qquad
f_H=M_H+L\alpha+\mathbf1_h b.
\tag{12}
\]

不能先使用无约束 \(R\) 算完截距后将其固定：式 (11) 中 \(V\) 会改变 \(b\)。

### 4.2 Dual 与完整 KKT

定义无约束全分数 \(F_0=M_T+PR\)，初始 slack

\[
s_0=\mathsf D\operatorname{vec}F_0-\delta,
\qquad Q=\mathsf D(I_C\otimes P)\mathsf D^\top\succeq0.
\tag{13}
\]

消去 \(g,b\) 后，原问题等价于

\[
\min_{\mu\ge0}\quad \tfrac12\mu^\top Q\mu+s_0^\top\mu.
\tag{14}
\]

其必要充分 KKT 条件是

\[
\mu\ge0,\quad s_{\mathrm{mar}}=s_0+Q\mu\ge0,
\quad \mu\odot s_{\mathrm{mar}}=0,
\tag{15}
\]

再与式 (11) 的 stationarity 和 \(\mathbf1^\top\alpha=0\) 联合。若无约束解已可行，\(\mu=0\) 是解；这是一种有条件的快捷路径，不意味着任意 head 都不需 QP。

式 (14) 的正负号可从一维 sanity check 确认：某个无约束 margin 低于下界时 \(s_0<0\)，非负乘子应推动该 true-vs-other 差值增加，而不是继续压低。

令 \(J_0\) 为无约束 affine ridge 的最优目标，则 Lagrange dual 原值为 \(J_0-s_0^\top\mu-\tfrac12\mu^\top Q\mu\)。在 stationarity 精确成立时，primal-dual gap 为 \(\mu^\top s_{\mathrm{mar}}\)。未来需要同时保存 primal 可行性、dual 非负性、互补残差、stationarity、截距方程和 gap，不能只看 QP solver 的 success 字符串。

由于 primal 是有解的严格凸二次函数加有限线性约束，多面体 normal cone 提供 KKT 乘子与强对偶；这里不需要假称存在严格可行点。零 residual 通常至少有一个 margin 紧约束，重复矛盾标签还可能使整个可行空间没有严格内部。

### 4.3 Class sum 与置换

每行 \(V\) 的 class sum 为零，因为差分算子一正一负。若 \(Y_T\mathbf1_C=M_T\mathbf1_C=0\)，则式 (11) 给出 \(\alpha\mathbf1_C=b\mathbf1_C=0\)，所有预测的 class sum 仍为零（held prior 同样需零和）。这不表示 \(\mathbf1_n^\top V=0\)，也不表示后文的截距上游为零。

同时置换全部 class 列、标签、旧类映射与差分行时，目标和约束形式不变，分数等变；同时置换训练物理行也只置换表示。严格 argmax 的预测随类映射一致；存在 ties 时，只能保证 argmax 集合等变，不能忽略既有列序 tie-break。

## 5. Active-set 求解与完整隐式梯度

### 5.1 适用区域

设最优 active 集为 \(\mathcal I\)，其大小为 \(a\)。在该局部区域假设 active 约束独立、\(Q_{\mathcal I\mathcal I}\succ0\)，并满足 strict complementarity：active 乘子严格为正、inactive slack 严格为正。于是

\[
\mu_{\mathcal I}=-Q_{\mathcal I\mathcal I}^{-1}s_{0,\mathcal I},
\qquad \mu_{\mathcal I^c}=0.
\tag{16}
\]

必须再检查式 (15) 的全部约束；式 (16) 只给出已知正确 active 集时的线性解，不提供免费获得 active 集的方法。未来求解器的增删约束、检查 inactive slack 和重分解都需要计费。

有限 feature 的 primal 向量记作 \(x=\operatorname{vec}\Theta\)，\(H_C=I_C\otimes H_0\)，\(A_{\mathcal I}=\mathsf D_{\mathcal I}(I_C\otimes X)\)，\(q_x=\operatorname{vec}(X^\top R)\)。对应 active KKT 为

\[
\begin{bmatrix}H_C&-A_{\mathcal I}^\top\\A_{\mathcal I}&0\end{bmatrix}
\begin{bmatrix}x\\\mu_{\mathcal I}\end{bmatrix}
=\begin{bmatrix}q_x\\h_{\mathcal I}\end{bmatrix}.
\tag{17}
\]

它的微分 RHS 为

\[
\begin{bmatrix}
\mathrm dq_x-(\mathrm dH_C)x+(\mathrm dA_{\mathcal I})^\top\mu_{\mathcal I}\\
\mathrm dh_{\mathcal I}-(\mathrm dA_{\mathcal I})x
\end{bmatrix}.
\tag{18}
\]

式 (18) 显示不能 detach 当前核、active 约束的响应或训练 feature。此类通过 KKT 微分的通用方法见 [Amos–Kolter，OptNet，§3，式 (3)–(8)](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf)。其论文的计算实验和效率结论不能移用为本项目已有效或几乎无反向成本。

### 5.2 不物化 feature 的 kernel 伴随

下面给出同一隐式梯度的低秩 kernel 形式，覆盖全部 registered class 列。令 \(G=\partial\ell/\partial f_H\in\mathbb R^{|H|\times C}\)。先解

\[
\begin{bmatrix}A&\mathbf1\\\mathbf1^\top&0\end{bmatrix}
\begin{bmatrix}T_G\\t_G\end{bmatrix}
=\begin{bmatrix}L^\top G\\\mathbf1_h^\top G\end{bmatrix}.
\tag{19}
\]

用已有 \(A\) 分解求 RHS \(B_G=L^\top G\)，有

\[
t_G=(z^\top B_G-\mathbf1_h^\top G)/s,\qquad
T_G=A^{-1}B_G-zt_G.
\tag{20}
\]

然后只在正确 active 子空间上解

\[
\eta=Q_{\mathcal I\mathcal I}^{-1}
\mathsf D_{\mathcal I}\operatorname{vec}T_G,
\quad B_\eta=\operatorname{unvec}(\mathsf D_{\mathcal I}^\top\eta),
\quad W_\eta=JB_\eta.
\tag{21}
\]

固定 \(M,Y,d,\mathsf D\) 时，完整 VJP 为

\[
\boxed{\overline L=G\alpha^\top,\qquad
\overline K=-\operatorname{sym}\bigl[(T_G+W_\eta)\alpha^\top\bigr].}
\tag{22}
\]

这里 \(\operatorname{sym}(B)=(B+B^\top)/2\)，矩阵内积采用全矩阵 Frobenius 约定。式 (22) 中 \(W_\eta\) 是 margin 约束随当前 kernel 改变产生的完整伴随；将最优乘子视为常数，只留下 \(-\operatorname{sym}(T_G\alpha^\top)\)，不是该 constrained head 的梯度。

一个直接验证途径是对式 (11) 微分：

\[
\mathrm d\alpha=J(\mathrm dR+\mathrm dV-(\mathrm dK)\alpha),
\qquad \mathrm dr_T=P(\mathrm dR+\mathrm dV)+J(\mathrm dK)\alpha.
\tag{23}
\]

在 active 约束上，

\[
Q_{\mathcal I\mathcal I}\mathrm d\mu_{\mathcal I}
=\mathrm d\delta_{\mathcal I}
-\mathsf D_{\mathcal I}\operatorname{vec}
\bigl[\mathrm dM_T+P\mathrm dR+J(\mathrm dK)\alpha\bigr].
\tag{24}
\]

将式 (24) 代入式 (19) 给出的 loss 微分，即得式 (22)。此推导没有漏掉自由截距的导数，也没有将 \(P,J,z,s\) 视作常数。

为便于未来独立 oracle，若暂把 \(R,M_T,\delta,M_H\) 当作独立连续输入，附加伴随为

\[
\overline R=T_G-PB_\eta,\quad
\overline{M_T}\big|_{R\text{ 独立}}=-B_\eta,\quad
\overline\delta_{\mathcal I}=\eta,\quad
\overline{M_H}=G.
\tag{25}
\]

若进一步使用 \(R=Y-M_T\)，则 \(\overline Y=T_G-PB_\eta\)，且不含 δ 链的 \(\overline{M_T}=-(T_G+W_\eta)\)。若实际还对 B 求导，必须继续通过式 (2) 的 min，唯一最小竞争者时按该对分数回传，最小者 ties 时是另一处非光滑边界。本候选明确冻结 B 与 \(d\)，因此 C 更新中这些 prior/阈值输入的微分为零，不能偷偷把跨阶段 B 重训练梯度混入。

没有 active 约束时，\(\eta,B_\eta,W_\eta\) 为空或零，式 (22) 自然退化为完整普通 affine ridge 伴随。

### 5.3 外层 RMS CE 与几何链

现有外层跨全部 inner-fold 汇总每类 CE 后计算 RMS。若该类均值为 \(\ell_c\)、该类 held 数为 \(n_c\)、\(\mathcal R=\sqrt{C^{-1}\sum_c\ell_c^2}>0\)，则 held 行上游为

\[
G_i=\frac{\ell_{y_i}}{C\mathcal R n_{y_i}}
\bigl[\operatorname{softmax}(f_i)-e_{y_i}\bigr].
\tag{26}
\]

每行 \(G_i\) 的 class sum 为零，**不**意味着 \(\mathbf1_h^\top G=0\)。必须保留式 (19) 的完整 \(g_b\)，不能借 CE 的平移不变性删去它。Outer 梯度为 CE-only，不加 \(+Z\)。

对正 τ 原核 \(k=\gamma\exp[-(d_0+d_U)/(2\tau)]\)，每个 kernel 元素的当前平方距离伴随为

\[
\overline{d_U}=-(\overline k\odot k)/(2\tau).
\tag{27}
\]

将式 (22) 的 train Gram 全部 old-old、old-new、new-new 块，以及 held-to-all-train 块传给同一实际几何 VJP；旧端点不能 detach。对称块只按所选全矩阵内积处理一次，不额外重复乘 2。继承 \(U=U_B+ZW^\top\) 时，\(\overline Z=\overline U W\)。

## 6. 退化、非唯一 dual 与导数边界

### 6.1 PSD Gram 不等于 SPD dual

任意 PSD \(K\) 都使 \(A=I+K\succ0\)，因此式 (9)–(14) 仍成立。若 \(K\succ0\)，则 \(P\succ0\)；\(\mathsf D\) 的每个旧点块有 \(C-1\) 个独立行，因此 \(Q\succ0\)，dual 唯一。退化 \(K\) 则可能令 \(P,Q\) 奇异。

最直接的例子是 \(K=0\)：\(J=I-\mathbf1\mathbf1^\top/n\)、\(P=\mathbf1\mathbf1^\top/n\)。Head 只剩唯一的受约束常数 \(b\)，而 \(q=m(C-1)\) 个不等式在至多 \(C-1\) 个 class-difference 方向上作用；dual 一般可能不唯一。不能将 \(K+I\) 可 Cholesky 当成 active dual 也必然可 Cholesky 的证明。

不同 dual 解只要满足完整 KKT，会给出相同 primal 函数；其 kernel 系数可能相差训练 Gram 的零空间向量。在来自同一 PSD kernel 的合法 cross block 上，它们仍表示同一函数。将不一致的任意 cross matrix 接入，则没有这项保证。

重复 physical rows 不能因为 kernel 行相同就删除 SSE 或 margin 条件。若以后做无损压缩，必须保留每个物理观测的损失权重与全部对应标签约束；不同标签、不同 frozen prior 或下界尤其不能静默合并。不得用 nugget/jitter、软化 margin、clamp \(d\) 或近似去重悄悄改变式 (3)。

### 6.2 Active 切换与奇异 KKT

式 (19)–(25) 的普通 Jacobian 要求第 5.1 节的局部正则性。即使 primal 唯一，也可能出现 active 乘子为零、inactive slack 到零、冗余 active 行或退化 dual；这时不能直接对奇异 \(Q_{\mathcal I\mathcal I}\) 使用一个随意伪逆并声称得到了唯一真实梯度。

在固定正则 active 区域内，状态随平滑 kernel 参数平滑；穿越边界时通常只有分段或方向导数。Active 集发生切换时，对任意选中的一侧公式不能承诺等于中央差分。对于退化边界，需要明确 rank-aware 的 primal/critical-cone 灵敏度处理，或在实现范围中明确报告该点无法提供普通 Jacobian。本文不选择新的 smoothing、性能相关阈值或启发式梯度。

因此“所有 PSD Gram 上存在唯一 primal head”与“所有点可用同一 SPD active-set 反向”是不同命题。后者不成立。完整方法未来必须处理这个边界，不能把出现概率小当作证明，也不能静默退回 Conditional 或普通无约束 head。

Zero-bandwidth 的等价核可在固定合法分组内使用相同 QP；若等价关系本身随参数变化发生离散切换，连续 Gaussian 的式 (27) 不适用。必须依据真实 τ0 几何定义另行确认可微域，不能通过给 τ 加小常数代替。

## 7. Query、自由截距与 new0 边界

有限旧 train 上的 margin 约束不保护 unseen query 的 margin、类别比例或风险。Query 仍逐样本面对全部注册列，只输出固定分数与原 tie-break argmax；不能在推理时读取真实类、旧新 role 或使用本约束筛选竞争列。

对一个仅用于数学论证的真实类 \(y\)，令 \(\mathcal N_g^2=\sum_c\|g_c\|_{\mathcal H_k}^2=\operatorname{tr}(\alpha^\top K\alpha)\)。由再生性质，

\[
m_{f,y}(x)\ge m_{M,y}(x)
-\max_{j\ne y}|b_y-b_j|-\sqrt{2k(x,x)}\,\mathcal N_g.
\tag{28}
\]

右端严格为正是一个充分条件，不是训练时允许读取 query 标签的理由。也可从同真实类旧锚点 \(o\) 出发，用 \(k(x,x)+k(o,o)-2k(x,o)\) 控制 \(g(x)-g(o)\)，此时常数截距相消，但仍需控制 frozen prior 的成对分数变化。没有这些条件，不能从式 (5) 推出邻域或 query 保护。

与 Conditional 的 \(r(O)=0\) 不同，这里 residual 在旧点不为零，不能沿用其 \(\sigma_\perp(O)=0\) 的界。若 raw Gaussian cross kernel 对全部训练点趋近零，\(r(x)\to b\)，一般不是零。单独控制 RKHS 范数不控制未惩罚 \(b\) 的各类差值；有限 support 的可行性也不能消除远处常数扰动。

**new0 不能由本公式自动得到 B 的精确复用。**即使没有新类，重新解式 (3) 仍可能通过 residual ridge 改变旧分数。若未来上层继续保持现行 new0 语义，应明确在没有新增类时不进入 C/QP，精确返回当次 B 对象；这属于边界定义，不是把一个不同目标的解伪称成恒等。K1/rank0/zero kernel 同样不能推断 head 无需计算；真实 K1 依然没有独立 held 指标，外层没有合法 held 时不得虚构 CE 训练证据。

## 8. 实际计算账及必要实现前事项

稀疏 \(\mathsf D\) 本身只需约 \(2q\) 个非零，但 dual Hessian 一般是稠密的。对两约束 \(a=(i,j),b=(\ell,k)\)，

\[
Q_{ab}=P_{i\ell}(e_{y_i}-e_j)^\top(e_{y_\ell}-e_k).
\tag{29}
\]

因此可从旧点块 \(P_{OO}\) 构造 Q，而无需物化 \(nC\times nC\) Kronecker 矩阵。获得 \(P_{OO}\) 仍需真实求解：例如对 \(A\) 解 \(m\) 个旧点选择 RHS，连同 \(z\) 和 target RHS 记账。这个结构只节省表示，不消除 QP。

| 工作 | 必须记录的实际量 |
|---|---|
| 几何 | 当前 train/train 与 held/train 的距离、kernel 次数和测得耗时；不能复用已变化 \(U\) 的旧数值。 |
| Affine 基础 | \(n\times n\) 的 \(A\) 分解次数；\(z\)、target、旧选择列及乘子响应的每次 solve。 |
| QP 前向 | 实际迭代、active 增删、全 slack 检查、分解/更新次数、实际 active size、失败与最后可行状态；不同 head 的次数不能用上界替代。 |
| 反向 | 式 (19) 一组 \(C\)-RHS 的 \(A\) solve；式 (21) active 系统一列 RHS；\(JB_\eta\) 又一组 \(C\)-RHS 的 \(A\) solve。已有分解复用仍有 solve 成本。无 active 约束时后两项可按实际省去。 |
| RHS 代理 | 每个真实 triangular call 分开累计 \(r\)、\(nr\)、\(n^2r\)；active 系统用其真实维度 \(a\)。若用 LDL/QR/迭代法，另按真实调用描述，不能套旧“两次 Cholesky”模板。 |
| 驻留与部署 | 训练中的 K、Q/active 因子、μ/slack、α/b、prior B、feature/adapter buffers 分开计字节；部署可只保留必要 α/b 展开和实际 B，但实际序列化前不能声称最小包已实现。 |
| 外层预算 | 每次 inner head、被拒 trial、最终 head 和 prior 准备都计费；固定 4×12 外层预算不限制每个 QP 求解器的内部工作。 |

例如仅作 shape 演示，若一个合法 train head 有每类 20 条、旧 6 类、新 20 类，则 \(n=520,m=120,C=26,q=3000\)。完整 float64 Q 需要 \(8q^2=72{,}000{,}000\) B，仅其 dense Cholesky 领先工作项即 \(q^3/3\approx9\times10^9\)。这不是当前任何已运行路径的实测，也不表示所有 active 系统都达到此维度；它说明“参数少”不能推出 QP 免费或星载省算力。Matrix-free Q 乘法可降低显式存储，但每次仍要差分 gather/scatter 与 \(P\) 的真实 solve，需另计迭代和误差。

未来实现前需要明确的直接正确性事项只有以下几类：

1. 定义可行 QP 的确定求解、停止残差尺度和技术失败保留方式，尤其是重复约束、奇异 dual、active 切换和 τ0。数值容差应来自机器精度、矩阵规模、条件与 RHS/残差尺度，不能由成绩选取；不得静默改变目标。
2. 用独立 finite-feature equality/inequality-constrained primal oracle 核对目标与完整 KKT，而非让同一 dual 实现互证；同时验证正 margin、负 margin、ties、全列竞争、类/行置换和物理重复权重。
3. 在正则 active 区域做 train/held 两端完整方向差分，覆盖非零 \(g_b\)，并构造 detach μ、漏 \(W_\eta\)、固定 b 的错误反例；在切换点验证方向导数边界，不要求不存在的单一中央 Jacobian。
4. 将现有 same-stage B、inner prior、new0 精确复用、K1 无 held 和完整实际资源账接入时保持明确。它们尚未由本文实现；本文不授予配置、发布或 launch，也不添加性能硬门槛。

**P0 判断：**未发现使式 (3) 本身不可行、非凸或无法定义唯一 primal 函数的不可消除缺陷。若后续实现声称任意退化输入都有唯一光滑 dual 梯度、把固定 active 解当成无需求 QP、遗漏自由截距伴随、用 query/inner-held 定义保护 margin，或将 new0 重解伪称为 B 精确复用，则会是具体正确性问题。当前这些边界已在数学层面明确；独立小矩阵数值证书已通过，尚无端到端实现或性能改善证据。

## 9. 独立数值证书及当前交付边界

独立query-blind符号审阅未发现P0/P1；随后主Agent在项目ssr-gpu环境串行执行[合成数学证书](../tests/test_d92_margin_joint_math_certificate.py)，19 passed，0.42 s，完整输出prefix为 `E:/type10-7/.codex_tmp/pytest_utf8_1790831747662706800`。测试不import生产候选、encoder或数据，也不读取任何实际实验产物。

证书采用独立有限feature primal KKT枚举与非负dual枚举核对自由截距、原目标、全saddle/strong-duality gap、零residual可行及Conditional同目标子集。覆盖正/零/负margin和所有注册竞争列、singular K的函数唯一而系数非唯一、zero kernel的dual非唯一，并以正则1/2 active小矩阵完整方向差分核对对称train K及合法held L、g_b、乘子依赖、R/M/δ/M_H和Y−M链；漏截距或detach乘子的负对照确实不能通过正确差分。切换点与singular active普通Jacobian明确拒绝。

这些证书不覆盖[工作集求解路线](D92_MARGIN_JOINT_SOLVER_FEASIBILITY_20261001.md)的生产实现、去循环/误差传播、完整U几何链或实际资源；new0只证明重解不是B复用，未测试不存在的生产router。没有新配置、实验或参数搜索，未证明旧query遗忘、新类性能或用户三个理想目标改善。仍需依据直接正确性要求完成求解组件与完整方法，不能以数学证书代替真实性能评估。
