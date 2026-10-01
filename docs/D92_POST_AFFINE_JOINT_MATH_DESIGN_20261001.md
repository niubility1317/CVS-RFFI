# D92：Affine 之后的联合 LocalRidge 数学设计

状态：`DESIGN_DOCUMENT / PARTIAL_IMPLEMENTATION / NOT_FROZEN_FOR_EXPERIMENT`。日期：2026-10-01。

本文推荐一条后续结构路线：**B 直接优化训练用 RMS CE；C 在解析 LocalRidge 头内部限制完整残差，使其在当前旧 support 上恒为零。** C 的约束覆盖全部注册列，包括新增类列。adapter 仍可更新，保护恒等式对每次当前 U 都成立。该路线保留解析头、实际 B 函数继承、固定旧尺度和现有有限更新预算，不引入 keep 权重、GEM 半空间、类别校准系数或参数网格。

这是一份供后续预登记和实现审查使用的设计。主任务后续已完成[正核解析头与完整伴随](D92_CONDITIONAL_AFFINE_KERNEL_IMPLEMENTATION_20261001.md)的 20 项独立 KKT/方向差分合成验证；联合 SFT 上层和完整实验尚未完成，不能把局部验证当成端到端验证。它不修改正在运行的 Affine，不宣称准确率提高或 query 零遗忘。下文的“精确”均指数学上的精确；浮点实现需要独立残差和容差核验。

## 1. 依据、权限与要解决的结构问题

允许的证据是当前数学文档、纯源码、一手文献，以及明确授权的 [AJLR training-only findings](D92_AJLR_TRAINING_FINDINGS_20261001.md)。未读取 support_summary、query/outer 评分、原始 snapshot/IQ/权重、全局实验索引、handoff 或其他真实结果。本次只读取该 findings 文档，没有打开真实派生流或其他真实产物，也没有选择训练最高步骤。

training-only findings 给出两个与结构有关的事实：C 的冻结 B prior 到 Z=0 注册头已经改变旧类的全注册竞争；adapter 随后的更新不能自动消除这一变化。另有 accepted step 使 RMS CE 上升、总目标下降，说明 `RMSCE + .5||Z||²` 的下降保证不等于 RMS CE 下降保证。这些是训练目标的诊断，不是泛化证据；inner held 的标签参与 optimizer，不能称为独立验证集。本文只据此定位需要改变的方程，不用训练正确率挑 checkpoint、预算或系数。

固定 Phase1 和合规 source-only 基座。B 只用当前合法旧 support，C 只用当前合法新旧 support。C 只能继承同一新 run、同一 row、同一物理 fold 的当次实际 B 状态。final C prior 是当次 full-support B 的真实函数；inner C prior 沿用当前实现的构造：固定实际 U_B，在该 inner fold 的旧 inner-train 上重新求解析旧类头，并冻结其 train/held 值。不得从其他 run 加载目标适应 head/adapter，不因 balanced B 的理论等价复用已结束状态。

每个 query 仍逐单样本对全部已注册类统一 argmax。训练、推理不使用 query truth、query role、类别数量、配额、全局重排或 query 回流。约束中的 O 是合法 train support；inner held 不能被加入该 fold 的头约束。final 的 O 才是当前全部旧 support。

## 2. 与已实现路线的具体差别

| 路线 | 实际控制对象 | 不能直接推出的性质 |
| --- | --- | --- |
| AJLR/Affine 的白化近端 | 当前 support 字典上的平均预切向位移；目标含 `.5||Z||²` | 全注册类 margin 不下降；旧 support 上完整 C 函数等于 B |
| AJLR/Affine 的实际 B prior | 完整 C 函数中的固定加法分量 `M=pad(f_B)` | 新 residual/intercept 不改变旧分数或新增类竞争 |
| 现有 MCResidual8 | teacher-deficit 的聚合 keep loss、方向半空间、trial keep 上限 | 每个旧点的所有注册列相等；有限步后的逐点 margin 恒等式 |
| 本设计 | 每次解析 C 头的函数空间：`r(O)=0`，覆盖所有 C 列 | O 之外的全局保护；新增类必能学好；B 错误会被修正 |

当前位置：Affine `_solve_affine_head` / `_affine_adjoint` 在 [核心第 186 行](E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt/code/cvsrffi/d92_affine_joint_local_ridge.py:186)；C prior 绑定在 `prepare_affine_joint_training` 第 455 行及其 `make`；RMSCE 与近端在 `evaluate_affine_joint_objective` 第 578 行；有限预算在 `fit_affine_joint_local_ridge` 第 723 行。白化由 `d92_function_coordinate_residual8_local_ridge.py:89` 的 `latent_coordinates` 实现。MC 的 `_hinge`、`evaluate_mc_objective`、`guarded_direction`、`_trial_acceptance` 分别在 `d92_margin_constrained_residual8_local_ridge.py:345,355,432,487`。

GEM 原论文把旧任务 loss 限制转换为基于局部线性近似的梯度内积约束，见 §3、式 (6)–(8)。本文约束作用于每次重新求解的完整分类函数，既不是把 keep 梯度投影到半空间，也不是放宽该类约束的系数。代价是它禁止旧锚点上的正向迁移，保护范围也依赖旧 support 的覆盖。[Lopez-Paz & Ranzato, 2017](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf)

## 3. 保留的几何与 B 的目标

### 3.1 函数坐标和固定尺度

沿用 736 维分支向量、固定 DCT8、精确 GELU、κ=1/4、切向范数保持 adapter，以及完整 interaction 的距离。记原始和适配后的 interaction 几何为 φ₀(x)、φ_U(x)。不物化 123616 维 φ；使用现有精确分支距离引擎。

对每个 stage 的合法 support 字典 H，沿用保留秩 r≤8 的白化 W：

\[
W^\top H^\top HW/n=I_r,\qquad U=U_{\rm anchor}+ZW^\top.
\]

因此

\[
\frac1n\sum_i\|(U-U_{\rm anchor})h_i\|_2^2=\|Z\|_F^2.
\]

这是预切向位移的经验平均，不是分类 margin 距离，也不是 CE Fisher 白化。C 的 H/W 由当前新旧 support 准备，不能把其球半径直接解释为旧类每点位移上限。未覆盖的参数零空间保持 anchor 中的实际分量。

旧原始 R0 给出的 τ、γ、s₀ 保持原定义和物理 old reference 集合；不随 U 重新调尺度。正 τ 的未中心化核为

\[
k_U(x,z)=\gamma\exp\left[-\frac{\tfrac12\|\phi_0(x)-\phi_0(z)\|^2+
\tfrac12\|\phi_U(x)-\phi_U(z)\|^2}{\tau}\right]. \tag{1}
\]

两部分距离和的 Gaussian 核是 PSD。Affine 的自由常数使固定核下的 q 中心化只改变表示 gauge；τ/γ 仍由旧参考统计量决定，不能删除或偷换。C 的新解析式使用式 (1) 的 raw gauge，B 可保留现有 q-centered Affine 表示；此等价要求同一原始核、U、τ/γ、Y 和 prior，不能拿跨 run 状态代换。当前 cache 中用于中心化的 `expm1(−d/τ)` 不能直接当成投影 A：A 必须是 γ 缩放的真正 Gaussian Gram，而不是减过常数 1 的非 PSD 表示。

### 3.2 B：解析 Affine 头，外层只下降 RMS CE

B 的 head 不做旧锚点约束。对一个 fold 的旧 inner-train，记当前 centered Gram 为 K，one-hot 每行减 `1/C_B` 得 Y，M=0。保留

\[
A_B=K+I,\quad F=A_B^{-1}Y,\quad z_B=A_B^{-1}{\bf1},\quad s_B={\bf1}^\top z_B,
\]
\[
b_B={\bf1}^\top F/s_B,\qquad \alpha_B=F-z_Bb_B,
\qquad f_B(H)=K_{HT}\alpha_B+{\bf1}b_B. \tag{2}
\]

训练样本数均衡且 q 为该旧 train 的均匀测度时，`K1=0`、`1ᵀY=0`，解析 b_B=0。它与现有无截距 B 的头等价，但不承诺位级一致。非均衡训练、非均匀 q 或非零 prior 不享有该等价。

把各 fold held 的 CE 按类累加，再求类平均 ℓ_c 和 RMS：

\[
\ell_c(Z)=\frac1{n_c}\sum_{f,i\in H_f:y_i=c}
\bigl[\log\sum_j e^{f_{f,j}(x_i;Z)}-f_{f,c}(x_i;Z)\bigr],
\quad \mathcal R(Z)=\sqrt{C^{-1}\sum_c\ell_c(Z)^2}. \tag{3}
\]

B 与 C 都推荐外层

\[
\min_{\|Z\|_F\le1/2}\mathcal R(Z),\qquad Z_0=0. \tag{4}
\]

不再加入 `.5||Z||²`；保留白化坐标，并把幅度限制写为硬预算。半径 1/2 来自当前 4 次更新、每次归一化方向最大步长 1/8 的确定上界，未从训练峰值或外层成绩选择。在下述 4 次预算内此球通常是冗余的显式证书：路径长度已经≤1/2。它不增加可训练方向或隐藏第 5 次更新。

目的仅是消除“近端下降而 CE 上升”的接受理由。接受步保证训练用 RMS CE 在数值容差内不升；它不能保证 0–1 正确率、独立 query 风险或用户理想目标达成。移除近端也会增加在有限 support 上拟合噪声的风险，硬位移预算不能消除这种风险。

## 4. C：在解析头中保护完整旧类函数

### 4.1 精确约束问题

一个 C fold 中，合法 train 为 `T=O∪N`：O 是旧 inner-train，N 是新增类 inner-train；final 中 O/N 是当前全部旧/新 support。记 m=|O|、p=|N|、h=|H|、C=C_old+C_new。所有列均用相同 one-hot−1/C 标签 Y。冻结

\[
M(x)=\operatorname{pad}_{\rm old\to all} f_B(x)
\]

到当次实际 B 路径。新列初值为零。当前 U 从实际 U_B 起步，但 M 的 feature mapping 和解析头都保持实际 B 状态，不随 C 的 U 变化。

令 `r=g+b`，每一列的 g 属于式 (1) 的 RKHS，b 是不惩罚的自由常数。对每个当前 U，求

\[
\min_{g,b}\frac12\|M_T+g(T)+{\bf1}b-Y_T\|_F^2+
\frac12\sum_{c=1}^C\|g_c\|_{\mathcal H_{k_U}}^2,
\quad g(O)+{\bf1}_m b=0. \tag{5}
\]

约束的是**完整 residual**，不能只限制 old 列、g 或训练 adapter 的瞬时梯度。旧 train 的误差项变成常数，但没有把旧类从 outer RMS CE 移除。O 之外的旧 inner held 仍通过式 (3) 的 CE 对 U 施加训练梯度。

式 (5) 的可行集合始终包含 `g=b=0`。它不要求 B 在旧 support 上正确；B 错误同样被保留。有限表达由半参数 representer theorem 支持，自由常数不应被误当作 RKHS 范数的一部分。[Schölkopf, Herbrich & Smola, 2001, Theorem 2](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)

### 4.2 条件 Affine 核与解析解

先讨论正 τ 且旧 raw Gram 正定的情况。令

\[
A=k_U(O,O),\quad B=k_U(O,N),\quad D=k_U(N,N),
\quad F=k_U(H,O),\quad E=k_U(H,N),
\]
\[
z=A^{-1}{\bf1}_m,\quad s={\bf1}_m^\top z>0,\quad J=A^{-1}B,
\quad c_N={\bf1}_p-B^\top z,\quad c_H={\bf1}_h-Fz.
\]

受约束函数空间的核为

\[
k_\perp(x,t)=k_U(x,t)-k_U(x,O)A^{-1}k_U(O,t)
+\frac{[1-k_U(x,O)z][1-z^\top k_U(O,t)]}{s}. \tag{6}
\]

它给出

\[
K_\perp=D-B^\top J+c_Nc_N^\top/s,
\qquad L_\perp=E-FJ+c_Hc_N^\top/s. \tag{7}
\]

式 (6) 的第一部分是 PSD Schur 补，最后一项是 PSD rank-one 项；`k_perp(O,x)=0`。最后一项来自自由常数的消元，不能省略为“简单条件 Gaussian 核”，也不能在求完式 (7) 后再加一个自由截距，否则破坏式 (5)。这与显式常数基函数的无信息先验极限相符；只用有限 Schur 式，不通过“极大常数方差”模拟自由截距。[Rasmussen & Williams, GPML §2.7, 式 (2.41)–(2.42)](https://gaussianprocess.org/gpml/chapters/RW2.pdf)

记 `R_N=Y_N−M_N`。式 (5) 等价于

\[
\alpha=(K_\perp+I_p)^{-1}R_N,
\quad f_C(H)=M_H+L_\perp\alpha. \tag{8}
\]

不再次减 residual 样本均值，不更改 ridge 系数 1，不新增 optimizer head 参数。C 需要的自由常数已经在条件核内部解析消元。

可部署为普通 raw-kernel 展开。设

\[
v=c_N^\top\alpha/s,\qquad \beta=J\alpha+zv,
\quad r(x)=k_U(x,N)\alpha-k_U(x,O)\beta+v. \tag{9}
\]

因为 `Aβ=Bα+1v`，故对任意当前 U：

\[
f_C(O)=M_O=\operatorname{pad}f_B(O). \tag{10}
\]

又有 `1ᵀβ=1ᵀα`，展开的 raw RKHS 系数总和为零，解释了固定 q 的 affine gauge 等价：中心化只在预测中产生可由 v 吸收的常数，并不改变该零和展开的 RKHS 二次范数。若实际 B 和标签每行 class sum 为零，则 α、β、v 以及 f_C 也保持 class sum 为零。数值实现仍须记录实际 class-sum 残差，不能根据恒等式省略审计。

## 5. 保护范围直接连接到最终全注册 margin

对已知理论标签 y 的旧点 x，定义 actual B padding 后的全注册 margin：

\[
m_{\rm padB,y}(x)=\min_{j\ne y}[M_y(x)-M_j(x)]
=\min\{m_{B,\rm old,y}(x),\ f_{B,y}(x)\}\quad(C_{\rm new}>0). \tag{11}
\]

新增列为零，因此只看旧类之间的 B margin 不足够。数学上，若 B 的类分数和为零且 y 在旧类中严格胜出，则 `f_B,y>0`，padding 后也严格胜出。旧类数为 1、零分数或 ties 不享有这一严格结论。式 (10) 因而保护旧锚点上所有原本严格正确的 B 决策，同时保留 B 的错误；它不把 B 的全部 accuracy 转化为一个无条件不变定理。

条件函数空间的范数为

\[
\mathcal N_r^2=\sum_c\|r_c\|_\perp^2
=\operatorname{tr}(\alpha^\top K_\perp\alpha),\qquad
\sigma_\perp^2(x)=k_\perp(x,x)\ge0.
\]

由再生性质和 Cauchy–Schwarz，

\[
\|r(x)\|_2\le\sigma_\perp(x)\mathcal N_r,
\quad m_{C,y}(x)\ge m_{\rm padB,y}(x)
-\sqrt{2}\,\sigma_\perp(x)\mathcal N_r. \tag{12}
\]

右端>0 是最终全注册 argmax 保持的充分条件，不是必要条件。O 上 `σ_perp=0`；在条件核意义下接近 O 的点可能有较小上界，远处不能保证。该界在当前 U 的核上成立，已经包含 adapter 改变后的 residual；它不是只在 Z=0 成立的一阶结论。还有一个明确失效边界：若 x 与全部 O/N 的 Gaussian cross kernel 趋近零，式 (9) 的 r(x) 趋近 v，而不是零；式 (6) 的对角趋近 `γ+1/s`。保留自由常数后，不能把条件核名称误读成远处无扰动的保证。

式 (12) 不要求训练时访问任何 query。未来可在冻结模型后的独立数学审计中检验合成点；不得据真实 query margin 选择 U、预算或 checkpoint。推理不按该界过滤样本或分 old/new role，仍返回同一完整分数和统一 argmax。

计算机实现的 `r(O)` 有误差 ε_O 时，锚点 margin 保护改为 `m_C≥m_padB−sqrt(2)||r(O)||₂`，应保存并验证真实最大残差。不能把理论零写成实际测得的零，或将失败残差静默截断。

## 6. 完整隐式梯度：两个系统，一路 CE

### 6.1 outer CE 与条件 ridge 伴随

式 (3) 的 held 行 i、类别 y 的上游梯度为

\[
G_i=\frac{\ell_y}{C\mathcal R n_y}
\bigl[\operatorname{softmax}(f_i)-e_y\bigr],\quad \mathcal R>0. \tag{13}
\]

类 sums/counts 跨全部 fold 合并后才能形成权重。温度保持 1。risk=0 时按当前极限/零梯度约定处理，不新增各 fold 平均替代式。

一个 fold 中令

\[
T=(K_\perp+I_p)^{-1}L_\perp^\top G,
\quad \overline K_\perp=-\operatorname{sym}(T\alpha^\top),
\quad \overline L_\perp=G\alpha^\top, \tag{14}
\]

其中 `sym(X)=(X+Xᵀ)/2`。M_N/M_H 固定，所以 `dR_N=dM_H=0`。式 (14) 使用一组完整 C-RHS adjoint，不能只对新类列回传。

### 6.2 条件 Affine 核的完整伴随

用下面的 saddle 记号推导，不直接对不定矩阵做 Cholesky：

\[
\mathscr H=\begin{bmatrix}A&{\bf1}_m\\{\bf1}_m^\top&0\end{bmatrix},
\quad V_N=\begin{bmatrix}B\\{\bf1}_p^\top\end{bmatrix},
\quad V_H=\begin{bmatrix}F^\top\\{\bf1}_h^\top\end{bmatrix},
\quad X=\mathscr H^{-1}V_N.
\]

式 (7) 可写为 `K_perp=D−V_NᵀX`、`L_perp=E−V_HᵀX`。任何 RHS `(b,c)` 的 saddle 解由 A 的已有 Cholesky 完成：

\[
\nu=(z^\top b-c)/s,\qquad x=A^{-1}b-z\nu. \tag{15}
\]

因此 X 的前 m 行是 `J+z c_Nᵀ/s`，最后一行是 `−c_Nᵀ/s`。

为避免 h 列 RHS，定义

\[
\Lambda=\mathscr H^{-1}(V_HG),\qquad X_\alpha=X\alpha,
\qquad X_T=XT.
\]

完整反向为

\[
\overline D=-\operatorname{sym}(T\alpha^\top),\qquad
\overline E=G\alpha^\top,
\]
\[
\overline V_N=(X_T-\Lambda)\alpha^\top+X_\alpha T^\top,
\quad \overline V_H=-X_\alpha G^\top,
\quad \overline{\mathscr H}=\operatorname{sym}[(\Lambda-X_T)X_\alpha^\top]. \tag{16}
\]

取 `barH` 的前 m×m 块给 A；取 `barV_N` 前 m 行给 B；`barV_H` 前 m 行转置给 F。其余行/列的常数 1 与 0 不求导。所有 old/current train/held kernel 端点都有梯度，特别是 A、B、F 的旧参考端点；不能把投影矩阵、J、z、s 或条件核当作固定缓存来 detach。

式 (16) 是一般矩阵伴随的精确低秩写法，利用 `barK/barL` 的 α、T、G 因子。Λ 只需要 A 上的 C-RHS SPD solve，不是 h-RHS solve，也不需要新的 CE/keep loss 通道。它的 saddle RHS 最后一行为 `1_hᵀG=g_b`，必须保留；逐行 CE 梯度的 class sum 为零不表示这个逐列总和为零。前向、反向的 triangular 调用和 RHS 宽度都必须分别计费。

正 τ 时对 raw kernel 的上游量 bar k 使用

\[
\overline d=-(\overline k\odot k)/\tau,
\quad d=\tfrac12d_0+\tfrac12d_U,
\quad \overline{d_U}=\tfrac12\overline d. \tag{17}
\]

将 A/B/D/F/E 各块上游按同一矩阵内积约定合并，调用现有完整距离/adapter VJP；对称训练块需一致处理两端和计数，不能把非对角项再次乘 2。最后 `g_Z=g_U W`，不再加 Z。B 的伴随继续使用现有 `_affine_adjoint`，包括完整 `g_b=sum_rows(G)`；C 的自由常数已在式 (16) 的 saddle 常数边界中传递。仅因 CE 对行 class-sum 的平移不敏感，不能推断 `g_b=0`。

## 7. 固定优化预算和失效条件

推荐在未来方法配置中一次性固定下列规则：每 stage 最多 4 次 accepted 更新；每次最多 12 次 Armijo 试探；初始步长 1/8；回溯因子 1/2；Armijo 系数 `1e-4`；方向 `−g/||g||`；stage 起点 Z=0；球半径 1/2。candidate 先投影到球，Armijo 使用实际 `delta_Z` 的内积。接受需要 RMS CE 的 Armijo 与 nonincrease 均通过，数值容差按现有 float64 比较规则预先声明。

保留最后 accepted 状态，不取训练曲线最高正确率或最低 CE 的旧 checkpoint。12 次失败记录 `TRIAL_BUDGET_EXHAUSTED` 并停止本 stage；零梯度/零可行位移记录独立原因；不反馈重跑，不扩大步数，也不以旧 MC keep 上限二次筛选。推荐沿用获准 training-only findings 的比较矩阵：K=1,5,10,20，旧类 6，新增类数 0,2,5,10,20，inner folds=`min(K,3)`，保留全部 row 和原 seed 角色，不增加参数扫描。具体 capsule/物理 IDs、row 名、seed、不可覆盖新 run_id 与输出路径由 root 后续一次性预登记；不能在结果出现后选择有利 row。

数值和容量边界必须在实现前固定：

| 边界 | 本设计的数学行为与限制 |
| --- | --- |
| 新增类数为 0 | 精确复用当次实际 B 状态，零 C prep/head/adapter fit；不重拟合近似 B |
| K=1 或 dictionary rank=0 | 不更新 adapter，仍求完整 B 解析头和 C 条件头；K=1 不虚构 inner CE 优化。rank=0 只表示 H/W 没有可训练坐标，不表示 raw kernel 为零；τ/γ 有效时仍执行完整核投影和解析头 |
| τ=0 且非零 γ 的等价核 | 沿用当前既定 τ0 前向，合并关系必须是原/适配等价核的整个核输入精确等价；不从标签或近似距离阈值推断组。不更新 U；重复旧等价组的同一零残差约束可解析合并，C 仍可在未被旧组占据的方向求完整头 |
| τ=None / s₀=0 / γ=None 的零核 | g=0；存在旧锚点时约束强制 b=0，所以 C 只能输出 M。这不同于当前无约束 Affine 零核的 `b=mean(Y−M)`，应明确记容量不足，不能偷偷借截距学习新类 |
| 正 τ，旧 raw Gram 病态 | 式 (5) 仍可行，A 的无 ridge 投影求解可能数值不可解；记录技术失败。不得在 A 加 jitter/nugget、重估带宽或改为软约束来伪称仍精确保护 |
| 精确重复旧点 | 仅对可证明在整个当前核中等价的约束做无损压缩，并保留全部物理 IDs。完全相同的原始五块是充分条件；近似相同不能按容差删掉约束 |
| 新点与受保护旧点核输入完全重合 | 该方向的条件核为零，r 必为零，新标签无法覆盖旧 B 决策；这是不可兼得的容量冲突，不是调大学习率可解决的问题 |
| B 旧点错误或 tie | C 的硬保护会保留错误；ties 仍按统一注册类 argmax 规则，无类别置换无关的唯一正确保证 |

正 τ 的 Gaussian 核对不同完整核输入在实数算术下严格正定；计算机上的 A 不一定能保留这个性质。实现应保存无损 raw A、分解、实际 solve/constraint residual 和条件核对称/PSD 残差，容差根据既定算术误差口径固定，不能根据模型效果调整。`expm1` 可用于稳定记录径向差值，不能把严重病态 A 的不可分辨特征变成新可训练方向。

此路线牺牲旧锚点上的正向迁移以获得函数恒等式。条件核容量在旧/new support 很接近时可能过小；旧 support 无法覆盖旧 query 时，式 (12) 也可能不给正界。强保护不是低遗忘和新类接近旧类的无条件同时保证。

## 8. 星载计算账：必须按真实矩阵与 RHS 报告

adapter 梯度坐标仍为 `736r`，r≤8；解析 α、β、v 与 B 的 α_B/b_B 单列为闭式 head 状态，不把它们算成 optimizer 参数或反过来声称“不需要计算”。不引入源样本、query 训练传输或额外骨干训练。

一个正核 C fold 的主要真实工作：

| 工作 | 尺寸/次数 | 必须记录的语义 |
| --- | --- | --- |
| old 投影分解 | A：m×m，1 次 Cholesky | 新增 `projection_factorization_count`，不能藏成 prior fit |
| old 投影前向 solve | A 上 `[B,1]`，p+1 列，2 次 triangular | RHS columns=`2(p+1)`，elements=`2m(p+1)`，dense work proxy=`2m²(p+1)` |
| 条件核构造 | `BᵀJ`：p×p；`FJ`：h×p | 前向需 `O(mp²+hmp)` 密集乘法与 rank-one 项 |
| residual head | K_perp+I：p×p，1 次 Cholesky；C 列 RHS | 2 次 triangular；columns=`2C`，elements=`2pC`，proxy=`2p²C` |
| CE residual adjoint | p×p 上 C 列 RHS，2 次 triangular | 式 (14)，每个实际 backward 计费 |
| CE projection adjoint | m×m 上 C 列 RHS，2 次 triangular | 式 (15) 的 Λ，不能漏算或声称仅 2 次 adjoint triangular |
| complete raw/distance VJP | A/B/D/F/E 全部端点 | 输出规模 `m²+mp+p²+hm+hp`；式 (16) 可用 C 宽低秩乘法，无 m³ 反向中心化 |
| prior | 当前 B old-only inner head、其 train/held 单样本 score | 沿用实际 B 规则，另列 factor、C_old+1 RHS、距离、核、adapter 和时间 |

上表 dense work 是 `n²×RHS` 的工作代理，不是实际 FLOP、能耗或硬件时间。若进行了数学上无损的重复约束压缩，分解账使用实际代表点数，另保留全部受保护物理点数，不能混淆。B/prior Affine forward 仍为 `(C_old+1)` RHS 的合并 Schur 解；zero-kernel 不虚构分解。失败和 rejected trial 的所有已发生 forward 也计费；缓存复用本身不增加 head/factor 次数。

若 F≤3，固定预算最多有 `1+4×12=49` 个新 objective forward，gradient 复用当前 forward 至多 4 次，最后 full-support head 1 次。单个 informative B 的 student forward/factor 上界为 `49F+1≤148`。单个 C 的 student forward 上界相同，但每个 positive-kernel forward 有两次分解；加至多 F 次旧 inner prior 分解，factor 上界为 `2(49F+1)+F≤299`，未包含其前序 B 的成本。C 每 stage 最多 `4F≤12` 组 CE backward，每组两个 SPD adjoint、4 次 triangular，合计至多 48 次 adjoint triangular。实际 no-information、零梯度、budget exhaustion 或零核应按物理执行计数，不能机械填上界。

矩阵分解的形式主项由现有 n³ 变成 m³+p³，但新增投影构造、old raw Gram 的条件数、额外 adjoint、实际 B 双 mapping 和缓存均有成本。例：当前授权矩阵的最大 full-support 形状是 m=120、p=400、C=26、n=520；这只给出数学工作尺寸，不是省算力证据。训练 dictionary、原/适配几何、完整距离引擎及 Phase1 的固定特征提取成本不因 optimizer 参数少而消失。

部署式 (9) 不需携带 A/J/Cholesky：保存 α(p×C)、β(m×C)、v(C)、原/适配 support 几何、U/W 及当次完整 B prior。仅这些条件 residual 系数的 float64 字节是 `8[(m+p)C+C]`；B prior、geometry、raw blocks、IDs/classes、adapter 和去重后的共享数组另计。训练峰值仍须覆盖两套 factor、完整 forward/adjoint、trial cache 和 prior。不能按全归档文件和同时常驻缓存混淆 peak/resident/deployment。

单 query 先算 actual B，再按当前 C geometry 算 O/N kernel 展开；两种 U 不相同时，真实有两种 adapter mapping 和各自距离/核工作。不能用当前 C mapping 替代 B prior，不能在部署时重新解投影，不能根据 query batch 的类配额改变预测。实际耗时、峰值 RAM/VRAM、硬件、线程、传输字节和计数必须测量；未测量项为 N/A。任何总量都应包含 R0 基线、B、C、prep、prior、被拒试探和独立分析成本，不能只报告 736r。

## 9. 后续实现与必要验证清单

本文件不指定新 schema/method，不改现有 core。若 root 决定预登记实现，应独立模块、入口、summary 和方法配置，继续保留现有 release；本设计不是现有 Affine 的热修补。

训练 archive 至少保留当前 prepared H/W/SVD/anchor、物理 IDs/classes/分组、实际 B lineage 和完整 prior 状态，以及每个 current head 的 A/B/D/F/E、J/z/s/c_N/c_H、K_perp/L_perp、R_N、α/β/v、scores、factor 和真实 τ/γ/s₀。gradient archive 保留 G、T、Λ、X_α/X_T 和完整 kernel/adapter 上游，INITIAL/GRADIENT/TRIAL/STEP/FINAL 的坐标、方向、RMS CE sums/counts、预算与失败原因。summary 应从原/适配几何及 actual B 独立重建式 (5)–(17)，不能只信打印分数或借 core forward 的结果。

真正必要的确定性合成验证是：

1. 用显式有限 feature 的 primal equality-constrained ridge + unpenalized b，以及独立 saddle/KKT oracle 验证式 (6)–(9)；不能以同一条件核函数互验。
2. 对原始/适配几何到完整 Z 的方向差分，分别移动 old/reference、new 和 held 端点；验证式 (16) 的非零自由常数伴随，故意 detach A 或去掉 rank-one 项应被测试识别。
3. 条件核 PSD、r(O)=0、class-sum、q gauge、逐单样本/批量 scores、row/class 置换的一致性；balanced B 与旧头等价以及非均衡不等价。
4. 用严格正确、错误、零 margin/tie 的 B 合成头核验式 (11)–(12)，包括一个远离 O、不能保证保护的反例；不能把 sufficient bound 不通过写成必然错误。
5. new0 精确 object/state 复用；K1/rank0 完整头；τ0 非零等价核及重复组；零核强制 b=0；新旧核输入重合的不可学习方向；病态 A 无 jitter 的显式失败与残差证据。
6. RMS CE-only Armijo 使用实际投影 delta、4×12 固定预算、最后 accepted 而非最高训练 step；所有 rejected trial/prior/projection/adjoint RHS 与 bytes 的实际计数、无损 archive 和独立篡改拒绝。

其中正核解析头、独立 primal KKT、五个原核块及 old/new/held 端点完整伴随、margin 界与远处反例已在上述独立模块通过合成测试；真实分支/adapter 到 Z、顺序状态、退化分支、预算、完整日志和端到端集成仍待验证。实际训练/发布/Git/远端仍由 root 统一负责，局部数值测试通过不能替代完整预登记实验及独立 scorer。该实现另外执行 A、raw Schur 和 K_perp 三次谱诊断，计算账必须包含这些额外立方规模成本。

## 10. 预登记三阶段报告与推荐结论

对同一 row、同一物理旧 support/query，预先固定 A=合法绑定的冻结地面 source-only 分类器、B=旧 support 适应、C=实际 B 继承后全类注册。A 必须核实实际 source-only head、类序、scale 和 raw-feature 契约；当前尚未接入合法 A，故 `A_old` 与 `B_old−A_old` 为 N/A。R0/B0 仍在目标旧 support 上拟合，不能冒充“适应前 A”。

完整报告 K×新增类数，并注明旧类数、新增类数和类别注册规则；分别给 A_old、B_old、C_old、C_new，在相关阶段可用时计算 `B_old−A_old`、`B_old−C_old`、`|C_old−C_new|` 和 H。C 必须全注册类统一竞争，不能用 old-only accuracy 代替；Nnew=0 时新类指标为 N/A。训练 CE 曲线、旧锚点残差、理论界和真实测试 accuracy 分开报告。

用户的适应提升≥10 个百分点、注册后旧类下降≤1 个百分点、新旧绝对差≤3 个百分点继续作为理想目标，不是本文能证明的结果，也不增设发布硬门槛。固定预测后由独立 scorer 连接 truth，禁止其结果回流挑预算、系数、rank、checkpoint 或选择性重跑。

推荐此一条结构，是因为它把旧类保护从平均位移/聚合 keep 转到最终完整函数，仍保留一个可微解析 LocalRidge 头；B 的目标变化也直接对应已授权训练诊断中的 CE/近端冲突。没有推荐软 keep 权重或重新校准旧/新类列，因为它们没有式 (10) 的逐点恒等式，并增加选择系数；没有推荐把 MC/GEM 改名重做。该推荐的代价和边界已经明确：旧锚点错误不能修正，新旧重合容量冲突无法消除，O 之外保护仅为式 (12) 的充分条件，病态 raw old Gram 可能阻止数值实现，计算总量必须实测。它是一项可被后续合成 oracle 和预登记实验否证的设计，不是性能结论。
