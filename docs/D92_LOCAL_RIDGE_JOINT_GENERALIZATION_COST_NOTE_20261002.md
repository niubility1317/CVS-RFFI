# LocalRidge 联合微调的风险关系、一次解析校正与成本边界

日期：2026-10-02。范围：source-blind 数学设计；未实施、未运行数值、未访问实际数据或结果。本文只提出一个候选 `D92-ProtoFrameTangent-GGN1-LocalRidge`，不改变当前 BranchLocalRidge、Margin B、GroupBarrier C、query 或预算。文中数值是源码已有常数、维度和字节公式，不是实验观测。

## 1. 有限结论与依据

1. 完整解析 Ridge 加小 adapter 的支持集 CE，可以构成明确的监督微调目标。一次小维度正定二次子问题可以解析求解，再用有限次真实目标回读接受更新。整个非线性联合 CE 没有因此获得闭式全局最优解。
2. CE 给分类错误的上界，Ridge 解给固定核的平方误差正则目标最优解。两者都不能单独保证最终 query 准确率提升。OOF CE 下降更不能直接证明最后全 support 重拟合的分类器风险下降。
3. 冻结地面原型可只定义固定特征参考几何。本文用六个旧类原型的五个固定差向量生成共享切向字典；可训练量仅由合法 target support 的 CE 拟合。原型不是回归标签、教师输出、额外 support 或持久分类头。
4. 相同核及相同 train/query 交叉核产生相同 LocalRidge 分类器。完整 interaction Hilbert 特征的共同平移、共同正交变换，以及归一化前逐分支统一正缩放，不能产生识别收益。本文的共享切向平移通常改变核，因此不是上述不变变换。
5. 五个微调坐标远少于当前最大 5888 个坐标，但每次真实目标回读仍需完整 Ridge；C 还需新头及 barrier gate。不能由参数少推断训练总计算、耗时或内存更少。

推导依据为 [BranchLocalRidge 源码](../code/cvsrffi/d92_branch_local_ridge.py)、[Margin joint 源码](../code/cvsrffi/d92_margin_joint_local_ridge.py)、[GroupBarrier joint 源码](../code/cvsrffi/d92_group_barrier_joint_local_ridge.py)、[gate 源码](../code/cvsrffi/d92_group_barrier_gate.py)、[Margin 数学稿](D92_MARGIN_JOINT_STRUCTURAL_DERIVATION_20261001.md)和 [Group 数学稿](D92_GROUP_FACTORIZED_JOINT_DERIVATION_20261001.md)。名称含 prototype 的 [PrototypeTransport 源码](../code/cvsrffi/d92_prototype_transport_local_ridge.py)实际使用 target train 类均值，不证明已有地面原型 bundle 满足本文接口。

本候选遵守项目协议 5.3.1/5.3.2：Phase1、encoder、原型、类映射和聚合统计冻结；只使用合法 target support 标签训练 adapter、Ridge 和 gate。原型坐标作为回归或教师目标来拟合持久 adapter 的另一种做法不属于本文主方案，也不由本文授权。已有 support 微调和冻结原型授权不因本候选命名而增加审批。本文不新增 bundle 字段、不登记或启动实验。

## 2. 从训练目标到分类风险：能证明什么

### 2.1 CE、Ridge 与最终风险不是同一个量

对所有注册类的归一化概率 $p_y(x)$，定义 $\ell(x,y)=-\log p_y(x)$。若真实类不是 argmax，则 $p_y\le 1/2$，包括真实类与另一类并列但规范 tie-break 未选择真实类的情形。因此逐样本有

\[
\mathbf1\{\widehat y\ne y\}\le \ell(x,y)/\log2. \tag{1}
\]

类均衡经验风险使用每类平均 CE $\ell_c$，于是

\[
\widehat R_{\rm bal}\le {1\over C\log2}\sum_c\ell_c
\le {F\over\log2},\qquad F=\sqrt{{1\over C}\sum_c\ell_c^2}. \tag{2}
\]

这是上界，不是错误率与 CE 的等式；CE 降低时 argmax 错误可以不变，甚至增加。对总体分布也可对 (1) 取期望，但需要总体 CE，而非仅训练 CE。类均衡目标与未知 query 类先验下的普通风险也不是同一口径，不可利用 query 数量或配额补齐二者。

OOF 中每个 held 样本由不含该样本标签的 inner head 评分。(2) 此时约束的是这些 fold 预测器的经验错误。最后在全部合法 support 上重拟合的 head 是另一个函数；没有额外稳定性论证，OOF 上界不是其训练或 query 风险证书。OOF 减少标签泄漏，不消除小样本方差，也不自动覆盖 receiver、时间或场景的分布差异。

固定 PSD 核的 LocalRidge 精确最小化

\[
\frac12\sum_{i=1}^n\|w(x_i)+b-Y_i\|^2+\frac12\|w\|_{\mathcal H}^2,\quad
Y_i=e_{y_i}-{\mathbf1\over C}. \tag{3}
\]

核特征值 $\lambda_k$ 对应的拟合收缩为 $\lambda_k/(1+\lambda_k)$，有效自由度为 $\sum_k\lambda_k/(1+\lambda_k)$。这说明 Ridge 的正则化机制，不证明其平方误差最优解也最小化 CE 或 0-1 风险。联合 adapter 优化的是由该解析解组成的 CE，不能把 (3) 的凸性转移给 adapter 目标。

### 2.2 固定几何的稳定性与 margin 条件

令 $K_c$ 为中心化核，$A=I+K_c$、$\alpha=A^{-1}Y_c$。对另一个 PSD 核 $K'_c$，有

\[
\alpha'-\alpha=-{A'}^{-1}(K'_c-K_c)\alpha,
\quad \|\alpha'-\alpha\|\le\|K'_c-K_c\|\,\|Y_c\|, \tag{4}
\]

因为两个逆矩阵的谱范数均不超过 1。用完整中心化交叉核 $L_c$ 评分时，

\[
\|L'_c\alpha'-L_c\alpha\|
\le \|L'_c-L_c\|\,\|\alpha\|+
\|L'_c\|\,\|\alpha'-\alpha\|. \tag{5}
\]

所有移动的 train 均值和 query/train 两端都必须包含在 $K_c,L_c$ 中。若某样本原 winner margin 为 $m>2\epsilon$，且每类分数变动至多 $\epsilon$，winner 保持不变。这可证明“小扰动不改变已有大 margin 决策”，不能证明“校正后更正确”。改变 support 成员、自由截距、带宽统计或 trace scale 时，不可只套用固定矩阵 (4)。

若另外假定每类有完整 Hilbert 特征锚点 $\mu_y$，其真实分类 margin 至少 $\delta>0$，所有 score gap 的 Lipschitz 常数不超过 $L$，则 $\|\Phi(x)-\mu_y\|<\delta/L$ 足以保持正确分类；由 Markov 不等式可得 $R\le L^2\mathbb E\|\Phi(X)-\mu_Y\|^2/\delta^2$。这些是额外的总体假设。本文只用 160 维 `z_id` 原型，不能据此得到 FFT、auxiliary 和 interaction 的完整锚点误差或 anchor margin，故不把此条件界当作本候选的泛化保证。

经典稳定性理论针对明确的样本分布、损失和算法稳定性条件。支持集同时改变 kernel、adapter、自由截距和最终 head 的流程，不能直接沿用固定核正则算法的结论。[Bousquet 与 Elisseeff，JMLR 2002](https://jmlr.org/papers/v2/bousquet02a.html)支持稳定性到泛化误差的研究框架，未证明本候选的风险界。

### 2.3 原型和少量矩不能识别任意域变换

有限旧类 support 与源类均值不能唯一确定 target 域的逆变换。即使给出少量方差或半径，也存在同均值、同有限矩、却在未观测区域具有不同类别边界的分布。对有限 support 之外交换两个类的标签即可产生同一观测但不同 query 风险。故必须另外假定域扰动共享、近似位于候选字典且旧、新类遵循相同机制，才能把旧类学习的校正解释为可能迁移的新类域校正；这些假设不能由原型包或小训练损失证明。

## 3. 核不变方向必须分清空间

BranchLocalRidge 的背景与辅助特征为

\[
b=\operatorname{unit}\bigl[\operatorname{unit}(z_{id}),4\operatorname{unit}(FFT)\bigr],\quad
a_{aux}={1\over\sqrt3}[\operatorname{unit}(t),\operatorname{unit}(f),\operatorname{unit}(pa)],
\]

并使用隐式 $\Psi(b,a_{aux})=(b,a_{aux},b\otimes a_{aux})$。维度分别为 256、480，隐式总维度 123616。以下结论指相应空间的全部 train 和单 query 同时变换：

| 变换 | 对核的结论 | 限制 |
|---|---|---|
| 完整 $\Psi$ 的共同平移或共同正交变换 | 所有 pair distance、原 trace、带宽和 raw kernel 相同 | 原始 $b,a_{aux}$ 共同平移一般不等于完整 $\Psi$ 平移，因有 bilinear 项 |
| $b\mapsto O_b b$、$a_{aux}\mapsto O_a a_{aux}$，两者正交 | $\Psi$ 变换为 block 正交矩阵，距离相同 | 必须对所有样本应用同一变换 |
| 归一化前对一整个分支乘同一正数 | unit 后不变 | 160 维所有 `z_id` 特征的统一缩放才是该不变方向 |
| 只对某个 8 维子空间等幅缩放 | 一般改变其相对正交补的角度和 kernel | 不能称为全 160 维统一缩放，也不能据此删掉一个“常数方向” |
| 完整 $\Psi$ 乘 $s>0$，带宽同步乘 $s^2$ | Gaussian radial 部分相同 | 原 trace 和 trace-matched $\gamma$ 变为 $s^2\gamma$，固定 Ridge 系数 1 下 kernel 及解未必相同 |

若最终 train 核、交叉核、中心化规则、$\tau,\gamma$ 均相同，(3) 的 canonical 解和预测相同；Group gate 的输入及 RHS 也全部相同时，注册结果同样相同。kernel 不变的重新参数化可能改变某个有限优化器的路径，但不扩展可达分类函数，不能把该路径差异当作结构收益证明。

本文不删除任何所谓 DCT 常数方向。字典直接由固定原型差构造；某个系数方向是否无信息，取决于其对真实 kernel 的 Jacobian，而非坐标名字。

## 4. 唯一候选的可执行数学定义

### 4.1 固定原型参考几何与共享变换

要求冻结 Phase1 bundle 中已有与当前 encoder、160 维 `z_id` 特征及 canonical 类映射兼容的六个旧类原型。原型的类均值、归一化次序、量化和缺槽语义必须由原有 schema 规定；本文不重估它们。记确定性解码后再单位化的向量为 $p_1,\ldots,p_6$，零向量保持零。以 lexicographic 最后一个旧类为固定参考，定义

\[
Q=[p_1-p_6,\ldots,p_5-p_6]\in\mathbb R^{160\times5}. \tag{6}
\]

不拟合 source covariance，不拟合 source classifier，不对 $Q$ 作 support 驱动的旋转、白化或选择。$Q$ 只是一组固定特征方向；原型不出现在监督 loss 的目标值中。保留全部五列，即使线性相关，也不求伪逆、不借 rank 分支删参数。名义参数维度严格为 5，有效 feature 维度可更小。$Q=0$ 时校正恒等。其他旧类数不是本文默认矩阵；数学上可用 $C_o-1$ 列，但不能静默改变本候选维度。

对任意物理样本，$u=\operatorname{unit}(z_{id})$、$w=Q\theta$、$\theta\in\mathbb R^5$，定义

\[
t=(I-uu^\top)w,\quad
\delta={\kappa t\over\sqrt{\kappa^2+\|t\|^2}},\quad
u_\theta={u+\delta\over\|u+\delta\|},\qquad \kappa=0.25. \tag{7}
\]

若 `z_id` 为零，直接保持零；否则输出 $\|z_{id}\|u_\theta$。其他四个分支不变。因为 $u^\top\delta=0$，分母至少为 1；$\|\delta\|<\kappa$，不需要截断零分母。$\theta=0$ 时输出为原特征，并保留其真实 Jacobian。

变换是每个样本无标签的同一公式。它没有按类或 old/new 路由。对小 $w$，(7) 等于 $\operatorname{unit}(u+w)$ 的一阶切向变化，可表示位于 $\operatorname{col}(Q)$ 的小共享方向偏移。它不能表示任意旋转、频率扰动或 class-specific 域变化。它通常改变角度：在二维中 $u_1=(1,0),u_2=(0,1),w=(\epsilon,0)$，第一点不动，第二点向第一点偏移，二者内积从 0 变正。这是代数例子，不是数值实验。

令 $\Psi_0(x)$ 为原分支特征，$\Psi_\theta(x)$ 为 (7) 后的分支特征。使用原源码的固定 old-train $\tau_0,\gamma_0$：

\[
d_\theta(x,x')={\|\Psi_0(x)-\Psi_0(x')\|^2+\|\Psi_\theta(x)-\Psi_\theta(x')\|^2\over2},\quad
k_\theta=\gamma_0\exp(-d_\theta/\tau_0). \tag{8}
\]

这是拼接 Hilbert 特征的 Gaussian PSD 核。固定几何阻止把每次 bandwidth/trace 变化作为未声明的校准自由度。半原距离只保证 $d_\theta\ge d_0/2$，不保证分类改善。

### 4.2 阶段、真实目标与固定数值

本候选另拟独立 B；它不是把当前实际 Margin B 的 `U` 转换成五维参数，也不重解释当前运行状态。当前 B/C 继续冻结。本候选未来若被实施，须从自己的合法 B 状态顺序继承自己的 C。

固定物理 split 采用当前源码规则：每类物理 ID 排序后，以位置模 $\min(K,3)$ 构造 inner folds。每个 inner head 只读 inner-train 标签。B 使用旧类，C 使用当前全部注册类。每类 held CE 先跨 fold 求均值，再计算 (2) 的 RMS；不使用 query。

| 项目 | 固定定义及来源 |
|---|---|
| B adapter 初值 | $\theta_B^0=0$，恒等特征 |
| C adapter 初值与 anchor | 实际本候选 B 的 $\theta_B$，不重置；优化变量 $\vartheta=\theta_C-\theta_B$ |
| 每阶段接受的更新数 | 最多 1；这是本候选明确的一次更新成本限定，非性能筛选 |
| 约束 | B 为 $\|\theta\|\le0.5$；C 为 $\|\vartheta\|\le0.5$。半径沿用当前 hard-ball 常数，但不存在跨参数化等成本或等效果保证 |
| 二次模型 damping | $I_5$，只确保解析子问题正定；不是监督目标中的 proximal loss |
| 真实监督目标 | RMS 类平均 CE，temperature 1；不加 prototype alignment、教师回归或 proximal 项 |
| 回读策略 | 初始步长 $0.125$，每次乘 $0.5$，至多 12 次，Armijo 系数 $10^{-4}$；均沿用当前 Margin 数值策略 |
| 浮点比较 | 源码比较容差乘数 128；实数接受式与浮点分辨率分别报告，不把容差内相等当成已证明的严格下降 |
| Ridge | 系数 1、每物理样本权重 1、完整自由截距；沿用 LocalRidge |
| gate | 沿用 all-pair 正 barrier 定义，$\zeta=N10^{-4}/(m q)$；其平均原目标 gap 预算来自 Group 源码，不由分数选取 |

B 的目标是旧类 analytic Ridge 的 OOF CE。默认 C 继续更新共享校正参数，最多接受一次 GGN 更新；不存在“冻结 C adapter、只注册”的第二默认路径。C 的旧类条件函数完全冻结在实际本候选 B，使用自己的 $\theta_B$ 而非 C 的 $\theta_C$；内折先在冻结实际 $\theta_B$ 上，仅用该折 old inner-train 重新拟合合法 old prior head；最终使用 actual full B。不得以 full B 中 held 标签构造 inner 教师或 margin。

C 的新条件头只用 new inner-train 的 raw $K_{NN}$ 解 (3)，具有 $q$ 个完整输出和自由截距。对 all legal inner-train support，gate 标签 old 为 1、new 为 0；令原 B padded 新列 0 的真实 margin 为 $d_i$，保留负值、零值和 ties。阈值为

\[
a_{ij}=d_i-\log p_B(y_i|x_i)+\log p_N(j|x_i).
\tag{9}
\]

其中 old 部分可稳定写为 $\operatorname{LSE}(f_B(x_i))-\max(\max_{k\ne y_i}f_{B,k}(x_i),0)$，再加 $\log p_N(j|x_i)$。不能 clamp $d_i$。固定新头时 gate 解

\[
\min_{v,b}\ \frac12\|v\|_{\mathcal H_{k_\theta}}^2+
\sum_{i=1}^N[\operatorname{softplus}(g_i)-t_i g_i]
-\zeta\sum_{i\in old}\sum_{j=1}^q\log(g_i-a_{ij}),\quad g_i=v(x_i)+b. \tag{10}
\]

所有 slack 严格为正，不做 max 折叠。固定 $\theta$ 时它是凸、光滑的 RKHS 问题；常数 $b>\max a_{ij}$ 给 Slater 点。old/new 两组均非空时，logistic 阻止 $b$ 向正负无穷，RKHS 正则和严格的 intercept 曲率给唯一函数及自由截距。PSD 奇异核不阻止 canonical 表示。有限 barrier 的原 hard 最优目标差至多 $m q\zeta=N10^{-4}$，不是“精确 hard 最优解”，也不是实测 dual residual 证书。

统一分数为

\[
s_y=\log\sigma(g)+\log p_B(y),\quad
s_j=\log\sigma(-g)+\log p_N(j). \tag{11}
\]

其指数总和为 1。C 真实 outer 目标仍为全部类 RMS CE，对新头、gate、阈值和两个 kernel 端点完整求导。新 head 的平方误差目标、gate 的 sum logistic 与 outer RMS CE 不相同；不能声称依次 inner 求解已全局最小化 outer CE。

每个单 query 都计算两组，面对全部注册类取 argmax；同分按物理类 ID lexicographic 规则。不得使用真实 role/count、配额、batch 统计或先硬路由组。实现 argmax 时可用 old raw B winner 与两组稳定最大值来保持数学排序，避免共同巨大 log offset 使不同 old float 分数舍入成相同；这不是按真实组路由。

## 5. 完整解析 head、gate 与 GGN 求导

### 5.1 Ridge：自由截距与移动两端

对任意 inner head，记 $A=I+K$、$F=A^{-1}Y$、$z=A^{-1}\mathbf1$、$s=\mathbf1^\top z>0$。用一次 Cholesky 和 RHS `[Y,1]` 求解

\[
b={\mathbf1^\top F\over s},\quad\alpha=F-zb,\quad
h=L\alpha+\mathbf1 b. \tag{12}
\]

$A\succ0$ 不要求 $K$ 可逆；无伪逆、jitter 或漏掉自由常数。即使 Y 的物理 class counts 不平衡，也保留 (12)。对固定标签的任意参数方向，

\[
dF=-A^{-1}(dK)F,\quad dz=-A^{-1}(dK)z,\quad
db={\mathbf1^\top dF-(\mathbf1^\top dz)b\over s},
\]
\[
d\alpha=dF-(dz)b-z(db),\quad dh=(dL)\alpha+L(d\alpha)+\mathbf1(db). \tag{13}
\]

五列方向共用同一 factor，不需要五次 Cholesky。若采用反向 VJP，对 score cotangent $G$，令 $A_\alpha=L^\top G$、$\beta=\mathbf1^\top G-z^\top A_\alpha$，则

\[
\bar F=A_\alpha+\mathbf1\beta/s,\quad
\bar z=-A_\alpha b^\top-\mathbf1\langle\beta,b\rangle/s,
\]
\[
T_F=A^{-1}\bar F,\ T_z=A^{-1}\bar z,\quad
\bar K=-\operatorname{sym}(T_FF^\top+T_z z^\top),\quad
\bar L=G\alpha^\top. \tag{14}
\]

这是完整 free-intercept adjoint。新头用于 old train（阈值）和 held（直接 CE）的 score cotangent 必须合并后应用 (14)；train-new 的 cross block 必须向对称 full train kernel 两端散射。

### 5.2 Barrier gate：不依赖 active set 的光滑隐式导数

设 $g=K\alpha+b\mathbf1$、$s_{ij}=g_i-a_{ij}>0$，则 canonical 方程是

\[
q_i=\sigma(g_i)-t_i-\mathbf1_{old(i)}\sum_j{\zeta\over s_{ij}},\quad
\alpha=-q,\quad \mathbf1^\top\alpha=0,
\]
\[
D_i=\sigma(g_i)\sigma(-g_i)+\mathbf1_{old(i)}\sum_j{\zeta\over s_{ij}^2}>0. \tag{15}
\]

$\zeta$ 的 N、m、q 是固定物理及 registry 元数据，对 adapter 不求导。对任意 adapter 方向，令 $r_i=\mathbf1_{old(i)}\sum_j\zeta\,da_{ij}/s_{ij}^2$，解

\[
(I+DK)d\alpha+D\mathbf1\,db=-D(dK)\alpha+r,
\qquad\mathbf1^\top d\alpha=0. \tag{16}
\]

不反演 $K$。以 $H=I+\sqrt D K\sqrt D\succ0$ 因子求 (16)。具体令 $M=I+DK$、$R=\sqrt D H^{-1}\sqrt D=M^{-1}D$、$c=-D(dK)\alpha+r$，则

\[
db={\mathbf1^\top M^{-1}c\over\mathbf1^\top R\mathbf1},\qquad
d\alpha=M^{-1}c-R\mathbf1\,db,\quad
M^{-1}c=c-\sqrt D H^{-1}\sqrt D Kc. \tag{16a}
\]

自由 intercept Schur $\mathbf1^\top R\mathbf1$ 在有限正常数 $D_i$ 时为正。五方向 RHS 可一起解；不反演 $D$。这里没有 tight/active 集合，也没有 strict-complementarity 条件。相应 eval 导数为

\[
dg_E=(dL)\alpha+L(d\alpha)+\mathbf1\,db. \tag{17}
\]

C 中 old conditional prior 冻结，所以 $da_{ij}=d\log p_N(j|x_i)$。若 $\bar a$ 是阈值 cotangent，新头 old-eval score 的 cotangent 必须包含

\[
\bar h_{ij}=\bar a_{ij}-p_N(j|x_i)\sum_k\bar a_{ik}. \tag{18}
\]

在方向模式中用同一 softmax Jacobian，绝不把 RHS 当常数。由 (11)，

\[
ds_y=(1-\sigma(g))dg,\qquad
ds_j=-\sigma(g)dg+dh_j-\sum_k p_N(k)dh_k. \tag{19}
\]

(13)、(16)、(17)、(19) 给出完整 C score Jacobian。gate、Ridge 的 raw kernel 均不预先中心化；Ridge 的自由截距已等价处理 train 均值，不能另省略移动均值项。

### 5.3 特征及核 Jacobian

在非零 `z_id` 上，$u$ 对 adapter 固定。令 $\sigma_t=\sqrt{\kappa^2+\|t\|^2}$、$v=u+\delta$，有

\[
{\partial\delta\over\partial\theta}={\kappa\over\sigma_t}
\left(I-{tt^\top\over\sigma_t^2}\right)(I-uu^\top)Q,
\quad
{\partial u_\theta\over\partial\theta}={I-u_\theta u_\theta^\top\over\|v\|}
{\partial\delta\over\partial\theta}. \tag{20}
\]

保留零分支规则，并经原 `b` 归一化及 implicit interaction 链式求导。FFT/auxiliary 固定不等于 interaction 的 derivative 为零：$\partial(b\otimes a_{aux})=(\partial b)\otimes a_{aux}$。不显式展开 123616 维。

当 $\tau_0>0$、$\gamma_0>0$，

\[
dk_{ij}=-{k_{ij}\over2\tau_0}\,d\|\Psi_\theta(x_i)-\Psi_\theta(x_j)\|^2,
\]
\[
d\|\Psi_i-\Psi_j\|^2=2(\Psi_i-\Psi_j)^\top(d\Psi_i-d\Psi_j). \tag{21}
\]

train/train 的两个端点，以及 held/train 的两个端点都移动。不得只更新 held，或对称核只记一端。B、new Ridge、gate 的各 covector 相加后再沿 (21)、(20) 回到五个坐标；没有 encoder backward、source 样本或 query feedback。

### 5.4 一次 SPD 子问题与真实接受

把全部 held score 展平为 $h$，每类平均 CE 向量为 $\ell$，记 $B_\ell=\partial\ell/\partial h$、$J=\partial h/\partial\theta$。使用 (13) 或 C 完整链构造 J。对 $F>0$，

\[
v_c={\ell_c\over C F},\quad
W=\sum_c v_c\nabla_h^2\ell_c+
B_\ell^\top\left({I\over C F}-{\ell\ell^\top\over C^2F^3}\right)B_\ell\succeq0,
\]
\[
g=J^\top B_\ell^\top v,\qquad H_5=I_5+J^\top WJ\succ0. \tag{22}
\]

每个 CE Hessian 是相应 softmax 的 $\operatorname{diag}(p)-pp^\top$，含每类物理平均权重。括号是向量 RMS Hessian，PSD；不能把 RMS 简化为 unweighted mean CE。GGN 丢弃 nonlinear score 的二阶项，因此不是完整真实 Hessian。

[Martens，Deep learning via Hessian-free optimization，ICML 2010，第 4.1、4.2 节](https://icml.cc/Conferences/2010/papers/458.pdf)讨论广义 Gauss–Newton 的 PSD 曲率，以及正 damping 限制低曲率方向步幅的作用。该论文采用调整 damping 的 Levenberg–Marquardt 策略；本文固定 $I_5$ 只为本项目的五维二次子问题提供 SPD 性质，不沿用其 damping 策略或性能结论。式 (22) 的 RMS 跨类曲率由本文直接求导，论文没有推导本候选的 RMS/解析 Ridge/barrier 组合。

从当前阶段的 anchor、相对坐标 0，仅解一次

\[
\min_{\|d\|\le\rho}\ g^\top d+\tfrac12d^\top H_5d,\qquad\rho=0.5. \tag{23}
\]

若 $\|H_5^{-1}g\|\le\rho$，$d=-H_5^{-1}g$。否则唯一 $\lambda>0$ 使

\[
d=-(H_5+\lambda I)^{-1}g,\quad\|d\|=\rho. \tag{24}
\]

五维 SPD 因子或一次 eigendecomposition 加一维单调 secular 求根即可；上界 $\lambda=\|g\|/\rho$ 已使 $\|d\|\le\rho$。求根容差属于数值求解定义，不是候选或性能搜索。奇异 Q、奇异 J 和零 CE 曲率都不破坏 $H_5$ 的正定性。$g=0$ 时 $d=0$，明确记录不更新。

对非零 d，(23) 比零解更优，故 $g^\top d<0$。按表中固定步长检查

\[
F(\theta_{anchor}+\eta d)\le F(\theta_{anchor})+10^{-4}\eta g^\top d. \tag{25}
\]

每个试点重做完整合法 inner head/gate，不复用其他参数点的解冒充真实目标。式 (25) 是实数接受条件；实现中的浮点比较和原 objective evaluation 误差需分别归档，容差内的观测不代表精确实数下降。数学连续性保证足够小步长下降，不保证固定 12 次预算一定找到它。全拒绝时保留本阶段 initial 合法拟合及零 adapter 更新，记录全部成本；C 的 initial 拟合仍是完整 new Ridge+gate，绝不用 B 替代一次失败注册。数值求解失败需保留失败和 partial 证据，不可回退、抹梯度或宣称完成。gate 资源接口继续要求显式正整数 Newton/line/factor-buffer 上限；本文不改变当前执行额度，理论可解不意味任何有限额度必成功。

更新的训练 gradient/J 为上述精确链；不是对整个训练策略作全局光滑假设。若需要对二次解作额外 meta-derivative，inactive 时 $dd=-H_5^{-1}(dg+dH_5d)$；active 时令 $A_5=H_5+\lambda I$，

\[
d\lambda=-{d^\top A_5^{-1}(dg+dH_5d)\over d^\top A_5^{-1}d},\quad
dd=-A_5^{-1}(dg+dH_5d+d\lambda\,d). \tag{26}
\]

active 转换和首次 Armijo 接受索引的边界可能不可微。本文不优化这些数值策略，也不在边界伪造零或任意 subgradient。source 原型几何和参数默认值始终固定。

## 6. 继承保证、无信息与技术边界

对所有输入，C 由 (11) 给出

\[
s_y(x)-s_k(x)=f_{B,y}(x)-f_{B,k}(x)\quad(y,k\in old). \tag{27}
\]

因此条件旧类排序严格继承实际 B；C 不会纠正 B 的 old-old 错误，也不会通过自己的共享校正破坏它。由 strict slack，old train 对每个 new j 有 $s_{y_i}-s_j>d_i$。这只保护当前 legal inner-train/final-train support 的原真实 margin；若 $d_i\le0$，不等于该样本分类正确。query 的 gate 可以选择新组，故没有 query 旧类不遗忘保证；没有新类接近旧类或提升 10 个百分点的数学保证。

| 情形 | 必须执行的数学语义 |
|---|---|
| new count 0 | C 直接复用实际本候选 B 状态，先于任何新头/gate；零新增拟合 |
| K1 / no-held | OOF adapter 目标不可用，B/C adapter 更新关闭并标 N/A；仍拟合完整 final support Ridge；有新类时仍拟合完整 new head 与 gate，不虚构 held 指标 |
| Q=0、有效 rank 0 或目标 g=0 | 恒等或零更新；head 仍按完整解析式解。不得把无信息声明为风险改善 |
| Q/J 线性相关、PSD K 奇异 | Ridge $I+K$、GGN $I+J^\top WJ$、gate $I+\sqrt D K\sqrt D$ 均无需伪逆/jitter；自由截距不得删 |
| $\gamma_0$ 为 None 或 0 | raw kernel 为零，feature gradient 为零；adapter 不更新。new/gate 的自由常数仍必须拟合 |
| $\tau_0=0$ | 使用原 exact-equivalence kernel。因为 (8) 含固定半原距离，不同原特征不能变为等价；同原特征受同一确定映射后仍等价，故该核与参数无关，adapter 不更新 |
| q=1 | new conditional log-probability 恒为 0，新头完整 target/自由截距仍保留；binary gate 与其 kernel derivative 通常非零，不能因此省掉 C |
| 极端 logits | 用 logsumexp、logsigmoid、label-aware logistic residual；曲率用 $e^{-|g|}/(1+e^{-|g|})^2$ 避免 $p(1-p)$ 的相减损失。真实 underflow、非有限、非正 Schur 或预算失败明确报技术失败，不补 jitter 或更换目标 |
| ties | canonical 物理类 ID tie-break；(9) 全 j barrier 不依赖 max ties 的梯度。冻结 old margin 的 max 不对 C 参数求导，不代表忽略 new RHS 导数 |
| 浮点 RMSCE=0 | 多类有限数学 logits 的 CE 严格为正；不能除零后给伪梯度。按数值失败/不支持 Jacobian 记录，不加 epsilon 偷换目标 |
| 原型缺槽或 lineage/schema 不兼容 | 本候选参考几何不可用，不能在 Phase2 重建 source 原型、导入 source 样本或借旧结果填补；当前已冻结方法不受影响 |

不存在旧 Margin QP 的独立 active/strict 互补 gate 条件；但这不保证任意幅度 logits 和有限资源下数值求解成功。barrier gap 预算、strict feasibility、实测残差和实际目标接受应分别记录。

## 7. 参数、计算、状态与统计传输

### 7.1 理论操作量

令 $d=160$、$r=5$、完整输入 $D=736$、每折 train 数 N、held 数 E、总类数 C、new train 数 p、旧 train 数 m、新类数 q。f 为实际 inner fold 数，正常 $f\le3$。下列为 dense 实现量级，不能替代实际运行测量。

| 工作 | 操作量及原因 |
|---|---|
| 固定字典与共享 feature map | Q 构造 $O(d r)$；每样本求 w、投影及五列 Jacobian 约 $O(d r)$。同一参数点 w 可共享，不能共享样本切向投影 |
| 一次 raw kernel | $O(N^2D+END)$，implicit interaction 避免 123616 维展开；仍有所有物理 pair |
| 一次 Ridge forward | Cholesky $O(N^3)$，RHS/free intercept $O(N^2(C+1))$，eval $O(EN C)$ |
| 五列 Ridge Jacobian | 复用 factor，$O(rN^2(C+1)+rEN C)$；另计 $O(r(N^2+EN)D)$ kernel sensitivity |
| GGN 五维解 | 曲率收缩及 $O(r^3)$ 小因子；用 softmax Hessian 乘法可避免存巨大的 score Hessian，但 J/score 方向仍需计算 |
| 一次 C new Ridge | 用 p 替换 head 的 N、q 替换 C，cross eval 包含全部 old train 和 held |
| C gate | 每个实际 Newton factor 约 $O(N^3)$，objective/line trial 约 $O(N^2+m q)$；五个方向复用最终 factor 但有 $O(rN^2+r m q)$ 伴随工作 |

一次“解析 adapter 更新”不是一次总求解。初点各 fold 的 heads、完整 J、最多 12 个真实试点各 fold 的 heads，以及最终 full-support head 均收费；C 每个试点还包括 gate 的实际 Newton、失败 line trials、阈值和新头。不得把解析 head 写成零耗时，也不得只计最后接受的一点。与当前四次 gradient 更新相比，本候选减少了规定的 outer 更新数，但五方向 GGN、额外参考几何和 gate 内求解可能抵消该差异；实际设备收益未知。

名义 adapter 为 5 个 float64 坐标，即 40 B；B/C 各保留一份时为 80 B，不包括 Q、head、gate 和 audit。当前 `U` 最大 $736\times8=5888$ 个坐标，即 47104 B，仅是 U 的数值数组，不能拿它与本候选全部状态作不等口径比较。若 Q 常驻，另有 $160\times5\times8=6400$ B；可从冻结原型确定性重建，但重建成本和原型常驻状态仍收费。

Ridge 部署通常保留 support 分支特征 $O(ND)$、系数 $O(NC)$、free b 和固定 centering/geometry 摘要；是否保留 $N^2$ kernel/factor 取决于实际 sealed 部署实现。C 至少保留实际 B、new head $O(pq)$、gate canonical 系数 $O(N)$、free constants 和必要 support 特征。训练还需 kernel/factor $O(N^2)$、cross $O(EN)$、五方向 head/gate 数组和试点状态；streaming 可降某些 buffer，不能把同驻 factor 与 RHS 忽略。

成本账本应分别记录：SUM 型 factor/solve/physical-pair/line/adjoint 次数及实际秒数；MAX 型同驻 factor/临时 buffer/内存峰值。一次失败的第二 triangular solve 仍是实际 attempt，保留 partial 数组和完成计数；未知工作不得默认为 0。部署常驻 bytes、archive 文件 bytes 和峰值训练内存也不是同一个量。硬件、train/query 秒数、CPU/GPU 峰值和 native 网络传输在本文全部 N/A。

### 7.2 原型包的字节公式

本文不假定已有 packet 必须重新传输。若兼容原型已经随合法冻结 bundle 驻留，本候选新增 ground 数据/统计 payload 为 0 B；code/schema 发布的 wire bytes 独立未知，不能算成 0 B。若需要计完整既有原型包，其格式必须用真实 schema，以下只是可审计公式：

\[
B_{proto}=C_o d\,b_q+n_s b_s+B_{mask}+B_{classmap}+B_{schema}+B_{framing}. \tag{28}
\]

$b_q$ 是每坐标字节，$n_s$ 是实际 quantization scale 数，$b_s$ 是每 scale 字节。若每类一 FP16 scale、int8 坐标、bit-packed 有效槽，数值部分为 $C_od+2C_o+\lceil C_o/8\rceil$ B；六类、160 维为 $960+12+1=973$ B。若 schema 使用 byte mask，则为 978 B。float32 六类原型坐标为 3840 B，另加真实 metadata；不能假造其量化格式。

现有白名单若使用 int8 domain×class 中心、S 个 domain、d 维，则数值项为 $S C_o d+2n_s+B_{mask}$。若使用已冻结的 compressed class 中心、rank R domain residual 方向及 domain 系数，则对应项为 $C_o d+R d+S R+C_o+2n_s+B_{mask}$，其中 $C_o$ 项仅在既有 schema 确有 int8 类半径时加入；可选 FP16 全局 feature location/scale 为 $4d$ B。上述统计只按已登记冻结公式使用，本候选不要求它们、不重估或增加它们。

int8 舍入未发生 clipping 且每类 scale 为 $\Delta$ 时，坐标解码误差范数至多 $\sqrt d\Delta/2$，还需加 FP16 scale 误差项 $\|q\|\,|\widehat\Delta-\Delta|$。clipping、归一化零点和低 rank 压缩需各自 schema 的误差界。此类量化界既不证明原型代表 target 分布，也不证明角度校正的收益。packet 文件大小、压缩容器大小、native wire bytes、重传和星载实测必须分开；只用 (28) 不能宣称实际传输已节省。

## 8. 文献支持范围与交付边界

[Bertinetto 等，Meta-learning with differentiable closed-form solvers，ICLR 2019](https://arxiv.org/abs/1805.08136)支持通过 Ridge 求解器反向传播、将解析分类头作为适应机制，以及小样本矩阵求解的设计。论文不证明本文五维字典、一次 GGN、冻结原型权限或 target query 提升。

[Snell 等，Prototypical Networks，NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)研究学习的嵌入度量与类别原型分类。本文不采用其原型分类器，论文也不证明仅凭源原型即可识别共享域逆变换。

(6)—(26) 是本项目候选推导。五维原型差字典、GGN1（每阶段最多一次接受更新）、固定 damping 与半径 0.5 硬球的组合是本项目的新候选；它不是 R2D2、Martens 的 Hessian-free 算法或其他论文已经验证的既有方法，不承接这些论文的性能结果。若实施，最小正确性验证应针对独立 synthetic：free-intercept Ridge、gate RHS、train/held 两端、完整五方向 Jacobian 与真实目标、实际 B→C lineage、K1/零核/奇异字典/new0/新 1、严格单样本全类决策、失败成本及 partial state。本文没有执行这些验证，不产生实现就绪或泛化已验证状态，也不授权新实验、参数选择、query 重跑或修改健康任务。

最终判断：一个低维共享域校正的“一次解析二次更新”在上述定义下可解且可完整求导；把整个联合分类问题说成一次闭式拟合、保证新旧类性能目标或必然降低总成本，均缺乏依据。
