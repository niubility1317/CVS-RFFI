# Margin 联合头的有界 working-set 求解可行性

状态：**仅求解路线推导，未实现、未数值验证、未进入配置或实验。**本文不修改[结构推导](D92_MARGIN_JOINT_STRUCTURAL_DERIVATION_20261001.md)、生产代码、方法或已有 run；没有读取真实权重、cache、trace、训练日志、summary、历史索引或成绩。

唯一选择是：**从零 residual 可行点出发，使用保持 primal 可行的确定性 working-set 法，只对当前独立约束集形成并分解小矩阵，始终扫描全部约束。**它利用已有可行点与旧点响应块，省去寻找初始可行点，也不预先分配完整 \(q\times q\) dual Hessian。它不是无条件低成本的闭式求解器：active 集可以很大，退化可造成零步长或循环，普通梯度也可能不存在。

本文能给出精确算术的子问题、下降、阻塞与完整 KKT 公式，以及有限资源下“满足明确数值证书或技术失败”的执行边界；**尚不能宣称给出了覆盖全部退化输入、保证有限成功并处处返回唯一光滑梯度的完整可靠生产解法**。未解决项在第 8 节具体列出。

## 1. 固定问题与选择理由

沿用原推导：\(n=m+p\)，旧 train 为 \(O\)，注册类数 \(C\)，pair 约束数 \(q=m(C-1)\)。\(M=\operatorname{pad}(\text{actual B})\)、\(R=Y-M_T\)、\(d_i=\min_{j\ne y_i}(M_{iy_i}-M_{ij})\) 均冻结。约束是全部旧 train 对全部 registered 竞争列的 margin，不 clamp \(d_i\)，不接触 inner-held/query 来形成约束。

记训练 Gram \(K\succeq0\)，
\[
A=I+K,\quad z=A^{-1}\mathbf1,\quad s=\mathbf1^\top z,\quad
J=A^{-1}-zz^\top/s,\quad P=I-J.
\tag{1}
\]
无约束 affine 解为 \(\bar\alpha=JR,\ \bar b=z^\top R/s\)，训练 residual \(\bar r=PR\)。令
\[
\mathsf D\operatorname{vec}F=(F_{iy_i}-F_{ij})_{(i,j)},\quad
\delta_{(i,j)}=d_i,\quad h=\delta-\mathsf D\operatorname{vec}M_T\le0,
\]
\[
s_0=\mathsf D\operatorname{vec}(M_T+\bar r)-\delta,\quad
Q=\mathsf D(I_C\otimes P)\mathsf D^\top.
\tag{2}
\]
原 QP 的非负 dual 为
\[
\min_{\mu\ge0}\ \tfrac12\mu^\top Q\mu+s_0^\top\mu.
\tag{3}
\]

本路线在 **primal** 侧维护可行性，而不是把 \(\mu=0\) 当成已满足 margin 的解。零 residual 是已证明可行的 primal 起点；无约束 ridge 解则可能不满足 margin，二者不能混用。

Primal feasible active-set 的一般做法是保持 primal 不等式、调整独立基并消除错误符号的乘子；见 [Forsgren、Gill、Wong，2015，§1 与 §3](https://optimization-online.org/wp-content/uploads/2015/03/4848.pdf)。本文采用其一般思想推导本项目结构，不移植该论文的 shifted/penalized 问题、不增加正则项，也不将其测试结果视为本项目证据。

## 2. 只保存旧点块与按需约束响应

令 \(E_O\in\mathbb R^{n\times m}\) 为旧行选择矩阵。只需
\[
P_O=E_O^\top P E_O\in\mathbb R^{m\times m},\qquad
\bar r_O=E_O^\top\bar r.
\tag{4}
\]
通过 \(A\) 的同一分解求 \(A^{-1}E_O\)，可算
\[
P_O=I_m-(A^{-1})_{OO}+z_Oz_O^\top/s.
\tag{5}
\]
当 \(K\) 很小，式 (5) 有 \(I-A^{-1}\) 的相消；同一恒等式可更稳定地写作
\[
P_O=K_{O,T}(A^{-1}E_O)+z_Oz_O^\top/s.
\tag{6}
\]
式 (6) 仍需计入真实乘法及误差；它没有消除病态 rank 问题。原理上 \(P_O\succeq0\)，数值上需保留实际对称误差和分解残差，不能用对角 jitter 将负方向改成正方向。

每个约束 \(a=(i,j)\) 的 class difference 为 \(v_a=e_{y_i}-e_j\)。任意所需 Hessian 元素由
\[
Q_{ab}=(P_O)_{i\ell}\,v_a^\top v_b,\qquad b=(\ell,k)
\tag{7}
\]
即时生成。class 内积只包含 Kronecker delta 运算。没有必要存储 \(\mathsf D\) 的 dense 形式或全 \(Q\)。

对任意 pair 向量 \(u\)，先将其 scatter 成 \(V_O\in\mathbb R^{m\times C}\)：
\[
(V_O)_{i,y_i}\mathrel{+}=u_{(i,j)},\qquad
(V_O)_{i,j}\mathrel{-}=u_{(i,j)}.
\]
然后
\[
Qu=\mathsf D_O\operatorname{vec}(P_O V_O).
\tag{8}
\]
因此全约束响应可用 \(m\times C\) 数组和 \(m\times m\) 块实现。约束扫描是全部 \(q\) 行，而不是挑选若干“困难负类”。工作集只缩小线性系统，不缩小数学约束。

## 3. 可行迭代的紧凑 primal 表示

### 3.1 不必每次重新解 \(n\times C\) RHS

为推导使用有限 feature 的严格凸 primal：
\[
H_0=X^\top X+\operatorname{diag}(I,0),\quad
X=[\Phi,\mathbf1],\quad H_C=I_C\otimes H_0\succ0.
\]
记 \(x\) 为全部 feature 系数和截距的向量，\(\bar x\) 为无约束最优解，\(\mathcal E_O\) 为旧点分数的线性求值算子。于是
\[
\mathcal E_O H_C^{-1}\mathcal E_O^\top=I_C\otimes P_O.
\]
这只是证明用表示，不要求生产运行构造 \(\Phi\) 或新做 SVD。

维护
\[
x=\rho\bar x+H_C^{-1}\mathcal E_O^\top\operatorname{vec}V,
\quad \rho\in\mathbb R,\quad V\in\mathbb R^{m\times C}.
\tag{9}
\]
初始 \(\rho=0,V=0\)，故 \(x=0\)，即 \(g=b=0\)。旧点 residual 为
\[
r_O=\rho\bar r_O+P_OV.
\tag{10}
\]
每次只需要式 (10) 与全 pair 差分来检查可行性。\(V\) 是当前 primal 响应表示，不必等于某个非负 dual 乘子；不得在迭代中把它写成“已满足 dual 可行”。

令
\[
\kappa=\langle R,PR\rangle_F,\qquad
c(V)=\langle V,\bar r_O\rangle_F,\qquad
v(V)=\langle V,P_OV\rangle_F.
\]
完整 primal 目标可由
\[
\mathcal J(\rho,V)=\tfrac12\|R\|_F^2
+\tfrac12\rho^2\kappa+\rho c(V)+\tfrac12v(V)
-\rho\kappa-c(V)
\tag{11}
\]
计算。这来自原二次型，不是新 loss。式 (11) 接近最优时也会相消；最后仍必须使用实际 \(\alpha,b,K\) 重算原 SSE 与 RKHS 范数，而非仅相信该紧凑表达。

### 3.2 当前工作集的 equality-constrained 最优点

工作集 \(W\) 仅含当前 tight 且在 primal Hessian 度量下独立的约束，令 \(a=|W|\)。要求 \(Q_{WW}\succ0\)。解
\[
Q_{WW}\lambda_W=-s_{0,W}.
\tag{12}
\]
把 \(\lambda_W\) 只在 \(W\) scatter 成 \(V^\star\)，则当前工作集最优点为
\[
x^\star_W=\bar x+H_C^{-1}\mathcal E_O^\top\operatorname{vec}V^\star.
\tag{13}
\]
此时 \(\lambda_W\) 允许负值，因为它还是 equality-constrained multiplier。只有完成全部 KKT 检查后，才能把非负 \(\lambda_W\) 扩展为完整 dual 解。

方向 \(p=x^\star_W-x\) 对应
\[
\Delta\rho=1-\rho,\qquad \Delta V=V^\star-V,\qquad
\Delta r_O=\Delta\rho\,\bar r_O+P_O\Delta V.
\tag{14}
\]
其 primal 能量为
\[
\|p\|_{H_C}^2
=(\Delta\rho)^2\kappa
+2\Delta\rho\langle\Delta V,\bar r_O\rangle_F
+\langle\Delta V,P_O\Delta V\rangle_F.
\tag{15}
\]
不能只以 \(\Delta r_O=0\) 判定 \(p=0\)：方向仍可能改变新增 train 上的函数或 RKHS 范数。也不能仅看 \(\Delta\rho,\Delta V\) 的欧氏大小，因为退化情况下该表示可能非唯一。式 (15) 的近零判断必须考虑相消误差；无法可靠判定时属于技术边界。

## 4. 确定性步长、增删集与精确算术性质

### 4.1 完整阻塞检查

当前所有 slack 与方向为
\[
s_{\mathrm{cur}}=\mathsf D_O\operatorname{vec}(M_O+r_O)-\delta,\qquad
t=\mathsf D_O\operatorname{vec}\Delta r_O.
\tag{16}
\]
对于 \(p\ne0\)，在所有非工作集约束上扫描
\[
\theta=\min\left(1,\ \min_{a\notin W:t_a<0}\frac{s_{\mathrm{cur},a}}{-t_a}\right).
\tag{17}
\]
没有阻塞者时内部最小值为 \(+\infty\)。更新
\[
\rho\leftarrow\rho+\theta\Delta\rho,\qquad
V\leftarrow V+\theta\Delta V.
\tag{18}
\]
精确算术下 \(\theta\in[0,1]\)，所有约束保持可行。若 \(\theta<1\)，将一个真正的阻塞约束加入 \(W\)。不能只扫之前 active 行，不能用 subsampling、top-k 竞争列或部分 slack 代替式 (17)。

用固定物理行顺序和固定注册列顺序定义 pair 的整数索引。精确相同的最小阻塞比例只加入最小索引一行；其余仍被下一次全扫描覆盖。若 \(\theta=1\)，先到工作集最优点，再重查全 KKT，不因终点碰到约束就同时加入一批可能冗余的行。

### 4.2 下降与独立性

当前 \(x\) 和 \(x^\star_W\) 都满足工作集等式，因此
\[
\nabla\mathcal J(x)^\top p=-p^\top H_Cp.
\]
沿方向的目标变化恰为
\[
\mathcal J(x+\theta p)-\mathcal J(x)
=-\theta(1-\theta/2)\|p\|_{H_C}^2.
\tag{19}
\]
所以 \(p\ne0,\theta>0\) 时严格下降；\(\theta=0\) 不下降，必须作为真实退化事件记录。

若真正阻塞行的 primal 法向量在工作集行的 span 中，则由于工作集满足 \(A_Wp=0\)，该行也有 \(a^\top p=0\)，不可能满足式 (17) 的 \(t_a<0\)。因此**精确算术下真正阻塞行必然独立**。这说明保留独立工作集能够覆盖退化 PSD 问题，而不是要求所有约束一开始独立。

这一证明不授权浮点实现跳过“看起来相关”的约束。若某行被计算成阻塞者，但新增 Schur pivot 无法可靠判正，应报告数值不一致或 rank 模糊；忽略该行可能直接破坏 margin。

### 4.3 零方向时的乘子检查

若 \(p=0\)，当前点就是工作集最优点：

- 若 \(\lambda_W\ge0\)，令非工作集乘子为零，并重新检查全部 slack、stationarity 与互补条件；通过才是全 QP 的最优证书。
- 若有负乘子，按固定规则删除最小索引的可靠负乘子行，再解下一个工作集子问题。这里只删工作集行，不删原约束。
- 若乘子符号在数值误差内无法确定，不能随意赋为正并宣称 strict complementarity。Forward 可按明确数值 KKT 容差判断；普通梯度资格另判。

最小索引规则提供确定性，**不构成退化情况下的 anti-cycling 定理**。多条零步长和零乘子可能使工作集反复变化。本文不把词典序“看起来固定”偷换成已证明的全局有限成功。

## 5. 因子更新、PSD 与成本上界的真实含义

加入阻塞行 \(b\) 时，只生成向量 \(q=Q_{Wb}\) 与标量 \(Q_{bb}\)。若 \(Q_{WW}=LL^\top\)，则
\[
w=L^{-1}q,\qquad \sigma=Q_{bb}-w^\top w.
\tag{20}
\]
当 \(\sigma>0\) 时，可用该 Schur pivot 扩展 Cholesky；删除工作集行时使用确定性正交旋转更新，或明确计费地重分解当前 \(Q_{WW}\)。不能将一次更新写成零分解成本，也不能对不确定或负 \(\sigma\) 加 jitter。

固定这一路线的最初实现应只选择一种已明确定义的因子维护方式。本文建议首个合成证书采用**每次工作集变化后按式 (7) 重建并 Cholesky 分解当前 \(Q_{WW}\)**，先避免未审清的 downdate 误差；式 (20) 用于解释独立性及后续可选实现优化，不构成第二种求解候选或当前已启用优化。重分解成本必须全部记录，不能拿未来 rank-one 更新成本替代当前实际成本。

\(K\succeq0\) 时 \(A\succ0\)，但 \(Q\) 可能奇异。维度满足
\[
\operatorname{rank}(Q)\le
\min\{q,\ (C-1)\operatorname{rank}(P_O)\}.
\tag{21}
\]
工作集最多达到这个 rank。零核时 \(P_O=\mathbf1_m\mathbf1_m^\top/n\)，rank 至多 1，故独立 pair 工作集最多 \(C-1\) 行；这并不允许忽略其余重复、不同下界或不同标签约束。它们仍参与式 (16)–(17) 的全扫描。

本路线不预先构造全 \(Q\)，但**不能保证最坏 active factor 仍小**。当 \(K\succ0\) 且全部 pair 最终独立、active 时，\(a\) 可以达到 \(q\)，当前因子仍会需要 \(O(q^2)\) 存储和 \(O(q^3)\) 分解。若资源上限不允许，正确结果是技术资源失败，不是 silently cap 工作集并忽略其他约束。

可在分配前按当前 \(a\) 估算必需 buffer：
\[
8m^2+8mC+16a^2+\text{实际 }A/R/z/\text{临时工作数组字节},
\tag{22}
\]
并对照事先声明的可用内存。式 (22) 中先按同时保留 working 矩阵与因子计两个 dense 数组；它仍只是示意，不得漏掉额外副本、slack、scratch 和库工作区。只有实际实现证明可安全覆盖复用时，才能减少这部分计数。最终应按实现实际数组计数；所有数量是浮点 numeric buffer 代理，不是进程峰值或真实星载部署测量。

## 6. 最终全 KKT、primal-dual 证书及失败分类

### 6.1 必须回到原 head 方程复核

成功候选的非负 \(\mu\) 仅在 \(W\) 有值。重新 scatter 到 \(V_O\)，在其余训练行补零成 \(V_T\)，再用原 \(A\) 解
\[
\alpha=J(R+V_T),\quad b=z^\top(R+V_T)/s,\quad
F_T=M_T+K\alpha+\mathbf1 b.
\tag{23}
\]
使用式 (23) 的真实分数重新扫描全部 margin，不能只沿用 compact 迭代中的 \(P_OV\)。

必要残差包括
\[
e_{\mathrm{stat}}=A\alpha+\mathbf1 b-R-V_T,\quad
e_b=\mathbf1^\top\alpha,\quad
s_{\mathrm{full}}=\mathsf D\operatorname{vec}F_T-\delta,
\tag{24}
\]
\[
v_p=\|(-s_{\mathrm{full}})_+\|_\infty,\quad
v_d=\|(-\mu)_+\|_\infty,\quad
v_c=\|\mu\odot s_{\mathrm{full}}\|_\infty.
\tag{25}
\]
还需核对 compact 与原展开一致、原目标有限、工作集方程残差以及 dual 目标。令无约束最优目标为 \(\mathcal J_0\)，则
\[
g(\mu)=\mathcal J_0-s_0^\top\mu-\tfrac12\mu^\top Q\mu.
\tag{26}
\]
\(Q\mu\) 用式 (8) 计算。只有 \(\mu\ge0\) 且对偶构造正确时，\(g(\mu)\) 才是有效下界；可行 primal 的 gap 为 \(\mathcal J(\alpha,b)-g(\mu)\)。在 stationarity 精确成立时，它等于 \(\mu^\top s_{\mathrm{full}}\)。未满足这些前提时，不能将一个很小的互补积单独称为最优证书。

浮点残差允许的误差尺度应来自机器精度、矩阵规模、实际 RHS、分解 backward error 与条件估计。举例，标准舍入模型中 \(\gamma_t=tu/(1-tu)\) 可界定长度 \(t\) 点积误差；pair slack 的误差还需传播 \(P_OV\)、prior 分数和下界的误差，不能仅用一个固定绝对常数与所有问题比较。这是数值实现尚需固定的规格，不是新增性能系数。

### 6.2 有限退出，不混淆“最优”与“最后可行点”

操作输入必须在运行前明确有限的内存与计算工作上限；它们是技术资源界限，不进入目标、margin 或成绩选择。当前 4×12 的外层预算不是每个 QP 的内部上限。本文不根据未观测性能替 owner 选择具体限值。

每轮至少计一个 working-set transition；每次全扫描、矩阵构建、分解和 solve 都增加单调工作计数。达到预先声明的有限上限后立即保留状态并技术失败，从而保证有界退出；不能无限尝试直到某次返回成功。

必须区分：

| 退出类别 | 能说什么 |
|---|---|
| 完整数值 KKT 通过 | 在报告的残差尺度内求得原 QP head；仍需单独判断梯度是否可用。 |
| 资源上限耗尽 | 尚未获得规定最优证书；保留最后可行 primal 与全部轨迹，不将其作为“最优 head”供训练悄悄继续。 |
| 工作集循环或无可靠进展 | 技术停滞；固定 tie-break 不是已证明的去循环规则。 |
| rank、pivot 或符号无法可靠判定 | 数值模糊，不改 Gram、不跳过阻塞约束、不调用随意 pinv 来伪装成功。 |
| 完整约束/原方程复核失败 | 保留原误差与状态，不能仅凭内层 solver 状态接受。 |
| 梯度资格不满足 | Forward head 可能合格，但该点普通反向没有得到证明；不得用零梯度或无约束梯度替代。 |

由于本问题已有零 residual 可行证据，数值失败不应报告成“原数学 QP 不可行”。也不能把同一工作集再次出现自动称为严格循环：它可能发生在不同可行点。实际 cycle 证据需要至少包含工作集、primal 状态及目标/步长；只凭重复工作集设置保守中止时，应准确标为保守停滞界限。

## 7. 能提供完整梯度的范围及实际账

### 7.1 普通 VJP 的充分资格

以下是清楚而保守的充分条件，不声称必要：

1. 原问题完整 KKT 已通过规定精度；
2. 全部 tight 约束已识别，且就是一个独立 active 集 \(\mathcal I\)，\(Q_{\mathcal I\mathcal I}\) 的求解误差与正定性可控；
3. active 乘子有可分辨的正间隔，inactive slack 有可分辨的正间隔，当前 kernel 参数化在邻域平滑。

这时可以提供结构推导中的完整 VJP。令 held 上游 \(G\)，先解
\[
\begin{bmatrix}A&\mathbf1\\\mathbf1^\top&0\end{bmatrix}
\begin{bmatrix}T_G\\t_G\end{bmatrix}
=
\begin{bmatrix}L^\top G\\\mathbf1_h^\top G\end{bmatrix},
\]
\[
\eta=Q_{\mathcal I\mathcal I}^{-1}
\mathsf D_{\mathcal I}\operatorname{vec}T_G,\quad
B_\eta=\operatorname{unvec}(\mathsf D_{\mathcal I}^\top\eta),\quad
W_\eta=JB_\eta,
\]
\[
\overline L=G\alpha^\top,\qquad
\overline K=-\operatorname{sym}[(T_G+W_\eta)\alpha^\top].
\tag{27}
\]
训练 old/new 全端点、held 全端点和 \(g_b=\mathbf1_h^\top G\) 都保留。一个 \(C\)-RHS affine 伴随、一个 \(a\)-维单 RHS active solve、又一个 \(C\)-RHS 的 \(JB_\eta\) solve 都有真实成本；分解复用不等于反向免费。

[KKT 微分的一手背景是 OptNet 的 §3、式 (3)–(8)](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf)；其 §3.1.1 讨论特定 batched solver 的反向分解复用。本文没有实现该 solver，也不能继承其 GPU 速度声明或据此宣布本项目省算力。

如果 working set 只是全部 tight 行的一组独立子集，另外还存在 tight 冗余行，**不能仅因这组基可逆就直接认证式 (27) 为完整局部梯度**。需证明被省略的约束在邻域保持相同冗余关系和下界；否则它们可能在扰动后限制不同方向。本路线目前没有补齐这项一般退化灵敏度证明。

Active 切换、零乘子、模糊 slack、rank 变化和 \(\tau=0\) 等价组切换时，不承诺存在单一 Jacobian。方向导数或 critical-cone 方法仍待明确定义。不得通过 jitter、随机 perturbation、clamp \(d\) 或伪逆把不可微问题变成“已光滑”。

### 7.2 每个 head 的实际计算项目

| 阶段 | 实际计费口径 |
|---|---|
| 预计算 | 原核与距离；一次 \(n\times n\) 的 \(A\) 分解；\(z\) 的 1 RHS、\(\bar\alpha\) 的 \(C\) RHS、\(P_O\) 的 \(m\) 选择 RHS；式 (6) 乘法。是否批量合并只改变调用分组，不改变 RHS 总量。 |
| 每轮 primal 响应 | \(P_OV\) 或 \(P_O\Delta V\) 的实际 \(O(m^2C)\) 工作；已计算可复用部分只按实际记。 |
| 全约束检查 | 每次扫描全部 \(q=m(C-1)\) 个 margin、方向与比例；包括拒绝/零步/最终审计。 |
| 工作集改变 | 按式 (7) 生成实际 \(a\times a\) 小矩阵并分解，真实 \(O(a^3)\)；乘子 solve 的维度和 RHS 逐次记录。初版不能按未实现的 rank-one update 优惠计费。 |
| 完成读回 | 式 (23) 的最终 \(C\)-RHS affine solve、原 train 分数/目标/KKT 全量重算；不是免费复用旧 compact 数字。 |
| 反向 | 式 (27) 全部 solve、kernel/geometry VJP；失败反向已完成工作也保留。 |
| 状态 | \(A\) 因子、\(P_O\)、\(R,z,\bar r_O,V\)、working 因子、slack/multipliers、prior B、最终 \(\alpha,b\) 及暂存数组；persistent、peak 和部署状态分开。 |

对每个实际三角 solve 记录维度 \(d\)、RHS 数 \(r\)、\(r\)、\(dr\)、\(d^2r\) 代理；完整 Cholesky solve 一般含两次三角调用，失败中的 partial 调用按实际保存。工作代理不是实测 FLOPs，更不是耗能。CPU 时间、进程/设备峰值和最终部署传输字节均须实测；目前全部未知。

## 8. 尚未解决项与是否进入实现的判断

这条路线的有利结构是确定的：可行起点已知；只需 \(P_O\) 与 pair 稀疏操作；任意真正阻塞行在精确算术下独立；所有工作集子问题都对应原目标；成功可由全 KKT 独立检查。它有希望将实际成本集中到较小工作集，但**实际 active 大小、transition 数、峰值和耗时没有测量**。

目前仍未解决的具体项是：

- 浮点下 Schur/rank、近零方向和乘子/slack 符号的统一误差传播与停止规格；不能只任意选一个容差就声称所有边界可靠。
- 不改变原目标或下界的退化 anti-cycling 实现与证明。最小索引规则仅使路径确定，有限资源界限仅保证失败或成功都能退出，不保证所有输入在界限内成功。
- 多个 tight 冗余约束、非唯一 dual、弱 active 行与 rank 变化处的完整方向导数规格；当前只承诺第 7.1 节正则区域的 VJP。
- 最坏 \(a\) 接近 \(q\) 时成本不可被结构本身消除；需要依据合成计算账确定是否符合真实资源预算，而非通过成绩选择 working-set 上限。
- 独立 finite-feature primal oracle、正则区完整端点差分、退化与循环反例、失败证据和实际资源 ledger 的实现证书。主 agent 可独立做合成数值工作，本文没有执行这些检查，也不预判通过。

因此当前结论是 **一条可推导且有明确技术失败边界的候选求解路径**，不是已完成的通用稳健 QP/反向组件。它没有新增性能调参系数、候选扫描或 launch 授权；任何尚未提供的正确性与资源证据都不能由“convex”“active-set”或论文引用代替。

## 结构证书与本求解器的区别

主Agent已完成结构推导的19项独立小矩阵证书（0.42 s）。该证书用枚举oracle验证head/KKT/VJP，不实现本文的working-set迭代。因此本文的compact迭代、全部阻塞检查、数值误差与去循环仍未通过实现测试；不能把结构证书登记为本求解器已验证或已启动。
