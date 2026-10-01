# ProtoFrame 五维坐标、GGN 阻尼与 support 度量

日期：2026-10-02。状态：**仅下一步数学推导，未实现、未数值验证、未形成实验配置**。本文只读取冻结方法源码和数学说明，未读取任何实际 support/query 产物、分数、缓存、权重、运行配置或索引。作者不是代码库唯一修改者；本次只新增本文，不修改现行 core、producer、analyzer、scorer 或运行状态。

## 1. 有限结论

现行五维坐标中的 `GGN + I5` 与半径 0.5 是明确、正定的优化定义，但不具有任意线性坐标缩放不变性。原型差向量的长度、夹角以及接近线性相关的程度会影响同一 `I5` 对真实特征变化的约束强度。完整 Ridge/gate 隐式导数使优化方向对应真实组合函数，并不消除这种尺度依赖，也不自动带来 support 到 query 的泛化保证。

本文只提出一条后续可检验候选：在每个阶段初始 anchor，以已有冻结字典 Gram 与当前合法 support 的 OOF 预测 Fisher 构成度量

\[
M_S=Q^\top Q+F_S.\tag{1}
\]

用它同时定义二次模型阻尼和方向信赖域，在该阶段内冻结。所有随数据变化的量只来自当前合法训练 support；`Q` 仍是已许可、冻结的参考几何，不是额外源数据或教师。此候选不需要参数网格，但系数 1、相加定义和度量半径仍是新的方法设计，不能说成由数学唯一确定或对当前冻结方法的等价实现。

在 `rank(Q)=5` 时，(1) 严格正定。在秩不足时，它只在真实可达特征空间上正定，五维冗余坐标中的系数不唯一。可在商空间证明唯一物理方向；**目前没有实现可靠的浮点秩判定与商空间求解器**。不能以 jitter、伪逆或静默截断小奇异值宣称一般退化输入已经解决。

## 2. 对照的冻结源码

本次直接核对了以下函数，而未重新审查完整项目：

| 源码 | 本文使用的事实 |
|---|---|
| `code/cvsrffi/d92_proto_frame_primitives.py:239`，`build_proto_frame_dictionary` | 六个单位化原型相对于固定第六类作差，形成未白化的 `Q[160,5]` |
| 同文件 `:264`，`transform_z_id` | 共享切向映射，`kappa=.25`；零特征保持零，零坐标仍保留真实 Jacobian |
| 同文件 `:416`，`raw_kernel` | 固定原始 old-train 带宽及 trace scale，原始/适应距离各占一半；两端导数齐全 |
| 同文件 `:494`，`fit_free_intercept_ridge` | `I+K` 的同一因子解 forward 与五方向 JVP；完整自由截距 |
| 同文件 `:566`，`class_balanced_rmsce` | score GGN 加 RMS 外层曲率，随后加 `I5`，不加真实 proximal loss |
| 同文件 `:632`，`solve_hard_ball_step` | 正定五维二次方向与欧氏球；特征值/secular 求解 |
| `code/cvsrffi/d92_proto_frame_joint_local_ridge.py:365`，`_gate_jvp` | `sqrt(D)` 缩放的 SPD 系统、五方向及自由截距 Schur |
| 同文件 `:464`、`:557` | 完整 OOF RMSCE、至多一步、固定 0.125 起始比例和 12 次折半；C 继承自己的实际 B |

数学定义参见 [既有源码数学稿](D92_LOCAL_RIDGE_JOINT_GENERALIZATION_COST_NOTE_20261002.md) 的式 (6)–(8)、(12)–(13)、(15)–(16) 及 GGN 部分。下文是新推导，不把已有稿件中的实现前状态或任何运行信息作为性能证据。

## 3. 为什么 `I5` 不是与尺度无关的“小阻尼”

记当前阶段参数为 \(\theta\)，anchor 为 \(\theta_0\)，真实目标为 \(R(\theta)\)，其梯度和 score GGN 为 \(g,G\)。现行方向解

\[
\min_{\|d\|_2\le .5}g^\top d+\tfrac12d^\top(G+I_5)d.\tag{2}
\]

实际 trial 是 \(\theta_0+\eta d\)，\(\eta=.125\,2^{-j}\)，并回读真实 RMSCE。`I5` 只在 (2) 中，真实目标不是 \(R+\|\theta-\theta_0\|^2/2\)。方向球、初始 trial 比例和真实 Armijo 分别起作用，不能互相替代。

作可逆线性换坐标 \(\theta=T\phi\)，同时 \(Q_\phi=QT\)。同一物理变换 \(Q\theta\) 完全不变，完整导数给出

\[
g_\phi=T^\top g_\theta,\qquad G_\phi=T^\top G_\theta T.\tag{3}
\]

若换坐标后仍机械使用 `I5` 和欧氏半径，则映回原坐标的阻尼为 \(T^{-\top}T^{-1}\)，球为 \(d_\theta^\top T^{-\top}T^{-1}d_\theta\le .25\)，通常与 (2) 不同。只在正交 \(T\) 等特殊情况下相同。

一维、不激活球的代数例子更直接。设 \(\theta=s\phi\)，当前梯度为 \(g\)，曲率 \(\lambda\ge0\)。在 \(\phi\) 中加单位阻尼后，物理方向为

\[
\Delta\theta=-\frac{s^2g}{1+s^2\lambda}
=-\frac{g}{\lambda+s^{-2}}.\tag{4}
\]

这只是同一函数的参数化变化，没有增加表达能力，却改变了有限方向；球激活时还改变可达物理步长。不能据此判断当前坐标一定“过小”或“过大”，也不能由该公式推断任何实际结果的原因。

若 \(Q=U\Sigma V^\top\)，物理变量 \(w=Q\theta\) 在有效空间中的现行最小坐标范数为 \(\sum_j w_j^2/\sigma_j^2\)。因此 `I5` 对短字典方向更强，对长方向更弱。\(\ker Q\) 中的任何变化对全部样本变换、所有核和预测都严格无影响；其真实梯度和 GGN 必须为零。单位阻尼使冗余坐标方向唯一为零，但并不修正有效方向间的尺度差异。

Martens 的一手论文在 §8–§10 讨论 GGN、Fisher 与 damping，§12 及式 (11)–(14) 区分无穷小不变性和实际有限更新。本文的 (3) 是线性换坐标下直接的链式法则；不把一般非线性重新参数化也说成有限步精确不变。[Martens, 2020，正式论文](https://jmlr.org/papers/v21/17-678.html)。

## 4. 完整隐式导数改变的是哪一种几何

### 4.1 从切向位移到核

对非零原始 `z_id`，记单位方向为 \(u\)。在 \(\theta=0\) 时，单位输出的 Jacobian 为

\[
J_u(0)=(I-uu^\top)Q.\tag{5}
\]

一般 anchor 下还包含切向饱和和最终单位化的 Jacobian。由于饱和映射的导数算子范数不超过 1、\(u\perp\delta\) 从而 \(\|u+\delta\|\ge1\)，有

\[
\|du_\theta\|_2\le\|dw\|_2=\|Q\,d\theta\|_2.\tag{6}
\]

零原始特征的 Jacobian 为零。范数保持使每个分支的 floor-normalization 分母对该方向保持不变；结合固定 FFT 与三个辅助块，隐式 interaction 特征满足保守界 \(\|d\Psi\|\le\sqrt2\|dw\|\)。这不是对独立 encoder 扰动的界。

固定正 \(\tau\)、\(\gamma\) 时，对 \(d_{ij}=\frac12\|\Psi_{0i}-\Psi_{0j}\|^2+\frac12\|\Psi_i-\Psi_j\|^2\)，

\[
dk_{ij}=-\frac{k_{ij}}\tau(\Psi_i-\Psi_j)^\top(d\Psi_i-d\Psi_j).\tag{7}
\]

训练核和交叉核都要计算两个移动端点；“query 逐样本、仅推理”不等于数学上把 query 端 Jacobian 删掉。这里的导数仅用于理论或合法 inner-held 训练，绝不授权读取实际 query 求度量。

由于每个归一化块的范数不超过 1，\(\|\Psi\|\le\sqrt3\)，故 (7) 给出宽松的逐元素界 \(|dk_{ij}|\le4\sqrt6(\gamma/\tau)\|dw\|\)。它说明物理位移可控制核扰动，但常数会受实际固定几何尺度影响，不能忽略 \(\gamma/\tau\)、矩阵规模或后续 head。\(\tau=0\) 的冻结原始 exact-equality 分支和零核分支必须按原分段定义处理，不能代入除以 \(\tau\) 的公式。

### 4.2 Ridge 的正则化不等于整个组合函数都不敏感

设 \(P=I-\mathbf1\mathbf1^\top/n\)、\(A_c=I+PKP\)。自由截距 Ridge 可写为

\[
\alpha=A_c^{-1}PY,\quad b=\bar Y-\mathbf1^\top K\alpha/n,
\quad h=L\alpha+\mathbf1 b.
\]

固定标签与物理行时，

\[
d\alpha=-A_c^{-1}P(dK)P\alpha,
\quad db=-\mathbf1^\top[(dK)\alpha+K(d\alpha)]/n,
\quad dh=(dL)\alpha+L(d\alpha)+\mathbf1(db).\tag{8}
\]

\(K\succeq0\) 时 \(\|A_c^{-1}\|\le1\)，所以 \(\|d\alpha\|\le\|dK\|\|PY\|\)。但预测还含 \(dL\)、自由截距及 \(L\)；不能把这个系数界误说成完整 score 的单位 Lipschitz 界。源码使用等价的 `[Y,1]` Schur 公式，无需显式形成中心化矩阵。

C 的新头还产生全部 old-train/new-class 的 \(d\log p_N\)，进入每个 gate 阈值。冻结实际 B 后，C 的旧条件函数不随 C 坐标变化；合法 inner prior 仍须在 old inner-train 上重拟合，不能换成看过 held 标签的 full B。

### 4.3 Gate 的 SPD 系统仍有敏感方向

设 slack 为 \(s_{ij}\)，\(D_i=\sigma(g_i)\sigma(-g_i)+\sum_j\zeta/s_{ij}^2\)（新类行没有第二项）。记 \(R=\sqrt D\)、\(H=I+RKR\succ0\)。对五个方向同时有

\[
r_i=\mathbf1_{old(i)}\sum_j\zeta\,da_{ij}/s_{ij}^2,\qquad
v=-R(dK)\alpha+R^{-1}r.
\]

源码用有界比值 \((\zeta/s_{ij}^2)/D_i\) 构造第二项，避免直接除以极小 \(R_i\)。解

\[
t=H^{-1}v,\quad u=H^{-1}R\mathbf1,\quad
db=\frac{\mathbf1^\top Rt}{\mathbf1^\top Ru},\quad
d\alpha=R(t-u\,db),\quad dg=(dL)\alpha+L(d\alpha)+\mathbf1db.\tag{9}
\]

精确有限输入、严格 slack、\(D_i>0\) 时 Schur 分母正；奇异 PSD 的 \(K\) 不破坏这个结论。然而较小的 Schur 分母、大曲率对比或窄 slack 可使导数与浮点抵消变得敏感。`H` 最小特征值至少为 1，并不证明自由截距或整个 head Jacobian具有统一小常数。真实 underflow、分解/残差失败仍是技术失败，不能加 curvature floor 冒充真实导数。

GGN 用的是完整 (8)–(9) 后的 \(J_i=\partial s_i/\partial\theta\)，不是仅 adapter 输出、固定 head 或只改核一端的 Jacobian。删掉 \(db\) 或阈值导数，所得度量即使数值正定，也不再度量本方法的预测变化。

## 5. 唯一候选：物理字典加 support 预测 Fisher

### 5.1 定义和合法数据范围

在 B 的零 anchor、C 的实际 \(\theta_B\) anchor，按原 inner-fold 规则取得每个合法训练 support 的 OOF score \(s_i\)、完整 \(J_i\)、概率 \(p_i=\operatorname{softmax}(s_i)\)。设当前注册类数为 \(C\)，第 \(c\) 类物理 support 数为 \(n_c\)。定义

\[
F_S=\sum_i\frac1{C n_{y_i}}J_i^\top[\operatorname{diag}(p_i)-p_ip_i^\top]J_i,\qquad M_S=Q^\top Q+F_S.\tag{10}
\]

这是在经验 support 输入分布上、对模型输出类别取期望的 Fisher；不是观测标签梯度外积的 empirical Fisher。真实标签只决定合法 support 训练和类均衡输入权重，不把 `y_i` 代替 Fisher 中的模型输出类别期望。当前 outer-held 诊断样本和 query 均不进入 (10)。

当前 RMSCE 的 GGN 不等于 \(F_S\)。记类 CE 均值向量为 \(\ell\)、\(R=\|\ell\|/\sqrt C\)、类均值 Jacobian 为 \(B\)，则源码计算

\[
G=\sum_i\frac{\ell_{y_i}}{C R n_{y_i}}J_i^\top S_iJ_i
+\frac1{CR}B^\top\left(I-\frac{\ell\ell^\top}{\|\ell\|^2}\right)B,
\qquad S_i=\operatorname{diag}(p_i)-p_ip_i^\top.\tag{11}
\]

两项均 PSD。候选保留原真实 RMSCE 及 (11)，只把方向定义改为

\[
\min_{d^\top M_S d\le .25} g^\top d+\tfrac12d^\top(G+M_S)d.\tag{12}
\]

`M_S` 在该阶段的单次方向及所有 trial 中冻结，不随每个 trial 重估或做额外校准；真实回读仍不含任何 \(d^\top M_Sd\) 项。现有 trial 比例等常数仅作为将来可能复用的明确约束，本文未登记新的预算或启动候选。

(10) 可解释为两个映射的直和拉回度量：物理参考位移 \(w=Q\theta\) 的欧氏度量，以及 OOF 预测分布的局部 KL 度量。\(\frac12d^\top F_Sd\) 是加权 KL 的二阶项，非有限位移 KL 的精确值。这里两个组成项的权重固定为 1，是定义而非从泛化理论求得的最优系数。

### 5.2 正定、退化与坐标不变性证明

对任意 \(v\)，

\[
v^\top M_Sv=\|Qv\|^2+\sum_i\frac1{Cn_{y_i}}\operatorname{Var}_{c\sim p_i}[(J_iv)_c]\ge0.\tag{13}
\]

所有模型依赖 \(\theta\) 的路径都通过 \(Q\theta\)，因此 \(Qv=0\Rightarrow J_iv=0\)。反过来 (13) 等于零必有 \(Qv=0\)。故

\[
\ker M_S=\ker Q,\qquad \ker(G+M_S)=\ker Q.\tag{14}
\]

若 `rank(Q)=5`，则两矩阵 SPD，(12) 有唯一解；即使 support 的预测 Fisher 为零也成立。若 `rank(Q)=r<5`，则在 \(\mathbb R^5/\ker Q\) 上 SPD，\(g\) 正交于核空间，二次目标与约束只依赖等价类，因而**物理方向 \(Qd\) 唯一，但五维系数不唯一**。`Q=0` 时可达物理变化只有零，无须伪造 update。K1 没有 OOF Jacobian，保留“不更新 adapter、完整拟合 final head”的语义；不以 final training 分数冒充 OOF 来补 Fisher。

精确数学上可取 \(\operatorname{col}(Q)\) 的正交基，在该 \(r\) 维物理空间求 (12)，再选择一个系数代表。浮点下极小非零奇异值与真实零不可凭任意阈值混同；可靠误差界、rank 不确定时的明确技术失败和表示回读尚待实现。本文没有通过 `pinv`、额外 ridge floor 或“保留前几维”解决这个问题。

对第 3 节的可逆线性换坐标，有 \(F_\phi=T^\top F_\theta T\)、\(M_\phi=T^\top M_\theta T\)。因此 (12) 的目标、约束、物理方向及同一 \(\eta\) 的 trial 函数完全对应，真实 Armijo 的 \(g^\top d\) 也相同。这是精确算术、线性坐标变化的证明，不保证不同浮点计算图的最后一位或接受标志一致。

只在求解器内部对白化后的矩阵求解，却仍保留原 (2) 的阻尼和球，只是等价数值预条件，不能改变现行方法的尺度偏好。要获得上述物理不变性必须改变方向定义为 (12)，所以不得热改当前运行。

## 6. 能联系到泛化的只有条件性结论

(12) 给出 \(\|Qd\|\le .5\)，结合 (6) 可控制所有单样本的特征位移；(7)–(9) 在相应条件下把它传播到完整 score。若某个固定样本的原 winner margin 大于两倍逐类 score 变化上界，则其 winner 保持不变。这是保持既有决策的条件，不是保持正确性，更不是纠正错误的保证。

\(d^\top F_Sd\) 只控制当前 support 上的加权局部平均预测变化。一个方向可在全部 support 上几乎不可见，在未观测输入上却明显改变预测；字典 Gram 项给它有限的物理约束，但没有证明这种变化更正确。小 Fisher、小参数数目、局部条件数改善和 OOF CE 下降都不能单独推出 query 风险下降。

要把算法稳定性转成泛化界，必须控制替换一个 support 样本后，adapter、字典度量估计、物理折分、带宽/trace 统计、所有重拟合 head 及最终全 support 模型的变化，并说明抽样/分布条件。现行流程尚无这条统一稳定性证明。Bousquet–Elisseeff 的定义 6 和定理 12 给的是带前提的稳定性框架，不是对本候选的自动认证；未约束 CE 也不能直接套其有界损失结论。[原论文 PDF](https://www.jmlr.org/papers/volume2/bousquet02a/bousquet02a.pdf)。

因此，本文证明的是尺度依赖、候选局部度量的半正定/商空间正定和线性坐标协变；**未证明最终泛化、实际改进幅度或成本优势**。

## 7. 实际计算账与下一步证书边界

| 工作 | 当前完整 JVP 的实际范围 | 候选新增范围 |
|---|---|---|
| 一个 Ridge head，输出宽度 \(c\) | 一次 `I+K` 因子；forward 两次三角、各 \(c+1\) RHS；五方向 JVP 两次三角、各 \(5(c+1)\) RHS；合计 RHS 列收费 \(12(c+1)\)，另有实际 PSD 谱检查 | (10) 复用现有 OOF \(J_i,p_i\)，不新增 Ridge 求解 |
| 一次 gate JVP | 一次当前 `H` 因子及其条件估计；五方向加截距共 6 RHS，两次三角共 12 RHS 列；forward Newton 和全部失败/拒绝 trial 成本另计 | 不新增 gate JVP；不能说已有 JVP 免费 |
| 五维度量 | 当前已有完整 score Jacobian 和 GGN | `Q.T@Q` 的 \(160\times5\) Gram；(10) 可用概率加权中心化 Jacobian 累计 25 个元素，约 \(O(NC\,25)\)，无需 \(C\times C\) 稠密 Fisher |
| 度量球方向 | 现行五维 SPD 特征分解及 secular 求解 | 至多五维的度量因子、白化矩阵/梯度的实际三角 RHS，再求五维球方向；秩不足还需有明确计费的基/秩处理 |
| 真实目标回读 | 每次试步完整 forward heads/gate；拒绝和失败均计费 | 保持同样的真实回读职责，不能凭方向更好预扣试步次数 |

例如满秩时 \(M=R^\top R\)，令 \(z=Rd\)，则白化曲率为 \(R^{-\top}(G+M)R^{-1}\)，梯度为 \(R^{-\top}g\)，约束 \(\|z\|\le .5\)。需要实际记录这些三角求解、因子尝试/完成、condition/rank 检查、秒数和临时 buffer；不能把五方向说成五次大 Cholesky，也不能把新增五维工作说成零。

以上是操作公式，不是设备实测。原有 gate explicit factor-buffer 上限只约束其自己的因子工作区，不是整阶段常驻内存或 RSS 上限。参数只有五个，不代表 kernel/JVP 张量、new Ridge、barrier Newton 或归档字节也只有五个量。星载能耗、传输与整机峰值全部未测。

若后续单独实现，最小可检验对象是：非正交线性换坐标的同物理方向证书、完整 free-intercept/阈值 JVP、rank 完整与已知精确退化例子、同一真实目标的有限试步，以及新增实际 RHS/分解账。浮点近退化的可靠求解仍是具体未解决项。这些是候选实现应证明的行为，不是给现行健康任务增加审批或实验门槛；本文不启动该工作。

## 8. 引用核实与冻结

- Martens, J. *New Insights and Perspectives on the Natural Gradient Method*. JMLR 21(146), 2020，[期刊页](https://jmlr.org/papers/v21/17-678.html)，[正式 PDF](https://jmlr.csail.mit.edu/papers/volume21/17-678/17-678.pdf)。已核实 §8–§11 的 GGN/Fisher/damping 区分与 §12 的坐标变换论证；本文 (10)–(14) 是针对当前组合函数的独立推导，论文未评估本候选。
- Bousquet, O. and Elisseeff, A. *Stability and Generalization*. JMLR 2, 499–526, 2002，[期刊页](https://www.jmlr.org/papers/v2/bousquet02a.html)。已核实定义 6 与定理 12 的前提；未宣称本流程满足其稳定性或有界损失假设。

交付状态：数学说明冻结。仅进行 UTF-8 文本读回与局部公式自查；无数值测试、真实结果访问、参数选择、实现或发布。
