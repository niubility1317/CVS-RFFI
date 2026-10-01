# MarginJoint LocalRidge 联合微调的数学依据与边界

这项方法保留固定的 Phase1 表示和 LocalRidge 适应头，在合法 support 上联合更新一个小 adapter。B 阶段学习旧类；C 阶段继承该次实际 B 的函数，并在求解新头时约束旧训练点的真实类 margin。结构和目标先由数学定义，再由固定预算的求解器执行；训练不依靠参数组合搜索或评分反馈选择路径。

本文解释当前 `D92-MarginJointLocalRidge-v1` 源码，不修改方法、参数或配置，不引用真实数据、训练产物或准确率。依据限于[Margin 结构推导](D92_MARGIN_JOINT_STRUCTURAL_DERIVATION_20261001.md)、[前序联合方法的论文映射](D92_JOINT_SFT_PAPER_MAPPING_20261001.md)和核心源码。前序映射的对象是 ConditionalJoint，其“旧点完整分数不变”定理和固定计算模板不能直接移用到本方法。

## 1. 相对 BranchLocalRidge 改了什么

BranchLocalRidge 在固定特征几何上，根据 support 标签求核 ridge 头，optimizer 不更新表示。其优势是适应头有明确的正则化目标和线性系统解；它本身没有为“先适应旧类、再继承注册新类”提供分数保护条件。

MarginJoint 保留这条核适应路线，增加以下结构。

| 部分 | 当前变化 | 作用与边界 |
| --- | --- | --- |
| 表示 | 固定 Phase1，更新五个 received feature 分支上的小 adapter | 学习当前 support 的几何，禁止 encoder 反向和源样本访问 |
| B 头 | 每个当前 adapter 下重新求带自由类截距的 Affine ridge 头 | B 不使用 margin QP；外层 CE 可通过解析头更新 adapter |
| C 先验 | 固定当次 B 的 adapter 和分类函数，旧列映射到真实类序，新列补零 | C 的 adapter 从实际 B 起步，但 C 更新不改变这个 prior |
| C 头 | 在全物理旧、新 support 上求带 margin 不等式的自由截距 ridge QP | 保护旧训练点的最差真实类 gap，允许其分数变化 |
| 外层 | 跨 inner-fold 合并每类 CE，再求类 RMS；在白化坐标硬球内更新 | CE-only，不加 teacher keep loss、梯度 guard 或 `.5||Z||²` 近端项 |

“联合”指 adapter 改变核后，头随之重新求解，外层梯度完整穿过这个解。头的内层目标是平方误差加 RKHS 正则，adapter 的外层目标是 RMS CE；两者组成有明确内外层的优化问题，并非把头当作固定分类器微调，也不是将所有参数直接放进同一个 CE 优化器。

核保留原始几何的一半与当前适配几何的一半。正带宽分支为

\[
k_U(x,t)=\gamma\exp\!\left[-\frac{d_0(x,t)+d_U(x,t)}{2\tau}\right].
\]

这里的 \(d_0,d_U\) 是完整 interaction 平方距离。\(\tau,\gamma\) 来自既定原旧类参考统计，在相应 head 中固定，不随 C 更新重新选取。B 的核中心参考是既定物理集合；C margin 头使用真正 raw 核，没有将旧参考中心化额外变成保护约束。实现定位：[冻结定义与核前向](../code/cvsrffi/d92_margin_joint_local_ridge.py)，`FROZEN_CONFIG`、`_forward`。

## 2. B→C 继承的究竟是什么

令 B 的实际输出为 \(f_B(x)\)。C 注册全部 \(C\) 类后固定

\[
M(x)=\operatorname{pad}_{\rm old\to all} f_B(x),\qquad
f_C(x)=M(x)+r_U(x).
\]

Padding 只把旧列按真实类序放入 C，将新增列补零；不重新校准 B，也不把 B 的分数再加第二次。C 的 anchor 为实际 \(U_B\)，新的 adapter 位移写成 \(U=U_B+ZW^\top\)。尽管 \(U\) 改变，\(M\) 始终使用冻结的 \(U_B\) 与 B 头。

这项继承有两层职责。

- Final C 使用同 run、row、split 的 full-support B 实际函数。源码核对旧物理 ID、标签、原特征和类序，不能加载历史 run 的目标适应状态。
- Inner C 固定实际 \(U_B\)，只在该 fold 的 old inner-train 上重求旧解析头，用它定义该 fold 的 prior。不能用直接拟合过 inner-held 的 full-support B 头代替。这不使 inner-held 成为独立验证集：\(U_B\) 已通过合法 support 适应，C 的 inner-held 标签也参与外层 adapter 训练。

源码定位：`prepare_margin_joint_training` 的继承核对和 `make` 内的 prior 构造，见[联合核心](../code/cvsrffi/d92_margin_joint_local_ridge.py)。没有新增类时，`fit_margin_joint_local_ridge` 直接返回 B 原对象；new0 不进入 C QP。重新拟合一个相似的头不等于这个对象和函数的精确复用。

三阶段报告中的 A 应是未适应的原冻结地面分类器。用旧目标 support 拟合的 R0/B0 仍属于目标适应，不能冒充 A。B−A、B−C_old 必须在同一物理旧 query 上配对；这只是报告定义，不给训练器访问 query truth 的权限。

## 3. C 的约束直接联系最终全类 margin

设当前 head 的合法训练集为 \(T=O\cup N\)，其中 \(O\) 是旧类 train support，\(N\) 是新类 train support。Held support 不加入该 head 的拟合或 margin 约束。标签用全部注册列的中心化 one-hot：

\[
Y_{ic}=\mathbf1[y_i=c]-1/C.
\]

对每个旧训练点，先从固定 prior 得到

\[
d_i=\min_{j\ne y_i}\{M_{i,y_i}-M_{i,j}\}.
\]

竞争列包括新增的补零列。\(d_i\) 保留原值，负值不 clamp 到零，也不加入新 margin 系数。C 求解

\[
\begin{aligned}
\min_{g,b}\quad &
\frac12\|M_T+g(T)+\mathbf1_n b-Y_T\|_F^2
+\frac12\sum_{c=1}^{C}\|g_c\|_{\mathcal H_{k_U}}^2,\\
\text{s.t.}\quad &
M_{i,y_i}+g_{y_i}(x_i)+b_{y_i}
-M_{i,j}-g_j(x_i)-b_j\ge d_i,\\
&i\in O,\quad j\ne y_i.
\end{aligned}
\]

\(g\) 是有 RKHS 范数惩罚的核残差，\(b\) 是不惩罚的自由类截距。所有物理旧、新行都保留平方误差权重；每个旧行与全部竞争类的约束都保留。

这一定义给出几项明确性质。

1. **零残差始终可行。** 取 \(g=0,b=0\)，约束由 \(d_i\) 的定义成立。B 错误、负 margin、重复输入或零核不会使数学问题失去这个可行点。求解器技术失败不能被称为理论不可行。
2. **保护的是原最小 gap。** 对可行解，最终最小 margin 至少为 \(d_i\)。若 \(d_i>0\)，旧训练点真实类严格胜过所有注册类；若 \(d_i=0\)，只保证真实类属于最大分数集合，列序 tie-break 可能选别类；若 \(d_i<0\)，不保证正确，也不逐对保留 B 原来的所有 margin。
3. **允许旧分数变化。** 同一固定核、prior、标签和正则下，Conditional 的 \(r(O)=0\) 可行函数都满足这里的不等式。Margin 因而释放部分分数方向，包括某些可能修正 B 错误的方向；最优解是否选择它们没有保证。这项可行集包含关系只比较固定核的正则化 head 目标，不能推出 CE 或准确率单调。

数值解只能依据真实违反量说明保护。若最大约束违反量为 \(\varepsilon\)，可宣称的下界是 \(d_i-\varepsilon\)，不是理论精确值；\(d_i>\varepsilon\) 才足以推出严格胜出。实现会保存可行性、stationarity、乘子非负性、互补性和 primal-dual gap，见[QP 头](../code/cvsrffi/d92_margin_qp_head.py)的 `fit_margin_qp_head`。

## 4. 自由截距、QP 与完整 KKT 导数

固定当前 \(U\)，令 \(K=k_U(T,T)\)、\(L=k_U(H,T)\)、\(R=Y_T-M_T\)。半参数核展开给出

\[
g(x)=k_U(x,T)\alpha,
\qquad f_C(x)=M(x)+k_U(x,T)\alpha+b.
\]

令稀疏差分算子 \(\mathsf D\) 对每个旧行取真实类减竞争类，约束数为 \(q=|O|(C-1)\)。采用非负乘子 \(\mu\)，定义

\[
A=I+K,\quad z=A^{-1}\mathbf1,\quad s=\mathbf1^\top z,
\quad J=A^{-1}-zz^\top/s,\quad P=I-J,
\quad V=\operatorname{unvec}(\mathsf D^\top\mu).
\]

给定乘子后的系数和截距满足

\[
\begin{bmatrix}A&\mathbf1\\\mathbf1^\top&0\end{bmatrix}
\begin{bmatrix}\alpha\\b\end{bmatrix}
=\begin{bmatrix}R+V\\0\end{bmatrix},
\qquad \alpha=J(R+V),\quad b=z^\top(R+V)/s.
\]

因此截距依赖最优乘子，不能先求无约束截距再将它冻结。C 的完整求解不是一次无约束 ridge：它还要找到满足全部不等式与互补条件的 \(\mu\)。若重复 \(d_i\) 得到向量 \(\delta\)，则

\[
s_0=\mathsf D\operatorname{vec}(M_T+PR)-\delta,\quad
Q=\mathsf D(I_C\otimes P)\mathsf D^\top,
\quad \min_{\mu\ge0}\ \tfrac12\mu^\top Q\mu+s_0^\top\mu.
\]

完整 KKT 要求 \(\mu\ge0\)、\(s_0+Q\mu\ge0\)、\(\mu\odot(s_0+Q\mu)=0\)，并满足上面的截距和 stationarity 方程。当前实现用紧凑旧响应和工作集求解，不预先物化完整 Kronecker 或整个 dual Hessian；它仍必须检查全部物理约束并记录真实工作。

外层 CE 经过这个最优解求导。在 active 集 \(\mathcal I\) 独立、\(Q_{\mathcal I\mathcal I}\) 正定且严格互补的局部区域，令 \(G=\partial\mathcal R/\partial f_H\)。先解

\[
\begin{bmatrix}A&\mathbf1\\\mathbf1^\top&0\end{bmatrix}
\begin{bmatrix}T_G\\t_G\end{bmatrix}
=\begin{bmatrix}L^\top G\\\mathbf1_h^\top G\end{bmatrix},
\quad \eta=Q_{\mathcal I\mathcal I}^{-1}
\mathsf D_{\mathcal I}\operatorname{vec}(T_G),
\quad W_\eta=J\operatorname{unvec}(\mathsf D_{\mathcal I}^\top\eta).
\]

冻结 B、标签和下界后，完整核梯度为

\[
\overline L=G\alpha^\top,
\qquad \overline K=-\operatorname{sym}\big[(T_G+W_\eta)\alpha^\top\big].
\]

\(W_\eta\) 体现乘子随核变化的响应；detach 乘子会漏掉它。\(g_b=\mathbf1_h^\top G\) 是自由截距上游；每行 CE 梯度的类列和为零，不代表每一列跨行求和为零，所以不能删除 \(g_b\)。源码对应 `margin_qp_head_vjp`。

这些梯度继续覆盖 train Gram 的 old-old、old-new、new-new 块和 held-to-all-train 块，再穿过距离与 adapter，最后得到 \(\overline Z=\overline U W\)。旧端点没有 detach。源码对应联合核心的 `_backward`；C 没有调用 B 的中心化 VJP 替代 raw margin 核的导数。

固定核的 primal 函数解唯一，不代表处处有唯一光滑 Jacobian。即使 \(K\) 奇异，自由截距的有限 feature primal Hessian 仍满足

\[
\|\Phi v+\mathbf1 c\|^2+\|v\|^2>0
\quad\text{对任意非零 }(v,c),
\]

所以最优函数与截距唯一；核系数、dual 乘子却可能不唯一。Active 切换、冗余紧约束或零乘子可能使普通隐式导数不适用。当前实现对此明确技术失败，保留已消耗工作，不用伪逆、jitter、无约束梯度或零梯度掩盖。

## 5. 小 adapter 与白化坐标的精确含义

固定 DCT8 字典生成每个合法 support 的向量 \(h_i\in\mathbb R^8\)。令 \(H\) 按行堆叠这些向量，保留其数值有效子空间：

\[
H/\sqrt N=P_H\Sigma_H R_H^\top,\qquad
W=R_{H,\rm retained}\Sigma_{H,\rm retained}^{-1},
\qquad U=U_{\rm anchor}+ZW^\top.
\]

有效秩 \(r\le8\)，optimizer 坐标为 \(Z\in\mathbb R^{736\times r}\)，最多 5888 个标量。B 的 anchor 为零，C 的 anchor 为实际 \(U_B\)；不覆盖的字典零空间分量继续保留在 anchor 中。

白化带来的等式是

\[
\frac1N\sum_{i=1}^{N}\|(U-U_{\rm anchor})h_i\|^2=\|Z\|_F^2.
\]

这控制当前 support 字典上的平均预切向位移。它不是 Fisher 白化、每个样本的分类 margin 界或未知 query 的位移保证。位移再经过分支切向投影、固定 \(\kappa=1/4\) 的有界映射和范数保持归一化；最终 feature 位移与该等式中的预切向位移不同。数学 helper 为[函数坐标核心](../code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py)的 `latent_coordinates`、`reconstruct_U`、`_adapt`，本方法只复用这些 helper，不继承该旧模块的 keep/proximal 训练目标。

有合法 inner-held 时，跨全部 fold 合并类别 \(c\) 的平均 CE 为 \(\ell_c\)，外层求

\[
\min_{\|Z\|_F\le1/2}\mathcal R(Z),\qquad
\mathcal R(Z)=\sqrt{\frac1C\sum_c\ell_c(Z)^2}.
\]

这会给较高平均 CE 的类更大梯度权重，但高 CE 不等于准确率最低；降低 RMS 也不保证每类 CE 都下降。Inner-held 标签用于优化，所以这是训练监督，不是独立泛化证据。实现至多进行 4 次更新、每次 12 次固定回溯试探，初始步长 0.125；Armijo 使用球投影后的真实位移，返回最后接受状态。接受只说明当前训练 RMS CE 在容差内不升，不保证全局最优或每次都能接受。源码对应 `evaluate_margin_joint_objective` 与 `fit_margin_joint_local_ridge`。

## 6. 论文依据与项目推导各负责什么

| 一手来源 | 原有内容 | 本项目的使用及新增部分 |
| --- | --- | --- |
| [Bertinetto 等，R2D2，2019](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)，§3.2 | 可对闭式 ridge 适应解求导，使表示学习与解析头耦合 | 支撑 B 的可微解析适应；本项目固定 Phase1，只更新当前合法 support adapter，不复现其跨 episode 骨干 meta-training，也不采用其可学习 ridge/输出校准机制 |
| [Schölkopf、Herbrich 与 Smola，2001](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)，Theorem 1、2 | RKHS 正则下的有限核展开与有限参数部分 | 支撑“核残差加自由截距”的有限表示；实际 B padding、全注册 margin 下界、内外层目标及完整伴随由本项目推导 |
| [Boyd 与 Vandenberghe，Convex Optimization](https://www.seas.ucla.edu/~vandenbe/cvxbook/bv_cvxbook.pdf)，§4.4、§5.5、§10.1 | 凸 QP、对偶、KKT 和求解的一般理论 | 支撑固定核问题的凸性与证书；不提供本项目的 query 保护或性能结论 |
| [Amos 与 Kolter，OptNet，2017](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf)，§3，式（3）至（8） | 通过优化问题的 KKT 条件对最优解隐式微分 | 支撑 C 的可微 QP 思路；这里含完整自由截距与乘子响应的 kernel VJP 是项目针对该目标的推导，不能照搬其效率结论 |
| [Lopez-Paz 与 Ranzato，GEM，2017](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf)，§3 | 用局部线性 loss 约束投影梯度，以控制旧记忆样本行为 | 只提供保护思想的对照。当前方法每次求头后直接约束完整分类函数的 margin，不是 GEM 的梯度半空间 QP，也没有源记忆 replay |

前序 Conditional 的 GPML/Schur 补依据解释其等式条件核和自由常数修正；Margin 的约束集合已经改变，不能沿用“旧点 residual 恒为零”的证明。上述文献提供理论工具，没有任何一篇直接提出并验证本项目整条目标-support 联合流程，也没有替项目证明理想准确率目标。

## 7. 保护、退化与计算的实际边界

旧 margin 保护只覆盖当前 head 的 \(O\)。它不自动延伸到未见 query。对纯数学论证中的真实类 \(y\)，若 \(\mathcal N_g^2=\sum_c\|g_c\|_{\mathcal H_k}^2\)，再生性质只给出充分界

\[
m_{f,y}(x)\ge m_{M,y}(x)
-\max_{j\ne y}|b_y-b_j|-
\sqrt{2k(x,x)}\,\mathcal N_g.
\]

右端严格为正才足以保证该点正确。自由截距未受 RKHS 范数惩罚；当 raw Gaussian cross kernel 在远处趋于零时，残差趋于 \(b\)，一般不趋于零。保护有限旧 support 的 margin，因此不能宣称 query 零遗忘、新旧准确率接近或固定幅度的适应提升。Query 仍逐样本面对全部注册列，不能根据旧/新真实 role 删列、用类别配额重排或反馈选模型。

退化分支也有明确含义。

- K1、rank0、无合法 inner-held 或没有连续核信息时不更新 adapter，仍求完整最终头；rank0 不等于 raw 核为零。
- \(\tau=0\) 使用既定原完整 interaction 输入的精确等价核，保留每个物理行和全部约束；它与零核不同，没有普通 Gaussian adapter 导数。
- \(\gamma\) 缺失或为零时 C 仍求受约束的自由常数头，不能沿用 Conditional 的“残差必为零”。
- new0 按上层定义直接复用实际 B；它不是重新求 C 得到的定理。技术失败和外层正常试探预算耗尽分别记录，不按低分选择重跑。

小 adapter 减少的是直接优化的坐标数，不是全部计算量。每次前向仍涉及核距离、\(n\times n\) affine 分解、真实 RHS solve、全部 margin slack 扫描以及工作集分解；反向即使复用因子，也有 affine、active 和乘子响应 solve。Prior 准备、被拒 trial、最终头、完整数值档案和独立数学分析都有成本，不能只计接受的更新。

仅作矩阵规模说明：旧 6 类、新 20 类、每类 20 条时，\(n=520,m=120,C=26,q=3000\)。若物化完整 float64 dual Hessian，仅该矩阵就是 72,000,000 字节；当前实现避免这种完整预分配，但不能由此推出实际工作集或进程内存很小。头的 \(\alpha,b\) 也是必要状态，须与 optimizer 坐标分开计数。

星载计算是否降低，必须比较实际训练和推理耗时、峰值 RAM/VRAM、常驻状态、完整部署包和新增传输字节，并注明硬件与口径。Factor buffer 上限不是进程 RSS，工作代理不是实际 FLOP；未测量的能耗或设备指标为 N/A。SFT、低秩或少量可训练参数都不能单独证明省算力。

已有人工小矩阵证书为公式和数值正确性提供证据；本说明未重新运行测试，也不将证书当成生产求解器在所有输入上的成功保证或真实性能验证。用户关注的适应提升、注册后旧类下降和新旧差距，最终仍需按同物理 query 的 A/B/C 三阶段独立评分报告，不能由这些数学性质猜补。
