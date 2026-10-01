# ConditionalJointLocalRidge：数学原理与论文映射

日期：2026-10-01。对象：已冻结的 `D92-ConditionalJointLocalRidge-v1`。

该方法在合法 support 上联合更新低维 adapter，并对每个当前 adapter 重新求解析 LocalRidge 头。B 阶段学习旧类；C 阶段继承当次实际 B 函数，用等式约束限制注册残差。数学依据来自可微闭式求解、RKHS 有限展开和带常数项的核投影；RMS CE、完整旧点约束与当前 adapter 坐标是项目自己的组合和推导，不能把它们的效果归给原论文。

本文只解释[冻结数学设计](D92_POST_AFFINE_JOINT_MATH_DESIGN_20261001.md)第 3 至 8 节和[核心源码](../code/cvsrffi/d92_conditional_joint_local_ridge.py)。不提出新变体、参数选择或实验结论，也不引用真实评分、训练产物或运行状态。

## 1. 逐项映射

| 来源与原文位置 | 借用的数学原理 | 项目中的方程与作用 | 项目改造及原论文没有证明的内容 |
| --- | --- | --- | --- |
| Bertinetto 等，2019，[R2D2](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)，§3.2，式（3）至（5） | ridge 头可闭式求解，损失可穿过求解器对表示求导；对偶表达使用样本 Gram | B 使用 Affine ridge；C 使用下文的条件 ridge。CE 的梯度经过解析头、核距离和 adapter，避免把 head 当成固定分类器 | 当前目标域 support 适应不是其跨 episode 的骨干 meta-training 复现；未借用其可学习 ridge 系数、输出 scale/bias 校准，也未获得其实验性能保证 |
| Schölkopf、Herbrich 与 Smola，2001，[Generalized Representer Theorem](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)，Theorem 1、2 | RKHS 范数正则化下的有限核展开；允许有限参数部分与不惩罚的常数基函数，硬约束可进入有限点代价 | 将 C 的每列 residual 写成 O/N 上的核展开与一个自由常数，得到式（2）、（3） | 实际 B padding、对全部注册列施加旧点零残差，以及完整伴随公式由项目推导；定理不证明分类准确率或 adapter 外层目标的全局最优 |
| Rasmussen 与 Williams，2006，[GPML](https://gaussianprocess.org/gpml/chapters/RW2.pdf)，§2.2 式（2.19）、§2.7 式（2.41）、（2.42） | 消去已约束点产生 Schur 补；显式基函数的消元产生额外非负修正项，宽先验极限须用有限解析式 | 条件核包含 Schur 补和自由常数对应的 rank-one 项，见式（3） | 这里求确定性的等式约束 ridge，不建立随机函数后验，不输出可信区间；RMS CE 分类器不是 Bayesian posterior，GPML 也未证明其 query 零遗忘 |
| Lopez-Paz 与 Ranzato，2017，[GEM](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf)，§3，式（6）至（8） | 显式控制过去样本上的行为；GEM 将记忆 loss 不增约束在局部线性假设下转成梯度内积约束，再投影梯度 | 只用于说明保护对象的区别：本方法直接限制每次求解后的完整函数，而非使用 GEM 半空间或其 QP | `r(O)=0` 是项目的函数等式约束。它比平均 loss 不增更直接地固定旧锚点分数，也禁止这些点上的正向迁移；GEM 不提供该等式或有限步、任意 query 的保护定理 |
| 项目推导：[数学设计](D92_POST_AFFINE_JOINT_MATH_DESIGN_20261001.md)§3.2、§6 | 对各类平均 CE 求 RMS，并对完整解析头求导 | 式（1）给较高平均 CE 的类更大梯度权重；接受规则只检查当前训练目标 | 这不是上述论文共同证明的困难类准确率优化定理，也不是新增类别重校准 |
| 项目实现：[函数坐标](../code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py)中的 `latent_coordinates`、`reconstruct_U` | 当前 support 字典的保留子空间白化；只更新该子空间内的 adapter 位移 | `U=U_anchor+ZWᵀ`，optimizer 参数为 `736r`，`r≤8`；保留 LocalRidge 作为解析适应头 | DCT 字典、切向范数保持映射与该坐标组合属于项目实现。少量 optimizer 参数不证明总训练计算、部署内存或星载能耗较低 |

## 2. 可微闭式适应与 RMS CE

令 H 为各 inner fold 中留出的合法 support，`n_c` 为跨 fold 合并后类别 c 的 held 行数，`ℓ_c` 为这些行的平均 CE。H 的标签参与 adapter 更新，所以 H 是训练监督的一部分，不能称为独立验证集。head 的 inner-train 与 H 分开，只说明该次解析头没有直接拟合 H；它不使外层 optimizer 对 H 的依赖消失。

对有训练信息的情形，冻结实现使用：

\[
\begin{aligned}
\ell_c(Z)&=\frac{1}{n_c}\sum_{i\in H:y_i=c}
 \operatorname{CE}\!\left(f_{\mathrm{fold}(i)}(x_i;Z),c\right),
&\mathcal R(Z)&=\sqrt{\frac1C\sum_{c=1}^{C}\ell_c(Z)^2},\\
G_i&=\frac{\ell_{y_i}}{C\mathcal R n_{y_i}}
 \left[\operatorname{softmax}(f_i)-e_{y_i}\right],
&\min_{\|Z\|_F\le1/2}&\ \mathcal R(Z),\quad U=U_{\rm anchor}+ZW^\top .
\end{aligned}\tag{1}
\]

梯度式适用于 `𝓡>0`；零风险按实现的零梯度分支处理。类均值先合并全部 fold 的 sums/counts。权重随类平均 CE 增大而增大，但“高 CE 类”不等于“0–1 准确率最低的类”，CE 还反映预测置信程度。降低 RMS CE 不保证每类 CE 都下降，更不保证 argmax 正确率上升。

R2D2 的相关依据是可以穿过闭式 ridge 解反向求导。其 §3.1、§3.4 在多个 episode 上训练共享表示；本项目固定合规 Phase1 基座，只在当前合法 support 上更新 adapter，不能称为复现 R2D2 的 meta-training。[Bertinetto 等，2019](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)

实现的 `evaluate_conditional_joint_objective` 只使用 RMS CE，没有 `.5||Z||²` 近端项，`g_Z` 也不加 Z。白化使当前字典上平均预切向位移平方等于 `||Z||²`；它不是 Fisher 白化、分类 margin 距离或每个旧点的位移界。归一化梯度、球投影和既有 4×12 有限预算使用真实投影位移进行 Armijo 检查，返回最后接受状态。接受保证仅限当前训练 RMS CE 的数值不升，不能保证全局收敛、每次必能接受或独立泛化。

## 3. C 的保护对象是完整残差函数

记 O/N 为当前 C head 的旧/新 train support，C 为全部注册类别数。M 来自同一 run、row、物理 fold 的当次 B：inner C 固定实际 U_B，在对应旧 inner-train 上重求旧解析头；final C 使用 full-support B 的实际完整函数。旧列按真实类序映射，新列补零。M 的 adapter、解析头与分数在 C 中冻结；不能随 C 的当前 U 重算，也不能替换为其他 run 的适应 B。

每列 `g_c` 属于当前正核 `k_U` 的 RKHS，常数 `v_c` 不计入 RKHS 正则项。令 `R_N=Y_N−M_N`，其中 Y 为全部注册列的中心化 one-hot。去掉约束下恒定的旧点拟合误差后，C 求：

\[
\begin{aligned}
\min_{g,v}\quad &\frac12\|g(N)+\mathbf1_pv^\top-R_N\|_F^2
 +\frac12\sum_{c=1}^{C}\|g_c\|_{\mathcal H_{k_U}}^2,\\
\text{subject to}\quad &r(O)=g(O)+\mathbf1_mv^\top=0_{m\times C},
\qquad f_C(x)=M(x)+r(x).
\end{aligned}\tag{2}
\]

约束覆盖全部注册列，包括新列；只限制旧列、只令 `g(O)=0` 或只限制 adapter 的瞬时梯度，都不是式（2）。对每个当前 U 求解后，旧锚点的完整 C 分数等于 padding 后的 B 分数。inner fold 的 O 仅含旧 inner-train，旧 inner-held 仍可通过 CE 产生梯度；final head 的 O 才是当前全部旧 support。

半参数 representer theorem 支持“有限核展开加独立常数”的表达，硬约束可由有限点代价表示。该定理没有替项目证明式（2）的特定保护目标、C 的分类效果或下面的完整消元与梯度。[Schölkopf、Herbrich 与 Smola，2001，Theorem 1、2](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)

这里固定的是分数，而非修正 B 的错误。新增列补零后，决策保护应相对 padding 后的全注册 margin 判断；原来的 old-only 排名不足以推出该 margin 为正。旧锚点原本严格正确时可据正 margin 保持决策，错误与 ties 不得到修复保证。O 之外没有分数恒等式，旧 query 零遗忘不由此成立。

## 4. 自由常数为何必须保留 rank-one 项

正核分支令 `A=k_U(O,O)`、`z=A⁻¹1_m`、`s=1_mᵀz>0`、`c(x)=1−k_U(x,O)z`；旧 Gram A 必须满足当前固定数值可解条件。受式（2）约束的核和部署表达为：

\[
\begin{aligned}
k_\perp(x,t)&=k_U(x,t)-k_U(x,O)A^{-1}k_U(O,t)
 +\frac{c(x)c(t)}{s},\\
K_\perp&=k_\perp(N,N),\qquad
\alpha=(K_\perp+I_p)^{-1}R_N,\\
v^\top&=c_N^\top\alpha/s,\qquad
\beta=A^{-1}k_U(O,N)\alpha+zv^\top,\\
r(x)&=k_U(x,N)\alpha-k_U(x,O)\beta+v^\top.
\end{aligned}\tag{3}
\]

前两项为 Schur 补，最后一项是自由常数消元产生的正半定 rank-one 修正；三者共同使 `k_perp(O,x)=0`。删掉最后一项会求得另一问题，求完后再随意加截距也会破坏旧点约束。GPML 的无噪声条件化和显式基函数宽先验极限提供相同的线性代数结构；项目直接使用有限解析式，不通过给常数设置极大方差模拟它。[GPML §2.2、§2.7](https://gaussianprocess.org/gpml/chapters/RW2.pdf)

式（3）在本项目中是确定性受约束 ridge 的解，不是对未知分类标签的 Bayesian posterior。R_N 和 M 均冻结；正 τ 时 k_U 是原始及适配 interaction 距离之和对应的 Gaussian 核，τ/γ 沿用原旧参考统计，不在 C 中重新选取。投影 A 使用真正 raw Gaussian Gram，不能用减去常数的 `expm1` 表示冒充。

[解析核源码](../code/cvsrffi/d92_conditional_affine_kernel.py)的 `conditional_affine_adjoint` 对旧 Gram、old/new cross、new Gram 和 held 两组 cross 都回传。反向有 residual 与 projection 两个系统，并保留常数上游 `g_b=sum_rows(G)`。CE 每行的 class sum 为零，不意味着逐列跨行总和为零；删掉该上游或 detach 旧投影都不再是式（2）的完整梯度。

GEM 原文用局部线性假设将旧记忆 loss 不增变成梯度内积约束。本方法没有使用其梯度 QP；它在每次闭式 head 求解中直接执行函数等式。这一改造固定旧锚点行为，也放弃这些点上的正向迁移，不能借 GEM 的论述声称全域遗忘得到控制。[Lopez-Paz 与 Ranzato，2017，§3](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf)

## 5. 权限、数值与计算边界

| 范围 | 可作出的结论 | 必须保留的边界 |
| --- | --- | --- |
| 输入与继承 | B/C 使用当前授权 support 及其合法监督标签；C prior 绑定当次实际 B | 拟合不使用 query、源域逐样本数据、query truth/role、真实类别配额或跨 run 的目标适应状态。query 逐样本面对全部注册类，预测固定后才由独立 scorer 连接 truth |
| 旧点保护 | 实数算术下，对当前 O 的全部 C 列有 `r(O)=0` | 实现须记录实际残差而非写理论零。O 外只存在条件核范数导出的 margin 充分界；远离全部 O/N 时 residual 可趋于非零 v，不能宣称 query 零遗忘 |
| 容量与退化 | new0 精确复用 B；零核有旧锚点时常数也被约束为零；K1/rank0 保留完整解析 head | K1 无 inner CE 更新；新旧核输入完全重合时不能通过 residual 改写受保护点的新标签。τ0 分组与旧约束压缩只接受完整核输入的精确等价，不按标签或近似距离删点 |
| 数值与证书 | 固定机器精度残差、形状及有限性检查限定可接受数值解；A 可解时采用无 ridge 投影 | 病态 A 明确失败，不加 jitter、改变带宽或改软约束。[独立 KKT helper](../tools/d92_conditional_analysis_math.py)的 rebuild 与 residual-only certificate 语义不同；后者不求谱，`positive_kernel_checked=False`，不能冒充独立 PSD 确认 |
| 训练计算 | adapter 的 optimizer 参数为 `736r`；α、β、v 为另列的闭式 head 状态 | 每个正核 C forward 有 m×m 投影和 p×p residual 两次 Cholesky，另有三次谱诊断；一次完整反向有两个 C 列 RHS solve、四次 triangular 调用。准备、actual B prior、距离、拒绝 trial 与缓存仍计费 |
| 独立分析与部署 | raw-kernel 展开可部署，部署无需重解投影 | 完整不定 KKT oracle 的矩阵阶数为 `m+p+1`，一般 dense solve 有立方主项，是额外独立核验成本。部署仍需 O/N 几何、系数、adapter 和 actual B prior；B/C mapping 不同时须各自计算 |

低维 adapter 保留 LocalRidge 的解析适应优先级，减少直接训练骨干和迭代 head 的需求。它不消除核矩阵、完整 VJP 或固定特征提取成本，参数数目和 `n²×RHS` 工作代理也不能替代硬件时间、峰值 RAM/VRAM、常驻状态、能耗与实际传输测量。本文不作星载省算力或性能达标结论。

实现与独立推导的详细接口分别见[条件核说明](D92_CONDITIONAL_AFFINE_KERNEL_IMPLEMENTATION_20261001.md)、[独立数学证书](D92_CONDITIONAL_MATH_CERTIFICATE_20261001.md)和[独立分析设计](D92_CONDITIONAL_JOINT_ANALYSIS_20261001.md)。这些链接用于定位方程和实现职责，不将其中的运行记录作为本文的论文或性能依据。
