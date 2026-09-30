# AJLR 分数尺度与 CE 数学审计

日期：2026-10-01。结论：平方标签目标的分数单位与 softmax 的置信度单位不同；共享正 scalar 在固定分类状态下不改善准确率，却可能改变联合训练梯度与近端折中。**这说明需要核对实际分数、margin 和梯度，不能证明 temperature=1 是当前性能原因，也不构成新候选。**

本审计仅依据[当前 AJLR 核心](../code/cvsrffi/d92_anchor_joint_local_ridge.py)与[冻结数学设计](D92_JOINT_AFTER_FCR8_DESIGN_20261001.md)第 4、5 节，以及下文两篇原始论文。未读取任何本项目 query、ABC、历史成绩、总索引、交接或本轮外层结果；未拟合、测试、修改算法或设定候选参数。

## 1. 当前目标与理想标签分数

注册类数记为 \(C\ge2\)，真实类为 \(y\)。AJLR 的 closed residual ridge 使用

\[
Y_{ij}=\mathbf1[y_i=j]-1/C,\quad \alpha=(K+I)^{-1}(Y-M),\quad s(x)=m(x)+k(x,S)\alpha.
\]

CE 使用自然对数与 temperature=1；跨折每类 mean CE 为 \(\ell_c\)，目标为 \(J=R+\tfrac12\|Z\|_F^2\)，其中 \(R=\sqrt{\sum_c\ell_c^2/C}\)。下文记录权重只用于有监督记录的类，即 \(n_c>0\)。M 是 C 阶段固定实际 B 函数的旧列与新列零 padding；不是概率标签，也不再次对 \(Y-M\) 做样本均值中心化。

假设某记录的分数**恰好等于** centered onehot：\(s_y=1-1/C\)，其余 \(s_j=-1/C\)。公共位移 \(-1/C\) 从 softmax 中消去，所以

\[
p_y=\frac{e}{e+C-1},\qquad \ell=-\log p_y=\log[1+(C-1)e^{-1}].
\]

所有错误类与真实类的 margin 都为 1，argmax 完全正确；但随 C 增加，\(p_y\) 严格下降、CE 严格上升，且 \(p_y\sim e/C\)、\(\ell\sim\log C-1\)。这是理想分数下的精确结论，不能外推为实际注册后的分数固定不变。C=1 时概率为 1、CE 为 0，没有错误类 margin。

该例的分数梯度也不趋零：\(\|p-e_y\|_2=\sqrt{C(C-1)}/(e+C-1)\to1\)。因此“概率低/CE 高”本身不能证明 CE 梯度弱；从分数到 Z 的 Jacobian、closed solve 和当前近端才决定实际 adapter 梯度。

**实际 ridge/held score 没有由 onehot 目标自动得到的逐元素 ±1 上界。** PSD 只给出 \(\|(K+I)^{-1}\|_2\le1\)、\(\|\alpha\|_F\le\|Y-M\|_F\)；它不使逆解成为逐元素凸组合，也不把 \(m(x)+k(x,S)\alpha\) 的 held 分数限制在标签范围。实际 prior、核尺度、交叉核和参考中心均参与分数。不得用理想标签值代替真实归档分数。

## 2. 固定正确 margin 的概率关系

对任意有限分数定义 \(\Delta_j=s_y-s_j\)，则无须假设分数幅度：

\[
p_y=\frac1{1+\sum_{j\ne y}e^{-\Delta_j}},\qquad
\ell=\log\left(1+\sum_{j\ne y}e^{-\Delta_j}\right).
\]

若所有 \(\Delta_j=d\)，精确为 \(p_y=[1+(C-1)e^{-d}]^{-1}\)。若只知**实际最小 margin** \(d=\min_{j\ne y}\Delta_j\)，至少一项达到 d，故

\[
\frac1{1+(C-1)e^{-d}}\le p_y\le\frac1{1+e^{-d}},\qquad
\log(1+e^{-d})\le\ell\le\log[1+(C-1)e^{-d}].
\]

仅知 \(\Delta_j\ge d_0\) 时，只有上述概率下界与 CE 上界可以将 d 换成 \(d_0\)；概率上界不能这样换。d>0 表示真实类严格胜出，d=0 有并列，d<0 有竞争类胜出。固定 margin 下的类数效应与实际注册造成的函数改变必须分别记录。

## 3. 共享正 scalar：决策不变与导数

设 a>0 在同一样本全部注册类列上共用，\(\widetilde s=a s\)。对固定 U、prior、残差头、输入和 class registry，\(\operatorname{argmax}\widetilde s=\operatorname{argmax}s\)，精确 tie 也保持。故同一冻结预测集合上的 A/B/C 旧、新准确率及其 H 不变；缺失 A 仍为 N/A。这是代数定理，不需要读取实际评分。

共同加 \(b\mathbf1\) 同样不改变 softmax 或 argmax。a=0 会制造全 tie，a<0 可能反转排序；不同类别 scale/bias 不满足本定理。仅缩放 residual 而保留 prior，或把缩放后的 B 再送入 C 的 \(Y-M\) 求解，会改变函数构造，不能援引上述固定状态结论。有限精度溢出、下溢或舍入 tie 也不属于精确代数保证。

固定 s，令 \(p=\operatorname{softmax}(a s)\)、\(B_p=\operatorname{diag}(p)-pp^T\)。单记录 CE 的导数为

\[
\ell_a=p^Ts-s_y=-\sum_{j\ne y}p_j\Delta_j,\qquad
\ell_{aa}=s^TB_ps=\operatorname{Var}_{p}(s_j)\ge0,
\]
\[
\nabla_s\ell=a(p-e_y),\qquad \nabla_s^2\ell=a^2B_p\succeq0.
\]

所以固定分数的 CE/NLL 对 a 凸；非恒定分数在有限 a 时严格凸。这不等于对 temperature \(T=1/a\) 或 log-scale 凸：若 \(u=\log a\)，则 \(\ell_{uu}=a^2\ell_{aa}+a\ell_a\)，符号不固定。

令 \(D_i=\partial s_i/\partial\operatorname{vec}Z\)。在闭式头与几何可微的区域，

\[
\nabla_Z\ell_i=aD_i^T(p_i-e_{y_i}),\quad
\nabla_Z^2\ell_i=a^2D_i^TB_{p_i}D_i+a\sum_j(p_{ij}-\mathbf1[j=y_i])\nabla_Z^2s_{ij}.
\]

第二项不保证 PSD，adapter 联合问题一般非凸。交叉导数为 \(\partial_a\nabla_Z\ell_i=D_i^T[p_i-e_{y_i}+aB_{p_i}s_i]\)；a 改变概率及不同竞争列的权重，不只是把原梯度乘常数。

AJLR 必须先跨 folds 汇总 class means。对 \(R_a>0\)，记录权重为 \(w_i=\ell_{y_i}(a)/(C R_a n_{y_i})\)，因此

\[
\partial_a R_a=\sum_iw_i\ell_{i,a},\qquad
\nabla_ZJ_a=a\sum_iw_iD_i^T(p_i-e_{y_i})+Z.
\]

固定 s 时各 \(\ell_c(a)\) 非负且凸，RMS 也凸；直接求导可得

\[
R_a''=\frac{\sum_c[(\ell_c')^2+\ell_c\ell_c'']}{C R_a}-\frac{(R_a')^2}{R_a}\ge0,
\]

其中最后不等式来自 Cauchy–Schwarz 与 \(\ell_c\ell_c''\ge0\)。若 R=0，须单独采用核心的零风险分支，不能除零。该一维性质不能证明联合 Z 优化凸。

当前 D 包含 residual closed solve、train/held 双端几何和移动参考中心的导数；C 的固定 prior 导数为 0，但完整 score 仍包含 prior。a 进入训练会改变概率、class weights、CE/prox 的相对方向及 Armijo 轨迹，可能改变最终 argmax，却不保证提高准确率，更不保证新旧类共享改善。

## 4. 无惩罚 scalar NLL 的边界与三种“尺度”

在固定分数的合法有标签 support 上，若每条记录真实类都严格最大，即全部 \(\Delta_{ij}>0\)，则每项 \(\ell_{i,a}<0\)，且 a→∞ 时 NLL→0。故无惩罚 scalar NLL 只有无穷远下确界，没有有限最优 scalar；当前 Z 近端在固定 Z 下是常数，不能阻止这个边界。

若真实类属于大小为 \(k_i\) 的最大分数组（计入真实类），则 a→∞ 时该记录 CE→\(\log k_i\)。若存在 \(s_{iy_i}<\max_js_{ij}\)，其 CE 随 a 至少线性发散，斜率趋于 \(\max_js_{ij}-s_{iy_i}>0\)。总 NLL 仍凸，但最优可能在有限 a，也可能在 a↓0 的均匀概率边界；恒定分数使 scalar 不可识别。是否可分、是否存在有限最优必须由实际分数确认，不能凭标签目标判断。

| 概念 | 改变什么 | 不可混同的边界 |
|---|---|---|
| 冻结状态的概率/logit 校准 | softmax 置信度、NLL；正共享 scalar 不改变决策 | support NLL 下降不证明独立概率校准或准确率提升 |
| 训练损失尺度 | scalar 改变 CE 的形状与 class 权重；单纯 CE 乘权重仅改变风险权重 | scalar 不等价于固定 loss weight 或只改学习率 |
| 函数坐标近端 | \(\tfrac12\|Z\|^2\) 约束当前 support 的 pre-tangent adapter 增量 | 不惩罚独立 output scalar；不是完整分类函数或概率的范数约束 |

当前 inner-held 标签参与 adapter 监督；若再用它们拟合 scalar，仍是合法 support 训练，不能改名为独立校准验证。信心提高、NLL 下降和经验泛化是不同命题。Guo 等在 §4.2、式 9 使用各类共享 T>0 的温度缩放，并指出固定模型准确率不变；其校准实验依赖验证集，不能据此宣称本项目的训练内标签形成独立校准证据。[原始论文](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)

## 5. R2-D2 支持的结论与本项目边界

Bertinetto 等的 §3.2 以 onehot 标签做 ridge，式 5 用 \(W=X^T(XX^T+\lambda I)^{-1}Y\)，式 6 写 \(\widehat Y=\alpha X'W+\beta\)，其中 α、β 为共享实 scalar；与 λ、CNN 参数一起由跨 episode 的 meta outer loop 学习。该结构支持“ridge regression output 接入 CE 时可以单独校准共享分数尺度”，并支持反传闭式求解；论文没有给出适用于 AJLR 的固定温度或 scale 值。α 的正性不是式 6 已声明的约束，本审计的 argmax 不变定理另要求 a>0。[ICLR 2019 原始论文，§3.2/式 3–6](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)

式 6 的公共 β 在本文这种多类 softmax 中数学上消去，不能解释成可学习旧/新偏置、逐类 intercept 或修复 AJLR 的旧参考均值限制。论文的 meta episode 训练机制也不等于当前冻结 Phase1、目标 support adapter 和实际 B prior 的顺序训练。

式 5 的小样本 N×N 求解给出相对特征维度大矩阵的复杂度理由；它不测量本项目反复 trial heads、先验推理、kernel/adapter、缓存和 NPZ 的总成本。论文结果不能证明 AJLR 的 support/query 性能或星载省算力。本审计的硬件时间、内存、部署包和传输均未测，全部为 N/A。

## 6. 只提出完整归档的诊断条件

| 需按阶段、C 与 fold/类别读取的已有量 | 可回答的问题 |
|---|---|
| C、每类 CE sums/counts/means、RMSCE、prox/total | 类数基线与监督风险如何同时变化；不得跨 C 只比较未说明的 CE 原值 |
| 真实 scores、labels、score norm/range、全部 \(\Delta_j\) 与最小 margin、tie/correctness | 分数是否接近理想目标、严格可分与否、置信变化是否伴随判决变化 |
| 初末 prior M、Z0 score、最终 inner score 的同记录差异 | 区分注册残差作用与后续 adapter/闭式头变化；缺失完整 Z0 final C head 保持 N/A，不补拟合 |
| \(g_Z\)、Z、\(g_{CE}=g_Z-Z\)、norm/dot/cosine、trial 接受与停止 | CE 信号经 D 后是否小、近端是否抵消、预算是否先耗尽；不以 CE 大小代替梯度证据 |

这些量属于监督训练机制证据，不能单独归因 temperature、证明 calibration 或替代独立结果。本任务不读取归档数值、不执行 scalar 优化、不给 grid/rank/LR/temperature 建议值，也不实现或冻结下一候选；决定等待 root 的完整独立结果。
