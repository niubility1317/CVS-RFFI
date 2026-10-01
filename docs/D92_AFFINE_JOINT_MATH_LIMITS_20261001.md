# 联合 LocalRidge 微调：可证明的范围与实验边界

日期：2026-10-01。本文补充当前 AJLR/AffineJoint 的数学解释。独立子 agent 仅阅读源码、已有数学审计和一手论文，没有读取真实评分、实验索引、训练快照或结果。本文不改变任何算法、参数、健康进程或预登记矩阵，不根据性能提出参数组合。

## 1. 微调直接优化解析分类器

当前路径是固定合规 Phase1 表示、训练合法目标 support 上的 adapter，再解析拟合 LocalRidge；监督损失沿解析解、核几何和 adapter 完整反传。它借鉴可微闭式求解，而没有重新进行地面跨任务 meta-training。[R2D2，§3.2](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf)。

对于当前残差头，B 的 prior 为零，C 的 prior 来自同一路径合法 B 状态。Affine 的完整最优性条件为：

```text
min_(g,b) .5||M + g(X) + e b - Y||_F² + .5||g||_RKHS²
A = K + I, E = Y - M
A alpha + e b = E
eᵀ alpha = 0
```

这里 `e` 是训练样本数长度的全 1 列，`b` 是每类截距行向量。`K` 是实际使用的中心化、缩放核。自由截距的处理与均值函数思想相关，具体公式是本项目残差目标的推导。[GPML，第 2 章 §2.7](https://gaussianprocess.org/gpml/chapters/RW2.pdf)。

增加自由截距后，固定 U 下的最优正则平方目标不会增加；这不推出外层 RMSCE、旧类准确率、新类准确率或 H 改善。理论依据与真实收益分开报告。

## 2. 白化近端项控制平均位移

固定字典行矩阵 `H` 有 N 个当前 support 行。对保留奇异子空间，令：

```text
H / sqrt(N) = P Sigma Vᵀ
W = V_r Sigma_r⁻¹
Delta U = Z Wᵀ
H W = sqrt(N) P_r
||H Delta Uᵀ||_F² / N = ||Z||_F²
```

所以 `.5||Z||_F²` 精确对应归一化前 hidden readout 位移的经验平均平方罚项。它白化固定字典的二阶矩，不是 CE Fisher、最终分类函数距离或旧类独立保持约束。后续切向投影、限幅、分支归一化与核求解均不能省略。

若 C 使用全体当前 support 白化，旧 support 行数为 N_old，则只能直接推出：

```text
mean_old ||Delta U h||² <= (N / N_old) ||Z||_F²
```

等量 K 时，比例为注册总类数/旧类数。以矩阵中的 6 个旧类、20 个新类为例，比例为 `26/6`；这是代数上界示例，不是实测旧类退化的因果解释。全体均方控制不能写成旧类均方受到相同上界保护。

对任意单点还只有 `||Delta U h|| <= ||Z||_F ||Wᵀh||`。小保留奇异值可能放大单点杠杆值，因此平均控制也不是逐样本或分布外保证。

实现位置：[坐标白化](../code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py)、[Affine preparation 与近端目标](../code/cvsrffi/d92_affine_joint_local_ridge.py)。

## 3. B 函数继承不等于零遗忘

Final C 使用 `f_C(x) = pad(f_B(x)) + delta_C(x)`，并继承实际 `U_B`。Inner prior 则在冻结的实际 `U_B` 上只拟合同折旧 inner-train 头，不能混用其他 fold、model、run 或历史适应状态。

B 状态与函数分量不会被 C 覆写，但当前目标没有要求 `delta_C` 的旧类列为零。即使 `Z=0`，完整 residual solve 也可能改变旧类分数。

若 B 的正确旧类胜者 y 对其他旧类的最小 margin 为 `m_B(x)>0`，则：

```text
m_C,old(x) >= m_B(x) - 2||delta_old(x)||_infinity
```

因此 `2||delta_old(x)||_infinity < m_B(x)` 是保持旧类内部胜者的充分条件。面对新增类，还必须满足 `f_C,y(x) > f_C,j(x)`，对每个新增类 j 都成立。保持旧列内胜者不能保证全注册类竞争的预测不变。

当前 RMSCE 与 Armijo 控制总目标，不保证每类 CE 或准确率单调。上述 margin 条件没有被当前方法强制执行，不能将函数先验或近端项表述成用户要求的“注册后旧类下降不超过 1 个百分点”保证。

## 4. 截距必须参与完整隐式反向

令 held 核为 L、分数伴随为 G，截距伴随 `g_b=e_heldᵀG` 通常非零。每行 CE 梯度的类别和为零，不代表跨样本和为零。正确 companion 为：

```text
A T + e eta = Lᵀ G
eᵀ T = g_b
bar_K = -sym(T alphaᵀ)
bar_L = G alphaᵀ
```

之后继续通过完整中心化、移动参考和实际尺度/几何反传。把 `eᵀT` 置零或只把截距加到预测中，会改变声称优化的目标。从完整最优性条件进行隐式求导的原则也见 [Blondel 等，NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html)。

当前核心与数学审计已有独立 primal/saddle oracle、有限差分及非零截距伴随验证；本文是范围说明，不新增运行门槛或重复测试要求。详见[截距审计](D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md)和[核心说明](D92_AFFINE_JOINT_CORE_20261001.md)。

## 5. 少参数与低计算分别验证

活动 adapter 坐标为 `736r`，`r<=8`，最多 5,888 个。解析系数、截距、prior 状态和工作缓存另计，不能因为不是 SGD 参数就忽略。

每个有信息 head 仍计算距离/kernel、n×n 分解、n×C 系数和评分。Affine 前向两次三角求解各有 C+1 个 RHS，companion 两次各有 C 个 RHS。所有拒绝 trial、actual B prior 评分和最终 head 均计费。小 r 不消除 n×n 分解的 O(n³) 主项。

SFT 是监督微调方式，不自带省算力结论。实际训练/推理耗时、峰值内存、常驻状态、传输字节和硬件口径由完整运行报告提供；未测量项记 N/A。大型只读分析/诊断传输的资源与部署训练资源分开，不能互相代替。

后续继续报告完整 K×新增类数及 A/B/C。A 缺失仍为 N/A，不以已经适应过的 R0/B0 代替。support 留出诊断不写成 query 准确率或新增独立数据验证；数学证明也不写成未经实测的性能承诺。
