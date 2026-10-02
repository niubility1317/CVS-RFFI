# D92 SupportMetric 的 support 数学诊断

日期：2026-10-02。本文只分析冻结源码与 root 提供的完整合法 support 汇总。本文不读取 query、历史目标评分、真实输入或产物，不修改算法、配置和运行中的任务，也不授权启动下一实验。

## 1. 有限结论与证据范围

当前 SupportMetric 已在数学定义中消除了用坐标单位矩阵 `I5` 充当物理阻尼的错误：它使用实际基底 Gram 与预测 Fisher，求解物理度量球中的一步。这个结论与“完整解析方向已经得到严格有限精度认证”“分类性能应当上升”不同。后两项均不能由当前模块的残差证据或接受更新数量推出。

当前外层目标是 RMS 类均交叉熵（RMSCE），不是旧、新组准确率的调和均值 H。即便精确 RMSCE 下降，准确率与 H 仍可下降。C 又冻结实际 B 的旧条件分类函数，因而不能修正 B 的旧类内部排序；它只能通过新条件头与组 gate 改变跨组竞争。这是有用的继承约束，也是明确的表达能力边界。

本文优先提出一个下一步假说：**只把 gate 的逐物理样本 Bernoulli 监督质量改为旧、新组各占一半，组内再按合法 support 类均衡**。它针对内层 gate 的 `6:q` 支持组质量与两组 H 的对称评价方式之间的错位；它不改变旧支持 barrier、实际 B 继承、新类解析 Ridge、外层 RMSCE、物理度量或一步更新预算。现有汇总没有条件排序与跨组挤出分解，因此尚未证实该假说，也不能保证它提升 H 或 query 性能。

root 告知并从完整 parent 重新核实的 support 诊断如下。OOF 均排除 new0，覆盖含新类的 96 parent；本文没有独立读取这些产物。下表把 root 提供的小数比例换成百分数显示。

| 合法诊断与覆盖 | 路径 | B 旧准确率 | C 旧准确率 | C 新准确率 | H |
| --- | --- | ---: | ---: | ---: | ---: |
| OOF，含新类的 96 parent | R0 独立拟合 | 71.0416667% | 63.7673611% | 54.8463542% | 58.4167964% |
| OOF，含新类的 96 parent | SupportMetric 顺序继承 | 71.0416667% | 64.8871528% | 50.1822917% | 56.0010757% |
| one-shot proxy，含新类的 96 parent | R0 独立拟合 | N/A（本次未提供） | 44.3540854% | 34.1075932% | 36.2753074% |
| one-shot proxy，含新类的 96 parent | SupportMetric 顺序继承 | N/A（本次未提供） | 45.7015006% | 30.5817495% | 34.3046066% |

OOF 的 A 旧准确率两条路径均为 62.6388889%。root 另核实：R0 的平均注册后旧下降为 7.2743056 个百分点、平均新旧绝对差为 10.7404514 个百分点；候选分别为 6.1545139 个百分点与 15.4756944 个百分点。这些是同一含新类 96 parent 口径，不能替换为含 new0 的 120 parent 汇总。平均绝对差也不等于汇总组均值之差的绝对值。

root 同时告知：实际接受更新 648 次、拒绝 0 次、有效 rank 为 5。本文没有读逐步 loss、参数变化或逐样本预测，不能把这些汇总补写为未提供的测量。B 准确率相同不代表参数未更新，也不代表两条路径的正确样本集合完全相同。

这些 support 汇总未显示候选 H 优于 R0。OOF 汇总中旧、新两组的变化方向不同，但不能据此单独认定数值实现错误、泛化失败或某个组校准机制已被证实。parent 级 H 的平均值一般也不等于用汇总旧、新准确率重新计算的 H。OOF 与 one-shot proxy 均不是 query 性能证据。

本文核对的冻结来源是 [SupportMetric 核心](../code/cvsrffi/d92_support_metric_joint_local_ridge.py)、[度量子问题](../code/cvsrffi/d92_support_metric_step.py)、[原型 frame 与解析头原语](../code/cvsrffi/d92_proto_frame_primitives.py)、[GroupBarrier gate](../code/cvsrffi/d92_group_barrier_gate.py)、[坐标尺度数学稿](D92_PROTO_FRAME_GGN_COORDINATE_SCALE_NOTE_20261002.md)及[有限精度数学稿](D92_PROTO_FRAME_SUPPORT_METRIC_NUMERICS_NOTE_20261002.md)。

## 2. 物理度量问题解决到哪一层

### 2.1 当前定义与精确坐标不变性

设冻结地面 frame 经 exact stored-binary64 rank 构造得到物理基底 \(U\in\mathbb R^{160\times r}\)，\(r\le5\)。物理校正为 \(w=U\theta\)。这里的 exact rank 指存储后的二进制有理数矩阵，不是存储前真实原型或无限精度总体几何的 rank。

令 \(J_i\in\mathbb R^{C\times r}\) 是每个合法 OOF 物理样本的完整分类 score JVP；它包括解析 head、free intercept、gate 及其约束右端的依赖。\(p_i\) 是对应 softmax 概率，\(n_c\) 是合法 OOF 中类 \(c\) 的样本数。当前预测 Fisher 为

\[
 F=\sum_{i=1}^{N}\frac{1}{C n_{y_i}}
 J_i^\top\{\operatorname{diag}(p_i)-p_ip_i^\top\}J_i.
\]

它是概率中心化 JVP 的 Gram；实现不需要建立稠密 \(C\times C\) 协方差。所有计数只来自合法 support，不来自 query 类数、配额或全局统计。当前实际物理 Gram 是 \(B=U^\top U\)，不能自动置为 \(I_r\)。设 \(G\succeq0\) 是现有 RMSCE 的 GGN 曲率，则

\[
 M=B+F,\qquad H=G+M,
\]

当前一步定义为

\[
 \min_d\quad g^\top d+\tfrac12d^\top H d
 \quad\text{s.t.}\quad d^\top M d\le\tfrac14.
 \tag{1}
\]

注意：\(M\) 是这一步模型的阻尼与信赖度量，不是额外加入实际 RMSCE 的 proximal loss。核心传入原语的 `curvature`，没有把旧原语的 `I5+curvature` 再作为 \(G\) 使用。

对于任意可逆重参数化 \(U'=UT\)、\(\theta'=T^{-1}\theta\)，精确计算下

\[
 J'_i=J_iT,\quad g'=T^\top g,\quad
 (B',F',G',M',H')=T^\top(B,F,G,M,H)T.
\]

故 \(d'=T^{-1}d\)，目标值、球约束及物理方向 \(U'd'=Ud\) 相同。旧坐标 \(I_5\) 阻尼不具备这个一般合同变换规律。Fisher/GGN、参数化与阻尼的联系有一手理论背景，但本文具体的 \(U^\top U+F\)、exact-rank basis、半径与预算是本项目定义，不是文献性能定理。[Martens, 2020](https://jmlr.org/papers/v21/17-678.html)

这类坐标变换没有改变物理校正，也没有改变 kernel，本身不能带来识别收益。另一种不同的恒等变换是对完整归一化特征做共同正缩放再归一化；只缩放某个子空间通常会改变其相对正交补的几何，不能把两者混为 kernel 不变性。

### 2.2 有限精度和实际接受边界

冻结模块对 \(M\) 做 SPD whitening，并以有界 secular 求解满足球约束的一步；没有经验 jitter、伪逆或负曲率裁零。它回读原坐标 stationarity、可行性与互补残差，按 binary64 计算尺度给出诊断。非 SPD、资源耗尽或残差不能解释时是明确技术失败。

这解决的是**已给定浮点 \(g,G,B,J,p\) 的子问题诊断**。基底的精确有理数 rank 与 enclosure 不会自动给出完整解析头及 JVP 的误差界。当前证据层级明确为 `FLOAT64_SUBPROBLEM_DIAGNOSTIC_NOT_COMPLETE_HEAD_CERTIFICATE`，完整 head/JVP 误差包络仍是 `None`。因此，小残差不能证明未知精确 head 下的物理方向完全正确；任意病态可逆 \(T\) 也不保证浮点求解得到逐 bit 相同方向。

核心在 stage anchor 固定 \(M\)，最多求一步，最多 12 个 Armijo trial，\(\eta=1,1/2,\ldots\)。实际接受使用

\[
 R(\theta+\eta d)\le R(\theta)+10^{-4}g^\top(\eta d)
 +128\epsilon_{64}\max(1,|R(\theta)|,|R(\theta+\eta d)|,|\mathrm{rhs}|).
 \tag{2}
\]

因此“accepted”表示满足源码定义的浮点比较；在下降小于比较尺度时，不能仅靠该词宣称精确 loss 严格下降。逐步状态另存精确比较是否成立与观察到的 objective increase。648 次接受也不能证明表示改善、旧、新 accuracy 改善或总体收敛。

## 3. RMSCE 与分类风险、H 为什么不同

### 3.1 连续 loss 与离散决策

当前类均 loss 与 RMSCE 为

\[
 \ell_c=\frac1{n_c}\sum_{i:y_i=c}-\log p_i(y_i),\qquad
 R=\sqrt{\frac1C\sum_c\ell_c^2}.
\]

对于 \(R>0\)，score 梯度的逐样本权重是

\[
 \frac{\ell_{y_i}}{C R n_{y_i}}(p_i-e_{y_i}).
\]

它更重视当前 CE 较大的类，同时优化概率幅度；accuracy 只检查真实类是否赢过全部竞争类。loss 可以通过提高已经正确的置信度、降低严重错误的 CE，或改变尚未越过 argmax 边界的 margin 而下降。

一个纯手算反例足以排除单调保证。某个二分类类的两个样本，真实类概率从 \((0.51,0.90)\) 变成 \((0.49,0.99)\)。由于 \(0.49\times0.99=0.4851>0.51\times0.90=0.459\)，该类平均 CE 下降，但正确数从 2 变成 1。其余类 loss 不变时 RMSCE 也下降。这个例子是数学构造，不是项目观测。

还存在有限但很弱的联系：多类 argmax 错误时 \(p_i(y_i)\le1/2\)，于是样本 CE 至少为 \(\log2\)。所以同一固定预测集上的类错误率 \(e_c\) 满足

\[
 e_c\le\ell_c/\log2,\qquad
 \frac1C\sum_c e_c\le R/\log2.
 \tag{3}
\]

右端可大于 1，且上界下降不表示实际错误率下降。它也不是从 OOF 到 query 的风险界。校准 surrogate 的人口 excess-risk 关系需要相应损失、分布与估计条件；不能拿该类定理替代本项目一步经验 RMSCE、约束解析头和 H 的单调保证。[Bartlett、Jordan 与 McAuliffe，一手作者稿，式 (1)](https://www.stat.berkeley.edu/~bartlett/papers/bjm-ccrb-05.pdf)

### 3.2 H 的组权重与当前目标不同

设旧、新组准确率为 \(a_O,a_N\)，两者正时

\[
 H=\frac{2a_Oa_N}{a_O+a_N},\quad
 \frac{\partial H}{\partial a_O}=\frac{2a_N^2}{(a_O+a_N)^2},\quad
 \frac{\partial H}{\partial a_N}=\frac{2a_O^2}{(a_O+a_N)^2}.
\]

较弱组的局部 accuracy 变化对 H 更敏感；RMSCE 的权重却由类 CE 决定。均匀类均目标中 6 个旧类与 \(q\) 个新类的组总权重还随 \(q\) 改变。连续概率误差、类数权重和 H 的离散两组调和权重不存在恒等映射。

因此本次“旧组相对 R0 略高、新组相对 R0 更低、H 更低”的合法 support 汇总与目标定义相容，不需要假定未观察到的 query 退化，也不能用其排除具体技术错误。应先分解条件头与跨组错误，再判断哪个机制受到限制。

## 4. 冻结旧条件函数对 C 的能力边界

记实际 B 的旧条件概率为 \(p_B(o\mid x)\)，新条件头为 \(p_N(j\mid x)\)，gate 为 \(g(x)\)。C 的全部注册类 log-score 是

\[
 s_o=\log\sigma(g)+\log p_B(o\mid x),\qquad
 s_j=\log\sigma(-g)+\log p_N(j\mid x).
 \tag{4}
\]

两组概率和为 1。每个物理样本同时竞争全部注册类，没有预先硬路由组。对于任意两个旧类，\(s_o-s_{o'}=f_B(o)-f_B(o')\)。这在任意输入上成立，不只在训练 support 上成立；C 的校正、Ridge 与 gate 都不能改变旧组内部赢家。

在同一实际 B、同一物理旧评估集和相同固定 tie 规则下，若 B 旧条件赢家已错，C 不可能输出正确旧类。因此

\[
 A_{C,O}\le A_{B,O}^{\mathrm{conditional}}.
 \tag{5}
\]

同样，给定某个 C 的新条件头，\(A_{C,N}\le A_N^{\mathrm{conditional}}\)。gate 只能保留或挤出已正确的组内赢家。C 更新可改变新条件排序，但旧条件上界冻结。这些上界不能跨 R0 与候选借用不同路径的 B。

定义跨组决策差

\[
 \Delta(x)=g(x)+\max_o\log p_B(o\mid x)-\max_j\log p_N(j\mid x).
 \tag{6}
\]

\(\Delta>0\) 时旧组最高 score 更大，\(\Delta<0\) 时新组更大，等号遵从固定 canonical 类 ID tie 规则。组内使用原始条件分数的比较，避免共同偏移在浮点输出中抹去原有排序。

当前 gate 对每个合法旧 inner-train 样本及每个新类保留严格 barrier slack

\[
 g(x_i)>a_{ij},\quad
 a_{ij}=d_i-\log p_B(y_i\mid x_i)+\log p_N(j\mid x_i).
 \tag{7}
\]

\(d_i\) 是实际 B 在 padded 新类零 logit 竞争下的原真实类 margin；负值及 ties 不 clamp。约束保护的是这些旧训练样本原有的旧对新 margin。负 margin 不能解释为原真实类必然正确，旧对旧排序更不能由它修复。约束没有延伸成 held 或 query 不遗忘保证。

内层用 old inner-train 在冻结实际 B 校正下重新拟合 teacher；final 用实际 full B。把含 held 标签的 full B teacher 用于 inner 目标会改变合法协议，本文不建议这样做。当前新头只由 new inner-train 的解析 centered Ridge 与完整 free intercept 得到，没有地面原型分类教师。

## 5. 唯一下一步假说：gate 的合法 support 组均衡监督

### 5.1 具体目标与固定权重

当前 gate 逐物理样本 Bernoulli loss 的权重都是 1。若一次 inner-train 每类有相同 \(K\)，旧组有 6 类、新组有 \(q\) 类，其监督质量为 \(6K:qK\)。这是训练 support 的采样质量，不是已知 query 先验。

令合法 train 共 \(N\) 个物理样本，\(C_O=6\)、\(C_N=q\)，类 \(c\) 的合法 train 数为 \(n_c>0\)。只在 \(q>0\) 的 C gate 中定义

\[
 w_i=\frac{N}{2C_{r(i)}n_{y_i}},\qquad r(i)\in\{O,N\}.
 \tag{8}
\]

于是 \(\sum_{i\in O}w_i=\sum_{i\in N}w_i=N/2\)，\(\sum_iw_i=N\)。没有可调组权重或新参数网格。权重只由合法 support 标签与计数确定，不能从 held/query 表现、真实 query 类数或配额估计。

优先候选只替换 gate 内层为

\[
 \Phi_{\mathrm{bal}}(g)=\tfrac12\|g_{\mathrm{RKHS}}\|^2
 +\sum_iw_i[\operatorname{softplus}(g_i)-t_i g_i]
 -\zeta\sum_{i\in O,j\in N}\log(g_i-a_{ij}),
 \tag{9}
\]

其中旧组 \(t_i=1\)，新组 \(t_i=0\)，free intercept、raw PSD kernel、全部 \(a_{ij}\)、正 slack 规则与现有 gate 求解形式不变。\(\zeta=N\times10^{-4}/m\)，\(m=n_Oq\)，沿用原按总监督质量 N 定义的 barrier approximation。理论中心点的 hard-objective gap 上界仍为 \(m\zeta/N=10^{-4}\)；这不是当前浮点解的精确实测 dual certificate。

外层仍为当前 RMSCE；B 仍为冻结候选的合法 old-only 联合一步；C 从其实际 B 出发，继续当前联合校正并完整重解新头与 gate。只改变式 (9) 的内层权重，避免同时改变外层目标、校正冻结方式或步数而无法辨认机制。new0 仍直接返回实际 B。

### 5.2 可解与完整求导路径

令 \(s_{ij}=g_i-a_{ij}>0\)。新梯度与有效曲率是

\[
 q_i^{\mathrm{eff}}=w_i(\sigma(g_i)-t_i)
 -\mathbf1_{i\in O}\sum_j\zeta/s_{ij},
\]

\[
 D_i^{\mathrm{eff}}=w_i\sigma(g_i)\sigma(-g_i)
 +\mathbf1_{i\in O}\sum_j\zeta/s_{ij}^2.
 \tag{10}
\]

正权重、两组样本与原 Slater/free-intercept 条件保持。固定新头时目标在 RKHS 函数与截距上仍严格凸；原 raw PSD kernel 的奇异性不需要伪逆或 jitter。Newton 与隐式求导仍通过 \(I+\sqrt D K\sqrt D\) 的 SPD 因子与 free-intercept Schur 系统处理。

所有支持计数、\(w_i\) 与 \(\zeta\) 在一个固定 fold 内对校正参数是常数，故 \(dw_i=d\zeta=0\)。完整伴随必须继续包含 raw kernel 两端、free intercept、每个 \(a_{ij}\) 中的新条件 log-probability，以及新 Ridge 的训练均值、free intercept 与 kernel。冻结 B 项的导数仍为零。max ties 不通过删约束或抹梯度解决；式 (9) 保留全部 \(i,j\) slack。

### 5.3 为什么这个假说值得先检验

在固定条件头时，全类 CE 的 gate 部分正是对应 Bernoulli loss。\(N^{-1}\sum_iw_i\mathrm{CE}_i\) 的 gate 部分等于旧、新组各占一半、组内类均的监督 CE；条件头部分对于求 gate 是常数。

一个没有 kernel 自由度、没有 barrier 的截距模型给出明确机制对照：原样本和 logistic 的最优 \(b=\log(n_O/n_N)=\log(6/q)\)，式 (9) 的最优 \(b=0\)。这只是解析极简模型，不是当前 constrained kernel gate 实际 b 的测量或断言。实际 \(g(x)=K\alpha+b\)、旧支持约束及条件头置信度均可改变结论。

该候选把内层监督目标转为两组对称，而不是声称 query 先验为 1/2。它仍不能保证 H：H 依赖离散错误与条件排序；RMSCE 外层权重也仍与 H 不同。如果新条件分类本身是瓶颈，或旧支持约束决定了更大的新组代价，权重变化可能无益甚至有害。当前汇总不足以区分这些情形。

### 5.4 只用合法 support 可验证的机制

下一次若由 root 独立预登记此单一候选，应固定完整矩阵与输出，保留 paired 物理身份，再做以下 support 分解；本文不启动它。

- 旧条件赢家正确率、新条件赢家正确率，以及各自被跨组竞争挤出的正确物理样本比例。这样能把式 (5) 的条件上界与 gate 损失分开。
- 每类 CE、RMSCE、真实类条件 margin、式 (6) 的跨组差，以及式 (7) 的合法训练 slack；按 K、new_count 和物理 parent 完整报告。不能只看 b，因为 RKHS 项同样参与组竞争。
- actual B 到 C 的实际物理校正变化、参数变化与接受不等式，区分置信度改善、组内排序变化和跨组挤出。B accuracy 不变不能替代这些记录。
- 预先定义的旧、新组质量各 N/2 是否在每个实际 inner-train 达成，held 标签是否仅在固定预测后的诊断中使用。不得用 query 结果选择权重、结构或选择性重跑。

如果新条件正确率低而跨组挤出少，当前假说不获支持；如果新条件正确率明显高于最终新准确率且主要损失来自跨组挤出，则与假说机制一致。两种情况都应保留，不能自动晋级。未提供上述分解前，本文不判断当前结果属于哪一种。

## 6. 退化规则与计算边界

new0 不定义式 (8)，继续 exact B reuse。K1/no-held 不做 OOF 校正更新，但最终解析 head 与有新类时的 gate 仍完整拟合。rank0 只表示没有可更新物理空间，不代表头梯度已认证为零；其完整新头与 gate 仍存在。\(\gamma=0/\mathrm{None}\)、\(\tau=0\) 等既有退化几何沿原源码处理，不新增 jitter、fallback 或省掉 free 常数项。

候选没有新增可训练坐标、源样本、源逐样本特征或地面统计。权重来自已有合法 support 计数；若以 float64 向量保留，新增 payload 是 \(8N\) 字节，亦可从 counts 重建。这个公式只描述数学数组，不是实测 wire、序列化或设备内存。冻结原型/frame 的既有传输成本保持原口径，不能再次计为新来源或假称总传输为零。

计算主体不变：全 raw kernel 为 \(N\times N\)，gate 有 \(m=n_Oq\) 个 slack，Newton/伴随涉及 \(N\) 阶因子，时间主项可为 \(O(N^3)\)、显式矩阵状态为 \(O(N^2+m)\)。new-only Ridge 的规模为 \(n_N\)，其因子主项 \(O(n_N^3)\)，多类 RHS 与完整 head JVP 另计。r 至多 5 只限制校正与方向维数，不能消除每个 fold、final fit、trial、伴随和失败尝试的解析头成本。

权重自身仅增加 \(O(N)\) 算术及可选向量状态，现有 barrier 运算仍为 \(O(m)\)。它不保证减少 Newton 次数或 wall time，不能以“五个参数”宣称星载省算力完成。真实硬件耗时、峰值 RSS/显存、能耗、部署 serializer 与 wire 字节均未在本文测量，应为 N/A；实际状态仍包含完整头、support、B 继承引用和审计数组。

## 7. 本文不作出的推断

本文不宣称现行方法已达到用户的适应提升、低遗忘或新旧接近目标；不把合法 support 结果外推为 query；不把未改 accuracy 当作未改参数；不把 exact stored rank、GGN PSD 或小 KKT 残差当作完整 head 误差证书；不把 barrier 的理论 approximation gap 当作测得的精确 dual certificate。

优先假说只解释一个可检验的内层监督错位。它保留 LocalRidge 优先、合法 support SFT、实际 B 继承与全部注册类逐物理样本竞争。是否实现、预登记或运行由 root 后续授权决定；现行代码、参数与健康任务保持冻结。
