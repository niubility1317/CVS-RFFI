# ProtoFrame support 度量的秩与 float64 求解边界

日期：2026-10-02。状态：**有界数学设计，未实现、未数值验证、未登记新实验**。本文只新增本文件。现行 core、entry、scorer、配置、五坐标状态和健康任务均保持冻结；未读取真实输入、产物、索引或指标，未执行数值、Git、Conda、SSH 或实验。

## 1. 有限结论

可以给出一条不使用经验 jitter、伪逆或奇异值截断网格的求解路径：**对已存 binary64 字典精确判秩，在其物理商空间构造完整导数，用正定度量球求解，并以包含舍入误差的可行性、KKT 残差和互补残差认证方向**。精确零方向可以删除；小而非零的方向必须保留。五维系数无法可靠表示物理解时，明确报告技术失败。

这个结论有两层边界。第一层是对已存数值及其明确误差区间的有限维子问题认证。第二层是把子问题输入联系到真实 RMSCE、完整 Ridge/free-intercept/gate 导数。目前第二层所需的统一 JVP 误差界尚未建立。**小线性残差、正定性或小 KKT 残差，不能独自证明完整方法的方向正确。** 缺少该误差链时只能报告“浮点子问题认证”，不得升级为原解析目标方向认证。

本文只定义这一条未来数值方案，保留现行 \(\theta\) 状态接口。将物理 \(w\) 设为权威状态是另一个将来 ABI，本文不据此绕过系数回读失败，也不改变当前状态。

## 2. 固定问题与输入含义

依据 [冻结坐标尺度说明](D92_PROTO_FRAME_GGN_COORDINATE_SCALE_NOTE_20261002.md) 式 (10)–(14)，在每个阶段 anchor 冻结

\[
M=Q^\top Q+F,\qquad
F=\sum_i\omega_iJ_i^\top S_iJ_i,\quad
\omega_i=\frac1{C n_{y_i}},\quad
S_i=\operatorname{diag}(p_i)-p_ip_i^\top .
\tag{1}
\]

\(Q\in\mathbb R^{160\times5}\) 是冻结原型差字典；\(p_i\) 和完整 \(J_i\) 来自合法训练 support 的 OOF 预测。外层 held 和 query 不参与。\(G\succeq0\) 是原 RMSCE 的完整 GGN；真实目标没有附加 proximal 项。候选方向问题为

\[
\min_{d^\top Md\le\rho^2}
g^\top d+\tfrac12d^\top(G+M)d,\qquad \rho=.5 .
\tag{2}
\]

全部可训练路径通过 \(w=Q\theta\)，故 \(\ker M=\ker Q\)、\(g\perp\ker Q\)。秩不足时只有物理方向 \(Qd\) 唯一，五维代表不唯一。这里的 \(M\) 定义本来就是新候选，不能把它当作现行 \(GGN+I_5\) 的等价修复。

本方案的“精确输入”首先指**已经存储的有限 binary64 位模式**。每个数都是精确二进制有理数，不是无限精度归一化原型。先前形成 \(Q\) 时的舍入可能改变数学秩；本文不声称恢复舍入前的理想几何。只有在输入区间明确包含理想量时，认证才覆盖该理想量。

## 3. 唯一方案的步骤

### 3.1 对五列作精确秩分解，不设小量阈值

按现有固定列顺序，将每个有限 binary64 分量解码为精确 dyadic rational。用任意精度整数及约分有理数作如下 Gram–Schmidt：

\[
v_j=q_j-\sum_{k\in\mathcal I}
v_k\frac{v_k^\top q_j}{v_k^\top v_k}.
\tag{3}
\]

仅当精确 \(v_j=0\) 时跳过该列。否则保留，哪怕其精确范数极小。最多五列，所得 \(r=|\mathcal I|\le5\) 是已存 \(Q\) 的精确秩；保存原列索引、精确投影系数和零关系。这里没有对病态矩阵求逆，也没有按经验阈值选择有效秩。

用精确整数指数选择二的幂缩放 \(v_k\)，再计算

\[
U_*[:,k]=v_k/\sqrt{v_k^\top v_k},\qquad U_*^\top U_*=I_r .
\tag{4}
\]

先缩放后舍入，不能先把微小 \(v_k\) 转成全零 float64。对平方根、转换和除法给出向外舍入区间，得到浮点 \(\widehat U\) 及误差包络；记录 \(\|\widehat U-U_*\|_2\) 上界和正交性误差。不能把浮点 \(\widehat U^\top\widehat U\) 强行写成 \(I\) 后遗漏误差。

精确有理数工作是有限且输入相关的整数计算。它不是“全流程只用 float64”，但保留 float64 的模型、head 和方向计算。若预先规定的整数内存/位长资源不足，返回技术失败，不把长整数截成近似值后宣布精确秩。本文未规定设备资源数值或启动该实现。

### 3.2 在物理基方向计算完整 JVP

令 \(\delta w=U_*a\)。在精确正交物理基中，定义 \(J_i^U=\partial s_i/\partial a\)，则

\[
F_U=\sum_i\omega_i(J_i^U)^\top S_iJ_i^U,\quad
M_U=I_r+F_U,\quad
A_U=G_U+M_U\succeq M_U\succeq I_r .
\tag{5}
\]

原 (2) 等价于

\[
\min_{a^\top M_Ua\le\rho^2}
(g_U)^\top a+\tfrac12a^\top A_Ua .
\tag{6}
\]

完整 JVP 应直接沿物理基方向求值；不要用已有五坐标 JVP 反除极小奇异值恢复它。它仍须经过移动训练均值、完整 Ridge 和自由截距、新头、gate 阈值、两端 raw kernel，以及旧条件函数冻结的全部路径。不能删除 ties、阈值导数或自由截距来获得较好的条件数。

构造 Fisher 时使用概率中心化因子：

\[
\mu_i=p_i^\top J_i^U,\qquad
B_{F,(i,c)}=\sqrt{\omega_i p_{ic}}\,[J^U_{ic,:}-\mu_i],\quad
F_U=B_F^\top B_F .
\tag{7}
\]

GGN 也保留冻结说明式 (11) 的两项 PSD 因子，包括 RMS 外层曲率；不能用 Fisher 替代 GGN。所有因子及平方根的舍入必须进入包络。极小概率、CE 或 barrier 曲率的 underflow 不可静默视为理论上的零。

得到 \([g_U],[G_U],[F_U]\) 的区间时，应注明它们包含什么：只是对已计算浮点 JVP 的矩阵累计误差，还是也包含 head/JVP 本身的误差。后者才可能覆盖原解析 (6)。本文没有证明完整 Ridge/gate 的这一统一包络已经可得。

### 3.3 正定度量白化与有界 secular 求解

以下精确符号均指物理商空间。对 \(M_U=R^\top R\) 作 Cholesky，令

\[
z=Ra,\quad c=R^{-\top}g_U,\quad
A=R^{-\top}A_UR^{-1}\succeq I_r .
\tag{8}
\]

在 float64 中得到近似因子后，必须验证因子/三角解误差；失败不能加对角量重试。理论目标变为 \(\min_{\|z\|\le\rho}c^\top z+\frac12z^\top Az\)。其解满足

\[
z(\lambda)=-(A+\lambda I)^{-1}c,\quad
\lambda\ge0,\quad
\lambda(\|z\|^2-\rho^2)=0 .
\tag{9}
\]

先解 \(\lambda=0\)。只有范数区间上端不超过 \(\rho\) 时才能判为内部解。若确为外部解，使用

\[
h(\lambda)=\|z(\lambda)\|^2-\rho^2,\qquad
h'(\lambda)=-2z^\top(A+\lambda I)^{-1}z<0
\tag{10}
\]

建立括区间。按向上舍入的 \(\lambda_U=\|c\|/\rho\) 有理论上 \(\|z(\lambda_U)\|\le\|c\|/(1+\lambda_U)<\rho\)；仍须验证实际计算的上端可行。若不能验证，不猜测括区间。

最多进行 128 次标量区间迭代，复用当前小球求解已有的有界规模。区间符号明确时二分；符号不明时仅用已认证负的 \([h']\) 作区间 Newton 收缩。不能把不确定符号当作零。每次小 SPD 解均计算包含舍入的残差和近似逆验证。128 次是明确数值预算，不能声称由性能或数学唯一确定；达到预算而未认证即失败。

返回经认证可行的高端方向，而不是以未知误差的中点假装精确根。允许最多两次向内的 nextafter 比例修正并重新回读；其位移进入最终误差账。修正不能替代方向误差认证。由于 \(M_U\succeq I\)，原 \(Q\) 的微小非零列尺度不直接使此物理系统失去正定性；但极大 Fisher/GGN、head 敏感性和浮点表示仍可使认证失败。

### 3.4 线性系统误差与最终方向证据

对真实包含在区间中的线性系统 \(Bx=b\)，取 float64 近似解 \(\widehat x\) 和由同一 SPD 因子解单位 RHS 得到的近似逆 \(V\)。用向外舍入计算

\[
q=\sup\|I-VB\|_\infty<1,\qquad
\beta=\frac{\sup\|V(b-B\widehat x)\|_\infty}{1-q}.
\tag{11}
\]

则 \(\|x-\widehat x\|_\infty\le\beta\)。这是 Rump 的定理 10.2、式 (10.8)–(10.9) 的直接应用；\(V\) 不要求是精确逆。若 \(q\ge1\)，该充分证据失败，不等价于证明矩阵奇异，也不能换成经验 condition 阈值通过。[Rump 原论文](https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf)。

最终认证直接在 (6) 的物理坐标中做，避免只认证白化后的不同问题。设实际输出对应 \(a_o\)，取得 \(\lambda_o\ge0\)。验证全部输入包络下

\[
a_o^\top M_Ua_o\le\rho^2,\quad
e=g_U+[G_U+(1+\lambda_o)M_U]a_o,\quad
E\ge\|e\|_2,\quad
\Gamma\ge\tfrac12\lambda_o(\rho^2-a_o^\top M_Ua_o).
\tag{12}
\]

\(E,\Gamma\) 用向外舍入；\(\Gamma\) 不得抹成零。设 \(a_*\) 是 (6) 真解，\(\Delta=\|a_o-a_*\|_{M_U}\)。强凸性、可行性和 (12) 给出

\[
\tfrac12(1+\lambda_o)\Delta^2\le E\Delta+\Gamma,\qquad
\Delta\le
\frac{E+\sqrt{E^2+2(1+\lambda_o)\Gamma}}{1+\lambda_o}.
\tag{13}
\]

证明：对二次目标在 \(a_o\) 展开，使用 \(A_U\succeq M_U\)；再代入
\[
-a_o^\top M_U(a_o-a_*)\le
\tfrac12[\rho^2-a_o^\top M_Ua_o-\Delta^2].
\]
梯度残差在对偶 \(M_U^{-1}\) 范数下至多 \(E\)，因为 \(M_U\succeq I\)。由最优目标差非负得到 (13)。于是物理误差 \(\|U_*(a_o-a_*)\|_2\le\Delta\)。

这是一项可计算的后验误差界，不依赖 strict complementarity，也不要求辨认“接近球边界”是严格内点还是边界。若需要固定成功标准，建议将物理误差上界与
\(\tau_{\rm dir}=128\,\operatorname{eps}_{64}\,5\,\rho\)
比较，其中 \(\operatorname{eps}_{64}=2^{-52}\)。128 沿用当前比较乘子的明确数值约定；它不是最优容差定理，不是秩阈值，也不是按分数选择的参数。任何未来实现必须明确登记这一误差标准，本文不将它加入现有运行。

若包络遗漏完整 JVP 误差，(13) 只能认证给定浮点子问题。即使界很小，也不能证明真实目标方向已正确；该证据层级须随记录保留。

### 3.5 回到原五坐标并回读真实目标

用 (3) 的已选原始独立列，在精确有理数关系中选择唯一固定代表：未选列系数为零，已选列系数解对应三角关系。不使用最小范数伪逆。输出舍入后的五维 \(d_o\) 后，把已存 \(Q,d_o\) 视为精确 dyadic 数，重新计算物理 \(Qd_o\) 的包络和其 \(U_*\) 坐标，再用该**实际输出**重新做 (12)–(13)。

近秩亏时 \(d_o\) 可非常大；系数溢出，或 \(Qd_o\) 的抵消使实际方向无法满足误差标准，均为 \(\mathrm{LIFT\_UNREPRESENTABLE}\) 技术失败。不能删除小方向、使用另一维数，或把物理权威状态写入现行 \(\theta\) ABI。若连 \(\theta_0+\eta d_o\) 的物理增量都无法可靠表示，同样失败。

完整方向认证还应验证 \(g_U^\top a_o<0\) 的区间上端为负；数学上已证 \(g_U=0\) 时零方向是合法结果。仅因浮点累计得到零而无包络证据，不可宣称真实梯度为零。试步仍按原真实 RMSCE、既定比例和 Armijo 回读完整 heads/gate。若比较区间重叠，报告数值不确定；不能以二次目标下降或接受标志一致代替真实目标比较。

## 4. 退化和失败规则

| 情形 | 唯一处理 |
|---|---|
| \(r=0\) | 物理方向必为零，不求伪逆；保留完整 final heads/gate 的原职责 |
| \(0<r<5\) 且精确关系已证 | 只删除精确核空间，在 \(r\) 维求解；五名义坐标和有效物理秩分别记录 |
| 近秩亏但精确 \(r=5\) | 全部保留；物理方向认证后仍须原五坐标 lift 回读 |
| K1 或不存在合法 OOF | 不用 final train loss 补 Fisher；保持不更新 adapter、完整拟合 final head |
| 极端 logits、CE/RMS 分母或 slack 导致 underflow/overflow | 稳定公式加真实误差包络；无法界定时技术失败，不加 curvature floor |
| Ridge/gate 分解、自由截距 Schur 或 JVP 包络缺失 | 保留已耗工作与原异常；不能称完整方向已认证 |
| 线性验证、括区间或最终 KKT 误差标准失败 | 明确失败，不 jitter、不 pinv、不截断、不默认旧 B |
| 输入非有限或精确秩计算超出资源 | 在任何适应更新前失败并保留诊断；不猜测秩 |

这些规则是未来实现的算法结果，不是为当前健康任务增加实验 gate。没有任何规则授权按性能重试或替换候选。

## 5. 理论不变性与有限精度边界

在实数精确算术下，任意可逆线性坐标变换 \(\theta=T\phi\)、\(Q_\phi=QT\) 给出
\(g_\phi=T^\top g\)、\(G_\phi=T^\top GT\)、\(M_\phi=T^\top MT\)。因此 (2) 的物理最优方向、度量球、同一比例的 trial 和真实 Armijo 坡度完全对应。不同正交商空间基也只作 \(r\) 维正交变换，不改变物理解。本文的线性论证是本项目推导；Martens §12 讨论一般重参数化的精确/有限步区别，不能据其结论宣称任意浮点实现完全不变。[Martens 正式论文](https://jmlr.org/papers/volume21/17-678/17-678.pdf)。

float64 中要分别看两种变化：

1. 同一个已存 \(Q\) 的不同求解坐标：认证均覆盖同一实数问题时，两方向的物理差可由各自 (13) 的界之和约束，但最后一位和 trial 接受标志可不同。
2. 先形成并舍入 \(QT\)，再视为新输入：新的 dyadic 矩阵可能具有不同精确秩或列空间。除非提供并传播 \(QT\) 的输入误差关系，它不是“同一问题”的证据。即使微小坐标变化，rank 本身也可能改变。

正定的物理 Gram 项消除了小列尺度直接进入线性解的困难，不能消除输入信息损失、完整 head 的敏感性或输出五坐标的表示问题。固定全 160 维共同正缩放后再归一化的不变性，也不能推广为任意子空间缩放不改变 kernel。

## 6. 实际计算、状态与未证明处

精确秩步骤为 \(O(160\cdot5^2)\) 次有理数运算；其**位复杂度**取决于输入指数跨度和中间分子/分母位长，不能按普通 float64 FLOP 免费记账。精确证书整数负载可写为
\[
B_{\rm rational}=\sum_{\text{stored integers }z}\left\lceil
\operatorname{bitlength}(|z|)/8\right\rceil
\]
再加实际序列化元数据；未实现时字节及秒数均为 N/A。

\(\widehat U\) 的数值负载为 \(8\cdot160r\) bytes，满秩时 6,400 bytes；原 \(Q\) 同为 6,400 bytes。若归档物理位移，另需 \(8\cdot160=1,280\) bytes，它只是诊断，不能因此改变状态权威。一个 \(r\times r\) float64 矩阵为 \(8r^2\) bytes，区间上下端为 \(16r^2\) bytes；完整 JVP 若保留则为 \(8NCr\) bytes，另有 factors、输入包络、head 状态及原归档。

Fisher/GGN 累计为 \(O(NCr^2)\)，小因子/白化为 \(O(r^3)\)。每次 secular SPD 因子只分解一次；构造 \(V\) 的 \(r\) 个单位 RHS、解向量以及导数 RHS 可在同一因子上实际计费，不能把它们称为 \(r+2\) 次 Cholesky。至多 128 次标量迭代是上限，实际尝试、完成、RHS、失败成本和区间验证成本分别记录。

物理 JVP 最多 \(r\le5\) 个方向。原 Ridge 的 \(N^3\) 因子、gate Newton/伴随、完整 free-intercept、新头和全部真实 trial 均保留；不因坐标只有五个就减少其成本。相较当前 core，直接物理方向 JVP、整数秩证书和误差包络都需要新实现，不能宣称复用后零成本。硬件时间、整机峰值、星载收益和实际新增传输全部未测，均为 N/A；冻结原型传输规则保持原方案，数值证书不是新源域统计。

尚未证明或实现的关键项只有：完整 head/JVP 的统一输入误差包络、稳定的向外舍入实现及其资源上限、原 \(\theta\) lift 的通用成功率，以及真实目标比较的误差闭合。本文证明物理商空间的严格凸性和 (13) 的条件性后验界，**没有证明所有有限输入都能通过认证，也没有证明 query 泛化或性能提升**。

## 7. 引用范围与冻结

- Rump, S. M. *Verification methods: Rigorous results using floating-point arithmetic*. Acta Numerica 19, 287–449, 2010。[作者机构 PDF](https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf)。已核实定理 10.2 与式 (10.8)–(10.9)。它支持线性系统的后验包络；不替本方法证明精确秩、完整 JVP 或 (13)。
- Martens, J. *New Insights and Perspectives on the Natural Gradient Method*. JMLR 21(146), 2020。[期刊页](https://jmlr.org/papers/v21/17-678.html)。其 GGN/Fisher、damping 和 §12 的重参数化讨论是背景；本文的精确 dyadic 商空间、support 度量球和 KKT 认证为新项目设计，不是论文已有性能承诺。

交付状态：本文冻结。仅文本、公式与链接静态核对；没有代码实现、数值验证、真实数据访问、新实验或现行方法修改。
