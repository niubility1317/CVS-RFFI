# D92 AJLR 无惩罚截距数学审计

日期：2026-10-01。范围：只审计在现有残差函数中增加每类无惩罚常数的数学后果。本文件不是候选冻结文档，不授权修改、停止或重启当前健康 AJLR runtime `a1a003`，也不登记或选择下一实验。本次未读取当前 partial 指标、历史 query/ABC、总实验索引或交接中的性能内容，未实现、拟合或执行数值测试。

**可证明的结论是：自由截距解除现有模型的固定旧参考均值限制；当前均衡旧 B 契约下，B 的截距精确为零，原 B 解保持不变。** 在 raw kernel、尺度和实际 B 先验均相同的条件下，中心参考测度只改变截距的表示，不改变完整预测函数。这些结论不保证 C 阶段旧类准确率保持、注册成功或任何 query 性能提升。

## 1. 审计对象、文献依据与符号

现有数学约定见 [AJLR 冻结设计第 3 至 5、7 节](D92_JOINT_AFTER_FCR8_DESIGN_20261001.md)。本审计只把原残差目标改写为以下待讨论的半参数问题：

\[
\min_{g\in\mathcal H_{k_U}^C,\ b\in\mathbb R^{1\times C}}
\frac12\|M+g(S)+e b-Y\|_F^2
+\frac12\sum_{c=1}^C\|g_c\|_{\mathcal H_{k_U}}^2.
\tag{1}
\]

这里的 \(b\) 只在合法 support 上由同一个平方损失估计，没有惩罚、扫描系数、旧/新组专属参数或 query 校准。B 的 \(M=0\)；C 的 \(M=m_B(S)\) 是本行实际 B 分类函数按注册类位置补零。增加 \(b\) 不改变实际 B 函数、其 adapter 或其已保存尺度。罚项只作用于残差 \(g\)，不是未经证明的同空间 \(\|f-m_B\|\) 结论。

文献提供模型形式和工具的依据，本文件后续 Schur 解、梯度和中心等价性由本次推导给出：

- [GPML 第 2 章 §2.7，印刷页 27 至 28、式 2.38 至 2.42](https://gaussianprocess.org/gpml/chapters/RW2.pdf) 区分固定均值的残差拟合与显式基函数的系数估计，并给出均值系数无信息先验极限的解析表达。这里用常数基函数和固定实际 B 先验；不构造趋于无穷的核常数，不称为继承 B 后验协方差的精确顺序 Bayes 更新。
- [Schölkopf、Herbrich、Smola，2001，Theorem 2，印刷页 421 至 422](https://alex.smola.org/papers/2001/SchHerSmo01.pdf) 给出带未惩罚有限维函数空间的半参数 representer 形式。常数基函数在非空 support 上满足其秩条件；残差可用训练核展开，另保留截距。
- [R2-D2，Bertinetto 等，ICLR 2019，§3.2、式 3 至 5](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/bertinetto19meta-learning.pdf) 展示闭式 ridge 头及其可微训练。该文式 6 的共享输出校准量由外层学习，和本审计由当前 support 估计的每类未惩罚 \(b\) 不同。论文没有验证本文的 AJLR 变体、数据协议或性能。

对一个合法训练头，令：

| 符号 | 含义与形状 |
|---|---|
| \(S\)、\(n>0\) | 当前合法 train support 及记录数 |
| \(C\)、\(j=\mathbf1_C\) | 全部注册类数及类别维度的全一向量 |
| \(e=\mathbf1_n\)、\(e_h=\mathbf1_h\) | train 与待评分记录维度的全一向量 |
| \(Y\in\mathbb R^{n\times C}\) | 每行 one-hot 减 \(1/C\)，故 \(Yj=0\) |
| \(M\in\mathbb R^{n\times C}\)、\(M_h\in\mathbb R^{h\times C}\) | 固定实际 B 先验在 train、待评分记录上的值；B 时均为零 |
| \(E=Y-M\) | 残差标签；没有默认对样本维度再中心化 |
| \(R\in\mathbb R^{n\times n}\)、\(Q\in\mathbb R^{h\times n}\) | 同一个 raw kernel 的 train Gram、cross Gram |
| \(q\in\mathbb R^n\)、\(e^Tq=1\) | 固定中心测度在 train 中的权重；现有 C 只在旧参考 support 上等权 |
| \(P_q=I-eq^T\) | 中心化矩阵 |
| \(K=\gamma P_qRP_q^T\)、\(L=\gamma(Q-e_hq^TR)P_q^T\) | residual train/cross kernel，\(\gamma\) 固定 |
| \(A=I+K\) | canonical SPD 系统矩阵；不使用 jitter |

第 7 节的 \(h\) 指合法 support 内用于监督 adapter 的 inner-held，\(G\) 来自该监督风险。最终 query 可以使用同一 cross-score 公式，但其 truth 不进入拟合或梯度。所有矩阵形状都按物理记录计数，不由 query 的真实类别构成决定。

## 2. 唯一函数解与 Schur 闭式

取中心核的任一特征映射 \(\psi_q\)，每类残差写为 \(g_c(x)=\langle w_c,\psi_q(x)\rangle\)。对 \(w,b\) 求一阶条件，定义 canonical 残差系数为训练目标与预测之差：

\[
\alpha=E-K\alpha-eb,\qquad e^T\alpha=0.
\tag{2}
\]

第一个等式来自 \(w=\sum_i\psi_q(x_i)\alpha_i\)，第二个来自截距的无惩罚最优条件。因而

\[
\begin{pmatrix}A&e\\e^T&0\end{pmatrix}
\begin{pmatrix}\alpha\\b\end{pmatrix}
=\begin{pmatrix}E\\0\end{pmatrix}.
\tag{3}
\]

定义

\[
F=A^{-1}E,\qquad z=A^{-1}e,\qquad s=e^Tz>0.
\tag{4}
\]

Schur 补给出

\[
\boxed{b=\frac{e^TF}{s},\qquad \alpha=F-zb.}
\tag{5}
\]

这里的 \(A^{-1}\) 是数学记号，实现应求解线性系统。对 PSD 的 \(K\)，\(A\succeq I\)，所以

\[
\frac{n}{1+\operatorname{tr}K}\le s\le n.
\tag{6}
\]

式 (6) 使用本次实际 \(K\) 的 trace，不用未随 \(U\) 更新的旧 \(s_0\) 冒充实际谱界，也不需要给 \(s\) 设置人为 floor。只需一个 \(n\times n\) SPD Cholesky。把 \([E,e]\) 合成 \(C+1\) 个右端，可用两次三角求解同时得到 \(F,z\)；再做标量 Schur 除法和秩一修正。不能对式 (3) 的不定矩阵直接宣称 SPD Cholesky，也不需要构造 \((n+1)\times(n+1)\) 的稠密逆。

即使 \(K\) 奇异，式 (3) 仍非奇异：齐次解满足 \(A\alpha+eb=0\)、\(e^T\alpha=0\)，左乘 \(\alpha^T\) 即得 \(\alpha=0\)，再得 \(b=0\)。原问题的函数解也唯一，因为对非零变化 \((\Delta w,\Delta b)\)，二次曲率为

\[
\|\Psi_q\Delta w+e\Delta b\|_F^2+\|\Delta w\|_F^2>0.
\]

核展开系数若只从 \(g=K\alpha\) 定义，可能有零空间歧义；式 (2) 指定的 canonical \(\alpha\) 没有此歧义。不能把零空间中的系数改成另一解来美化诊断。

完整 train 与 held score 是

\[
F_S=M+K\alpha+eb=Y-\alpha,
\qquad F_h=M_h+L\alpha+e_hb.
\tag{7}
\]

可直接审计的等式残差是 \(A\alpha+eb-E\) 和 \(e^T\alpha\)。前者的尺度可由 \((1+\operatorname{tr}K)\|\alpha\|_F+\sqrt n\|b\|_2+\|E\|_F\) 表达。本文不冻结浮点阈值或新增流程门槛；真正非有限值、Cholesky 或等式检验失败仍是技术问题，不能改公式当成性能退化。

## 3. 哪个确定性限制被解除

记旧参考记录数为 \(n_o\)，旧类数为 \(C_B\)，新类数为 \(C_N>0\)，\(C=C_B+C_N\)。在当前平衡旧参考契约下，实际 B 函数及中心残差在旧参考上的均值都为零：

\[
q^TM=0,\qquad q^TK\alpha=0.
\]

全注册类标签的旧参考均值为行向量 \(v\)：

\[
v_c=\begin{cases}
1/C_B-1/C,&c\text{ 为旧类},\\
-1/C,&c\text{ 为新类}.
\end{cases}
\qquad \|v\|_2^2=\frac{C_N}{C_BC}.
\tag{8}
\]

无截距时，任何 \(U\) 的旧参考半平方数据误差至少为

\[
\frac{n_o}{2}\|v\|_2^2=\frac{n_oC_N}{2C_BC}>0.
\tag{9}
\]

加入截距后，旧参考预测均值变成 \(q^TF_S=b\)，同一个 Jensen 推导只给出

\[
\frac12\sum_{i\in O}\|F_i-Y_i\|_2^2
\ge\frac{n_o}{2}\|b-v\|_2^2.
\tag{10}
\]

\(b=v\) 在模型中可行且没有截距罚项，因此式 (9) 的固定正下界不再由旧参考均值约束强制成立。这里解除的是**无法表达该均值分量**的限制，不是证明总体 SSE 可以为零。最优 \(b\) 由完整当前 support 的损失共同确定，通常不能预先指定为 \(v\)；RKHS 正则、重复输入、kernel 秩、固定尺度、实际 B 先验以及不同类的可分性仍可能限制拟合。

原无截距解是新问题取 \(b=0\) 的可行解，因此在同一固定 \(U\) 上，式 (1) 的最小正则化平方目标不会增大。这个弱单调结论不等于数据 SSE 单独下降，也不等于 adapter 所优化的 RMS CE、任何类准确率或 query 性能改善。训练目标与泛化评价不能互换。

## 4. 当前均衡旧 B 是否保持不变

本节只使用当前旧 B 契约：所有旧类拥有相同合法 support 数；每个旧类等量切分的 inner-train 与完整 final support 仍满足该平衡；B 的先验为零。于是 B 中 \(q=u=e/n\)、\(e^TY_B=0\)，而中心核满足

\[
K_Be=0,\qquad A_Be=e,\qquad z_B=e,\qquad s_B=n.
\]

由对称性和式 (5)，

\[
b_B=\frac{e^TA_B^{-1}Y_B}{n}
=\frac{e^TY_B}{n}=0,\qquad
\alpha_B=A_B^{-1}Y_B.
\tag{11}
\]

这恰是原无截距 B 解。该恒等式对每个固定 \(U\) 成立；若其余准备、监督风险和边界均相同，B 的精确函数、训练目标及其对 \(U\) 的梯度也相同。既有实际 B 可以继续作为固定先验，无须为了本审计重新拟合。浮点操作顺序可能产生舍入差异，本文不声称 bitwise 相同。

**式 (11) 不推广到不均衡 B。** 若旧类样本数不同，通常 \(e^TY_B\ne0\)，新增截距会改变 B。类维度的每行零和 \(Y_Bj=0\) 不能代替样本维度的 \(e^TY_B=0\)。本节也不证明 C 注册后的旧列 score、argmax 或准确率保持原 B。

## 5. 两种零和条件不能混淆

所有自由截距头都有

\[
e^T\alpha=0,
\tag{12}
\]

这是**样本维度**的条件，来自未惩罚截距。

如果实际 B 的每条 score 在其旧类列上零和，补零后的 \(M,M_h\) 也满足 \(Mj=M_hj=0\)。结合 \(Yj=0\)，式 (4) 至 \(5\) 给出

\[
Ej=0,\quad Fj=0,\quad bj=0,\quad\alpha j=0,
\quad F_Sj=F_hj=0.
\tag{13}
\]

这是**类别维度**的零和。\(b\) 可存为 \(C\) 个浮点数，但其独立 class-contrast 自由度为 \(C-1\)。式 (12) 与式 (13) 形状和来源不同，不能互相替代。若先验不满足类别零和，就不能自动声明式 (13)；必须从真实保存函数核验，不能对 prior 任意减均值以隐藏差异。所有类共同的 score 常数不影响 argmax/softmax，但平方目标仍明确固定该分量的规范解。

## 6. 自由截距下的参考测度等价性

### 6.1 同一 raw kernel 下的代数证明

本节固定同一 \(S,R,Q,U,\tau,\gamma,Y,M,M_h\)，只改变满足 \(e^Tq=1\) 的中心权重。它不改变带宽、trace scale、训练记录、实际 B 先验或正则强度。

因为 \(e^T\alpha=0\)，\(P_q^T\alpha=\alpha\)。定义 raw 表示中的截距

\[
\beta=b_q-\gamma q^TR\alpha.
\tag{14}
\]

式 (3) 与式 (7) 立即变成

\[
(I+\gamma R)\alpha+e\beta=E,\qquad e^T\alpha=0,
\qquad F_h=M_h+\gamma Q\alpha+e_h\beta.
\tag{15}
\]

式 (15) 完全不含 \(q\)，且有唯一 canonical 解。因此任意两个这样的测度 \(q,q'\) 的 \(\alpha\) 与完整预测函数相同，只有

\[
\boxed{b_{q'}=b_q+\gamma(q'-q)^TR\alpha}
\tag{16}
\]

发生变化。罚项也相同：

\[
\operatorname{tr}(\alpha^TK_q\alpha)
=\gamma\operatorname{tr}(\alpha^TR\alpha).
\tag{17}
\]

从共同 raw 特征空间看，中心化只是把所有特征减去同一均值向量；自由常数吸收这个平移，系数和为零则消除残差权重中的均值项。式 (15) 至 \(17\) 已证明本题中“旧 support 测度”和“完整 train support 测度”的精确等价，不要求两个测度具有相同类频率。

### 6.2 全 train 均值表示

令 \(u=e/n\)、\(H=I-eu^T\)，则同一个函数还可写为

\[
K_u=\gamma HRH,\qquad
\alpha=(I+K_u)^{-1}HE,\qquad b_u=u^TE.
\tag{18}
\]

恢复旧测度的截距需用

\[
b_q=b_u+\gamma(q-u)^TR\alpha.
\tag{19}
\]

式 (18) 是带截距完整 score 的等价表达。仅把 \(E\) 减去样本均值却丢掉 \(b_u\)，会改变模型；不能借此修改当前无截距实现。

### 6.3 可以与不能归因的内容

在式 (15) 的条件下，**新模型不能再把固定旧中心 \(q\) 本身作为预测性能差异的原因**。改变 \(q\) 的数学函数和对 \(U\) 的完整导数都相同，最多改变有限精度计算的条件数、舍入或工作组织。这不是对当前无截距 run 的性能归因，也不说明该 run 的结果。

旧 support 确定 \(\tau_B,\gamma_B\) 的作用仍保留。改变旧尺度估计为新集合尺度，会改变 raw kernel 或其缩放，超出等价证明；改变 prior、support 集合、截距惩罚、正则系数或仅近似求解也超出条件。中心测度的平移不变性不消除这些统计与模型选择。

## 7. 完整 held score 的伴随梯度

### 7.1 合法训练风险与 score 梯度

沿用现有跨 inner-fold 汇总的每类 mean CE 和 RMS 风险：

\[
\ell_c=\frac1{n_c}\sum_{i:y_i=c}\operatorname{CE}(F_i,y_i),
\quad R_{CE}=\sqrt{\frac1C\sum_c\ell_c^2}.
\]

本段仅定义合法 support 内的监督 held 梯度。先跨折形成每类 \(\ell_c,n_c\)，再向每折分配梯度，不能平均不等长 fold 的 RMS。若 \(R_{CE}>0\)，每条类别 \(c\) 的记录对应

\[
G_i=\frac{\ell_c}{CR_{CE}n_c}
\big(\operatorname{softmax}(F_i)-\operatorname{onehot}(y_i)\big).
\tag{20}
\]

\(R_{CE}=0\) 时取零 CE 梯度。以下推导也适用于任意已经正确形成的 \(G=\partial R/\partial F_h\in\mathbb R^{h\times C}\)，不依赖 CE 以外的特殊性质。

### 7.2 截距必须进入隐式求导

本阶段 actual B prior、\(E,M_h\) 固定。对式 (3) 微分得

\[
A\,d\alpha+e\,db=-(dK)\alpha,
\qquad e^Td\alpha=0,
\quad dF_h=(dL)\alpha+L\,d\alpha+e_h\,db.
\tag{21}
\]

令 \(g_b=e_h^TG\in\mathbb R^{1\times C}\)。使用同一个对称 saddle 系统定义伴随：

\[
\begin{pmatrix}A&e\\e^T&0\end{pmatrix}
\begin{pmatrix}T\\\eta\end{pmatrix}
=\begin{pmatrix}L^TG\\g_b\end{pmatrix}.
\tag{22}
\]

复用前向 Cholesky、\(z,s\)，只需

\[
V=A^{-1}L^TG,\qquad
\eta=\frac{e^TV-g_b}{s},\qquad T=V-z\eta.
\tag{23}
\]

特别地，\(e^TT=g_b\)，通常不是零。CE 的 \(Gj=0\) 只说明 \(g_bj=0\)，不说明向量 \(g_b=0\)。遗漏式 (22) 下方的非零右端会漏掉 \(db\)，不能机械沿用当前无截距头的伴随。

对称 \(K\) 和任意 cross \(L\) 的完整 adjoint 为

\[
\boxed{\bar L=G\alpha^T,\qquad
\bar K=-\operatorname{sym}(T\alpha^T)},
\quad\operatorname{sym}(B)=(B+B^T)/2.
\tag{24}
\]

其中 \(\langle\bar K,dK\rangle+\langle\bar L,dL\rangle=\langle G,dF_h\rangle\)。若把 prior 也作为变量，形式上还有 \(\bar M=-T\)、\(\bar M_h=G\)；但本题实际 B prior 对 C 残差坐标固定，这两条路径的 \(dM,dM_h\) 严格为零。不更新 \(U_B\)，不通过 prior 给 B 或源域参数反传。

### 7.3 从中心核到 raw kernel

固定 \(q,\gamma\) 时，完整中心化 VJP 为

\[
\bar Q=\gamma\bar LP_q,\qquad
\bar R=\gamma\operatorname{sym}
\{P_q^T\bar KP_q-q(e_h^T\bar LP_q)\}.
\tag{25}
\]

现在利用 \(e^T\alpha=0\)、\(e^TT=g_b\)：

\[
\bar LP_q=G\alpha^T,
\quad
P_q^T\bar KP_q
=-\operatorname{sym}(T\alpha^T)
+\operatorname{sym}(qg_b\alpha^T).
\]

式 (25) 的第二项恰好抵消后一个中心项，得到

\[
\boxed{\bar Q=\gamma G\alpha^T,\qquad
\bar R=-\gamma\operatorname{sym}(T\alpha^T).}
\tag{26}
\]

这一消去与第 6 节的参考测度等价性一致。它只对完整自由截距伴随成立。新模型可通过等价 raw 表达求完整梯度；不能把当前无截距设计中“参考项必须反传”的论断原样移植为独立的新限制。若保留中心化计算图，应反传全部项后得到式 (26)；只删除若干参考贡献而保留其他不配套项仍可能求错梯度。所有 raw train 几何中的旧 support 点仍有实际 \(U\) 导数；不能因此冻结其 residual embedding。

### 7.4 Gaussian 和 adapter 链

正带宽分支沿用

\[
D_U(x,x')=\tfrac12\|\Phi_0(x)-\Phi_0(x')\|^2
+\tfrac12\|\Phi_U(x)-\Phi_U(x')\|^2,
\quad R_U=\exp(-D_U/\tau_B).
\]

因此 raw train/cross 的距离 adjoint 分别是

\[
\bar D_{SS}=-(R\odot\bar R)/\tau_B,
\qquad\bar D_{hS}=-(Q\odot\bar Q)/\tau_B.
\tag{27}
\]

对任一有向 pair \((a,b)\)，adapted 平方距离乘 \(1/2\) 的导数给端点贡献 \(\bar D_{ab}(\Phi_U(a)-\Phi_U(b))\) 及其相反数。train 全矩阵的两方向都要按其系数累计；若仅计算对称唯一 pair，应合并两方向系数，不能少一倍。随后按现有 interaction 与 adapter 映射做完整链式求导。

对固定准备得到的 \(W\)、\(U=U_{anchor}+ZW^T\)，

\[
\nabla_ZJ=(\nabla_UR_{CE})W+Z.
\tag{28}
\]

\(\tau_B,\gamma_B,q,W,U_{anchor}\) 在当前阶段固定，actual B prior 无 C 梯度。式 (28) 中的 \(Z\) 是现有 proximal 罚项，不是对 \(b\) 的惩罚；\(b\) 的全部作用已经通过式 (21) 至 \(27\) 的闭式头导数进入。

## 8. 明确边界

| 条件 | 自由截距的数学结果与本审计边界 |
|---|---|
| \(\tau_B>0,\gamma_B>0\) | 使用式 (3) 至 \(7\)；PSD 与 \(A\succeq I\) 保证规范解。改变 \(U\) 时实际 trace 可变，尺度仍固定 |
| \(\tau_0=0,s_0>0\) | 沿用原始 interaction 完全特征等价关系的 PSD raw kernel 与原 \(\gamma\)。截距仍可闭式求解；不对该边界求 adapter 导数，不合成小正带宽。完全相同输入仍得到同一 score，不能同时区分不同 truth |
| 缺失 \(\tau\) 或 \(\gamma\)，按现约定 \(K=L=0,M=M_h=0\) | \(A=I,z=e,s=n,b=n^{-1}e^TY,\alpha=Y-eb\)，残差函数 \(g=0\)，score 为常数 \(b\)。若当前头全部注册类等量，\(b=0\)，仍全类并列；自由截距不能制造缺失的 kernel 信息。一般固定 prior 的零核公式是 \(b=n^{-1}e^T(Y-M),F_h=M_h+e_hb\)，不能用于静默替换当前缺失尺度分支 |
| retained rank \(r=0\) 或 \(H\) 全零 | adapter 坐标空，\(U=anchor\)，不监督更新 adapter；合法完整 support 的 residual head 和 \(b\) 仍需按公式求解。rank 0 不等于完整方法没有执行 |
| 真正 \(K=1\) 或合法 train \(K=1\) | 无独立 inner-held，不监督更新 adapter，不产生虚构 OOF 风险；完整 support 闭式头仍有定义。均衡旧 B 的 \(b_B=0\)，C 的 \(b\) 只用当前 support。proxy 不能冒充真正 K1 独立验证 |
| \(N_{new}=0\) | 仍须精确复用本行实际 B，既无新 C head，也无 C 更新。增加截距不能把此边界改为重拟合 |
| 相同输入、不同标签 | 保留相同核行，规范系统仍可解；公共截距不会让相同输入得到不同 score，不使用 truth 路由、类配额或标签修改 |
| \(C=1\) | 类别中心化目标为零；零类间 contrast 自由度。合法单类预测是平凡情形，不作为多类注册成功证据 |
| 非有限数、稳定距离失败或 Cholesky/等式残差失败 | 属于实现正确性问题；保留产物，不加 jitter、floor 或结果驱动 fallback |

特别强调 \(N_{new}=0\)：它是流程上的精确复用约定，不是新的残差目标自动产生的结果。若在原 B 数据上另做一次 C 残差拟合，原 B 规范式给出 \(E=Y-K_B\alpha_B=\alpha_B\)，通常仍有可拟合的非零残差；因此新闭式拟合会再次改变函数。即使新截距为零，也不能把它误写成 B 复用。

## 9. 解析自由度、求解工作与字节

本节给出单个 head 的代数成本，未运行任何新实现；实际耗时、峰值内存/显存、包体和传输字节均为 N/A。调用次数、右端宽度和 FLOP 工作不能混为一个数。

### 9.1 自由度

\(b\) 有 \(C\) 个保存标量、\(C-1\) 个独立类间 contrast。它们由 support 的解析解估计，不加入 adapter optimizer，也不新增 SGD 参数或迭代。若 \(\operatorname{rank}K=r_k\le n-1\)，固定 kernel 下残差训练函数最多有 \(r_k(C-1)\) 个独立类间方向，截距再提供 \(C-1\) 个方向。canonical \(\alpha\) 虽满足样本与类别两种零和，仍可有核零空间分量，不能把保存的 \(nC\) 个系数全称为独立函数自由度。

如需讨论有效自由度，必须以固定 kernel、固定 prior 的条件平滑器为对象。令

\[
P_A=A^{-1}-\frac{zz^T}{s},\qquad
\alpha=P_AE,\qquad F_S-M=(I-P_A)E.
\]

每个独立标量输出的 conditional EDF 是

\[
\operatorname{df}=n-\operatorname{tr}A^{-1}
+\frac{\|z\|_2^2}{s}.
\tag{29}
\]

相对同一 \(K_q\) 的无截距平滑器，增加量为 \(\|z\|^2/s\in(0,1]\)。全 train 均值表示下增加量恰为 1，等价地

\[
\operatorname{df}=1+\operatorname{tr}
\{K_u(I+K_u)^{-1}\}.
\tag{30}
\]

零和类别子空间有 \(C-1\) 个这样的输出；这不是包含 B prior 估计和监督 adapter 的全方法 EDF。式 (29) 的 trace 如通过求解 \(A^{-1}I_n\) 测量，会额外执行两次、各有 \(n\) 个右端的三角求解。该诊断不是求 \(b\) 的必要成本，本审计不要求追加它，也不把未执行的 EDF 求解计入实际工作。

### 9.2 前向与伴随求解

| 路线与一次 head 工作 | SPD 分解 | 三角调用 | 每次右端宽度 | 与当前无截距路线的区别 |
|---|---:|---:|---:|---|
| 现有无截距前向 | 1 个 \(n\times n\) Cholesky | 2 | \(C\) | 基准 |
| 一般 \(q\)，合并 \([E,e]\) 的自由截距前向 | 1 个同阶 Cholesky | 2 | \(C+1\) | 调用数相同，多 1 个右端；另有 Schur reduction 与秩一修正 |
| 一般 \(q\)，单独求 \(F\) 与 \(z\) | 1 个同阶 Cholesky | 4 | 两次 \(C\)，两次 1 | 比基准多 2 次三角调用；不能隐去 |
| 自由截距伴随，缓存 \(z,s\) | 复用前向 | 2 | \(C\) | 与原伴随同调用数，另计算 \(g_b,\eta,T\) |
| 伴随未缓存 \(z\)，额外重求 | 复用前向 | 额外 2 | 1 | 重求成本另列，不能按缓存路线记账 |
| 等价 \(q=u\) 表示，直接使用 \(z=e,s=n\) | 1 个同阶 Cholesky | 2 | \(C\) | \(b_u=\operatorname{mean}E\)，无需额外全一右端；必须保存/加回截距 |
| 零 residual kernel 的显式退化分支 | 0 | 0 | 不适用 | 仅均值与广播，不能虚报 Cholesky |

标准 dense Cholesky 的主阶工作为 \(n^3/3\) FLOP；两次三角求解合计约 \(2n^2r_{rhs}\) FLOP。因此一般 \(q\) 的合并路线新增全一右端约增加 \(2n^2\) FLOP，而不是新增分解。前向形成 \(b\) 与 \(\alpha\)、伴随形成 \(\eta,T\) 都有 \(O(nC)\) 的 reduction/秩一工作；每条评分记录加 \(b\) 需要 \(C\) 个标量加法。实际 backend、对称存储和复用决定精确计数及耗时，以上主阶式不充当测量。

若以后审计某份实现，应按真正选择的路线报告头数、分解数、三角调用数及各次 RHS/物理规模。不能同时用一般 \(q\) 的状态和均值规范的少求解预算，或把旧 B 的已证明 \(b_B=0\) 当成所有 C 都不需截距求解。当前既有 head/prior/outer score 计数合同保持冻结，本文件不修改启动预算。

### 9.3 状态与部署字节

以 float64 numeric buffer 为口径，新增量有如下明确边界：

- head 保存完整 \(b\in\mathbb R^{1\times C}\) 时，新增 \(8C\) bytes；独立 contrast 数为 \(C-1\) 不会自动缩减实际数组。若明确实现了压缩，才可按其真实保存量计。
- 训练/反传保留 \(z\in\mathbb R^n,s\in\mathbb R\) 时，新增 \(8n+8\) bytes。部署只需 score 截距，不需保留 Schur 缓存。
- 合并 RHS 多一列的单个 buffer 需要额外 \(8n\) bytes。是否与 \(z\)、求解输出或 scratch 重叠取决于具体实现；不能把这些理论量直接相加宣称实际峰值。
- 原有 \(nC\) 的 \(\alpha\)、\(n^2\) 的 factor、adapter 和实际 B prior 缓存仍有成本。固定公式不增加分解阶数，不意味着其原有费用消失。
- 若最终 deployment 新增完整 \(b_C\)，最小额外 numeric 状态为 \(8C\) bytes。实际 B 在当前平衡契约下可数学上省略零 \(b_B\)，但只有实现确实省略时才不计；若 prior 也保存零数组，应另计 \(8C_B\) bytes。
- NPZ/JSON 元数据、压缩、多个 archived fold、日志状态和打包会改变文件及传输字节；它们不是上述 numeric buffer 字节。新增传输总量、训练/推理耗时、峰值资源及硬件口径在未实现测量前一律 N/A。

## 10. 科学权限与交付边界

本审计与现行 [本机项目科学协议](E:/type10-7/项目.md) 及 [optimizer workflow](../tools/optimizer_workflow_contract.md) 的授权相容，条件是将 \(b\) 与 head 一样只从合法 support 估计。Phase1 合规 source-only 基座保持固定；不读取源域样本、逐样本 feature 或目标 query 拟合 head/adapter。C 只绑定本行真实 B；inner prior 只用同折旧 inner-train，完整 final C 才用实际完整 B，不能混入 outer-held。

最终每条 query 使用同一已冻结函数

\[
f(x)=\operatorname{pad}_C(f_B(x))+k_U(x,S)\alpha+b
\]

面对全部注册类做同一 argmax；并列仍按统一物理 class-ID 规则。\(b\) 没有 query truth、old/new role、真实 query 类别数量、配额、全局重排或逐 query 选择。预测固定后才由独立 scorer 连接 truth。无监督 query batch 均值也不进入本题的截距估计。类注册元数据及 support 标签是合法训练信息，不等于 query 真实类别构成。

本文件完成的是结构审计：Schur 闭式、balanced B 保持、固定均值下界的解除、参考测度等价性、完整伴随和解析资源口径。未验证新模型数值实现或 query 性能，未冻结后续候选。当前完整 run 结束后，是否另立方法、实施和开展实验由主任务后续决定；本文件没有据此添加任何运行或重复验证要求。
