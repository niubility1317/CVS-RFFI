# TD-HTRC M2.2模块二改造说明

## 1.定位与结论

TD-HTRC M2.2是模块二的一个显式可选实现。它不替换默认D92 E0，也不改变默认`module2_mode="baseline"`的数学路径；只有调用方明确传入`module2_mode="td_htrc_m22"`时才启用。

M2.2解决的是一个明确问题：目标域support中的类别中心不仅可能发生共同平移，还可能在少量、可辨识的方向上发生尺度或低秩形变。它使用6个旧类的地面聚合中心作为锚点，先估计共享偏移，再估计受约束的传输矩阵；随后用目标旧类残差重建扰动谱，用稳健中心后验表达support中心的不确定性，并把该不确定性以对角项加入D92的旧/新任务共享协方差。

它不是梯度更新训练，不读取query，不回放source样本，也不把6个旧类锚点扩展成一个自由的288维仿射网络。M2.2的全部状态在一次support注册中计算完成，之后将变换编译进仿射分类头，query端仍是一次普通的线性打分。

## 2.输入、输出和协议边界

### 2.1输入

|符号/对象|形状|含义|
|---|---:|---|
|\(\mathbf Z\)|\(N\times288\)|当前注册批次的目标域support特征；\(N=CK\)，\(C\)是注册后总类数，\(K\)是每类shot数|
|\(\mathbf y\)|\(N\)|support整数标签，标签顺序必须与注册类顺序一致|
|\(\mathbf G_{160}\)|\(6\times160\)|Phase1不可变旧类地面聚合中心；前6行与当前旧类注册表一一对应|
|\(\mathbf G_{288}\)|\(6\times288\)，可选|完整Phase1旧类中心。提供时可估计FFT96和RF32两块的目标尺度；不提供时只对identity160做传输，辅助块尺度固定为1|
|\(\mathbf U\)|\(160\times r\)|冻结的地面身份扰动谱基；\(r\)是保留方向数|
|\(\boldsymbol\rho\)|\(r\)|地面扰动谱权重，非负且归一化|

这里的\(\mathbf Z\)只来自目标域support。query特征、query标签、query批次类别比例、真实query类别计数和任何全局重分配信息都不进入M2.2拟合。

### 2.2输出

M2.2内部产生以下注册态：

1.共享偏移\(\widehat{\mathbf b}\in\mathbb R^{288}\)；
2.正则低秩/块尺度传输矩阵\(\widehat{\mathbf A}\in\mathbb R^{288\times288}\)及其逆；
3.目标自适应扰动谱\((\mathbf U_t,\boldsymbol\rho_t)\)；
4.每类后验中心\(\mathbf m_c\in\mathbb R^{288}\)和逐维后验方差\(\mathbf v_c\in\mathbb R^{288}\)；
5.由最终support拟合的D92 full/block仿射头。

传输矩阵和后验方差是注册过程中的临时计算对象。传输矩阵对support的作用被吸收到返回的raw坐标仿射头中，因此常驻query状态不额外保存288×288传输矩阵；审计中`persistent_transport_state_bytes=0`表示这一点，而不是说注册时没有矩阵计算。

## 3.符号约定

|符号|意义|
|---|---|
|\(c\)|类别索引；\(c=1,\ldots,6\)表示旧类，\(c>6\)表示本次新注册类|
|\(k\)|类内support索引；\(k=1,\ldots,K\)|
|\(C\)|注册后总类别数|
|\(K\)|每类support数量|
|\(\mathbf z_{c,k}\)|类别\(c\)的第\(k\)条288维目标support|
|\(\mathbf g_c\)|第\(c\)个地面旧类中心；只有旧类存在该对象|
|\(\boldsymbol\mu_c^t\)|目标support计算出的类别中心|
|\(\widehat{\mathbf b}\)|估计的共享目标域偏移|
|\(\mathbf U\)|地面身份扰动子空间基|
|\(\mathbf R\)|低秩身份传输系数矩阵，形状为\(r\times r\)|
|\(s_{\mathrm{id}},s_{\mathrm{fft}},s_{\mathrm{rf}}\)|identity160、FFT96、RF32三块尺度|
|\(\mathbf A\)|把canonical地面坐标映射到目标坐标的传输矩阵|
|\(\mathbf q\)|一条待预测query的288维特征|
|\(\mathbf W,\mathbf a\)|canonical坐标下D92仿射分类系数和截距|
|\(\mathbf W_{\mathrm{raw}},\mathbf a_{\mathrm{raw}}\)|编译到原始query坐标后的分类系数和截距|
|\(\ell_{c,j}\)|类别\(c\)在第\(j\)维的support均值似然方差|
|\(\tau_j^2\)|旧类中心的地面先验方差|
|\(v_{c,j}\)|后验中心方差|
|\(\operatorname{diag}(\mathbf v_c)\)|用逐维后验方差组成的对角矩阵|

## 4.完整计算流程

### 4.1第一步：计算目标support的稳健类别中心

对每个类别，M2.2先调用现有D81类内Cauchy稳健中心规则。普通类别均值为

\[
\bar{\mathbf z}_c=\frac{1}{K}\sum_{k=1}^{K}\mathbf z_{c,k}.
\]

式中，\(\mathbf z_{c,k}\)是一条目标support，\(K\)是该类support数，\(\bar{\mathbf z}_c\)是该类普通均值。该均值只用于计算类内残差，不是最终一定采用的中心。

在固定扰动谱\((\mathbf U,\boldsymbol\rho)\)下，对残差计算投影能量和Cauchy权重，再得到稳健中心\(\widehat{\boldsymbol\mu}_c^t\)。Cauchy步骤不删除support，也不改变标签；它只降低异常support对类别中心的影响。K=1或K=2时仍使用原有低shot安全分支，不因样本协方差不足而失败。

**本式涉及的符号：**\(\widehat{\boldsymbol\mu}_c^t\)是目标类稳健中心；\(\mathbf U\)和\(\boldsymbol\rho\)是冻结的地面谱；“Cauchy权重”是随扰动能量增大而减小的可靠性权重；它不是概率标签，也不是梯度更新系数。

### 4.2第二步：用旧类配对估计共享偏移

旧类地面中心和目标support中心形成6组跨域配对差：

\[
\mathbf d_c=\widehat{\boldsymbol\mu}_c^t-\mathbf g_c,
\qquad c=1,\ldots,6.
\]

式中，\(\mathbf d_c\)是第\(c\)个旧类从地面中心到目标中心的差向量；\(\widehat{\boldsymbol\mu}_c^t\)来自当前目标support；\(\mathbf g_c\)来自不可变Phase1地面聚合中心；\(c\)只遍历6个旧类。新类没有地面配对，因此不直接参与共享偏移的锚点估计。

代码对\(\mathbf d_c\)执行固定步数的稳健加权平均：

\[
\widehat{\mathbf b}
=\frac{\sum_{c=1}^{6}\omega_c\mathbf d_c}
        {\sum_{c=1}^{6}\omega_c},
\qquad
\omega_c\ge0.
\]

式中，\(\widehat{\mathbf b}\in\mathbb R^{160}\)（M2.2的identity偏移部分）是6个旧类共同支持的域偏移；\(\omega_c\)是旧类锚点可靠性权重；分母把权重归一化，避免类数或权重总量改变偏移尺度。M2.1的谱外分量在这里保留，不能强行投影回\(\mathbf U\)子空间。

如果提供完整\(\mathbf G_{288}\)，M2.2还按FFT96和RF32块计算共享偏移；否则\(\widehat{\mathbf b}_{160:288}=\mathbf0\)，表示没有合法的完整地面锚点来辨识辅助块的共同偏移。

### 4.3第三步：拟合低秩身份传输和三块尺度

M2.2不是拟合自由的\(288\times288\)矩阵，而是使用以下结构：

\[
\mathbf A=
\begin{bmatrix}
s_{\mathrm{id}}\mathbf I_{160}+\mathbf U\mathbf R\mathbf U^{\mathsf T}&0&0\\
0&s_{\mathrm{fft}}\mathbf I_{96}&0\\
0&0&s_{\mathrm{rf}}\mathbf I_{32}
\end{bmatrix}.
\]

式中，\(\mathbf A\)是288维传输矩阵；\(\mathbf I_d\)是\(d\times d\)单位矩阵；\(s_{\mathrm{id}}\)、\(s_{\mathrm{fft}}\)、\(s_{\mathrm{rf}}\)分别是三块尺度；\(\mathbf U\in\mathbb R^{160\times r}\)是冻结地面谱基；\(\mathbf R\in\mathbb R^{r\times r}\)是只在该低维谱子空间内估计的修正；零块表示M2.2不引入跨identity/FFT/RF的自由交叉耦合。

矩阵的含义是：若\(\mathbf g_c\)表示canonical地面中心，则目标中心近似满足

\[
\boldsymbol\mu_c^t-\widehat{\mathbf b}
\approx \mathbf A\mathbf g_c.
\]

式中，左侧是去掉共享偏移后的目标中心；右侧是传输矩阵作用于地面中心；近似号表示support噪声和模型残差仍存在。实现采用行向量存储，所以代码把目标样本变换为\((\mathbf z-\widehat{\mathbf b})\mathbf A^{-\mathsf T}\)；若改写成列向量，就是\(\mathbf A^{-1}(\mathbf z-\widehat{\mathbf b})\)。

低秩修正通过带岭正则的最小二乘求解：

\[
\widehat{\mathbf R}
=\arg\min_{\mathbf R}
\sum_{c=1}^{6}\omega_c
\left\|\mathbf y_c-s_{\mathrm{id}}\mathbf g_c
-\mathbf U\mathbf R\mathbf U^{\mathsf T}\mathbf g_c\right\|_2^2
 +\lambda\|\mathbf R\|_F^2.
\]

式中，\(\mathbf y_c=\boldsymbol\mu_c^t-\widehat{\mathbf b}\)是去偏移目标中心；\(\omega_c\)是旧类锚点权重；\(\lambda=10^{-2}\)是固定岭系数；\(\|\mathbf R\|_F\)是Frobenius范数，用于抑制低秩修正过大；\(\mathbf U\mathbf R\mathbf U^{\mathsf T}\mathbf g_c\)只改变地面扰动子空间内的方向。代码还把\(\|\mathbf R\|_F\)限制在0.5以内，并检查\(\mathbf A\)的最小特征值为正，保证逆矩阵存在。

三块尺度由对应块的加权能量比得到，并裁剪到\([0.75,1.25]\)：

\[
s_b=\operatorname{clip}_{[0.75,1.25]}
\left(
\sqrt{\frac{\sum_c\omega_c\|\mathbf y_{c,b}-\bar{\mathbf y}_b\|_2^2}
{\sum_c\omega_c\|\mathbf g_{c,b}-\bar{\mathbf g}_b\|_2^2}}
\right).
\]

式中，\(b\in\{\mathrm{id},\mathrm{fft},\mathrm{rf}\}\)表示三个特征块；\(\mathbf y_{c,b}\)和\(\mathbf g_{c,b}\)分别是目标和地面中心的对应块；\(\bar{\mathbf y}_b\)和\(\bar{\mathbf g}_b\)是按\(\omega_c\)加权的块均值；\(\operatorname{clip}\)把尺度限制在0.75到1.25之间，避免6个锚点把尺度估计推到不稳定的极端。没有完整\(\mathbf G_{288}\)时，FFT96和RF32的尺度不估计，代码固定为1。

### 4.4第四步：把support变换到canonical空间

对任意support，M2.2使用同一共享偏移和同一传输矩阵：

\[
\widetilde{\mathbf z}_{c,k}
=\left(\mathbf z_{c,k}-\widehat{\mathbf b}\right)\mathbf A^{-\mathsf T}.
\]

式中，\(\mathbf z_{c,k}\)是原始目标support；\(\widehat{\mathbf b}\)是注册时由旧类锚点得到的共享偏移；\(\mathbf A^{-\mathsf T}\)是传输矩阵逆的转置，因为代码以行向量存储；\(\widetilde{\mathbf z}_{c,k}\)是canonical坐标support。该公式对旧类和新类使用完全相同的变换；新类不单独拟合一个私有变换。

### 4.5第五步：由跨域残差构造目标自适应扰动谱

先把地面旧类中心也变到canonical空间：

\[
\widetilde{\mathbf g}_c
=\left(\mathbf g_c-\widehat{\mathbf b}\right)\mathbf A^{-\mathsf T}.
\]

式中，\(\widetilde{\mathbf g}_c\)是canonical地面旧类中心；它与变换后的目标旧类中心处于同一坐标系，不能把raw地面中心直接和canonical目标中心相减。

旧类配对残差为

\[
\mathbf r_c
=\widetilde{\boldsymbol\mu}_c^t-\widetilde{\mathbf g}_c,
\qquad
\mathbf C_r
=\frac{\sum_{c=1}^{6}\omega_c\mathbf r_c\mathbf r_c^{\mathsf T}}
{1-\sum_{c=1}^{6}\omega_c^2}.
\]

式中，\(\widetilde{\boldsymbol\mu}_c^t\)是变换后的目标旧类中心；\(\mathbf r_c\)是第\(c\)个跨域残差；\(\mathbf C_r\)是6个旧类残差的二阶矩；分母是加权有限样本修正。这里确实涉及多个旧类，但不是对“一个特征向量”计算协方差：每个旧类提供一个288维中心残差，6个残差共同形成跨类别的方向统计。

冻结地面扰动谱对应的identity协方差为

\[
\mathbf G=\mathbf U\operatorname{diag}(\boldsymbol\rho)\mathbf U^{\mathsf T}.
\]

式中，\(\mathbf G\in\mathbb R^{160\times160}\)是地面身份扰动协方差；\(\operatorname{diag}(\boldsymbol\rho)\)把谱权重放到对角线上；\(\mathbf U\)把这些谱方向放回160维坐标。

M2.2先把\(\mathbf G\)的trace调整到与残差能量同量级，再按有效锚点数进行收缩：

\[
\beta=\frac{n_{\mathrm{eff}}}{n_{\mathrm{eff}}+6},
\qquad
\mathbf G_t=(1-\beta)\mathbf G+\beta\mathbf C_r+\epsilon\mathbf I_{160}.
\]

式中，\(n_{\mathrm{eff}}=1/\sum_c\omega_c^2\)是旧类锚点的有效数量；6是旧类锚点总数；\(\beta\)越大表示越信任目标残差统计；\(\mathbf C_r\)是目标跨域残差二阶矩；\(\epsilon=10^{-10}\)是数值正定保护项；\(\mathbf I_{160}\)是160维单位矩阵。对\(\mathbf G_t\)做特征分解并按参与率保留正方向，得到\((\mathbf U_t,\boldsymbol\rho_t)\)，再将其送回Cauchy稳健中心步骤。

因此，“自适应谱”不是把每一个support当成一个独立协方差矩阵，也不是根据query动态更新；它是由6个旧类中心的跨域残差集合估计出的共享目标扰动方向。

### 4.6第六步：旧类地面先验与新类宽先验后验

在canonical support上，先计算类别似然方差：

\[
\ell_{c,j}
=\max\left(
\frac{1}{n_{c,\mathrm{eff}}}
\frac{1}{K}\sum_{k=1}^{K}
\left(\widetilde z_{c,k,j}-\widehat\mu_{c,j}\right)^2,
\epsilon
\right).
\]

式中，\(j\)是288个特征维度的索引；\(n_{c,\mathrm{eff}}\)是Cauchy权重给出的该类有效样本数；\(\widetilde z_{c,k,j}\)是canonical support的第\(j\)维；\(\widehat\mu_{c,j}\)是稳健中心第\(j\)维；\(\ell_{c,j}\)是该类中心观测的逐维方差；\(\epsilon\)避免K=1时出现精确零方差。

旧类的地面先验方差固定为旧类似然方差中位数的4倍：

\[
\tau_j^2
=4\cdot\operatorname{median}_{c\le6}\ell_{c,j}.
\]

式中，\(\tau_j^2\)是第\(j\)维地面先验方差；4是固定的宽先验倍数；中位数只在6个旧类之间计算，因此不受某一个异常旧类的极端方差支配。旧类在有地面锚点的维度使用高斯共轭融合：

\[
v_{c,j}
=\left(\frac{1}{\tau_j^2}+\frac{1}{\ell_{c,j}}\right)^{-1},
\qquad
m_{c,j}
=v_{c,j}\left(
\frac{\widetilde g_{c,j}}{\tau_j^2}
 +\frac{\widehat\mu_{c,j}}{\ell_{c,j}}
\right).
\]

式中，\(v_{c,j}\)是旧类中心的后验方差；\(m_{c,j}\)是旧类后验中心；\(\widetilde g_{c,j}\)是canonical地面中心；\(\widehat\mu_{c,j}\)是canonical目标稳健中心。M2.2当前默认把旧类identity160作为有地面先验的维度；只有提供完整\(\mathbf G_{288}\)时，FFT96和RF32也启用地面先验。

新类没有地面类别中心，因此不伪造新类先验：其后验中心保持目标support稳健中心，后验方差保持\(\ell_{c,j}\)。这就是“旧类有地面先验、新类使用宽/无类别特定先验”的实际代码语义。

### 4.7第七步：把后验不确定性送入D92协方差

D92先按旧类和新类分别估计自动收缩协方差，然后把类中心后验不确定性作为对角项加入：

\[
\boldsymbol\Sigma_o'
=\boldsymbol\Sigma_o
 +\operatorname{diag}\left(
\frac{1}{6}\sum_{c=1}^{6}\mathbf v_c
\right),
\]

\[
\boldsymbol\Sigma_n'
=\boldsymbol\Sigma_n
 +\operatorname{diag}\left(
\frac{1}{C-6}\sum_{c=7}^{C}\mathbf v_c
\right).
\]

式中，\(\boldsymbol\Sigma_o\)和\(\boldsymbol\Sigma_n\)是D92从旧/新support得到的原始任务协方差；\(\boldsymbol\Sigma_o'\)和\(\boldsymbol\Sigma_n'\)是加入中心不确定性后的协方差；\(\mathbf v_c\)是类别\(c\)的逐维后验方差；\(\operatorname{diag}(\cdot)\)把向量放到对角线上。最后仍使用D92固定等权共享规则：

\[
\boldsymbol\Sigma_{\mathrm{bal}}'
=0.5\boldsymbol\Sigma_o'+0.5\boldsymbol\Sigma_n'.
\]

式中，\(\boldsymbol\Sigma_{\mathrm{bal}}'\)是M2.2传给D92 full/block判别头的共享协方差；0.5和0.5是D92锁定的旧/新任务等权系数，不是M2.2重新搜索出来的超参数。这样做的含义是：中心越不确定，对应维度的共享方差越大，LDA在该方向上的判别权重会相对降低。

### 4.8第八步：把canonical仿射头编译回raw query

D92在canonical特征上得到类别分数：

\[
s_c(\widetilde{\mathbf q})
=\mathbf w_c^{\mathsf T}\widetilde{\mathbf q}+a_c.
\]

式中，\(\widetilde{\mathbf q}\)是canonical query；\(\mathbf w_c\)是canonical坐标下第\(c\)类权重；\(a_c\)是canonical截距；\(s_c\)是该类分数。

对raw query\(\mathbf q\)，M2.2不在query端重新估计任何量，而是把变换代数地吸收到头中：

\[
\mathbf W_{\mathrm{raw}}=\mathbf W\mathbf A^{-1},
\qquad
\mathbf a_{\mathrm{raw}}
=\mathbf a-\mathbf W_{\mathrm{raw}}\widehat{\mathbf b}.
\]

式中，\(\mathbf W\in\mathbb R^{C\times288}\)是所有类别的canonical系数矩阵；\(\mathbf a\in\mathbb R^C\)是canonical截距；\(\mathbf W_{\mathrm{raw}}\)和\(\mathbf a_{\mathrm{raw}}\)是返回给部署端的raw坐标头；\(\mathbf A^{-1}\)是传输矩阵逆；\(\widehat{\mathbf b}\)是共享偏移。于是

\[
\mathbf W\left(\mathbf A^{-1}(\mathbf q-\widehat{\mathbf b})\right)+\mathbf a
=\mathbf W_{\mathrm{raw}}\mathbf q+\mathbf a_{\mathrm{raw}}.
\]

式中，左侧是“先变换query再用canonical头”，右侧是“直接使用编译后的raw头”；两者代数等价。代码用右侧形式，所以query额外传输MAC为0，也不会因query数量增加而更新注册状态。

## 5.逐类注册还是批量注册

M2.2的外部入口一次接收完整注册集合\(\mathbf Z\in\mathbb R^{CK\times288}\)，因此推荐一次批量注册多个新类。内部确实按类别循环计算中心和有效样本数，但传输矩阵只由6个旧类锚点估计一次，D92协方差和最终仿射头对全部注册类联合计算一次。

如果把N个新类拆成N次“一类注册”，每次都会重新计算：

- 旧类与当前support的共享偏移；
- 低秩传输和目标扰动谱；
- 旧/新任务协方差；
- 全部已注册类的D92仿射头。

因此逐类注册不是简单地把最终批量结果拆开；旧类判别行和共享协方差也会随注册集合改变。批量注册通常更省固定矩阵分解开销，也保证同一批新类共享同一个任务平衡协方差。若业务必须逐类到达，可以逐次调用入口，但应把每次结果视作新的注册状态，不能把前一次头和后一次新类行机械拼接。

## 6.计算量、存储和星上部署含义

### 6.1注册阶段

注册阶段主要包括：6个旧类的Cauchy中心、6个锚点差、低秩岭求解、三块尺度、160维特征分解、所有类后验方差、D92 full/block协方差和仿射头。代码将保守估计写入`estimated_registration_macs`；MAC是一次乘法加一次加法的乘加操作单位，不是秒，也不是字节。

M2.2没有反向传播，不更新冻结编码器参数；但由于仍调用D92 full/block收缩协方差和多次组件拟合，注册时不能称为“零计算”。与梯度适配相比，它的优势是没有网络参数、优化器状态、epoch和反向传播图；与M2.1相比，它增加了288维传输矩阵、谱重建和后验不确定性计算。

### 6.2query阶段

query阶段只计算冻结特征并执行最终仿射头：

\[
\mathbf s=\mathbf W_{\mathrm{raw}}\mathbf q+\mathbf a_{\mathrm{raw}}.
\]

式中，\(\mathbf q\)是单条query特征；\(\mathbf W_{\mathrm{raw}}\)是\(C\times288\)常驻系数；\(\mathbf a_{\mathrm{raw}}\)是\(C\)维常驻截距；\(\mathbf s\)是C类分数。传输矩阵不在query端重新相乘，额外query MAC审计为0。

### 6.3精度和常驻状态

M2.2注册内部使用FP64进行中心、低秩和矩阵求解，以减少6个锚点和288维矩阵运算的舍入误差；最终仍沿用D92量化仿射头封存路径。M2.2本身没有把传输矩阵量化成INT8，也没有宣称原生INT8矩阵乘法加速；它只把变换编译进现有D92头。若将来在星上实现，可把注册放在地面或高算力节点完成，仅下传D92已有的常驻头；query端仍只需要小型线性分类计算。

## 7.与M2.1和默认D92 E0的区别

|项目|默认D92 E0|TD-HTRC M2.1|TD-HTRC M2.2|
|---|---|---|---|
|域传输|无显式ground-target传输|160维共享偏移|共享偏移+低秩identity传输+可选三块尺度|
|旧类地面中心|只由原D81路径使用|用于6个旧类偏移锚点|用于偏移、传输、旧类后验和谱收缩|
|辅助块FFT/RF|不做目标传输|保持原值|有完整288维地面中心时可估计块尺度，否则固定1|
|目标扰动谱|冻结地面谱|冻结地面谱|旧类跨域残差与地面谱收缩后的目标谱|
|中心不确定性|不进入D92协方差|不进入D92协方差|以逐类对角方差加入旧/新任务协方差|
|query额外变换|无|偏移编译进截距|完整仿射传输编译进系数和截距|
|默认是否启用|是|否|否|

三者都遵守support-only边界；M2.2不是把D92 E0默认结果改写成新方法，而是一个需显式选择、可审计、可成对实验的升级入口。

## 8.代码入口和验证

核心实现：`code/cvsrffi/stage2_td_htrc_m22.py`。

D92探针入口：`code/scripts/probe_d92_registration_balanced_covariance.py`中的`build_td_htrc_m22_fit`。

通用执行器示例：

```python
state = fit_stage2_ablation(
    ablation_id="P2-E0",
    old_support_features=old_support,
    old_support_labels=old_labels,
    old_classes=old_classes,
    new_support_features=new_support,
    new_support_labels=new_labels,
    new_classes=new_classes,
    ground_basis=ground_basis,
    ground_spectral_weights=ground_weights,
    ground_audit=ground_audit,
    ground_class_centers=ground_old_centers_160,
    ground_full_centers=ground_old_centers_288_or_none,
    module2_mode="td_htrc_m22",
    seed=seed,
    device="cpu",
)
```

`ground_full_centers`可以为`None`；此时M2.2仍然实现identity160传输、目标谱和后验，但FFT96/RF32的块尺度固定为1。代码不允许用query或临时统计量伪造完整Phase1中心。

本次实现的聚焦测试覆盖：M2.2矩阵逆与正定性、完整/不完整地面中心分支、后验旧/新类先验语义、raw-query仿射等价性、D92不确定性注入和通用执行器入口。测试只证明代码闭环和协议边界，不等同于目标域准确率提升；性能结论必须使用同一support/query物理划分的paired实验。
