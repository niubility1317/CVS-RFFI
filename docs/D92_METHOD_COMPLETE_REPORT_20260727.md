# D92 E0完整技术报告：identity160＋FFT96的256维注册方法

版本：2026-08-24

报告对象：D92 E0的P2-A1冻结配置

特征口径：identity160＋FFT96，共256维

适用任务：跨接收机目标域少样本旧类适应与新类注册
证据状态：development screening；不是fresh confirmation，不构成真实在轨部署结论

## 摘要

D92 E0把一条固定received IQ映射为256维联合特征：其中160维来自冻结身份编码器，96维来自确定性频谱描述。注册阶段只读取Phase1封存bundle、当前目标域带标签support和冻结配置；不读取source样本，不用query真值、query角色或整批query类别数量更新状态。方法先从封存的域×类聚合知识中提取“常见跨域漂移方向”，再用该方向集合降低异常support对类中心的影响；随后为旧类任务和新类任务分别估计收缩协方差，以固定50%∶50%的任务权重形成共享判别几何；它同时保留完整协方差与两块对角协方差，并以support内留一结果逐类融合；最后把全部类别的判别行编译为双层INT8系数、尺度和截距，得到可冻结的单一预测状态。

这是一种完整的support-only注册方法，不是训练期域适应算法的补丁，也不是逐query近邻检索器。它确实利用Phase1的跨域知识给support中心估计提供方向先验，但不会对目标域编码器做梯度更新、对抗对齐或分布匹配。因此，它的作用应表述为“注册期的域扰动感知稳健化”，不能表述为已经启动了训练式域对齐。

本文档只使用identity160＋FFT96、256维配置的机制、资源与实验证据。其他特征配置的数值不进入本文的结论。

## 1.问题、边界与输出

### 1.1任务对象

Phase1在地面训练并封存身份编码器与聚合知识。Phase2面对目标接收机已经收到的固定LEO弱信道IQ。注册类别集合由旧类集合\(\mathcal Y_{\mathrm o}\)和本次新类集合\(\mathcal Y_{\mathrm n}\)组成：

$$
\mathcal Y=\mathcal Y_{\mathrm o}\cup\mathcal Y_{\mathrm n},
\qquad
\mathcal Y_{\mathrm o}\cap\mathcal Y_{\mathrm n}=\varnothing,
\qquad
C=|\mathcal Y|.
$$

**符号说明：**\(\mathcal Y_{\mathrm o}\)是Phase1已见、在目标域重新提供合法support的旧类集合；\(\mathcal Y_{\mathrm n}\)是本次登记的新类集合；\(\varnothing\)表示空集；\(C\)是注册完成后需要同时竞争的类别数；\(|\cdot|\)表示集合中元素的数量。

每个类别\(c\in\mathcal Y\)有\(K\)条互不重复的物理support：

$$
\mathcal S_c=
\left\{
\left(\mathbf x^{\mathrm{recv}}_{c,k},c\right)
\right\}_{k=1}^{K}.
$$

**符号说明：**\(\mathcal S_c\)是类别\(c\)的support集合；\(\mathbf x^{\mathrm{recv}}_{c,k}\)是第\(k\)条固定received IQ；\(k\)是类内shot索引；\(K\)是每类support数量；\((\mathbf x^{\mathrm{recv}}_{c,k},c)\)中的第二个元素是合法support标签。对同一物理样本计算不同数学view不增加\(K\)。

### 1.2输入、状态与输出

|阶段|输入|允许做的事|输出|
|---|---|---|---|
|注册|冻结编码器、Phase1 bundle、所有注册类support、固定配置|构造一次性的256维分类状态|类别表、量化判别行、尺度、截距、审计字段|
|单条query|一条固定received IQ和已冻结状态|只做前向特征、解码与全类打分|唯一预测类别与分数|
|评分|不可变预测artifact和独立truth连接|计算准确率、调和均值、遗忘等指标|报告指标，不回流注册状态|

最终query决策为

$$
\widehat y(\mathbf x)=
\arg\max_{c\in\mathcal Y}
s_c\!\left(\Phi_\theta(\mathbf x)\right).
$$

**符号说明：**\(\mathbf x\)是一条query received IQ；\(\Phi_\theta\)是冻结的256维特征映射；\(s_c(\cdot)\)是类别\(c\)的已冻结判别分数；\(\arg\max\)返回得分最大的类别索引；\(\widehat y\)是唯一预测。该式对每条query独立执行，不包含query真值、old/new角色、类别配额或跨query全局重分配。

### 1.3四状态指标口径

|状态|含义|允许报告的主要指标|
|---|---|---|
|DA0_REG0|域适应前、未注册新类|旧类准确率、资源状态|
|DA1_REG0|域适应后、未注册新类|旧类准确率、资源状态|
|DA0_REG1|域适应前、已注册新类|旧类准确率、新类准确率、调和均值|
|DA1_REG1|域适应后、已注册新类|旧类准确率、新类准确率、调和均值、遗忘|

本文的D92 E0注册结果属于\(DA1\_REG1\)口径。需要注意：D92 E0本身不在Phase2执行训练式域适应；表中的\(DA1\)表示实验row使用的已冻结适应状态，而不是说模块二对编码器进行了在线优化。

## 2.总流程：六个模块如何接起来

~~~text
固定received IQ
  ↓
模块一：identity160＋FFT96→256维联合特征
  ↓
模块二：Phase1扰动基→稳健类别中心
  ↓
模块三：每类自动收缩协方差
  ↓
模块四：旧/新任务均衡→等先验LDA行
  ↓
模块五：full/block两种几何的support内留一融合
  ↓
模块六：双层INT8编译→不可变全类预测状态
  ↓
单条query独立全类判别
~~~

六个模块并不都在每次query出现时运行。模块一在注册与query时都运行；模块二到模块六只在注册期运行。注册完成后，query不再需要重新计算协方差、特征分解、留一交叉熵或任何梯度。

## 3.统一符号表

|符号|形状或取值|含义|
|---|---:|---|
|\(\mathbf x^{\mathrm{recv}}\)|复IQ序列|固定received IQ|
|\(E_\theta\)|冻结映射|160维身份编码器|
|\(\mathbf f^{\mathrm{id}}\)|160维|身份特征块|
|\(\mathbf f^{\mathrm{fft}}\)|96维|频谱描述块|
|\(\mathbf z\)|256维|模块一的联合特征|
|\(\widetilde{\mathbf z}_{c,k}\)|256维|模块二稳健化后的第\(k\)条support|
|\(\mathbf U\)|\(160\times r\)|Phase1聚合知识派生的扰动基|
|\(\boldsymbol\rho\)|\(r\)维|扰动方向的归一化谱权重|
|\(\boldsymbol\mu_c\)|256维|类别\(c\)的稳健化support均值|
|\(\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_c\)|\(256\times256\)|类别\(c\)的自动收缩协方差|
|\(\boldsymbol\Sigma_{\mathrm o},\boldsymbol\Sigma_{\mathrm n}\)|\(256\times256\)|旧类任务、新类任务协方差|
|\(\boldsymbol\Sigma_{\mathrm{bal}}\)|\(256\times256\)|固定任务均衡协方差|
|\(\mathbf w_c,b_c\)|256维、标量|类别\(c\)的仿射判别行与截距|
|\(\eta_{c,h}\)|\([0,1]\)|类别\(c\)对几何分支\(h\)的可靠性权重|
|\(\mathbf Q^{(1)},\mathbf Q^{(2)}\)|INT8矩阵|两层量化判别系数|

## 4.模块一：从一条IQ得到256维联合特征

### 4.1两条确定性路径

对任意固定received IQ，冻结编码器给出身份块：

$$
\mathbf f^{\mathrm{id}}
=E_\theta\!\left(\mathbf x^{\mathrm{recv}}\right)
\in\mathbb R^{160}.
$$

**符号说明：**\(\mathbf f^{\mathrm{id}}\)是160维实数身份特征；\(E_\theta\)的参数\(\theta\)在Phase2冻结；\(\mathbf x^{\mathrm{recv}}\)是已经接收并固定的复IQ；\(\mathbb R^{160}\)表示160维实数向量空间。

同一条IQ经固定频谱计算得到

$$
\mathbf f^{\mathrm{fft}}\in\mathbb R^{96}.
$$

**符号说明：**\(\mathbf f^{\mathrm{fft}}\)是96维频谱描述；它由同一条\(\mathbf x^{\mathrm{recv}}\)确定性计算得到；该计算不重新生成信道观测、不访问clean IQ，也不增加shot数。

### 4.2带保护的\(L_2\)归一化

定义

$$
\mathcal N_\varepsilon(\mathbf v)
=
\frac{\mathbf v}
{\max\!\left(\|\mathbf v\|_2,\varepsilon\right)},
\qquad
\varepsilon=10^{-8}.
$$

**符号说明：**\(\mathbf v\)是任意非必然非零的向量；\(\|\mathbf v\|_2=\sqrt{\sum_i v_i^2}\)是欧氏二范数；\(\varepsilon\)是防止零向量除零的正数；\(\max(a,b)\)返回较大的那个数。若\(\mathbf v\ne\mathbf0\)，输出范数为1；若\(\mathbf v=\mathbf0\)，输出保持零向量而不产生数值异常。

### 4.3拼接与固定几何权重

联合特征定义为

$$
\mathbf z
=
\mathcal N_\varepsilon
\left(
\begin{bmatrix}
\mathcal N_\varepsilon(\mathbf f^{\mathrm{id}})\\
4\mathcal N_\varepsilon(\mathbf f^{\mathrm{fft}})
\end{bmatrix}
\right)
\in\mathbb R^{256}.
$$

**符号说明：**\(\mathbf z\)是最终256维联合特征；方括号表示纵向拼接；第一块有160维，第二块有96维，因此总维数为\(160+96=256\)；常数4是冻结的几何权重；外层\(\mathcal N_\varepsilon\)把拼接后的整体范数重新规范化。

若两个内层块都非零，则拼接、外层归一化之前的长度为

$$
\sqrt{1^2+4^2}=\sqrt{17}.
$$

**符号说明：**第一个1来自归一化后的身份块范数；4来自频谱块的固定权重；两块在拼接坐标中正交，因此总平方范数等于两块平方范数之和。外层归一化后，两块总能量比固定为\(1:16\)。这不是“频谱块的每一维都比身份块每一维重要16倍”，而是方法对两个块整体长度的设定。

### 4.4输入输出与边界

|输入|输出|不做什么|
|---|---|---|
|一条固定IQ|一条256维\(\mathbf z\)|不更新\(E_\theta\)，不使用标签，不读取query真值|
|K条某类support|K条256维特征|不把同一物理样本的不同数学view当成多条support|
|一条query|一条256维特征|不写入后续state|

模块一的主干前向通常是query端的主要耗时来源。后面报告的query head MAC只计量化仿射头，不包含冻结主干或FFT96的成本。

## 5.模块二：从封存聚合知识构造扰动基，并稳健化类中心

### 5.1模块二的作用

模块二不把Phase1地面原型当作目标域旧类support，也不让它们直接参加类别匹配。它只从跨域聚合中心的共同漂移中构造一个160维方向坐标系\(\mathbf U\)。当前类别的target support在这个坐标系中计算偏移能量；偏移更集中在常见跨域方向的样本，在计算类中心时获得更小的Cauchy权重。

因此，Phase1 bundle与target support的职责不同：

|来源|提供|不提供|
|---|---|---|
|Phase1 bundle|类无关扰动方向和方向权重|当前target类别中心、query分数、可训练源数据|
|当前target support|普通中心、残差、Cauchy权重、稳健中心|对Phase1 bundle的更新|

### 5.2Phase1 bundle的构成

当前报告对应的封存知识组件为int8_domain_class_center_lowrank_residual_radius_v2。它在Phase1离线阶段将域×类的160维聚合中心压缩保存；注册期只读取以下聚合层信息：

|组件|含义|注册期用途|
|---|---|---|
|域×类中心的压缩码与尺度|恢复近似聚合中心\(\widehat{\mathbf g}_{d,c}\)|计算跨域残差|
|低秩残差与中心项|提高聚合中心恢复精度|构成\(\widehat{\mathbf g}_{d,c}\)|
|P90半径|每个域×类聚合单元的分散度摘要|为跨域单元分配可靠性|
|重构RMSE|离线压缩对完整聚合中心的平均误差摘要|定义量化噪声底|

这里的聚合中心不是Phase2可访问的source样本，也不是模块三中用来估计类内协方差的support行。它们在构造扰动基后不进入query预测状态。

### 5.3聚合扰动协方差\(\mathbf G\)

令\(d\)表示封存地面域、\(c\)表示Phase1旧类。先以本类的跨域加权中心去除类别位置：

$$
\bar{\mathbf g}_c
=
\sum_{d=1}^{D_{\mathrm g}}
\beta_{d,c}\widehat{\mathbf g}_{d,c},
\qquad
\mathbf e^{\mathrm g}_{d,c}
=
\widehat{\mathbf g}_{d,c}-\bar{\mathbf g}_c.
$$

**符号说明：**\(\widehat{\mathbf g}_{d,c}\in\mathbb R^{160}\)是从封存组件恢复的域\(d\)、旧类\(c\)聚合中心；\(D_{\mathrm g}\)是地面域数量；\(\beta_{d,c}\ge0\)是同一类别内的可靠性权重，且\(\sum_d\beta_{d,c}=1\)；\(\bar{\mathbf g}_c\)是该旧类的跨域加权中心；\(\mathbf e^{\mathrm g}_{d,c}\)是域×类单元相对本类中心的残差；上标\(\mathrm g\)表示它来自ground aggregate，而不是target support。

可靠性权重使用跨域漂移能量\(\nu_{d,c}\)与P90半径\(\widehat R_{d,c}\)：

$$
\nu_{d,c}=
\left\|
\widehat{\mathbf g}_{d,c}-\bar{\mathbf g}^{(0)}_c
\right\|_2^2,
\qquad
\gamma_{d,c}=
\frac{\nu_{d,c}}{\nu_{d,c}+2\widehat R_{d,c}},
\qquad
\beta_{d,c}=
\frac{\gamma_{d,c}}{\sum_{d'=1}^{D_{\mathrm g}}\gamma_{d',c}}.
$$

**符号说明：**\(\bar{\mathbf g}^{(0)}_c=D_{\mathrm g}^{-1}\sum_d\widehat{\mathbf g}_{d,c}\)是未加权跨域均值；\(\nu_{d,c}\ge0\)是当前域×类中心相对该均值的漂移能量；\(\widehat R_{d,c}\ge0\)是恢复的P90半径；\(\gamma_{d,c}\)是未归一化可靠性；\(d'\)只是分母中的求和索引；\(\beta_{d,c}\)是归一化后权重。大半径单元在相同漂移能量下获得较小权重，因为其聚合中心本身更分散。

把所有旧类的中心化跨域残差汇总，得到

$$
\mathbf G=
\frac{1}{C_{\mathrm o}}
\sum_{c=1}^{C_{\mathrm o}}
\sum_{d=1}^{D_{\mathrm g}}
\beta_{d,c}
\mathbf e^{\mathrm g}_{d,c}
\left(\mathbf e^{\mathrm g}_{d,c}\right)^{\mathsf T}
\in\mathbb R^{160\times160}.
$$

**符号说明：**\(C_{\mathrm o}\)是Phase1旧类数量；\((\mathbf e^{\mathrm g}_{d,c})^{\mathsf T}\)是转置；\(\mathbf e^{\mathrm g}_{d,c}(\mathbf e^{\mathrm g}_{d,c})^{\mathsf T}\)是\(160\times160\)外积矩阵；\(\mathbf G\)是聚合跨域扰动协方差。每个旧类先去除自己的中心，再按类等权汇总，因此\(\mathbf G\)描述的是共同漂移的形状，而不是某个类别的位置。

\(\mathbf G\)的非对角元素解释为

$$
G_{ij}>0\Rightarrow i,j\text{维在跨域漂移中倾向同向变化},
\qquad
G_{ij}<0\Rightarrow i,j\text{维倾向反向变化}.
$$

**符号说明：**\(G_{ij}\)是\(\mathbf G\)的第\(i\)行第\(j\)列元素；\(i\ne j\)时它是两个不同身份坐标的协方差；正、负号描述中心化偏移的线性共同变化方向。对角项\(G_{ii}\)是第\(i\)维跨域漂移的方差，理论上非负。这个解释来自协方差的定义，不意味着任何单条IQ同时属于两个维度。

### 5.4量化噪声底为什么是重构RMSE平方

离线压缩审计定义重构误差为

$$
\epsilon_{\mathrm{rec}}=
\sqrt{
\frac{1}{D_{\mathrm g}C_{\mathrm o}\cdot160}
\sum_{d=1}^{D_{\mathrm g}}
\sum_{c=1}^{C_{\mathrm o}}
\sum_{i=1}^{160}
\left(g^{\mathrm{dense}}_{d,c,i}
-\widehat g_{d,c,i}\right)^2
},
\qquad
\sigma_{\mathrm q}^2=\epsilon_{\mathrm{rec}}^2.
$$

**符号说明：**\(g^{\mathrm{dense}}_{d,c,i}\)是Phase1离线压缩前完整聚合中心的第\(i\)个坐标；\(\widehat g_{d,c,i}\)是解码后相同坐标；\(\epsilon_{\mathrm{rec}}\)是全部域、类别、坐标误差的均方根；\(\sigma_{\mathrm q}^2\)是该误差的平方。Phase2不读取\(g^{\mathrm{dense}}\)；它只读取封存的\(\epsilon_{\mathrm{rec}}\)标量。

因此，\(\sigma_{\mathrm q}^2\)不是只表示一个整数舍入误差。它是低秩近似、整数码和尺度解码共同造成的每坐标平均重构误差能量的各向同性近似。

### 5.5去噪特征分解与扰动基

先数值对称化并扣除噪声底：

$$
\mathbf G_+
=
\frac{\mathbf G+\mathbf G^{\mathsf T}}{2}
-\sigma_{\mathrm q}^2\mathbf I_{160}.
$$

**符号说明：**\(\mathbf G^{\mathsf T}\)是\(\mathbf G\)的转置；\(\mathbf I_{160}\)是160维单位矩阵；\((\mathbf G+\mathbf G^{\mathsf T})/2\)消除浮点累积引入的微小不对称；\(\sigma_{\mathrm q}^2\mathbf I_{160}\)从每个方向减去相同的平均重构噪声底；\(\mathbf G_+\)是用于寻找主扰动方向的去噪矩阵。

特征分解为

$$
\mathbf G_+\mathbf u_j=\lambda_j\mathbf u_j,
\qquad
\|\mathbf u_j\|_2=1,
\qquad
\mathbf u_i^{\mathsf T}\mathbf u_j=0\ (i\ne j).
$$

**符号说明：**\(\lambda_j\)是第\(j\)个特征值，表示方向强度；\(\mathbf u_j\in\mathbb R^{160}\)是单位特征向量；\(i,j\)是方向索引；正交条件说明保留方向互不重复。一个\(\mathbf u_j\)是160个原始坐标的线性组合，不是某一个原始坐标，也没有单独的物理名称。

保留数值上有效的正谱方向，使用participation ratio确定秩：

$$
r_{\mathrm{eff}}=
\frac{\left(\sum_{j\in\mathcal J_+}\lambda_j\right)^2}
{\sum_{j\in\mathcal J_+}\lambda_j^2},
\qquad
r=\min\!\left(\left\lceil r_{\mathrm{eff}}\right\rceil,\left|\mathcal J_+\right|\right),
\qquad
\mathcal J_+=\{j:\lambda_j>\tau_{\mathrm{eig}}\}.
$$

**符号说明：**\(\mathcal J_+\)是超过数值容差\(\tau_{\mathrm{eig}}\)的正特征值索引集合；\(r_{\mathrm{eff}}\)是有效秩；\(\lceil\cdot\rceil\)表示向上取整；\(|\mathcal J_+|\)表示可用正方向的数量；\(r\)是实际保留方向数。该规则由封存谱一次确定，不搜索query或target标签。

扰动基与谱权重为

$$
\mathbf U=
\begin{bmatrix}
\mathbf u_1&\cdots&\mathbf u_r
\end{bmatrix}
\in\mathbb R^{160\times r},
\qquad
\rho_j=
\frac{\lambda_j}{\sum_{\ell=1}^{r}\lambda_\ell}.
$$

**符号说明：**\(\mathbf U\)的每一列是一条保留方向；\(\rho_j\in(0,1)\)是第\(j\)条方向的归一化强度；\(\ell\)是分母求和索引；\(\sum_j\rho_j=1\)。\(\mathbf U\)和\(\boldsymbol\rho\)只在注册期用于评价support残差。

### 5.6当前support如何获得稳健中心

类别\(c\)的普通身份中心、残差和投影为

$$
\bar{\mathbf z}^{\mathrm{id}}_c=
\frac{1}{K}\sum_{k=1}^{K}\mathbf z^{\mathrm{id}}_{c,k},
\qquad
\mathbf e_{c,k}=
\mathbf z^{\mathrm{id}}_{c,k}-\bar{\mathbf z}^{\mathrm{id}}_c,
\qquad
\mathbf h_{c,k}=\mathbf U^{\mathsf T}\mathbf e_{c,k}.
$$

**符号说明：**\(\mathbf z^{\mathrm{id}}_{c,k}\in\mathbb R^{160}\)是第\(k\)条support的身份块；\(\bar{\mathbf z}^{\mathrm{id}}_c\)是普通身份均值；\(\mathbf e_{c,k}\)是该样本相对均值的160维残差；\(\mathbf h_{c,k}\in\mathbb R^r\)是残差在扰动基中的坐标。

扰动能量、类内尺度与Cauchy权重为

$$
E_{c,k}=\sum_{j=1}^{r}\rho_jh_{c,k,j}^2,
\qquad
\tau_c=\frac{1}{K}\sum_{k=1}^{K}E_{c,k},
\qquad
\omega_{c,k}=
\frac{\left(1+E_{c,k}/\tau_c\right)^{-1}}
{\sum_{k'=1}^{K}\left(1+E_{c,k'}/\tau_c\right)^{-1}}.
$$

**符号说明：**\(E_{c,k}\ge0\)是第\(k\)条support的扰动谱能量；\(h_{c,k,j}\)是其在第\(j\)个扰动方向上的投影；\(\tau_c\)是本类平均能量；\(\omega_{c,k}\)是归一化Cauchy权重；\(k'\)是分母中的support索引。能量较大时权重变小，但样本不被删除，且每类有\(\sum_k\omega_{c,k}=1\)。

稳健身份中心和统一平移量为

$$
\mathbf m_c^{\mathrm{rob}}=
\sum_{k=1}^{K}\omega_{c,k}\mathbf z^{\mathrm{id}}_{c,k},
\qquad
\boldsymbol\delta_c=
\mathbf m_c^{\mathrm{rob}}-\bar{\mathbf z}^{\mathrm{id}}_c.
$$

**符号说明：**\(\mathbf m_c^{\mathrm{rob}}\in\mathbb R^{160}\)是稳健身份中心；\(\boldsymbol\delta_c\in\mathbb R^{160}\)是它相对普通中心的位移；\(\omega_{c,k}\)决定每条support对中心的贡献。

模块二输出为

$$
\widetilde{\mathbf z}_{c,k}
=
\mathbf z_{c,k}+
\begin{bmatrix}
\boldsymbol\delta_c\\
\mathbf0_{96}
\end{bmatrix}
\in\mathbb R^{256}.
$$

**符号说明：**\(\widetilde{\mathbf z}_{c,k}\)是稳健化后的联合support；\(\mathbf z_{c,k}\)是模块一输出；\(\mathbf0_{96}\)是96维零向量；同一类别的所有support加同一个\(\boldsymbol\delta_c\)，所以该平移改变类别位置，不改变类内两两差值。模块二只调整160维身份块；频谱块保持原值。

当\(K\le2\)或\(\tau_c\)退化时，模块二回退为\(\widetilde{\mathbf z}_{c,k}=\mathbf z_{c,k}\)。这是避免极少样本制造不可信稳健性的明确边界。

## 6.模块三：每类自动收缩协方差

### 6.1协方差不是给一条向量内部求协方差

随机向量\(\mathbf Z\in\mathbb R^{256}\)表示从某一类别的目标域support生成机制中抽取的一条联合特征。第\(i\)和第\(j\)个坐标的总体协方差定义为

$$
\Sigma_{ij}
=
\operatorname{Cov}(Z_i,Z_j)
=
\mathbb E\!\left[
(Z_i-\mu_i)(Z_j-\mu_j)
\right],
\qquad
\mu_i=\mathbb E[Z_i].
$$

**符号说明：**\(\mathbf Z\)是随机向量，不是一条已经固定的具体样本；\(Z_i,Z_j\)是其第\(i,j\)个随机坐标；\(\mu_i,\mu_j\)是对应坐标在该类别抽样分布上的均值；\(\mathbb E[\cdot]\)表示对重复抽样的期望；\(\Sigma_{ij}\)是协方差矩阵的一个元素。

在实际注册中，我们没有无限多次抽样，只看到类别\(c\)的\(K\)条support。因此\(\mu_i\)的估计不是某条向量第\(i\)个元素，而是K条support第\(i\)个坐标的平均：

$$
\widehat\mu_{c,i}
=
\frac{1}{K}\sum_{k=1}^{K}\widetilde z_{c,k,i}.
$$

**符号说明：**\(\widehat\mu_{c,i}\)是类别\(c\)第\(i\)维均值的样本估计；\(\widetilde z_{c,k,i}\)是第\(k\)条稳健化support的第\(i\)个坐标；\(K\)是同类support数。它与一条向量的元素不同：前者是跨K行的统计量，后者只是一个观察值。

### 6.2均值、标准化残差与经验协方差

每类的256维均值为

$$
\boldsymbol\mu_c=
\frac{1}{K}\sum_{k=1}^{K}\widetilde{\mathbf z}_{c,k}
\in\mathbb R^{256}.
$$

**符号说明：**\(\boldsymbol\mu_c\)是类别\(c\)的联合特征中心；\(\widetilde{\mathbf z}_{c,k}\)是模块二输出；\(K\)是类内样本数；\(\mathbb R^{256}\)表示均值有256个坐标。

令\(\mathbf D_c\)为逐维标准差构成的对角矩阵，则

$$
\mathbf u_{c,k}
=
\mathbf D_c^{-1}
\left(\widetilde{\mathbf z}_{c,k}-\boldsymbol\mu_c\right),
\qquad
\mathbf S_c^{(u)}
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf u_{c,k}\mathbf u_{c,k}^{\mathsf T}.
$$

**符号说明：**\(\mathbf D_c\in\mathbb R^{256\times256}\)的对角元素是各维support标准差；\(\mathbf D_c^{-1}\)执行逐维标准化；\(\mathbf u_{c,k}\)是标准化残差；\(\mathbf S_c^{(u)}\)是标准化空间的经验协方差；外积\(\mathbf u_{c,k}\mathbf u_{c,k}^{\mathsf T}\)记录同一条support在任意两维上的共同偏离。

### 6.3球形目标与Ledoit–Wolf自动收缩

经验协方差在\(K\ll256\)时最多只有\(K-1\)个独立中心化方向，直接求逆会奇异。D92 E0用球形目标

$$
\mathbf T_c=\zeta_c\mathbf I_{256},
\qquad
\zeta_c=
\frac{\operatorname{tr}(\mathbf S_c^{(u)})}{256}.
$$

**符号说明：**\(\mathbf T_c\)是类别\(c\)的球形收缩目标；\(\mathbf I_{256}\)是256维单位矩阵；\(\operatorname{tr}(\cdot)\)是矩阵迹，即对角元素之和；\(\zeta_c\)是经验协方差的平均对角方差。该目标在标准化空间中让每个方向具有相同方差、让非对角项为0；它不是类别原型，也不是额外生成的support。

自动收缩估计为

$$
\widehat{\boldsymbol\Sigma}^{(u)}_c
=
(1-\alpha_c)\mathbf S_c^{(u)}
+\alpha_c\mathbf T_c,
\qquad
\alpha_c\in[0,1].
$$

**符号说明：**\(\widehat{\boldsymbol\Sigma}^{(u)}_c\)是标准化空间的收缩协方差；\(\alpha_c\)是Ledoit–Wolf从当前类别support自动确定的收缩强度；\(\alpha_c=0\)表示仅用经验协方差；\(\alpha_c=1\)表示完全使用球形目标。

恢复原始联合特征尺度：

$$
\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_c
=
\mathbf D_c
\widehat{\boldsymbol\Sigma}^{(u)}_c
\mathbf D_c.
$$

**符号说明：**\(\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_c\in\mathbb R^{256\times256}\)是类别\(c\)的最终收缩协方差；左右两侧的\(\mathbf D_c\)把标准化空间的尺度恢复到联合特征坐标。球形只严格指标准化空间；恢复尺度后，保守目标一般表现为轴对齐椭球。

## 7.模块四：旧/新任务均衡与等先验LDA

### 7.1为什么要先按任务汇总

新类数量增加时，如果把所有类别直接平均，新类任务将因类别数更多而拥有更大总权重。D92 E0先分别计算旧、新任务的类内协方差均值：

$$
\boldsymbol\Sigma_{\mathrm o}
=
\frac{1}{C_{\mathrm o}}
\sum_{c\in\mathcal Y_{\mathrm o}}
\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_c,
\qquad
\boldsymbol\Sigma_{\mathrm n}
=
\frac{1}{C_{\mathrm n}}
\sum_{c\in\mathcal Y_{\mathrm n}}
\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_c.
$$

**符号说明：**\(C_{\mathrm o}=|\mathcal Y_{\mathrm o}|\)是旧类数；\(C_{\mathrm n}=|\mathcal Y_{\mathrm n}|\)是新类数；\(\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_c\)是单类收缩协方差；\(\boldsymbol\Sigma_{\mathrm o}\)和\(\boldsymbol\Sigma_{\mathrm n}\)是两个任务内部的类别等权平均。

共享协方差固定为

$$
\boldsymbol\Sigma_{\mathrm{bal}}
=
\frac{1}{2}\boldsymbol\Sigma_{\mathrm o}
+\frac{1}{2}\boldsymbol\Sigma_{\mathrm n}.
$$

**符号说明：**\(\boldsymbol\Sigma_{\mathrm{bal}}\)是旧、新任务总权重各为50%的共享协方差；两个\(1/2\)是方法定义，不由query准确率拟合。它不表示每个类别有相同权重：每个旧类权重为\(0.5/C_{\mathrm o}\)，每个新类权重为\(0.5/C_{\mathrm n}\)。

### 7.2两种几何

full分支保留全部坐标关系：

$$
\boldsymbol\Sigma_{\mathrm{full}}
=
\boldsymbol\Sigma_{\mathrm{bal}}.
$$

**符号说明：**\(\boldsymbol\Sigma_{\mathrm{full}}\)是full几何的协方差；它包括身份块内部、频谱块内部以及两个块之间的全部协方差元素。

block分支只保留两块内部关系：

$$
\boldsymbol\Sigma_{\mathrm{blk}}
=
\begin{bmatrix}
\boldsymbol\Sigma_{\mathrm{id}}&\mathbf0_{160\times96}\\
\mathbf0_{96\times160}&\boldsymbol\Sigma_{\mathrm{fft}}
\end{bmatrix}.
$$

**符号说明：**\(\boldsymbol\Sigma_{\mathrm{id}}\in\mathbb R^{160\times160}\)是身份块的协方差子矩阵；\(\boldsymbol\Sigma_{\mathrm{fft}}\in\mathbb R^{96\times96}\)是频谱块的协方差子矩阵；两个零矩阵表示block分支不使用跨块相关性。置零不是在宣称真实数据独立，而是在少样本下采用更保守的判别结构。

两种协方差都须满足

$$
\lambda_{\min}(\boldsymbol\Sigma)>0.
$$

**符号说明：**\(\lambda_{\min}(\boldsymbol\Sigma)\)是当前协方差矩阵的最小特征值；\(\boldsymbol\Sigma\)可代表full或block分支；严格为正表示正定，可以稳定地求解线性系统。若数值检查失败，状态构造失败闭合，不以伪逆静默改变方法。

### 7.3等先验LDA仿射行

在等先验高斯模型下，类别\(c\)的分数可写为

$$
s_c^{(h)}(\mathbf q)
=
\mathbf q^{\mathsf T}\mathbf w_c^{(h)}
+b_c^{(h)},
\qquad
\mathbf w_c^{(h)}
=
\left(\boldsymbol\Sigma_h\right)^{-1}\boldsymbol\mu_c,
\qquad
b_c^{(h)}
=
-\frac12
\boldsymbol\mu_c^{\mathsf T}
\left(\boldsymbol\Sigma_h\right)^{-1}
\boldsymbol\mu_c,
$$

其中\(h\in\{\mathrm{full},\mathrm{blk}\}\)。

**符号说明：**\(\mathbf q\in\mathbb R^{256}\)是当前query特征；\(\boldsymbol\Sigma_h\)是分支\(h\)的共享协方差；\(\mathbf w_c^{(h)}\in\mathbb R^{256}\)是类别\(c\)的判别方向；\(b_c^{(h)}\)是截距；\(\boldsymbol\mu_c\)是类别中心；上标\(\mathsf T\)表示转置。实现使用线性方程求解，不显式形成逆矩阵。

## 8.模块五：support内留一的双几何融合

### 8.1为什么不把full或block固定为唯一答案

full分支能利用跨块相关，表达力较强；block分支只使用块内关系，估计方差较低。少样本下，这两种偏好会随类别而变化。D92 E0不使用query来选择分支，而是让support轮流作为未参与本折拟合的held样本。

第\(t\)折held集合为

$$
\mathcal H_t=
\{(c,t):c\in\mathcal Y\},
\qquad
\mathcal S_{-t}=
\mathcal S\setminus\mathcal H_t,
\qquad
t=1,\ldots,K.
$$

**符号说明：**\(\mathcal S\)是全部注册support集合；\(\mathcal H_t\)从每个类别各取第\(t\)条support；\(\mathcal S_{-t}\)是删除该折held后的拟合集；\(\setminus\)表示集合差；每折保留每类\(K-1\)条support。每一折重新计算与support有关的稳健中心、协方差和LDA行，避免held样本通过统计量间接泄漏。

### 8.2类别级留一交叉熵

对类别\(c\)、分支\(h\)，留一损失为

$$
\ell_{c,h}^{\mathrm{LOO}}
=
-\frac{1}{K}
\sum_{t=1}^{K}
\log
\frac{
\exp\!\left(s_{c,t,c}^{(h)}/r_h\right)
}{
\sum_{j=1}^{C}
\exp\!\left(s_{c,t,j}^{(h)}/r_h\right)
}.
$$

**符号说明：**\(\ell_{c,h}^{\mathrm{LOO}}\)是类别\(c\)在分支\(h\)的平均留一交叉熵；\(s_{c,t,j}^{(h)}\)是第\(t\)折held的类别\(c\)样本对候选类别\(j\)的logit；\(r_h>0\)是该分支从support logits估计的RMS尺度；\(C\)是注册类别数；分子是真实类别的指数分数，分母是全部候选类的指数分数和。损失越小，说明该几何在未见的本类support上更可信。

分支权重为

$$
\eta_{c,h}
=
\frac{\exp\!\left(-K\ell_{c,h}^{\mathrm{LOO}}\right)}
{\sum_{h'\in\{\mathrm{full},\mathrm{blk}\}}
\exp\!\left(-K\ell_{c,h'}^{\mathrm{LOO}}\right)}.
$$

**符号说明：**\(\eta_{c,h}\)是类别\(c\)对分支\(h\)的可靠性权重；\(h'\)是分母中的分支索引；\(-K\ell_{c,h}^{\mathrm{LOO}}\)对应K个held样本的总对数证据。对固定类别\(c\)，有\(\eta_{c,\mathrm{full}}+\eta_{c,\mathrm{blk}}=1\)。

### 8.3融合成一条最终仿射行

$$
\mathbf w_c^{(0)}
=
\eta_{c,\mathrm{full}}
\frac{\mathbf w_c^{(\mathrm{full})}}{r_{\mathrm{full}}}
+
\eta_{c,\mathrm{blk}}
\frac{\mathbf w_c^{(\mathrm{blk})}}{r_{\mathrm{blk}}},
\qquad
b_c^{(0)}
=
\eta_{c,\mathrm{full}}
\frac{b_c^{(\mathrm{full})}}{r_{\mathrm{full}}}
+
\eta_{c,\mathrm{blk}}
\frac{b_c^{(\mathrm{blk})}}{r_{\mathrm{blk}}}.
$$

**符号说明：**\(\mathbf w_c^{(0)}\)和\(b_c^{(0)}\)是量化前的基础融合行；\(\mathbf w_c^{(\mathrm{full})},b_c^{(\mathrm{full})}\)与\(\mathbf w_c^{(\mathrm{blk})},b_c^{(\mathrm{blk})}\)是两个分支输出；\(r_{\mathrm{full}},r_{\mathrm{blk}}\)先消除分支logit绝对尺度差异；\(\eta_{c,h}\)再逐类加权。权重只由当前row的support留一结果产生，不读取outer held或query。

当\(K\le2\)时，留一链不满足稳定构造条件，方法使用冻结回退，不把训练内分数伪装成留一证据。

## 9.模块六：量化编译与不可变预测状态

### 9.1两层残差INT8量化

对类别\(c\)第\(j\)个浮点权重\(w_{c,j}\)，第一层量化为

$$
q_{c,j}^{(1)}
=
\operatorname{clip}
\left(
\operatorname{round}
\left(
\frac{w_{c,j}}{s_{c,g(j)}^{(1)}}
\right),
-127,127
\right),
\qquad
\widehat w_{c,j}^{(1)}
=
s_{c,g(j)}^{(1)}q_{c,j}^{(1)}.
$$

**符号说明：**\(q_{c,j}^{(1)}\)是第一层INT8码；\(w_{c,j}\)是原始浮点权重；\(g(j)\)返回第\(j\)维所属量化组；\(s_{c,g(j)}^{(1)}>0\)是该组的第一层尺度；\(\operatorname{round}\)表示四舍五入；\(\operatorname{clip}\)把整数限制在对称INT8范围；\(\widehat w_{c,j}^{(1)}\)是第一层解码近似。

量化残差和第二层为

$$
e_{c,j}=w_{c,j}-\widehat w_{c,j}^{(1)},
\qquad
q_{c,j}^{(2)}
=
\operatorname{clip}
\left(
\operatorname{round}
\left(
\frac{e_{c,j}}{s_{c,g(j)}^{(2)}}
\right),
-127,127
\right),
\qquad
\widehat w_{c,j}
=
s_{c,g(j)}^{(1)}q_{c,j}^{(1)}
+s_{c,g(j)}^{(2)}q_{c,j}^{(2)}.
$$

**符号说明：**\(e_{c,j}\)是第一层未表达的残差；\(q_{c,j}^{(2)}\)是第二层INT8码；\(s_{c,g(j)}^{(2)}\)是第二层尺度；\(\widehat w_{c,j}\)是最终解码权重。双层结构是两组INT8近似相加，不是把单个元素改写为INT16。

### 9.2状态内容与query计算

冻结状态抽象为

$$
\mathcal S_{\mathrm{E0}}
=
\left(
\mathcal C,
\mathbf Q^{(1)},
\mathbf Q^{(2)},
\mathbf S^{(1)},
\mathbf S^{(2)},
\widehat{\mathbf b},
\mathbf m_{\mathrm o},
\mathcal A
\right).
$$

**符号说明：**\(\mathcal C\)是有序类别表；\(\mathbf Q^{(1)},\mathbf Q^{(2)}\)是两层INT8系数矩阵；\(\mathbf S^{(1)},\mathbf S^{(2)}\)是尺度；\(\widehat{\mathbf b}\)是FP16截距；\(\mathbf m_{\mathrm o}\)是旧类适配使用的冻结对角metric；\(\mathcal A\)是配置、绑定与数值审计字段。状态不保存历史support IQ、query样本、query真值或可更新的query统计量。

query分数为

$$
s_c^{\mathrm q}(\mathbf q)
=
\mathbf q^{\mathsf T}\widehat{\mathbf w}_c+\widehat b_c,
\qquad
\widehat y=\arg\max_{c\in\mathcal C}s_c^{\mathrm q}(\mathbf q).
$$

**符号说明：**\(\mathbf q\in\mathbb R^{256}\)是当前query联合特征；\(\widehat{\mathbf w}_c\)和\(\widehat b_c\)是从冻结量化状态解码出的类别行；\(s_c^{\mathrm q}\)是类别分数；\(\widehat y\)是最终预测。当前实现的主要价值是状态压缩；若运行时先解码为浮点再乘加，就不能把INT8存储压缩误写成INT8算力加速。

## 10.注册是逐类统计，还是整批重编译

两种说法都正确，但对应不同层次：

1.每条support先独立提取特征；可串行也可批量前向。
2.模块二和模块三按类别分别计算中心、权重和类内协方差。
3.模块四开始把本批全部旧类和新类一起汇总；共享协方差、LDA、留一融合和量化状态都重新编译。

设本次一起注册\(N\)个新类，则

$$
C=C_{\mathrm o}+N,
\qquad
\boldsymbol\Sigma_{\mathrm n}^{(N)}
=
\frac1N\sum_{n=1}^{N}
\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_n,
\qquad
\boldsymbol\Sigma_{\mathrm{bal}}^{(N)}
=
\frac12\boldsymbol\Sigma_{\mathrm o}
+\frac12\boldsymbol\Sigma_{\mathrm n}^{(N)}.
$$

**符号说明：**\(N\)是本次批量登记的新类数量；\(C_{\mathrm o}\)是旧类数；\(C\)是总注册类数；\(\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_n\)是第\(n\)个新类的类内协方差；\(\boldsymbol\Sigma_{\mathrm n}^{(N)}\)是本批新类的任务协方差；\(\boldsymbol\Sigma_{\mathrm{bal}}^{(N)}\)是注册后共享几何。

这意味着增加新类并非只增加一条新类别行。它还会改变共享协方差，进而改变旧类和新类全部判别行。一次一个新类连续到达时，每次都可按同一算法重新注册；但这不是低成本的局部追加。一次同时登记多个新类时，局部统计量数量随\(N\)增加，而一次共享协方差求解、全类LDA和留一重建则作用于更大的总类集合。

## 11.计算量、存储与星上适用性

### 11.1MAC是什么

MAC是multiply–accumulate的缩写：一次乘法后立即加到累加器中，记作一次乘加。长度为\(p\)的向量与一条长度\(p\)的权重行做点积，约需要\(p\)MAC；面对\(C\)个类别，仿射头的理想化规模约为

$$
\mathcal M_{\mathrm{head}}\approx C p,
\qquad p=256.
$$

**符号说明：**\(\mathcal M_{\mathrm{head}}\)是每条query的分类头MAC数量；\(C\)是注册类别数；\(p=256\)是联合特征维数。该量只描述点积头，不包含编码器、FFT96、内存搬运、解码、控制流或外部I/O。

### 11.2注册与query不能混为一个数字

|开销阶段|主要计算|随什么增长|特点|
|---|---|---|---|
|support特征|冻结主干与FFT96|support总数\(CK\)|可批处理，通常主干占主导|
|扰动基|160维聚合矩阵与特征分解|封存域×类规模|一次注册一次，不随query重复|
|单类协方差|每类256维残差与收缩|类别数与\(K\)|按类独立|
|共享几何与LDA|矩阵分解/线性求解|特征维数和总类别数|注册期重计算|
|留一融合|K次重建两种几何|\(K\)、\(C\)、256维|注册期最重部分之一|
|query头|全类仿射点积|类别数\(C\)|轻量、每query重复|

### 11.3当前256维配置的资源证据

下表来自P2-A1已完成screening的资源审计；它不是目标星上硬件实测。

|项目|均值/中位数/最大值|解释|
|---|---:|---|
|完整registration time|235699.082/276033.354/676693.398ms|包括D92 E0完整注册链，不能当作纯LDA微基准|
|deployment state|17216/17216/17216B|封存部署状态总大小|
|state bytes|14722/16492/16492B|资源表中的压缩状态计数|
|query head MAC|6624/7488/7488|只计判别头审计口径|
|row peak RSS|1146965524/1148645376/1183404032B|约1.068/1.070/1.102GiB的进程峰值|
|row peak VRAM|17443533/17405952/17715200B|约16.64/16.60/16.89MiB的GPU显存峰值|

结论分两层：

- 注册时不轻。原因是收缩协方差、两种几何、K折重建和量化编译都集中发生在这一阶段。
- query头轻。冻结后每条query只需256维特征、量化行解码和全类点积；无需重算协方差或留一。

从只比较分类头计算量的角度，它通常比每query对所有类别、每类全部support都计算距离的KNN式方法轻。若qKNN的身份距离维数为\(d\)，则其量级约为

$$
\mathcal M_{\mathrm{qKNN}}\approx C K d.
$$

**符号说明：**\(\mathcal M_{\mathrm{qKNN}}\)是qKNN的query距离计算量级；\(C\)是类别数；\(K\)是每类support数；\(d\)是距离特征维数。与D92 E0的\(Cp\)头相比，qKNN额外随\(K\)线性增长；但它不一定承担D92 E0的注册期协方差与留一成本，二者应分别比较注册、状态和query三个阶段。

### 11.4量化、FP64与部署优化

当前算法中，特征分解、正定性检查和协方差求解是数值敏感的注册期操作；FP64有助于避免近奇异矩阵带来的误判。更合理的星上优化顺序是：

1.在地面或高资源端完成bundle派生与数值敏感注册。
2.将已验证的量化仿射状态下传。
3.query端使用INT8或混合精度kernel执行已冻结头。
4.先做量化误差、类别排序一致性和端到端时延验证，再宣称整数加速。

不能把状态中包含INT8码直接等同于所有协方差计算都可以安全替换为INT8。对注册期的矩阵分解，低比特替换需要专门的数值稳定性验证；对query头，则更适合做专门的整数kernel与校准。

## 12.实验结果、模块消融与证据边界

### 12.1当前配置的screening结果

P2-A1固定identity160＋FFT96和完整B–F注册链，在75个logical identity row、225个场景单元的development screening中得到：

|指标|均值±总体标准差/中位数|
|---|---:|
|\(A_{\mathrm o}^{DA1\_REG0}\)|0.749444±0.132670/0.766667|
|\(A_{\mathrm o}^{DA1\_REG1}\)|0.594630±0.168673/0.616667|
|\(A_{\mathrm n}^{DA1\_REG1}\)|0.548556±0.205489/0.582500|
|\(H_{\mathrm{old,new}}\)|0.565038±0.186408/0.581880|
|\(F\)|0.154815±0.071628/0.150000|
|最低旧类准确率|0.275333±0.195512/0.250000|
|最低新类准确率|0.222222±0.220695/0.150000|

其中

$$
H_{\mathrm{old,new}}
=
\frac{2A_{\mathrm o}^{DA1\_REG1}A_{\mathrm n}^{DA1\_REG1}}
{A_{\mathrm o}^{DA1\_REG1}+A_{\mathrm n}^{DA1\_REG1}},
\qquad
F=
A_{\mathrm o}^{DA1\_REG0}-A_{\mathrm o}^{DA1\_REG1}.
$$

**符号说明：**\(A_{\mathrm o}^{DA1\_REG0}\)是注册前旧类准确率；\(A_{\mathrm o}^{DA1\_REG1}\)是注册后旧类准确率；\(A_{\mathrm n}^{DA1\_REG1}\)是注册后新类准确率；\(H_{\mathrm{old,new}}\)是两者调和均值；\(F\)是旧类准确率下降量。调和均值会惩罚只提高一侧而另一侧很低的结果。

这些结果是开发筛选证据：目前只有3个development seed和1个new-class draw，不能替代fresh confirmation、多draw稳定性或真实星载硬件测量。

### 12.2六个模块的同配置消融状态

|模块|当前256维配置中的状态|可以得出的结论|
|---|---|---|
|一：联合特征|P2-A1完整路径已完成screening|当前报告的特征输入与性能数字以此为准|
|二：扰动基与稳健中心|完整链中启用|尚无同配置的单模块去除结果，不将其他配置的数字移入本文|
|三：自动收缩协方差|完整链中启用|其必要性由\(K\ll256\)的可逆性问题直接支持；独立效果待同配置实验|
|四：任务均衡与LDA|完整链中启用|方法固定旧/新任务各占50%；独立效果待同配置实验|
|五：双几何留一融合|完整链中启用|避免预设全局几何开关；独立效果待同配置实验|
|六：量化编译|完整链中启用并有资源审计|可确认状态压缩；尚无目标硬件整数kernel证据|

这张表刻意不把不同特征配置的模块消融数值拼入256维结论。缺失的同配置消融应通过新的冻结矩阵补齐，而不是用不匹配的历史行代替。

## 13.与论文复现方法的比较

### 13.1与域适应方法

|方法类别|典型机制|训练/状态更新时机|与D92 E0的可比性|
|---|---|---|---|
|MRIOR-SDA|目标域适应、迭代优化或可靠样本机制|训练或适应阶段|可比较旧类目标域性能；不等价于同一support-only注册权限|
|DADDA-SDA|域判别与对抗式表征对齐|训练阶段，需要优化器|重点是缩小域分布差异；D92 E0不做梯度对齐|
|ProtoNet CDA|support原型与度量推理|episode内构造原型|可比较少样本分类；D92 E0额外建模协方差、旧/新任务均衡和量化状态|

MRIOR、DADDA等训练式域适应方法通过更新表征或显式优化对齐目标来减少域差异。D92 E0的模块二只利用冻结Phase1跨域漂移方向来重新估计target support中心，既不改变编码器参数，也不将target分布与source分布重新匹配。因此，D92 E0有域扰动感知效果，但不是启动域对齐。

### 13.2与类增量方法

|方法类别|典型机制|新类到达后的代价结构|与D92 E0的差异|
|---|---|---|---|
|CSIL类增量|训练、蒸馏、回放或分类器扩展|主要成本在重训练或更新epoch|D92 E0不做梯度更新，只用support重编译判别状态|
|MoPC-HR类增量|原型维护与类间关系约束|维护原型或记忆状态|D92 E0同时重估共享协方差，旧类行不只是追加|
|Orthogonal Incremental SEI|增量表示约束与分类头学习|需要学习阶段|D92 E0不学习新投影，而是用冻结特征和闭式判别|
|qKNN|保存support并在query时计算近邻距离|query成本随\(CK\)增长|D92 E0把主要复杂度移到注册期，query使用固定仿射头|

这些方法的训练权限、是否可使用base/source数据、是否保留历史样本、是否允许梯度更新均不同。正式实验比较必须在相同receiver、K、新类规模、LEO场景与评分规则下报告；不能只摘取某一篇论文的最佳数值并与本文screening平均值排名。

## 14.优点、限制与适用结论

### 14.1方法优势

- support和query职责清楚：所有可变统计量只由注册support形成。
- 对极少样本协方差有明确的收缩与正定性处理。
- 旧类与新类在共享几何中总权重相等，不因新类数量增多自动压低旧任务。
- 注册完成后，query分类头是固定的全类仿射计算，不需要逐query重建近邻集合。
- 状态量化可减少常驻存储，并保留可审计的类别顺序和数值检查。

### 14.2方法限制

- 256维协方差与K折重建使注册期成本显著，不能把它描述为轻量在线学习。
- Cauchy稳健中心只处理已知跨域扰动方向上的support可靠性，不保证消除所有目标域偏移。
- full/block融合只用support内留一证据；\(K\le2\)时必须回退。
- 当前资源数据来自服务器screening，不是抗辐照、功耗、实时性或航天级硬件验证。
- 当前结果仍缺少同配置的完整模块单因素消融和fresh confirmation。

### 14.3星上部署判断

若“星上部署”指注册后只执行冻结编码器、FFT96和量化仿射头，D92 E0具有可部署的结构：query不需要矩阵分解、留一或梯度更新。若“星上部署”还要求在极低资源节点实时完成新类注册，则当前完整注册链仍偏重，宜迁移到地面站、边缘高资源节点或离线维护窗口。是否真正适合具体星载平台，必须以目标硬件的功耗、时延、内存、整数kernel和辐射环境测试为准。

## 15.可复核证据与报告维护原则

本文的实验和资源表引用仓库内已完成的P2-A1 artifact汇总：docs/D92_E0_ALL_ABLATION_EXPERIMENTS_REPORT_20260819.md。该汇总记录75个logical row、225个场景单元的screening结果及资源审计。

本报告只描述identity160＋FFT96、256维D92 E0配置。代码默认配置、其他特征配置、未启动实验和未完成的同配置消融不因本次文档修订而被改写或视为完成。
