# TD-HTRC-M2.1：模块二目标域共享传输—稳健中心升级说明

日期：2026-08-19

## 1.定位与实现边界

TD-HTRC（Target-Domain Hierarchical Transport and Robust Center）是对模块二的可选升级。当前实现的是改造路线中的最低风险版本M2.1：

\[
\boxed{\text{旧类ground-target配对}\rightarrow\text{共享目标域偏移}\rightarrow\text{Cauchy稳健中心}}
\]

它不是对现有D92 E0默认路径的静默替换。默认`build_d92_fit`和已有D92 E0执行器保持原行为；调用`build_td_htrc_fit`或在`fit_stage2_ablation`中显式指定`module2_mode="td_htrc_m21"`时才启用升级。

本次没有实现以下高自由度部分：低秩仿射矩阵、物理nuisance/z-dom回归、目标自适应扰动谱、贝叶斯类别中心后验以及将中心不确定性注入模块三。它们列为后续M2.2/M2.3研究项，不能把本实现描述成完整的TD-HTRC研究主版本。

## 2.要解决的问题

原模块二对每一类目标support先去均值：

\[
\bar{\mathbf z}_{c}^{t}=\frac{1}{K}\sum_{k=1}^{K}\mathbf z_{c,k}^{t},\qquad
\mathbf e_{c,k}=\mathbf z_{c,k}^{t}-\bar{\mathbf z}_{c}^{t}.
\]

式中，\(c\)是注册类别索引，\(k\)是该类的shot索引，\(K\)是每类support数，\(\mathbf z_{c,k}^{t}\in\mathbb R^{288}\)是目标域support特征，\(\bar{\mathbf z}_{c}^{t}\)是该类普通均值，\(\mathbf e_{c,k}\)是类内残差。若目标域给同一类别的所有support增加共同偏移\(\mathbf b_t\)，该偏移会在减去类均值时消失，因此原模块无法从类内残差识别共同目标域偏移。

TD-HTRC-M2.1额外使用6个旧类的地面聚合中心和目标support中心。对旧类集合\(\mathcal Y_o=\{0,\ldots,5\}\)，形成真正的跨域配对：

\[
\mathbf d_c=\mathbf m_c^{t}-\mathbf p_c^{g},\qquad c\in\mathcal Y_o.
\]

式中，\(\mathbf p_c^{g}\in\mathbb R^{160}\)是Phase1不可变聚合bundle中第\(c\)个旧类的160维身份中心，\(\mathbf m_c^{t}\in\mathbb R^{160}\)是当前目标域support形成的旧类稳健中心，\(\mathbf d_c\)是第\(c\)个旧类的跨域中心差。若多个\(\mathbf d_c\)具有共同部分，该共同部分就是目标域共享传输的可识别证据。

## 3.输入与输出

### 3.1输入

|输入|形状|来源|是否允许更新|
|---|---:|---|---|
|目标域注册support特征|\(N\times288\)|冻结编码器对当前support的输出|否，只读|
|support整数标签|\(N\)|当前注册表|否，只读|
|Phase1旧类聚合中心|\(6\times160\)|不可变地面domain×class聚合中心|否，只读|
|地面扰动基\(\mathbf U\)|\(160\times r\)|Phase1聚合扰动谱|否，只读|
|谱权重\(\boldsymbol\rho\)|\(r\)|地面扰动谱归一化权重|否，只读|

这里\(N=C K\)，\(C\)是当前注册类别总数；前6个类别位置是旧类，后面的类别位置是本次新类。Phase1输入是聚合中心，不是地面原始样本、成员样本或query。

### 3.2输出

模块输出包括三部分：

1.共享偏移\(\widehat{\mathbf b}_t\in\mathbb R^{160}\)及其\(160\times160\)协方差估计；
2.经过共享偏移规范化、再经过原Cauchy稳健中心处理的support，交给D92协方差和LDA组件；
3.已经换回raw-query坐标的affine head，即query不需要持久化或重新估计一个变换。

## 4.计算流程

### 4.1目标旧类稳健中心

TD-HTRC先在旧类support上复用原模块二的地面谱Cauchy规则。对旧类\(c\)，先得到160维均值和残差：

\[
\bar{\mathbf z}_{c}^{\mathrm{id}}=\frac{1}{K}\sum_{k=1}^{K}\mathbf z_{c,k}^{\mathrm{id}},\qquad
\mathbf e_{c,k}=\mathbf z_{c,k}^{\mathrm{id}}-\bar{\mathbf z}_{c}^{\mathrm{id}}.
\]

式中，\(\mathbf z_{c,k}^{\mathrm{id}}\in\mathbb R^{160}\)是288维联合特征的identity160块，\(\bar{\mathbf z}_{c}^{\mathrm{id}}\)是该块的普通中心，\(\mathbf e_{c,k}\)是类内身份残差。

投影和扰动能量为：

\[
\mathbf h_{c,k}=\mathbf U^{\mathsf T}\mathbf e_{c,k},\qquad
E_{c,k}=\sum_{j=1}^{r}\rho_j h_{c,k,j}^{2}.
\]

式中，\(\mathbf U\)的第\(j\)列是一个地面扰动方向，\(\mathbf h_{c,k}\in\mathbb R^r\)是残差在扰动基上的坐标，\(h_{c,k,j}\)是第\(j\)个坐标，\(\rho_j>0\)是该方向的归一化谱权重，\(E_{c,k}\)是该support沿已知扰动方向的加权能量。

令\(\tau_c\)为该类能量平均值，未归一化权重和归一化权重分别为：

\[
\tau_c=\frac{1}{K}\sum_{k=1}^{K}E_{c,k},\qquad
a_{c,k}=\frac{1}{1+E_{c,k}/\max(\tau_c,\varepsilon)},\qquad
\omega_{c,k}=\frac{a_{c,k}}{\sum_{\ell=1}^{K}a_{c,\ell}}.
\]

式中，\(\tau_c\)是类内能量尺度，\(a_{c,k}\)是第\(k\)条support的Cauchy可靠性权重，\(\omega_{c,k}\)是归一化后的贡献权重，\(\ell\)只是归一化分母的求和索引，\(\varepsilon\)是数值保护常数。能量越高，\(a_{c,k}\)越小，但样本不会被删除。

旧类目标中心为：

\[
\mathbf m_c^{t}=\sum_{k=1}^{K}\omega_{c,k}\mathbf z_{c,k}^{\mathrm{id}}.
\]

式中，\(\mathbf m_c^{t}\)是目标域旧类中心；它来自当前target support，不是地面prototype，也不读取query。

有效样本数为：

\[
K_{\mathrm{eff},c}=\frac{1}{\sum_{k=1}^{K}\omega_{c,k}^{2}}.
\]

式中，\(K_{\mathrm{eff},c}\)是第\(c\)类的稳健有效样本数。若所有support权重接近\(1/K\)，它接近\(K\)；若某条support被明显降权，它小于\(K\)。当\(K=1\)或\(K=2\)时，原Cauchy步骤按既有规则回退为等权，TD-HTRC仍可使用6个旧类中心差估计共享偏移。

### 4.2共享偏移的稳健估计

旧类中心差为：

\[
\mathbf d_c=\mathbf m_c^{t}-\mathbf p_c^{g}.
\]

代码先按每个旧类的\(K_{\mathrm{eff},c}\)和类内不确定性构造质量权重\(q_c\)，再归一化为初始权重。随后进行固定3步Cauchy IRLS：

\[
\mathbf b^{(0)}=\sum_{c\in\mathcal Y_o}q_c\mathbf d_c,
\]

\[
u_c^{(s)}=\frac{1}{1+\|\mathbf d_c-\mathbf b^{(s)}\|_2^2/\eta^2},\qquad
\widetilde q_c^{(s)}=\frac{q_cu_c^{(s)}}{\sum_{j\in\mathcal Y_o}q_ju_j^{(s)}},
\]

\[
\mathbf b^{(s+1)}=\sum_{c\in\mathcal Y_o}\widetilde q_c^{(s)}\mathbf d_c,qquad s=0,1,2.
\]

式中，\(\mathbf b^{(0)}\)是质量加权初始偏移，\(s\)是IRLS迭代编号，\(\eta\)是由旧类中心差残差的中位数确定的固定尺度，\(u_c^{(s)}\)是第\(s\)步对异常旧类锚点的Cauchy降权，\(\widetilde q_c^{(s)}\)是重新归一化后的旧类权重，\(\mathbf b^{(s+1)}\)是下一步偏移。最终共享偏移为\(\widehat{\mathbf b}_t=\mathbf b^{(3)}\)。迭代次数固定，不使用query指标调参。

M2.1不把偏移强行限制在地面扰动基内：

\[
\widehat{\mathbf b}_t=\mathbf U\mathbf a_t+\widehat{\mathbf b}_t^{\perp},\qquad
\widehat{\mathbf b}_t^{\perp}=(\mathbf I-\mathbf U\mathbf U^{\mathsf T})\widehat{\mathbf b}_t.
\]

式中，\(\mathbf a_t=\mathbf U^{\mathsf T}\widehat{\mathbf b}_t\)是已知地面扰动子空间坐标，\(\widehat{\mathbf b}_t^{\perp}\)是正交于该子空间的谱外目标偏移，\(\mathbf I\)是160维单位矩阵。代码允许\(\widehat{\mathbf b}_t^{\perp}\neq\mathbf0\)，因此不会把地面未覆盖的新receiver或LEO方向误删。

### 4.3偏移不确定性和诊断量

令\(\mathbf r_c=\mathbf d_c-\widehat{\mathbf b}_t\)，最终归一化权重为\(w_c\)，模块给出有限锚点协方差：

\[
\widehat{\mathbf V}_{b}=\frac{\sum_{c\in\mathcal Y_o}w_c\mathbf r_c\mathbf r_c^{\mathsf T}}
{\max\left(1-\sum_{c\in\mathcal Y_o}w_c^2,\,10^{-6}\right)}+10^{-10}\mathbf I.
\]

式中，\(\widehat{\mathbf V}_{b}\in\mathbb R^{160\times160}\)是共享偏移的不确定性估计，\(\mathbf r_c\)是旧类锚点相对共享偏移的残差，\(w_c\)是IRLS后的旧类权重，分母是有限样本自由度保护项，\(10^{-10}\mathbf I\)确保数值正定。该协方差当前只作为注册审计输出，不进入模块三协方差。

实现还输出两个诊断比率：

\[
R_{\mathrm{shared}}=1-\frac{\sum_cw_c\|\mathbf d_c-\widehat{\mathbf b}_t\|_2^2}
{\sum_cw_c\|\mathbf d_c\|_2^2+\varepsilon},
\qquad
R_U=\frac{\|\mathbf U^{\mathsf T}\widehat{\mathbf b}_t\|_2^2}
{\|\widehat{\mathbf b}_t\|_2^2+\varepsilon}.
\]

式中，\(R_{\mathrm{shared}}\)是旧类中心差被共同偏移解释的比例，\(R_U\)是共享偏移落在地面扰动子空间内的比例，\(\varepsilon\)防止零偏移分母为零。两者是诊断指标，不通过query性能选择模块开关。

### 4.4support和query的一致性

support在拟合前被映射到canonical空间：

\[
\mathbf y_{c,k}=\mathbf z_{c,k}-\begin{bmatrix}\widehat{\mathbf b}_t\\\mathbf0_{128}\end{bmatrix}.
\]

式中，\(\mathbf z_{c,k}\in\mathbb R^{288}\)是原始目标support，\(\mathbf y_{c,k}\)是共享偏移校正后的288维support，\(\widehat{\mathbf b}_t\)只作用于identity160块，\(\mathbf0_{128}\)表示FFT96和RF32不被M2.1改变。

若canonical空间拟合得到的affine head为\((\mathbf W,\mathbf a)\)，则：

\[
\mathbf W\mathbf y+\mathbf a
=
\mathbf W\mathbf z+\left(\mathbf a-\mathbf W_{[:,1:160]}\widehat{\mathbf b}_t\right).
\]

式中，\(\mathbf W\)是类别数×288的分类系数矩阵，\(\mathbf a\)是canonical空间截距，\(\mathbf W_{[:,1:160]}\)是其identity160列，\(\mathbf z\)是raw query或raw support，右侧括号内是编译后的raw坐标截距。代码返回右侧等价head，因此query只做普通评分，不参与变换估计，也不需要额外query缓存。

### 4.5保留原Cauchy稳健中心

共享偏移校正之后，M2.1再次调用原类内Cauchy中心步骤：

\[
\widetilde{\mathbf y}_{c,k}
=
\mathbf y_{c,k}+\left(\mathbf m_c^{\mathrm{rob}}-\bar{\mathbf y}_c\right).
\]

式中，\(\bar{\mathbf y}_c\)是共享偏移校正后的类均值，\(\mathbf m_c^{\mathrm{rob}}\)是同一类的Cauchy稳健中心，\(\widetilde{\mathbf y}_{c,k}\)是交给D92协方差头的最终support。该步骤仍只平移类内中心，不删除样本、不改变标签；新类没有地面类别中心，但会使用同一共享偏移规范化和同一Cauchy规则。

## 5.代码入口

核心模块：`code/cvsrffi/stage2_td_htrc_target_transport.py`。

D92集成入口：`code/scripts/probe_d92_registration_balanced_covariance.py`中的`build_td_htrc_fit`。

通用注册执行器提供显式模式：

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
    ground_class_centers=ground_old_centers,
    module2_mode="td_htrc_m21",
    seed=seed,
    device="cpu",
)
```

`ground_old_centers`必须是与当前旧类注册顺序一致的\(6\times160\)Phase1聚合中心。没有这些中心时，代码会拒绝TD-HTRC模式，而不是退化成无锚点的伪域适应。默认`module2_mode="baseline"`保持原D92 E0路径。

## 6.计算量与状态

M2.1不训练网络，不执行梯度更新，不增加query MAC。注册时主要计算：

1.6个旧类support的160维残差投影和Cauchy权重；
2.6个旧类中心与地面中心的差向量；
3.固定3步160维IRLS和一个160×160协方差估计；
4.将160维共享偏移吸收到分类头截距。

代码审计字段`estimated_registration_macs`给出这部分注册上界，`td_htrc_query_extra_macs=0`表示后续query评分没有新增变换乘法。偏移协方差只在注册过程中作为临时诊断对象；由于共享偏移已编译进截距，持久化分类状态不额外保存160维偏移或160×160协方差。

## 7.协议与证据边界

- 允许：当前目标域support及标签、Phase1不可变聚合中心、Phase1扰动谱。
- 禁止：query特征参与偏移估计、query真值、query批次比例、source样本回放、按receiver或seed切换公式。
- M2.1输出的是可运行的support-only代码路径和审计量，不是性能结论。
- 当前测试使用合成配对数据和既有D81/D92回归测试；尚未进行同row目标域适应paired性能实验。

## 8.与M2.2的边界

M2.1保留为较低自由度的共享偏移入口；M2.2已经在独立文件
`code/cvsrffi/stage2_td_htrc_m22.py`和探针入口中实现。M2.2在M2.1偏移之上增加低秩/块尺度传输、目标自适应扰动谱、旧类地面先验后验和D92对角不确定性注入，详细定义见
`docs/TD_HTRC_M22_MODULE2_UPGRADE.md`。以下能力仍不属于M2.1或M2.2：

|未接入项|原因|
|---|---|
|物理nuisance/z-dom连续状态融合|当前Phase2冻结缓存没有经过协议批准的连续域状态映射；不能用receiver、scene或query统计量替代|
|自由的全维仿射或可训练适配器|6个旧类锚点不足以辨识288维自由矩阵，且会改变support-only注册边界|
