# D92-BranchInteraction-v1：固定分支交互核

日期：2026-09-29。Phase1完全冻结；只用当前row合法target support及既有五块特征，不读取source样本、样本级source特征、source统计或query反馈。本文和实现未读取任何query评分或历史方法排名，也未运行真实实验。

用户希望K=1/5/10/20同时兼顾旧类与新类。本文预设一个可检验候选，不承诺所有K都有增益。此前support-only探查对辅助分支的结果支持继续利用这些出口，但不能证明新的二阶交互或K1有效。开发已接触获准support证据，不能称为未见目标support的盲设计。

## 1. 机制选择

考虑过三条方向：

1. **background×auxiliary二阶交互**：让两组特征共同决定判别方向，保留所有原线性项，K1也能直接拟合，不需估计类内方差。这是本次唯一实现的候选。
2. 支持内类间/类内散度驱动的分支权重或收缩度量：可在K≥5估计噪声，但K1没有独立类内变化证据，需要预设退化规则。若K1回到原头，就不能提供K1的机制增量。本次不实现、不扫描权重。
3. 多类margin或交叉熵头：改变原平方损失，但会引入迭代优化、停止规则及小样本过拟合风险，不能直接利用已有分支互补结构。本次不实现。

所选方法是标准核岭框架下的自定义双线性交互核，不宣称新理论，也未调用scikit-learn。核岭一般定义见[scikit-learn官方KernelRidge文档](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)。以下核、中心化及控制臂是本次明确预设。

## 2. 固定五块特征与三臂

沿用[BranchRidge设计](D92_BRANCH_AUGMENT_DESIGN_20260929.md)的特征、norm floor和FFT96：

\[
u(v)=v/\max(\|v\|_2,10^{-12}),\quad
B=u([u(z),4u(FFT96)])\in\mathbb R^{256},\quad
A=[u(t),u(f),u(p)]/\sqrt3\in\mathbb R^{480}.
\]

对任意两条独立接收记录，令\(k_B(x,x')=B_x^TB_{x'}\)、\(k_A(x,x')=A_x^TA_{x'}\)。固定三臂，不扫描系数：

| 臂 | 核 | 用途 |
|---|---|---|
| `linear`（L） | \(k_B+k_A\) | 与现有736维、ridge=1线性头等价 |
| `energy_control`（E） | \(1.5(k_B+k_A)\) | 控制新增特征能量；等价于原线性特征配ridge=2/3 |
| `interaction`（I） | \(k_B+k_A+k_Bk_A\) | 唯一候选，保留线性项并引入跨组交互 |

交互核的显式特征为\(\phi_I(x)=[B_x,A_x,\mathrm{vec}(B_xA_x^T)]\)，共123616维；实现仅计算核，不展开122880维张量。乘积核为正半定，线性项之和也为正半定。交互不是重复特征或换ridge：它允许“某background方向与某auxiliary方向同时出现”形成类别证据。测试用分处两组的XOR验证这一表达能力区别；这只是合成机制验证，不是实际性能证据。

常规非零且不受norm floor限制时，\(\|B\|^2=\|A\|^2=1\)，三臂对角分别为2、3、3，E与I匹配能量。若零块或极小块使能量\(e_B,e_A\)不再为1，则

\[
k_I(x,x)-k_E(x,x)=e_Be_A-\tfrac12(e_B+e_A).
\]

因此不能宣称每个输入严格匹配。日志逐臂记录kernel diagonal的min/max/mean及I−E实际差，不因此切换算法或参数。对所有合法输入，核对角上界分别为2、3、3。

## 3. 拟合、中心化与归纳式预测

每row有C个实际注册类、每类K个物理support，共N=CK。类别和support分别按物理class ID、physical ID稳定排序，类别角色仅用于审计和support结果分组，不参与拟合或推理。标签\(Y_i=onehot(y_i)-1/C\)，均衡support的数学均值严格为0。

对每个固定核，最小化

\[
\frac12\sum_{i=1}^N\|f(x_i)+b-Y_i\|_2^2+
\frac12\|f\|_{\mathcal H}^2.
\]

ridge恒为1，对应物理损失之和；截距不正则化，不除以N后保留同一ridge。令\(H=I-11^T/N\)、训练Gram为G，\(G_c=HGH\)。使用float64 Cholesky求

\[
\alpha=(G_c+I)^{-1}Y,\qquad
s(q)=k_c(q,S)\alpha,
\]

其中

\[
k_c(q,S)=k(q,S)-\overline{k(q,S)}1^T
-\tfrac1N1^TG+\tfrac1{N^2}1^TG1\,1^T.
\]

所有训练均值仅由当前fit的support计算。式中的\(\overline{k(q,S)}\)是单条q相对固定support的核均值，是固定RKHS中心化映射的一部分；它不读取其他query、不更新状态。实现用第一条训练support作参考差分，再做均值中心化，以改善恒定特征的数值表现；数学上与上述HGH相同。query逐行调用同一算术路径，分块、重排不改变分数。

状态保存当前row归一化target support的B/A、alpha、核中心化向量和标量。这些是合法target support，不是source特征库。源样本及源样本级特征仍为0。类表、所有数值数组和audit均不可变；不保存query状态。

若全部support的B及A精确相同，则数学上f=0。实现直接输出零分数，以免BLAS舍入产生虚假的类别偏好；这是一项精确相等判断，不使用经验阈值。平局始终按物理class ID字典序处理。C=1合法。

\(G_c+I\)最小特征值至少为1；条件数上界\(1+\mathrm{tr}(G_c)\le1+3N\)。有限值检查失败或Cholesky失败均为技术错误，不静默修改ridge、核或输入。

## 4. K1及固定support-only诊断

正式fit对所有K都使用同一I核和一次全support求解，没有运行时选择、OOF权重或新旧类专属参数。K1可以拟合当前C条标签对应的交互判别边界，但不能由此估计类内变化，也不能声称增益已验证。

`probe_branch_interaction`只接收当前support。K1仅记录核对角数值诊断，不拟合、不造holdout：`oof=null`、`folds=[]`、`factorization_count=0`，原因`K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT`。同一物理记录的数学view不能冒充K1独立验证。

K≥2沿用既有按类physical ID排序位置取模的完整\(\min(K,3)\)折协议。每折、每臂从train physical重新构建Gram、中心化和alpha；held physical不进入任何拟合状态。K=5/10/20每episode固定9次解析分解。不得用全support中心化或头复用到fold。保持三臂相同physical分配，并输出逐physical配对结果。

support诊断同时报告旧类accuracy、新类accuracy、H、macro accuracy、固定softmax未校准NLL，以及I−L、E−L、I−E的逐physical正确性差。预登记门槛由本次运行记录承载：K=5/10/20含新类任务上I相对两对照的H与new均须为正，旧类允许退化上限1个百分点；K1只做数值检查。这些是候选开发的support证据，不是正式query验收或K1收益证明。复用物理记录的多个draw不能被当成独立样本构造置信区间。

若I仅胜L而不胜E，不能把变化归因于交互。若旧类与新类收益方向相反，也不能只用总accuracy掩盖。任一正式query预测须先固化artifact，再由独立scorer连接truth；评分不反馈调参、排序、停止或选择性重跑。

## 5. 成本、风险和实现

设D=736。核训练约\(O(N^2D+N^3+N^2C)\)，每query约\(O(ND+NC)\)。相对原线性头，交互需要当前target support记忆及与全部support比较。当前数值状态的确切口径是六个float64数组和两个float64标量：

\[
8\bigl[N(736+C+2)+C+2\bigr]\ \mathrm{bytes}.
\]

C=26、K=20时为3178464 B；原线性W/b为153296 B。新数值口径包含support B/A，二者不可混淆。类名、audit、Python对象、模型、特征缓存和文件开销另记。每query乘加量由原来的约736C增加到约N(736+C)，内存及推理成本是实质代价。任何实际耗时都必须运行后记录，本文未给出真实数据速度。

可能失败的原因包括：两组共有接收机/信道变化相乘后被放大；冻结分支没有可泛化的交互信息；乘积项更依赖小样本support偶然方向；既有FFT权重使交互仍偏向FFT；K1缺少物理重复而无法验证类内稳定性；新增核方向改变有效正则化，能量相同也不保证谱相同。E只控制总能量，不能证明所有收益都来自理想的身份因果结构。

API为：

```python
fit_branch_interaction(*, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes=(), arm='interaction')
probe_branch_interaction(*, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes=())
state.score(*, z_id, fft, t_emb, f_emb, pa_local)
state.predict(**features)
state.audit_dict()
```

probe没有query接口、文件I/O或可部署state；三臂不会触发选择。配置位于[冻结配置](../configs/d92_branch_interaction_frozen_20260929.json)，顶层为`algorithm`。核心位于[方法代码](../code/cvsrffi/d92_branch_interaction.py)，测试位于[核心测试](../tests/test_d92_branch_interaction.py)。日志记录实际损失、RKHS惩罚、dual coefficient梯度、正常方程残差、截距残差、K/N/IDs、Gram/状态字节及耗时；optimizer steps=0，LR/epoch/source validation为null并说明原因。

合成验证覆盖显式张量展开等价、原BranchRidge等价、ridge=2/3能量对照、XOR机制、训练与未见行的中心化、K1/C1、全零及近零恒定特征、旧/新角色不影响拟合、类/physical排列、query分块与重排逐值一致、状态只读、实际字节、K1不拟合、完整OOF与独立fold重拟合一致、非法输入及JSON有限值。近零恒定输入曾暴露约10^-21的伪类别分差，已用上述精确退化规则修正；没有借此改动一般算法或实验参数。
