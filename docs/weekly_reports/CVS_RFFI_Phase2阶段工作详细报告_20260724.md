# CVS-RFFI Phase2阶段工作详细报告（截至2026年7月24日）

## 从跨接收机域适应到少样本新类注册：任务、方法与实验结果

**汇报对象：**导师

**报告范围：**整合2026年7月16日与7月24日两轮工作

**研究主线：**Stage2-A无标签目标域参考→Stage2-B旧类K-shot适应→Stage2-C旧类保持与新类注册

**协议口径：**`p2_min_v1`

**涉及方法：**ProtoNet CDA、MRIOR-SDA、DADDA-SDA、CSIL、MoPC-HR

> **阶段概览：**Phase2需要依次解决跨接收机迁移、旧类少样本适应和新类注册。MRIOR-SDA与DADDA-SDA证明target-old support具有域校准价值，但依赖source replay和完整backbone更新；CSIL与MoPC-HR分别偏向稳定性和可塑性，仍未同时解决旧类保持与新类学习。当前工作完成了任务定义、方法对照和失败归因，尚不能声明主方法已经达到晋级性能。

## 1.汇报摘要与工作主线

### 1.1报告讲述逻辑

本报告按“任务是什么→实验如何组织→方法做了什么→结果说明什么”的顺序展开：

1. 先用Stage2-A/B/C说明Phase2的递进任务、数据权限和评价口径。
2. 再分别介绍Stage2-B的三种域适应方法和Stage2-C的两种类增量方法，包括数据、更新对象和损失函数。
3. 最后汇总正式LEO结果、matched诊断、方法边界和下一步路线。

### 1.2近期工作的递进关系

|时间|Phase2阶段|核心问题|对比方法|主要输出|
|---|---|---|---|---|
|统一参考|Stage2-A|没有target标签时，Phase1模型能否直接跨接收机|直接ADV3B02|5个target receiver的\(A_{\mathrm{old}}^{\mathrm{pre}}\)参考|
|截至7月16日|Stage2-B|换到新接收机后，旧类准确率如何恢复|ProtoNet CDA、MRIOR-SDA、DADDA-SDA|375个域适应任务，分析K-shot、receiver和计算开销|
|截至7月24日|Stage2-C|如何注册新发射机，同时保留旧发射机|CSIL、MoPC-HR|800个正式LEO cell/2400个场景row及同规模matched无LEO诊断|

## 2.Phase2任务的递进关系

### 2.1Stage2-A/B/C

Phase1在地面source receiver上训练并封存ADV3B02 deployment bundle。Phase2将该模型部署到互斥的target receiver，三个Stage按target标签权限和类别范围逐级增加：

|阶段|target标签权限|类别集合|核心任务|主要输出|
|---|---|---|---|---|
|Stage2-A|无target TX标签|旧类参考|建立无标签目标域参考|reference/diagnostic|
|Stage2-B|旧类K-shot标签|\(Y_{\mathrm{old}}\)|完成旧类域适应|旧类预测器|
|Stage2-C|旧类与新类K-shot标签|\(Y_{\mathrm{old}}\cup Y_{\mathrm{new}}\)|保持旧类并注册新类|全注册类统一预测器|

Stage2-A只回答“没有target标签时能否直接迁移”；Stage2-B用旧类support校准域偏移；Stage2-C进一步加入新类support，并要求旧类与新类在同一分类空间竞争。Phase3的unknown拒识不属于本报告范围。

### 2.2集合、样本角色与成功条件

$$
Y_{\mathrm{old}}\cap Y_{\mathrm{new}}=\varnothing
$$

- **旧类\(Y_{\mathrm{old}}\)：**Phase1已见发射机；更换receiver不会改变类别身份。
- **新类\(Y_{\mathrm{new}}\)：**Phase1未见、在Stage2-C通过合法support注册的发射机。
- **support：**带标签目标域样本，可用于状态更新。
- **query：**纯测试样本，不能训练、调参、选择候选、设阈值或回滚。
- **K-shot：**每个已注册类别有K个互不重复的物理support样本；同一接收IQ的FFT、均衡或裁剪view不增加K。

Stage2-C必须在同一row同时报告旧类保持和新类注册。只提高\(A_{\mathrm{old}}^{\mathrm{post}}\)或只提高\(A_{\mathrm{new}}\)都不构成完整成功。

## 3.数据、权限与评价框架

### 3.1从Phase1训练到Phase2部署

WiSig/ManySig在本项目中是地面代理数据，不是真实卫星数据。两轮实验统一使用ADV3B02地面域泛化checkpoint，避免backbone差异干扰方法比较。

|环节|可用数据|允许更新|输出|
|---|---|---|---|
|Phase1地面训练|source数据及物理启发的LEO增强|身份backbone与训练状态|不可变deployment bundle|
|Phase2适应/注册|bundle、固定LEO target IQ、当前row的support标签|由方法声明的prototype、分类头、adapter或backbone|冻结后的目标域预测器|
|独立评分|prediction artifact与query真值|不得回流到predictor|同一row评价指标|

`p2_min_v1`主方法不能在Phase2运行时读取clean/raw/source样本、source feature/cache/replay或其他可影响决策的外部source状态。外部论文对比若使用更宽权限，必须单独标注，不能反向证明主方法满足协议。

### 3.2固定LEO观测与query隔离

每个clean/raw物理IQ在进入Phase2前只能随机选择一种允许的LEO弱信道：

$$
c_i\in\{\mathrm{leo\_clear\_weak},\mathrm{leo\_low\_elev\_weak},\mathrm{leo\_rain\_weak}\}
$$

同一物理样本不能复制后叠加多个LEO场景再扩充K。support可以更新状态；query只在模型冻结后逐样本测试，不能参与训练、调参、回滚或跨query重排。

### 3.3实验矩阵

|工作包|基座与数据|target设置|实验规模|输出|
|---|---|---|---|---|
|旧类域适应|ADV3B02＋target-old support/query；MRIOR/DADDA另读source|5个target receiver；\(K\in\{1,2,5,10,20\}\)；5个seed|5×5×5×3=375个方法任务|\(A_{\mathrm{old}}^{\mathrm{pre/post}}\)、\(G_{\mathrm{old}}\)、正/负迁移、时延|
|新类注册|ADV3B02接口＋Phase2前base状态＋target-new support；old/new query只评分|5个target receiver；5个seed；\(K\in\{1,5,10,20\}\)；新类数\(\in\{2,5,10,20\}\)|800个正式LEO cell/2400个场景row|\(A_{\mathrm{old}}^{\mathrm{pre/post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)、\(A_{\min,\mathrm{old}}\)|
|信道归因诊断|保持方法、物理ID、split、K、seed和旧类条件一致，仅替换新类IQ为无LEO版本|与正式矩阵逐row配对|800个非正式cell/2400个场景row|\(\Delta A_{\mathrm{new}}\)、\(\Delta A_{\mathrm{old}}\)、\(\Delta H\)、\(\Delta F_{\mathrm{old}}\)、\(\Delta A_{\min,\mathrm{old}}\)|

本报告沿用已完成外部对比运行中的“正式LEO”命名，表示该冻结矩阵内的LEO条件，而不表示这些宽权限方法已经获得`p2_min_v1`主方法晋级资格。三个LEO场景分别训练、锁定和评分，不能把同一物理样本的多场景结果合并成更多K-shot support；其汇总仅用于对比方法机制分析。

### 3.4评价指标

记\(Q_{\mathrm{old}}\)和\(Q_{\mathrm{new}}\)分别为旧类、新类query集合，\(Q_c\)为旧类\(c\)的query集合；\(y_i\)是真值，\(\hat y_i^{(0)}\)和\(\hat y_i^{(1)}\)分别表示适应/注册前后的预测，\(\mathbb I[\cdot]\)为指示函数。为同时保证数学表达和实验字段可追溯，本文采用\(A_{\mathrm{old}}^{\mathrm{pre}}\)、\(A_{\mathrm{old}}^{\mathrm{post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)和\(A_{\min,\mathrm{old}}\)，分别对应结果字段`old_acc_before`、`old_acc_after`、`seen_new_acc`、`H_old_new`、`forgetting`和`min_old`。

#### 旧类适应

**适应前旧类准确率：**

$$
A_{\mathrm{old}}^{\mathrm{pre}}
=\frac{1}{|Q_{\mathrm{old}}|}
\sum_{i\in Q_{\mathrm{old}}}
\mathbb I[\hat y_i^{(0)}=y_i]
$$

**适应后旧类准确率：**

$$
A_{\mathrm{old}}^{\mathrm{post}}
=\frac{1}{|Q_{\mathrm{old}}|}
\sum_{i\in Q_{\mathrm{old}}}
\mathbb I[\hat y_i^{(1)}=y_i]
$$

**适应收益：**

$$
G_{\mathrm{old}}
=A_{\mathrm{old}}^{\mathrm{post}}-A_{\mathrm{old}}^{\mathrm{pre}}
$$

\(G_{\mathrm{old}}\)为正表示正迁移，为负表示适应损伤了原有能力。

#### 新类注册

**已注册新类准确率：**

$$
A_{\mathrm{new}}
=\frac{1}{|Q_{\mathrm{new}}|}
\sum_{i\in Q_{\mathrm{new}}}
\mathbb I[\hat y_i^{(1)}=y_i]
$$

该指标为0意味着新类support没有形成可用的分类身份，或者增量训练根本没有产生有效更新。

#### 旧新联合评价

**旧新调和均值：**

$$
H_{\text{old,new}}
=\frac{
2A_{\mathrm{old}}^{\mathrm{post}}A_{\mathrm{new}}
}{
A_{\mathrm{old}}^{\mathrm{post}}+A_{\mathrm{new}}
}
$$

调和均值会惩罚“只保旧类、不学新类”和“只学新类、忘掉旧类”。任意一侧接近0，\(H_{\text{old,new}}\)都会接近0。

**遗忘：**

$$
F_{\mathrm{old}}
=A_{\mathrm{old}}^{\mathrm{pre}}-A_{\mathrm{old}}^{\mathrm{post}}
$$

遗忘越大，说明增量学习对旧知识破坏越严重。

**最低旧类准确率：**

$$
A_c
=\frac{1}{|Q_c|}
\sum_{i\in Q_c}
\mathbb I[\hat y_i^{(1)}=y_i]
$$

$$
A_{\min,\mathrm{old}}
=\min_{c\in Y_{\mathrm{old}}}A_c
$$

\(A_{\min,\mathrm{old}}\)直接检查最差旧发射机是否接近失效，避免平均\(A_{\mathrm{old}}^{\mathrm{post}}\)掩盖局部崩塌。

### 3.5统一输入、状态更新与输出

|阶段|输入|允许更新的状态|输出|
|---|---|---|---|
|直接基线|Phase1 bundle＋target query|无|旧类预测|
|Stage2-B适应|Phase1 bundle＋target-old support|prototype、adapter或backbone，取决于方法权限|冻结后的旧类预测器|
|Stage2-C注册|冻结/适应后状态＋\(Y_{\mathrm{old}}/Y_{\mathrm{new}}\) support|旧类统计、新类prototype、分类头或受控参数|面向全部注册类的统一预测器|
|独立评分|不可变prediction artifact＋query真值|不得回流到predictor|\(A_{\mathrm{old}}^{\mathrm{post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)等指标|

## 4.Stage2-B：跨接收机旧类域适应

### 4.1研究问题与统一实验设置

Stage2-B要回答的问题是：Phase1已经识别过的发射机换到未见target receiver后，少量target-old support能否恢复旧类识别能力？

统一实验设置如下：

|维度|设置|
|---|---|
|基座模型|同一ADV3B02地面域泛化checkpoint|
|target receiver|20-1、3-19、7-14、7-7、8-8|
|K-shot|1、2、5、10、20|
|随机性|5个独立seed|
|target观测|support和query均为固定LEO弱信道接收IQ|
|query权限|只测试，不训练、不调参、不回滚|
|方法|ProtoNet CDA、MRIOR-SDA、DADDA-SDA|

### 4.2ProtoNet CDA：度量型少样本分类

Snell等人提出的Prototypical Networks属于度量型少样本方法[1]。**其特点是固定特征空间，只用support闭式计算类别中心；不访问source、不反向传播。**

**CVS数据与更新：**加载ADV3B02的160维身份特征\(z=f_\theta(x)\)，每个场景只读取6个旧类的K-shot target-old support。对类别\(c\)的support集合\(S_c\)，prototype是类内平方距离最小化问题的闭式解：

$$
p_c^*
=\arg\min_p\sum_{i\in S_c}\|z_i-p\|_2^2
=\frac{1}{|S_c|}\sum_{i\in S_c}z_i
$$

本轮没有可训练损失：

$$
\mathcal L_{\text{ProtoNet-CDA}}=0,
\qquad
\mathrm{gradient\_updates}=0
$$

模型锁定后，query按欧氏距离最近的prototype分类：

$$
\hat y(x)
=\arg\min_{c\in Y_{\mathrm{old}}}
\|f_\theta(x)-p_c^*\|_2^2
$$

**核心局限：**它只能移动类别参考点，不能修正接收机造成的特征旋转、拉伸或类内多峰，因此“使用了support”不等于“完成了域适应”。

### 4.3MRIOR-SDA：域对齐与target监督

Yang等人的MRIOR原本属于单源无监督域适应[2]。CVS版本用真实target-old support标签替代伪标签，形成监督式域适应。**其特点是用可学习的DV-KL估计网络对齐source与target，同时更新完整ADV3B02身份backbone。**

记\(B_s,B_t\)为source和target-support batch，\(p_\theta(c|x)\)为旧类预测概率，\(w_c\)为根据target-support类别频数得到并归一化到均值1的类别权重：

$$
w_c
=\frac{\left(n_c+\epsilon\right)^{-1}}
{\frac{1}{C}\sum_{r=1}^{C}\left(n_r+\epsilon\right)^{-1}}
$$

source和target的加权交叉熵为：

$$
\mathcal L_{\mathrm{wCE}}^{D}
=-\frac{1}{|B_D|}
\sum_{(x_i,y_i)\in B_D}
w_{y_i}\log p_\theta(y_i|x_i),
\qquad D\in\{s,t\}
$$

令\(T_\phi(z)\)为DV估计网络，域差异项为：

$$
\mathcal L_{\mathrm{DV-KL}}
=\frac{1}{|B_s|}\sum_{i\in B_s}T_\phi(z_i^s)
-\log\left[
\frac{1}{|B_t|}\sum_{j\in B_t}
\exp T_\phi(z_j^t)
\right]
$$

ADV3B02的外层优化目标为：

$$
\mathcal L_{\text{MRIOR-SDA}}
=0.5\mathcal L_{\text{wCE}}^{\text{source}}
+0.5\mathcal L_{\text{wCE}}^{\text{target-support}}
+0.005\mathcal L_{\text{DV-KL}}
$$

**CVS执行：**每个场景使用封存source LEO弱信道标签缓存和6类K-shot target-old LEO support。先用7个内层step最大化DV估计，再更新backbone；两个Adam学习率均为\(6\times10^{-4}\)，每场景200个外层step。query只在训练结束后测试。该方法的代价是source replay和完整backbone更新，因此只作为宽权限机制对照。

### 4.4DADDA-SDA：全局与类条件动态分布对齐

Feng等人的DADDA属于基于统计距离的跨接收机域适应[3]。**其特点是同时计算全局MMD和类条件LMMD，再用数据驱动的\(\alpha\)动态分配二者权重。**

source与target-support的监督分类项均采用标准交叉熵：

$$
\mathcal L_{\mathrm{CE}}^{D}
=-\frac{1}{|B_D|}
\sum_{(x_i,y_i)\in B_D}
\log p_\theta(y_i|x_i),
\qquad D\in\{s,t\}
$$

采用RBF核：

$$
k(u,v)
=\exp\left(
-\frac{\|u-v\|_2^2}{2\sigma^2}
\right)
$$

全局MMD为：

$$
\mathcal L_{\mathrm{MMD}}
=\frac{1}{n_s^2}\sum_{i,i'}k(z_i^s,z_{i'}^s)
+\frac{1}{n_t^2}\sum_{j,j'}k(z_j^t,z_{j'}^t)
-\frac{2}{n_sn_t}\sum_{i,j}k(z_i^s,z_j^t)
$$

对类别\(c\)，令\(w_s^c,w_t^c\)为按该类样本数归一化的source/target-support权重，类条件LMMD为：

$$
\mathcal L_{\mathrm{LMMD}}
=\sum_{c=1}^{C}
\left[
(w_s^c)^\top K_{ss}w_s^c
+(w_t^c)^\top K_{tt}w_t^c
-2(w_s^c)^\top K_{st}w_t^c
\right]
$$

动态系数和总损失分别为：

$$
\alpha
=\frac{\mathcal L_{\mathrm{MMD}}}
{\mathcal L_{\mathrm{MMD}}+\mathcal L_{\mathrm{LMMD}}+\epsilon}
$$

$$
\mathcal L_{\text{DADDA-SDA}}
=\mathcal L_{\text{CE}}^{\text{source}}
+\mathcal L_{\text{CE}}^{\text{target-support}}
+(1-\alpha)\mathcal L_{\text{MMD}}
+\alpha\mathcal L_{\text{LMMD-sum}}
$$

**CVS执行：**数据与MRIOR-SDA相同；更新完整ADV3B02身份backbone和旧类分类输出。SGD初始学习率为\(10^{-4}\)、momentum为0.9、weight decay为\(5\times10^{-4}\)，每场景200步。query不参与CE、MMD、LMMD或\(\alpha\)估计。DADDA比MRIOR更直接地约束类条件分布，但K很小时\(w_t^c\)估计噪声较大，可能产生负迁移。

### 4.5机制、权限与结果位置

|方法|方法分类|部署输入|更新对象|source访问|计算特点|本轮能否注册新类|
|---|---|---|---|---|---|---|
|ProtoNet CDA|度量型少样本分类|target-old support|prototype|否|0次backbone更新|未评估|
|MRIOR-SDA|模型更新型监督域适应|source LEO缓存＋target-old support|完整身份backbone＋DV-KL估计网络|是|200步/场景，3场景合计600步|否，闭集旧类|
|DADDA-SDA|统计距离型监督域适应|source LEO缓存＋target-old support|完整身份backbone|是|200步/场景，3场景合计600步|否，闭集旧类|

ProtoNet CDA、MRIOR-SDA与DADDA-SDA的复现实验数值统一放在附录A，正文只保留理解方法和权限差异所需的信息。

## 5.Stage2-C：类增量学习与新类注册

### 5.1研究问题、代码口径与数据

Stage2-C在同一target receiver中注册新类，并要求旧类与新类统一竞争。本轮使用论文作者官方执行语义：CSIL锁定`pcwhy/CSIL@8ce8637`，MoPC-HR锁定`xmuLdz/MoPC-HR@ae65543`；编码器接口替换为ADV3B02的160维\(z_{\mathrm{id}}\)，其余训练器、损失和更新范围按官方实现保留。

|环节|CSIL|MoPC-HR|
|---|---|---|
|Phase2前基座|5879条source base train＋2521条Fisher validation|8400条source base训练并保存6个旧类prototype|
|Phase2真实训练数据|当前cell的target-new K-shot support|当前cell的target-new K-shot support|
|query作用|旧类测遗忘、新类测注册；均不参与训练|旧类测遗忘、新类测注册；均不参与训练|
|冻结矩阵|5个receiver×5个seed×4个K×4个新类规模，共400个cell/1200个LEO row|同左|

这里的source数据用于构造进入Phase2前的旧类状态，不是每个增量cell中的source replay。实验属于“官方方法语义＋CVS基座/数据接口适配”，不是原论文数据集数值复现。

#### 5.1.1少样本学习的严格定义

少样本学习（few-shot learning，FSL）不是泛指“总数据量较小”，而是指：**模型已经从base数据、相关任务或预训练模型中获得先验知识，在一个新的目标任务中，每个待识别类别只有极少量带标签样本时，仍要利用这些样本建立能够泛化到独立query的预测器。**少样本描述的是目标任务的监督数据条件，不限定必须采用某一种网络或优化算法。

令一个分类任务为\(\mathcal T=(\mathcal Y_{\mathcal T},S_{\mathcal T},Q_{\mathcal T})\)。其中，\(\mathcal Y_{\mathcal T}\)是本任务的类别集合，\(S_{\mathcal T}\)是允许模型读取并用于适配的support集合，\(Q_{\mathcal T}\)是只用于评价的query集合，并满足样本级互斥：

$$
S_{\mathcal T}\cap Q_{\mathcal T}=\varnothing .
$$

标准\(N\)-way \(K\)-shot分类表示任务中有\(N\)个类别，每个类别提供\(K\)个带标签support样本：

$$
\left|\mathcal Y_{\mathcal T}\right|=N,\qquad
S_{\mathcal T}
=\bigcup_{c\in\mathcal Y_{\mathcal T}}
\left\{(x_{c,k},c)\right\}_{k=1}^{K},
\qquad
\left|S_{\mathcal T}\right|=NK.
$$

学习算法\(\mathcal A\)接收先验状态\(\theta_0\)和support，生成针对当前任务的状态或预测器：

$$
\theta_{\mathcal T}
=\mathcal A(\theta_0,S_{\mathcal T}),\qquad
\hat y=f_{\theta_{\mathcal T}}(x),\quad x\in Q_{\mathcal T}.
$$

少样本学习的目标不是记住support，而是降低独立query上的期望风险。若任务从分布\(p(\mathcal T)\)中抽取，则目标可写为：

$$
\min_{\mathcal A}\;
\mathbb E_{\mathcal T\sim p(\mathcal T)}
\left[
\frac{1}{|Q_{\mathcal T}|}
\sum_{(x,y)\in Q_{\mathcal T}}
\ell\!\left(f_{\mathcal A(\theta_0,S_{\mathcal T})}(x),y\right)
\right].
$$

因此，少样本学习至少包含四个必要要素：

1. **先验知识：**来自base类训练、相关任务、预训练表示或已有模型；完全从随机初始化学习几个样本通常只是极小数据训练。
2. **少量带标签support：**\(K\)统计相互独立的真实样本，而不是数据增强产生的view数量。
3. **support/query隔离：**support可以更新模型、prototype或分类头；query不能参与调参、早停、阈值选择或状态更新。
4. **对独立query泛化：**评价对象是未参与适配的query，而不是support训练准确率。

经典闭集少样本分类通常满足\(\mathcal Y_{\mathrm{support}}=\mathcal Y_{\mathrm{query}}\)，query只在当前\(N\)个类别中竞争。ProtoNet[1]属于度量型少样本学习：它不必在support上反向传播，而是用support计算类别prototype，再按距离预测query。其他少样本方法也可以通过微调参数、生成分类权重或学习优化器完成适配；“少样本”描述数据与任务条件，不等同于“只使用prototype”。

#### 5.1.2少样本学习与类增量学习的区别

类增量学习（class-incremental learning，CIL）描述的是**类别随时间分批到达、模型状态持续更新、推理标签空间不断扩大的学习过程**。设base阶段类别为\(\mathcal Y^{(0)}\)，第\(t\)次增量到达的新类别为\(\mathcal Y^{(t)}\)，不同阶段的新增类别互不重叠。完成第\(t\)阶段后，模型必须在全部已学习类别的并集上统一预测：

$$
\mathcal Y^{(\le t)}
=\bigcup_{j=0}^{t}\mathcal Y^{(j)},\qquad
\hat y
=\arg\max_{c\in\mathcal Y^{(\le t)}}p_{\theta_t}(c|x).
$$

类增量学习的关键不在于当前批次样本是否少，而在于更新\(\theta_{t-1}\rightarrow\theta_t\)后，模型既要学习\(\mathcal Y^{(t)}\)，又要保持\(\mathcal Y^{(<t)}\)的识别能力，并且推理时通常不提供样本来自哪个阶段的task ID。其核心矛盾是新类可塑性与旧类稳定性之间的平衡。

|比较维度|少样本学习|类增量学习|少样本类增量学习|
|---|---|---|---|
|主要约束|目标任务中每类标签样本很少|类别分阶段到达，模型要持续更新|新类别分阶段到达，且每个新类只有少量样本|
|标签空间|一个episode内通常固定为\(N\)类|随阶段扩展为\(\mathcal Y^{(\le t)}\)|随阶段扩展，但每阶段新类为K-shot|
|历史状态|可以对每个任务重新构造临时预测器|必须从\(\theta_{t-1}\)继续更新到\(\theta_t\)|必须持续更新，同时避免少样本过拟合|
|旧类保持|普通闭集FSL通常不要求保留此前episode性能|必须评价旧类遗忘|必须同时解决旧类遗忘与新类欠学习|
|推理范围|通常在当前任务的\(N\)类内预测|在全部已学习类别中统一竞争|在全部base类和历次新类中统一竞争|
|主要风险|support过拟合、类别中心估计不准|灾难性遗忘、分类器偏向新类|灾难性遗忘、新类欠拟合及严重类别不平衡|
|本报告对应|Stage2-B中的K-shot旧类域适应、ProtoNet CDA|CSIL和MoPC-HR的增量更新机制|Stage2-C在少量新类support下注册新类并保留旧类|

两者不是互斥的方法类别，而是描述两个不同维度：**少样本学习规定“当前任务能看到多少标注”，类增量学习规定“类别和模型状态如何随时间演化”。**当增量阶段每个新类只有\(K\)个support时，任务同时属于少样本学习和类增量学习，即少样本类增量学习（few-shot class-incremental learning，FSCIL）。

在本项目中，Stage2-B保持标签空间为\(\mathcal Y_{\mathrm{old}}\)，主要解决新receiver下的K-shot域适应，因此属于跨域少样本适应；Stage2-C把标签空间从\(\mathcal Y_{\mathrm{old}}\)扩展为\(\mathcal Y_{\mathrm{old}}\cup\mathcal Y_{\mathrm{new}}\)，并要求两部分在同一分类器中竞争，因此属于少样本类增量新类注册。仅在新类support上取得较高训练准确率，不代表类增量成功；必须同时报告独立query上的\(A_{\mathrm{new}}\)、\(A_{\mathrm{old}}^{\mathrm{post}}\)、\(H_{\mathrm{old,new}}\)和\(F_{\mathrm{old}}\)。

### 5.2CSIL：通道隔离型无exemplar类增量学习

Liu等人提出的CSIL属于无exemplar、结构扩展型类增量方法[4]。**其特点是冻结ADV3B02 backbone，以通道扩展和mask限制新类更新，再用EWC与KD保护旧类响应。**

**CVS数据与更新：**进入Phase2前，5879条source样本训练`fc_bf_fp→zero-bias Fingerprints`基座，2521条互斥source样本估计Fisher。Phase2只用target-new support训练；ADV3B02冻结，fc只更新新增行和bias，fingerprint只允许old-old与new-new块更新，两个交叉块保持为0。

记\(B_{\mathrm{new}}\)为新类support训练batch，\(q_\theta(c|x)\)为全部注册类概率。新类交叉熵为：

$$
\mathcal L_{\mathrm{CE,new}}
=-\frac{1}{|B_{\mathrm{new}}|}
\sum_{(x_i,y_i)\in B_{\mathrm{new}}}
\log q_\theta(y_i|x_i)
$$

EWC使用Phase2前参数\(\theta^*\)和Fisher重要性\(F_j\)：

$$
\mathcal L_{\mathrm{EWC}}
=\frac{1}{2}\sum_j
F_j(\theta_j-\theta_j^*)^2
$$

令\(r_{\mathrm{old}}^*(x)\)和\(r_{\mathrm{old}}(x)\)为旧模型与当前模型的旧类fingerprint响应，官方KD为：

$$
\mathcal L_{\mathrm{KD}}
=\frac{1}{32}
\sum_{x\in B_{\mathrm{new}}}
\|r_{\mathrm{old}}^*(x)-r_{\mathrm{old}}(x)\|_2^2
$$

实际执行总损失为：

$$
\mathcal L_{\text{CSIL}}
=\mathcal L_{\text{CE,new}}
+\mathcal L_{\text{EWC}}
+0.2\mathcal L_{\text{KD}}
$$

**执行特点：**3个epoch、batch size 20、`drop_last=True`，手工SGDM学习率为\(0.01/(1+0.01t)\)。query用zero-bias全注册类argmax，训练步数为0。该方法旧类约束最强，但低K时可能没有完整batch，或者新类更新被旧类约束压制。

### 5.3MoPC-HR：prototype校正与分层正则化

Li等人提出的MoPC-HR属于无exemplar、prototype校正型类增量方法[5]。**其特点是允许完整ADV3B02随新类support更新，同时用旧prototype伪特征和分层参数正则限制遗忘。**

**CVS数据与更新：**进入Phase2前用全部8400条source旧类样本训练基座并保存6个old prototype。Phase2真实样本只来自target-new support；每个step另采样16个old prototype并加入\(\epsilon\sim\mathcal N(0,0.05^2I)\)。ADV3B02 backbone和Linear classifier均更新。

新类真实样本交叉熵为：

$$
\mathcal L_{\mathrm{CE,new}}
=-\frac{1}{|B_{\mathrm{new}}|}
\sum_{(x_i,y_i)\in B_{\mathrm{new}}}
\log q_\theta(y_i|x_i)
$$

令\(\tilde z_r=p_{c_r}+\epsilon_r\)为增强后的旧类prototype，温度\(\tau=2\)，prototype augmentation损失为：

$$
\mathcal L_{\mathrm{protoAug}}
=-\frac{1}{B_p}
\sum_{r=1}^{B_p}
\log\operatorname{softmax}
\left(
\frac{g_\theta(\tilde z_r)}{\tau}
\right)_{c_r}
$$

公开trainer对第\(\ell\)组参数使用递减系数\(a_\ell=1-(\ell-1)/L\)，分层正则为：

$$
\mathcal L_{\mathrm{HR}}
=\sum_{\ell=1}^{L}
a_\ell\|\theta_\ell-\theta_\ell^*\|_2
$$

实际执行总损失为：

$$
\mathcal L_{\text{MoPC-HR}}
=\mathcal L_{\text{CE,new}}
+\mathcal L_{\text{protoAug}}
+\mathcal L_{\text{HR}}
$$

公开代码还计算KD用于日志，但`KD不进入总loss`。训练后prototype校正为：

$$
\tilde P_{\mathrm{old}}
=0.97P_{\mathrm{old}}
+0.03\operatorname{softmax}
\left(
P_{\mathrm{old}}(P_{\mathrm{new}}^*)^\top
\right)
\left(
P_{\mathrm{new}}-P_{\mathrm{new}}^*
\right)
$$

**执行特点：**20个epoch、batch size 16、SGD学习率0.01。正式CVS query仍使用当前模型的全注册类classifier logits，而不是校正prototype。该方法新类可塑性强于CSIL，但完整backbone更新会放大旧类遗忘。

### 5.4CSIL与MoPC-HR机制对比

|比较维度|CSIL|MoPC-HR|
|---|---|---|
|主要方法类别|结构扩展与通道隔离|prototype校正与分层正则|
|历史原始样本|不保存|不保存|
|Phase2真实训练样本|target-new support|target-new support＋旧prototype伪特征|
|保留的历史状态|旧模型、fingerprint结构、Fisher重要参数统计|旧类prototype、旧模型参数|
|ADV3B02增量更新|冻结|不冻结|
|实际损失|CE-new＋EWC＋0.2KD|CE-new＋protoAug＋逐参数非平方L2 HR；KD仅记录|
|旧类保护|mask、冻结、KD、EWC|prototype增强、HR、旧参数参考|
|最终query规则|zero-bias全注册类argmax|当前模型全注册类classifier logits argmax|
|主要风险|过度保护旧类，新类不注册|新类学得快，但旧类遗忘|
|部署特征|结构复杂、容量增长|状态较紧凑，但仍需多轮训练|

### 5.5正式LEO弱信道结果

正式LEO矩阵的800/800个cell、2400/2400个场景row和800/800份prediction/评分收据均完成，独立审计`failures=[]`。下表每行聚合1200个同条件、同方法的场景row：

|方法|\(A_{\mathrm{old}}^{\mathrm{pre}}\)|\(A_{\mathrm{old}}^{\mathrm{post}}\)|\(A_{\mathrm{new}}\)|\(H_{\mathrm{old,new}}\)|\(F_{\mathrm{old}}\)|\(A_{\min,\mathrm{old}}\)|
|---|---:|---:|---:|---:|---:|---:|
|CSIL|42.83%|23.17%|8.65%|1.18%|19.66%|0.82%|
|MoPC-HR|45.32%|22.14%|26.61%|10.85%|23.19%|3.89%|

这些是跨5个target receiver、5个seed、4个\(K\)、4个新类规模和3个LEO场景的全矩阵均值，不是挑选最佳\(K\)或最佳receiver后的结果。\(A_{\min,\mathrm{old}}\)表示每个row最低旧类准确率再求均值，用于观察少数旧类是否被严重牺牲。

#### 结果解读

**两种方法都没有在全矩阵上解决旧新平衡。**CSIL的正式LEO\(A_{\mathrm{new}}\)只有8.65%，\(H_{\mathrm{old,new}}\)仅1.18%；MoPC-HR的\(A_{\mathrm{new}}\)较高，为26.61%，但\(A_{\mathrm{old}}^{\mathrm{post}}\)仍只有22.14%，\(F_{\mathrm{old}}\)达到23.19个百分点。

**MoPC-HR比CSIL更具可塑性，但不是无代价提升。**其\(A_{\mathrm{new}}\)比CSIL高17.96个百分点，\(H_{\mathrm{old,new}}\)高9.67个百分点；与此同时，完整backbone更新使旧类表征更容易漂移。全矩阵结果不能写成“MoPC-HR已经解决新类注册”，只能说明它在当前执行语义下更倾向于学习新类。

**CSIL偏向稳定约束，但新类通道经常没有形成有效竞争力。**冻结backbone、EWC、KD和mask共同保护旧状态，但正式LEO下旧类均值仍从42.83%降到23.17%，说明结构隔离不能自动抵消target receiver域偏移。

**逐类floor揭示均值掩盖的问题。**两种方法的\(A_{\min,\mathrm{old}}\)均值都低于4%，说明至少一部分旧发射机在增量后接近失效。Phase2方法不能只看\(A_{\mathrm{old}}^{\mathrm{post}}\)或\(A_{\mathrm{new}}\)。

### 5.6matched无LEO新类归因诊断

该诊断保持方法、物理样本ID、support/query划分、K-shot、seed和旧类评测条件一致，只把新类support/query替换为未叠加LEO的同一物理记录。结果必须标为：

`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`

|方法|无LEO\(A_{\mathrm{old}}^{\mathrm{post}}\)|无LEO\(A_{\mathrm{new}}\)|无LEO\(H_{\mathrm{old,new}}\)|\(\Delta A_{\mathrm{old}}^{\mathrm{post}}\)|\(\Delta A_{\mathrm{new}}\)|\(\Delta H_{\mathrm{old,new}}\)|\(\Delta F_{\mathrm{old}}\)|
|---|---:|---:|---:|---:|---:|---:|---:|
|CSIL|23.78%|12.15%|1.67%|+0.60pp|+3.50pp|+0.49pp|−0.60pp|
|MoPC-HR|21.45%|52.50%|12.98%|−0.68pp|+25.89pp|+2.13pp|+0.92pp|

移除新类LEO扰动后，MoPC-HR的\(A_{\mathrm{new}}\)平均提高25.89个百分点，说明LEO弱信道显著破坏了新类可分性；但\(A_{\mathrm{old}}^{\mathrm{post}}\)下降0.68个百分点、\(F_{\mathrm{old}}\)增加0.92个百分点，\(H_{\mathrm{old,new}}\)只提高2.13个百分点。CSIL的\(A_{\mathrm{new}}\)只提高3.50个百分点，说明其主要瓶颈还包括零步训练和过强稳定约束。

该诊断不能用于CVS卫星场景性能声明、方法晋级或Stage2协议有效性证明。

### 5.7低K条件下的零注册问题

若增量阶段只有\(N_{\text{new}}\times K\)个样本，batch size为\(B\)，并且DataLoader使用`drop_last=True`，则：

$$
\mathrm{floor}((N_{\text{new}}\times K)/B)=0
$$

当结果为0时，一个完整batch都不会产生，optimizer step也为0。新增分类权重没有学习，query自然继续被预测为旧类。

CSIL还要先执行约60%的官方训练切分，因此其有效样本数会进一步减少。正式LEO的400个CSIL cell中有175个零步cell，平均每场景5.625个optimizer step；400个MoPC-HR cell中有100个零步cell，平均每场景97.5步。两种条件均未启用缩batch、补采样或small-K训练适配。

低新类数并不必然意味着任务更简单。新增类之间可用于形成相对边界的样本更少，新增权重的方向和尺度可能不稳定；旧类logit经过充分训练，新类logit接近初始化，单头竞争会强烈偏向旧类。零步行只能说明官方trainer在该K/新类数组合下没有产生有效更新，不能解释成“训练后证明方法无效”。

## 6.统一比较与下一步

### 6.1五种方法的横向定位

|方法|任务类型|主要数据与权限|更新对象|核心特点与风险|
|---|---|---|---|---|
|ProtoNet CDA|度量型少样本分类|target-old support|prototype|最轻量，但不能修正复杂特征形变|
|MRIOR-SDA|监督域适应|source缓存＋target-old support|backbone＋DV-KL网络|域对齐能力强，但需要source replay|
|DADDA-SDA|统计距离域适应|source缓存＋target-old support|完整backbone|联合MMD/LMMD，低K时类条件估计不稳|
|CSIL|无exemplar类增量|target-new support＋旧模型/Fisher|新增分类结构；backbone冻结|稳定约束强，但可能不学习新类|
|MoPC-HR|prototype类增量|target-new support＋旧prototype/参数|backbone＋classifier|新类可塑性较强，但旧类遗忘明显|

方法不能只按最高准确率排序。比较时必须同时考虑任务类型、数据权限、更新范围、计算资源和同一row的\(A_{\mathrm{old}}^{\mathrm{post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)及\(A_{\min,\mathrm{old}}\)。

### 6.2当前实验发现

1. Phase2的困难具有递进性：先出现receiver域偏移，再叠加少样本新类注册和旧类遗忘。
2. MRIOR-SDA与DADDA-SDA证明target-old support具有域校准价值，但它们使用source replay和完整backbone更新，只能作为宽权限对照；ProtoNet CDA则说明单一prototype不足以修正复杂target embedding形变。
3. 正式LEO全矩阵中，MoPC-HR的\(A_{\mathrm{new}}\)为26.61%、\(H_{\mathrm{old,new}}\)为10.85%，高于CSIL的8.65%和1.18%；但两者\(A_{\mathrm{old}}^{\mathrm{post}}\)均只有约22%至23%，旧类稳定性仍未解决。
4. matched无LEO诊断使MoPC-HR的\(A_{\mathrm{new}}\)提高25.89个百分点，但\(H_{\mathrm{old,new}}\)只提高2.13个百分点且遗忘略增，说明LEO信道失真不是唯一瓶颈。
5. 固定batch与`drop_last`使部分低K配置产生零个optimizer step；这属于执行失效，不能解释为“训练后方法无效”。
6. 当前工作完成了任务定义、论文机制复现、CVS接口适配、完整结果审计和失败归因，尚未形成可晋级的Phase2主方法。

### 6.3下一步路线

1. **support-only域校准：**优先研究共享协方差、ridge/LDA、低秩adapter和归一化距离，在不回读source样本的条件下修正receiver偏移。
2. **统一非参数分类头：**研究KNN、加权KNN与qKNN，使新类support直接进入统一记忆库，避免固定batch和多轮增量训练：

$$
\hat y=\arg\max_c\sum_{i\in N_k(z_q)}w_i\mathbf 1(y_i=c)
$$

3. **安全训练与配对评价：**训练前检查有效batch和optimizer step；正式比较固定receiver、seed、K、类别集合、LEO观测和split，并在统一全类竞争下报告旧类、新类、调和均值、遗忘和逐类floor。

### 6.4证据边界

- WiSig/ManySig是地面代理数据，LEO弱信道是物理启发的仿真压力条件，不代表真实在轨验证。
- MRIOR-SDA与DADDA-SDA使用更宽source-access权限，只能作为机制对照。
- CSIL与MoPC-HR采用官方方法语义和CVS数据接口，其结果不等于`p2_min_v1`主方法已经满足support-only协议。
- matched无LEO结果仅用于归因，不能用于正式卫星场景声明或方法晋级。

## 参考文献

[1] SNELL J, SWERSKY K, ZEMEL R S. Prototypical Networks for Few-shot Learning[C]//Advances in Neural Information Processing Systems 30. 2017. https://papers.neurips.cc/paper/6996-prototypical-networks-for-few-shot-learning

[2] YANG L, LI Q, REN X, et al. Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation[J]. IEEE Internet of Things Journal, 2024, 11(13):24024-24034. DOI:10.1109/JIOT.2024.3389491.

[3] FENG J, FANG S, FAN Y. Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment[J]. IEEE Internet of Things Journal, 2025, 12(16):33202-33214. DOI:10.1109/JIOT.2025.3573713.

[4] LIU Y, WANG J, LI J, NIU S, SONG H. Class-Incremental Learning for Wireless Device Identification in IoT[J]. IEEE Internet of Things Journal, 2021, 8(23):17227-17235. DOI:10.1109/JIOT.2021.3078407.

[5] LI D, CHEN Z, SHAO M, et al. Non-Exemplar Class-Incremental Learning via Prototype Correction and Hierarchical Regularization for Specific Emitter Identification[J]. IEEE Transactions on Intelligent Transportation Systems, 2025, 26(8):12632-12646. DOI:10.1109/TITS.2025.3559174.

## 附录A：非类增量对比方法复现实验结果

本附录仅汇总ProtoNet CDA、MRIOR-SDA与DADDA-SDA的375项CVS复现实验。CSIL和MoPC-HR采用论文作者公开的官方代码，其CVS接口实验与结果见第5节。三种方法共享第4.1节的基座、target receiver、K-shot和seed矩阵，但数据权限不同：MRIOR-SDA与DADDA-SDA使用封存source LEO弱信道标签缓存与target-old support，ProtoNet CDA只使用冻结特征与target-old support。因此，附录结果只作机制对照，不构成同权限Phase2主方法排名。

### A.1总体实验结果

|方法|适应前\(A_{\mathrm{old}}\)|适应后\(A_{\mathrm{old}}\)|平均\(G_{\mathrm{old}}\)|正/负迁移任务|平均时延|3场景backbone更新|
|---|---:|---:|---:|---:|---:|---:|
|MRIOR-SDA|73.60%|82.58%|+8.98pp|105/20|17.90s|600|
|DADDA-SDA|73.60%|78.35%|+4.75pp|99/26|14.62s|600|
|ProtoNet CDA|73.60%|66.85%|−6.75pp|10/115|0.046s|0|

MRIOR-SDA获得最高旧类准确率，平均提升8.98个百分点；DADDA-SDA平均提升4.75个百分点；ProtoNet CDA平均下降6.75个百分点。target-old support确实包含可利用的域校准信息，但固定embedding上的单prototype不足以恢复跨接收机偏移。

### A.2不同K-shot下的结果

|K|直接ADV3B02|MRIOR-SDA|DADDA-SDA|ProtoNet CDA|
|---:|---:|---:|---:|---:|
|1|73.60%|77.22%|74.94%|58.67%|
|2|73.60%|79.51%|76.14%|64.70%|
|5|73.60%|82.59%|78.19%|68.98%|
|10|73.60%|85.82%|80.31%|70.42%|
|20|73.60%|87.74%|82.16%|71.48%|

K从1增加到20时，MRIOR-SDA与DADDA-SDA持续受益，说明更多target support提高了梯度估计和类条件对齐的稳定性。ProtoNet CDA也随K增加而改善，但K=20时仍低于直接ADV3B02。误差并非只来自prototype均值方差，目标域embedding还发生了系统性偏移。

### A.3不同target receiver下的结果

|target receiver|直接ADV3B02|MRIOR-SDA|DADDA-SDA|ProtoNet CDA|
|---|---:|---:|---:|---:|
|20-1|64.61%|83.43%|75.91%|60.98%|
|3-19|60.33%|69.06%|65.31%|48.50%|
|7-14|90.06%|89.93%|89.93%|83.53%|
|7-7|80.22%|86.78%|82.49%|74.81%|
|8-8|72.78%|83.69%|78.10%|66.42%|

- **20-1：**direct基线较低，MRIOR-SDA提升到83.43%，support提供了明确的目标域校准信号。
- **3-19：**最高结果为69.06%，是本轮最困难的receiver，说明后续方法必须报告逐receiver结果和逐类floor。
- **7-14：**direct已经达到90.06%，继续更新后轻微下降，高基线receiver存在负迁移风险。
- **7-7与8-8：**MRIOR-SDA保持明显正收益，DADDA-SDA获得中等收益，ProtoNet CDA仍低于direct。

### A.4复现实验结果的证据边界

1. 不同receiver的direct基线差距接近30个百分点，跨接收机域偏移是需要单独处理的性能来源。
2. 在允许source replay和完整backbone更新时，MRIOR-SDA与DADDA-SDA多数任务获得正迁移，证明target-old support能够提供域校准信号。
3. K增加后两种监督式域适应方法持续改善，说明support数量影响梯度估计和类条件分布估计的稳定性。
4. ProtoNet CDA的结果说明单prototype分类不等于域适应，尤其不能处理系统性的embedding旋转、拉伸或类内多峰。
5. 本附录不能证明support-only Phase2主方法已经解决旧类域适应；MRIOR-SDA与DADDA-SDA使用更宽权限，只能作为外部机制对照。
