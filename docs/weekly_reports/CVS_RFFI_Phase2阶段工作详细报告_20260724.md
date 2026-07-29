# CVS-RFFI Phase2阶段工作详细报告（截至2026年7月24日）

## 从跨接收机域适应到少样本新类注册：任务定义、方法复现、数据协议与实验结论

**汇报对象：**导师

**报告范围：**整合2026年7月16日与7月24日两轮工作

**研究主线：**Stage2-A无标签目标域参考→Stage2-B旧类K-shot适应→Stage2-C旧类保持与新类注册

**协议口径：**`p2_min_v1`

**涉及方法：**ProtoNet CDA、MRIOR-SDA、DADDA-SDA、CSIL、MoPC-HR

> **核心结论：**Phase2不是单一的“少样本分类”问题，而是由跨接收机域偏移、少样本参数估计、旧类保持、新类注册和部署权限共同构成的联合任务。MRIOR-SDA与DADDA-SDA证明target-old support包含有效的域校准信息，但它们依赖source replay和多轮backbone更新；CSIL与MoPC-HR能够扩展新类，却分别表现出“过度保护旧类导致新类不注册”和“新类可塑性增强但旧类遗忘”的问题。当前结果建立了清晰的问题边界和方法对照，还不能表述为Phase2主方法已经达到可晋级性能。

## 1.报告目的与近期工作全景

### 1.1报告主要介绍内容

本报告聚焦Phase2部署阶段，不再重复射频指纹识别的一般性背景。正文依次回答五个问题：

1. Phase2为什么划分为Stage2-A、Stage2-B和Stage2-C，各阶段的输入、允许更新的状态、输出和声明边界是什么。
2. 近期实验使用了哪些source/target数据，support、query、K-shot和LEO弱信道观测如何组织。
3. ProtoNet CDA、MRIOR-SDA、DADDA-SDA、CSIL和MoPC-HR分别属于什么方法类别，各自最突出的机制是什么。
4. 五种方法在ADV3B02-CVS接口上实际更新了什么参数，优化目标和损失函数如何定义，query是否参与更新。
5. Stage2-B域适应和Stage2-C新类注册得到了什么结果，哪些结论只是宽权限对比或matched诊断，哪些问题仍未解决。

### 1.2两轮工作的对应关系

|时间|Phase2阶段|核心问题|对比方法|主要输出|
|---|---|---|---|---|
|统一参考|Stage2-A|没有target标签时，Phase1模型能否直接跨接收机|直接ADV3B02|5个target receiver的\(A_{\mathrm{old}}^{\mathrm{pre}}\)参考|
|截至7月16日|Stage2-B|换到新接收机后，旧类准确率如何恢复|ProtoNet CDA、MRIOR-SDA、DADDA-SDA|375个域适应任务，分析K-shot、receiver和计算开销|
|截至7月24日|Stage2-C|如何注册新发射机，同时保留旧发射机|CSIL、MoPC-HR|800个正式LEO cell/2400个场景row及同规模matched无LEO诊断|

## 2.Phase2任务定义与三个Stage

### 2.1Phase2总体目标

Phase1在地面source receiver上训练并封存ADV3B02 deployment bundle。Phase2把该模型部署到与source receiver互斥的target receiver，使用固定LEO弱信道接收IQ和不同程度的target监督完成目标域推理。三个Stage的差异只由target标签权限和类别集合决定：

|阶段|target标签权限|类别集合|核心任务|主要输出|
|---|---|---|---|---|
|Stage2-A|无target TX标签|旧类参考|建立无标签目标域参考|reference/diagnostic|
|Stage2-B|旧类K-shot标签|\(Y_{\mathrm{old}}\)|完成旧类域适应|旧类预测器|
|Stage2-C|旧类与新类K-shot标签|\(Y_{\mathrm{old}}\cup Y_{\mathrm{new}}\)|保持旧类并注册新类|全注册类统一预测器|

Phase3的unknown拒识不属于Phase2-A/B/C。本报告只讨论已经注册的旧类和新类。

### 2.2Stage2-A：无标签目标域参考

**输入：**Phase1 bundle和不带TX标签的固定LEO target IQ；没有target support标签。

**任务：**测量Phase1模型直接跨接收机、跨信道后的旧类参考性能，或运行明确声明的无标签目标域诊断。

**输出边界：**Stage2-A不能声明旧类few-shot适应，也不能计算新类注册准确率。它回答的是“没有target标签时，模型能够直接迁移到什么程度”。

### 2.3Stage2-B：旧类K-shot域适应

**输入：**Phase1 bundle、\(Y_{\mathrm{old}}\)的K-shot target-old support及其标签。

**允许更新：**可根据方法更新prototype、adapter、分类头或backbone；具体权限必须逐方法说明。query只在状态冻结后测试。

**任务与输出：**用少量目标域旧类样本校准接收机/信道偏移，输出仍只覆盖\(Y_{\mathrm{old}}\)的预测器。ProtoNet CDA、MRIOR-SDA和DADDA-SDA属于本阶段对比方法。

### 2.4Stage2-C：旧类适应与新类注册

**输入：**Phase1 bundle、\(Y_{\mathrm{old}}\cup Y_{\mathrm{new}}\)的K-shot target support及其标签；其中\(Y_{\mathrm{new}}\)在Phase1从未出现。

**任务：**在同一target receiver中同时完成两件事：保持或校准旧类，注册新类。模型冻结后，每个query必须在全部已注册旧类和新类中统一竞争。

**输出与评价：**必须在同一row报告\(A_{\mathrm{old}}^{\mathrm{post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)和\(A_{\min,\mathrm{old}}\)。只保旧类或只学新类都不能称为Stage2-C成功。CSIL和MoPC-HR属于本阶段的外部类增量对比。

### 2.5关键集合与样本角色

$$
Y_{\mathrm{old}}\cap Y_{\mathrm{new}}=\varnothing
$$

- **旧类\(Y_{\mathrm{old}}\)：**Phase1已经见过的发射机；更换target receiver不会把旧类变成新类。
- **新类\(Y_{\mathrm{new}}\)：**Phase1未见、在Stage2-C获得合法support后注册的发射机。
- **support：**带标签目标域样本，可用于状态更新。
- **query：**纯测试样本，不能训练、调参、选择候选、设阈值或回滚。
- **K-shot：**每个已注册类别有K个互不重复的物理support样本；同一接收IQ的FFT、均衡或裁剪view不增加K。

类增量学习需要同时控制稳定性和可塑性。稳定性指旧类知识不被破坏，可塑性指新类能够形成有效决策区域；CSIL偏向稳定约束，MoPC-HR允许更强参数更新。

## 3.项目数据、Phase边界与权限

### 3.1Phase1：地面弱标注域泛化

Phase1在source receiver集合上训练身份表征。WiSig/ManySig在项目中属于地面代理数据或接收机代理域，不是真实卫星数据。Phase1可以使用source数据及物理启发的LEO增强训练模型，训练结束后封存不可变deployment bundle。

Phase1的目标是学到尽量稳定的发射机身份表征，不是执行部署few-shot。当前两轮工作统一使用ADV3B02地面域泛化checkpoint作为基座，避免不同backbone造成不公平比较。

### 3.2Phase2：目标接收机域部署

Phase2部署到与source receiver互斥的target receiver。当前`p2_min_v1`主方法运行时只允许读取：

1. 不可变Phase1 deployment bundle。
2. 已验证并封存的固定LEO弱信道接收IQ capsule。
3. 当前row的support标签、注册类别表和不含query真值的split。
4. 与数据无关的算法配置。

Phase2主方法不得运行时读取clean/raw/source样本、样本级source feature、source cache、source replay或能够影响决策的外部source状态。唯一受控例外是与checkpoint共同封存的只读、多样本聚合、不可逆int8 Phase1知识。

### 3.3单物理样本单LEO观测

每个clean/raw物理IQ在进入Phase2前只能随机选择一种允许的LEO弱信道：

$$
c_i\in\{\mathrm{leo\_clear\_weak},\mathrm{leo\_low\_elev\_weak},\mathrm{leo\_rain\_weak}\}
$$

同一物理样本不能复制后叠加多个LEO场景，再作为多份support或query。正式结果因此反映固定接收观测下的方法能力，而不是通过多次信道采样人为扩大K。

### 3.4本报告涉及的数据与实验矩阵

|工作包|基座与数据|target设置|实验规模|输出|
|---|---|---|---|---|
|旧类域适应|ADV3B02＋target-old support/query；MRIOR/DADDA另读source|5个target receiver；\(K\in\{1,2,5,10,20\}\)；5个seed|5×5×5×3=375个方法任务|\(A_{\mathrm{old}}^{\mathrm{pre/post}}\)、\(G_{\mathrm{old}}\)、正/负迁移、时延|
|新类注册|ADV3B02接口＋Phase2前base状态＋target-new support；old/new query只评分|5个target receiver；5个seed；\(K\in\{1,5,10,20\}\)；新类数\(\in\{2,5,10,20\}\)|800个正式LEO cell/2400个场景row|\(A_{\mathrm{old}}^{\mathrm{pre/post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)、\(A_{\min,\mathrm{old}}\)|
|信道归因诊断|保持方法、物理ID、split、K、seed和旧类条件一致，仅替换新类IQ为无LEO版本|与正式矩阵逐row配对|800个非正式cell/2400个场景row|\(\Delta A_{\mathrm{new}}\)、\(\Delta A_{\mathrm{old}}\)、\(\Delta H\)、\(\Delta F_{\mathrm{old}}\)、\(\Delta A_{\min,\mathrm{old}}\)|

本报告沿用已完成外部对比运行中的“正式LEO”命名，表示该冻结矩阵内的LEO条件，而不表示这些宽权限方法已经获得`p2_min_v1`主方法晋级资格。三个LEO场景分别训练、锁定和评分，不能把同一物理样本的多场景结果合并成更多K-shot support；其汇总仅用于对比方法机制分析。

## 4.评价指标与输入输出

记\(Q_{\mathrm{old}}\)和\(Q_{\mathrm{new}}\)分别为旧类、新类query集合，\(Q_c\)为旧类\(c\)的query集合；\(y_i\)是真值，\(\hat y_i^{(0)}\)和\(\hat y_i^{(1)}\)分别表示适应/注册前后的预测，\(\mathbb I[\cdot]\)为指示函数。为同时保证数学表达和实验字段可追溯，本文采用\(A_{\mathrm{old}}^{\mathrm{pre}}\)、\(A_{\mathrm{old}}^{\mathrm{post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)和\(A_{\min,\mathrm{old}}\)，分别对应结果字段`old_acc_before`、`old_acc_after`、`seen_new_acc`、`H_old_new`、`forgetting`和`min_old`。

### 4.1旧类适应指标

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

### 4.2新类注册指标

**已注册新类准确率：**

$$
A_{\mathrm{new}}
=\frac{1}{|Q_{\mathrm{new}}|}
\sum_{i\in Q_{\mathrm{new}}}
\mathbb I[\hat y_i^{(1)}=y_i]
$$

该指标为0意味着新类support没有形成可用的分类身份，或者增量训练根本没有产生有效更新。

### 4.3旧新联合指标

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

### 4.4统一输入、状态更新与输出

|阶段|输入|允许更新的状态|输出|
|---|---|---|---|
|直接基线|Phase1 bundle＋target query|无|旧类预测|
|Stage2-B适应|Phase1 bundle＋target-old support|prototype、adapter或backbone，取决于方法权限|冻结后的旧类预测器|
|Stage2-C注册|冻结/适应后状态＋\(Y_{\mathrm{old}}/Y_{\mathrm{new}}\) support|旧类统计、新类prototype、分类头或受控参数|面向全部注册类的统一预测器|
|独立评分|不可变prediction artifact＋query真值|不得回流到predictor|\(A_{\mathrm{old}}^{\mathrm{post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)等指标|

## 5.工作一：跨接收机旧类域适应

### 5.1研究问题与统一实验设置

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

### 5.2ProtoNet CDA：度量型少样本分类

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

### 5.3MRIOR-SDA：域对齐与target监督

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

### 5.4DADDA-SDA：全局与类条件动态分布对齐

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

### 5.5三种方法的机制与权限对比

|方法|方法分类|部署输入|更新对象|source访问|计算特点|本轮能否注册新类|
|---|---|---|---|---|---|---|
|ProtoNet CDA|度量型少样本分类|target-old support|prototype|否|0次backbone更新|未评估|
|MRIOR-SDA|模型更新型监督域适应|source LEO缓存＋target-old support|完整身份backbone＋DV-KL估计网络|是|200步/场景，3场景合计600步|否，闭集旧类|
|DADDA-SDA|统计距离型监督域适应|source LEO缓存＋target-old support|完整身份backbone|是|200步/场景，3场景合计600步|否，闭集旧类|

### 5.6非类增量对比方法的复现实验结果位置

为使正文先完成Phase2概念、任务、方法机制和类增量主线的讲解，ProtoNet CDA、MRIOR-SDA与DADDA-SDA的复现实验数值统一移至报告最后的“附录A：非类增量对比方法复现实验结果”。附录按总体结果、K-shot分组、target receiver分组和证据边界依次汇总。

CSIL与MoPC-HR使用论文作者公开的官方代码，并在此基础上完成CVS数据接口和评测矩阵适配。因此，第6节继续保留两种类增量方法的实验结果，但不把它们列入“对比方法复现实验结果”附录。

## 6.工作二：类增量学习与新类注册

### 6.1Stage2-C研究问题与数据

Stage2-C要求在同一个target receiver域中同时处理：

- target-old support：校准已经在Phase1见过的旧类。
- target-new support：注册Phase1未见的新发射机。
- old/new query：模型冻结后，在全部已注册类别中统一竞争。

本轮类增量对比严格采用论文作者公开仓库的执行语义，而不是早期简化移植版：CSIL锁定`pcwhy/CSIL@8ce8637`[12]，MoPC-HR锁定`xmuLdz/MoPC-HR@ae65543`[13]。原始编码器统一替换为预训练ADV3B02的160维\(z_{\mathrm{id}}\)特征接口，再保留CSIL或MoPC-HR各自的分类头、训练器、batch、`drop_last`、损失和更新范围。因此，实验定位是“官方方法语义＋CVS基座/数据接口适配”，不是原论文数据集上的数值复现。

#### Phase1基座状态与Phase2输入必须分开

**进入Phase2前的source base状态：**6个旧类、7个source receiver、2天、每个组合100条，共8400条地面source物理样本。MoPC-HR使用全部8400条训练base；CSIL按官方全局shuffle语义切成5879条base train和2521条互斥Fisher validation。它们用于构造进入Phase2前的旧类模型、Fisher统计或old prototype，不是每个Phase2增量cell反复回放的训练输入。

**Phase2真正用于新增类更新的数据：**仅使用当前cell中带合法标签的target-new K-shot support。正式条件下，新类support和query均为固定LEO弱信道观测；support/query物理ID互斥。旧类target query用于测量遗忘，新类target query用于测量注册能力，二者都不参与训练。

**冻结矩阵：**5个target receiver×5个seed×4个K（1、5、10、20）×4个新类规模（2、5、10、20）×2种方法，共800个cell；每个cell包含`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`三个正式场景，共2400个正式LEO row。每个query在模型锁定后独立对全部实际注册类竞争。

### 6.2CSIL：通道隔离型无exemplar类增量学习

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

### 6.3MoPC-HR：prototype校正与分层正则化

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

### 6.4CSIL与MoPC-HR机制对比

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

### 6.5正式LEO弱信道完整结果

正式LEO矩阵的800/800个cell、2400/2400个场景row和800/800份prediction/评分收据均完成，独立审计`failures=[]`。下表每行聚合1200个同条件、同方法的场景row：

|方法|\(A_{\mathrm{old}}^{\mathrm{pre}}\)|\(A_{\mathrm{old}}^{\mathrm{post}}\)|\(A_{\mathrm{new}}\)|\(H_{\mathrm{old,new}}\)|\(F_{\mathrm{old}}\)|\(A_{\min,\mathrm{old}}\)|
|---|---:|---:|---:|---:|---:|---:|
|CSIL|42.83%|23.17%|8.65%|1.18%|19.66%|0.82%|
|MoPC-HR|45.32%|22.14%|26.61%|10.85%|23.19%|3.89%|

这些是跨5个target receiver、5个seed、4个\(K\)、4个新类规模和3个LEO场景的全矩阵均值，不是挑选最佳\(K\)或最佳receiver后的结果。\(A_{\min,\mathrm{old}}\)表示每个row最低旧类准确率再求均值，用于观察少数旧类是否被严重牺牲。

### 6.6正式结果中的主要现象

**两种方法都没有在全矩阵上解决旧新平衡。**CSIL的正式LEO\(A_{\mathrm{new}}\)只有8.65%，\(H_{\mathrm{old,new}}\)仅1.18%；MoPC-HR的\(A_{\mathrm{new}}\)较高，为26.61%，但\(A_{\mathrm{old}}^{\mathrm{post}}\)仍只有22.14%，\(F_{\mathrm{old}}\)达到23.19个百分点。

**MoPC-HR比CSIL更具可塑性，但不是无代价提升。**其\(A_{\mathrm{new}}\)比CSIL高17.96个百分点，\(H_{\mathrm{old,new}}\)高9.67个百分点；与此同时，完整backbone更新使旧类表征更容易漂移。全矩阵结果不能写成“MoPC-HR已经解决新类注册”，只能说明它在当前执行语义下更倾向于学习新类。

**CSIL偏向稳定约束，但新类通道经常没有形成有效竞争力。**冻结backbone、EWC、KD和mask共同保护旧状态，但正式LEO下旧类均值仍从42.83%降到23.17%，说明结构隔离不能自动抵消target receiver域偏移。

**逐类floor揭示均值掩盖的问题。**两种方法的\(A_{\min,\mathrm{old}}\)均值都低于4%，说明至少一部分旧发射机在增量后接近失效。Phase2方法不能只看\(A_{\mathrm{old}}^{\mathrm{post}}\)或\(A_{\mathrm{new}}\)。

### 6.7matched无LEO新类归因诊断

该诊断保持方法、物理样本ID、support/query划分、K-shot、seed和旧类评测条件一致，只把新类support/query替换为未叠加LEO的同一物理记录。结果必须标为：

`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`

|方法|无LEO\(A_{\mathrm{old}}^{\mathrm{post}}\)|无LEO\(A_{\mathrm{new}}\)|无LEO\(H_{\mathrm{old,new}}\)|\(\Delta A_{\mathrm{old}}^{\mathrm{post}}\)|\(\Delta A_{\mathrm{new}}\)|\(\Delta H_{\mathrm{old,new}}\)|\(\Delta F_{\mathrm{old}}\)|
|---|---:|---:|---:|---:|---:|---:|---:|
|CSIL|23.78%|12.15%|1.67%|+0.60pp|+3.50pp|+0.49pp|−0.60pp|
|MoPC-HR|21.45%|52.50%|12.98%|−0.68pp|+25.89pp|+2.13pp|+0.92pp|

移除新类LEO扰动后，MoPC-HR的\(A_{\mathrm{new}}\)平均提高25.89个百分点，说明LEO弱信道显著破坏了新类可分性；但\(A_{\mathrm{old}}^{\mathrm{post}}\)下降0.68个百分点、\(F_{\mathrm{old}}\)增加0.92个百分点，\(H_{\mathrm{old,new}}\)只提高2.13个百分点。CSIL的\(A_{\mathrm{new}}\)只提高3.50个百分点，说明其主要瓶颈还包括零步训练和过强稳定约束。

该诊断不能用于CVS卫星场景性能声明、方法晋级或Stage2协议有效性证明。

### 6.8为什么低K、低新类数会出现零注册

若增量阶段只有\(N_{\text{new}}\times K\)个样本，batch size为\(B\)，并且DataLoader使用`drop_last=True`，则：

$$
\mathrm{floor}((N_{\text{new}}\times K)/B)=0
$$

当结果为0时，一个完整batch都不会产生，optimizer step也为0。新增分类权重没有学习，query自然继续被预测为旧类。

CSIL还要先执行约60%的官方训练切分，因此其有效样本数会进一步减少。正式LEO的400个CSIL cell中有175个零步cell，平均每场景5.625个optimizer step；400个MoPC-HR cell中有100个零步cell，平均每场景97.5步。两种条件均未启用缩batch、补采样或small-K训练适配。

低新类数并不必然意味着任务更简单。新增类之间可用于形成相对边界的样本更少，新增权重的方向和尺度可能不稳定；旧类logit经过充分训练，新类logit接近初始化，单头竞争会强烈偏向旧类。零步行只能说明官方trainer在该K/新类数组合下没有产生有效更新，不能解释成“训练后证明方法无效”。

## 7.五种方法的统一横向比较

|方法|解决方向|主要数据|更新对象|新类能力|旧类保护|资源与状态|报告定位|
|---|---|---|---|---|---|---|---|
|ProtoNet CDA|少样本度量分类|target support|prototype|机制上可扩展|不主动保护或校准backbone|极轻量|非类增量对比方法；复现实验结果见附录A|
|MRIOR-SDA|跨接收机域适应|source LEO缓存＋target-old support|完整身份backbone＋DV-KL估计网络|本轮无|source/target加权CE|200步/场景，source replay|非类增量对比方法；复现实验结果见附录A|
|DADDA-SDA|全局＋类条件域对齐|source LEO缓存＋target-old support|完整身份backbone|本轮无|source CE＋动态MMD/LMMD|200步/场景，source replay|非类增量对比方法；复现实验结果见附录A|
|CSIL|无exemplar类增量|target-new support＋旧模型/Fisher|新增fc行＋old-old/new-new fingerprint块；backbone冻结|可以|KD、EWC、mask与冻结|3epoch、batch20；存在零步cell|官方代码语义＋CVS基座/数据接口|
|MoPC-HR|prototype类增量|target-new support＋旧prototype/参数|完整backbone＋classifier|较强|prototype增强、HR|20epoch、batch16；存在零步cell|官方代码语义＋CVS基座/数据接口|

### 7.1这些方法实际上修改了什么

- ProtoNet只修改类别参考点，不修改特征空间。
- MRIOR与DADDA修改特征空间，使target旧类重新靠近正确决策区域。
- CSIL扩展分类结构，但Phase2只更新新增fc行和新fingerprint块，ADV3B02保持冻结。
- MoPC-HR用新类support与旧prototype伪特征更新完整ADV3B02和classifier，并记录prototype校正状态。

方法之间的差异不是“用了不同loss”这么简单。它们读取的数据、保存的历史状态、允许更新的参数和执行生命周期都不同。

### 7.2为什么不能只按最高准确率排序

MRIOR-SDA和DADDA-SDA使用source replay并完整更新backbone；ProtoNet只读target support；CSIL与MoPC-HR执行增量训练并保留不同历史状态。把这些结果放进同一数值排行榜，会把数据权限和计算开销误当成算法优劣。

正确比较至少需要同时报告：

1. 方法任务：闭集旧类域适应还是旧新类增量。
2. 数据权限：是否读取source、旧样本、旧prototype或历史模型。
3. 更新范围：prototype、分类头、adapter还是完整backbone。
4. 资源：训练步数、时延、存储状态和是否需要反向传播。
5. 同一row指标：\(A_{\mathrm{old}}^{\mathrm{post}}\)、\(A_{\mathrm{new}}\)、\(H_{\mathrm{old,new}}\)、\(F_{\mathrm{old}}\)和\(A_{\min,\mathrm{old}}\)。

## 8.当前困难与原因归纳

### 8.1域适应强度与部署权限冲突

MRIOR与DADDA表明完整backbone更新可以显著利用target support，但它们需要source replay和较高计算。Phase2主方法不能在部署时回读source样本，因此下一步需要把域校准能力压缩到support-only、低秩、闭式或轻量adapter中。

### 8.2轻量prototype依赖embedding质量

ProtoNet的失败说明当前target embedding并非简单的“同类中心平移”。receiver响应可能带来旋转、拉伸、类内多峰和类间重叠。后续prototype或KNN路线必须结合归一化、共享协方差、局部度量或轻量域校准。

### 8.3类增量存在稳定性—可塑性冲突

CSIL偏稳定：旧类保留但新类可能不注册。MoPC-HR偏可塑：新类能学到，但旧类下降。Stage2-C的目标不是选择任一极端，而是在统一全类竞争中同时提高\(A_{\mathrm{old}}^{\mathrm{post}}\)与\(A_{\mathrm{new}}\)。

### 8.4低K条件下训练流程本身可能失效

固定batch和`drop_last`使部分低K配置没有optimizer step。这是执行语义问题，不是LEO或RFFI机制本身。后续实验必须显式记录样本数、batch数和实际optimizer step，避免把“没有训练”误写成“方法训练后失败”。

### 8.5LEO弱信道放大support估计误差

K-shot support只能覆盖少量固定LEO观测。support与query的信道扰动不一致时，prototype、分类权重和类条件统计会把信道差异误当作类别差异。增加数学view不能增加物理K，因此需要用稳健估计、共享统计或不确定度建模降低误差。

## 9.下一步研究路线

### 9.1support-only旧类域校准

优先研究不读取Phase2 source数据的方法，包括低秩adapter、共享协方差、ridge/LDA、归一化弦距离和support驱动的轻量校准。适应前需要判断receiver是否真的需要更新，避免7-14这类高基线receiver发生负迁移。

### 9.2KNN、加权KNN与qKNN

Cover与Hart提出的KNN是非参数、实例型、局部度量分类方法[10]。它不把每类压缩成单一prototype，而是保存support局部结构：

$$
\hat y=\arg\max_c\sum_{i\in N_k(z_q)}w_i\mathbf 1(y_i=c)
$$

它适合Stage2-C的原因是：新类support可直接加入统一记忆库，不需要固定batch或多轮增量训练，也不会因更新encoder产生参数级灾难性遗忘。量化KNN（qKNN）可进一步压缩support embedding存储。

KNN仍需要解决目标域embedding质量、类别support数量不平衡、距离归一化、内存增长和逐query计算开销。

### 9.3统一全类竞争与旧类floor

旧类和新类必须使用同一推理规则，在全部实际注册类别中竞争。后续方法需要同时报告：

- \(A_{\mathrm{old}}^{\mathrm{post}}\)。
- \(A_{\mathrm{new}}\)。
- \(H_{\mathrm{old,new}}\)。
- \(F_{\mathrm{old}}\)。
- \(A_{\min,\mathrm{old}}\)。
- receiver级收益与负迁移比例。

不能通过old/new角色Oracle、类别配额、TX白名单或跨query重排修饰结果。

### 9.4小样本安全训练入口

对仍需要梯度更新的复现或候选方法，训练前应计算可用样本数和实际step：

1. 记录\(N_{\text{new}}\)、K、batch size和`drop_last`。
2. 确认至少产生一个有效batch。
3. 报告实际optimizer step，而不是只报告epochs。
4. 把论文原始trainer、CVS接口适配和项目优化版本分开命名。

### 9.5matched、同row、同权限的评价

后续正式比较需要固定target receiver、seed、K、old/new类别集合、LEO观测和split。只有在相同数据权限和资源范围内，数值差异才能用于方法晋级。

## 10.导师可直接带走的结论

1. Phase2的第一层困难是跨接收机域偏移，第二层困难是少样本条件下的新类注册与旧类遗忘，两者必须联合处理。
2. MRIOR-SDA和DADDA-SDA表明target-old support包含可利用的域校准信息，但两者使用source replay和完整backbone更新，只能作为宽权限外部对照；具体数值见附录A。
3. ProtoNet CDA说明轻量prototype分类不能自动替代接收机域适应；其总体、K-shot和target receiver结果见附录A。
4. CSIL强保护旧类，但低K、低新类数时经常没有有效训练或完全不输出新类。
5. 官方代码语义全矩阵中，MoPC-HR正式LEO的\(A_{\mathrm{new}}\)为26.61%、\(H_{\mathrm{old,new}}\)为10.85%，高于CSIL的8.65%和1.18%，但两者\(A_{\mathrm{old}}^{\mathrm{post}}\)都只有约22%至23%，稳定性问题仍然突出。
6. matched无LEO诊断中，MoPC-HR新类准确率平均提高25.89个百分点，但H只提高2.13个百分点且遗忘略增，因此LEO信道失真不是唯一矛盾。
7. 下一步应优先研究support-only轻量域校准＋KNN/qKNN或稳健prototype头，并用统一全类竞争、旧新调和均值和逐类floor评价。
8. 当前工作已经完成任务定义、论文机制复现、CVS接口适配、完整结果审计和失败原因定位；尚未得到可表述为Phase2主方法晋级成功的结论。

## 参考文献

[1] SNELL J, SWERSKY K, ZEMEL R S. Prototypical Networks for Few-shot Learning[C]//Advances in Neural Information Processing Systems 30. 2017. https://papers.neurips.cc/paper/6996-prototypical-networks-for-few-shot-learning

[2] YANG L, LI Q, REN X, et al. Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation[J]. IEEE Internet of Things Journal, 2024, 11(13):24024-24034. DOI:10.1109/JIOT.2024.3389491.

[3] FENG J, FANG S, FAN Y. Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment[J]. IEEE Internet of Things Journal, 2025, 12(16):33202-33214. DOI:10.1109/JIOT.2025.3573713.

[4] LIU Y, WANG J, LI J, NIU S, SONG H. Class-Incremental Learning for Wireless Device Identification in IoT[J]. IEEE Internet of Things Journal, 2021, 8(23):17227-17235. DOI:10.1109/JIOT.2021.3078407.

[5] LI D, CHEN Z, SHAO M, et al. Non-Exemplar Class-Incremental Learning via Prototype Correction and Hierarchical Regularization for Specific Emitter Identification[J]. IEEE Transactions on Intelligent Transportation Systems, 2025, 26(8):12632-12646. DOI:10.1109/TITS.2025.3559174.

[6] PAN S J, YANG Q. A Survey on Transfer Learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2010, 22(10):1345-1359. DOI:10.1109/TKDE.2009.191.

[7] GRETTON A, BORGWARDT K M, RASCH M J, et al. A Kernel Two-Sample Test[J]. Journal of Machine Learning Research, 2012, 13:723-773.

[8] KIRKPATRICK J, PASCANU R, RABINOWITZ N, et al. Overcoming Catastrophic Forgetting in Neural Networks[J]. Proceedings of the National Academy of Sciences, 2017, 114(13):3521-3526. DOI:10.1073/pnas.1611835114.

[9] HINTON G, VINYALS O, DEAN J. Distilling the Knowledge in a Neural Network[EB/OL]. arXiv:1503.02531, 2015. https://arxiv.org/abs/1503.02531

[10] COVER T, HART P. Nearest Neighbor Pattern Classification[J]. IEEE Transactions on Information Theory, 1967, 13(1):21-27. DOI:10.1109/TIT.1967.1053964.

[11] DE LANGE M, ALJUNDI R, MASANA M, et al. A Continual Learning Survey: Defying Forgetting in Classification Tasks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022, 44(7):3366-3385. DOI:10.1109/TPAMI.2021.3057446.

[12] LIU Y, et al. CSIL official implementation[CP/OL]. GitHub, commit 8ce8637daf4dc60eeb1c56bff64c050c5b2353e9. https://github.com/pcwhy/CSIL

[13] LI D, et al. MoPC-HR official implementation[CP/OL]. GitHub, commit ae6554316ad1a2175920e330133a2f103408bf78. https://github.com/xmuLdz/MoPC-HR

## 项目内部依据与证据边界

- `E:\type10-7\项目.md`：当前科学场景、Phase1/Phase2数据协议、Stage2-A/B/C权限和claim边界。
- `E:\type10-7\github_publish\CVS-RFFI-repo\docs\weekly_reports\学习进展情况_20260716_详细扩展版.md`：旧类域适应方法、375任务结果和解释。
- `E:\type10-7\github_publish\CVS-RFFI-repo\docs\weekly_reports\学习进展情况_20260724_详细扩展版.md`：CSIL/MoPC-HR机制、正式LEO结果与matched无LEO诊断。
- `E:\type10-7\github_publish\CVS-RFFI-repo\analysis\official_repo_execution_lock_csil_mopc_hr_20260723.md`：CSIL/MoPC-HR官方commit、训练日程、loss和接口适配锁。
- `E:\type10-7\github_publish\CVS-RFFI-repo\paper_reproduction\cvs_aligned\adv3b02_official_repo_ci.py`：类增量方法在ADV3B02-CVS上的实际参数更新、损失和query决策实现。
- `E:\type10-7\github_publish\CVS-RFFI-repo\paper_reproduction\cvs_aligned\supervised_da.py`与`adv3b02_supervised_da_runner.py`：ProtoNet CDA、MRIOR-SDA和DADDA-SDA的实际数据入口、优化目标和训练边界。
- `E:\type10-7\automation_reports\CV-SincNet\adv3b02_officialrepo_csil_mopc_20260723_v1\report.md`：8400条source base构建、800-cell完整矩阵、固定batch、zero-step和执行证据。
- 原始周报：`C:\Users\lh594\Desktop\周报\学习进展情况+7.16.docx`、`C:\Users\lh594\Desktop\周报\学习进展情况+7.24.docx`。

本文中的WiSig/ManySig只表示地面代理数据；LEO弱信道是物理启发的仿真压力条件，不是真实在轨验证。MRIOR-SDA和DADDA-SDA结果来自更宽source-access权限，不能与`p2_min_v1` support-only主方法同权限排名。CSIL和MoPC-HR的source base状态属于外部论文方法所需历史状态，其结果不能反证主方法已经满足support-only协议。无LEO新类结果仅为`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`归因诊断。

## 附录A：非类增量对比方法复现实验结果

本附录仅汇总ProtoNet CDA、MRIOR-SDA与DADDA-SDA的375项CVS复现实验。CSIL和MoPC-HR采用论文作者公开的官方代码，其CVS接口实验与结果见第6节。三种方法共享第5.1节的基座、target receiver、K-shot和seed矩阵，但数据权限不同：MRIOR-SDA与DADDA-SDA使用封存source LEO弱信道标签缓存与target-old support，ProtoNet CDA只使用冻结特征与target-old support。因此，附录结果只作机制对照，不构成同权限Phase2主方法排名。

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
