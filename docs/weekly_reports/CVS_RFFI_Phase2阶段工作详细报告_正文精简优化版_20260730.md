# 1.核心概念及其与Phase2阶段的对应

本章先区分域适应、少样本学习和类增量学习，再说明它们在CVS-RFFI Phase2中的具体位置。三者描述的不是同一维度：域适应关注数据分布变化，少样本学习关注目标任务的标注数量，类增量学习关注标签空间和模型状态随时间扩展。

## 1.1三个概念的核心区别

**域适应（domain adaptation，DA）：类别语义保持不变，但source域与target域的数据分布不同；模型利用协议允许的target信息降低target风险。**

**少样本学习（few-shot learning，FSL）：目标任务中每个类别只有少量带标签support；模型必须利用已有先验，在独立query上实现泛化。**

**类增量学习（class-incremental learning，CIL）：新类别分阶段加入，模型状态持续保存；每次更新后都要在全部已学习类别中统一预测。**

**少样本类增量学习（few-shot class-incremental learning，FSCIL）：类增量过程中的每个新类只有K-shot support，因而必须同时解决新类欠学习和旧类遗忘。**

**核心区别：DA回答“换了域如何适应”，FSL回答“标注很少如何学习”，CIL回答“类别持续增加如何保持旧知识”。Stage2-C同时包含receiver域偏移、K-shot新类和标签空间扩展，因此属于跨域FSCIL。**

|比较维度|域适应DA|少样本学习FSL|类增量学习CIL/FSCIL|
|---|---|---|---|
|发生变化的对象|数据域或接收条件|目标任务的监督样本数量|类别集合与持久化模型状态|
|标签空间|通常保持不变|一个episode内通常固定|随增量阶段持续扩大|
|主要学习输入|source知识＋target域信息|每类K个support＋预训练先验|当前新类数据＋允许保留的历史状态|
|query范围|target域既定类别|标准FSL通常为当前novel类|全部旧类与历次新类统一竞争|
|主要风险|域偏移导致决策边界失效|support过拟合、类别中心估计不准|灾难性遗忘、新类偏置与类别不平衡|
|CVS对应|Stage2-B|K-shot是Stage2-B/C的数据条件|Stage2-C；K-shot时为FSCIL|

## 1.2RFFI中的类别轴与域轴

RFFI接收信号可抽象为

$$
x=\Psi\!\left(\mathcal R_d\!\left[\mathcal H_d\!\left(\mathcal T_y(s)\right)\right]+n_d\right),
$$

其中，$\mathcal T_y$表示发射机硬件非理想性，决定身份类别；$\mathcal H_d$和$\mathcal R_d$分别表示信道和接收机链路，决定域条件；$\Psi$表示接收后的同步、裁剪、归一化或时频变换。**发射机身份变化属于类别变化，receiver或信道变化属于域变化，两者不能混为同一任务。**

旧类、增量新类和累计已注册类别分别记为

$$
\mathcal C_{\mathrm{old}},\qquad
\mathcal C_t^{\mathrm{new}},\qquad
\mathcal C^{(\le t)}
=\mathcal C_{\mathrm{old}}\cup\bigcup_{i=1}^{t}\mathcal C_i^{\mathrm{new}}.
$$

## 1.3少样本学习的严格任务定义

少样本学习不是泛指“总数据量较小”，而是指模型已经从base数据或预训练状态中获得先验，在目标任务每类只有少量带标签样本时，仍能泛化到未参与适配的query。

一次$N$-way $K$-shot任务记为$\tau=(S_\tau,Q_\tau)$。support包含$N$个类别，每类$K$个独立带标签样本：

$$
S_\tau
=\bigcup_{c\in\mathcal C_\tau}
\left\{(x_{c,k},c)\right\}_{k=1}^{K},
\qquad
|\mathcal C_\tau|=N,
\qquad
|S_\tau|=NK.
$$

学习算法读取先验状态$\Omega_0$和support，生成当前任务预测器：

$$
h_\tau=\mathcal A(S_\tau;\Omega_0),\qquad
\widehat y_q=\arg\max_{c\in\mathcal C_\tau}h_\tau(x_q)_c,
\quad(x_q,y_q)\in Q_\tau.
$$

support与query必须样本级互斥：

$$
S_\tau\cap Q_\tau=\varnothing.
$$

标准novel-class FSL还要求base类与novel类互斥，query通常只在当前novel类中竞争。若训练与测试类别相同，只是每类样本较少，更准确的名称是low-shot closed-set classification。ProtoNet[1]属于度量型FSL方法，但把prototype估计器用于旧类target support，并不会自动把任务变成标准novel-class FSL。

**K-shot统计的是每类K个独立物理IQ记录；同一IQ的FFT、裁剪、均衡或数据增强view均不增加K。query只用于冻结后评价，不能训练、调参、早停、设阈值或回滚。**

## 1.4域适应及Stage2-B定位

source域与target域可写为

$$
\mathcal D_s=(\mathcal X,P_s(X,Y)),\qquad
\mathcal D_t=(\mathcal X,P_t(X,Y)).
$$

跨接收机RFFI保持旧类标签空间不变，但类别条件分布发生变化：

$$
\mathcal C_s=\mathcal C_t=\mathcal C_{\mathrm{old}},
\qquad
P_s(X\mid Y)\neq P_t(X\mid Y).
$$

域适应的目标是利用source知识和允许读取的target数据，降低

$$
R_t(h)=\mathbb E_{(X,Y)\sim P_t}\!\left[\ell(h(X),Y)\right].
$$

Stage2-B每个旧类仅提供K个target-old support，因此属于**少样本监督域适应** ：少样本描述target标签预算，域适应才是任务本质。Phase1域泛化训练不能读取未来target receiver；Phase2域适应发生在部署以后，可以读取协议允许的target support。是否接触target数据，是DG与DA的关键边界。

## 1.5类增量、FSCIL与新类注册

类增量学习要求新类按阶段到达，并把更新后的状态持续保存：

$$
\mathcal C^{(\le t)}
=\mathcal C^{(\le t-1)}\cup\mathcal C_t^{\mathrm{new}},
\qquad
\widehat y
=\arg\max_{c\in\mathcal C^{(\le t)}}h_t(x)_c.
$$

推理时不提供task ID，模型必须让旧类与新类在同一标签空间竞争。当每个增量新类只有K个support时，任务成为FSCIL。一次新类集合注册只能评价single-session/one-step类扩展；若要声明持续FSCIL，还需连续执行多个增量session，并在每个session后评价累计类别和遗忘。

CVS中的“新类注册”是部署操作：利用带标签新类support建立prototype、分类权重、adapter或其他持久状态，使该身份从未注册unknown转为seen-new class。应严格区分

$$
\text{unknown rejection}\neq\text{new-class registration}.
$$

unknown尚未获得可信标签，不能直接进入注册集合；只有获得合法标签和support后，才能成为已注册新类。

## 1.6与CVS-RFFI Phase2阶段的对应

|阶段|域是否变化|类别是否增加|可用target标签|准确任务定位|冻结后query范围|
|---|---|---|---|---|---|
|Stage2-A|是|否|无|zero-label transfer/reference；仅在实际使用无标签target更新时才属于UDA|旧类target query|
|Stage2-B|是|否|旧类K-shot|少样本监督域适应|旧类target query|
|Stage2-C|是|是|新类K-shot；旧类状态按协议保留|跨域单步FSCIL/新类注册；连续多session时为完整FSCIL|旧类＋已注册新类统一query|
|Phase3|是|可能|unknown无真值|开集拒识与开放世界扩展|旧类＋已注册新类＋未注册unknown|

**Stage2-A不属于K-shot学习；Stage2-B类别不增加，不是标准novel-class FSL；Stage2-C同时扩大标签空间并要求旧新统一竞争，是少样本类增量任务。**

## 1.7样本角色与成功条件

- **旧类$\mathcal C_{\mathrm{old}}$：** Phase1已经见过的发射机；更换receiver不改变其身份。
- **新类$\mathcal C_t^{\mathrm{new}}$：** Phase1未见、在Stage2-C通过合法support注册的发射机。
- **未注册unknown：** 尚未获得可信标签或尚未加入已注册类别集合的发射机。
- **support：** 唯一允许参与适应、注册和状态更新的带标签样本。
- **query：** 模型冻结后用于评价的独立样本，不得影响predictor。
- **K-shot：** 每类K个互不重复的物理接收观测。

# 2.数据、权限与评价框架

## 2.1从Phase1训练到Phase2部署

WiSig/ManySig在本项目中是地面代理数据，不是真实卫星数据。两轮实验统一使用ADV3B02地面域泛化checkpoint，避免backbone差异干扰方法比较。

| **环节**        | **可用数据**                                    | **允许更新**                                     | **输出**                |
|-----------------|-------------------------------------------------|--------------------------------------------------|-------------------------|
| Phase1地面训练  | source数据及物理启发的LEO增强                   | 身份backbone与训练状态                           | 不可变deployment bundle |
| Phase2适应/注册 | bundle、固定LEO target IQ、当前row的support标签 | 由方法声明的prototype、分类头、adapter或backbone | 冻结后的目标域预测器    |
| 独立评分        | prediction artifact与query真值                  | 不得回流到predictor                              | 同一row评价指标         |

p2_min_v1主方法不能在Phase2运行时读取clean/raw/source样本、source feature/cache/replay或其他可影响决策的外部source状态。外部论文对比若使用更宽权限，必须单独标注，不能反向证明主方法满足协议。

## 2.2固定LEO观测与query隔离

每个clean/raw物理IQ在进入Phase2前只能随机选择一种允许的LEO弱信道：

$$c_{i} \in \{ leo\_ clear\_ weak,leo\_ low\_ elev\_ weak,leo\_ rain\_ weak\}$$

同一物理样本不能复制后叠加多个LEO场景再扩充K。support可以更新状态；query只在模型冻结后逐样本测试，不能参与训练、调参、回滚或跨query重排。

## 2.3实验矩阵

| **工作包**   | **基座与数据**                                                          | **target设置**                                                                  | **实验规模**                    | **输出**                                                                                |
|--------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------|-----------------------------------------------------------------------------------------|
| 旧类域适应   | ADV3B02＋target-old support/query；MRIOR/DADDA另读source                | 5个target receiver；$K \in \{ 1,2,5,10,20\}$；5个seed                           | 5×5×5×3=375个方法任务           | $A_{old}^{pre/post}$、$G_{old}$、正/负迁移、时延                                        |
| 新类注册     | ADV3B02接口＋Phase2前base状态＋target-new support；old/new query只评分  | 5个target receiver；5个seed；$K \in \{ 1,5,10,20\}$；新类数$\in \{ 2,5,10,20\}$ | 800个正式LEO cell/2400个场景row | $A_{old}^{pre/post}$、$A_{new}$、$H_{old,new}$、$F_{old}$、$A_{\min,old}$               |
| 信道归因诊断 | 保持方法、物理ID、split、K、seed和旧类条件一致，仅替换新类IQ为无LEO版本 | 与正式矩阵逐row配对                                                             | 800个非正式cell/2400个场景row   | $\Delta A_{new}$、$\Delta A_{old}$、$\Delta H$、$\Delta F_{old}$、$\Delta A_{\min,old}$ |

本报告沿用已完成外部对比运行中的“正式LEO”命名，表示该冻结矩阵内的LEO条件，而不表示这些宽权限方法已经获得p2_min_v1主方法晋级资格。三个LEO场景分别训练、锁定和评分，不能把同一物理样本的多场景结果合并成更多K-shot support；其汇总仅用于对比方法机制分析。

## 2.4评价指标

记$Q_{old}$和$Q_{new}$分别为旧类、新类query集合，$Q_{c}$为旧类$c$的query集合；$y_{i}$是真值，${\widehat{y}}_{i}^{(0)}$和${\widehat{y}}_{i}^{(1)}$分别表示适应/注册前后的预测，$\mathbb{I}\lbrack \cdot \rbrack$为指示函数。为同时保证数学表达和实验字段可追溯，本文采用$A_{old}^{pre}$、$A_{old}^{post}$、$A_{new}$、$H_{old,new}$、$F_{old}$和$A_{\min,old}$，分别对应结果字段old_acc_before、old_acc_after、seen_new_acc、H_old_new、forgetting和min_old。

### 旧类适应

**适应前旧类准确率：**

$$A_{old}^{pre} = \frac{1}{\left| Q_{old} \right|}\sum_{i \in Q_{old}}^{}\mathbb{I}\left\lbrack {\widehat{y}}_{i}^{(0)} = y_{i} \right\rbrack$$

**适应后旧类准确率：**

$$A_{old}^{post} = \frac{1}{\left| Q_{old} \right|}\sum_{i \in Q_{old}}^{}\mathbb{I}\left\lbrack {\widehat{y}}_{i}^{(1)} = y_{i} \right\rbrack$$

**适应收益：**

$$G_{old} = A_{old}^{post} - A_{old}^{pre}$$

$G_{old}$为正表示正迁移，为负表示适应损伤了原有能力。

### 新类注册

**已注册新类准确率：**

$$A_{new} = \frac{1}{\left| Q_{new} \right|}\sum_{i \in Q_{new}}^{}\mathbb{I}\left\lbrack {\widehat{y}}_{i}^{(1)} = y_{i} \right\rbrack$$

该指标为0意味着新类support没有形成可用的分类身份，或者增量训练根本没有产生有效更新。

### 旧新联合评价

**旧新调和均值：**

$$H_{\text{old,new}} = \frac{2A_{old}^{post}A_{new}}{A_{old}^{post} + A_{new}}$$

调和均值会惩罚“只保旧类、不学新类”和“只学新类、忘掉旧类”。任意一侧接近0，$H_{\text{old,new}}$都会接近0。

**遗忘：**

$$F_{old} = A_{old}^{pre} - A_{old}^{post}$$

遗忘越大，说明增量学习对旧知识破坏越严重。

**最低旧类准确率：**

$$A_{c} = \frac{1}{\left| Q_{c} \right|}\sum_{i \in Q_{c}}^{}\mathbb{I}\left\lbrack {\widehat{y}}_{i}^{(1)} = y_{i} \right\rbrack$$

$$A_{\min,old} = \min_{c \in Y_{old}}A_{c}$$

$A_{\min,old}$直接检查最差旧发射机是否接近失效，避免平均$A_{old}^{post}$掩盖局部崩塌。

## 2.5统一输入、状态更新与输出

| **阶段**     | **输入**                                           | **允许更新的状态**                           | **输出**                                                    |
|--------------|----------------------------------------------------|----------------------------------------------|-------------------------------------------------------------|
| 直接基线     | Phase1 bundle＋target query                        | 无                                           | 旧类预测                                                    |
| Stage2-B适应 | Phase1 bundle＋target-old support                  | prototype、adapter或backbone，取决于方法权限 | 冻结后的旧类预测器                                          |
| Stage2-C注册 | 冻结/Stage2-B适应后状态＋target-new K-shot support | 旧类统计、新类prototype、分类头或受控参数    | 面向旧类∪已注册新类的统一预测器                             |
| 独立评分     | 不可变prediction artifact＋query真值               | 不得回流到predictor                          | $A_{old}^{post}$、$A_{new}$、$H_{old,new}$、$F_{old}$等指标 |

# 3.Stage2-B：跨接收机旧类域适应仿真实验

## 3.1仿真问题与统一实验设置

Stage2-B要回答的问题是：Phase1已经识别过的发射机换到未见target receiver后，少量target-old support能否恢复旧类识别能力？

统一实验设置如下：

| **维度**        | **设置**                              |
|-----------------|---------------------------------------|
| 基座模型        | 同一ADV3B02地面域泛化checkpoint       |
| target receiver | 20-1、3-19、7-14、7-7、8-8            |
| K-shot          | 1、2、5、10、20                       |
| 随机性          | 5个独立seed                           |
| target观测      | support和query均为固定LEO弱信道接收IQ |
| query权限       | 只测试，不训练、不调参、不回滚        |
| 方法            | ProtoNet CDA、MRIOR-SDA、DADDA-SDA    |

## 3.2 ProtoNet CDA：原型式K-shot目标域校准基线

Snell等人提出的Prototypical Networks原本面向训练阶段未见的新类别少样本分类\[1\]。本轮实验的support与query均属于Phase1已见旧类，类别不增加，只是receiver域发生变化。因此，ProtoNet CDA在本报告中被定位为借用prototype估计器的少样本监督域适应基线，而不是标准novel-class FSL。其特点是固定特征空间，只用target-old support闭式计算类别中心；不访问source、不反向传播。

**CVS数据与更新：** 加载ADV3B02的160维身份特征$z = f_{\theta}(x)$，每个场景只读取6个旧类的K-shot target-old support。对类别$c$的support集合$S_{c}$，prototype是类内平方距离最小化问题的闭式解：

$$p_{c}^{*} = \arg\min_{p}\sum_{i \in S_{c}}^{} \parallel z_{i} - p \parallel_{2}^{2} = \frac{1}{\left| S_{c} \right|}\sum_{i \in S_{c}}^{}z_{i}$$

本轮没有可训练损失：

$$\mathcal{L}_{\text{ProtoNet-CDA}} = 0,\quad\quad gradient\_ updates = 0$$

模型锁定后，query按欧氏距离最近的prototype分类：

$$\widehat{y}(x) = \arg\min_{c \in Y_{old}} \parallel f_{\theta}(x) - p_{c}^{*} \parallel_{2}^{2}$$

**核心局限：** 它只能移动类别参考点，不能修正接收机造成的特征旋转、拉伸或类内多峰，因此“使用了support”不等于“完成了域适应”。

## 3.3MRIOR-SDA：域对齐与target监督

Yang等人的MRIOR原本属于单源无监督域适应\[2\]。CVS版本用真实target-old support标签替代伪标签，形成监督式域适应。其特点是用可学习的DV-KL估计网络对齐source与target，同时更新完整ADV3B02身份backbone。

记$B_{s},B_{t}$为source和target-support batch，$p_{\theta}\left( c|x \right)$为旧类预测概率，$w_{c}$为根据target-support类别频数得到并归一化到均值1的类别权重：

$$w_{c} = \frac{\left( n_{c} + \epsilon \right)^{- 1}}{\frac{1}{C}\sum_{r = 1}^{C}\left( n_{r} + \epsilon \right)^{- 1}}$$

source和target的加权交叉熵为：

$$\mathcal{L}_{wCE}^{D} = - \frac{1}{\left| B_{D} \right|}\sum_{\left( x_{i},y_{i} \right) \in B_{D}}^{}w_{y_{i}}\log p_{\theta}\left( y_{i}|x_{i} \right),\quad\quad D \in \{ s,t\}$$

令$T_{\phi}(z)$为DV估计网络，域差异项为：

$$\mathcal{L}_{DV - KL} = \frac{1}{\left| B_{s} \right|}\sum_{i \in B_{s}}^{}T_{\phi}\left( z_{i}^{s} \right) - \log\left\lbrack \frac{1}{\left| B_{t} \right|}\sum_{j \in B_{t}}^{}\exp T_{\phi}\left( z_{j}^{t} \right) \right\rbrack$$

ADV3B02的外层优化目标为：

$$\mathcal{L}_{\text{MRIOR-SDA}} = 0.5\mathcal{L}_{\text{wCE}}^{\text{source}} + 0.5\mathcal{L}_{\text{wCE}}^{\text{target-support}} + 0.005\mathcal{L}_{\text{DV-KL}}$$

**CVS执行：** 每个场景使用封存source LEO弱信道标签缓存和6类K-shot target-old LEO support。先用7个内层step最大化DV估计，再更新backbone；两个Adam学习率均为$6 \times 10^{- 4}$，每场景200个外层step。query只在训练结束后测试。该方法的代价是source replay和完整backbone更新，因此只作为宽权限机制对照。

## 3.4DADDA-SDA：全局与类条件动态分布对齐

Feng等人的DADDA属于基于统计距离的跨接收机域适应\[3\]。其特点是同时计算全局MMD和类条件LMMD，再用数据驱动的动态分配二者权重。

source与target-support的监督分类项均采用标准交叉熵：

$$\mathcal{L}_{CE}^{D} = - \frac{1}{\left| B_{D} \right|}\sum_{\left( x_{i},y_{i} \right) \in B_{D}}^{}\log p_{\theta}\left( y_{i}|x_{i} \right),\quad\quad D \in \{ s,t\}$$

采用RBF核：

$$k(u,v) = \exp\left( - \frac{\parallel u - v \parallel_{2}^{2}}{2\sigma^{2}} \right)$$

全局MMD为：

$$\mathcal{L}_{MMD} = \frac{1}{n_{s}^{2}}\sum_{i,i\prime}^{}k\left( z_{i}^{s},z_{i\prime}^{s} \right) + \frac{1}{n_{t}^{2}}\sum_{j,j\prime}^{}k\left( z_{j}^{t},z_{j\prime}^{t} \right) - \frac{2}{n_{s}n_{t}}\sum_{i,j}^{}k\left( z_{i}^{s},z_{j}^{t} \right)$$

对类别$c$，令$w_{s}^{c},w_{t}^{c}$为按该类样本数归一化的source/target-support权重，类条件LMMD为：

$$\mathcal{L}_{LMMD} = \sum_{c = 1}^{C}\left\lbrack \left( w_{s}^{c} \right)^{\top}K_{ss}w_{s}^{c} + \left( w_{t}^{c} \right)^{\top}K_{tt}w_{t}^{c} - 2\left( w_{s}^{c} \right)^{\top}K_{st}w_{t}^{c} \right\rbrack$$

动态系数和总损失分别为：

$$\alpha = \frac{\mathcal{L}_{MMD}}{\mathcal{L}_{MMD} + \mathcal{L}_{LMMD} + \epsilon}$$

$$\mathcal{L}_{\text{DADDA-SDA}} = \mathcal{L}_{\text{CE}}^{\text{source}} + \mathcal{L}_{\text{CE}}^{\text{target-support}} + (1 - \alpha)\mathcal{L}_{\text{MMD}} + \alpha\mathcal{L}_{\text{LMMD-sum}}$$

**CVS执行：** 数据与MRIOR-SDA相同；更新完整ADV3B02身份backbone和旧类分类输出。SGD初始学习率为$10^{- 4}$、momentum为0.9、weight decay为$5 \times 10^{- 4}$，每场景200步。query不参与CE、MMD、LMMD或$\alpha$估计。DADDA比MRIOR更直接地约束类条件分布，但K很小时$w_{t}^{c}$估计噪声较大，可能产生负迁移。

## 3.5机制、权限与结果位置

| **方法**     | **方法分类**         | **部署输入**                       | **更新对象**                    | **source访问** | **计算特点**               | **本轮能否注册新类** |
|--------------|----------------------|------------------------------------|---------------------------------|----------------|----------------------------|----------------------|
| ProtoNet CDA | 少样本监督DA基线     | target-old support                 | prototype                       | 否             | 0次backbone更新            | 未评估               |
| MRIOR-SDA    | 模型更新型监督域适应 | source LEO缓存＋target-old support | 完整身份backbone＋DV-KL估计网络 | 是             | 200步/场景，3场景合计600步 | 否，闭集旧类         |
| DADDA-SDA    | 统计距离型监督域适应 | source LEO缓存＋target-old support | 完整身份backbone                | 是             | 200步/场景，3场景合计600步 | 否，闭集旧类         |

ProtoNet CDA、MRIOR-SDA与DADDA-SDA的复现实验数值统一放在附录A，正文只保留理解方法和权限差异所需的信息。

# 4.Stage2-C：少样本类增量仿真实验

## 4.1仿真问题、代码口径与数据

Stage2-C在同一target receiver中注册新类，并要求旧类与新类在同一输出空间统一竞争，因此它不是标准“仅在当前新类query中评价”的FSL，而是类扩展/FSCIL设置。本轮每个cell只执行一次新类集合注册，严格来说属于single-session（one-step）跨域FSCIL评价；只有在多个增量session之间持续继承模型状态并逐轮扩大类别集合时，才能形成完整的multi-session FSCIL实验。本轮使用论文作者官方执行语义：CSIL锁定pcwhy/CSIL@8ce8637，MoPC-HR锁定xmuLdz/MoPC-HR@ae65543；编码器接口替换为ADV3B02的160维，其余训练器、损失和更新范围按官方实现保留。

| **环节**           | **CSIL**                                                        | **MoPC-HR**                                 |
|--------------------|-----------------------------------------------------------------|---------------------------------------------|
| Phase2前基座       | 5879条source base train＋2521条Fisher validation                | 8400条source base训练并保存6个旧类prototype |
| Phase2真实训练数据 | 当前cell的target-new K-shot support                             | 当前cell的target-new K-shot support         |
| query作用          | 旧类测遗忘、新类测注册；均不参与训练                            | 旧类测遗忘、新类测注册；均不参与训练        |
| 冻结矩阵           | 5个receiver×5个seed×4个K×4个新类规模，共400个cell/1200个LEO row | 同左                                        |

这里的source数据只用于构造进入Phase2前的旧类状态，不是每个增量cell中的source replay。每个cell的唯一真实增量训练数据是target-new K-shot support；old query与new query均只用于模型冻结后的统一评价，不能参与训练、调参、回滚或阈值选择。实验属于“官方方法语义＋CVS基座/数据接口适配”，不是原论文数据集数值复现。

## 4.2CSIL：通道隔离型无exemplar类增量学习

Liu等人提出的CSIL属于无exemplar、结构扩展型类增量方法\[4\]。其特点是冻结ADV3B02 backbone，以通道扩展和mask限制新类更新，再用EWC与KD保护旧类响应。

**CVS数据与更新：** 进入Phase2前，5879条source样本训练fc_bf_fp→zero-bias Fingerprints基座，2521条互斥source样本估计Fisher。Phase2只用target-new support训练；ADV3B02冻结，fc只更新新增行和bias，fingerprint只允许old-old与new-new块更新，两个交叉块保持为0。

记$B_{new}$为新类support训练batch，$q_{\theta}\left( c|x \right)$为全部注册类概率。新类交叉熵为：

$$\mathcal{L}_{CE,new} = - \frac{1}{\left| B_{new} \right|}\sum_{\left( x_{i},y_{i} \right) \in B_{new}}^{}\log q_{\theta}\left( y_{i}|x_{i} \right)$$

EWC使用Phase2前参数$\theta^{*}$和Fisher重要性$F_{j}$：

$$\mathcal{L}_{EWC} = \frac{1}{2}\sum_{j}^{}F_{j}\left( \theta_{j} - \theta_{j}^{*} \right)^{2}$$

令$r_{old}^{*}(x)$和$r_{old}(x)$为旧模型与当前模型的旧类fingerprint响应，官方KD为：

$$\mathcal{L}_{KD} = \frac{1}{32}\sum_{x \in B_{new}}^{} \parallel r_{old}^{*}(x) - r_{old}(x) \parallel_{2}^{2}$$

实际执行总损失为：

$$\mathcal{L}_{\text{CSIL}} = \mathcal{L}_{\text{CE,new}} + \mathcal{L}_{\text{EWC}} + 0.2\mathcal{L}_{\text{KD}}$$

**执行特点：** 3个epoch、batch size 20、drop_last=True，手工SGDM学习率为$0.01/(1 + 0.01t)$。query用zero-bias全注册类argmax，训练步数为0。该方法旧类约束最强，但低K时可能没有完整batch，或者新类更新被旧类约束压制。

## 4.3MoPC-HR：prototype校正与分层正则化

Li等人提出的MoPC-HR属于无exemplar、prototype校正型类增量方法\[5\]。其特点是允许完整ADV3B02随新类support更新，同时用旧prototype伪特征和分层参数正则限制遗忘。

**CVS数据与更新：** 进入Phase2前用全部8400条source旧类样本训练基座并保存6个old prototype。Phase2真实样本只来自target-new support；每个step另采样16个old prototype并加入$\epsilon \sim \mathcal{N}\left( 0,{0.05}^{2}I \right)$。ADV3B02 backbone和Linear classifier均更新。

新类真实样本交叉熵为：

$$\mathcal{L}_{CE,new} = - \frac{1}{\left| B_{new} \right|}\sum_{\left( x_{i},y_{i} \right) \in B_{new}}^{}\log q_{\theta}\left( y_{i}|x_{i} \right)$$

令${\widetilde{z}}_{r} = p_{c_{r}} + \epsilon_{r}$为增强后的旧类prototype，温度$\tau = 2$，prototype augmentation损失为：

$$\mathcal{L}_{protoAug} = - \frac{1}{B_{p}}\sum_{r = 1}^{B_{p}}\log softmax\left( \frac{g_{\theta}\left( {\widetilde{z}}_{r} \right)}{\tau} \right)_{c_{r}}$$

公开trainer对第$\ell$组参数使用递减系数$a_{\ell} = 1 - (\ell - 1)/L$，分层正则为：

$$\mathcal{L}_{HR} = \sum_{\ell = 1}^{L}a_{\ell} \parallel \theta_{\ell} - \theta_{\ell}^{*} \parallel_{2}$$

实际执行总损失为：

$$\mathcal{L}_{\text{MoPC-HR}} = \mathcal{L}_{\text{CE,new}} + \mathcal{L}_{\text{protoAug}} + \mathcal{L}_{\text{HR}}$$

公开代码还计算KD用于日志，但KD不进入总loss。训练后prototype校正为：

$${\widetilde{P}}_{old} = 0.97P_{old} + 0.03softmax\left( P_{old}\left( P_{new}^{*} \right)^{\top} \right)\left( P_{new} - P_{new}^{*} \right)$$

**执行特点：** 20个epoch、batch size 16、SGD学习率0.01。正式CVS query仍使用当前模型的全注册类classifier logits，而不是校正prototype。该方法新类可塑性强于CSIL，但完整backbone更新会放大旧类遗忘。

## 4.4CSIL与MoPC-HR机制对比

| **比较维度**       | **CSIL**                                    | **MoPC-HR**                                   |
|--------------------|---------------------------------------------|-----------------------------------------------|
| 主要方法类别       | 结构扩展与通道隔离                          | prototype校正与分层正则                       |
| 历史原始样本       | 不保存                                      | 不保存                                        |
| Phase2真实训练样本 | target-new support                          | target-new support＋旧prototype伪特征         |
| 保留的历史状态     | 旧模型、fingerprint结构、Fisher重要参数统计 | 旧类prototype、旧模型参数                     |
| ADV3B02增量更新    | 冻结                                        | 不冻结                                        |
| 实际损失           | CE-new＋EWC＋0.2KD                          | CE-new＋protoAug＋逐参数非平方L2 HR；KD仅记录 |
| 旧类保护           | mask、冻结、KD、EWC                         | prototype增强、HR、旧参数参考                 |
| 最终query规则      | zero-bias全注册类argmax                     | 当前模型全注册类classifier logits argmax      |
| 主要风险           | 过度保护旧类，新类不注册                    | 新类学得快，但旧类遗忘                        |
| 部署特征           | 结构复杂、容量增长                          | 状态较紧凑，但仍需多轮训练                    |

## 4.5正式LEO弱信道结果

正式LEO矩阵的800/800个cell、2400/2400个场景row和800/800份prediction/评分收据均完成，独立审计failures=\[\]。下表每行聚合1200个同条件、同方法的场景row：

| **方法** | $A_{old}^{pre}$ | $A_{old}^{post}$ | $A_{new}$ | $H_{old,new}$ | $F_{old}$ | $A_{\min,old}$ |
|----------|-----------------|------------------|-----------|---------------|-----------|----------------|
| CSIL     | 42.83%          | 23.17%           | 8.65%     | 1.18%         | 19.66%    | 0.82%          |
| MoPC-HR  | 45.32%          | 22.14%           | 26.61%    | 10.85%        | 23.19%    | 3.89%          |

这些是跨5个target receiver、5个seed、4个$K$、4个新类规模和3个LEO场景的全矩阵均值，不是挑选最佳$K$或最佳receiver后的结果。$A_{\min,old}$表示每个row最低旧类准确率再求均值，用于观察少数旧类是否被严重牺牲。

### 结果解读

**两种方法都没有在全矩阵上解决旧新平衡。** CSIL的正式LEO$A_{new}$只有8.65%，$H_{old,new}$仅1.18%；MoPC-HR的$A_{new}$较高，为26.61%，但$A_{old}^{post}$仍只有22.14%，$F_{old}$达到23.19个百分点。

**MoPC-HR比CSIL更具可塑性，但不是无代价提升。** 其$A_{new}$比CSIL高17.96个百分点，$H_{old,new}$高9.67个百分点；与此同时，完整backbone更新使旧类表征更容易漂移。全矩阵结果不能写成“MoPC-HR已经解决新类注册”，只能说明它在当前执行语义下更倾向于学习新类。

**CSIL偏向稳定约束，但新类通道经常没有形成有效竞争力。** 冻结backbone、EWC、KD和mask共同保护旧状态，但正式LEO下旧类均值仍从42.83%降到23.17%，说明结构隔离不能自动抵消target receiver域偏移。

**逐类floor揭示均值掩盖的问题。** 两种方法的$A_{\min,old}$均值都低于4%，说明至少一部分旧发射机在增量后接近失效。Phase2方法不能只看$A_{old}^{post}$或$A_{new}$。

## 4.6matched无LEO新类归因诊断

该诊断保持方法、物理样本ID、support/query划分、K-shot、seed和旧类评测条件一致，只把新类support/query替换为未叠加LEO的同一物理记录。结果必须标为：

DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL

| **方法** | **无LEO** $A_{old}^{post}$ | **无LEO** $A_{new}$ | **无LEO** $H_{old,new}$ | $\Delta A_{old}^{post}$ | $\Delta A_{new}$ | $\Delta H_{old,new}$ | $\Delta F_{old}$ |
|----------|---------------------------|--------------------|------------------------|-------------------------|------------------|----------------------|------------------|
| CSIL     | 23.78%                    | 12.15%             | 1.67%                  | +0.60pp                 | +3.50pp          | +0.49pp              | −0.60pp          |
| MoPC-HR  | 21.45%                    | 52.50%             | 12.98%                 | −0.68pp                 | +25.89pp         | +2.13pp              | +0.92pp          |

移除新类LEO扰动后，MoPC-HR的$A_{new}$平均提高25.89个百分点，说明LEO弱信道显著破坏了新类可分性；但$A_{old}^{post}$下降0.68个百分点、$F_{old}$增加0.92个百分点，$H_{old,new}$只提高2.13个百分点。CSIL的$A_{new}$只提高3.50个百分点，说明其主要瓶颈还包括零步训练和过强稳定约束。

该诊断不能用于CVS卫星场景性能声明、方法晋级或Stage2协议有效性证明。

## 4.7低K条件下的零注册问题

若增量阶段只有$N_{\text{new}} \times K$个样本，batch size为$B$，并且DataLoader使用drop_last=True，则：

$$floor\left( \left( N_{\text{new}} \times K \right)/B \right) = 0$$

当结果为0时，一个完整batch都不会产生，optimizer step也为0。新增分类权重没有学习，query自然继续被预测为旧类。

CSIL还要先执行约60%的官方训练切分，因此其有效样本数会进一步减少。正式LEO的400个CSIL cell中有175个零步cell，平均每场景5.625个optimizer step；400个MoPC-HR cell中有100个零步cell，平均每场景97.5步。两种条件均未启用缩batch、补采样或small-K训练适配。

低新类数并不必然意味着任务更简单。新增类之间可用于形成相对边界的样本更少，新增权重的方向和尺度可能不稳定；旧类logit经过充分训练，新类logit接近初始化，单头竞争会强烈偏向旧类。零步行只能说明官方trainer在该K/新类数组合下没有产生有效更新，不能解释成“训练后证明方法无效”。

# 5.统一比较与下一步

## 5.1五种方法的横向定位

| **方法**     | **任务类型**     | **主要数据与权限**                   | **更新对象**               | **核心特点与风险**                |
|--------------|------------------|--------------------------------------|----------------------------|-----------------------------------|
| ProtoNet CDA | 少样本监督DA基线 | target-old support                   | prototype                  | 最轻量，但不能修正复杂特征形变    |
| MRIOR-SDA    | 监督域适应       | source缓存＋target-old support       | backbone＋DV-KL网络        | 域对齐能力强，但需要source replay |
| DADDA-SDA    | 统计距离域适应   | source缓存＋target-old support       | 完整backbone               | 联合MMD/LMMD，低K时类条件估计不稳 |
| CSIL         | 无exemplar类增量 | target-new support＋旧模型/Fisher    | 新增分类结构；backbone冻结 | 稳定约束强，但可能不学习新类      |
| MoPC-HR      | prototype类增量  | target-new support＋旧prototype/参数 | backbone＋classifier       | 新类可塑性较强，但旧类遗忘明显    |

方法不能只按最高准确率排序。比较时必须同时考虑任务类型、数据权限、更新范围、计算资源和同一row的$A_{old}^{post}$、$A_{new}$、$H_{old,new}$、$F_{old}$及$A_{\min,old}$。

## 5.2当前实验发现

1.  Phase2的困难具有递进性：先出现receiver域偏移，再叠加少样本新类注册和旧类遗忘。

2.  MRIOR-SDA与DADDA-SDA证明target-old support具有域校准价值，但它们使用source replay和完整backbone更新，只能作为宽权限对照；ProtoNet CDA则说明单一prototype不足以修正复杂target embedding形变。

3.  正式LEO全矩阵中，MoPC-HR的$A_{new}$为26.61%、$H_{old,new}$为10.85%，高于CSIL的8.65%和1.18%；但两者$A_{old}^{post}$均只有约22%至23%，旧类稳定性仍未解决。

4.  matched无LEO诊断使MoPC-HR的$A_{new}$提高25.89个百分点，但$H_{old,new}$只提高2.13个百分点且遗忘略增，说明LEO信道失真不是唯一瓶颈。

5.  固定batch与drop_last使部分低K配置产生零个optimizer step；这属于执行失效，不能解释为“训练后方法无效”。

6.  当前工作完成了任务定义、论文机制复现、CVS接口适配、完整结果审计和失败归因，尚未形成可晋级的Phase2主方法。

## 5.3下一步路线

1.  **support-only域校准：** 优先研究共享协方差、ridge/LDA、低秩adapter和归一化距离，在不回读source样本的条件下修正receiver偏移。

2.  **统一非参数分类头：** 研究KNN、加权KNN与qKNN，使新类support直接进入统一记忆库，避免固定batch和多轮增量训练：

$$\widehat{y} = \arg\max_{c}\sum_{i \in N_{k}\left( z_{q} \right)}^{}w_{i}\mathbf{1}\left( y_{i} = c \right)$$

1.  **安全训练与配对评价：** 训练前检查有效batch和optimizer step；正式比较固定receiver、seed、K、类别集合、LEO观测和split，并在统一全类竞争下报告旧类、新类、调和均值、遗忘和逐类floor。

## 5.4证据边界

- WiSig/ManySig是地面代理数据，LEO弱信道是物理启发的仿真压力条件，不代表真实在轨验证。

- MRIOR-SDA与DADDA-SDA使用更宽source-access权限，只能作为机制对照。

- CSIL与MoPC-HR采用官方方法语义和CVS数据接口，其结果不等于p2_min_v1主方法已经满足support-only协议。

- matched无LEO结果仅用于归因，不能用于正式卫星场景声明或方法晋级。

# 参考文献

[1] SNELL J, SWERSKY K, ZEMEL R S. Prototypical Networks for Few-shot Learning[C]//Advances in Neural Information Processing Systems 30. 2017. https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning

[2] YANG L, LI Q, REN X, et al. Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation[J]. IEEE Internet of Things Journal, 2024, 11(13):24024-24034. DOI:10.1109/JIOT.2024.3389491.

[3] FENG J, FANG S, FAN Y. Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment[J]. IEEE Internet of Things Journal, 2025, 12(16):33202-33214. DOI:10.1109/JIOT.2025.3573713.

[4] LIU Y, WANG J, LI J, NIU S, SONG H. Class-Incremental Learning for Wireless Device Identification in IoT[J]. IEEE Internet of Things Journal, 2021, 8(23):17227-17235. DOI:10.1109/JIOT.2021.3078407.

[5] LI D, CHEN Z, SHAO M, et al. Non-Exemplar Class-Incremental Learning via Prototype Correction and Hierarchical Regularization for Specific Emitter Identification[J]. IEEE Transactions on Intelligent Transportation Systems, 2025, 26(8):12632-12646. DOI:10.1109/TITS.2025.3559174.

# 附录A：非类增量对比方法复现实验结果

本附录仅汇总ProtoNet CDA、MRIOR-SDA与DADDA-SDA的375项CVS复现实验。CSIL和MoPC-HR采用论文作者公开的官方代码，其CVS接口实验与结果见第4节。三种方法共享第3.1节的基座、target receiver、K-shot和seed矩阵，但数据权限不同：MRIOR-SDA与DADDA-SDA使用封存source LEO弱信道标签缓存与target-old support，ProtoNet CDA只使用冻结特征与target-old support。因此，附录结果只作机制对照，不构成同权限Phase2主方法排名。

## A.1总体实验结果

| **方法**     | **适应前** $A_{old}$ | **适应后** $A_{old}$ | **平均** $G_{old}$ | **正/负迁移任务** | **平均时延** | **3场景backbone更新** |
|--------------|---------------------|---------------------|-------------------|-------------------|--------------|-----------------------|
| MRIOR-SDA    | 73.60%              | 82.58%              | +8.98pp           | 105/20            | 17.90s       | 600                   |
| DADDA-SDA    | 73.60%              | 78.35%              | +4.75pp           | 99/26             | 14.62s       | 600                   |
| ProtoNet CDA | 73.60%              | 66.85%              | −6.75pp           | 10/115            | 0.046s       | 0                     |

MRIOR-SDA获得最高旧类准确率，平均提升8.98个百分点；DADDA-SDA平均提升4.75个百分点；ProtoNet CDA平均下降6.75个百分点。target-old support确实包含可利用的域校准信息，但固定embedding上的单prototype不足以恢复跨接收机偏移。

## A.2不同K-shot下的结果

| **K** | **直接ADV3B02** | **MRIOR-SDA** | **DADDA-SDA** | **ProtoNet CDA** |
|-------|-----------------|---------------|---------------|------------------|
| 1     | 73.60%          | 77.22%        | 74.94%        | 58.67%           |
| 2     | 73.60%          | 79.51%        | 76.14%        | 64.70%           |
| 5     | 73.60%          | 82.59%        | 78.19%        | 68.98%           |
| 10    | 73.60%          | 85.82%        | 80.31%        | 70.42%           |
| 20    | 73.60%          | 87.74%        | 82.16%        | 71.48%           |

K从1增加到20时，MRIOR-SDA与DADDA-SDA持续受益，说明更多target support提高了梯度估计和类条件对齐的稳定性。ProtoNet CDA也随K增加而改善，但K=20时仍低于直接ADV3B02。误差并非只来自prototype均值方差，目标域embedding还发生了系统性偏移。

## A.3不同target receiver下的结果

| **target receiver** | **直接ADV3B02** | **MRIOR-SDA** | **DADDA-SDA** | **ProtoNet CDA** |
|---------------------|-----------------|---------------|---------------|------------------|
| 20-1                | 64.61%          | 83.43%        | 75.91%        | 60.98%           |
| 3-19                | 60.33%          | 69.06%        | 65.31%        | 48.50%           |
| 7-14                | 90.06%          | 89.93%        | 89.93%        | 83.53%           |
| 7-7                 | 80.22%          | 86.78%        | 82.49%        | 74.81%           |
| 8-8                 | 72.78%          | 83.69%        | 78.10%        | 66.42%           |

- **20-1：** direct基线较低，MRIOR-SDA提升到83.43%，support提供了明确的目标域校准信号。

- **3-19：** 最高结果为69.06%，是本轮最困难的receiver，说明后续方法必须报告逐receiver结果和逐类floor。

- **7-14：** direct已经达到90.06%，继续更新后轻微下降，高基线receiver存在负迁移风险。

- **7-7与8-8：** MRIOR-SDA保持明显正收益，DADDA-SDA获得中等收益，ProtoNet CDA仍低于direct。

## A.4复现实验结果的证据边界

1.  不同receiver的direct基线差距接近30个百分点，跨接收机域偏移是需要单独处理的性能来源。

2.  在允许source replay和完整backbone更新时，MRIOR-SDA与DADDA-SDA多数任务获得正迁移，证明target-old support能够提供域校准信号。

3.  K增加后两种监督式域适应方法持续改善，说明support数量影响梯度估计和类条件分布估计的稳定性。

4.  ProtoNet CDA的结果说明单prototype分类不等于域适应，尤其不能处理系统性的embedding旋转、拉伸或类内多峰。

5.  本附录不能证明support-only Phase2主方法已经解决旧类域适应；MRIOR-SDA与DADDA-SDA使用更宽权限，只能作为外部机制对照。
