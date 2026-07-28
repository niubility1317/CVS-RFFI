# CVS-RFFI Phase2阶段工作详细报告（截至2026年7月24日）

## 从跨接收机域适应到少样本新类注册：任务定义、方法复现、数据协议与实验结论

**汇报对象：**导师

**报告范围：**整合2026年7月16日与7月24日两轮工作

**研究主线：**Stage2-B旧类目标域适应→Stage2-C旧类适应与新类注册

**协议口径：**`p2_min_v1`

**涉及方法：**ProtoNet CDA、MRIOR-SDA、DADDA-SDA、CSIL、MoPC-HR

> **核心结论：**Phase2不是单一的“少样本分类”问题，而是由跨接收机域偏移、少样本参数估计、旧类保持、新类注册和部署权限共同构成的联合任务。MRIOR-SDA与DADDA-SDA证明target-old support包含有效的域校准信息，但它们依赖source replay和多轮backbone更新；CSIL与MoPC-HR能够扩展新类，却分别表现出“过度保护旧类导致新类不注册”和“新类可塑性增强但旧类遗忘”的问题。当前结果建立了清晰的问题边界和方法对照，还不能表述为Phase2主方法已经达到可晋级性能。

## 1.报告目的与近期工作全景

### 1.1为什么把两份周报合并

7月16日的工作集中在Stage2-B：给定目标接收机域中旧发射机的K-shot support，研究模型如何恢复跨接收机后的旧类识别性能。7月24日的工作进入Stage2-C：目标域不仅包含旧发射机，还出现Phase1从未见过的新发射机，模型需要用少量support注册新类，同时避免遗忘旧类。

两轮工作不是相互独立的实验。Stage2-C建立在Stage2-B之上：如果模型连目标接收机上的旧类域偏移都无法处理，新类注册得到的类别中心、分类权重或特征子空间也会受到相同偏移影响。因此，本报告把两轮工作组织成一条连续路线：

1. 识别并量化跨接收机域偏移。
2. 比较轻量prototype分类与强模型更新型域适应。
3. 在统一目标域中加入新发射机support，转入类增量学习。
4. 比较结构隔离型CSIL与prototype校正型MoPC-HR。
5. 通过正式LEO弱信道和matched无LEO诊断，区分信道失真、训练步数不足与灾难性遗忘。
6. 收敛到下一阶段需要解决的联合目标：support-only域校准、轻量新类注册和旧类floor保护。

### 1.2两轮工作的对应关系

|时间|Phase2阶段|核心问题|对比方法|主要输出|
|---|---|---|---|---|
|截至7月16日|Stage2-B|换到新接收机后，旧类准确率如何恢复|ProtoNet CDA、MRIOR-SDA、DADDA-SDA|375个域适应任务，分析K-shot、receiver和计算开销|
|截至7月24日|Stage2-C|如何注册新发射机，同时保留旧发射机|CSIL、MoPC-HR|24个正式LEO配置与18个matched无LEO诊断配置|

## 2.从零理解RFFI与Phase2

### 2.1什么是射频指纹识别

射频指纹识别（Radio Frequency Fingerprint Identification，RFFI）利用发射机硬件在制造误差、器件老化和工作状态上的微小差异识别设备身份。功率放大器非线性、I/Q不平衡、载波频偏、滤波器响应和时钟误差会在无线信号中留下相对稳定的设备特征。

接收端获得的IQ可抽象为：

$$
x=R_d(H_d(T_y(s)))+n
$$

其中，\(T_y\)表示类别\(y\)对应发射机的硬件非理想性，\(H_d\)表示传播信道，\(R_d\)表示接收机\(d\)的链路响应，\(n\)表示噪声。模型真正希望识别的是\(T_y\)，但实际输入同时混入了信道和接收机效应。

### 2.2为什么换接收机会掉点

同一发射机经不同接收机采集时，接收机的频率响应、增益、滤波、采样时钟和噪声会改变观测分布。模型如果把接收机特征误当作发射机特征，就会在训练接收机上表现很好，在未见目标接收机上明显下降。

本项目要求source receiver集合与target receiver集合互斥：

$$
R_s\cap R_t=\varnothing
$$

这里的“域”主要指接收机域，也可以包含信道场景、信噪比、日期和采样链路等采集条件。Phase2最核心的变化是：模型从source receiver训练分布迁移到未见target receiver的LEO弱信道接收分布。

### 2.3域泛化、域适应、少样本和类增量有什么区别

|概念|训练或部署时可见信息|类别集合是否变化|解决的问题|
|---|---|---|---|
|域泛化（DG）|训练时只见source域|通常不变|希望模型直接泛化到未见target域|
|无监督域适应（UDA）|可见无标签target数据|通常不变|利用target分布缩小source-target差异|
|监督域适应（SDA）|可见少量有标签target support|通常不变|用target标签定向校准旧类|
|少样本学习（FSL）|每类只有K个support|可不变，也可增加|用极少样本完成分类或注册|
|类增量学习（CIL）|后续阶段持续出现新类训练数据|持续增加|学习新类并抑制旧类遗忘|
|开放集识别（OSR）|query可能来自未注册类别|注册集合不一定变化|拒绝unknown，而不是强制归入已知类|

本项目的Stage2-B是有标签target-old support条件下的旧类域适应。Stage2-C同时包含目标域旧类适应和少样本新类注册。Phase3才独立研究未注册unknown拒识。把这三个任务混在一起，会导致指标和方法目标失真。

### 2.4旧类、新类和未知类

类别集合定义为：

$$
Y_{\text{old}}\cap Y_{\text{new}}=\varnothing
$$

- **旧类\(Y_{\text{old}}\)：**Phase1已经见过的发射机。即使它在Phase2换了接收机，身份仍然属于旧类。
- **新类\(Y_{\text{new}}\)：**Phase1从未见过，在Stage2-C通过target-new support完成注册的发射机。
- **未知类\(Y_{\text{unknown}}\)：**没有注册support，推理时只能拒识或暂缓判断的发射机。

“新接收机上的旧发射机”不是新类。它的困难来自域偏移；“地面训练从未出现、部署后提供合法support的发射机”才是新类。

### 2.5support、query与K-shot

- **support：**带标签的少量目标域样本，可用于域适应、prototype计算、分类头扩展或模型更新。
- **query：**测试样本，只能在模型和状态冻结后逐样本推理，不能用于训练、调参、选择候选、设阈值或回滚。
- **K-shot：**每个已注册类别有K个互不重复的物理support样本。由同一接收IQ计算FFT、均衡、裁剪或归一化view不会增加K。

同一实验row中，support与query的物理样本ID必须不相交。每个query必须独立面对全部已注册类别，不能利用真实old/new角色、batch类别数量、每类配额或跨query全局重排。

### 2.6什么是灾难性遗忘

类增量模型用新类support更新参数时，梯度会改变特征提取器和分类边界。若训练只关注新类，旧类性能可能快速下降，这一现象称为灾难性遗忘。

类增量学习的核心不是只提高新类准确率，而是平衡：

- **稳定性：**旧类知识尽量不下降。
- **可塑性：**新类能够被真正学习和预测。

CSIL更偏向稳定性，MoPC-HR更偏向可塑性。两者在CVS实验中的失败模式恰好对应这两个极端。

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

### 3.4Stage2-A、Stage2-B与Stage2-C

|阶段|可用target信息|分类空间|核心任务|不能声明|
|---|---|---|---|---|
|Stage2-A|无target TX标签；可有无标签LEO IQ|旧类参考空间|zero-label目标域参考/诊断|旧类few-shot适应、新类注册|
|Stage2-B|\(Y_{\text{old}}\)的K-shot support|全部旧类|旧类目标域适应与校准|新类注册性能|
|Stage2-C|\(Y_{\text{old}}\cup Y_{\text{new}}\)的K-shot support|全部已注册旧类与新类|同时完成旧类适应和新类注册|只报告一侧就称为完整成功|
|Phase3|未注册类作为独立评估|旧类、新类与unknown|unknown拒识/open-world扩展|用unknown结果替代Phase2旧新类结果|

### 3.5为什么论文对比方法会有不同权限

MRIOR、DADDA、CSIL和MoPC-HR的原论文任务、训练生命周期和数据权限并不相同。为了研究机制，这些外部对比方法可按论文或CVS适配流程访问base/source数据、历史统计和多轮训练，但报告必须标明权限。

因此需要区分两种结论：

- **机制比较：**可以比较方法如何对齐域、保护旧类、扩展新类和控制遗忘。
- **同权限排名：**只有数据、生命周期、query边界和资源权限一致时才能成立。

历史MRIOR-SDA和DADDA-SDA读取source数据并联合更新backbone，只能作为“更宽权限下域适应是否可行”的外部对照，不能与当前support-only主方法混成一个正式排行榜。

### 3.6本报告涉及的数据与实验矩阵

|工作包|基座与数据|target设置|实验规模|输出|
|---|---|---|---|---|
|旧类域适应|ADV3B02＋target-old support/query；MRIOR/DADDA另读source|5个target receiver；K={1,2,5,10,20}；5个seed|5×5×5×3=375个方法任务|适应前后old_acc、收益、正/负迁移、时延|
|新类注册|ADV3B02接口＋target-old/target-new support/query|K={5,10,20}；新类数按方法配置变化|24个正式LEO配置|old_acc_after、seen_new_acc、H、forgetting|
|信道归因诊断|保持方法、物理ID、split、K和seed一致，仅替换新类IQ为无LEO版本|18个matched配置|18个诊断配置|Δnew、Δold、ΔH|

## 4.评价指标与输入输出

### 4.1旧类适应指标

**适应前旧类准确率：**

$$
\mathrm{old\_acc\_before}
$$

它表示直接使用Phase1模型在target receiver旧类query上的性能。

**适应后旧类准确率：**

$$
\mathrm{old\_acc\_after}
=(正确预测的旧类query数)/(旧类query总数)
$$

**适应收益：**

$$
\mathrm{gain}
=\mathrm{old\_acc\_after}-\mathrm{old\_acc\_before}
$$

正值表示正迁移，负值表示适应损伤了原有能力。

### 4.2新类注册指标

**已注册新类准确率：**

$$
\mathrm{seen\_new\_acc}
=(正确预测的新类query数)/(新类query总数)
$$

该指标为0意味着新类support没有形成可用的分类身份，或者增量训练根本没有产生有效更新。

### 4.3旧新联合指标

**旧新调和均值：**

$$
H_{\text{old,new}}
=(2\cdot\mathrm{old\_acc\_after}\cdot\mathrm{seen\_new\_acc})
/(\mathrm{old\_acc\_after}+\mathrm{seen\_new\_acc})
$$

调和均值会惩罚“只保旧类、不学新类”和“只学新类、忘掉旧类”。任意一侧接近0，\(H_{\text{old,new}}\)都会接近0。

**遗忘：**

$$
\mathrm{forgetting}
=\mathrm{old\_acc\_before}-\mathrm{old\_acc\_after}
$$

遗忘越大，说明增量学习对旧知识破坏越严重。

### 4.4统一输入、状态更新与输出

|阶段|输入|允许更新的状态|输出|
|---|---|---|---|
|直接基线|Phase1 bundle＋target query|无|旧类预测|
|Stage2-B适应|Phase1 bundle＋target-old support|prototype、adapter或backbone，取决于方法权限|冻结后的旧类预测器|
|Stage2-C注册|冻结/适应后状态＋old/new support|旧类统计、新类prototype、分类头或受控参数|面向全部注册类的统一预测器|
|独立评分|不可变prediction artifact＋query真值|不得回流到predictor|old/new/H/forgetting等指标|

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

Snell等人在2017年提出Prototypical Networks，它是典型的度量型少样本学习方法[1]。对类别\(c\)的K个support样本，先用固定特征提取器\(f_\theta\)得到embedding，再计算prototype：

$$
p_c=(1/K)\cdot\sum_{i:y_i=c}f_\theta(x_i)
$$

query选择距离最近的prototype：

$$
\hat y=\arg\min_c\|f_\theta(x)-p_c\|_2^2
$$

**本轮使用的数据：**只使用固定ADV3B02特征和target-old support，不读取source，不更新backbone。

**更新对象：**每类一个target prototype。

**方法分类：**度量型少样本分类、prototype classifier。

**优点：**注册快、无反向传播、每类状态小、机制上可扩展新类。

**局限：**如果target receiver使embedding整体旋转、拉伸或类内多峰化，单一均值prototype不能修正encoder。“有support”不等于“完成域适应”。

### 5.3MRIOR-SDA：域对齐与target监督

Yang等人提出MRIOR，用于缓解不同接收机造成的RFFI性能下降[2]。原论文属于单源无监督域适应，核心包括域对齐和自适应伪标签。本轮CVS版本利用有标签target-old support，把机制改写为监督式域适应：

$$
\mathcal L_{\text{MRIOR-SDA}}
=\mathcal L_{\text{CE}}^{\text{source}}
+0.5\mathcal L_{\text{CE}}^{\text{target}}
+0.005\mathcal L_{\text{DV-KL}}
$$

**本轮使用的数据：**source有标签数据、target-old support和固定target query。

**更新对象：**完整backbone和分类相关状态；每个任务进行600次更新。

**方法分类：**模型更新型监督域适应；原论文基础是域对齐＋自适应伪标签的UDA。

**为什么可能有效：**它不是只移动prototype，而是改变特征提取网络，使target support回到正确旧类区域。

**权限与部署代价：**需要source replay、多轮反向传播和更多模型状态，不满足当前Phase2主方法的support-only权限。

### 5.4DADDA-SDA：全局与类条件动态分布对齐

Feng等人提出DADDA，通过动态分布对齐解决跨接收机RFFI[3]。方法同时考虑：

1. **MMD全局对齐：**缩小source与target整体分布差异[7]。
2. **LMMD类条件对齐：**对齐同一类别的source和target特征。
3. **动态权重：**平衡全局对齐与局部对齐。
4. **多尺度特征：**从多个层级提取细粒度发射机表征。

本轮监督式改写为：

$$
\mathcal L_{\text{DADDA-SDA}}
=\mathcal L_{\text{CE}}^{\text{source}}
+\mathcal L_{\text{CE}}^{\text{target}}
+\mathcal L_{\text{MMD}}
+\alpha\mathcal L_{\text{LMMD}}
$$

**本轮使用的数据：**source有标签数据、target-old support和固定target query。

**更新对象：**backbone和分类相关状态；每个任务进行600次更新。

**方法分类：**统计距离型监督域适应；原论文属于动态全局＋局部对齐UDA。

**局限：**K很小时，target类条件分布估计噪声大，强行对齐可能破坏source中已经形成的决策边界。

### 5.5三种方法的机制与权限对比

|方法|方法分类|部署输入|更新对象|source访问|计算特点|本轮能否注册新类|
|---|---|---|---|---|---|---|
|ProtoNet CDA|度量型少样本分类|target-old support|prototype|否|0次backbone更新|未评估|
|MRIOR-SDA|模型更新型监督域适应|source＋target-old support|完整backbone|是|600次更新|否，闭集旧类|
|DADDA-SDA|统计距离型监督域适应|source＋target-old support|完整backbone|是|600次更新|否，闭集旧类|

### 5.6总体实验结果

|方法|适应前old_acc|适应后old_acc|平均收益|正/负迁移任务|平均时延|backbone更新|
|---|---:|---:|---:|---:|---:|---:|
|MRIOR-SDA|73.60%|82.58%|+8.98pp|105/20|17.90s|600|
|DADDA-SDA|73.60%|78.35%|+4.75pp|99/26|14.62s|600|
|ProtoNet CDA|73.60%|66.85%|−6.75pp|10/115|0.046s|0|

MRIOR-SDA获得最高旧类准确率，平均提升8.98个百分点；DADDA-SDA平均提升4.75个百分点；ProtoNet CDA平均下降6.75个百分点。结果说明target-old support确实包含可利用的域校准信息，但固定embedding上的单prototype不足以恢复跨接收机偏移。

### 5.7不同K-shot下的结果

|K|直接ADV3B02|MRIOR-SDA|DADDA-SDA|ProtoNet CDA|
|---:|---:|---:|---:|---:|
|1|73.60%|77.22%|74.94%|58.67%|
|2|73.60%|79.51%|76.14%|64.70%|
|5|73.60%|82.59%|78.19%|68.98%|
|10|73.60%|85.82%|80.31%|70.42%|
|20|73.60%|87.74%|82.16%|71.48%|

K从1增加到20时，MRIOR与DADDA持续受益，说明更多target support提高了梯度估计和类条件对齐的稳定性。ProtoNet也随K增加而改善，但K=20仍低于直接ADV3B02，问题不只是prototype均值方差，而是目标域embedding发生了系统性偏移。

### 5.8不同target receiver下的结果

|target receiver|直接ADV3B02|MRIOR-SDA|DADDA-SDA|ProtoNet CDA|
|---|---:|---:|---:|---:|
|20-1|64.61%|83.43%|75.91%|60.98%|
|3-19|60.33%|69.06%|65.31%|48.50%|
|7-14|90.06%|89.93%|89.93%|83.53%|
|7-7|80.22%|86.78%|82.49%|74.81%|
|8-8|72.78%|83.69%|78.10%|66.42%|

- **20-1：**direct基线较低，MRIOR提升到83.43%，support提供了明确的目标域校准信号。
- **3-19：**最高结果只有69.06%，是本轮最困难receiver，说明方法必须报告逐receiver和逐类floor。
- **7-14：**direct已经达到90.06%，继续更新反而轻微下降，证明高基线receiver存在负迁移风险。
- **7-7与8-8：**MRIOR保持明显正收益，DADDA获得中等收益，ProtoNet仍低于direct。

### 5.9Stage2-B结果能够证明什么

1. 跨接收机域偏移真实且强烈，不同receiver的direct基线差距接近30个百分点。
2. target-old support含有有效适应信息；在允许source replay和完整backbone更新时，MRIOR与DADDA多数任务获得正迁移。
3. K增加能够稳定监督式域适应。
4. 单prototype分类不等于域适应，尤其无法处理系统性embedding偏移。

这轮结果不能证明当前support-only Phase2主方法已经解决旧类域适应，因为最高性能方法使用了更宽的数据权限和计算预算。

## 6.工作二：类增量学习与新类注册

### 6.1Stage2-C研究问题与数据

Stage2-C要求在同一个target receiver域中同时处理：

- target-old support：校准已经在Phase1见过的旧类。
- target-new support：注册Phase1未见的新发射机。
- old/new query：模型冻结后，在全部已注册类别中统一竞争。

正式实验的新类support和query均叠加固定LEO弱信道。K取5、10和20，新类数量按CSIL和MoPC-HR的可执行配置取1、3、5、10、20或25。报告必须把old_acc_after、seen_new_acc、H_old_new和forgetting保留在同一row中。

### 6.2CSIL：通道隔离型无exemplar类增量学习

Liu等人在2021年提出CSIL（Channel Separation Enabled Incremental Learning），用于无线设备识别中的类增量学习[4]。论文关注在不保存历史原始样本时，新增设备类别如何避免与旧类fingerprint通道发生冲突。

**方法分类：**无exemplar类增量学习、结构扩展型、通道隔离型方法。

**核心机制：**

1. 保留旧类已经使用的fingerprint子空间。
2. 为新类扩展新的通道或特征块。
3. 屏蔽会破坏旧类结构的交叉连接。
4. 用旧模型输出约束新模型，减少旧类预测漂移。

**主要组件：**

- zero-bias余弦分类器，降低新旧类权重尺度不平衡。
- 通道扩展与mask，尽量把新类学习限制在新增容量中。
- 知识蒸馏（KD），让新模型接近旧模型输出[9]。
- EWC/Fisher约束，对旧任务重要参数施加更强保护[8]。

概念性目标可写为：

$$
\mathcal L_{\text{CSIL}}
=\mathcal L_{\text{CE,new}}
+\lambda_{\text{KD}}\mathcal L_{\text{KD}}
+\lambda_{\text{EWC}}\mathcal L_{\text{EWC}}
$$

**使用的数据与状态：**新类support、旧模型、旧类重要参数统计和CVS适配训练流程；不保存全部旧类原始IQ。

**优点：**旧类保护强，不要求完整历史原始样本。

**局限：**网络容量随增量阶段扩展；固定batch和`drop_last`会导致低K、低新类数时没有有效optimizer step；过度保护旧类时，新类可能完全不被预测。

**CVS适配边界：**本轮是“CSIL机制驱动的CVS适配”，不是原论文在ADS-B数据上的数值复现。ADV3B02特征、LEO新类数据、类别规模和训练日程均与原论文不同。

### 6.3MoPC-HR：prototype校正与分层正则化

Li等人在2025年提出MoPC-HR（Momentum-based Prototype Correction and Hierarchical Regularization），用于specific emitter identification的无exemplar类增量学习[5]。

**方法分类：**无exemplar类增量学习、prototype校正型、正则化型方法。

**prototype的作用：**对类别\(c\)的特征集合\(D_c\)，prototype可以理解为类特征均值：

$$
p_c=(f_\theta(x_1)+…+f_\theta(x_n))/n,\quad x_i\in D_c
$$

MoPC-HR不保存原始IQ，而是保留旧类prototype，用它近似旧类在特征空间中的位置。

**动量prototype校正：**

$$
\tilde p_c^{(t)}
=m p_c^{(t-1)}+(1-m)\Delta p_c^{(t)}
$$

论文默认动量\(m=0.97\)。模型根据新旧类关系平滑校正旧prototype，减少encoder变化造成的中心漂移。

**Gaussian prototype augmentation：**

$$
\tilde z_c=p_c+\epsilon,\quad\epsilon\sim N(0,\sigma^2I)
$$

论文默认\(\sigma=0.05\)。它不回放原始旧样本，而是在prototype附近生成旧类伪特征，为分类器提供旧类约束。

**分层正则化（HR）：**在多个网络层控制新旧模型参数或表示差异，把更多可塑性留给适合学习新类的层，同时保护通用旧类表征。

概念性目标为：

$$
\mathcal L_{\text{MoPC-HR}}
=\mathcal L_{\text{CE}}
+\lambda_{\text{PA}}\mathcal L_{\text{prototype augmentation}}
+\lambda_{\text{HR}}\mathcal L_{\text{hierarchical regularization}}
$$

**使用的数据与状态：**新类support、旧类prototype、旧模型或参数状态，以及CVS接口中的增量训练流程。

**优点：**状态比保存历史IQ紧凑；新类学习能力通常强于只做结构隔离的方法；prototype校正直接处理特征漂移。

**局限：**prototype只近似旧类分布；新类训练较强时仍会遗忘旧类；公开trainer与CVS适配执行路径存在实现差异，必须区分论文公式、官方代码和项目接口结果。

### 6.4CSIL与MoPC-HR机制对比

|比较维度|CSIL|MoPC-HR|
|---|---|---|
|主要方法类别|结构扩展与通道隔离|prototype校正与分层正则|
|历史原始样本|不保存|不保存|
|保留的历史状态|旧模型、fingerprint结构、重要参数统计|旧类prototype、旧模型或参数状态|
|旧类保护|mask、冻结、KD、EWC|prototype增强、prototype校正、HR|
|新类学习|扩展新通道与分类输出|用新类support训练并校正新旧prototype关系|
|主要风险|过度保护旧类，新类不注册|新类学得快，但旧类遗忘|
|部署特征|结构复杂、容量增长|状态较紧凑，但仍需多轮训练|

### 6.5正式LEO弱信道完整结果

|K-shot|方法|新类数|old_acc_after|seen_new_acc|H_old_new|forgetting|
|---:|---|---:|---:|---:|---:|---:|
|5|CSIL|1|83.70%|0.00%|0.00%|0.00%|
|5|CSIL|3|83.70%|0.00%|0.00%|0.00%|
|5|CSIL|20|0.78%|5.17%|0.54%|82.92%|
|5|MoPC-HR|1|87.47%|0.00%|0.00%|0.00%|
|5|MoPC-HR|3|87.47%|0.02%|0.04%|0.00%|
|5|MoPC-HR|5|77.44%|25.65%|37.97%|10.02%|
|5|MoPC-HR|10|63.80%|22.08%|32.30%|23.67%|
|5|MoPC-HR|25|58.73%|14.82%|23.43%|28.73%|
|10|CSIL|1|83.70%|0.00%|0.00%|0.00%|
|10|CSIL|3|83.70%|0.00%|0.00%|0.00%|
|10|CSIL|20|8.33%|5.49%|3.17%|75.37%|
|10|MoPC-HR|1|87.47%|0.00%|0.00%|0.00%|
|10|MoPC-HR|3|66.78%|50.40%|56.78%|20.69%|
|10|MoPC-HR|5|47.38%|50.52%|48.05%|40.09%|
|10|MoPC-HR|10|44.69%|33.15%|37.57%|42.78%|
|10|MoPC-HR|25|44.99%|22.31%|29.32%|42.48%|
|20|CSIL|1|83.70%|0.00%|0.00%|0.00%|
|20|CSIL|3|0.08%|34.04%|0.14%|83.62%|
|20|CSIL|20|59.84%|5.64%|9.96%|23.86%|
|20|MoPC-HR|1|60.76%|96.53%|72.69%|26.71%|
|20|MoPC-HR|3|48.96%|71.47%|57.23%|38.51%|
|20|MoPC-HR|5|33.28%|61.31%|41.09%|54.19%|
|20|MoPC-HR|10|34.07%|41.36%|35.48%|53.40%|
|20|MoPC-HR|25|36.66%|27.86%|30.23%|50.81%|

### 6.6正式结果中的主要现象

**低新类数、低K时新类可能完全不被预测。**CSIL在K=5/10且新类数为1或3时seen_new_acc为0；MoPC-HR在K=5新类数1、K=10新类数1时也为0。后续流程审计确认，固定batch和`drop_last`会使部分配置没有optimizer step。

**MoPC-HR的可塑性强于CSIL。**K=20、新类数1时，MoPC-HR的seen_new_acc达到96.53%，H_old_new为72.69%，是表中旧新联合性能最平衡的切片。但old_acc从87.47%下降到60.76%，遗忘26.71个百分点。

**新类规模扩大后联合性能下降。**MoPC-HR在K=20时，新类数从1增加到25，seen_new_acc从96.53%下降到27.86%，old_acc_after下降到36.66%。更多新类带来更复杂的类间竞争和更强的旧类边界扰动。

**CSIL出现两个极端。**小规模时旧类几乎完全保留但新类不注册；部分大规模或高K切片中新类开始学习，却伴随旧类崩塌。通道隔离不能自动解决CVS中的目标域变化和小样本优化问题。

### 6.7matched无LEO新类归因诊断

该诊断保持方法、物理样本ID、support/query划分、K-shot、seed和旧类评测条件一致，只把新类support/query替换为未叠加LEO的同一物理记录。结果必须标为：

`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`

|K-shot|方法|新类数|无LEO seen_new|Δnew|Δold|ΔH|
|---:|---|---:|---:|---:|---:|---:|
|5|CSIL|3|0.00%|+0.00pp|+0.00pp|+0.00pp|
|5|CSIL|20|8.05%|+2.88pp|−0.78pp|−0.54pp|
|5|MoPC-HR|3|0.00%|−0.02pp|+0.00pp|−0.04pp|
|5|MoPC-HR|5|54.33%|+28.68pp|−33.44pp|+6.75pp|
|5|MoPC-HR|10|63.29%|+41.21pp|−38.09pp|+3.56pp|
|5|MoPC-HR|25|51.02%|+36.20pp|−40.32pp|+3.23pp|
|10|CSIL|3|0.00%|+0.00pp|+0.00pp|+0.00pp|
|10|CSIL|20|4.98%|−0.50pp|+48.69pp|+3.10pp|
|10|MoPC-HR|3|63.04%|+12.64pp|−12.68pp|−0.98pp|
|10|MoPC-HR|5|90.91%|+40.39pp|−19.80pp|−6.12pp|
|10|MoPC-HR|10|75.63%|+42.48pp|−29.12pp|−12.28pp|
|10|MoPC-HR|25|64.38%|+42.07pp|−35.71pp|−13.64pp|
|20|CSIL|3|45.09%|+11.04pp|−0.08pp|−0.14pp|
|20|CSIL|20|5.03%|−0.61pp|+2.00pp|−0.98pp|
|20|MoPC-HR|3|95.64%|+24.18pp|+5.46pp|+10.85pp|
|20|MoPC-HR|5|94.85%|+33.55pp|−10.28pp|−4.69pp|
|20|MoPC-HR|10|81.81%|+40.45pp|−20.38pp|−12.51pp|
|20|MoPC-HR|25|70.26%|+42.40pp|−29.78pp|−18.16pp|

移除新类LEO扰动后，MoPC-HR的新类准确率在多数切片明显提高，说明信道失真破坏了新类特征与旧类prototype之间的几何关系。但新类学得更好不一定提高H：新类梯度变强时，模型更容易向新类偏移，旧类准确率可能进一步下降。

例如K=10、新类数10时，无LEO使seen_new_acc提高42.48个百分点，但old_acc_after下降29.12个百分点，最终H反而下降12.28个百分点。LEO不是唯一原因，稳定性—可塑性冲突仍然存在。

该诊断不能用于CVS卫星场景性能声明、方法晋级或Stage2协议有效性证明。

### 6.8为什么低K、低新类数会出现零注册

若增量阶段只有\(N_{\text{new}}\times K\)个样本，batch size为\(B\)，并且DataLoader使用`drop_last=True`，则：

$$
\mathrm{floor}((N_{\text{new}}\times K)/B)=0
$$

当结果为0时，一个完整batch都不会产生，optimizer step也为0。新增分类权重没有学习，query自然继续被预测为旧类。

低新类数并不必然意味着任务更简单。只有1个新类时，新类样本缺少其他新类作为对比，新增权重的方向和尺度可能不稳定；旧类logit经过充分训练，新类logit接近初始化，单头竞争会强烈偏向旧类。

## 7.五种方法的统一横向比较

|方法|解决方向|主要数据|更新对象|新类能力|旧类保护|资源与状态|本轮主要结论|
|---|---|---|---|---|---|---|---|
|ProtoNet CDA|少样本度量分类|target support|prototype|机制上可扩展|不主动保护或校准backbone|极轻量|速度快，但Stage2-B平均负迁移|
|MRIOR-SDA|跨接收机域适应|source＋target-old support|完整backbone|本轮无|source CE维持旧类|600次更新，source replay|旧类平均提升最高，但权限较宽|
|DADDA-SDA|全局＋类条件域对齐|source＋target-old support|完整backbone|本轮无|source分类和动态对齐|600次更新，source replay|有正收益，但低K分布估计不稳|
|CSIL|无exemplar类增量|新类support＋旧模型统计|通道、mask、分类输出|可以|KD、EWC、冻结与隔离|容量增长，多轮训练|旧类保护强，但低K经常零注册|
|MoPC-HR|prototype类增量|新类support＋旧prototype/模型状态|prototype、分类器及受控参数|较强|prototype增强、校正、HR|状态较紧凑，多轮训练|新类可塑性强，但旧类遗忘明显|

### 7.1这些方法实际上修改了什么

- ProtoNet只修改类别参考点，不修改特征空间。
- MRIOR与DADDA修改特征空间，使target旧类重新靠近正确决策区域。
- CSIL修改网络结构和可训练通道，用容量隔离保护旧类。
- MoPC-HR修改新旧prototype关系和分类状态，用伪特征与分层正则控制遗忘。

方法之间的差异不是“用了不同loss”这么简单。它们读取的数据、保存的历史状态、允许更新的参数和执行生命周期都不同。

### 7.2为什么不能只按最高准确率排序

MRIOR-SDA和DADDA-SDA使用source replay并完整更新backbone；ProtoNet只读target support；CSIL与MoPC-HR执行增量训练并保留不同历史状态。把这些结果放进同一数值排行榜，会把数据权限和计算开销误当成算法优劣。

正确比较至少需要同时报告：

1. 方法任务：闭集旧类域适应还是旧新类增量。
2. 数据权限：是否读取source、旧样本、旧prototype或历史模型。
3. 更新范围：prototype、分类头、adapter还是完整backbone。
4. 资源：训练步数、时延、存储状态和是否需要反向传播。
5. 同一row指标：old_acc、seen_new_acc、H、forgetting和逐类floor。

## 8.当前困难与原因归纳

### 8.1域适应强度与部署权限冲突

MRIOR与DADDA表明完整backbone更新可以显著利用target support，但它们需要source replay和较高计算。Phase2主方法不能在部署时回读source样本，因此下一步需要把域校准能力压缩到support-only、低秩、闭式或轻量adapter中。

### 8.2轻量prototype依赖embedding质量

ProtoNet的失败说明当前target embedding并非简单的“同类中心平移”。receiver响应可能带来旋转、拉伸、类内多峰和类间重叠。后续prototype或KNN路线必须结合归一化、共享协方差、局部度量或轻量域校准。

### 8.3类增量存在稳定性—可塑性冲突

CSIL偏稳定：旧类保留但新类可能不注册。MoPC-HR偏可塑：新类能学到，但旧类下降。Stage2-C的目标不是选择任一极端，而是在统一全类竞争中同时提高old_acc_after与seen_new_acc。

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

- old_acc_after。
- seen_new_acc。
- H_old_new。
- forgetting。
- 最低旧类准确率或逐类floor。
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
2. MRIOR-SDA和DADDA-SDA证明target-old support包含有效域校准信息；MRIOR平均提升8.98个百分点，但两者使用source replay和完整backbone更新，只能作为宽权限外部对照。
3. ProtoNet CDA极轻量，但Stage2-B平均下降6.75个百分点，说明当前target embedding不能简单用单prototype解决。
4. CSIL强保护旧类，但低K、低新类数时经常没有有效训练或完全不输出新类。
5. MoPC-HR的新类学习能力更强，K=20、新类数1时H达到72.69%，但旧类遗忘26.71个百分点，稳定性问题仍然突出。
6. 无LEO诊断说明LEO弱信道确实破坏新类几何结构，但去掉LEO后旧类可能进一步下降，因此信道失真不是唯一矛盾。
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

## 项目内部依据与证据边界

- `E:\type10-7\项目.md`：当前科学场景、Phase1/Phase2数据协议、Stage2-A/B/C权限和claim边界。
- `E:\type10-7\github_publish\CVS-RFFI-repo\docs\weekly_reports\学习进展情况_20260716_详细扩展版.md`：旧类域适应方法、375任务结果和解释。
- `E:\type10-7\github_publish\CVS-RFFI-repo\docs\weekly_reports\学习进展情况_20260724_详细扩展版.md`：CSIL/MoPC-HR机制、正式LEO结果与matched无LEO诊断。
- `E:\type10-7\github_publish\CVS-RFFI-repo\analysis\csil_mopc_hr_cvs_port_audit_20260723.md`：CSIL/MoPC-HR官方源码、论文公式与CVS接口一致性审计。
- `E:\type10-7\automation_reports\CV-SincNet\adv3b02_officialrepo_csil_mopc_20260723_v1\report.md`：固定batch、zero-step和执行证据。
- 原始周报：`C:\Users\lh594\Desktop\周报\学习进展情况+7.16.docx`、`C:\Users\lh594\Desktop\周报\学习进展情况+7.24.docx`。

本文中的WiSig/ManySig只表示地面代理数据；LEO弱信道是物理启发的仿真压力条件，不是真实在轨验证。MRIOR-SDA和DADDA-SDA结果来自更宽source-access权限，不能与`p2_min_v1` support-only主方法同权限排名。无LEO新类结果仅为`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`归因诊断。
