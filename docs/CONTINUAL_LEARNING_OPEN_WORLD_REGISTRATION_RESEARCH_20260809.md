# 持续学习、开放世界学习与CVS-RFFI持续注册：联网调研与项目映射报告

- 日期：2026-08-09
- 适用项目：CVS-RFFI/CV-SincNet
- 协议语义：以本地`E:\type10-7\项目.md`和`protocol_schema=p2_min_v1`为准
- 证据状态：文献综述与研究路线分析，不构成Phase3已实现、实验已完成或性能已达标声明

## 摘要

本报告回答四个问题：持续学习究竟是什么；它与增量学习、在线学习、域适应、开放集识别、开放世界学习和新类发现有什么区别；哪些思想可以被CVS-RFFI吸收；它与项目拟研究的开放世界持续注册是什么关系。

结论很明确：**当前Stage2-C与少样本类增量学习在“少量带标签新类support、累计已注册类统一竞争、兼顾旧类保持”三个方面相邻，但一次Stage2-C适应/注册不等于持续学习。**持续学习的必要证据是时间有序的多个session、状态从上一session延续到下一session，以及每次更新后的旧知识保留、遗忘、新知识吸收和资源增长证据。

项目完整的开放世界持续注册不是一个分类器方法，而是一个跨阶段生命周期：

```text
未注册观测
→ registered/unknown/defer逐样本判决
→ anonymous entity跨时/跨节点关联
→ 外部多源证据确权与registration_authorized
→ 重新采集K个独立物理发射事件作为带标签support
→ 新split_id与相应数据验证
→ Stage2-C旧类适应与授权新类注册
→ 封存新版本并在后续授权事件中继续保留/扩展
```

其中，开放集识别只解决第一段的unknown/defer；新类发现只形成匿名簇；FSCIL从“新类标签已经存在”开始；持续学习只约束跨时间更新与抗遗忘；可信确权和授权属于项目额外且不可省略的运营语义。任何一类文献都不能单独替代完整闭环。

## 1.调研范围与证据方法

本轮采用三条独立联网研究线，并由主研究线统一去重和协议审查：

1. 持续学习理论线：定义、Task-IL/Domain-IL/Class-IL、online CL、稳定性—可塑性、方法谱系和评价矩阵。
2. 开放世界边界线：OSR、OWR、NCD、GCD、C-GCD、OWCL、FSCIL及unknown到注册的生命周期。
3. RFFI/SEI监督线：直接检索射频指纹类增量、少样本增量、时间/域增量、开放集与开放世界论文，并逐项检查`p2_min_v1`兼容性。

检索优先使用论文DOI、IEEE/Elsevier/Nature/PMLR/OpenReview、CVF和NeurIPS官方页面；arXiv只用于尚未正式出版或开放全文补充。关键结论不依赖博客、媒体报道或二手模型摘要。三条研究线返回的记录按DOI优先、题名与第一作者补充去重。本报告不是系统综述或meta-analysis，不对跨论文数值作直接排名；不同数据、session、回放权限和测试信息下的结果不可横向拼接为项目性能预期。

## 2.什么是持续学习

持续学习（Continual Learning，CL）研究的是模型如何从时间有序、非独立同分布的数据、任务、域或类别中持续获取知识，同时保留旧能力，并控制内存、计算和模型增长。[Parisi等，2019](https://doi.org/10.1016/j.neunet.2019.01.012)和[Wang等，2024](https://doi.org/10.1109/TPAMI.2024.3367329)将核心目标概括为稳定性—可塑性权衡、跨任务泛化与资源效率。

“持续”不是修辞。至少应存在：

- 多个按时间顺序发生的学习session；
- session之间延续的模型或注册状态；
- 新session只能读取事先声明的数据和历史状态；
- 新知识学习后，仍在冻结的旧测试切片上评价旧能力；
- 明确的内存、计算、参数和延迟预算。

如果每个实验row都从同一个初始bundle独立重启，不把上一row状态交给下一row，那么它是多条件矩阵，不是持续学习序列。若只完成一次新类注册，即使同时报告旧类与新类指标，也只能证明一次增量事件；持续学习的必要下限是至少两个有序更新session，正式研究更适合使用三个以上session检验顺序敏感性和累积遗忘。

### 2.1稳定性—可塑性不是单指标

稳定性指后续学习后仍保留旧知识；可塑性指模型能够吸收新域、新类或新任务。完全冻结模型可获得“零参数漂移”，但可能学不会新类；只追求新类拟合又会破坏旧类。这解释了为什么CVS不能只报`seen_new_acc`，也不能只报旧类遗忘或floor。[De Lange等，2021](https://doi.org/10.1109/TPAMI.2021.3057446)和[Riemannian Walk](https://openaccess.thecvf.com/content_ECCV_2018/html/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.html)都把遗忘与新任务不可塑性分开评价。

令`A[t,j]`表示完成第`t`个session后，在第`j`个冻结测试切片上的性能。持续学习常用：

```text
最终平均性能：ACC_T = (1/T) * Σ_j A[T,j]
第j个session遗忘：F_j = max_{l∈[j,T-1]} A[l,j] - A[T,j]
后向迁移：BWT关注新学习对旧session的平均影响
前向迁移：FWT关注学习新session前已有知识对它的帮助
```

[GEM](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)提出了accuracy、forward transfer和backward transfer矩阵。对CVS而言，这些指标必须使用同一receiver/scene/K/seed/session链和冻结测试切片，不能把不同run的old峰值与new峰值拼成一条“持续学习提升”。

## 3.相关概念的区别

|概念|变化的对象|标签与测试信息|是否要求跨时间保留|是否处理unknown|与CVS的关系|
|---|---|---|---|---|---|
|迁移学习/域适应|从source迁到一个target|依设置而定|不要求|通常不处理|Phase1→Phase2首先属于迁移/域适应|
|增量学习|实例、域、任务或类别分批到达|依设置而定|常见但术语本身不保证|不一定|是操作性上位描述|
|持续/终身学习CL/LL|非平稳经验随时间到达|必须声明session和可见信息|核心要求|不一定|未来多次授权注册的学习框架|
|在线学习|样本到达后频繁更新，常优化在线损失/遗憾|反馈何时到达是关键|不一定显式评价遗忘|不一定|Phase2 query零更新，因此当前不是online CL|
|Task-IL|任务随session变化|测试时给task ID或任务可区分|是|通常不处理|依赖task/role路由的方法通常不合项目统一竞争|
|Domain-IL|类别语义基本不变，输入域随session变化|测试时通常不给域ID|是|通常不处理|未来多接收机/多时间域序列；单次`R_s→R_t`还不是DIL|
|Class-IL/CIL|类别集合随session增长|测试时无task ID，对全部已见类统一分类|是|通常只评价已纳入的类|Stage2-C具有CIL式决策形态|
|Few-Shot CIL/FSCIL|每个增量session加入少量带标签新类样本|新类标签已存在|是|不负责unknown发现|最接近授权后的Stage2-C注册段|
|开放集识别OSR|测试时可能出现训练外类别|只需输出known/unknown/defer|通常不更新|核心任务|Phase3本地拒识证据；不是注册|
|开放世界识别OWR|OSR后，新类获得标签并可增量加入|标签获取通常作为外部后续条件|是|是|概念上接近Phase3→Stage2-C闭环|
|新类发现NCD|无标签新类集合|通常做聚类，簇号可置换|多为批量|发现簇|可启发anonymous entity，不等于身份|
|广义类别发现GCD|无标签池同时含旧类与新类|部分旧类有标签，常为传导式聚类|多为批量|发现旧/新结构|全批聚类不能直接进入逐query Phase2决策|
|持续GCD/C-GCD|多个session持续遇到旧/新混合无标签数据|常用聚类或伪标签更新|是|是|可启发发现与记忆分离，但不提供可信身份|
|持续测试时适应CTTA|测试流域分布持续变化|以测试数据熵、伪标签或统计量更新|是|通常不解决身份unknown|与Phase2 query零更新直接冲突|
|开放式学习Open-ended Learning|任务、目标和技能本身持续产生|高度自主|是|可能包含|范围远大于本项目注册问题|

[van de Ven等，2022](https://www.nature.com/articles/s42256-022-00568-3)表明Task-IL、Domain-IL和Class-IL因测试信息和输出空间不同，难度与有效方法显著不同。[Tao等，2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Tao_Few-Shot_Class-Incremental_Learning_CVPR_2020_paper.html)把FSCIL明确为“少量带标签新类样本+保留旧类”。[Scheirer等，2013](https://doi.org/10.1109/TPAMI.2012.256)和[Bendale与Boult，2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html)则把unknown拒识与后续类别纳入分开处理。

### 3.1开放集、开放世界与持续学习的三轴坐标

只用“开放”或“持续”两个词容易混淆三个彼此独立的问题：

|轴|低端|高端|CVS要回答的问题|
|---|---|---|---|
|推理集合开放性|query只来自已注册类|可能出现`Y_unknown`|这条观测能否由注册库解释？|
|时间与可塑性|一次性固定模型|多session版本化更新|系统是否在学新身份时保留旧身份？|
|身份语义与治理|匿名簇或伪标签|外部证据确权、有效期、授权|这个类别能否代表并获准注册某个物理身份？|

OSR主要位于第一轴；CIL主要位于第二轴；FSCIL从“第三轴的标签已经给定”开始；GCD可产生匿名类别结构，却没有物理身份确权；项目Phase3→Stage2-C闭环必须同时覆盖三轴。

## 4.持续学习方法谱系及项目兼容性

绿色表示原则可被现有协议容纳；黄色表示需要新设计或状态审计；红色表示当前Phase2主方法不可直接采用。颜色不代表已有实现或性能。

|方法族|典型机制与论文|优点|对CVS的判断|
|---|---|---|---|
|经验回放|保存旧样本，和新数据混合训练；[GEM](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)、[DER](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)|常是强基线，直接约束旧任务损失|🔴Phase2禁止clean/raw/source、样本级feature/logit cache；不能直接迁入|
|生成/伪特征回放|拟合旧类分布并采样旧特征|减少raw exemplar存储|🔴/🟡不因“合成”自动合法；生成状态可逆性、来源和历史target状态需另行协议|
|参数重要度正则|EWC用Fisher估计旧任务关键参数；[Kirkpatrick等，2017](https://doi.org/10.1073/pnas.1611835114)|无需保存全部旧样本|🟡重要度如何生成、封存和跨session携带必须审计；强约束也可能损害新类可塑性|
|功能/表示蒸馏|旧模型约束新模型输出；[Learning without Forgetting](https://doi.org/10.1007/978-3-319-46493-0_37)|保持旧决策函数|🟡若只在合法support上运行旧教师可研究；旧输入/cache来源必须合法|
|原型/非参数分类|最近类均值、多原型、固定表征累计；[iCaRL](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html)、[RanPAC](https://papers.neurips.cc/paper_files/paper/2023/hash/2793dc35e14003dd367684d93d236847-Abstract-Conference.html)|少样本、可扩类、推理可解释|🟢吸收同规则类几何和统一竞争；🔴不能照搬iCaRL exemplar|
|表示—分类器解耦|冻结backbone，只更新类均值/分类器；[CEC](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.html)|降低表示遗忘，适合FSCIL|🟢/🟡与deployment bundle+轻量注册头相容，但要验证目标域适应是否被过度冻结|
|参数隔离/动态网络|task mask、剪枝、扩展列或专家；[HAT](https://proceedings.mlr.press/v80/serra18a.html)、[PackNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.html)|旧参数稳定|🟡常依赖task ID且状态随session增长，不适合默认的无role统一竞争|
|prompt/adapter|冻结大模型，只更新少量可塑参数；[L2P](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html)、[CODA-Prompt](https://openaccess.thecvf.com/content/CVPR2023/html/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.html)|参数高效|🟡可吸收“慢表征+快适配状态”，但视觉预训练和prompt选择结果不能外推到RF|
|元学习/伪增量训练|在base/source数据中模拟未来增量session；[MetaFSCIL](https://openaccess.thecvf.com/content/CVPR2022/html/Chi_MetaFSCIL_A_Meta-Learning_Approach_for_Few-Shot_Class_Incremental_Learning_CVPR_2022_paper.html)|提前学习如何少样本更新|🟢适合作为Phase1 source-only训练原则；不得用target query或Phase2结果反向构造episode|
|开放空间/尾部分布|距离、能量、OpenMax、EVT、每类半径；[OpenMax](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)|提供unknown/defer接口|🟢可做Phase1 proxy unknown研发和Phase3冻结本地证据；不等于确权或注册|
|GCD/C-GCD双分支|发现分支寻找新簇，记忆分支保留旧类别；[MetaGCD](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_MetaGCD_Learning_to_Continually_Learn_in_Generalized_Category_Discovery_ICCV_2023_paper.html)、[DYDM](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_Decouple_Your_Discovery_and_Memory_in_Continual_Generalized_Category_Discovery_CVPR_2026_paper.html)|同时处理未知发现和遗忘|🟡可吸收“发现状态与注册推理状态隔离”；簇号仍不能成为真实身份或直接support|
|CTTA|用无标签测试流更新模型；[Tent](https://openreview.net/forum?id=uXl3bZLkr3c)、[CoTTA](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)、[EATA](https://proceedings.mlr.press/v162/niu22a.html)|适应连续域漂移|🔴query更新模型、阈值、统计或伪标签与`p2_min_v1`直接冲突|

### 4.1为何“source-free”或“无exemplar”仍不等于合规

论文中的source-free通常只表示部署阶段不读取原始source数据；它仍可能保存旧模型、全精度prototype、Fisher、logit、生成器、无标签目标池或可更新memory。项目Phase2还限制样本级source派生状态、query更新、真实角色、批量类别构成、配额和全局重排。因此必须逐项审查：状态来自何处、能否逆推出样本、何时更新、是否由query触发、推理是否依赖task/role ID。

同理，“non-exemplar”只排除了raw样本回放，不等于它满足与checkpoint共同封存的只读int8聚合知识、不可替换sidecar和无query更新要求。

## 5.RFFI/SEI中的持续学习研究现状

直接RFFI论文表明，领域已经从静态closed-set分类转向类增量、few-shot增量、时间漂移和开放集联合建模；但这些论文的权限和生命周期差异很大，不能因标题含`incremental`或`open-set`就视为项目协议合规。

|论文|研究问题与机制|可吸收观点|与当前项目的主要差异|
|---|---|---|---|
|[CSIL，IEEE IoT Journal 2021](https://doi.org/10.1109/JIOT.2021.3078407)|无线设备RFFI类增量，强调不使用历史原始数据的channel separation|RFFI可研究无raw replay的类扩展|仍需审计旧状态形式、support来源和query是否更新|
|[Radio Frequency Fingerprint Collaborative Intelligent Identification Using Incremental Learning，2022](https://doi.org/10.1109/TNSE.2021.3103805)|分布式传感器决策融合和新数据增量微调|多接收节点与增量更新可联合研究|不是unknown确权闭环；数据共享和更新权限不同|
|[A Novel RFFI Method Using Incremental Learning，VTC 2022](https://doi.org/10.1109/VTC2022-Fall57202.2022.10012703)|指纹度量、EVT与新增设备处理|距离/尾部分布适合RF开放空间风险|论文流程不自动满足目标support/query隔离|
|[CISP，IEEE TIFS 2024](https://doi.org/10.1109/TIFS.2023.3343193)|teacher-student、自训练、无标签池召回旧类和原型增广|指出old/new偏置和伪标签风险|🔴旧样本召回或无标签query自训练不能直接进入Phase2|
|[时间不变SEI，Engineering Applications of AI 2024](https://doi.org/10.1016/j.engappai.2024.109324)|15个时间段，adversarial DA+continual learning跟踪指纹漂移|时间漂移是独立于新类增长的domain轴|其选择性标注、数据库迭代和历史访问不同于`p2_min_v1`|
|[FSCIL-SEI，IEEE TIM 2025](https://doi.org/10.1109/TIM.2025.3529056)|base prototype、自监督对比、old/new权重分离、课程学习|RFFI中确有少样本类增量和旧新偏置问题|新类标签已存在；不负责unknown发现、确权或跨接收机协议|
|[MoPC-HR，IEEE TITS 2025](https://doi.org/10.1109/TITS.2025.3559174)|non-exemplar CIL、动量原型校正与层级正则|原型校正和无raw replay值得研究|旧原型的精度、来源和可逆性需按bundle规则重审|
|[AFD-IL，IEEE TCCN 2026](https://doi.org/10.1109/TCCN.2025.3583703)|16次增量更新、重要性感知蒸馏、old/new局部相似分类|长期更新曲线比单次注册更接近CL证据|蒸馏状态、样本权限和域条件并不等同本项目|
|[Meta-RFF，IEEE TCCN 2026](https://doi.org/10.1109/TCCN.2025.3592942)|few-shot open-set incremental learning，meta-task、open loss、自适应阈值|把少样本增量与unknown检测放在同一RF任务中|仍未提供外部真实身份确权；阈值不得由CVS query结果调节|
|[OFSCIL-SEI，IEEE TCCN 2026](https://doi.org/10.1109/TCCN.2025.3565589)|演化原型、prototype calibration、open-set距离、unknown半监督聚类|原型注册与匿名聚类可以分层|聚类只可形成anonymous候选，不能直接变成授权身份|
|[Class-Incremental Open-Set RFFI，JSEE 2026](https://www.jseepub.com/EN/10.23919/JSEE.2025.000180)|类prototype、自注意变换、Gaussian threshold、旧模型冻结和新类损失|RF领域已出现CIL+OSR联合方法|没有证明项目的跨接收机、K-shot物理事件、外部授权和query零更新闭环|
|[Exemplar-Free Class-Incremental RFFI，arXiv 2026](https://arxiv.org/abs/2601.03063)|冻结backbone、adapter、对角GMM伪特征和多教师蒸馏|低存储、冻结表征和adapter可作为候选原则|GMM/伪特征属于可更新历史状态，不能未经新协议直接进入Phase2|

### 5.1该领域尚未解决的组合空白

本轮没有找到一篇直接RFFI论文同时证明以下全部条件：

- 跨接收机与LEO压力下的身份表征；
- Phase2运行时无source/raw/clean或样本级cache；
- K按独立物理发射事件计数；
- query逐样本、全注册类竞争且零更新；
- unknown拒识、跨节点匿名关联、外部可信确权和授权；
- 授权后重新采K-shot并做多session注册；
- 每轮报告旧类保持、新类可塑性、unknown风险和资源增长；
- 真实在轨同步多星验证。

这个缺口说明CVS的完整研究问题具有独立价值，也意味着当前不能把零散部件写成已经闭合的开放世界持续注册系统。

## 6.与CVS Phase1—Phase3的准确映射

|项目阶段|当前科学任务|与持续学习的关系|不能提升的声明|
|---|---|---|---|
|Phase1|地面weak-label/semi-supervised source-domain DG，学习跨接收机、LEO压力鲁棒和open-world-ready表征|为未来CL提供稳定的慢表征和开放空间几何；自身不是部署期CL|不能称few-shot、真实unknown拒识或持续注册|
|Stage2-A|无target TX标签的zero-label target-domain reference/diagnostic|可诊断域漂移；除非另有合法reference更新和多session证据，否则不是DIL/CTTA|不能称旧类few-shot适应、新类注册或OSR闭环|
|Stage2-B|`Y_old`合法K-shot目标域适应与校准|一次target-domain adaptation；未来多接收机/多时间域序列才可能成为Domain-IL|不能称CIL、新类注册或unknown发现|
|Stage2-C|`Y_old∪Y_new`均有合法K-shot support，旧类适应与授权新类注册，全部注册类统一竞争|单次FSCIL/CIL相邻事件；多次版本化授权注册并评测遗忘后才构成continual registration|不能称已完成CL、open-world发现、可信确权或query在线学习|
|Phase3|多节点冻结本地证据上的unknown拒识、anonymous entity关联、可信确权和注册授权|为开放世界持续注册提供发现与准入前置层；若只做拒识/关联而不更新注册状态，也还不是CL|当前是计划研究；不能称已实现或已完成真实在轨多星验证|

这一映射可浓缩为：

> Phase1负责把特征学成“可持续更新的稳定底座”；Phase3负责发现、关联和确权尚未注册的实体；Stage2-C只在获得授权和新support后正式更新注册库；多次Phase3→Stage2-C循环及跨版本保留证据才构成开放世界持续注册。

### 6.1当前代码具有哪些准备，缺什么闭环

仓库已有[open_world_head.py](../code/cvsrffi/open_world_head.py)、[losses.py](../code/cvsrffi/losses.py)中的开放世界几何组件，以及[collaborative_open_set_qknn_eval.py](../code/evaluation/collaborative_open_set_qknn_eval.py)的unknown与协同诊断。这些说明项目已有部件级研发基础。

但当前项目说明仍明确标注：完整unknown拒识主线、anonymous entity关联、可信标签生成、证据冲突消解和Phase3协同推理尚未完成；代码存在也不等于完整artifact或性能证据。本地`E:\type10-7\项目.md`是当前协议真源，[项目介绍.md](项目介绍.md)提供当前能力说明。现有Phase2主线可以表述为“持续学习兼容/可扩展的单次授权注册结构”，不能表述为“已实现持续学习或Phase3开放世界闭环”。

## 7.项目建议术语：受控开放世界时序持续注册

“开放时间持续注册”容易被理解为服务开放时段，也不是稳定的学术任务名。建议使用：

> **受控开放世界时序持续注册**（controlled open-world temporal continual registration）

定义如下：

> 在部署期面对时间上不断到达、身份集合不预先封闭的射频观测，系统先将每条观测保留为冻结的分类、拒识与匿名关联证据；只有在合法外部证据完成可信确权并给出`registration_authorized=true`后，才重新采集K个独立物理发射事件作为带标签support，经新的数据切分与验证交由Stage2-C注册。每次注册形成可追溯版本，并在累计已注册身份的统一竞争下评价旧类保留、新类识别、unknown风险和跨版本稳定性。

令第`t`个注册版本的已注册集合和状态分别为`Y_t`与`M_t`：

```text
Q_t只产生冻结prediction/unknown/defer artifact，不更新M_t
Phase3(Q_t, external evidence)产生registration_authorized或继续defer
获批后重新采集S_t：每类K个独立物理发射事件
M_(t+1) = Update(M_t, immutable Phase1 bundle, legal S_t)
Y_(t+1) = Y_t ∪ Y_new_authorized_t
```

历史`Q_t`不能追溯改写为`S_t`。若support物理ID或support/query划分改变，应形成新的`split_id`并执行相应数据验证；算法、adapter、loss或原型规则变化本身不触发数据重验。

## 8.应该吸收哪些观点

以下是研究原则，不是已冻结候选，也不授权扩展当前实验矩阵。

### 8.1把域漂移和类增长作为两条正交轴

CVS同时面对接收机/信道/时间域变化和授权身份集合增长。未来持续注册不能只称CIL或只称Domain-IL，而应在每个注册session中保持四状态：

|状态|含义|主要回答的问题|
|---|---|---|
|`DA0_REG0`|域适应前、注册前|初始旧类基线|
|`DA1_REG0`|域适应后、注册前|纯域适应效应|
|`DA0_REG1`|域适应前、注册后|纯注册效应|
|`DA1_REG1`|域适应后、注册后|联合适应与注册结果|

每个session的最终`DA1_REG1`状态只有在封存、审计并明确成为下一session起点后，才形成持续状态链。仍应报告`DA1_REG0-DA0_REG0`、`DA1_REG1-DA0_REG1`、`DA0_REG1-DA0_REG0`、`DA1_REG1-DA1_REG0`及difference-in-differences。

### 8.2采用“慢表征+快注册状态”的双时间尺度

Phase1表征/只读bundle承担长期稳定性；target support驱动的轻量adapter、原型或分类头承担可塑性。这与FSCIL中表示—分类器解耦和近期C-GCD中的发现—记忆分离一致，但项目还需保留：

- 所有类使用标签置换对称的同一更新公式；
- query不更新任一时间尺度；
- 历史状态来源、精度、大小和可逆性可审计；
- 没有真实role、task ID、类别配额或全局重排；
- 状态增长有上限，不能为每个session无限增加专家。

### 8.3在Phase1模拟未来session，而不是在target query上学习

MetaFSCIL、CEC和forward-compatible FSCIL的共同启发是：用source-only数据构造伪增量episode，使表征提前学会为未来类别预留几何空间、减少base/new偏置。项目可以在TX互斥的Phase1开发划分中研究这一原则；任何target query结果、真实old/new角色或Phase2矩阵结果都不能回流构造episode。

### 8.4把anonymous discovery与registered predictor彻底隔离

GCD/C-GCD的聚类可用于Phase3提出`anonymous_entity_id`和关联假设；已注册身份预测器只读取冻结的合法注册状态。发现分支的簇号可置换、会漂移、可能合并/拆分，不能直接成为`Y_new`。这一隔离比“一个模型同时自标并注册”更符合安全关键身份系统。

### 8.5把主动学习的“请求标签”改成项目的“请求确权”

[Active GCD](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Active_Generalized_Category_Discovery_CVPR_2024_paper.html)说明纯无标签类别发现存在不可辨识性，需要有限oracle标签。CVS中的oracle不能是query真值接口，而应是运营登记、密码学认证、调度、TDOA/FDOA、轨迹、维护、现场调查等合法外部确权流程。模型可以提出高价值复核对象，不能自行批准注册。

### 8.6将forward compatibility纳入身份表征版本管理

持续注册会产生`M_0,M_1,…,M_T`。若表征空间每次更新都大幅旋转，旧prototype和历史匿名关联会失效。可研究旧/新embedding兼容、显式变换、原型校正或冻结稳定子空间；但任何兼容状态都必须符合bundle和support权限。forward-compatible原则值得吸收，其视觉方法数值不能直接外推。

### 8.7用非补偿门评估完整系统

unknown FAR低不能补偿旧类崩塌；旧类稳定不能补偿新类学不会；新类准确率高不能补偿错误注册；全部defer也不能伪造安全。未来成功条件应至少包含：

- 已注册旧类保持与每类floor；
- 当前授权新类准确率和累计新类保持；
- `H_old_new`及old/new偏置；
- unknown FAR、已注册类误拒和defer；
- anonymous association与ID switch；
- 错误确权/错误注册率；
- 状态、时延、计算和通信增长。

## 9.未来持续注册的最小评价骨架

这是一套研究设计骨架，不是本报告发起的新实验。

### 9.1有序session与状态账本

每个授权注册session至少保存：

```text
session_id / event_time
bundle_id_before / bundle_id_after
registered_set_before / authorized_new_set / registered_set_after
capsule_id / split_id / protocol_schema
support物理事件ID、每类K和场景
允许读取的历史状态及其字节数
精确update入口与参数变化范围
冻结prediction artifact与独立scorer
query_zero_update / no_role_oracle / all_registered_competition审计
```

必要下限是`T≥2`个不可覆盖注册版本；正式研究建议`T≥3`并运行预注册的session顺序或顺序置换，防止仅对某一到达顺序有效。

### 9.2每个session的联合指标

|层次|建议指标|解释边界|
|---|---|---|
|持续学习|`ACC_t`、BWT、FWT、逐session forgetting、历史最差类|必须来自同一状态链与冻结测试切片|
|域适应|四状态中的DA前后旧类变化、跨历史域保持|一次target提升不等于DIL|
|注册|`seen_new_acc`、`old_acc`、`H_old_new`、min-old/min-new、注册效应|只在`REG1`定义new与harmonic|
|开放集|unknown FAR、known误拒、AUROC、FPR95、OSCR、defer率|AUROC不能单独证明运营安全|
|关联|pairwise F1、track purity、ID switch、跨节点一致性|anonymous ID不等于真实身份|
|确权|Top-k候选召回、证据覆盖/独立性、冲突发现、错误授权率|必须独立于分类器置信度|
|系统|注册时延、状态字节数、参数增长、峰值内存、推理时延、能耗/通信|资源必须随session报告|

### 9.3必要基线与权限标签

至少保留：无更新、naive fine-tune、冻结表征+单prototype、受控正则/蒸馏、合法状态下的非exemplar方案，以及允许全部历史数据的joint retrain上界。joint retrain、raw replay、论文原流程等可作为权限不同的外部对比，但必须标注数据与训练许可，不能用于证明Stage2主方法合规或晋级。

### 9.4Phase3不能省略的评价

拒识、关联、确权和注册必须分别评分：

1. 拒识：每条观测独立输出registered/unknown/defer，阈值在query前冻结；registered query被reject/defer按识别错误计。
2. 关联：同一物理发射事件的多节点接收仍只计一个shot；非同步数据只能声明多接收节点代理协同。
3. 确权：记录候选身份、证据来源、证据相关性、冲突、有效期、人工/现场复核和授权决定。
4. 注册：只使用授权后重新采集的独立K-shot support，形成新split并回到Stage2-C。
5. 跨版本保持：注册后继续对所有历史注册身份、当前新身份和未来unknown统一评价。

## 10.当前可以怎样写，不能怎样写

### 10.1严谨表述

- 当前Phase2实现和证据面向目标接收机域的少样本旧类适应与授权新类注册。
- Stage2-C具有少样本类增量学习相邻的任务结构：少量带标签新类support、旧/新联合评价和全部注册类统一竞争。
- Phase1正在提供跨接收机稳定、开放世界就绪的表征与几何基础。
- Phase3规划部署期unknown拒识、anonymous entity关联、多源可信确权和注册授权。
- 多次Phase3→Stage2-C授权循环、版本化状态和跨session遗忘证据完成后，项目才可正式讨论开放世界持续注册。

### 10.2当前不可声明

- “Stage2-C已经实现持续学习/终身学习。”
- “一次new5/new10/new20独立矩阵就是多个持续学习session。”
- “unknown分数、AUROC或聚类簇已经发现真实发射机身份。”
- “历史unknown query在后续被确认后可以改成support。”
- “Tent、CoTTA、EATA或GCD可直接用于当前Phase2 query流。”
- “无exemplar/source-free论文天然满足`p2_min_v1`。”
- “已有open-world头或协同诊断代码，因此Phase3已完成。”
- “WiSig/ManySig是卫星数据，LEO弱信道是实测在轨、多接收机代理是同步多星。”

## 11.综合判断

持续学习为CVS提供了比“域适应+一次新类注册”更长的时间维度：它要求系统解释在第2次、第5次、第10次授权注册后，最早注册的身份是否仍可识别，接收机和信道变化是否污染了身份表征，新类可塑性是否随版本衰减，状态和资源是否保持有界。

开放世界学习又为持续学习增加了集合开放性：系统必须承认注册库不能解释所有观测。CVS比通用OWR再多一层运营约束：射频簇不能自行获得真实身份，必须通过多源证据确权和显式授权，且未知检测证据不能倒灌为训练support。

因此，本项目最有价值的研究定位不是简单宣称“采用持续学习”，而是提出并验证一种**协议受控、身份可审计、域变化与类增长并行、query零污染的开放世界时序持续注册框架**。当前Phase1/Phase2已覆盖该框架的表征底座与授权后注册中段；Phase3发现—关联—确权—授权闭环，以及多session状态链和遗忘证据，仍是后续研究工作。

## 12.核心参考文献

### 12.1持续学习与FSCIL

1. Parisi等，*Continual Lifelong Learning with Neural Networks: A Review*，Neural Networks，2019。[DOI](https://doi.org/10.1016/j.neunet.2019.01.012)
2. De Lange等，*A Continual Learning Survey: Defying Forgetting in Classification Tasks*，IEEE TPAMI，2021。[DOI](https://doi.org/10.1109/TPAMI.2021.3057446)
3. van de Ven等，*Three Types of Incremental Learning*，Nature Machine Intelligence，2022。[DOI](https://doi.org/10.1038/s42256-022-00568-3)
4. Wang等，*A Comprehensive Survey of Continual Learning: Theory, Method and Application*，IEEE TPAMI，2024。[DOI](https://doi.org/10.1109/TPAMI.2024.3367329)
5. Masana等，*Class-Incremental Learning: Survey and Performance Evaluation on Image Classification*，IEEE TPAMI，2023。[DOI](https://doi.org/10.1109/TPAMI.2022.3213473)
6. Kirkpatrick等，*Overcoming Catastrophic Forgetting in Neural Networks*，PNAS，2017。[DOI](https://doi.org/10.1073/pnas.1611835114)
7. Rebuffi等，*iCaRL: Incremental Classifier and Representation Learning*，CVPR，2017。[CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html)
8. Lopez-Paz和Ranzato，*Gradient Episodic Memory for Continual Learning*，NeurIPS，2017。[NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)
9. Buzzega等，*Dark Experience for General Continual Learning*，NeurIPS，2020。[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)
10. Tao等，*Few-Shot Class-Incremental Learning*，CVPR，2020。[CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Tao_Few-Shot_Class-Incremental_Learning_CVPR_2020_paper.html)
11. Zhang等，*Few-Shot Incremental Learning With Continually Evolved Classifiers*，CVPR，2021。[CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Few-Shot_Incremental_Learning_With_Continually_Evolved_Classifiers_CVPR_2021_paper.html)
12. Zhou等，*Forward Compatible Few-Shot Class-Incremental Learning*，CVPR，2022。[CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Forward_Compatible_Few-Shot_Class-Incremental_Learning_CVPR_2022_paper.html)

### 12.2开放集、开放世界与类别发现

13. Scheirer等，*Toward Open Set Recognition*，IEEE TPAMI，2013。[DOI](https://doi.org/10.1109/TPAMI.2012.256)
14. Bendale和Boult，*Towards Open World Recognition*，CVPR，2015。[CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html)
15. Bendale和Boult，*Towards Open Set Deep Networks*，CVPR，2016。[CVF](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)
16. Vaze等，*Generalized Category Discovery*，CVPR，2022。[CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Vaze_Generalized_Category_Discovery_CVPR_2022_paper.html)
17. Ahmad等，*Variable Few Shot Class Incremental and Open World Learning*，CVPR Workshops，2022。[CVF](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/html/Ahmad_Variable_Few_Shot_Class_Incremental_and_Open_World_Learning_CVPRW_2022_paper.html)
18. Zhao和Mac Aodha，*Incremental Generalized Category Discovery*，ICCV，2023。[CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Incremental_Generalized_Category_Discovery_ICCV_2023_paper.html)
19. Wu等，*MetaGCD: Learning to Continually Learn in Generalized Category Discovery*，ICCV，2023。[CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_MetaGCD_Learning_to_Continually_Learn_in_Generalized_Category_Discovery_ICCV_2023_paper.html)
20. Ma等，*Active Generalized Category Discovery*，CVPR，2024。[CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Active_Generalized_Category_Discovery_CVPR_2024_paper.html)
21. Yu等，*Decouple Your Discovery and Memory in Continual Generalized Category Discovery*，CVPR，2026。[CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_Decouple_Your_Discovery_and_Memory_in_Continual_Generalized_Category_Discovery_CVPR_2026_paper.html)

### 12.3持续测试时适应：仅作边界反例

22. Wang等，*Tent: Fully Test-Time Adaptation by Entropy Minimization*，ICLR，2021。[OpenReview](https://openreview.net/forum?id=uXl3bZLkr3c)
23. Wang等，*Continual Test-Time Domain Adaptation*，CVPR，2022。[CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)
24. Niu等，*Efficient Test-Time Model Adaptation without Forgetting*，ICML，2022。[PMLR](https://proceedings.mlr.press/v162/niu22a.html)

### 12.4直接RFFI/SEI文献

25. Liu等，*Class-Incremental Learning for Wireless Device Identification in Internet of Things*，IEEE IoT Journal，2021。[DOI](https://doi.org/10.1109/JIOT.2021.3078407)
26. Liu等，*Radio Frequency Fingerprint Collaborative Intelligent Identification Using Incremental Learning*，IEEE TNSE，2022。[DOI](https://doi.org/10.1109/TNSE.2021.3103805)
27. Liu等，*Specific Emitter Identification Unaffected by Time Through Adversarial Domain Adaptation and Continual Learning*，Engineering Applications of Artificial Intelligence，2024。[DOI](https://doi.org/10.1016/j.engappai.2024.109324)
28. Li等，*FSCIL-SEI: Few-Shot Class-Incremental Learning Approach for Specific Emitter Identification*，IEEE TIM，2025。[DOI](https://doi.org/10.1109/TIM.2025.3529056)
29. Li等，*Non-Exemplar Class-Incremental Learning via Prototype Correction and Hierarchical Regularization for Specific Emitter Identification*，IEEE TITS，2025。[DOI](https://doi.org/10.1109/TITS.2025.3559174)
30. Li等，*Meta-RFF: Meta-Task Adaptive-Based Few-Shot Open-Set Incremental Learning for RF Fingerprint Recognition*，IEEE TCCN，2026。[DOI](https://doi.org/10.1109/TCCN.2025.3592942)
31. Xie等，*Class-Incremental Open-Set Radio-Frequency Fingerprints Identification Based on Prototypes Extraction and Self-Attention Transformation*，JSEE，2026。[Publisher](https://www.jseepub.com/EN/10.23919/JSEE.2025.000180)
32. Jiang等，*Study of Class-Incremental Radio Frequency Fingerprint Recognition Without Storing Exemplars*，arXiv，2026。[arXiv](https://arxiv.org/abs/2601.03063)

## 13.局限

1. 本报告没有复现文献代码或重跑论文实验，方法兼容性以论文原文、官方摘要和项目权限进行初筛；正式吸收前仍需全文、代码和数据流审计。
2. 视觉CL/GCD提供的是任务结构与方法原则，不是RFFI性能外推证据。
3. 直接RFFI持续学习论文数量正在快速增加，但数据集、receiver/time划分、回放权限和open-set定义并不统一。
4. 项目当前实现状态以本地Git、实验报告和完整artifact为准；本报告不会把设计、代码部件或文献可行性写成实验完成。
