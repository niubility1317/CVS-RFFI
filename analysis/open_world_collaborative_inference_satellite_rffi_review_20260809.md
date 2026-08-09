# 开放世界拒识、卫星协同推理与天基射频指纹识别综合调研

日期：2026-08-09
状态：`LITERATURE_REVIEW_COMPLETE / INDEPENDENT_REVIEW_ALLOW / DESIGN_PROPOSAL / NO_NEW_EXPERIMENT / NO_PERFORMANCE_RESULT`
适用项目：CVS-RFFI／CV-SincNet
作者角色：主Agent综合；三个独立子Agent分别负责开放世界拒识、协同推理、天基现实约束检索

## 1.结论先行

本轮并行检索形成三个相互补充的证据集：开放世界识别与OOD核心论文43篇，分布式检测／相关融合／风险控制／跟踪／凭证核心论文46篇，RFFI／NTN／轨道时间／星上计算／干扰定位官方或一手命名来源54项，并额外整理18个现实可获取、可注册或可申请的数据入口。54项中GSSC与IGS按两个独立入口计；标准镜像只作为可访问副本，不重复计为新证据。论文按DOI、题名和作者去重，不把同一作品的arXiv、会议页和PDF重复计数。

综合判断如下。

1. **高性能协同的核心不是多星投票，而是正确识别“哪些证据真的独立、属于同一发射事件且仍处于有效校准域”。**简单平均、多数投票、最高置信节点和普通PoE会在相关节点、重复接收、共享前端、时钟误差和高置信错误下产生系统性过度自信，只能作为基线。
2. **本项目现有CIRF-Track v3方向是正确的，真正的瓶颈已从公式转移到真实资产与完整纵切。**事件账本、双轴相关QP、区间传播、`N_sat=1`恒等、transcript校准、固定滞后MHT和零回写边界已经通过技术G0；尚缺的不是另一个GAT、Set Transformer或DS规则，而是G1/G2执行器、真实same-event数据、授权unknown校准、真实凭证链和完整A/B/C/D×四状态证据。
3. **开放世界拒识必须把三件事分开：已注册类判别、未注册拒识、决策拒答。**MSP、energy、Mahalanobis、kNN、ViM、原型距离等首先只是连续分数；只有合法、互斥、与运行环境匹配的校准资产，才能把分数变成风险声明。Conformal的边际coverage不自动等于unknown FAR、条件可靠性或Byzantine安全。
4. **天基RFFI首先是观测系统问题，其次才是分类器问题。**LEO的高速Doppler、range-rate、传播时延、星钟漂移、前端温度和功率循环、星历误差、可见窗口、接收机差异、下传间歇和DTN队列都会改变观测。没有`emission_event_id`、逐节点`reception_id`、星历／时钟／前端校准和授权unknown真值，多接收机离线拼接不能升级为真实多星协同。
5. **当前WiSig／ManySig仍是地面代理。**它们可以支持跨TX／RX／day／channel的机制筛选和协议结构验证，但不能证明真实在轨Doppler、TDOA／FDOA、星载前端、星间同步、真实unknown FAR或星上资源性能。
6. **推荐的唯一Phase3主线是“CIRF-Track v3-G2确定性认证融合纵切”。**它复用冻结本地bundle，不增加学习式跨节点head：事件验证→origin去重→已注册／unknown双轴相关QP→量化和校准区间→独立风险门→分级通信→隔离MHT→外部凭证→fresh-K交Stage2-C。
7. **Phase1仍有一个值得继续设计但尚未冻结的方向：最坏TX×LEO场景风险＋冻结类条件Gaussian-NLL连续诊断。**它不能复活OE、Q98、双读出、clean–LEO对齐、CB-SFCE或CP-SFCE；还必须解决checkpoint未见验证集谱系和U标签零读取问题，当前不得直接发布实验。

## 2.指定对话`019fe217-6276-7f70-9992-f8d504d9158c`审计

该对话确实完成了实质工作，不是空任务。其主要产出包括：

- 定位并使用最新目标附件，完成CIRF-Track v3设计冻结；
- 实现事件账本、相关QP、有限先验／区间、scheduler、replay authority、MHT与CLI技术路径；
- 经多轮独立P0/P1复核，修复QP极小正定、跨进程replay、ledger截断恢复等缺陷；
- 形成Git实现和N607 CPU-only G0技术run；唯一launch成功退出，回收7个小工件并完成hash／schema闭环。

对应当前可核验入口：

- [CIRF-Track v3多轴设计](/E:/type10-7/code/snapshots/phase3_responsibility_20260807_wt/analysis/phase3_cirf_track_v3_multiaxis_revision_20260809.md)
- [CIRF-Track v3 G0报告](/E:/type10-7/code/snapshots/phase3_responsibility_20260807_wt/analysis/phase3_cirf_track_v3_g0_20260809_v1_report.md)
- 本地正式实验报告：[phase3_cirf_track_v3_g0_20260809_v1](/E:/type10-7/automation_reports/CV-SincNet/phase3_cirf_track_v3_g0_20260809_v1/report.md)

该对话没有完成真实same-event G2、正式unknown FAR、真实Byzantine容错、真实凭证或在轨多星性能。因此它应被定义为“设计＋实现＋技术G0已完成”，不能写成Phase3性能完成。本文在此基础上补充文献、现实采集和G2纵切设计，不从零重新命名同一机制。

## 3.项目约束与评价语义

本文严格继承[项目协议](/E:/type10-7/项目.md)，尤其是以下边界。

- 项目采用“地面训练、星上部署”。Phase1学习开放世界就绪表征；Phase2只用目标接收机合法support和不可变Phase1知识；Phase3才执行unknown拒识、anonymous entity关联和可信确权。
- 每个clean/raw物理IQ在Phase2前只能产生一次固定的`leo_*_weak`接收观测；数学view不增加K。
- 同一物理发射事件被多星接收仍是一个shot。不同事件或无法证明同源的记录不能拼成一个协同事件。
- 本地证据必须先封存，再进入协同；正式预测先封存，独立scorer后接truth。
- registered query的unknown或defer都按身份错误计数；unknown defer不算safe rejection，禁止reject-all规避责任。
- `anonymous_entity_id`不是语义身份；只有外部凭证明确`registration_authorized=true`后，才能重新采集fresh-K进入Stage2-C。历史unknown query永不变成support。
- 协同贡献必须以A/B/C/D因果矩阵分离；Phase2结果必须使用`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`四状态命名。

## 4.检索方法与证据等级

### 4.1检索范围

检索覆盖1979—2026年的以下主题：

1.开放集识别、OOD检测、选择性分类、conformal与风险控制；
2.RF专门open-set／open-world RFFI、跨接收机与时间漂移；
3.分布式检测、概率意见池、未知相关融合、可信多视图、Byzantine鲁棒；
4.异步、丢包、迟到量测、分级通信、端边协同；
5.MHT、JPDA、RFS／PMBM匿名轨迹；
6.3GPP NTN、CCSDS、NASA／JPL、ESA、ITU、FCC的轨道、时间、干扰与运营要求；
7.真实空间RF、GNSS-R、卫星RFFI和多接收数据入口。

### 4.2来源等级

|等级|来源|用途|
|---|---|---|
|T1|正式期刊／会议原文、官方标准、官方任务和数据页|支持机制、数学条件、标准字段和数据可得性|
|T2|作者arXiv、机构仓储、官方accepted manuscript|补充新近方法或受访问限制的正式论文|
|T3|企业产品页、项目展示、新闻稿|仅支持“可申请／商业能力／项目存在”，不支持模型性能|
|排除|二手博客、聚合站、论坛、未核原文的转述|不用于结论|

当前环境未暴露Crossref／Semantic Scholar／arXiv专用学术MCP，因此使用原会议、出版社、DOI、arXiv和标准机构页面逐项核验；同一作品的多个入口只保留一个主记录。

## 5.开放世界拒识方法全景

### 5.1方法族与项目适配

|方法族|代表方法|核心机制|优势|主要风险|本项目裁决|
|---|---|---|---|---|---|
|软最大分数|MSP、max-logit、temperature、ODIN|以最大后验或校准置信度拒识|便宜、零改模|错误高置信；跨RX／LEO阈值漂移|只作基线|
|能量／激活|Energy、ReAct、DICE|能量、激活裁剪或贡献稀疏化|易与分类器连接|固定边界受域移位影响|连续诊断，禁止proxy调阈值|
|距离／密度|Mahalanobis、kNN、GDA／DDU、ViM|类条件距离、非参数邻域、残差密度|可解释、连续排序通常较强|协方差／样本库／绝对阈值迁移|保留连续证据，不直接accept；kNN只作离线基线，deployment bundle不得携带source逐样本cache，只能封存类级聚合状态|
|开空间模型|OpenMax、W-SVM、RPL／ARPL、PROSER|EVT尾部、反向点、placeholder|显式建模open space|尾样本、虚拟未知和新head敏感|论文基线；不复活已拒阈值路线|
|重建／自监督|CROSR、CSI、SSD|重建或分布移位对比表征|无需真实外部unknown|增强可能不物理、训练成本高|Phase1后备，只允许物理合法view|
|外部／虚拟异常|OE、VOS、合成unknown|用异常样本压低置信|通用OOD基准常有效|异常分布权限与代表性决定结果|RealOE已否决，不再换名复活|
|确定性不确定性|SNGP、DUQ、DDU|谱约束、RBF或GDA密度|单模型、可估不确定性|改变head／骨干或依赖密度假设|未来独立候选，不与当前路线叠加|
|最坏组风险|Group-DRO、Fishr|优化最差组或跨域梯度统计|直击fold／floor失败|组稀疏、显式domain、过拟合|只允许TX×scene，不用RX/domain|
|选择性／conformal|SelectiveNet、APS、RCPS／CRC|coverage-risk或集合输出|可报告有限样本风险|exchangeability、条件覆盖与shift失效|只在冻结合法校准上做风险门|
|RF专门OSR|prototype、Gaussian、Siamese、open-world SSL|利用RF几何、噪声或半监督新类|领域接近|多为地面／预印本，协议常允许unknown回流|作为机制参考，不直接移植|

### 5.2关键认识

- “不确定”不等于“未知”。未知样本可能被模型高置信地吸入已知类，而困难的已知样本也可能低置信。
- 强闭集表征是开放集能力的必要基础。项目本地结果同样显示，任何proxy AUROC改善都不能补偿known floor下降。
- OOD排序与可部署阈值是两件事。WRC-NCT的proxy AUROC较高，但固定阈值在LEO和held条件下失败，正是这一差异的实例。
- RF unknown具有多来源：未注册同协议TX、异协议发射、干扰、spoofing、波形边界和接收机异常。一个AUROC不能代表全部unknown mixture。
- Conformal可以解释冻结分数，却不能替代分数本身，也不能在校准和运行分布不匹配时自动给出跨轨道保证。

## 6.协同推理方法全景

|方法族|可吸收机制|不能直接成立的假设|本项目用法|
|---|---|---|---|
|经典分布式检测|量化局部决策、可靠度加权|本地观测条件独立、检测率已知|leader／Chair-Varshney式基线|
|线性／对数意见池|类别证据相加、先验修正|概率语义一致、重复专家不相关|只在origin去重和相关折扣后使用|
|Bayesian committee／CI|共享prior消除、未知相关保守融合|softmax不是likelihood；CI不保证分类风险|形成prior-corrected evidence和相关QP|
|Dempster-Shafer／主观逻辑|显式ignorance和冲突量|高冲突归一化可反直觉；需联合训练|诊断／基线，不作正式核心|
|Trusted multi-view|不确定性加权、缺失视图|依赖真实联合多视图训练|未来G2消融，不用于首版|
|Deep Sets／GAT／Set Transformer|变长节点、置换不变、节点交互|需要大量真实same-event fit数据；无风险保证|数据充分后的二级消融|
|Byzantine鲁棒|明确`f`、节点下界、删组审计|Krum／Bulyan针对训练梯度，不等于后验安全|只吸收威胁模型和压力测试|
|通信高效协同|Tier、量化、选择性请求|请求由当前难度触发会产生选择偏差|冻结codec，枚举／校准完整transcript|
|异步／OOS|显式丢包、迟到和事件时间|线性高斯或独立丢包不一定成立|不插补缺节点；迟到只生成审计revision|
|MHT／JPDA／RFS|birth、death、clutter、miss、多假设|需要真实visibility、clutter和P_D|只跟踪sealed unknown／defer，零回写|
|PKI／attestation|来源、freshness、撤销和状态证明|签名不证明内容诚实|外接真实registry／verifier，不由模型自我授权|

## 7.项目既有实验证据如何改变方法选择

下表只引用同一run内的配对证据；proxy／held均不是正式unknown结果。

|路线|已观察的正信号|不可补偿失败|对新方案的约束|
|---|---|---|---|
|RealOE|真实source OE可以产生能量信号，个别fold有效|known保护仅局部通过，平均RX floor下降；partial held FAR反向恶化|不再用proxy／外部unknown训练本地门|
|GI-EpiOR|episodic相对几何有连续排序信号|known min-RX平均下降，只有2/6通过|不增加冻结表征上的小拒识head|
|WRC-NCT|clean known零下降，proxy AUROC高|WRC附加门仅2/18，paired-clean LEO门0/18，proxy FAR很高|绝对Q98／ratio阈值族关闭|
|CCPC|clean 6/6保护，proxy FAR 6/6下降|LEO四floor仅6/18；F5/F6集中退化|不再做clean↔LEO表征／teacher对齐|
|CB-SFCE|clean 6/6，LEO 16/18，18格整体均值正|两个原子floor失败，proxy仅3/6|保留“直接监督LEO风险”思想，拒绝同实现重调|
|CP-SFCE|clean 6/6；逐foldLEO overall为正|LEO仅14/18，proxy FAR恶化|冲突投影不能替代尾部风险控制|
|PAMR|训练技术合同闭合|postfreeze原生崩溃，无性能结果|不以技术闭环推断开放集能力|

证据入口：

- [WRC-NCT LEO报告](/E:/type10-7/automation_reports/CV-SincNet/phase1_wrc_nct_leo6x3_20260809_v2/report.md)
- [CCPC postfreeze报告](/E:/type10-7/automation_reports/CV-SincNet/phase1_ccpc_leo12_20260809_v4_postfreeze_v1/report.md)
- [CB-SFCE postfreeze报告](/E:/type10-7/automation_reports/CV-SincNet/phase1_cb_sfce_postfreeze_20260809_v1/report.md)
- [CP-SFCE postfreeze报告](/E:/type10-7/automation_reports/CV-SincNet/phase1_cp_sfce_postfreeze_20260809_v1/report.md)
- [RealOE postfreeze报告](/E:/type10-7/automation_reports/CV-SincNet/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1/report.md)
- [GI-EpiOR报告](/E:/type10-7/automation_reports/CV-SincNet/phase1_gi_epior_score6_oneshot_20260809_v1/report.md)
- [PAMR PAIR6技术失败报告](/E:/type10-7/automation_reports/CV-SincNet/phase1_pamr12_20260809_v1_pair6_20260809_v1/report.md)

这些结果共同说明：当前最危险的错误不是“分数不够复杂”，而是跨TX、RX、day和LEO场景的类条件尾部不稳定；任何协同方法若直接乘置信度，只会放大同一失配。

## 8.天基RFFI的现实需求

### 8.1必须进入数据合同的物理量

|域|强制字段|原因|
|---|---|---|
|物理事件|`emission_event_id`、`reception_id`、satellite／receiver／antenna／channel|证明same-event并区分多节点接收|
|信号|raw complex IQ或无损复数变换、sample rate、center frequency、bandwidth、waveform|保留可复核的物理证据|
|轨道|TLE／SPICE／精轨来源、epoch、satellite／ground state、range／range-rate、elevation／azimuth|LEO运动直接决定Doppler、时延和可见性|
|时间|UTC／TAI／GPS／SCLK／ET转换、PPS／GNSS lock、clock offset／drift／uncertainty|TDOA、事件绑定和迟到处理都依赖有界时间误差|
|前端|gain、noise figure、ADC、IQ imbalance、LO／CFO、AGC、filter、antenna pattern、temperature、power cycle|RX失真可能压过TX指纹|
|环境|SNR／C/N0、Doppler、propagation delay、rain／multipath、visibility window|支持context条件校准和最坏场景评估|
|多星几何|capture overlap、TDOA／FDOA／AOA及其uncertainty、independent front-end group|验证事件归属和相关组|
|unknown真值|授权／许可、干扰工单、jammer／spoofer类别、外部定位、有效期|source-held TX不能充当运营unknown|
|资源|CPU／FPGA／NPU、memory、power、thermal、burst latency、compression、downlink bytes、DTN queue|星上方案不能只报告FLOPs|
|复现|raw／meta／calibration／ephemeris hash、event／pass／receiver／satellite隔离split|防止同事件、同burst和同前端泄漏|

### 8.2官方资料支持的现实约束

- [3GPP NTN概览](https://www.3gpp.org/technologies/ntn-overview)和[TR 38.821](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3525)明确指出LEO／NGSO的快速Doppler、时延变化和moving cells。
- [CCSDS出版物](https://ccsds.org/publications/allpubs/)规定轨道／跟踪消息，[CCSDS DTN](https://ccsds.org/publications/allpubs/entry/3222/)针对大时延、间歇链路和store-and-forward。
- [NASA／JPL SCLK](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/sclk.html)说明任务时钟必须通过correlation转换到统一时间系。
- [ITU-R SM.2355-2](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-SM.2355-2-2023-PDF-E.pdf)说明多星TDOA／FDOA依赖同步、几何、钟差和带宽。
- [NASA SmallSat avionics](https://www.nasa.gov/smallsat-institute/sst-soa/small-spacecraft-avionics/)和[ESA On-board Data Processing](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Onboard_Data_Processing/What_is_On-board_Data_Processing)表明星上处理受功耗、存储和下传瓶颈约束。
- [FCC 25-69](https://docs.fcc.gov/public/attachments/FCC-25-69A1.pdf)要求运营者记录并处置疑似有害干扰；这类记录可以成为`interference_event_id`和truth sidecar的合规来源，但不是模型自动产生的标签。

## 9.现实可采集数据路线

目前没有任何公开入口同时提供“星载raw IQ＋前端／时钟校准＋same-emission多星配对＋授权unknown／干扰真值”。可用资产应按缺口组合，而不是假设一个数据集已经完整。

本节是面向方法决策的重点比较，不与附录第20节相加计数；第20节的18行是本轮“实际可获取／可申请入口”的canonical完整清单。

|入口|可得内容|适合用途|关键缺口|
|---|---|---|---|
|[CYGNSS L1 Raw IF](https://podaac.jpl.nasa.gov/dataset/CYGNSS_L1_RAW_IF)|Earthdata注册；空间原始IF counts、时间／几何|真实空间RF、RFI与轨道观测|无预封装same-event和unknown真值|
|[COSMIC-2](https://www.cosmic.ucar.edu/what-we-do/cosmic-2/data)|Level-0仪器数据、姿态／POD／tracking|轨道、时间和多平台流程|不保证复数IQ／发射机指纹标签|
|[MERRByS TDS-1](https://merrbys.co.uk/data-access/downloads)|注册下载DDM／GNSS-R产品|空间GNSS-R环境代理|主发布非raw IQ、单星|
|[HydroGNSS](https://www.hydrognss.org/mission/mission-information/)|双星任务、portal注册、短时未处理采样能力|未来双星原始采集申请|是否开放raw和same-event需任务确认|
|EUMETSAT／NOAA商业RO|RO L0／RDR可申请|真实LEO receiver／clock／orbit|许可和原始字段受合同限制|
|[Spire RF interference](https://spire.com/space-reconnaissance/interference-detection/)|商业历史／实时干扰、谱图／CSV|真实干扰事件和外部定位|raw IQ、真值和校准受合同限制|
|[ESA NAVISP EL2-117](https://navisp.esa.int/project/details/159/show)|多星干扰采集／定位项目资料|受控SSG／MSG场景合作|原始数据未公开|
|[OPS-SAT Space Lab](https://opssat.esa.int/)|开放实验申请、PRETTY SDR＋GNSS|自定义在轨采集和星上推理|单星；需审批和下传预算|
|[SatNOGS](https://network.satnogs.org/about/)|公开多地面站观测API|多接收代理、事件候选|默认无IQ且同步／校准不统一|
|[Iridium物理层数据](https://www.sciencedirect.com/science/article/pii/S2352340923000239)|Iridium raw IQ及包／卫星元数据|卫星RFFI、跨位置／时间|地面接收，不是多星接收同一地面源|
|[OrbID](https://www.ndss-symposium.org/ndss-paper/auto-draft-623/)|约899万Orbcomm包、多SDR／地点|卫星发射机RFF与spoofing|公开可得性和same-event多接收需核；卫星间识别AUC有限|
|[ORACLE](https://www.genesys-lab.org/oracle)|16台USRP的raw／demodulated IQ|地面硬件指纹基准|无轨道、多星、unknown运营真值|
|[WiSig](https://cores.ee.ucla.edu/downloads/datasets/wisig/)|174 TX、41 RX、跨月约千万包|跨RX／time／channel代理|不是真实卫星或same-event多星|

推荐采集顺序：先用Iridium／OrbID／CYGNSS验证空间信号与星历绑定；再以SatNOGS或自建同步地面阵列验证same-event多接收协议；最后申请OPS-SAT／HydroGNSS／Spire类真实星载采集。任何阶段都不得把较早代理结果改名为在轨协同。

## 10.推荐方案：CIRF-Track v3-G2确定性认证融合纵切

### 10.1为什么不是再做一个学习式融合器

Set Transformer、GAT、trusted multi-view和端到端evidential fusion在充足、同分布、真实联合多视图数据上可能具有更高上限，但当前项目没有足够的真实same-event多星fit数据。此时学习式融合器最容易记住节点ID、receiver、轨道场景或proxy角色，而且无法天然给出复制不增益、`N_sat=1`恒等、unknown FAR、Byzantine边界和迟到零回写保证。

因此首个高性能可行版本应使用冻结本地模型和确定性融合核，把学习容量留在经过Phase1／Phase2验证的本地表示，而不是在缺数据的跨节点层继续增加参数。

### 10.2六层架构

```mermaid
flowchart LR
  A["节点本地冻结bundle"] --> B["事件与来源账本"]
  B --> C["已注册轴/unknown轴证据"]
  C --> D["相关QP与区间传播"]
  D --> E["三态风险门与Tier调度"]
  E --> F["封存event decision"]
  F --> G["隔离anonymous MHT"]
  G --> H["外部凭证与fresh-K"]
  H --> I["Stage2-C"]
```

1. **节点层**：每个节点只读不可变bundle，输出全类`prior-corrected log-evidence`、独立unknown轴、质量和量化区间、context、event／origin hash、签名和attestation引用。
2. **事件层**：用时间、频率、波束、可见性、轨道传播边界、waveform digest和TDOA／FDOA不确定度验证same-event；按`reception_id`、artifact hash和`evidence_origin_id`去重。
3. **融合层**：registered轴和unknown轴分别使用`K_R`、`K_U`；相关核来自query前拓扑registry和合法fit split，PSD／对称／单位对角／范围失败即拒绝。
4. **风险层**：把校准有限样本误差、量化、top-L遗漏和有限运行先验传播为逐类区间；每个`availability×context`cell只使用已封存并hash的双轴核，样本不足时采用预注册topology fallback或defer，不在formal阶段传播或选择连续的“核不确定性”。registered必须形成conformal singleton并通过registered-error门；unknown必须有authorized unknown校准，并同时通过FAR和safe-rejection风险门，否则defer。
5. **通信层**：Tier-0只传完整性和摘要；Tier-1传int16逐类中心／半径与遗漏质量；只有所有合法区间下决策不变才结束，否则请求Tier-2或deadline后defer。
6. **生命周期层**：event在deadline封存；迟到证据只生成审计revision。MHT只消费sealed unknown／defer事件，不反馈local predictor、fusion、shot、threshold或凭证。

### 10.3本地证据

每个节点先以冻结的正斜率unknown converter和正温度registered calibrator形成联合simplex：

```text
log p_tilde_m(U)=log u_tilde_m
log p_tilde_m(k)=log(1-u_tilde_m)+log q_tilde_m(k)
e_m(k)=log p_tilde_m(k)-log pi_ref,m(k)
d_R,m(k)=e_m(k)-e_m(r)
d_U,m=e_m(U)-e_m(r)
```

其中`r`是canonical class registry固定的参考registered类；类别置换必须同时置换registry和参考映射。`p_tilde`必须在float64下finite且严格归一，并通过节点roster中预封存的base bundle、class registry、unknown converter、registered calibrator和reference prior验证。receiver-specific state允许不同，但其hash必须命中query前roster；不同概率语义或不同class order禁止共同融合。`d_U`仍不是unknown后验，除非其authorized unknown校准资产和适用unknown mixture已经冻结。

本地continuous score可以由energy、class-relative distance、density或其冻结组合产生，但不得根据formal proxy／unknown选择权重。当前Phase1表征只允许把这些量作为底层输入；不得把source proxy阈值冒充运行unknown门。

### 10.4相关性感知融合

对每个实际transcript先得到通过完整性和deadline门的活跃origin集合`A`。若`|A|=1`，直接进入`N_sat=1`非协同恒等分支；若`|A|≥2`，registered轴和unknown轴分别使用该`availability A×context c`预先封存的相关核`K_a^{A,c}`和fit误差尺度`D_a^{A,c}=diag(sigma^a_m)`，形成`Sigma_a=D_a K_a D_a`。完全同源origin先去重；独立component之间允许累积证据，component内复制不能增加有效证据量。`sigma`、核、origin上限和component上限都来自query前fit资产；节点自报质量只能降低自身上限，不能借当前置信度提高权重。

```text
F_A={beta_A>=0:1^T beta_A=1,beta_m<=c_m,
     sum_{m in component_j intersect A} beta_m<=c_j}
v_A,m=(sigma_m)^(-2)/sum_{i in A}(sigma_i)^(-2)
beta_0,A=Proj_F_A(v_A)
beta*_a,A=lexicographic_argmin_F_A(beta^T Sigma_a beta,
                                   ||beta-beta_0,A||_2^2)
nu_a=min(|A|,1/(beta*_a,A^T K_a beta*_a,A))
L_R(k)=nu_R*sum_{m in A}beta*_R,m*d_R,m(k)
L_U=nu_U*sum_{m in A}beta*_U,m*d_U,m
```

该表达的目标是：独立等质量证据近似按log-evidence累积，完全相关副本的`nu`接近1，异质量时偏向fit误差较小的origin。二级严格凸目标只负责在主最优集合中给出唯一解，不按hash或当前类别选择权重。每个活跃子集必须重建`F_A／v_A／beta_0,A`并使用该cell的核重新求解，禁止从全节点权重或核删行后归一化。

运行先验不从query、预测直方图或track估计。运营方在query前给出有限、严格正且归一的先验集`Pi_b={pi^(1),...,pi^(J)}`；无可信业务先验时只使用预注册uniform prior。对每个`pi∈Pi_b`分别计算：

```text
S(r;pi)=log pi_r
S(k;pi)=log pi_k+L_R(k)
S(U;pi)=log pi_U+L_U
```

量化、top-L遗漏和`Pi_b`按有限集合逐类取max-envelope。核本身必须是当前cell的单一冻结版本及hash；不足时只允许预注册`K_top` fallback或defer。如果未来另立版本保留有限核集合，则必须对每个核重新求解QP、再对全部`K×Pi_b`结果取包络，不能只给固定权重加一个核误差项。该算子仍是相关性策略，不是严格Bayesian posterior。

### 10.5三态决定

```text
registered(c*)：
  Gamma={c*}
  且所有合法transcript、区间和pi in Pi_b下c*仍胜出
  且formal registered-error一侧精确上界通过

unknown：
  authorized unknown calibration存在
  且包含U的conformal集合严格为Gamma={U}
  且所有合法transcript、区间和pi in Pi_b下U仍胜出
  且至少2个独立component支持
  且formal unknown FAR与R_unknown_safe一侧精确上界均通过

defer：其他全部情况
```

fit、interval-calibration、conformal-calibration和formal-test以`event_opportunity_block`四方互斥；unknown还要求TX／anonymous entity身份级互斥。每个正式risk先在block内取event loss最大值，再用预注册置信度的一侧精确Clopper-Pearson上界审计。proxy unknown不得进入证书；独立block样本量不足时只能defer或扩大未来采集。registered的unknown／defer均按身份错误计数；unknown defer计入`R_unknown_safe`失败，不算safe rejection。

### 10.6故障与攻击边界

- 只有1个有效component时降级为single-node并明确标记，不宣称协同。
- 2个component冲突时defer；不得宣称容忍任意坏组。
- 普通完整性模式的leave-one-component-out只作敏感性诊断，不构成Byzantine声明。正式authenticated Byzantine `f=1`模式需要至少3个已认证且可证明独立的component，约束每个component总权重`<=0.5`，并要求删除任一component后区间winner、`Gamma` singleton、三态和每一项风险门全部不变；否则只能冲突检测后defer。该条件性声明不覆盖两组串谋、gateway／registry／root key失陷或物理spoofing。
- 签名、PKI和attestation证明来源与状态，不证明传感内容诚实。
- 缺cell、非有限、不可行QP、未知transcript、失效校准、时钟／星历超界或codec区间不闭合均defer。

### 10.7通信与复杂度

|项|首版设计预算|
|---|---|
|双轴QP|`O(2N_sat^3)`，`N_sat<=5`|
|逐类区间|`O(J*N_sat*C)`|
|删组审计|`O(G*N_sat^3+G*N_sat*C)`|
|MHT|`O(H_max*B)`，当前`H_max=32`|
|内存|`O(N_sat^2+N_sat*C+H_max)`|
|Tier-1|约`242+4(C+1)`bytes／node／event|
|Tier-2|约`230+12(C+1)`bytes／node／event|
|local4＋unknown、5节点全Tier-2|约1.45KiB／event，不含证书链|

这些只是设计预算，必须由真实序列化、签名、链路、缓存和DTN trace实测后才能成为工程结果。

## 11.相对现有CARE／CIRF的真正新增价值

本文不建议把现有v3换名为另一套复杂模型。真正需要完成的是以下纵切：

1.把G0纯函数连接成完整G1／G2执行器；
2.把本地continuous score与合法calibration资产、class registry、checkpoint、receiver-specific state绑定；
3.为`availability×context`稀疏cell冻结topology fallback、defer和样本量门；
4.实现真实Tier codec、逐类量化区间、top-L遗漏上界和全transcript replay；
5.接入真实PKI／撤销／RATS attestation／freshness，而不是合成credential字段；
6.用真实visibility拟合MHT的`P_D`、clutter、birth／death，并报告online-as-of和lag-final两套结果；
7.完成A/B/C/D×四状态×`N_sat={1,...,5}`完整矩阵。

这套纵切比增加一个学习式fusion head更可能带来可复现收益，因为它允许独立节点真正累积证据，同时消除重复、相关、高置信坏节点和通信选择偏差。

## 12.Phase1后续研究：只保留一个尚未冻结的候选

文献和本地结果共同支持一个窄方向：**P1-GD-ProtoNLL**，即最坏TX×LEO场景风险重加权＋冻结类条件Gaussian-NLL连续诊断。

它必须满足以下修订后边界：

- C完全保持GeoSat-C；G只在source-known L标签上计算辅助风险，U真实TX标签零读取；
- scene严格round-robin，q使用旧值反传、detach EMA更新、全12格softmax供下一批；不按active组重新归一；
- 名称应为lagged-EMA entropy-regularized group reweighting，不宣称严格min-max GroupDRO；
- 后冻结Gaussian只用L标签，float64 log-NLL，冻结ddof、variance floor、shrinkage、每类最小样本数和归一化；
- V必须继承warm-start checkpoint从未见的原source_val physical谱系，不能重新切一个“20% V”后自称unseen；
- proxy零fit、零校准、零选参，只作每fold连续guardrail；
- 任一clean或18格LEO非补偿门失败即永久拒绝，不调整lambda、温度、EMA或采样。

该候选当前仍是`DESIGN_REVISION_REQUIRED`，不是已冻结实验，也不是本文主线。Phase3协同可以先复用现有最佳合法本地bundle推进G1／G2，避免把Phase1和Phase3同时变化而失去归因。

## 13.实验与数据路线

### 13.1G0：技术性质

- `N_sat=1`validate后canonical证据与本地决定逐字节恒等；
- 节点排列、类别置换、重复reception、重复origin不改变结果；
- QP的PSD、singular、极小正定、量化边界、top-L遗漏和finite门；
- scheduler全部合法prefix、missingness、arrival order、delay bucket、重传和Tier路径；
- replay、撤销、counter、ledger截断、跨进程authority；
- MHT高clutter、crossing tracks、late OOS、capacity、visible／non-visible opportunity和零回写；
- CPU-only资源、peak memory、primitive operation上界和不可覆盖artifact。

G0不读性能，不形成unknown或协同增益结论。

### 13.2G1：代理多接收节点

预测前使用truth-blind元数据构造proxy event／group并封存；独立scorer之后连接truth。比较：

1.leader single-node；
2.等权平均；
3.质量加权；
4.普通PoE；
5.CARE-PoE；
6.CIRF-Track v3；
7.G2纵切实现的同codec／同风险外壳版本。

所有方法使用相同local artifact、节点子集、deadline、字节预算和prior语义。G1只能声称`PROXY_MULTI_RECEIVER`，不能声称same-event或真实unknown。

### 13.3G2：真实same-event矩阵

数据必须先通过第8节验收表。固定因素至少包括：

- `N_sat={1,2,3,4,5}`和预注册节点子集；
- 独立component数、复制节点、共享前端、丢包、迟到和deadline；
- clock offset／drift、ephemeris uncertainty、event collision和false-binding；
- day／pass／elevation／SNR／Doppler／receiver／temperature；
- unknown来源族：未注册同协议TX、异协议、干扰／spoofing、近边界hard unknown；
- `f=0`和满足节点下界时的`f=1`；
- Tier-0／1／2、实际bytes／latency／energy／DTN queue；
- A/B/C/D与`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`全交叉。

主要效果：`C-A`是协同贡献，`B-A`是本地表征贡献，`D-B-C+A`是交互。新类准确率和`H_old_new`只在REG1定义。

### 13.4非补偿门

1.技术与协议：event、origin、checkpoint、calibrator、prior、class order、credential、codec和artifact全部闭合；
2.registered：所有reject／defer计错，报告overall、min-class、min-RX、min-satellite、min-pass、min-scenario；
3.unknown：正式operational mixture上FAR≤5%，safe rejection≥95%；各unknown来源族分别报告；
4.协同：每个`N_sat`和预注册子集相对single-node的收益与退化都完整报告，平均值不得补偿最坏子集；
5.关联：false merge、fragmentation、IDF1、online-as-of和lag-final分别报告；
6.资源：latency、bytes、energy、memory、deadline miss和DTN queue不能以registered／unknown／track失败补偿；
7.任何门失败即拒绝该冻结版本，禁止用formal结果重调threshold、K、kernel、scheduler、Tier或unknown mixture。

## 14.明确拒绝或暂缓的路线

|路线|原因|
|---|---|
|多数投票／简单平均／最高置信节点|不处理相关性、重复和高置信坏节点；仅作基线|
|普通PoE|共享prior和相关证据会被重复计数，易过度自信|
|Dempster归一化作正式核心|高冲突时可能反直觉，且evidence语义需联合训练|
|直接用Krum／Bulyan／trimmed mean融合后验|这些方法的主要保证面向分布式训练梯度，不自动迁移到事件级概率|
|端到端Set Transformer／GAT|缺少足够真实same-event fit数据，无法闭合风险和复制不变性|
|在线test-time adaptation|query零更新边界不允许，且会污染风险校准|
|proxy unknown或held TX调门|不能代表运营unknown mixture，本地实验已显示跨TX方向反转|
|track后验回写event|会把跨事件关联变成预测器适配并改写已经封存的shot|
|签名等同诚实|签名只证明来源，不证明节点传感内容正确|
|把技术G0写成在轨性能|缺真实same-event、unknown、链路和凭证资产|

## 15.实施优先级

|优先级|工作|是否需要GPU|完成标志|
|---:|---|---|---|
|P0|冻结G2输入schema、事件／时钟／轨道／前端验收表|否|schema、negative tests、hash绑定|
|P0|完成G1／G2 deterministic fusion executor与真实Tier codec|否|G0性质、bytes／latency实测、不可覆盖artifact|
|P0|实现authorized unknown四层隔离和独立scorer|否|fit／calibration／formal／truth sidecar互斥|
|P1|接入真实PKI／revocation／RATS／freshness|否|外部verifier闭环；节点内容不被误称可信|
|P1|完成G1代理矩阵|CPU优先|同artifact同预算的完整基线与消融|
|P1|申请／采集真实spaceborne IQ和same-event数据|任务资源|通过第8节acceptance spec|
|P2|完成G2全矩阵|CPU＋本地模型推理GPU|所有非补偿门和分层结果|
|P3|在真实G2数据充分后消融学习式Set／GAT融合|可选GPU|只作为附加消融，不替代确定性风险核|

## 16.声明边界

本文可以支持以下结论：

- 现有CIRF-Track v3技术方向与主流分布式检测、相关融合、风险控制、异步跟踪和可信凭证文献一致；
- 当前最有效的工程路径是完成G2纵切，而不是继续增加无数据支撑的跨节点学习容量；
- 项目已有Phase1结果排除了多类看似合理但跨TX／RX／LEO不稳定的路线；
- 已找到多个可获取或可申请的空间RF／轨道／干扰数据入口，但没有单一公开数据集满足全部正式G2需求。

本文不能支持以下结论：

- CIRF、CARE或本文方案已经提高真实unknown性能；
- 当前项目已经完成真实在轨多星协同；
- 当前系统已经达到Byzantine、unknown FAR、safe rejection或轨迹性能目标；
- WiSig／ManySig、LEO synthetic transform或非同步多接收代理等价于真实卫星数据；
- 技术G0、bundle、schema或签名测试可以替代完整G2性能。

## 17.开放世界与OOD核心文献索引

下表按原始论文去重。这里的“可借鉴”只表示机制相关，不表示可以绕过本项目的数据权限、校准条件或既有否定实验。

|#|论文／入口|机制与本项目边界|
|---:|---|---|
|1|Bendale、Boult，[Towards Open Set Deep Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)，CVPR 2016|OpenMax＋EVT尾部；有开空间意识，但尾部拟合和阈值必须有合法校准。|
|2|Hendrycks、Gimpel，[A Baseline for Detecting Misclassified and OOD Examples](https://arxiv.org/abs/1610.02136)，ICLR 2017|MSP零改模基线；不能区分错分known与semantic unknown。|
|3|Guo等，[On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)，ICML 2017|温度校准改善概率可靠性，但不自动跨RX、LEO或unknown成立。|
|4|Geifman、El-Yaniv，[Selective Classification for Deep Neural Networks](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)，NeurIPS 2017|coverage-risk拒答；本项目registered defer仍计错。|
|5|Liang等，[ODIN](https://openreview.net/forum?id=H1VGkIxRZ)，ICLR 2018|温度＋输入扰动；后处理超参数不能由query或proxy反选。|
|6|Lee等，[A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks](https://proceedings.neurips.cc/paper_files/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)，NeurIPS 2018|类条件Mahalanobis；连续几何有用，协方差与层融合易过拟合。|
|7|Yoshihashi等，[Classification-Reconstruction Learning for Open-Set Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoshihashi_Classification-Reconstruction_Learning_for_Open-Set_Recognition_CVPR_2019_paper.html)，CVPR 2019|分类＋重建；增加decoder与训练目标，不是当前最小路线。|
|8|Hendrycks等，[Outlier Exposure](https://iclr.cc/virtual/2019/poster/772)，ICLR 2019|外部异常训练；本项目缺合法运营unknown且RealOE已显示proxy／held反转。|
|9|Geifman、El-Yaniv，[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)，ICML 2019|端到端选择头；新增head与coverage目标，不能换名复活已拒小拒识头。|
|10|Liu等，[Deep Gamblers](https://proceedings.neurips.cc/paper_files/paper/2019/hash/0c4b1eeb45c90b52bfb9d07943d855ab-Abstract.html)，NeurIPS 2019|显式abstain输出；改变输出空间且不能以拒答逃避registered责任。|
|11|Chen等，[Learning Open Set Network with Discriminative Reciprocal Points](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480511.pdf)，ECCV 2020|反向点开空间建模；引入新原型结构。|
|12|Sun等，[Conditional Gaussian Distribution Learning for Open Set Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Conditional_Gaussian_Distribution_Learning_for_Open_Set_Recognition_CVPR_2020_paper.html)，CVPR 2020|条件高斯生成；可借鉴log-NLL，不能直接引入重型生成路径。|
|13|Liu等，[Energy-based Out-of-distribution Detection](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)，NeurIPS 2020|energy连续分数；带OE训练仍依赖外部异常。|
|14|Tack等，[CSI](https://proceedings.neurips.cc/paper/2020/hash/8965f76632d7672e7d3cf29c87ecaa0c-Abstract.html)，NeurIPS 2020|对比自监督分布移位；增强必须物理合法且不得构成clean↔LEO配对。|
|15|Liu等，[Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://papers.nips.cc/paper_files/paper/2020/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html)，NeurIPS 2020|SNGP；距离感知强但会改变骨干约束和head。|
|16|van Amersfoort等，[Uncertainty Estimation Using a Single Deep Deterministic Neural Network](https://proceedings.mlr.press/v119/van-amersfoort20a.html)，ICML 2020|DUQ／RBF中心；需要新的输出机制。|
|17|Romano等，[Classification with Valid and Adaptive Coverage](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html)，NeurIPS 2020|自适应conformal集合；边际coverage不是unknown FAR。|
|18|Sagawa等，[Distributionally Robust Neural Networks for Group Shifts](https://openreview.net/forum?id=ryxGuJrFvS)，ICLR 2020|Group DRO；可用于最坏TX×场景风险，但组、更新顺序和验证谱系必须冻结。|
|19|Sehwag等，[SSD](https://openreview.net/forum?id=v5gjXpmR8J)，ICLR 2021|自监督特征＋Mahalanobis；无需外部OOD但训练成本高。|
|20|Chen等，[Adversarial Reciprocal Points Learning for Open Set Recognition](https://doi.org/10.1109/TPAMI.2021.3106743)，TPAMI 2021|对抗反向点；虚拟混淆点不是项目当前最小机制。|
|21|Sun等，[ReAct](https://proceedings.neurips.cc/paper_files/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)，NeurIPS 2021|激活裁剪；便宜但依赖source分位数，不能解释为跨LEO保证。|
|22|Angelopoulos等，[Uncertainty Sets for Image Classifiers Using Conformal Prediction](https://iclr.cc/virtual/2021/poster/3246)，ICLR 2021|conformal集合；可作离线风险审计。|
|23|Vaze等，[Open-Set Recognition: A Good Closed-Set Classifier Is All You Need?](https://arxiv.org/abs/2110.06207)，ICLR 2022|强调closed-set表征质量；支持本项目先守known floor。|
|24|Du等，[VOS](https://openreview.net/forum?id=u2GZOiUTbt)，ICLR 2022|特征空间虚拟异常；仍是合成unknown训练。|
|25|Wang等，[ViM](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html)，CVPR 2022|残差空间＋virtual logit；本项目已有双读出／虚拟logit相邻路线否定证据。|
|26|Sun等，[Out-of-Distribution Detection with Deep Nearest Neighbors](https://proceedings.mlr.press/v162/sun22d.html)，ICML 2022|归一化特征kNN；需要样本库并受receiver shift影响。|
|27|Sun、Li，[DICE](https://doi.org/10.1007/978-3-031-20053-3_40)，ECCV 2022|分类器贡献稀疏化；mask选择需合法验证。|
|28|Hendrycks等，[Scaling Out-of-Distribution Detection for Real-World Settings](https://proceedings.mlr.press/v162/hendrycks22a.html)，ICML 2022|现实OOD基准；说明方法排名随unknown拓扑改变。|
|29|Yang等，[OpenOOD](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d201587e3a84fc4761eadc743e9b3f35-Abstract-Datasets_and_Benchmarks.html)，NeurIPS 2022|广义OOD评测框架并区分near／far OOD；covariate-shift known与semantic unknown双轴主要由第32项Full-Spectrum OOD支持。|
|30|Rame等，[Fishr](https://proceedings.mlr.press/v162/rame22a.html)，ICML 2022|跨domain梯度方差不变；显式domain对齐不符合当前Phase1候选边界。|
|31|Mukhoti等，[Deep Deterministic Uncertainty](https://openaccess.thecvf.com/content/CVPR2023/html/Mukhoti_Deep_Deterministic_Uncertainty_A_New_Simple_Baseline_CVPR_2023_paper.html)，CVPR 2023|谱约束＋GDA；需改变骨干训练。|
|32|Yang等，[Full-Spectrum Out-of-Distribution Detection](https://doi.org/10.1007/s11263-023-01811-z)，IJCV 2023|明确区分covariate-shift known与semantic unknown；对RX／LEO和新TX双轴最有启发。|
|33|Angelopoulos等，[Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg)，ICLR 2024|一般损失风险控制；需要独立、匹配的校准样本。|
|34|Wang等，[Open-Set RF Fingerprinting via Improved Prototype Learning](https://arxiv.org/abs/2306.13895)，2023预印本|RF原型开集；unknown门仍需独立证据。|
|35|Huang等，[RF Fingerprint Extraction and Authentication Towards Open Set](https://doi.org/10.1016/j.dsp.2023.104363)，DSP 2024|时频表示＋对比＋open classifier；新增open head且依赖实验噪声设定。|
|36|Wang等，[Noise Robust Open-Set RFFI](https://livrepository.liverpool.ac.uk/3178859/)，INFOCOM Workshops 2024|RF噪声鲁棒OSR；需按公开稿和数据条件重现。|
|37|Han等，[Open-world RFFI via Augmented Semi-supervised Learning](https://ojs.aaai.org/index.php/AAAI/article/download/32003/34158)，AAAI 2025|无标签新类发现；unknown回流建类与当前Phase1／Phase3边界不兼容。|
|38|Xie等，[Few-Shot Open-set RFFI](https://doi.org/10.1109/JIOT.2025.3559183)，IEEE IoTJ 2025|高斯原型／Mahalanobis；更适合授权后的future registration，不是当前unknown决策。|
|39|Cai等，[Joint Prediction and Siamese Comparison RFFI](https://arxiv.org/abs/2501.15391)，2025预印本|预测＋Siamese比对；双支路和样本比较需新证据。|
|40|Oligeri等，[Probe-and-Model Benchmark for Open-Set RF Fingerprinting](https://arxiv.org/abs/2607.21564)，2026预印本|把接收机probe纳入开放集基准；提醒接收链路不可忽略，结论仍待正式出版验证。|
|41|Scheirer等，[Toward Open Set Recognition](https://doi.org/10.1109/TPAMI.2012.256)，TPAMI 2013|W-SVM与open-space risk的基础工作；阈值和尾部仍需匹配校准。|
|42|Zhou等，[Learning Placeholders for Open-Set Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Learning_Placeholders_for_Open-Set_Recognition_CVPR_2021_paper.html)，CVPR 2021|PROSER以placeholder建模未知方向；引入额外输出结构。|
|43|Bates等，[Distribution-Free, Risk-Controlling Prediction Sets](https://doi.org/10.1145/3478535)，JACM 2021|RCPS在独立holdout上控制一般损失；不自动给跨轨道条件风险。|

## 18.协同推理、风险与跟踪核心文献索引

|#|论文／入口|可吸收机制与边界|
|---:|---|---|
|1|Tenney、Sandell，[Detection with Distributed Sensors](https://doi.org/10.1109/TAES.1981.309118)，1981|Bayesian分布式检测；需明确观测模型。|
|2|Chair、Varshney，[Optimal Data Fusion](https://doi.org/10.1109/TAES.1986.310699)，1986|独立本地决策的可靠度加权；相关节点不满足前提。|
|3|Tsitsiklis，[Decentralized Detection](https://doi.org/10.1007/BF02551407)，1988|有限消息分布式检测；经典保证依赖分布假设。|
|4|Kam等，[Fusion of Correlated Local Decisions](https://doi.org/10.1109/7.256317)，1992|直接说明相关local decisions需要相关项。|
|5|Genest、Zidek，[Combining Probability Distributions](https://doi.org/10.1214/SS/1177013825)，1986|概率意见池公理和失效条件。|
|6|Heskes，[Selecting Weighting Factors in Logarithmic Opinion Pools](https://proceedings.neurips.cc/paper_files/paper/1997/hash/59f51fd6937412b7e56ded1ea2470c25-Abstract.html)，1997|加权log-pool及KL解释；重复专家会过度自信。|
|7|Tresp，[A Bayesian Committee Machine](https://doi.org/10.1162/089976600300014908)，2000|共享prior修正；任意softmax不等同likelihood。|
|8|Deisenroth、Ng，[Distributed Gaussian Processes](https://proceedings.mlr.press/v37/deisenroth15.html)，2015|rBCM和专家不确定度。|
|9|Liu等，[Generalized Robust Bayesian Committee Machine](https://proceedings.mlr.press/v80/liu18a.html)，2018|说明专家分割与聚合可能不一致。|
|10|Julier、Uhlmann，[Covariance Intersection](https://doi.org/10.1109/ACC.1997.609105)，1997|未知交叉相关下的保守融合；不自动给分类风险证书。|
|11|Dempster，[Upper and Lower Probabilities Induced by a Multivalued Mapping](https://doi.org/10.1214/aoms/1177698950)，1967|上下概率基础。|
|12|Smets、Kennes，[The Transferable Belief Model](https://doi.org/10.1016/0004-3702%2894%2990026-4)，1994|credal层与决策层分离。|
|13|Yager，[On the Dempster-Shafer Framework and New Combination Rules](https://doi.org/10.1016/0020-0255%2887%2990007-7)，1987|指出高冲突归一化问题。|
|14|Sensoy等，[Evidential Deep Learning](https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html)，2018|Dirichlet evidence需专门训练，不可直接套冻结softmax。|
|15|Han等，[Trusted Multi-View Classification](https://doi.org/10.1109/TPAMI.2022.3171983)，TPAMI 2023|动态多视图证据；依赖联合训练和可信视图假设。|
|16|Liu等，[Trusted Multi-View Opinion Aggregation](https://doi.org/10.1609/aaai.v36i7.20724)，AAAI 2022|一致性与vacuity；也暴露高冲突风险。|
|17|Kailkhura等，[Distributed Bayesian Detection with Byzantine Data](https://arxiv.org/abs/1307.3544)，2013|小网络存在被致盲临界比例。|
|18|Nadendla等，[M-ary Quantized Data Fusion in the Presence of Byzantine Attacks](https://doi.org/10.1109/TSP.2014.2314072)，2014|量化消息、攻击与信誉。|
|19|Blanchard等，[Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)，2017|Krum面向训练梯度，不是事件后验融合保证。|
|20|El Mhamdi等，[The Hidden Vulnerability of Distributed Learning in Byzantium](https://proceedings.mlr.press/v80/mhamdi18a.html)，2018|Bulyan；收敛不等于传感内容安全。|
|21|Yin等，[Byzantine-Robust Distributed Learning](https://proceedings.mlr.press/v80/yin18a)，2018|median／trimmed mean依赖节点数和恶意比例。|
|22|Zaheer等，[Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)，2017|置换不变集合函数；无默认复制不变和风险保证。|
|23|Lee等，[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)，2019|集合内注意力；需要真实same-event训练数据。|
|24|Veličković等，[Graph Attention Networks](https://iclr.cc/virtual/2018/poster/299)，2018|拓扑注意力；没有内生校准或Byzantine保证。|
|25|Prakash等，[TransFuser](https://openaccess.thecvf.com/content/CVPR2021/html/Prakash_Multi-Modal_Fusion_Transformer_for_End-to-End_Autonomous_Driving_CVPR_2021_paper.html)，CVPR 2021|跨模态attention；要求对齐的联合训练资产。|
|26|Teerapittayanon等，[BranchyNet](https://doi.org/10.1109/ICPR.2016.7900006)，2016|early exit；请求／退出策略本身需校准。|
|27|Teerapittayanon等，[Distributed Deep Neural Networks over the Cloud, the Edge and End Devices](https://doi.org/10.1109/ICDCS.2017.226)，2017|端边云切分与通信减少。|
|28|Kang等，[Neurosurgeon](https://doi.org/10.1145/3037697.3037698)，2017|按延迟和能耗选择切分。|
|29|Li等，[Edgent](https://arxiv.org/abs/1806.07840)，2018|early exit和动态网络联合调度。|
|30|Shao、Zhang，[BottleNet++](https://arxiv.org/abs/1910.14315)，2019|中间特征压缩与信道噪声；压缩误差必须进入决策区间。|
|31|Sinopoli等，[Kalman Filtering with Intermittent Observations](https://doi.org/10.1109/TAC.2004.834121)，2004|丢包会出现稳定性相变。|
|32|Bar-Shalom，[Update with Out-of-Sequence Measurements](https://doi.org/10.1109/TAES.2002.1039398)，2002|迟到量测按事件时间处理。|
|33|Besada-Portas等，[Asynchronous Fusion of Out-of-Sequence and Erroneous Data](https://doi.org/10.1016/j.automatica.2011.02.030)，2011|异步、迟到与损坏量测联合处理。|
|34|Reid，[An Algorithm for Tracking Multiple Targets](https://doi.org/10.1109/TAC.1979.1102177)，1979|MHT的birth、miss、false alarm和多假设。|
|35|Fortmann等，[Sonar Tracking of Multiple Targets Using JPDA](https://doi.org/10.1109/JOE.1983.1145560)，1983|互斥关联事件的概率边缘化。|
|36|Mahler，[Multitarget Bayes Filtering via First-Order Multitarget Moments](https://doi.org/10.1109/TAES.2003.1261119)，2003|PHD／RFS；一阶矩牺牲身份连续性。|
|37|Vo、Ma，[The Gaussian Mixture Probability Hypothesis Density Filter](https://doi.org/10.1109/TSP.2006.881190)，2006|线性高斯下GM-PHD闭式近似。|
|38|Williams，[Marginal Multi-Bernoulli Filters](https://doi.org/10.1109/TAES.2015.130550)，2015|连接MHT、JIPDA和RFS。|
|39|García-Fernández等，[Poisson Multi-Bernoulli Mixture Filter](https://doi.org/10.1109/TAES.2018.2805153)，2018|clutter、未检测目标和track假设。|
|40|Tibshirani等，[Conformal Prediction Under Covariate Shift](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)，2019|加权exchangeability需可靠density ratio。|
|41|Romano等，[Classification with Valid and Adaptive Coverage](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html)，2020|分类prediction set的边际coverage。|
|42|Bates等，[Distribution-Free, Risk-Controlling Prediction Sets](https://doi.org/10.1145/3478535)，JACM 2021|独立holdout上的一般风险控制。|
|43|Gibbs、Candès，[Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)，2021|在线更新与本项目query零更新冲突。|
|44|Podkopaev、Ramdas，[Distribution-Free Uncertainty Quantification for Classification Under Label Shift](https://proceedings.mlr.press/v161/podkopaev21a.html)，2021|label shift破坏普通校准。|
|45|Angelopoulos等，[Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)，ICLR 2024|单调损失的期望风险控制。|
|46|Prinster等，[JAWS-X](https://proceedings.mlr.press/v202/prinster23a.html)，ICML 2023|请求／选择策略产生feedback covariate shift。|

凭证和确权的实现还必须独立服从[NIST SP 800-207](https://csrc.nist.gov/pubs/sp/800/207/final)、[RFC 5280](https://www.rfc-editor.org/info/rfc5280/)、[RFC 6960](https://www.rfc-editor.org/info/rfc6960/)、[RFC 9334 RATS](https://www.rfc-editor.org/rfc/rfc9334.html)、[RFC 9052 COSE](https://www.rfc-editor.org/rfc/rfc9052)及[W3C Verifiable Credentials 2.0](https://www.w3.org/TR/vc-data-model-2.0/)。它们能证明来源、生命周期与状态，不能证明节点传感内容诚实。

## 19.天基RFFI、NTN、轨道时间与在轨计算来源

|类别|官方或一手来源|对方案的约束|
|---|---|---|
|RFFI数据|[WiSig官方数据页](https://cores.ee.ucla.edu/downloads/datasets/wisig/)、[WiSig论文](https://arxiv.org/abs/2112.15363)|174个TX、41个USRP RX和跨月采集可验证receiver／channel／time漂移；不是卫星数据。|
|跨RX RFFI|[Receiver-Agnostic Radio-Frequency Fingerprinting Through Physical-Layer Deep Learning](https://arxiv.org/abs/2207.02999)|接收机硬件失真可压过TX特征；支持独立receiver holdout。|
|多RX SEI|[Specific Emitter Identification with Different Codes and Multiple Receivers](https://doi.org/10.1109/TAES.2024.3456090)|不同RX可造成严重性能下降；数值仅限其地面实验。|
|时间漂移|[The Day After Tomorrow](https://research.tue.nl/en/publications/the-day-after-tomorrow-on-the-performance-of-radio-fingerprinting/)、[Reliability of Radio Frequency Fingerprinting](https://arxiv.org/abs/2408.09179)|跨天、功率循环和FPGA reload会改变指纹；必须做day／pass分层。|
|LoRa鲁棒RFFI|[Scalable and Channel-Robust RFFI](https://arxiv.org/abs/2107.02867)|metric learning和channel-independent特征；仍是地面LoRa。|
|数据采集|[A Comprehensive RF Dataset Collection and Release](https://arxiv.org/abs/2201.02213)|SigMF IQ／FFT及环境元数据可借鉴。|
|NTN总体|[3GPP NTN overview](https://www.3gpp.org/technologies/ntn-overview)|LEO高速产生快速Doppler、时延和moving cell；需要星历和位置／运动。|
|NR-NTN研究|[3GPP TR 38.821](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3525)、[TR 38.811归档](https://www.3gpp.org/ftp/Specs/archive/38_series/38.811?sortby=daterev)|场景与解决方案必须锁定标准版本。|
|NR架构|[ETSI TS 138 300 V17.12](https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/17.12.00_60/ts_138300v171200p.pdf)|NTN接入仍处于完整NR架构和时序约束内。|
|轨道消息|[CCSDS出版物目录](https://ccsds.org/publications/allpubs/)|ODM 502.0-B-3、Tracking Data Message 503.0-B-2规定可复现轨道／跟踪消息。|
|任务时间|[JPL NAIF Time](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html)、[JPL NAIF SCLK](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/sclk.html)|UTC／TAI／GPS／TDB和SCLK需要显式correlation，不能混用时间戳。|
|TLE／传播|[CelesTrak TLE格式](https://celestrak.org/NORAD/documentation/tle-fmt.php)、[TLE／SGP4说明](https://celestrak.org/columns/v04n03/)|保存TLE epoch、来源和传播误差；TLE不是高精度真值。|
|星历复核|[JPL Horizons API](https://ssd-api.jpl.nasa.gov/doc/horizons.html)|可独立核对位置／可见性，但仍继承输入与模型误差。|
|DTN|[CCSDS DTN工作组](https://ccsds.org/publications/allpubs/entry/3222/)|大时延、间歇链路和store-and-forward要求记录schedule、queue和deadline。|
|星上计算|[NASA SmallSat avionics](https://www.nasa.gov/smallsat-institute/sst-soa/small-spacecraft-avionics/)|自治和编队带来算力、功耗、热和FDIR约束。|
|星上数据处理|[ESA On-board Data Processing](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Onboard_Data_Processing/What_is_On-board_Data_Processing)|原始载荷数据可能超过下传能力，必须度量压缩、存储和计算。|
|在轨网络|[JPL/NASA New Observing Strategies](https://ai.jpl.nasa.gov/public/documents/papers/NOS-IGARSS-2024.pdf)|星上分析和星间链路可降低响应时延；仍需任务级SLA。|
|干扰监测|[ESA GNSS Interference Monitoring from LEO](https://navisp.esa.int/news/article/Detect%20interference%20from%20orbit%20on%20a%20wider%20scale)|LEO原始IF可发现和定位非合作干扰；不是RFFI分类精度证据。|
|卫星定位|[ITU-R SM.2355-2](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-SM.2355-2-2023-PDF-E.pdf)|TDOA／FDOA需多星、同步时钟；精度依赖几何、带宽和钟差。|
|定位测试|[ITU-R SM.2139-0](https://www.itu.int/dms_pubrec/itu-r/rec/sm/R-REC-SM.2139-0-202108-I%21%21PDF-E.pdf)|固定带宽、采集时长和非共线测试位置应进入验收矩阵。|
|测量设施|[ITU-R SM.2182-3](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-SM.2182-3-2023-PDF-E.pdf)|GSO／NGSO测量和TDOA／FDOA需要任务设施校准。|
|干扰工单|[ITU-R SM.2149](https://www.itu.int/dms_pubrec/itu-r/rec/sm/R-REC-SM.2149-0-202209-I%21%21PDF-E.pdf)|事件记录应含卫星、频率、带宽、时间和定位方法。|
|合规记录|[FCC 25-69](https://docs.fcc.gov/public/attachments/FCC-25-69A1.pdf)|运营者需调查、记录有害干扰及处置；可作为事件schema来源，不是模型标签。|
|卫星RFFI|[OrbID](https://www.ndss-symposium.org/ndss-paper/auto-draft-623/)|真实卫星识别研究入口；需严格核其数据、轨道和攻击模型边界。|

补充一手资料如下。它们用于交叉核对接收机漂移、数据链、定位和任务可行性，不单独支撑本文的性能结论：

- RFFI与接收机：[GAN-RXA](https://arxiv.org/abs/2303.14312)、[Virginia Tech RFFI扩展研究](https://vtechworks.lib.vt.edu/items/a3d0a005-054b-41cf-a6de-b57e157f71cb)、[Multiple Receiver Specific Emitter Identification](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12606)、[SignCRF](https://arxiv.org/abs/2303.12811)、[CSI-RFF](https://arxiv.org/abs/2403.15739)。
- 标准与时间：[ATIS托管TR 38.821 V16.0.0](https://www.atis.org/wp-content/uploads/3gpp-documents/Rel16/ATIS.3GPP.38.821.V1600.pdf)、[NASA托管CCSDS ODM](https://www.nasa.gov/wp-content/uploads/2023/09/ccsds-orbit-data-messages.pdf)、[CCSDS Tracking Data Message副本](https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/document/trackingdatamessagestandard.pdf)。副本用于可访问性，版本仍以标准机构为准。
- 星上处理与运营：[NASA Ground Data Systems](https://www.nasa.gov/smallsat-institute/sst-soa/ground-data-systems-and-mission-operations/)、[NASA ESTO／AIST NOS报告](https://ntrs.nasa.gov/api/citations/20210010318/downloads/CP%E2%80%9320210010318%20ESTO-AIST_NOS-Workshop-Report_Final_2021-02-17.pdf)、[ESA Onboard Processing](https://www.esa.int/Applications/Connectivity_and_Secure_Communications/Onboard_processing)、[ESA Onboard Computers](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Onboard_Computers_and_Data_Handling/Onboard_Computers)、[ESA TychoBoB](https://incubed.esa.int/portfolio/tychobob/)。
- 干扰与定位项目：[NAVISP GINKO-2](https://navisp.esa.int/project/details/313/show)、[NAVISP EL1-043](https://navisp.esa.int/project/details/145/show)、[NAVISP GIDAS](https://navisp.esa.int/project/details/13/show)、[ITU-R SM.2356-2](https://www.itu.int/hub/publication/r-rep-sm-2356-2-2018/)、[ESA GNSS干扰定位演示](https://navisp.esa.int/uploads/files/documents/5de90bd70b90c298827454.pdf)。
- 新近空间定位研究：[Low-Complexity Direct Geolocation from LEO](https://arxiv.org/abs/2606.22751)、[Complexity-Scalable Direct Geolocation and Cancellation](https://arxiv.org/abs/2607.02190)、[Three Years of GNSS Interference Monitoring from LEO](https://arxiv.org/abs/2009.04093)、[Single-Satellite GNSS Spoofer Geolocation](https://arxiv.org/abs/2503.17791)、[Space-Based Electromagnetic Spectrum Sensing and Situation Awareness](https://spj.science.org/doi/10.34133/space.0109)。2025—2026条目均按预印本或项目状态引用，不当作已普遍复现的性能事实。

## 20.现实数据入口与可采集性

|#|数据／计划|访问与可得内容|正式G2缺口|
|---:|---|---|---|
|1|[CYGNSS L1 Raw IF](https://podaac.jpl.nasa.gov/dataset/CYGNSS_L1_RAW_IF)|Earthdata注册；DDMI原始IF counts、三输入天线、空间时间元数据。|无授权unknown／干扰真值；同事件多星不保证。|
|2|[CYGNSS Full DDM](https://podaac.jpl.nasa.gov/dataset/CYGNSS_L1_FULL_DDM_V3.0)|公开L1 DDM和几何元数据。|不是raw IQ或RFFI专用波形。|
|3|[COSMIC-2 Data](https://www.cosmic.ucar.edu/what-we-do/cosmic-2/data)|Level-0仪器二进制、L1a姿态／POD、L1b SP3等。|未承诺复数IQ，也无unknown标签。|
|4|[MERRByS TechDemoSat-1](https://merrbys.co.uk/data-access/downloads)|注册后FTP，主要为GNSS-R L1B／L2 DDM；CC BY-NC。|常规L0／IQ不公开，单星。|
|5|[HydroGNSS](https://www.hydrognss.org/mission/mission-information/)|注册数据入口，双LEO；支持短时未处理采样。|原始采样开放程度、同事件窗口与真值需专项申请。|
|6|[EUMETSAT Commercial RO](https://api.eumetsat.int/data/browse/collections/EO%3AEUM%3ADAT%3A0374?format=html)|公开L1B；原始L0／L1a可向Helpdesk申请。|第三方许可，公开主产品不是IQ。|
|7|[NOAA PlanetiQ RO RDR](https://catalog.data.gov/dataset/commercial-comm-radio-occultation-ro-raw-data-record-rdr-from-planetiq)|商业RO原始记录目录。|访问条款和IQ／时钟字段需逐项确认。|
|8|[NOAA／NCEI PlanetiQ archive](https://www.ncei.noaa.gov/data/commercial-radio-occultation-data/planetiq-rdr/archive/)|实际归档入口。|目录存在不等于无限制公开下载。|
|9|[Spire Interference Detection](https://spire.com/space-reconnaissance/interference-detection/)|商业历史／近实时干扰CSV和谱图，可商务申请。|未公开raw IF／IQ，标签与精轨受合同约束。|
|10|[ESA NAVISP EL2-117](https://navisp.esa.int/project/details/159/show)|曾以Spire采集开展单星／多星干扰定位。|公开资料不是原始数据；需后续合作。|
|11|[ESA OPS-SAT Space Lab](https://opssat.esa.int/)|可注册申请SDR在轨实验并设计受控采集。|不是现成数据集；受审批、带宽和下传预算限制。|
|12|[OPS-SAT SDR技术说明](https://opssat.esa.int/docs/opssat1/2021_OPS-SAT_in-orbit-a_technical_rundown_of_this_open_experimentation_platform.pdf)|公开AFE、频段、采样和处理能力。|能力说明不等于已存在的开放RFFI数据。|
|13|[SatNOGS Network／API](https://network.satnogs.org/about/)|全球多地面站公开观测、waterfall、audio和metadata。|默认不上传IQ；同步、前端和时钟校准不统一。|
|14|[SatNOGS artifact policy](https://wiki.satnogs.org/Artifacts)|明确IQ默认关闭、waterfall为图像、audio有损。|需要逐站授权原始文件，不能当公开raw-IQ库。|
|15|[OPSSAT-AD](https://doi.org/10.5281/zenodo.12588358)|公开真实卫星遥测段和异常标签。|不是RF／IQ或发射机unknown数据。|
|16|[ESA MUST／WebMUST](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Mission_Utility_and_Support_Tools_MUST)|授权伙伴可访问任务遥测、事件和辅助数据。|非公开库，未承诺IQ。|
|17|[ICE GNSS-R data server](https://gnssr-data.ice.csic.es/)|注册可取地面／航空complex raw和DDM。|可验证IQ流程，但不是天基或同事件多星。|
|18|[ESA GSSC](https://gssc.esa.int/activities/ftp-and-web-access-to-gnss-repository/)／[IGS](https://igs.org/data/)|GNSS观测、轨道、时钟和气象辅助。|通常是RINEX而非IQ；适合时间／几何校准代理。|

没有一个公开入口同时满足“星载原始复数IQ＋接收机／振荡器／时钟校准＋同一发射事件多星配对＋授权unknown／干扰真值”。最可执行的组合路径是：以CYGNSS／HydroGNSS解决真实空间RF和轨道观测，以OPS-SAT或商业Spire／NAVISP获取受控事件，以CCSDS／SPICE／SCLK和前端校准sidecar补齐时间与几何，再独立建立注册known、授权unknown、形式校准和formal测试四层互斥资产。

## 21.独立复核与质量收口

|复核角色|范围|结果|
|---|---|---|
|开放世界／Phase1科学复核|方法分类、原始论文、项目既有实验推论、P1-GD-ProtoNLL权限与设计状态|`P0=0，P1=0，ALLOW`|
|协同融合科学复核|双轴相对证据、`Sigma=DKD`、活跃集QP、有限先验、unknown风险证书、`f=1`、MHT零回写和G0／G1／G2|`P0=0，P1=0，ALLOW`|
|机械文档核验|本地／外部链接、DOI编码、54／18计数、13个表格、代码围栏、Mermaid和章节闭合|`PASS`|

两条科学复核共同审查的正文SHA256为`97227ADC303E1CA20DEE2AFCC3535083149E57286280EB8A4EDDE9B3D6606729`。本节只记录复核结果，不改变算法、实验矩阵或性能结论。本轮未访问N607、未创建或启动新实验，也未把设计完成度升级为性能证据。
