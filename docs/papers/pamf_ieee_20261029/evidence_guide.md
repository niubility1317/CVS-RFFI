# PAMF 论文第四版：方法设计、天基泛化与实验依据

PAMF manuscript v4: generalization, deployment footprint, and evidence. 2026-10-30.

本次只整合已有冻结结果、整理作图和撰写论文，没有训练、预测或新增目标评分。中文与英文论文共用段落结构、公式和数据，便于逐段核对。英文采用 IEEEtran journal、10 pt、双栏；中文沿用该双栏骨架供审阅，不视为正式英文投稿模板。目标期刊尚未指定，因此没有虚构刊名、录用日期、作者和基金信息。

Only frozen results are reused. No training, prediction, or new truth-side scoring is performed. The English draft uses IEEEtran journal mode; the Chinese counterpart uses the same equation, figure, and table structure for review. Journal-specific submission requirements and author details remain to be supplied.

## 1. 核心定位 / Core positioning

研究问题：部分源域身份标签可见、源接收机元数据可见、目标接收机不可访问时，如何以紧凑网络识别已知发射机，并承受额外信道退化。应用动机为单星被动监听；当前实验证据为地面 WiSig 测量及模拟星地压力，不写成真实卫星终端接入认证或在轨验证。

Research question: identify known transmitters at inaccessible, unseen receivers from partially labeled source IQ, while coping with additional channel degradation. Passive satellite listening motivates the setting; the evidence is terrestrial WiSig plus simulated channel stress.

论文主线：星地信道损伤下的跨接收机泛化 → 在较小参数和权重存储预算下保持识别优势 → 结构、训练和可选恢复分别解释收益 → 评价适用条件。半监督是支撑泛化的训练手段，DURA 是可选稳健性扩展，二者不取代“泛化与紧凑结构”的核心定位。PAMF 始终指三视图 SIG-TIME + Tri-SIG-TIME SSL 的识别器；PAMF+DURA 明确包含独立恢复器，不能混名。

Narrative: receiver generalization under space--ground impairment → recognition with a small weight footprint → architecture/training/restoration attribution → conditions and limits. SSL supports the main generalization objective; DURA is optional. PAMF names the identifier; PAMF+DURA includes the front end.

## 2. 可写的贡献与支持范围 / Contributions and claim boundaries

| 编号 | 可写贡献 / Defensible contribution | 已有证据 / Evidence | 边界 / Boundary |
|---|---|---|---|
| C1 | 紧凑结构化表征：时频、非线性记忆、参考响应和局部路径 / Compact complementary signal representation | E200 CE 15结构×8种子；E50 CE/SSL两层消融；单/三视图8种子 | 结构整体优于已比较骨干；不宣称各分支普遍必要或唯一恢复TX物理参数 |
| C2 | 接收机感知的半监督训练：同伪身份跨RX配对、扰动与L梯度锚定 / Receiver-aware source SSL | 监督/PL/完整方法；增强×PL；同排列监督；F/G U权限；11设置配对差 | 当前完整方案相对PL七正四负；普通PL与完整方案同时改变排列和U算法；不宣称全条件普遍领先 |
| C3 | 系统泛化与资源折中，含可选独立恢复 / Generalization--resource trade-off with optional restoration | 多划分、标签预算、源数量、十类、六景；参数与权重存储；DURA五识别器及权限控制 | 不把历史G消融改为当前完整SSL绑定证据；不宣称优于未匹配的恢复算法，不从参数量推出星上加速或功耗 |
| D | 参数小与留出判别表现同时成立 / Compactness with competitive discrimination | F 232993参数、0.89MiB FP32权重；比ResNet-18-1D少93.94%；F+G 499944参数、1.91MiB | 参数表比较的是已计数常见架构，不冒充四种外部RFFI方法的完整成本比较；权重存储不含激活和运行工作区 |
| E | 结构与训练作用有条件性 / Condition-dependent structural effects | 无SIG在CE双跨六景+0.85pp，SSL对应-2.10pp；PA删除跨RX CE/SSL -7.21/-6.78pp | 作为实验发现；不将差异直接当作物理因果证明 |

不作为独立新发明：Sinc 滤波、记忆多项式、路径签名、MixStyle、高置信伪标签、CG或CUDA Graph。本稿主张具体任务下的组合结构与训练流程。新颖性仍需在最接近论文全文与实际实现版本上逐项核对。

Not claimed as inventions: Sinc filtering, memory polynomials, signatures, MixStyle, pseudo-labeling, CG, or CUDA Graph. The contribution is the particular task-grounded design and validated integration. A full closest-work novelty audit remains necessary before asserting priority.

## 3. 完整正文框架 / Full manuscript structure

| 节 | 中文 / English | 该节回答的问题 |
|---|---|---|
| I | 引言 / Introduction | 天基载荷面对的泛化和资源联合要求；相关工作；紧凑PAMF及支撑机制 |
| II | 问题建模与信道 / Problem formulation and channel model | 观测、源/目标权限、闭集风险；绝对FSPL与相对归一化；六景及接收处理 |
| III | PAMF识别器 / Proposed PAMF identifier | 分支核心思路与必要公式；三视图局部编码；配对SSL；梯度保护与日程 |
| IV | 独立源域恢复 / Independent source-only restoration | 粗恢复、实线性作用、两阶段prior+CG；训练权限与F/G分离 |
| V | 实验设置 / Experimental setup | 数据角色、划分、场景、种子、预算、对照与统计口径 |
| VI | 结果与讨论 / Results and discussion | 泛化优势能否保持；是否依赖大模型；SSL解释多少；最难信道瓶颈；源组成与标签；两层结构；载荷配置与边界 |
| VII | 结论 / Conclusion | 已支持结果、适用边界及高价值缺项 |

三个被引用写作对话均已读取。Introduction 复用问题动机但改为被动监听语义；方法采用用户最后指定的核心机制说明，保留必要公式而省略逐层实现堆叠；信道采用最后完整版并保留绝对FSPL→相对归一化→固定新增噪声底→质量相关接收处理的逻辑。修正旧稿中把8种子扩写成所有主方法比较的描述。

All three writing chats were read. Their motivation, concise mechanism descriptions, and final channel formulation are reconciled with the completed results. Outdated statements expanding eight-seed evidence to all main comparisons are removed. These chat drafts are source material, not instructions to modify their chats or launch work.

## 4. 正文图表 / Main figures and tables

编号一律指“第三版 PDF 页码”，同时保留原 Fig 名，防止误认。第三版第11至22页为外部网络逐一介绍页，按用户要求排除；第8至10页的PAMF结构专页保留补充候选。其余候选均逐页列入已选50页_正文补充映射.csv，不删除原图。

Page numbers below refer to the selected r3 PDF, not historical Fig labels. External per-network detail pages 11–22 are excluded; PAMF detail pages 8–10 remain supplementary candidates. Every page is accounted for in the mapping CSV.

| 正文编号 | 来源页 / Source pages | 正文用途与取舍 / Purpose |
|---|---|---|
| Fig.1 | 新绘系统流程 | 源域开发与两条冻结推理路径；概念性部署示意 |
| Fig.2 | 新绘 PAMF 架构 | 多分支、三视图与残差融合；另有两种可选画法 |
| Fig.3 | 新绘 DURA 结构 | 粗恢复、作用估计与两阶段先验/一致性；另有阶段细节图 |
| Fig.4 | 第1、2页；原Fig001A、Fig001 | D1--D3跨RX 126000条／D0双留出42000条相邻；每图完整15模型、八种子；均值±SD贴点，F1放补充 |
| Fig.5 | 第25、26页；原Fig022、Fig031 | 有增强监督/PL/完整方法六景面板；clean在Table II；无增强23/24页放补充 |
| Fig.6 | 第31、32页；原Fig025、Fig034 | 七场景曲线，跨RX/双跨并排，相同坐标范围 |
| Fig.7 | 第27至30页；原Fig023/032、024/033 | RX划分+标签比例六景四面板；clean完整数据保留补充 |
| Fig.8 | 第35、36页；第37、38页为同实验替代版 | 六景跨RX/双跨原图相邻；第33、34页的clean图保留补充 |
| Fig.9 | 第42页；39至41页为同实验替代版 | DURA五识别器恢复增益，保留数值、曲线箭头和pp提升；替代画法不再占正文 |
| Fig.10 | 第43至50页同实验，并复用逐种子数据 | CE/完整SSL双留出原始配对图相邻，含clean/六景与准确率/F1；跨RX完整数据随附 |
| Table I | data_settings与已核实物理角色 | 主数据构成、源/目标RX、样本数；区分训练池10%与源全池7% |
| Table II | 01_main_comparison.csv | 5识别器×两种holdout，clean、raw六景、G-LU六景及F1，3种子 |
| Table III | 07_U_permission.csv | 自然序监督/PL、同排列监督与完整SSL，分别接raw/G-L/G-LU；复用已有归因控制 |
| Table IV | 第三版效率表，参数+部分运算计数 | 不画效率图；MAC/FLOPs覆盖范围写清；统一延迟缺项不填造 |

正文实验图直接复用已选第三版矢量PDF，只裁切、等比例排版和增加黑色范围标题；保留原配色、线型、标记、误差棒及坐标，不重新绘制数据。柱状图保留明确零基线以避免截断夸大；折线图使用标明范围的非零起点，同组坐标一致。所有图为PDF矢量，另附PNG预览；每图有对应CSV，来源见figures/manifest.json。跨RX和双跨始终相邻。

Main figures are reformatted from frozen statistics without changing values. Bars have explicit zero baselines; paired line plots use consistent nonzero limits. Vector PDFs, PNG previews, and CSV data are included. Cross-RX and double holdout are adjacent throughout.

Fig.4 修正：第二版误用汇总表中不完整的 `cross_rx` 字段。第三版直接嵌入已选图 `Fig001A_network_cross_rx` 和 `Fig001_network_accuracy_f1` 的原始准确率面板，完整15模型、八种子，分别对应126000条跨RX与42000条双留出；两类物理记录互不重叠。跨RX原图数据已由冻结混淆矩阵精确计算，本次不新增计算、预测或真值访问。

## 5. 每项结论对应哪些实验 / Conclusion-to-experiment map

| 结论 / Conclusion | 结果与证据 / Evidence | 可复用内容 / Reuse |
|---|---|---|
| 完整PAMF优于四种已比较外部识别器 | Main跨RX六景71.16 vs SDG65.30；同G79.90 vs67.12；双跨68.81/76.83 vs63.59/65.66 | 01_main_comparison；Table II；Fig.5/4 |
| 轻量结构在纯CE下有效 | E200双跨79.73 vs ComplexResNet75.05；232993参数 | 15_architecture；Fig.4；Table IV |
| 完整SSL增益随条件变化 | 相对PL跨RX clean+2.25，raw六景+0.11，同G六景+1.39；11设置七正四负 | 14_paired_effects；02_factorial；Table III；Fig.7/6 |
| 相同排列下U包不必提高raw六景 | 同排列Sup/完整raw72.53/71.16；G-LU79.51/79.90 | 07_U_permission；不新增重排PL |
| DURA在五种识别器上有平均收益 | Main五识别器raw/匹配G-LU配对；PAMF跨RX+8.74、双跨+8.02pp | 13_DURA_generality；Fig.9；原39–42页只选一版 |
| PA分支贡献较明确 | 跨RX六景删除PA：CE -7.21、SSL -6.78pp | 03_two_layer；Fig.10；04嵌套CE |
| SIG作用依赖训练条件 | 删除SIG：CE跨RX/双跨 -0.88/+0.85，SSL -3.85/-2.10pp | 03_two_layer；04嵌套；不混写E200CE |
| 三视图相对单视图提升有限 | E200双跨clean +0.10pp；完整SSL8种子跨RX/双跨六景+0.59/+0.22pp | 15_architecture与06_view_comparison分别报告 |
| RX划分影响明显 | PAMF+G跨RX六景79.90/66.58/75.26 | 10_receiver_splits；Fig.7 |
| 标签增加总体有利于本方法 | 1/5/10%恢复跨RX六景74.43/77.41/79.90 | 09_label_fraction；Fig.7；10%复用Main |
| 源数量不能概括为越多越好 | clean总体改善；六景非单调，且样本量同步增加 | 08_source_receiver_curve_common6；Fig.8 |
| 十类规模仍有恢复收益 | raw62.09→72.60；PL63.31→72.44 | 11_ten_class_scale；补充S5；不是大规模174TX验证 |

## 6. 补充材料与版本隔离 / Supplementary material and version separation

S1：第3至10页，架构综合、RX/TX分层、参数散点、SIG删除、PAMF结构专页。S2：架构Macro-F1、逐种子表。S3：第23/24页无增强监督/PL，加完整增强×PL四格与U曝光表。S4：第27至38页未入主文的clean或替代画法。S5：十类及完整11设置。S6：第43至50页逐景、F1、池化和配对结果，以及完整9结构、去SIG五组CE。S7：历史T1/自然序REPL内部机制、G七组及历史成本，显式标“历史配方”，不用于替代当前缺项。

S1–S7 preserve aggregate architecture detail, per-seed F1, full augmentation/PL controls, unselected clean panels, ten-class results, full structural evidence, and explicitly historical mechanism/cost panels. Alternative plot styles are not extra experiments. No current result is silently relabeled from T1 or ordered REPL.

## 7. 关键缺项及优先级 / Critical gaps and priorities

| 优先级 | 缺项 / Gap | 已有可复用材料 / Reuse first | 本次处置 / Current action |
|---|---|---|---|
| P0 写作 | 最接近工作逐项新颖性；独立贡献是否足够清晰 | 三写作对话、已核对原始文献、当前模型与配置 | 已给可辩护贡献；不写“首次”或“全部机制必要” |
| P0 整理 | 方法版本、原生预算、source RX标签权限与baseline版本 | 已有复现审计、resolved config、日志与契约审计 | 正文披露预算与权限；DRIFT引用v2与实际执行版本须最终核对 |
| P0 统计 | 主比较3种子的小幅差异与域级变动 | 逐种子、RX/day/TX分层CM与净纠错 | 已使用配对差；不伪造p值或置信区间，不把包数当独立n |
| P1 独立性 | 冻结方法后的第二独立采集/数据集 | 优先查未用于研发的现有真实多RX数据 | 不启动；若补新证据优先这一项 |
| P1 效率 | 同硬件/软件/精度/输入与边界的当前F和F+G时延 | 历史资源表、普通网络部署时延、当前架构参数 | Table IV只保留可比参数与部分运算；不拼接旧4.53ms为当前时延 |
| 条件项 DURA | 匹配外部恢复方法主表 | 先查已完成传统与学习恢复比较，核对L/U及教师来源 | 当前包尚未形成该主表；不足时列建议，不自动训练 |
| 条件项 SSL | 当前固定排列下内部机制的匹配删除/强SSL对照 | 已有同排列Sup、F/G权限四格、自然序历史四机制 | C2写为整体训练流程，收窄必需性与普遍优越主张；不重启取消的Permuted PL |
| 投稿信息 | 目标刊物、作者、机构、基金、完整数据/代码可用性声明 | 待用户提供或确定 | 作者信息明确省略；不编造基金、仓库公开状态或投稿适配完成 |

本稿采用足够明确的机制公式，但尚不等于完成全部代码公开与最接近方法全文审查。当前已核实参考文献以基础方法、四个对比识别器与NTN技术报告为主；投稿前应结合目标期刊扩充真实相关工作，不能用凑数量的引用替代论证。

The draft includes core mechanism equations but is not a completed public reproduction package or an exhaustive novelty review. References cover verified foundational methods, the four comparison identifiers, and an NTN report. Target-journal related work should be completed with relevant sources, not citation-count padding.

## 8. IEEE 格式与交付 / IEEE format and delivery

使用 IEEEtran journal 10pt；不使用geometry改页边距，不虚构IEEE版权/刊期。图题在图下，表题在表上；章节罗马序号、子节字母；公式自动编号；文献用cite+IEEEtran BibTeX按首次出现排序；跨栏图表通过figure*/table*处理。英文为可编辑LaTeX项目；中文使用XeLaTeX和中文字体，仅作对应审阅稿。实际编译、交叉引用与页面检查结果写入validation_final.json。

IEEEtran journal mode is retained without margin manipulation. Figures have bottom captions, tables top captions, equations automatic numbering, and references numbered using IEEEtran BibTeX. The English version is the submission-language working draft; the Chinese version is an aligned review copy. Final compilation and visual checks are recorded separately.

入口：english/main.pdf、chinese/main.pdf、中英文逐段对照稿.md、已选50页_正文补充映射.csv、figures/data、data。原第三版PDF与全部历史结果原路径保持。新增文件只在本次论文目录和专用源码文件中；不操作远端训练任务，也不向其他写作对话发送消息。
