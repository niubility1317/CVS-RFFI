# PAMF Phase2：域适应与新类注册的衔接、公平比较和优势证据

2026-10-28，r04 补充分析。**DESIGN_ONLY：未实现新的组合 runner，未读取目标 IQ／truth，未启动实验。**沿用 G10 八套与 D-LS／D-MU／D-LU 三个单景、相同源角色、4 模型、7 RX、5 draw、K1/5/10/20、N_new0/2/5/10/20。

## 1. 核心判断

旧总表 T1 检验旧类适应，T2 检验已有注册方法及其 B 扩展；二者都需要，但仅把两张表并列，还不足以证明 PAMF 的完整 Phase2 优势。闭集 DA 原方法没有 C，只能列能力边界，不能把 C_new 填 0 或 N/A 后当作性能胜利。应在同一分阶段任务下，给它们接上明确的、可复现的注册扩展；原生 C 方法则从自己的合法 B 状态继续注册。

“同时处理域变化与类别扩展”本身并非无人研究。ECCV2020 的 [Class-Incremental Domain Adaptation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580052.pdf) 已研究在域偏移下识别共享类和新增类；ECCV2024 的 [Versatile Incremental Learning](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4550_ECCV_2024_paper.php) 讨论域／类别增量干扰。它们是问题定位依据，不自动加入当前项目 14 个 RFFI 家族清单；其数据和标签假设不能直接当作本任务协议。

合理主张是：PAMF 的物理表征与 MILKR 统一 support 拟合机制，在固定接收观测、极少目标标签和新旧类共同竞争下，能否比合理串联系统取得更好的适应、注册平衡与成本。是否优越需要新实验，不是由能力表推出。

## 2. 统一任务与完整状态

设源模型为 F_s，旧类目标 support 为 S_o，新类 support 为 S_n。三次服务状态为：

1. A：只有源模型和旧类名册，测试旧 query。
2. B：开放 S_o，得到 M_B；在新类到来前必须能够识别旧类。
3. C：才开放 S_n，得到 M_C；所有 query 面对 old6＋N_new 个已注册类。

每个阶段的 query 只用于冻结后的逐样本推理；旧 query 在 A/B/C 保持同一物理记录。对每个 N_new 都从同一个 B 独立分叉，不能 C2→C5→C10 串行冒充规模实验。

完整 B 状态不仅是 encoder 权重，还包括分类头／核／原型／memory、BN 和其他 buffer、类别映射、规范化规则，以及 C 需要的参考模型／teacher／合法固定 Fisher 等。C 不能偷偷加载另一条 run 或重新从源模型开始，却声称继承 B。只继承 encoder 而换读出的组合允许，但必须明确说明。

旧 support 在 C 仍可被当前协议允许的方法复用，累计标签数仍是 6K，而非再次付出 6K 标签。重复训练曝光、保存原 IQ／特征／原型的成本分别计量。原生 exemplar-free 方法不强制额外保存旧 IQ，权限匹配扩展另列。

## 3. MILKR 的 B/C 到底怎样衔接

冻结 PAMF 表征 φ_s 后：

\[
M_B=\operatorname{MILKR}(\phi_s(S_o),Y_o),\qquad
M_C=\operatorname{MILKR}(\phi_s(S_o\cup S_n),Y_{o\cup n}).
\]

B 的域适应是目标标签驱动的判决／核状态适应，不更新 encoder，也没有显式源目标分布对齐损失。C 使用同一套分支交互、support 带宽、中心化、trace 匹配和 λ=1 解析规则，但重新估计全部当前状态，不继承 B 的 α、τ、中心化均值或 trace 尺度。源码证据：[fit_branch_local_ridge](../code/cvsrffi/d92_branch_local_ridge.py)；原设计已明确 B/C 独立拟合。

因此，在相同 φ_s、最终 S_o/S_n、类别名册、物理排序和配置下，跳过 B 后直接做 C 的最终预测，应与 B→C 最终预测相同，至数值容差和既定 tie-break。这是方法的一致性性质，不是失败，也不能用不等价数据人为制造 B 对终态的增益。

可以主张“同一机制服务旧类适应与新类扩展”“无梯度、无 source runtime 输入的任务升级”；不能据此主张“B 的参数在 C 累积优化”“B 必然提高最终 C”“编码器冻结即旧类永不下降”。C 重拟合与新类竞争仍可改变旧类判决。0 optimizer step 不等于无需拟合，完整重拟合不等于常数时间追加或无需保留旧 support。

单看 B−A 也不能证明学习到了更域不变的特征：A 默认头到 B MILKR 同时改变了读出与目标监督。应结合共享表征控制解释。

## 4. 三类合法衔接方式

|类型|接法|保留什么|怎样命名|
|---|---|---|---|
|原生可扩展表示／memory|ProtoNet B→追加原型；RadioNet B→追加 kNN；LoRa B→追加 enrollment|自己的 encoder、旧 target 决策状态、统一距离／投票规则|保留原方法，C 明确为原生行为或项目扩展|
|增量方法补 B|source-only→合法旧 support B→自己的 native C|自己的架构、源正交／隔离结构、Fisher／原型定义、C 算法|CSIL/MoPC/Orthogonal/ISSL 的 B 匹配扩展明示|
|闭集 DA 补 C|自己的原生 B→原生判决结构扩展；另接通用 NCM／JointCE|结构扩展保留 B 决策；通用控制明确更换读出，使用相同新旧 support|例如 DADDA-SDA→扩头 FT，不能叫 DADDA 原生 CIL|

不用所有 DA×所有 CIL 的笛卡尔积。RadioNet encoder 不自动具有 CSIL 的 projection/fingerprint 或 Orthogonal 的源正交几何；强拼后的模型是新方法，也会混入不同源训练与输入权限。

### 4.1 已有原生注册方法的衔接表

|方法|B|C 必须绑定的状态／注意事项|
|---|---|---|
|ProtoNet|自己的冻结 episode 表示，目标旧原型|保留目标旧原型，追加新原型；不能混入另一个 encoder 的坐标|
|RadioNet ADA＋FT＋kNN|自己的完整 ADA／FT 和旧 target memory|复制实际 B encoder 与旧 memory，再编码新 support；不是回退源 encoder|
|LoRa triplet＋KNN|自己的表示与旧 target enrollment|冻结表示、追加新 enrollment；保持 KNN 小样本规则与 memory 成本|
|CSIL corefix|声明的旧 support SFT|C 的参考／KD 对象按实现绑定 B；源 Fisher 保持合法预计算定义，不擅自用 target 重估为“源 Fisher”|
|MoPC-HR|声明的 B10 与原型校正|B 改变了表示时，旧原型需按其已声明机制校正；参考模型、原型和多 session 状态向后传递|
|Orthogonal SEI|自身源正交结构＋旧权重校准|保留 B 旧权重、冻结规则与新权重初始化；不能从普通 PAMF 头冒充|
|ISSL fixed_v1＋B10|自己的 base 经合法 B10|从实际 B 建立 C 所需 teacher/key/queue 状态；不加载未适应的另一 base；replay 输入明示|

已有管线证据：[pipeline.py](../comparison_suite/pipeline.py) 的 B/C 分叉、[registration.py](../comparison_suite/registration.py) 的 prepare/register、[adaptation.py](../comparison_suite/adaptation.py) 的状态与逐样本 score。源码证明部分组合结构已存在，不能据此宣称所有当前接口／权重已验证。

## 5. 新增 T2-J：完整 Phase2 串联系统主比较

原 T1 和 T2 保留为能力与分机制表。T2-J 专门比较所有阶段合法可用的完整系统：

|行族|数量|B|C|当前状态|
|---|---:|---|---|---|
|PAMF＋MILKR|1|原 MILKR 旧类拟合|原 MILKR 新旧联合重拟合|冻结方法；当前 PAMF 导出兼容仍待实施|
|已有注册系统|7|各自明确的合法 B|CSIL、MoPC、Orthogonal、ISSL、LoRa、ProtoNet、RadioNet 各自 C|保留原生机制及扩展标识|
|闭集 DA→原生结构扩展|7|下表各自原生 B|六种神经头扩展微调；Tweak 追加类校准|优先结构保持对照，新组合尚未实现|
|闭集 DA→NCM|7|下表各自原生 B|冻结 B 表示，目标新旧原型联合判决|新组合设计，尚未实现 runner|
|闭集 DA→JointCE|7|同一实际 B 复用|冻结 B 表示，统一新旧 CE 头|新组合设计，尚未实现 runner|

T2-J 设计 **29 条完整链路槽位**：1＋7＋7＋7＋7。其中原生结构扩展是防止丢掉 B 决策收益的重要对照，不能只用两个替换头后缀代表各 DA 方法的全部能力。29 条不是 29 篇论文，也不改变已有“14 RFFI 家族／22 原代码版本”统计。闭集未扩展原方法的 C 仍为 N/A；接口尚未合法绑定者报告缺项，不补造性能。

主文固定展示 **15 行**：PAMF 1 行＋已有注册系统 7 行＋原生结构保持扩展 7 行；14 个 NCM／JointCE 统一读出组合放完整附表。这个分工在结果前固定，不能为每个前缀按 query 分数挑最有利后缀。原 22 代码版本的版本对照仍单独保留。

|闭集 DA 前缀|保留原生结构的组合|统一控制 1|统一控制 2|
|---|---|---|---|
|POSTER FT|保留 Dense，扩最后输出层后 FT|POSTER FT→NCM|POSTER FT→JointCE|
|MRIOR-SDA|保留 B 分类网络，扩最后类输出后 FT|MRIOR-SDA→NCM|MRIOR-SDA→JointCE|
|DADDA-SDA|保留 B 分类网络，扩最后类输出后 FT|DADDA-SDA→NCM|DADDA-SDA→JointCE|
|Two-stage SDA＋FT|保留 B 分类网络，扩最后类输出后 FT|Two-stage→NCM|Two-stage→JointCE|
|Hu2024 Feature Separation|保留 B TX 分类路径，扩 TX 输出后 FT，RX 路径仍冻结|Hu2024→NCM|Hu2024→JointCE|
|RA-Collab 单节点 FT|保留 B 单节点分类器，扩类输出后 FT|RA-FT→NCM|RA-FT→JointCE|
|Tweak M1|保留 B 旧 centroid/radius，追加新类校准|Tweak-M1→NCM|Tweak-M1→JointCE|

各前缀使用自己的源网络和 B；不能只给弱方法接简单 NCM，再把 PAMF 的复杂局部核优势全部归因于“统一任务”。三种预定扩展都报告，不根据 query 结果替每个前缀选择更好的一条。统一后缀不再保留其原生决策，与结构保持版本不能混名。

### 5.1 保留 B 原生判决的扩展

六个神经分类器保留 B 的全部前置层与 buffer，在最后可扩展的类输出层复制旧类权重／bias，增加新类输出；冻结前置表示与 buffer，只更新扩展后的输出权重／bias。新类权重方向由其 B 坐标下 support 均值确定，norm 取 B 旧权重范数中位数，新 bias=0；零均值行置零。保持原生 head 的距离／cosine 温度等固定定义。所有新旧 support 等权、full-batch CE、Adam 0.001、100 更新、weight_decay=0；这是明确项目扩展预算。若某 head 不是兼容的类输出结构，须先明确映射或如实记缺项，不能把更换头伪装成继承头。

Tweak 直接追加新类 centroid 与平均 L2 radius，保留 B 旧校准；所有注册类一起调用原 closed-set 判决。现有 `calibration.py` 在 K1 时 radius=0，所有类别均为 K1 时退化为最近 centroid，不根据 query 修改半径；M=1 不获得多包组或真实场景路由。它是自然的类校准扩展，不冒称原论文正式 CIL 实验。[原校准代码](../paper_reproduction/gaskin_tweak_2023/calibration.py)

### 5.2 NCM 后缀的固定定义

取每个模型实际 B 后的已声明原生向量表示 z_B，所有新旧 support 均重新用该表示编码；不使用 B 前缓存。逐样本归一化 u=z/max(||z||,1e−12)，逐类求均值并归一化。C 使用 cosine 分数面向全部已注册类判决，无 query 统计、无优化器。距离／归一化统一，不按模型或场景目标分数调参。

它保留 B 表示，但更换 B 分类头／核／半径决策。尤其 POSTER FT 若只训练最后层，B 的变化可能被 C 的读出替换部分移除；如只变最终头则可能完全消失。这是此组合的机制边界，不等于原 DA 无效；用 A/B/C 与共享表征分析解释。

### 5.3 JointCE 后缀的固定定义

同样冻结 B 后的表示与所有 buffer，以相同 u 建立一个全部 C 类共享的线性头。所有类用其目标 support 均值的归一化向量初始化，bias=0；零均值行置零，tie-break 固定物理类别顺序。全部新旧 support 等权、full-batch CE、Adam lr=0.001、100 更新、weight_decay=0，无 query 早停。这是预定项目控制预算，不称原论文默认。

该版本替换而非继承原生旧分类头，目的是在每种 B 表示上提供同等、非零梯度的联合读出。结构保持扩展单独列在上一节；异构头不能强行复制权重。不得把本 JointCE 隐写成原生 head expansion。

所有组合 C 的可见目标标签仍仅是当前 S_o＋S_n；不增加 source replay、不增加 target U、不用场景／旧新 query 真身份路由。原方法 B 允许的 source 输入继续披露并计成本；“C 不回放 source”不等于整个组合 source-free。

这里的等权是每条 support 等权；各类 K 相同时各类等权，不是 old 组与 new 组各占一半。新 20 类时两组样本数量不同，主指标 H 与组间损伤必须另报；不能悄悄改组权重后继续称同一目标。

## 6. 如何分离系统、表示与读出收益

|证据层|固定条件|要比较什么|可以说明什么|
|---|---|---|---|
|完整系统|相同源角色、received、K、标签、阶段访问|PAMF＋MILKR 与 T2-J 全部合理组合|系统部署表现；不单独归因于 MILKR|
|共享 PAMF 表示|同一冻结 PAMF、S_o/S_n、query|NCM、kNN、线性 ridge、JointCE、BranchRidge、BranchInteraction、MILKR|目标读出／核机制贡献|
|相同注册后缀|各自 B，C 同 NCM 或 JointCE|不同 DA 表示之间；结合 A/B/C|B 后表示与源学习共同影响；不是纯 DA 因果效应|
|相同源模型，B 开／关|同一 F_s，最终 C 都获得相同 S_o＋S_n，使用同一后缀|A 表示→C 与 B 表示→C|B 表示改变对终态注册的条件影响|
|目标权限匹配|披露 source/U_s/replay/history，source-free 子表单列|PAMF-Sup＋MILKR 的已有合规权重桥接；全 PAMF 与额外 U_s 使用分列|区分标签／数据与方法效应|

可检验的机制假设是：固定多分支表征避免将 encoder 只针对少量旧类重新塑形；局部交互读出可由当前目标 support 调整决策，而新类仍使用同一表示和目标。证据应来自 B收益与 C_new/H 的共同变化、共享 PAMF 消融，以及真实时延。它不是“冻结一定更好”的定理；少样本域偏移过强时固定表示也可能不足，局部核仍会过拟合或发生旧新竞争。

外部 encoder 没有 t/f/PA 五块时，不能直接给它接完整 MILKR 并补零后继续同名。可以另定义单 embedding 的 local ridge，或将合法 DA 机制移植到 PAMF 后接 MILKR，但都属于明确的新控制，不是现有原版 MILKR 的直接跨骨干移植。

## 7. 无 B、直接 C 与可服务性

需要区分两个直接注册对照：

1. **数据匹配的直接 C**：不进行中间 B 更新，但到 C 仍提供相同 S_o＋S_n，且使用相同注册规则、初始化和预算。用于检验 B 的表示适应是否改变终态。
2. **原生 A→C**：按原增量算法只给其原生允许的新增输入／历史状态。它可能没有使用 S_o，标签利用量不同，不能把与 B→C 的差全部归为 B 的优化轨迹。

对冻结 encoder＋全 support 重拟合的 MILKR，两条“有 B／无 B但最终数据相同”的 C 应相同。ProtoNet 或某些 memory 追加方法也可能终态等价，因此 PAMF 不能靠这一点自称独有。

B 的实际价值是新类尚未到来时已能提高旧类服务表现；不能让“直接 C”对照提前看到 S_n。展示 A→B→C 时间轴，并同时画 B 阶段旧类准确率和最终 C 新旧平衡。若定义整个服务周期平均效用，阶段持续时间／权重须事前给定且有任务依据；没有真实时长时不把任意加权均值作为主指标。

## 8. 证明优势需要哪些指标与图表

完整链路主表采用：

`系统 | B规则 | C规则 | source/history权限 | A_old | B_old | C_old | C_new | H | ΔDA | Dreg | Gretain | B/C成本 | 状态`

其中 ΔDA=B_old−A_old，Dreg=B_old−C_old，Gretain=C_old−A_old；三者用相同旧物理 query，均为有符号百分点差。Gretain=ΔDA−Dreg，说明适应增益注册后还剩多少。优先报告差值，不用易因小分母失真的“增益保持率”。H=2C_old C_new/(C_old+C_new)，逐 cell 后平均，分母零时为零。

Gretain 只评价旧类，要与 C_new/H 同时看；冻结旧头但无法接纳新类不能靠旧类分高获胜。H 也不能替代 Macro-F1、类下限、不同新增类数与实际成本。

|展示|横纵轴／内容|证据目的|
|---|---|---|
|A→B→C 阶段图|相同旧 query 的 A_old、B_old、C_old；C_new 单独曲线|同时看到适应收益与注册损伤|
|新旧 Pareto 图|x=C_old，y=C_new；分 K／新增类数面板|右上方向较优；不能只挑一个好看的综合分|
|适应与终态平衡图|x=ΔDA，y=H；标出 A/B 绝对值|B 提升大是否以新类转移能力为代价|
|注册规模图|N_new0/2/5/10/20；C_old/C_new/H 三项|是否只在少量新类时有效|
|成本 Pareto 图|x=实测适应／注册成本，y=H，标注内存／source权限|检验精度与部署代价的共同优势|
|四场景条件面板|G10 八套＋D-LS／D-MU／D-LU|不能将混合场景结论直接推广到三个单景|

### 8.1 不把所有旧类下降都叫遗忘

固定全部 C 分数后，独立 scorer 计算 C_old,restricted：仅离线屏蔽新类分数列，在旧 query 上评价。正式预测始终面对全部注册类，不按 query 旧新身份缩小候选。

\[
B_{old}-C_{old,all}=(B_{old}-C_{old,restricted})+
(C_{old,restricted}-C_{old,all}).
\]

第一项为旧类内部判决变化，不专指 encoder 遗忘；第二项为新类竞争损失，统一 tie-break 下非负。MILKR encoder 冻结，第一项仍可能因带宽／中心化／系数重拟合而改变。保存 B对→C错、B错→C对与 old→new/new→old 混淆，避免净变化掩盖伤害和纠错抵消。

## 9. 预定研究问题与可否支持的结论

|问题|支持证据|不能越界的表述|
|---|---|---|
|Q1：旧类适应是否有效|A/B 绝对值与配对 ΔDA，各 K／场景|B−A 不单独证明域不变表示或消除信道|
|Q2：能否接纳新类并保留旧类|C_new/H、Dreg、Gretain、最差类、全部类别竞争|不靠 closed DA 的 C=N/A 或新类 0 分证明胜利|
|Q3：是否优于合理串联|T2-J 原生结构扩展、NCM 和 JointCE 三类组合、原生增量+B|不只对比没有注册能力的原方法，或丢掉 B 原头的替换版|
|Q4：收益是否来自 MILKR|共享 PAMF 的读出／交互／局部核控制|跨不同源网络的差值不能全归给 MILKR|
|Q5：是否实用|完整原生预算、标签数、raw/memory/prototype、source依赖、实测时延|0 optimizer step 不等于零训练成本或所有方法中必然最快|
|Q6：两阶段是否参数协同|数据匹配的 B开/关、路径状态记录|原 MILKR 无 B→C 核状态继承，不能预设协同终态收益|

若结果是 PAMF 的 B 不最高、C_new/H 更好，应称“适应与注册平衡较好”；若 H 接近、成本低，可称“精度接近而部署成本较低”；若只在 K5/20 或某些场景有效，应限定条件。只有实测支持时才说全面优于，不因设计目标预填结论。

## 10. 对总表与工作量的修订

保留原 14 家族／22 代码版本、T1/T2、全部原生机制及三单景。新增 T2-J 29 条完整链路槽位、共享表示中的 JointCE 控制、数据匹配直接 C 对照、Gretain 和旧类下降分解。当前新增组合尚未实现，旧 289520 fit row／294140 阶段槽位仅是原 15 系统基础矩阵，不包含它们。

7 个闭集 B 复用，不重新训练 source、不重新构建 target 数据、不重复 B。每个新增 C 后缀完整核心有 24640 个正新增 C row；14 个统一后缀另增 344960，7 个原生结构后缀另增 172480，合计 **517440 个 C row／阶段预测槽位**。若完整覆盖，则基础＋衔接表共 **806960 个 B/C 逻辑 row、811580 个阶段预测槽位**；源模型绑定仍至多 60 个，物理数据仍 132132 条。计数不含其他共享控制、直接 C 切片、消融、分次注册或补充 G20/G25；未接通项保留缺口，不按槽位数声称实际完成。

先在固定 source／合法 support 的最小同 row 切片验证接口、坐标、类别映射和预算，再按预定矩阵执行；不依据小型 target 分数删组合或选择后缀。完整扩展需要单独登记执行身份和成本估计，本次不启动、不制造新的发布审批。

主表位置：[全面总实验表](PAMF_PHASE2_TOTAL_EXPERIMENT_TABLE_20261028.md)；[机器设计](../configs/pamf_phase2_milkr_experiment_design_20261028.json)。
