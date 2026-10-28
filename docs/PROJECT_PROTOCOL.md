# CVS项目场景与数据协议

## 本轮 Phase2 纳入冻结 DURA 的双输入路径设计（2026-10-28，用户最新补充）

本轮总表增加 P0（原固定 received IQ）和 P1（同一 received 经冻结 DURA 后的估计恢复视图），所有方法均有匹配路径，A/B/C 的 support 与 query 始终在同一路径。原始 received 不重生成；恢复视图不是新增物理观测，不增加 K 或独立数据套数。只能从当前观测计算，不获准读取真实 clean、模拟器真值、源样本或 query 标签。query 恢复逐窗口固定推理，不更新 DURA、BN 或跨 query 统计。

DURA 候选固定为 v1.0/R69 stage2、Graph-RR、FP32/TF32off 的既定版本。Phase1 证据不自动证明 Phase2 资格；其源物理角色、旧新 TX、RX/day、教师祖先、选择背景及表示须与本轮契约核对，未核实前不宣称可加载。若不合规，明确 P1 缺项，不静默换权重或新增训练。本轮不微调 DURA，不改变当前源识别器或恢复暂停任务；源训练也使用 DURA 的匹配方案仅列后续扩展。此条覆盖此前本轮 Phase2“不新增 DURA”的设计表述，不追溯修改 Phase1 历史路线。详见 `docs/PAMF_PHASE2_TOTAL_EXPERIMENT_TABLE_20261028.md` r05。

## 本轮 Phase2 完整链路比较与声明边界（2026-10-28）

域适应与新类注册的完整比较允许在已声明的闭集 DA 的实际 B 表示上接明确命名的 target-support 注册控制；这属于项目组合扩展，不追认为原论文原生 C。各方法在 C 只能使用自身已允许的当前旧／新 support 和历史状态；新增组合不获得额外 source、target U 或 query 拟合权限。未扩展闭集方法的 C 记 N/A，不填 0 分来证明联合方法优越。

原 MILKR 使用冻结 PAMF，在 B 对旧 support 拟合，在 C 对新旧 support 重新独立拟合；它统一算法但不继承 B 核参数。相同最终输入下，直接 C 与 B 后 C 的终态应一致；不得称为已证实的两阶段参数协同或旧类永不下降。正式 C 始终面对全部已注册类别，旧类候选集限制只作预测固定后的离线损伤分解。衔接设计见 `docs/PAMF_PHASE2_DA_REGISTRATION_LINKAGE_20261028.md`，本次不启动实验。

## 本轮独立单场景固定为三组（2026-10-28，用户最新指定）

本轮 PAMF Phase2 独立单场景仅为低仰角郊区 `practical_low_suburban`、中仰角城区 `practical_mid_urban`、低仰角城区 `practical_low_urban` 三组，覆盖此前六个单场景诊断的安排。每组全部目标 support/query 使用该组同一种场景，组内各物理记录仍有自己的固定随机信道实现；不是将三景按 1/3 混入一个任务。原生仰角范围分别为 10°–30°、20°–45°、10°–30°。

综合主场景仍从六种 practical 分量抽取并固定 8 套；三组独立单场景不改变综合主概率。历史 MILKR 已完成的 high suburban／mid suburban／low urban 三景及原 received IQ 保留原身份，不改名或重生成。所有新数据组物理 ID 互斥、单记录单观测、相同源训练及 S→A→B→C 目标访问边界保持。本次只更新设计，不启动实验。完整总表见 `docs/PAMF_PHASE2_TOTAL_EXPERIMENT_TABLE_20261028.md`。

## 本轮应用为单星被动监听（2026-10-28，用户进一步明确）

PAMF本轮目标应用为单颗卫星被动监听，不控制地面发射机，不选择服务卫星。仰角采样应区分空间覆盖、单星过境时间、天线视场及已采集IQ总体；不得直接借用通信星座最高仰角选星或主动发射调度形成的概率。未声明接收视场／检测筛选时，中高仰角主导不能作为默认事实。

既有600km、10°–80°球面面积概率保留为明确的宽视场空间几何参考，不解释为实际载荷频率。任务实测概率取决于轨道／姿态、TX位置与活动、收发天线、采集过程；改变权重或接收门限不得依据识别成绩。条件于有效观测的识别性能与所有监听机会上的检测／采集／识别性能分开。单物理样本单received观测、target访问时点及truth-last保持；本次说明不修改概率数值、不启动实验，也不增加发布审批或数据重验证门槛。定量分析见 `docs/analysis/pamf_passive_elevation_20261028/REPORT.md`。

## 本轮 PAMF Phase2 使用 practical residual 与分阶段目标访问（2026-10-28）

用户本轮明确：星地信道使用 **practical residual**；所有域适应对比方法在目标适应前，均与 PAMF 使用同一套训练源域。此条覆盖下文将本轮 Phase2 场景限定为三种 `leo_weak` 的旧默认；不改写历史 received IQ、场景名称或已完成结果。

本轮允许的 practical 标识为 `practical_high`、`practical_mid`、`practical_low_suburban`、`practical_high_urban`、`practical_mid_urban`、`practical_low_urban`，实际场景矩阵在本轮设计／run 中固定。执行路径固定 `processing_route=residual`、`mode=post_sync`、`equalization_enabled=false`、`fs_hz=25e6`，保留输入的真实预处理身份。单物理样本只绑定一份固定 received IQ，跨场景物理 ID 互斥，support/query 互斥；不能以同一物理 IQ 的六次叠加冒充六个独立观测。

本轮主评估采用综合 practical residual：每条物理样本按事前固定概率抽取一个场景并处理一次；多套综合数据使用互不相交的物理记录，且与单场景诊断物理ID互斥。概率来自明确的参考几何与环境先验，不能依据目标识别结果设定或修改。原生low/mid仰角区间有重叠时，综合采样可采用其支持范围内的互斥分档和外部geometry，必须保留实际采样规则；内部衰落／接收机／residual处理公式不变。实际套数、概率、seed和矩阵属于本轮实验设计，不写成真实部署频率。

地面源训练 S 阶段只允许匹配 PAMF 的源接收机、源日期、旧 TX、L_s/U_s/V 物理角色。L_s 的 TX 标签可见，U_s 的 TX 始终隐藏，V 只验证，不更新持久状态；不同方法可保留原生 source-only 目标，是否使用 U_s、增强和计算成本分别披露。禁止在 S 阶段或源 checkpoint 祖先中用目标 support、目标无标签 IQ、目标统计／BN／原型／伪标签、query 或目标指标训练、校准、早停和选模。外部方法的 source/history 权限例外不豁免这一时间边界，不能照搬原论文提前混入 target 的默认训练器。

A 是源模型冻结后的目标 query 只读评估，不产生适应状态，评分不反馈选模；B 开放当前 row 的旧类有标签 target support，原生 UDA 可仅取该合法 support 的 IQ 做域对齐，不增设 query 无标签池；C 才加入合法新类有标签 support。原论文需 source replay、Fisher 或历史原型的外部方法可在 B/C 使用各自明确的合法源状态，但其产生必须在 source-only 边界内，且不能将该例外扩展给 MILKR 主方法。所有阶段的 query 仅逐样本测试，全部预登记预测固定后独立 truth-last 评分。源阶段派生的类原型／enrollment 仅由 L_s 生成并冻结；不能借用目标 support 构造 A。

本条只固定本轮科学输入、阶段权限与声明边界，不启动实验、不改变其他 owner 或历史固定矩阵；未变化且已 `VALIDATED_ONCE` 的数据直接复用。新增场景若需要新的物理观测，只验证实际改变的数据事实，不因方法或文档修改重复建库。

## 星地测试与普通PL的当前训练约定（2026-10-19）

用户最新要求：凡正式比较包含星地信道测试，训练须使用默认clean＋satellite拼接式星地增强。full与自身CE使用相同增强；卫星视图仅承载TX交叉熵，权重0.68，不附加星地一致性损失。本条覆盖此前“关闭训练增强仍默认进行星地测试”的后续正式比较约定；既有产物保留并标为无增强历史对照，不追溯改写。

本轮外部方法统一从首个训练batch启用拼接辅助CE，保留原生最大预算、源V选模和早停；这是用户授权自行判断的日程适配，须与Core90历史E80启用区分。默认LEO_WEAK场景及分阶段视图概率继承4.3节，并按固定最大训练预算的相对进度缩放；当前默认测试仍为clean及六种practical，不恢复旧的三LEO最终测试要求。

用户要求涉及半监督时仅使用普通置信度PL，并保留监督对照。U_s只提供源域IQ，不导出TX真值；当前联合训练模型以不更新持久状态的前向生成硬伪标签，超过固定阈值才计入伪标签CE。不使用EMA教师、近邻批次门控、强一致性或另训分类头。阈值、权重、无标签batch比例及实际接受率属于各run预登记；不得根据目标分数更改。源/目标物理ID、V权限、默认头和truth-last规则保持。

用户已授权撤下尚未开始GPU训练的旧扩展部分并重新发布；正在训练的行和已完成产物保留。具体run、矩阵及资源预算在独立执行报告登记，不写入本协议。

## 用户指定全标签 80/20 监督参考（2026-10-12）

用户明确授权单独的全标签监督参考：保持当前ManySig源RX索引1/3/4/6/8、源day1/2/3和全部6TX，源池按TX/RX/day/eq内sig_i排序，前80%全部有标签训练，后20%仅验证；L_s/U_s/V=0.8/0/0.2。该诊断例外不替代默认弱标注7/63/30协议，不解封任何旧run的U标签。源角色与目标物理ID必须不交，目标不参与训练或选模，prediction全部固定后独立truth-last评分。改变划分必须从零训练；不得继承旧权重。结果只作为固定训练预算下的全标签参考，不构成伪标签算法的数学上限；与弱标注比较披露划分和曝光差异。


## 用户指定历史 ADV3B02 划分比较（2026-10-12）

本次用户明确要求当前 E20 纯 CE＋网络内部门控＋cos 使用历史 `ADV3B02_CORE90_SOFT_E200` 的数据配置和 seed392002，并明确只测 clean。该单次历史比较沿用 ManySig 的全部 6 个 TX，源 RX 索引0至6、源 day0/1，按每个 TX/RX/day/eq 内 sig_i 排序，前10%为 L_s、中间70%为 U_s、末20%为 V，数量8400/58800/16800。实际训练池标注率为12.5%，作为用户指定的历史协议例外单独披露，不替代当前统一7/63/30和训练池10%标注协议。

固定 E20 后自动 clean 测试三个互不相交集合：未见 day2/3×已见 RX0至6（84000），已见 day0/1×未见 RX7至11（60000），未见 day2/3×未见 RX7至11（60000）。三个集合均与全部源角色物理ID不交；全部预测固定后由独立 scorer 连接 truth。训练从零开始，不继承当前划分 E20 或历史 ADV3B02 权重；U_s标签隐藏，V只监测，不以目标成绩选模或重跑。该已暴露历史基准比较不声明新的独立盲测；训练成本、损失和架构差异分别披露。

## 后续探索实验默认纯CE基线（2026-10-10）

- 用户为加快训练，要求后续探索实验基线只使用纯CE，关闭半监督和星地信道增强。新探索基线保留当前身份网络结构及分类头，只以合法有标签源域L_s的clean IQ计算身份交叉熵；label_smoothing=0。只使用L_s不等于解封U_s标签或扩大有标签数据比例，V仅验证，数据契约保持。
- 基线关闭U_s训练、伪标签、EMA教师及教师前向/更新、伪标签置信度/邻近批次/强增强一致门控、无标签熵损失和130+70半监督阶段；关闭卫星视图生成、clean+satellite拼接及其CE/一致性损失。基线不默认附加其他随机增强、多视图前向、GRL、正交、互信息、原型、域损失或L/T/R辅助网络。
- 保留身份网络中的时频/PA/参考响应结构、分类头和cosine学习率日程；优化器与weight decay属于优化设置，按本次配置显式记录。纯CE指损失目标，不要求改成线性分类器或删除身份网络内部融合。新的待研究机制只加入对应变体，和同训练预算的纯CE基线对照。
- 新探索launcher必须实际关闭无用教师、无标签迭代、增强生成和辅助前向，不能仅把loss权重设零后仍执行昂贵分支。日志及resolved config核对实际执行状态。未指定的新epoch/seed预算不自行缩减；速度收益需实测，不预先承诺倍数。
- 此项是探索实验训练默认的明确例外，覆盖历史默认半监督/星地拼接配方；当前健康任务和历史实验保持原配置，不重新训练。训练关闭信道增强不改变测试：仍按既有约定自动测试clean及六种practical，prediction固定后独立truth-last评分。

版本：2026-09-08
协议模式：`p2_min_v1`

## 1. 文件职责

所有CVS、外部对比、联邦、诊断与独立评估均使用[实验管理skill](../.agents/skills/experiment-management/SKILL.md)及[统一登记规范](EXPERIMENT_MANAGEMENT.md)。每次启动的名称、说明、实际数据契约、seed角色、矩阵、模型来源及存储位置保存在对应run记录，通过[实验总索引](../experiment_registry/README.md)查找；本文件继续只定义科学语义，不保存活动seed清单或性能排名。登记是既有预登记的一部分，不新增数据验证或审批，也不改变外部对比方法的显式权限例外。

本文件只定义CVS-RFFI/CV-SincNet的科学场景、数据集合、数据生成、Phase1/Phase2/Phase3边界、Stage2-A/B/C权限、Phase3部署期协同输入及可声明范围。

本文件不保存活动性能目标、当前候选方法、实验矩阵、seed清单、epoch/参数/显存上限、优化路线、N607操作、Git流程或某次实验结论。这些内容分别属于独立目标文档、方法设计、实验报告和`AGENTS.md`。

协议的稳定Git镜像相对路径为[PROJECT_PROTOCOL.md](PROJECT_PROTOCOL.md)；一次性Phase2数据builder/validator实现边界见[PHASE2_DATA_VALIDATION_APPENDIX.md](PHASE2_DATA_VALIDATION_APPENDIX.md)。`项目.md`始终定义科学语义，临时隔离worktree不另行成为协议权威。当前Phase1修订的隔离Git分支为`codex/phase1-mirage-owdg-20260817`，当前核验提交为`ab047abc`，其中实现提交为`50cf8fde`。最终集成前，主检出中的上述Git镜像可能仍是旧文本；只有最终集成后，匹配的Git正文才在主检出可见。本文件定义语义，后两份文档分别承担Git承载和实现说明，不额外创造数据权限。

## 2. 项目场景

CVS的主场景是：

> 天基射频指纹识别中的地面弱标注跨接收机域泛化、目标接收机域少样本适应与新类注册，以及部署阶段多接收节点协同的未知拒识、匿名实体关联和可信确权。

项目采用“地面训练、星上部署”。Phase1在地面学习开放世界就绪的发射机身份表征并封存checkpoint及类原型；Phase2只使用目标接收机已经接收到的LEO弱信道IQ、合法support标签、随checkpoint上传并冻结的类原型及类别映射、冻结checkpoint和预登记算法配置，完成旧类域适应与新类注册；Phase3发生在部署阶段，每个接收节点先形成冻结的本地身份、域、质量和拒识证据，再由协同推理完成unknown拒识、anonymous entity关联和可信确权。Phase3不替代Phase2，也不得把unknown query直接转成Stage2-C support。

Phase3是正式研究阶段，不是Phase2的同义改名。项目当前是否已经实现Phase3、是否完成真实在轨多星验证以及是否达到具体性能门槛，只能由对应Git实现、实验报告和完整artifact证明；阶段定义本身不构成完成声明。

WiSig/ManySig是地面代理数据，不是真实卫星数据；LEO弱信道叠加是物理启发的部署压力代理，不等价于真实在轨验证。

### 2.1 N607实验承载角色

N607是CVS-RFFI/CV-SincNet大规模训练、Phase2方法实验、125稳定性screen、独立确认矩阵和资源审计的主要计算与证据承载面。N607不是`R_s`或`R_t`中的接收机，不是source/target数据来源，不是卫星实体，也不构成clean/source访问、query拟合或其他Phase2协议的例外。所有代码与协议修改先在本地Git承载面完成并验证，再按`AGENTS.md`规定同步到N607；具体SSH、环境、GPU、launcher、日志和报告操作只由`AGENTS.md`及实验报告管理。

## 3. 观测模型与集合

接收IQ抽象为：

```text
x = R_d(H_d * T_y(s)) + n
```

- `T_y`：发射机硬件非理想性，是身份来源。
- `H_d`：传播或星地信道，是域扰动。
- `R_d`：接收机链路响应，是跨接收机偏移。
- `n`：噪声。

接收机集合：

```text
R_s = source training receivers
R_t = target receiver domain
R_t ∩ R_s = ∅
```

发射机集合：

```text
Y_old = Phase1已见发射机
Y_new = Phase1未见、Phase2注册发射机
Y_unknown = 未注册发射机
Y_old ∩ Y_new = ∅
Y_unknown ∩ (Y_old ∪ Y_new) = ∅
```

在Phase1出现过的TX在Phase2只能是旧类，不能重命名为新类。

## 4. Phase1地面开放世界就绪表征协议

Phase1是weak-label/semi-supervised source-domain DG，不是部署few-shot：

```text
L_s = {(x_i,y_i,d_i): receiver(x_i) ∈ R_s}
U_s = {(x_j,d_j): receiver(x_j) ∈ R_s, y_j hidden or unavailable}
rho_label = |L_s| / (|L_s| + |U_s|) ≤ 0.1
```

当前统一划分语义为相对source全池`L_s/U_s/V=0.07/0.63/0.30`：有TX标签训练集、无TX标签训练集和单一source validation。三个角色均不得包含`R_t`。`L_s`、`U_s`与`V`可以共享source已知TX身份，但物理样本ID在所有角色间必须两两不交。`V`可用于source侧校准、阈值冻结和checkpoint选择，但不得反向传播、更新EMA、prototype、normalization或其他持久状态；不得再把`V`拆成`V_cal/V_select`等方法角色。需要研究标注率时，可使用`rho_label∈{0.005,0.01,0.02,0.05,0.1}`，但不得改变集合含义。

Phase1可以使用source数据的clean与卫星信道增强视图训练；这不授予Phase2或Phase3部署期推理读取这些样本、样本级派生状态或训练期scorer结果的权限。Phase1具体模型、loss、训练轮数和选择规则属于方法/实验文档，不写入本协议。

### 4.1 Phase1职责与输出

Phase1只在地面训练，不执行多接收节点消息传递、anonymous track维护或真实运营身份确权。它优化底层射频特征提取器，使身份表征具备TX可分性、跨接收机稳定性、LEO弱信道鲁棒性、类内紧致和类间margin，并降低未见TX上的过度置信，为后续距离、能量、尾部分布或不确定性拒识提供可用几何。

Phase1可以重新训练Sinc或时域前端、频域及PA分支、卷积层、`z_id`身份表征、`z_dom`域扰动表征、normalization、projection、fusion以及prototype、radius、energy和不确定性输出。具体启用哪些组件由独立方法设计冻结，本协议不要求一个候选同时堆叠全部机制。

Phase1最终交付：

```text
开放世界就绪的特征提取器
已注册类基础几何
类别半径／能量／尾部分布先验
接收质量与域不确定性输出
不可变deployment bundle
```

这些输出只是Phase3本地证据提取的底层输入，不等于已经完成真实unknown拒识、多节点协同或可信确权。

### 4.2 Source-only proxy unknown研发边界

`proxy_train`只由`L_s`生成。它是相对当前episode注册类别表的source代理角色，而非真实未见TX；训练proxy可参与拒识相关反向传播。`U_s`不生成proxy，因为训练过程不可读取其TX真值。

validation proxy只由单一`V`生成，可用于source侧校准、阈值冻结和模型选择。validation proxy不得反向传播，不得更新EMA、prototype、normalization或其他持久状态。source proxy指标只能写作代理未知研发性能，不能替代真实target unknown性能。

target unknown TX身份与source训练/validation TX身份必须互斥。任何target角色，包括target-known与target unknown，均不得用于训练、校准、选模、候选重排（`CANDIDATE_RERANK`）或触发选择性重跑。模型、几何与阈值在target访问前冻结；预测artifact先封存，独立scorer之后才能连接truth。真实unknown结论只来自这一次性、role/truth-blind的target评估，不反馈研发。

target-known与target-unknown在连接角色或truth前，必须对每个物理样本采用相同的单物理样本单LEO weak观测规则，并使用相同预处理、模型前向和决策规则；不得依据known/unknown角色改变scene、seed、处理路径或阈值。

Phase1不得读取目标接收机query真值，不得使用Phase3后续确认的unknown作为训练数据，不得把source proxy unknown指标写成Phase3真实unknown结果，也不得输出真实运营身份或`registration_authorized`。上述内容完全冻结后，允许对预注册的单一候选开展一次性、role/truth-blind、零适配的`R_t`LEO弱信道确认性评测；registered与unknown必须同样遵守单物理样本单观测、scene/seed分配和预测先封存规则。独立scorer只能在预测封存后评分；结果仅可用于该冻结候选的预先声明确认记录或下一阶段准入，不得用于候选重排、阈值或模型调整、重训、重跑选择或任何反馈。该单节点确认也不构成Phase3运营unknown、协同、anonymous entity、可信确权或`registration_authorized`声明。

### 4.3 Phase1星地信道增强默认

自2026-08-21起，新建Phase1训练默认严格复用`ADV3B02_CORE90_SOFT_E200`的拼接式星地信道增强：每个clean训练批次生成一个卫星视图批次并执行`clean+satellite`拼接，卫星视图只承载TX交叉熵监督（`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`），不默认启用其他星地一致性损失或其他星地增强族。训练及其最终星地信道测试的默认场景族固定为`LEO_WEAK`：

```text
leo_clear_weak
leo_low_elev_weak
leo_rain_weak
```

默认视图日程为E1–40使用`leo_clear_weak,p=0.30`，E41–90使用`leo_low_elev_weak,leo_rain_weak,p=0.60`，E91–200使用三场景并集`p=0.80`；严格复用Core90时，卫星辅助CE从E80开始计入训练总损失。最终测试必须分别报告三种LEO弱场景；clean测试继续作为无星地增强的对照，不能替代任一LEO弱信道结果。`mixed_orbit`及其他星地场景仅保留为历史复现、已声明对照或诊断性压力测试，必须在命令和报告中显式指定，不能再作为默认训练或测试路径。该默认不追溯改写历史命令、checkpoint、日志或已启动实验。

#### 4.3.1 HCF-DG专用单前向例外

经用户于2026-08-30明确冻结，HCF-DG及其A0–A12 matched实验不继承本节的`clean+satellite`拼接双前向默认。HCF-DG在每个主训练batch进入共享身份主干前，固定将70%样本保持clean、30%样本替换为一次`mixed_orbit`接收视图；每个样本只保留一个训练输入位置，整个batch只执行一次共享身份主干前向。增强参数同时形成receiver/day/channel因子化中的channel标签。该例外只适用于显式`phase1_method=hcfdg`的独立入口，不改变ADV3B02、ADV3B03、历史复现或其他新Phase1方法的默认行为。

HCF-DG的clean/satellite成对一致性默认关闭；若后续独立消融启用，只允许每4个optimizer step在25%的batch样本上计算，并单独报告资源增量。无论训练使用何种`mixed_orbit`实现，最终checkpoint仍必须分别报告clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，不得用`mixed_orbit`聚合指标替代三个正式LEO弱场景。

### 4.4 Checkpoint训练数据一致性与继承污染禁令

本节约束正式CVS主方法的完整模型来源，覆盖Phase1初始化、恢复训练、教师蒸馏、特征提取、模型融合及Phase2/Phase3所用基座。数据权限约束整个产生流程，不能只检查本轮训练配置。旧报告中的“成熟基座”“固定基线”、默认checkpoint路径或历史可加载记录均不授予复用权限。

1. **禁止跨训练划分继承。**用于初始化或续训的checkpoint，其产生流程必须与本次预登记的Phase1数据契约一致：数据集及版本、raw/equalized与物理记录映射、source/target接收机与日期、TX集合及标签映射、L_s/U_s/V比例和各角色实际物理样本ID、划分算法及影响划分的seed。仅数据集文件名、split名称、样本数或比例相同，不构成一致。任一不一致即禁止用于该正式run；无污染但训练集不同，也不能作为匹配初始化或严格同条件基线。
2. **物理样本身份优先。**复制文件、改路径、重编号、换seed、重新归一化、切片或叠加新的LEO信道，不会消除底层物理IQ的历史归属。衍生视图必须追溯原始物理记录；不得把同一记录的不同视图放入互斥角色。只改变模型随机初始化seed且实际数据契约不变，不算数据划分改变。
3. **禁令追溯所有继承来源。**核对直接checkpoint及其全部上游预训练、resume、warm-start、teacher/EMA、蒸馏、集成与融合来源。部分加载骨干、丢弃分类头、冻结参数、只取embedding、重新训练adapter或重置optimizer，均不能解除既有污染。原型、归一化统计、阈值和其他随模型继承的持久状态同样受限；未知祖先不能默认视为从零训练。
4. **禁止历史目标暴露与测试选模。**任一上游使用当前target的物理样本（有标签或无标签）、统计量、伪标签，或其指标参与训练、校准、早停、checkpoint选择、候选排序、超参数调整及选择性重跑，该来源及其后代均不得用于当前干净泛化结论。仅检查“本轮不反向传播目标标签”不够。历史源域V若转为当前target，同样构成污染。任何测试集选模（包括含测试指标的joint_safe）均不满足source-only选模要求；即使旧测试集与当前target不同也不得作为合规来源。模型完全冻结后的独立最终评分本身不等于训练污染，但其结果不得再参与选择或反馈研发。
5. **缺证据即禁止使用。**加载前依据实际生效配置、数据索引/可复现划分、训练和选模记录核对来源，禁止只相信launcher默认值、文件名或口头声明。当前数据契约、每个继承来源及选模依据、差异和结论写入现有run报告即可，无需另建签名、哈希链或审批系统。无法确认实际物理ID或祖先来源时记为CHECKPOINT_PROVENANCE_UNVERIFIED；不一致记为CHECKPOINT_DATA_CONTRACT_MISMATCH；已证实目标参与训练或选择记为CHECKPOINT_TARGET_CONTAMINATED。三者均禁止该来源进入正式run，不得先跑后补证据。
6. **恢复路径与比较边界。**无合规checkpoint时，仅可在已有训练授权内按本次数据契约从零训练，并只以单一source V或预登记固定轮次选模；不能私自改变训练集、目标集或借用旧权重节省预算。共享初始化的消融使用同一合规checkpoint；从零训练的对比固定数据划分，披露训练随机性。公平比较还须披露预训练/续训总预算，不能把历史额外训练隐去。改变合法数据契约后须重新判断checkpoint适用性。
7. **不得追认或洗白污染结果。**发现问题后保留原checkpoint、日志和结果，在受影响报告中标明来源及后代范围；污染结果记为PROTOCOL_INVALID_FOR_CLEAN_GENERALIZATION，来源不明结果记为PROTOCOL_VALIDITY_UNVERIFIED，不进入正式有效结果汇总、晋级或干净泛化对比。产物完整不代表协议有效；共享污染源也不能抵消污染。既有运行的停机、删除、覆盖或重跑仍遵守原授权，不由本条自动授权。
8. **阶段权限不互相替代。**Phase2/Phase3继承的地面基座必须匹配预登记Phase1契约，而非要求其训练集等于target support；合法Stage2 support适配保持原权限，query始终不得参与拟合或选择。外部论文方法的显式权限例外不能用于给CVS主方法洗白，也不自动豁免训练/测试污染披露。只有用户明确另立跨数据预训练研究时，才可作为单独任务披露权限差异，不能标为本契约下的匹配source-only实验。

此项是现有输入权限检查的一部分。VALIDATED_ONCE只证明数据capsule自身已验证，不证明任意checkpoint均适用；换checkpoint只需核对其与现有契约及历史来源的兼容性，不触发未改变数据capsule的重复builder验证。相同来源与契约已有可复用核对结论时直接引用；来源、契约或新污染证据改变才重查受影响部分。

### Phase1 有标签源内部开发折（2026-10-15）

当预登记要求 V 只作健康诊断时，可在原 L_s 内固定物理样本互斥的训练/开发子集；不得把 U_s 或 V 移入训练，也不增加可见 TX 标签。划分算法、seed、实际物理 ID 和用途必须在首次训练前固定。L_dev 仅用于无更新的源强度校准、候选及组合选择，不参与 CE、伪标签、EMA、归一化或其他持久模型状态更新；训练仅使用 L_train，仍称外层 L_s 的内部开发协议，不改变外层 source/target 或固定测试 IQ。V 不再拆分。

该内部分折是实际训练数据契约的一部分。所有匹配消融使用相同 L_train/L_dev；完整 L_s 上训练的历史权重曾见过开发物理样本，不能作为该折的初始化、教师或严格匹配基线。新模型从零训练或从已核实相同内部分折来源继续；历史完整 L_s 模型可作明确标注的冻结操作诊断，不得把 L_dev 宣称为全历史未见盲测。正式 target 指标不参与内部折、强度、候选、组合或扩种子的选择。

### 4.3.2 Phase1最终测试默认更新（2026-10-10）

用户明确要求后续每次训练完成自动测试，星地环境只测六种practical。对后续新发布Phase1实验，最终测试默认采用clean对照加`practical_high`、`practical_mid`、`practical_low_suburban`、`practical_high_urban`、`practical_mid_urban`、`practical_low_urban`，覆盖4.3及4.3.1中默认必须追加三种LEO弱场景的最终测试要求。训练增强配置不因此改变，Phase2固定received IQ、support/query及单物理样本单观测规则保持。

固定对比row训练完成并按预登记规则冻结后，由发布队列自动接续测试；涉及跨候选源域选择时先完成必要source-only选择。逐row所有预登记视图预测固定后独立truth-last评分，测试结果不回流其他训练或选模。既有健康运行和历史实验保持各自已冻结计划；用户明确指定的测试范围优先。

### Phase1 源接收机未见日期的冻结诊断（2026-10-19）

跨日期诊断可以在已用于源训练的RX上，选择从未进入该模型L_s／U_s／V的日期进行冻结评估。该集合不定义为主任务的未见接收机R_t，须明确标为source-RX unseen-date diagnostic，与未见RX主测试分别报告。全部诊断物理ID与该模型全部源角色互斥，模型和默认头冻结后只预测，不校准、适配、更新状态或选模；本row全部预登记预测固定后独立truth-last评分。

改变训练日期或源接收机集合时使用独立数据契约及从零模型，不继承其他划分权重。此诊断不改变现有Phase1未见接收机主任务的R_s∩R_t=∅，也不改变Phase2输入权限或当前健康实验。

## 5. Phase2最小数据协议

### 5.1 单物理样本单LEO接收观测

每个clean/raw物理IQ记录在进入Phase2前只允许进行一次随机LEO弱信道叠加：

```text
received_i = H(c_i, seed_i)(clean_i)
c_i ∈ {leo_clear_weak, leo_low_elev_weak, leo_rain_weak}
```

对每个稳定`physical_sample_id`，只能绑定一个`c_i`、一个随机`seed_i`和一份固定`received_i`。禁止把同一clean/raw物理样本复制后分别叠加多种场景、多个随机信道实现或多个LEO状态，再作为多份support/query进入Phase2。

三种LEO场景用于评估不同接收条件；同一matched实验切片中，三个场景的物理样本ID集合必须两两不交。单场景内support与query的物理样本ID也必须不交。

### 5.2 K-shot与接收后计算view

`K-shot`表示每个已注册类别有K个互不重复的物理support样本。由同一固定接收IQ计算的均衡、裁剪、相位/幅度归一化、FFT或其他确定性/随机数学表征仍属于同一个物理样本，不增加K。

接收后计算view必须满足：

- 输入只能是固定、已封存的`received_i`；
- 不得调用LEO信道模拟器，不得恢复clean参考，不得产生第二个LEO观测；
- support view可以参与适配、注册和状态更新；
- query view只用于该query的逐样本推理，不得更新模型、原型、阈值、温度、门控、候选选择、早停或回滚状态。

### 5.3 Phase2允许输入

Phase2运行时采用穷尽式白名单，只允许读取：

1. `p2_min_v1`、`VALIDATED_ONCE`固定目标域LEO received IQ及其匹配的`capsule_id/split_id`；
2. 当前row合法target support标签、必要注册类别表和不含query真值/角色的split；
3. 地面预先计算、随checkpoint上传并在Phase2保持不可变的类原型、必要类别映射，以及符合5.3.2节的量化聚合Phase1分布摘要；
4. 冻结checkpoint和预登记算法配置。

除此之外，Phase2不得读取、构造或恢复任何地面source/clean样本、源域数据加载器、source replay、source cache、未按5.3.2节联合冻结的源域特征或统计量、样本级embedding、源域BatchNorm状态、伪源域样本、生成式源数据、可还原源样本的中间信息、dataset构建路径或能够影响决策的其他外部source状态。不得读取query真值、query角色、真实query batch类别集合/数量或其他query反馈。

#### 5.3.1 冻结类原型边界

白名单内的地面类原型只能作为不可训练的类别锚点和冻结判决依据。Phase2不得对其反向更新、在线重估、追加源域信息，或由类原型在线拟合D92式协方差、LDA、持久分类头、样本级记忆、类条件源域统计或其他可训练判决状态。符合5.3.2节的量化聚合Phase1分布摘要不是由Phase2扩展类原型得到的状态。类原型和该摘要均不增加K、不生成第二个LEO view，也不能替代Stage2-B/C的合法target support。缺少合规原型或摘要时，Phase2不得回读地面样本重建。

#### 5.3.2 量化聚合Phase1分布摘要

经2026-08-30用户明确授权，`p2_min_v1`允许一个可选的量化聚合Phase1分布摘要随checkpoint联合冻结并上传。该摘要只能在Phase1地面阶段由多个source物理样本或多个source接收域聚合产生，运行时保持不可训练、不可追加、不可在线重估。允许成员穷尽为：int8域×类聚合中心及有效槽mask，或其int8类中心、int8低秩域残差方向、int8域系数和int8类半径压缩形式，以及反量化所需的FP16尺度；如方法预登记确有需要，可增加不含BatchNorm运行状态的FP16全局逐特征location/scale。类别、域和特征维度映射必须随摘要冻结。

摘要不得包含source/clean IQ、样本级embedding、逐样本索引、可访问source数据的路径、源域BatchNorm状态、生成式源样本或能够恢复单个物理样本的信息。Phase2只能按预登记公式将摘要用作固定分布锚点、确定性虚拟特征点或normalization参考；不得对摘要反向传播，不得将反量化结果持久化为新的源特征库，也不得用query更新、选择或校准摘要。

该可选摘要不改变target received IQ、物理ID、receiver/TX集合、scenario、K或support/query划分，因此不使匹配的`VALIDATED_ONCE`数据capsule失效，也不触发数据重验证。未携带该摘要的既有`p2_min_v1`bundle仍然合法；需要该摘要的方法必须在运行前核对其与checkpoint、类别映射和特征schema一致。

### 5.4 query只用于测试

Phase2 predictor的状态只能由Phase1 bundle和注册support决定。每个query必须独立面对全部已注册类别；禁止访问或利用：

- query真值以及真实old/new/unknown角色；
- 真实query batch类别集合或各类数量；
- 每类query配额、标签排序或分块；
- Hungarian、optimal transport、global quota或其他跨query全局重排。

预测必须先形成不可变预测artifact；之后独立scorer才可按opaque query ID连接真值并计算指标。scorer输出不得回流到适配、注册、阈值、选择、排序或重跑决策。

### 5.5 一次验证、跨方法复用

Phase2数据builder完成唯一观测、物理ID互斥、support/query互斥、接收机/TX集合和禁止成员检查后，输出：

```text
protocol_schema = p2_min_v1
phase2_data_status = VALIDATED_ONCE
capsule_id = <content identity>
split_id = <receiver/TX/scenario/K/support-query identity>
single_leo_observation = true
clean_source_runtime_access = false
query_fit_access = false
query_decision_policy = per_sample_all_registered_classes
```

研发和评估流程只核对以上最小句柄并直接运行。只有以下数据事实变化时才重做数据验证：固定接收IQ字节、物理ID、receiver/TX集合、scenario分配、K、support/query划分或协议schema。

候选方法、adapter、超参数、epoch、原型更新规则、method lock、checkpoint推理状态、资源预算或报告格式变化，不得使数据capsule失效，也不得触发重新追溯clean/source构建过程。若验证失败，只修复直接失败的数据项；其他`VALIDATED_ONCE`切片继续实验。

hash、签名、allowlist、访问账本和pre-open检查由数据builder/validator一次性自动完成，详见Git承载面的`docs/PHASE2_DATA_VALIDATION_APPENDIX.md`。它们是实现证据，不在每个方法目标或实验报告中重复展开。

## 6. Stage2阶段权限

| 阶段 | 可用target信息 | 任务 | 不可声明 |
|---|---|---|---|
| Stage2-A | 无target TX标签；可有无标签LEO接收IQ | zero-label target-domain reference/diagnostic | 旧类few-shot适应、新类identity accuracy |
| Stage2-B | `Y_old`的K-shot target support标签 | 旧类目标域适应与校准 | 新类注册性能 |
| Stage2-C | `Y_old∪Y_new`的K-shot target support标签 | 在同一目标域同时完成旧类适应和新类注册 | 缺少任一侧时的完整Stage2-C成功 |
| Phase3 | 多接收节点冻结本地证据、未注册类观测及合法外部确权上下文 | 部署期unknown拒识、anonymous entity关联、可信确权与注册授权 | 直接用unknown query训练/更新Phase2，或以unknown结果替代旧类/新类结果 |

Stage2-C中的旧类适应与新类注册是同等任务；旧类注册前/后必须来自同一row、同一旧类query和同一推理规则。具体指标门槛与确认矩阵属于独立目标文档。

以上Phase2数据权限、query访问和逐样本决策约束只用于Stage2主方法及其内部候选的合法性与晋级判定，不限制为论文复现和性能比较而运行的外部对比方法。CSIL、MoPC-HR等对比方法可按原论文完整训练/增量流程访问其所需的base/source数据、历史统计、训练批次和评估流程，也不受Stage2主方法资源预算约束。正式对比方法默认必须继承的项目数据条件是：凡作为新类注册或新类评测输入的样本均须叠加并明确记录LEO星地信道；其他`p2_min_v1`限制不作为对比方法结果有效性的门禁。经用户显式要求，可以另跑仅用于归因的matched无LEO新类诊断：保持方法、物理样本ID、support/query划分、K-shot、seed和旧类评测条件不变，只把新类support/query换成同一物理记录的未叠加IQ。该结果必须标为`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`，与正式LEO结果隔离，不能用于CVS卫星场景性能声明、方法晋级或Stage2协议有效性证明。对比结果必须标明其数据与训练权限，不能反向用作Stage2主方法满足协议的证据。

floor是面向全部实际注册类的通用下界评价，不是预选难类清单。任何算法、loss、校准或更新规则必须对类标签置换保持同一形式；禁止按具体TX/class ID设置白名单、专属分支、专属权重、专属阈值或专属超参数。允许从各类合法support自适应估计半径、不确定度或权重，但所有类必须使用同一公式和同一超参数生成规则。

## 7. Phase3部署期多接收节点协同协议

### 7.1 单节点本地证据

每个接收节点部署同一冻结或合法适配后的Phase1特征提取器。节点`m`只根据当前已接收IQ形成：

```text
z_id^(m)       发射机身份表征
z_dom^(m)      接收机／信道域表征
q^(m)          观测质量与可靠性
d_class^(m)    到各注册身份的距离
e_unknown^(m)  本地未知能量或拒识分数
p_local^(m)    本地已注册身份预测
```

本地证据必须先成为不可变artifact，之后才能进入协同推理。单节点决策只能是`registered identity`、`unknown`或`defer`。本地预测不得读取query真值、真实old/new/unknown角色、其他query的真实类别构成、类别配额或独立scorer结果。

正式Phase3未知类拒识同样必须在LEO弱星地信道下评估。registered query与unknown query均须在连接真值或角色前，按与Phase2相同的单物理样本单观测原则，将每个接收记录固定为且仅固定为一份`leo_clear_weak`、`leo_low_elev_weak`或`leo_rain_weak`接收IQ；方法只能读取该固定`received_i`。禁止只给unknown叠加弱信道而让registered query使用clean，禁止依据真实known/unknown角色选择scene或seed，也禁止由同一物理接收记录生成多scene、多随机实现后挑选或融合。clean unknown只能作为隔离的matched诊断，必须标为`DIAGNOSTIC_UNKNOWN_NO_LEO_NON_FORMAL`，不得用于正式拒识指标、候选晋级或CVS卫星场景声明。

若对已冻结Phase1 bundle开展单节点目标域确认性unknown评测，registered与unknown均须在连接真值或角色前，按本节相同的单物理样本单LEO观测、scene／seed分配和三scene物理ID两两不交规则构建；候选及其unknown决策规则／阈值必须已由source-only冻结，全部预测artifact封存后才可由独立scorer评分。该评测零训练、零适配、零更新、零调参和零重试反馈，只能报告冻结bundle的单节点确认性指标，不构成Phase3多节点协同、anonymous entity、可信确权或注册授权结果。

### 7.2 协同输入与决策边界

Phase3可以融合各节点冻结的本地证据，以及与当前观测合法绑定的卫星可见性、时间窗、频率、波束、轨迹、位置约束和历史anonymous track。融合位置可以在星间链路、星座协同节点或地面网关，但必须在报告中明确。

正式协同方法必须显式处理接收机响应差异、传播/SNR差异、节点缺失或延迟、本地预测冲突、同一发射事件的多节点观测、跨过境匿名实体关联以及同源或高度相关证据的重复计权。简单平均、多数投票和选择最高置信节点只能作为基线，不能单独代表完整Phase3方法。

任何registered query被reject或defer，都必须在对应已注册类准确率中按错误计数。Phase3不得通过全部拒绝规避身份识别责任。

### 7.3 事件、接收与shot计数

Phase3区分物理发射事件与节点接收记录：

```text
一个emission_event_id
多个satellite_reception_id
仍然只计为一个shot
```

只有能够证明来自同一物理发射事件的多节点接收记录，才能作为严格same-event协同证据。不同时间、不同物理发射或无法建立事件绑定的记录不得拼接成一个K-shot样本。若现有数据不是同步多接收机采集，必须标为“多接收节点代理协同”，不得声称真实在轨同步多星验证。

### 7.4 Anonymous entity、可信确权与Stage2-C交接

被拒识的观测只能先关联为`anonymous_entity_id`，该ID表示多个观测可能来自同一物理射频链，不是最终语义身份。可信确权至少输出候选物理身份、证据来源、证据独立性、冲突标志、标签置信度、有效期和`registration_authorized`。

只有`registration_authorized=true`后，系统才能为获批身份重新采集K个独立物理发射事件作为新类support。历史unknown query保持为不可变检测证据，不得追溯改成support。新support按`p2_min_v1`形成新的`split_id`并完成对应数据验证后，交由Stage2-C执行旧类适应、新类注册、全部已注册类统一竞争以及遗忘/floor评价。

### 7.5 多节点数量与因果归因

活动目标可以在独立目标文档中冻结`N_sat`、receiver、seed、K和新类规模。协议要求至少保留`N_sat=1`单节点基线，并在任何协同收益声明中报告具体节点数、节点子集选择规则、缺失节点处理和证据相关性控制。

同时改变Phase1底层表征和Phase3协同方法时，必须使用同一输入口径分别评价：

```text
A：原Phase1基座+单节点
B：新Phase1特征提取器+单节点
C：原Phase1基座+多节点协同
D：新Phase1特征提取器+多节点协同
```

由`B-A`估计底层表征贡献，由`C-A`估计协同推理贡献，由`D-B-C+A`估计交互贡献。不得只比较A和D后把全部收益归因于协同推理。

## 8. 数据集角色与声明边界

- WiSig/ManySig可作为terrestrial proxy benchmark、source receiver family或目标接收机代理域。
- `R_t`必须与`R_s`不相交；当现有合法目标接收机覆盖不足时，可使用未进入Phase1的其他接收机/数据子集，但仍须满足集合与唯一观测协议。
- target-old、target-new与正式unknown query必须来自已定义的目标接收机域，并使用同一LEO弱信道生成与场景分配口径；不得用clean unknown替代正式拒识输入。
- 具体文件路径、样本数量、receiver/TX清单和hash属于数据资产登记表，不写入场景协议。
- 实验记录引用实际数据资产/物理ID/角色划分与配置来源，分别记录model、split、data、augmentation、support和evaluation seed；共享数据契约跨方法复用，不因登记或索引更新重复验证。历史缺证据字段保持未知，artifact完成与协议有效性分别记录；管理索引不进入训练、query决策或目标驱动选择。

可以声明CVS研究的是地面弱标注跨接收机DG、LEO压力下少样本旧类适应与新类注册，以及部署阶段多接收节点代理协同的Phase3方法。不能把WiSig称为真实卫星数据，不能把LEO模拟称为真实在轨验证，不能把非同步多接收机代理称为真实同步多星验证，不能把source-only DG称为few-shot适应，也不能把source proxy unknown、旧类提升、unknown拒识或协议无效结果当作新类注册成功。

## 项目资料管理入口（2026-09-16）

代码、报告、冒烟测试、临时文件及归档的查询与整理，先读[统一导航](../PROJECT_NAVIGATION.md)和[资料管理技能](../.agents/skills/project-file-management/SKILL.md)。当前清单、历史治理库桥接及恢复映射见[资料管理](project_governance/README.md)。先查现有索引，处理前核实当前路径/引用；保留代码镜像、独立变体和有追溯价值的冒烟证据，不凭同名、日期或tmp命名批量删除。实验管理及科学协议继续按原规则执行。
