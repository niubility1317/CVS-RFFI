# PAMF Phase2 总实验表：综合主场景与三个指定单场景

**r04 衔接补充：**T1/T2 保留分阶段原生比较；完整 Phase2 优势另用 T2-J，包含 PAMF、7 个已有注册系统，以及 7 个闭集 DA 各接原生结构扩展／NCM／JointCE 的 21 个明确组合，共 29 条完整链路槽位。结构保持对照防止更换头时丢掉 B 的适应收益。MILKR 的统一机制不等于 B→C 参数继承。衔接规则、直接 C 控制、声明边界与新增工作量见[域适应与注册衔接分析](PAMF_PHASE2_DA_REGISTRATION_LINKAGE_20261028.md)。下文原 289520／294140 仅是 15 系统基础矩阵；完整覆盖新增组合后为 806960 个 B/C 逻辑 row、811580 个阶段预测槽位，新增接口尚未实现。本次不启动实验。

2026-10-28，r03。**DESIGN_ONLY：只整理实验设计，未启动训练、目标适应、预测或评分。**

本表覆盖此前六单场景诊断的安排：本轮独立单场景恰好三组，分别为 low 郊区、mid 城区、low 城区。每组全部目标 support/query 采用该组同一种 practical 场景；各记录独立生成自己的固定信道快照，不共享同一组信道系数。综合主场景仍包含六种分量，固定 8 套。

历史 MILKR 的 high 郊区／mid 郊区／low 城区结果保持原身份；本轮三组是新的指定对照，不能将旧 high/mid 郊区成绩改名填入。当前表征为 PAMF 三视图 Tri-SIG-TIME SSL E50，历史 DAOT/FastTrust-RC4 E200 成绩也不能自动继承。

## 1. 场景总表

|ID|组别|目标样本的信道规则|数据套数|地位|
|---|---|---|---:|---|
|G10|六分量综合 practical residual|每条物理记录按下表概率选择一种场景，只处理一次；support/query 均来自综合分布|8：MIX01–08|主比较、规模曲线、成本和稳定性|
|D-LS|低仰角郊区|全部 support/query 为 `practical_low_suburban`；仰角均匀取 10°–30°|1|指定独立实验组 1|
|D-MU|中仰角城区|全部 support/query 为 `practical_mid_urban`；仰角均匀取 20°–45°|1|指定独立实验组 2|
|D-LU|低仰角城区|全部 support/query 为 `practical_low_urban`；仰角均匀取 10°–30°|1|指定独立实验组 3|
|G25|最低有效仰角 25° 的综合参考|六分量条件概率；低分量只取 25°–30°，其他两档为 30°–45°、45°–80°|如实施则另固定 8 套|补充敏感性设计，不计入核心工作量|
|G20|最低有效仰角 20° 的综合参考|低分量只取 20°–30°，其他两档同上|如实施则另固定 8 套|过渡敏感性设计，不计入核心工作量|
|ENV|环境占比敏感性|城区 20%／50%／80%；先报告固定 support 状态下的 query 重加权|复用对应固定预测|不能冒称另一 support 分布下的完整适应|
|HIST|历史 MILKR 原三景|只保留原数据、模型、划分和结果，原四 RX／两 capsule 不变|历史值|历史参照，不计本轮新数据或新主均值|

核心是 **4 类场景条件、11 个独立物理数据池**：8 套综合＋3 个单景。不是新增三个场景的 1/3 混合任务，也不是把 MIX01–08 分成不同概率的 3＋3＋2 套。所有新池之间、support/query 之间物理 ID 互斥；历史复用排除重复物理记录。

### 1.1 综合概率

|分量|G10 主参考|G25 补充参考|G20 过渡参考|
|---|---:|---:|---:|
|low 郊区|38.312874%|14.346548%|24.947300%|
|mid 郊区|7.317985%|22.324687%|15.686943%|
|high 郊区|4.369141%|13.328766%|9.365757%|
|low 城区|38.312874%|14.346548%|24.947300%|
|mid 城区|7.317985%|22.324687%|15.686943%|
|high 城区|4.369141%|13.328766%|9.365757%|

G10 采用 600 km、10°–80° 覆盖环内地面位置按球面面积均匀的明确参考，城区／郊区各 0.5。G25/G20 是同一空间参考的条件化，不能与特定轨道的时间积分比例混用。综合档位互斥，mid 为 30°–45°；三个指定单景保留原生均匀仰角，D-MU 为 20°–45°。表中仅显示舍入值，配置与实际执行使用完整精度。

这些都不是实际载荷频率。当前单星被动监听不控制地面发射机、不选服务星；未确定天线／采集条件前，G25 不自动替换 G10。外层概率不改变 practical 内部 LOS／PARTIAL／BLOCKED 状态规则。所有场景固定 `residual/post_sync/noEQ`、25 MHz、FP32 `[2,256]`；保留真实输入预处理身份，noEQ 不等于原始 WiSig 从未经均衡。不加入 DURA。

## 2. 公共数据、阶段与矩阵

|项目|统一设置|
|---|---|
|当前主系统|PAMF 三视图 Tri-SIG-TIME SSL E50／11100 更新＋冻结 MILKR `D92-BranchLocalRidge-v1`|
|源 RX／日期|Main split 名义索引 `[1,3,4,6,8]`；D1–D3；实际物理映射与 checkpoint 全部祖先须匹配|
|源角色|L_s=6300；U_s=56700，TX 隐藏；V=27000，只验证，不更新持久状态|
|目标 RX／日期|`[0,2,5,7,9,10,11]`，7 RX；D0–D3|
|旧类／新增类|旧 6；新增 0／2／5／10／20；新增集合按固定顺序嵌套|
|每类 target 标签|K=1／5／10／20；K5 为主展示点；不是无标签研究|
|主模型种子|2026092701–04，4 个；相同数据在各方法及模型种子间共享|
|support 抽样|2026092711–15，5 次；按 dataset/RX/TX 命名随机流；K 前缀嵌套|
|每池每 RX／类|36 条 support 候选＋30 条 query；每 row 只开放自己的 K 条 support|
|日期覆盖|support 候选每日期 9 条；query 每日期 7／8 条轮换，综合 8 套合计各日期 60 条|
|query 配对|同 dataset/RX/class 的 query 跨 K／draw 固定；新增时旧 query 保持；逐样本面对全部注册类|
|信道随机性|同套输入跨方法／模型种子固定；改变模型 seed 不重新叠加；receiver_seed=2027，只绑定 RX/day|
|旧新可见性|旧类在 S 可见；新增 TX 不能进入源模型及祖先训练、验证或选模；标签映射不符则明确缺项|

|阶段|允许输入与行为|输出|
|---|---|---|
|S：源训练|所有方法只使用与 PAMF 相同的源角色；各自原生网络／目标、U_s 使用及源预算披露；target IQ／统计访问为 0|合法 source-only 模型；需要的源 L_s 原型／enrollment／Fisher 按方法权限生成并冻结|
|A：适应前|冻结源状态，只读逐 query 推理；不做 BN、阈值、原型等更新|A_old|
|B：旧类适应|才开放当前 row 的旧类有标签 target support；UDA 对齐只能用这些 support 的 IQ；外部方法明确允许的 source replay 可用|B_old、B−A|
|C：新类注册|才开放新类 support；仅继承本 row 自己的 B。MILKR 例外：用当前新旧 support 独立重拟合，不继承 B 核状态|C_old、C_new、H、B−C_old|

源原始容器若同时含 source/target，须先准备 source-only 材料；不能仅依赖 `target_access=False` 字段。所有源模型选择只用合法源 V 或固定轮次；不能照搬作者 target-test-best。query 不用于域对齐、伪标签、中心化、带宽或任何状态更新。

|K|B 旧类 support 数|C 新 20 类 support 数|C 总 support 数|
|---:|---:|---:|---:|
|1|6|20|26|
|5|30|100|130|
|10|60|200|260|
|20|120|400|520|

N_new=0／2／5／10／20 时总类数分别为 6／8／11／16／26；每类 30 条 query，总 query 分别为 180／240／330／480／780。N_new=0 复用 B，不新增 C fit，新类与 H 为 N/A。

历史旧 TX：`14-10,14-7,20-15,20-19,6-15,8-20`。新 TX 固定顺序：`11-4,3-13,12-20,5-20,20-12,17-11,14-14,2-17,4-1,19-13,19-8,2-19,8-1,14-20,13-3,1-18,20-7,7-7,2-3,18-12`。本轮使用前核实当前 PAMF 标签映射与未见性；不能仅因同名 split 沿用。

## 3. 论文总表索引

|表号|内容|场景／范围|行与列|
|---|---|---|---|
|T0|公平性、源训练与额外输入|全部方法版本|源角色、U_s、增强、源更新、target 首次访问、回放／Fisher／teacher／原型、参数与源成本|
|T1|旧类域适应主比较|G10 八套；D-LS／D-MU／D-LU 各独立面板|下表 D0–D9；A_old、B@1／5／10／20、各 K 的配对 Δ=B−A|
|T2|新类注册主比较|同上；N_new=20；四个 K 各一子表|下表 R0–R7；A、B、C_old、C_new、H、B−A、B−C_old、Macro-F1|
|T2-J|完整 Phase2 链路比较|G10＋三个指定单景；相同标签／阶段权限|8 个已有完整系统＋21 个 DA→原生结构扩展／NCM／JointCE 组合；另报 C_old−A_old 与成本；新组合未实现|
|T3|新增规模与标签曲线|同上；完整 4K×5N；主文展示 K5／20|C_old、C_new、H、总 support；N0 单列 B，不混入 H|
|T4|适应、注册与部署成本|G10，K5／20、N_new20；三单景配套分层|source 成本、B fit、C 总 fit／每 session、编码、B1 median／p95、RSS／VRAM、持久状态／传输|
|T5|共享 PAMF 表征控制|G10＋三单景|S0–S10；完整冻结 MILKR 与简单读出／梯度适应区分|
|T6|MILKR 机制消融|核心 M0／M2／M3 完整矩阵；其他项固定切片|不得从目标分数挑选“有效消融”后只报告它们|
|T7|分次注册|G10；K5／20；最终新 20 类|1×20、2×10、4×5、10×2；逐 session 保持、新类、H、累计成本|
|T8|随机性与失效分层|全部核心条件|8 套逐套、4 模型、5 draw、7 RX、日期、实际 scene、逐 TX、混淆与类下限|
|S1|完整代码版本对照|22 个外部版本全部列入；有合法接口者按同核心条件|不能将不同版本计为新论文；有接口缺口／观测不匹配明确记 N/A|
|S2|G25／G20 综合条件敏感性|补充设计，分别独立数据池|若完整适应，support 与 query 均更换对应分布，不只加权旧分数|
|S3|环境比例与参数敏感性|固定参考 support 下的 query 重加权；参数为固定切片|解释各自改变的因素，不用目标结果重新选择默认值|
|S4|历史版本桥接|历史 DAOT/RC4＋MILKR 原身份；当前 PAMF＋MILKR 新身份|分别报告数据契约与分数，不以旧结果填当前表缺项|

G10 与三个单景分别报告，不能将 11 个数据池等权混成一个主分数。D-MU／D-LU 同为城区可考察条件差异；D-LS／D-LU 同为低仰角可考察环境差异；由于物理样本互斥，它们是分布条件比较，不是同一 IQ 的逐样本因果配对。

## 4. T1：旧类域适应的 10 行主方法

所有方法各自先完成 S，A 使用自己的源决策；不能用 PAMF 的 A 替代外部方法。以下目标预算是设计中已有的匹配设置，原论文预算与项目扩展必须分列。

|ID|方法|B 阶段的算法与预算|B 的额外源输入|
|---|---|---|---|
|D0|PAMF＋MILKR|冻结 encoder；旧 support 一次 FP64 解析局部核岭回归；λ=1；0 optimizer step|无 source／源摘要运行时输入|
|D1|POSTER Fine-Tuning|冻结卷积，重建并训练声明的 Dense；Adam 0.001，10 epochs，batch≤256|无|
|D2|RadioNet ADA＋FT＋kNN|source/support 对齐 10000 iterations；随后 support FT 30 epochs；最后 kNN，k=K|合法 source L_s replay|
|D3|ProtoNet CDA|自己的 episode encoder；旧 support 类均值＋平方欧氏判决；0 optimizer step|B 无；A 用 S 末合法 L_s 原型|
|D4|MRIOR-SDA，CPL-off|GAD／DV-KL＋support CE；100 外层更新，估计器每步 7 次；lr 0.0006|合法 source L_s replay|
|D5|DADDA-SDA|source/support CE＋动态 MMD／LMMD；100 更新；support 真标签用于条件对齐|合法 source L_s replay|
|D6|Two-stage SDA＋FT|DANN 100→LMMD 100→FT 100 更新|前两步合法 source L_s replay|
|D7|Hu2024 Feature Separation|共享 encoder／TX 路径 FT，RX 路径冻结；100 epochs，Adam 0.005，batch≤256|按选定完整实现披露；不得暗中扩大 source|
|D8|RA-Collab 单节点 FT|全网 SGD 20 epochs，lr 1e−5，momentum 0.9，batch≤32；不做多节点融合|无 target 外数据|
|D9|Tweak M1 扩展|自己的 triplet encoder；target centroid／radius 校准；单 query 判决|S 末合法 source calibration 冻结，实际载荷披露|

CSIL／MoPC／Orthogonal／ISSL 的 B 是为了比较“适应后注册”而明确添加或匹配的步骤，在 T2 中报告；不能把它们称为原论文自带的旧类 DA。LoRa 旧类 enrollment 同样在 T2 给出。

## 5. T2：新类注册的 8 行主方法

|ID|方法|B 状态|C 的输入、继承与预算|
|---|---|---|---|
|R0|PAMF＋MILKR|同 D0|新旧全部合法 support 独立重拟合带宽／中心化／trace／核解；不继承 B 系数|
|R1|CSIL empirical-Fisher corefix|旧 support SFT 10 epochs，显式 B 扩展|继承自己的 B；新 support、通道隔离／mask／KD／合法源 Fisher；C 3 epochs；经验 Fisher 修复版独立命名|
|R2|MoPC-HR paper-semantic|旧 support 匹配 B10＋声明的原型校正|继承自己的 B 与历史原型；新 20 类按 4×5 session，每 session 20 epochs；原型增强、HR、漂移校正向后传递|
|R3|Orthogonal Incremental SEI|匹配 B10 权重校准|保留自己正交源训练；冻结 encoder／旧权重，新类均值初始化后校准 50 epochs|
|R4|ISSL fixed_v1＋B10 扩展|旧 support 全网 Adam 1e−4，10 epochs，batch=min(32,6K)|继承自己的 B／teacher；扩头→SSL 2000 epochs→CE transfer 300 epochs；合法旧训练样本 replay；source 模板 300 epochs|
|R5|LoRa triplet＋KNN|自己的源 encoder，旧 target enrollment|冻结表示，继承旧 memory，追加新 support；KNN15，小样本取 min(15,Nsupport) 为显式兼容扩展|
|R6|ProtoNet 注册扩展|同 D3|保留 B 旧原型，追加新 support 均值，全部注册类统一竞争；不重新训练 encoder|
|R7|RadioNet ADA＋FT＋kNN 注册扩展|同 D2|冻结自己的 B encoder，保留旧 target memory，追加新 support；不重新进行 ADA|

MoPC 的 N_new=2 为一个 2 类短 session，N5／10／20 分别为 1／2／4 个 5 类 session；短 session 是明确的项目任务适配。ISSL 原生 A→C 与作者行为保留版另列，不伪造原生 B。LoRa 另给 `k=min(15,K)` 的 K-adaptive 控制，不能因该控制分数更高而替换原生兼容主行。闭集 DA 方法没有原生 C 时记 N/A，不强接新头后仍称原方法。

## 6. 全部完整外部代码：14 家族、22 版本覆盖表

|序号|版本|主表／辅助位置|A／B／C 的边界|
|---:|---|---|---|
|1|POSTER FT|D1；T1|A/B；扩头注册不属于现有原生 C|
|2|RadioNet ADA-only→kNN|S1，与 D2 配对|A/B；追加 C 是项目扩展；1000 次历史预算与 10000 次完整版不能同名|
|3|RadioNet ADA→FT→kNN|D2、R7|A/B/C 扩展，正式主路径|
|4|RadioNet DF-FT|S1|A/B，末层 FT 30 epochs；不能代表 ADA|
|5|RadioNet Triplet→kNN|S1|源 triplet 30 epochs；B enrollment；C 追加是项目扩展；256 点接口待接|
|6|ProtoNet CDA／PTN|D3、R6|独立 episode 源训练；C 是追加原型扩展|
|7|MRIOR-SDA，CPL-off|D4|原 UDA 的 SDA 扩展；闭集 C=N/A|
|8|DADDA-SDA|D5|原 UDA 的 SDA 扩展；闭集 C=N/A|
|9|Receiver-Agnostic Two-stage SDA＋FT|D6|DANN→LMMD→FT；闭集 C=N/A|
|10|Hu2024 Feature Separation 完整新版|D7|A/B；源特征分离；目标 RX 路径冻结|
|11|Feature Separation 旧匹配版|S1|同一论文的旧复现；单独版本，不与新版成绩互填|
|12|RA-Collab 单节点 FT|D8|单节点 A/B；去除旧 support/query 重叠后接入；不加协同|
|13|Tweak M1|D9|显式单观测扩展；A 只用合法 source calibration|
|14|Tweak 原生 M10|S1 多观测适用性|需要真实、无标签泄漏的 10 包事件组；当前单包数据不提供时 N/A，禁止按 TX 真值事后凑组|
|15|CSIL empirical-Fisher corefix|R1|匹配 IQ、经验 Fisher 修复；B 是扩展|
|16|CSIL 作者 MATLAB 行为|S1 版本对照|原生源码存在，正式 256 点接口／原生数值行为须明示；未绑定不能填主表|
|17|MoPC-HR 论文语义版|R2|cosine 校正、平方 L2 HR、完整分类器判决|
|18|MoPC-HR 公开代码语义版|S1|raw-dot／softmax 等差异独立记录，不与论文语义混名|
|19|Orthogonal Incremental SEI|R3|保留自己的源正交目标；不得用未经正交训练的 PAMF 头冒充|
|20|ISSL fixed_v1|R4；另列 fixed 原生 A→C|KD 图、zero_grad、queue、teacher/key BN 修复；B10 明确是扩展|
|21|ISSL 作者行为保留版|S1|原队列／梯度／KD 行为原样说明；不称为修复版；无原生 B|
|22|LoRa scalable/channel-robust triplet＋KNN|R5|Torch 移植与 256 点 STFT 为显式扩展；窗 64／重叠 32；A 用源 enrollment|

以上 22 行全部保留在总设计，不因尚未接当前数据而删方法。14 行外部代表方法加 PAMF，核心主表为 15 个不同系统；T1/T2 交集只算一次。其余版本列 S1，能合法绑定的按相同核心场景／预算矩阵比较；原生输入条件不满足者报告缺口，不换成 NCM/SFT 冒充。

完整源码路径、版本来源与当前接入状态见[代码盘点](PAMF_PHASE2_METHOD_CODE_INVENTORY_20261028.md)。CSCNet、SHOT、独立 DAN／DSAN／CDAN／WD／Deep CORAL、独立 LwF／EWC／iCaRL 等未找到完整独立实现的项列缺失表，不计“已有完整代码”。

## 7. 共享表征、通用 DA 与内部变体

|ID|固定同一 PAMF 后比较|性质|
|---|---|---|
|S0|原默认分类头|只作 A；source 头不另训|
|S1|NCM|support 均值＋cosine，冻结表示|
|S2|kNN|support memory；预定 k／距离规则，不能目标选 k|
|S3|线性 ridge|λ=1；明确原实现目标／截距与 MILKR 不同；严格机制比较需匹配中心化目标|
|S4|线性 CE|只训练 Phase2 头；既有固定 20 epochs／Adam 0.001，披露初始化|
|S5|BranchRidge|相同分支，无乘性交互／局部核|
|S6|BranchInteraction|保留交互，无局部径向核|
|S7|MILKR|完整固定方法；与 R0／D0 复用，不重复运行|
|S8|head SFT|Phase2 默认头 support CE，固定预算|
|S9|tail SFT＋MILKR|固定尾部和预算的表示适应扩展；B→C 继承与 reset 分列|
|S10|full SFT＋MILKR|全网适应参照，不能叫冻结 MILKR|

通用控制完整列入：SFT、AdaBN、CORAL＋SFT、DANN、MCD。AdaBN 没有 BN 的路径记 N/A；CORAL-SFT 是项目监督扩展；DANN/MCD 完整通用代码已存在，接入 RFFI 后 S 仍须 source-only，B 才对齐，不能继承 target-test-best。通用模板预算不冒充 RFFI 原论文预算，具体 RFFI 源／目标预算在实现时依据 source-only 证据固定，当前不宣称已可启动。

内部历史完整候选全部保留追溯入口：原 D92、SFHead、SCV、SGJoint、MVKME、BNNA、OSC/OrbitShared、MVRidge；BranchMetric／BranchOrbitCE／BranchLocalMargin／WithinClassMetric；SequentialResidual／JointChannel／JointSpectral／PrototypeTransport／MCResidual8／FCR8；Anchor／Affine／Conditional／Margin／GroupBarrier／GroupBalancedSupportMetric／SupportMetric／ProtoFrame Joint；FisherSubspace／Adaptive／Calibrated／Iterated Fisher；NativeSupportTail／NativeLocalRidge／NativeCrossfitRidge／NativeFullCrossfitRidge。它们不是新增外部论文，也不全部变成新主表训练行；完整矩阵优先原 D92、BranchRidge、BranchInteraction、MILKR，其他按冻结历史配置做指定诊断。

## 8. MILKR 消融与参数敏感性

|ID|变更|回答的问题|
|---|---|---|
|M0|完整 MILKR|参照，与主行复用|
|M1|原 z_id＋FFT96 的 D92|历史整体方法参照，不作单因素解释|
|M2|BranchRidge|M2→M3 检验乘性交互|
|M3|BranchInteraction|M3→M0 检验局部径向核|
|M4|去 z_id|联合身份块的条件贡献|
|M5|去 FFT96|频谱背景的条件贡献|
|M6|去 t_emb|时域辅助读出块的贡献|
|M7|去 f_emb|频域辅助读出块的贡献|
|M8|去 pa_local|PA 辅助读出块的贡献|
|M9|关闭 trace matching|核尺度／有效正则的影响|
|M10|改为类盲 pairwise-distance 中位数带宽|最近异类标签规则的影响；分类标签仍保留|
|M11a|只用 z_id 的匹配局部核|单一联合身份信息参照|
|M11b|只用 FFT96 的匹配局部核|单一频谱信息参照|

M0/M2/M3 做核心完整矩阵；其他项事前固定 MIX01/02、K1/5/20、N_new0/5/20、4 模型、7 RX、5 draw 的机制切片。切片全部报告，不据 query 分数筛选扩展项。删除块的归一化、范数匹配与 active-block 公式须预先固定；M6–M8 只删读出输入，z_id 仍可能含该分支信息，不能称为网络删除消融。本轮不自动重训 Phase1 网络分支。

参数单因素敏感性使用相同固定切片：λ∈{0.01,0.1,1,10,100}；τ 倍率∈{0.25,0.5,1,2,4}；FFT 权重∈{0,1,2,4,8}。默认始终为 λ=1、τ 倍率=1、FFT 权重=4；不运行三轴笛卡尔积，不从目标曲线选择新默认。现有 frozen runner 不接受这些变体，需要独立命名的配置／接口，当前不称已实现。

## 9. 分次注册、标签依赖与可靠性

|实验|固定设置|必须报告|
|---|---|---|
|标签预算|K1/5/10/20，完整新增规模|A/B/C、H、类下限、总 support 与实际标签数；不改称无标签|
|注册规模|N_new0/2/5/10/20|旧类保持与新类识别同时报告；N0 无 H|
|分次注册|最终新 20 类；1×20、2×10、4×5、10×2；K5／20；G10 八套|每 session、最终、累计成本；原生状态继承；MoPC 大 session 是显式扩展|
|类顺序|固定顺序为主；另用 2026102841/42/43|顺序敏感性单列，不改主顺序|
|模型稳定性|4 个主模型；PAMF05–08 仅已有合规权重时补充|4 模型主比较与 8 模型补充分开，前 4 复用|
|稀有场景覆盖|记录实际 support/query 场景频数|K1/5 自然缺景，不强制每类补齐六景，不额外开放标签|
|分层|RX、D0 与 D1–D3、TX、实际场景、状态|D0 为未见日期，不称未来日期；状态只在独立评分侧使用|

MILKR 最终累计 support／特征一致时，独立重拟合的最终解不应由到达顺序产生不同；逐 session 指标、历史存储与累计成本仍不同。各方法不能从 C2 接着训练到 C5/C10，再将它们冒充独立规模实验。

## 10. 核心工作量与物理数据需求

下表按 4 模型×7 RX×5 draw×4K×4 个正新增规模计算。A 不拟合；“fit row”是逻辑 B/C 任务，不等于源训练次数、optimizer step 或独立样本数。

|每个完整 A/B/C 方法|A 唯一参考|B|C，正新增规模|B＋C fit row|A＋B＋C 阶段预测|
|---|---:|---:|---:|---:|---:|
|G10 八套|224|4480|17920|22400|22624|
|一个指定单景|28|560|2240|2800|2828|
|三个指定单景合计|84|1680|6720|8400|8484|
|核心合计|308|6160|24640|30800|31108|

闭集只做 A/B：G10 4704、一个单景 588、三单景 1764、核心合计 6468 个阶段预测。主表 8 个支持完整 C 的系统＋7 个闭集 DA 系统，合计为 **289520 个逻辑 B/C fit row、294140 个阶段预测槽位**。其中所有 15 系统的 B 为 92400，8 系统的 C 为 197120，A 为 4620。A 无合法来源时明确缺项，不能补造；成本只计真实执行。

这些合计不含 22 版本辅助扩展、共享读出新增行、机制切片、分次注册、PAMF 额外 4 模型及 G20/G25。不能把它们含糊算作已包括。一个新增 8 套综合条件若覆盖相同 15 主系统，会另增加 210560 个 B/C row、213920 个阶段预测槽位。

|数据部分|互异物理观测数|每 RX／类需要记录|FP32 IQ 理论字节数|
|---|---:|---:|---:|
|G10 八套|96096|528|196804608|
|一个指定单景|12012|66|24600576|
|三个指定单景|36036|198|73801728|
|核心合计|132132|726|270606336，约 258.07 MiB|

每个额外 8 套综合条件另需 96096 条、每 RX／类 528 条互异原记录。数据覆盖尚未核实；不足时报告具体 RX／TX／日期缺口，不静默复用物理 ID 或以同一 IQ 的新随机信道代替独立数据。上述字节仅 IQ 数组，不含索引、metadata、模型、特征、预测和历史状态。

核心 15 种系统×4 模型种子为至多 60 个 source 模型绑定；已有合规模型可复用，不表示新增训练 60 次。不同源机制／版本不可共用不匹配权重。每个 source 模型跨所有场景复用；每组 B 从自身 source 状态恢复，包含 buffer/BN/optimizer/teacher 等；C 按方法定义继承。

## 11. 指标、统计与交付

主指标：A_old、B_old、C_old、C_new、H；配对 B−A、B−C_old；Macro-F1、最差类 recall、recall P10、新旧绝对差、旧→新／新→旧错误、净纠错及完整混淆。H 在每个 cell 由旧／新准确率计算，再平均；不能由两项平均值倒算。整体准确率会受新旧类数量影响，不能代替 H。

主综合先在 model×dataset×RX×draw×K×N 内配对，再对固定 8 套／7 RX／5 draw 汇总为每个模型的结果，报告 4 模型均值±样本标准差；8 套逐套值、套间标准差／范围和 draw 波动另列。每个单景独立按同样规则聚合。共享模型／RX／query 的相关性不能忽略，不把 8×5×4 当 160 个独立训练重复。每类 30 query 的单 cell recall 分辨率为 3.33 个百分点。

所有预登记预测完成固定后，独立 scorer 才读取 truth。失败／N/A／实际 seed 数和缺项逐行披露；不删低分套，不按成绩追加重复，不依据目标结果选 checkpoint、超参、场景概率或方法版本。历史“适应提升 10 pp、注册旧类下降不超过 1 pp、新旧差不超过 3 pp”仅是理想解释目标，不作为删结果或技术重跑理由。

完整源码存在、当前数据接口接通、正式结果完成必须分别说明。本轮只是设计：当前 PAMF 的 736 维五块导出兼容、综合 geometry 接口、各外部方法 source-only 绑定与新物理池覆盖仍待实施；不将任何设计值填为实测结果，不恢复暂停任务，不操作其他 owner。

交付位置：[主表设计](PAMF_PHASE2_MAIN_TABLES_20261028.md)、[机制与完整设计](PAMF_PHASE2_MILKR_EXPERIMENT_DESIGN_20261028.md)、[方法代码盘点](PAMF_PHASE2_METHOD_CODE_INVENTORY_20261028.md)、[机器设计](../configs/pamf_phase2_milkr_experiment_design_20261028.json)、[综合与三单景规则](../configs/pamf_phase2_practical_mixture_20261028.json)、[观测分布分析](analysis/pamf_passive_elevation_20261028/REPORT.md)。
