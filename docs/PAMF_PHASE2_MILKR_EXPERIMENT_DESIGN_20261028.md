# PAMF Phase2：MILKR 有标签域适应与新类注册完整实验设计

r05 已合并源训练、29条完整链路、状态继承和预算，并纳入所有方法匹配的 P0 received／P1 冻结DURA双路；DURA本轮资格未核实，不微调G或自动重训F。旧文中的单输入路径计数与“不新增DURA”安排由[完整总表](PAMF_PHASE2_TOTAL_EXPERIMENT_TABLE_20261028.md)覆盖，历史结果和Phase1路线不追溯改写。

r04 新增[完整链路衔接分析](PAMF_PHASE2_DA_REGISTRATION_LINKAGE_20261028.md)：保留所有原生机制，给 7 个闭集 DA 各接原生结构扩展／NCM／JointCE 三类明确 C 控制，组成 T2-J 的 29 条完整链路槽位；新增组合 runner 尚未实现。MILKR C 的独立重拟合与 B 后参数继承有严格区别，不能用闭集方法 C=N/A 作为性能优势证据。

本轮 r03 总表已按用户最新指定将独立单场景收敛为 **low 郊区、mid 城区、low 城区三组**；每组全部 target support/query 使用该组同一种场景，各物理记录的随机信道实现独立且固定。综合仍为六分量、8 套，历史 MILKR 原三景不改名。完整清单、主表、22 版本、消融和修订后的工作量以[全面总实验表](PAMF_PHASE2_TOTAL_EXPERIMENT_TABLE_20261028.md)为准。

日期：2026-10-28，r03 综合 practical residual、三个指定单景与总表修订。状态：**DESIGN_ONLY，尚未启动任何训练、预测或评分**。

本设计落实用户本轮范围：**仅设计现有有标签 MILKR 的完整 Phase2 实验**。不加入无标签适应、匿名新类发现、Phase3 unknown 拒识、多节点协同或联邦学习。K-shot 标签预算曲线属于现有有标签任务，不把它改称无标签研究。

本轮主表与综合场景的最终口径见[主表详细设计](PAMF_PHASE2_MAIN_TABLES_20261028.md)。它覆盖前版的三种leo_weak主面板安排：主实验为固定8套综合practical residual，三单场景作为诊断。所有源训练严格source-only，到B才开放合法target support。

应用现已明确为单星被动监听。综合概率仍是空间几何参考，不自动代表已采IQ或实际任务频率；接收条件与仰角占比分析见[被动监听分析](analysis/pamf_passive_elevation_20261028/REPORT.md)。本次不改变概率数值，不使用通信选星逻辑。

配套文件：

- [方法代码清单与版本边界](PAMF_PHASE2_METHOD_CODE_INVENTORY_20261028.md)：逐方法源码、入口、权限、现有完成度和未接入项。
- [综合场景概率与8套数据](../configs/pamf_phase2_practical_mixture_20261028.json)：600km参考几何、场景权重、条件采样、独立物理池与随机性。
- [机器可读实验设计](../configs/pamf_phase2_milkr_experiment_design_20261028.json)：矩阵、方法、预算引用和非可启动状态。
- [当前 PAMF 定义](PAMF_METHOD_DEFINITION_20261027.md)、[科学协议](PROJECT_PROTOCOL.md)、[实验管理](EXPERIMENT_MANAGEMENT.md)。

## 1. 核心决策与研究问题

主方法固定为 **MILKR：Multi-branch Interaction and Local Kernel Ridge，多分支交互与局部核岭回归**，实现 ID 为 `D92-BranchLocalRidge-v1`。地面表征与 Phase2 决策分别标识：`PAMF(Tri-SIG-TIME SSL, E50) + MILKR`。历史 `DAOT A1/FastTrust-RC4 E200 + MILKR` 独立保留，不能给旧成绩换上新表征名称。

本方案采用三层比较：

1. **完整系统比较**：各 RFFI 方法保留自己的源训练、骨干与原生适应／注册机制，比较完整方法在匹配目标任务上的表现。
2. **共享表示比较**：在同一冻结 PAMF 表征上比较 NCM、k-NN、线性／核岭回归及明确的移植适应版本，分离表征与决策收益。
3. **MILKR 机制消融**：固定同一源 checkpoint、support/query 和预算，只改变表示块、交互、局部性或尺度中的一项。

不采用“全部外部方法换成 PAMF＋NCM”作为主对比；这会删除它们的算法。也不将完整系统表中的差值全部归因于 MILKR，因为源训练方式、额外源域无标签使用和计算预算可能不同。

需回答六个问题：

|编号|问题|直接证据|
|---|---|---|
|Q1|旧类 target support 能否改善目标域识别？|同一物理旧类 query 的 A→B|
|Q2|加入新类后，旧类保持、新类识别与两者平衡如何？|B→C、C_old/C_new/H、逐类下限|
|Q3|MILKR 相对现有完整 RFFI 方法处于什么位置？|原生系统主表、相同目标切片及权限表|
|Q4|提升来自 PAMF 表征还是 MILKR 决策？|共享表示面板与源模型×读出桥接|
|Q5|多分支交互、局部核及 support 带宽分别有何贡献？|单因素消融及冻结参数敏感性|
|Q6|部署适应／注册需要多少时间、内存和状态？|从 received IQ 起的实测成本，及单独缓存读出成本|

## 2. 固定方法，不把新研发混入复现

### 2.1 MILKR 的冻结公式

五个输入块为 `z_id(160), FFT96(96), t_emb(160), f_emb(160), pa_local(160)`，显式总维度 736。设 `unit` 为既有带 norm floor 的归一化：

\[
b=\operatorname{unit}([\operatorname{unit}(z_{id}),4\operatorname{unit}(FFT96)]),
\qquad a=[\operatorname{unit}(t),\operatorname{unit}(f),\operatorname{unit}(pa)]/\sqrt3.
\]

交互表示为 \(\phi=[b,a,b\otimes a]\)，隐式维度 123616，不显式展开该向量。距离为该交互空间中的平方欧氏距离，按既有稳定 QR 实现计算。

\[
\tau=\operatorname{median}_i\min_{j:y_j\ne y_i}\|\phi_i-\phi_j\|^2,
\qquad R_{ij}=\exp(-\|\phi_i-\phi_j\|^2/\tau).
\]

只用当前训练 support 完成中心化；径向核中心迹匹配交互核中心迹，得到 \(K_c\)。目标为 \(Y_c=onehot(y)-1/C\)，固定 \(\lambda=1\)：

\[
\alpha=(K_c+I)^{-1}Y_c.
\]

使用 FP64 Cholesky 和两次三角求解，无 jitter、无 optimizer、无 query 选参。零带宽及完全相同特征按冻结实现的精确退化规则处理。预测逐 query 面对全部已注册类别，保持确定 tie-break。方法定义以 [冻结配置](../configs/d92_branch_local_ridge_frozen_20260929.json) 和 [核心实现](../code/cvsrffi/d92_branch_local_ridge.py) 为准。

### 2.2 当前 PAMF 与历史 D92 的桥接

当前 PAMF 为三视图 SIG-TIME＋固定局部排列＋Tri-SIG-TIME SSL，六类模型 232993 参数、160 维联合特征、固定 E50/11100 更新。优先使用 `20261027-phase1-pamf-repl-shuffled-views-m16-r02` 中 view_count=3 的既有权重；不恢复暂停的普通 PL 队列，不自动追加地面训练。

当前 PAMF 是在已公开 Phase1 目标结果之后由用户指定的主方法；其选择背景必须披露。冻结后的 Phase2 新评估可以提供新任务证据，但不能据此把整个方法开发过程称为从未接触目标基准的独立盲选。

旧导出器期望原生 aux 中存在上述四个 160 维块；当前 PAMF 使用自己的安装和读取路径。**源码层面的相似字段不能代替实际兼容性验证**。实现阶段需用当前官方 reader 严格加载原权重，核对固定排列、联合特征是否含 SIG-TIME、默认头 logits 及分支语义，再建立只读导出接口。不得沿用旧架构 loader 后部分加载，不得把签名描述向量冒充 `t_emb`，不得直接复用旧模型特征缓存。

这只是导出／推理适配，不训练或重训 Phase1 分类头。Phase2 的 support 分类器是本任务允许的适应状态，与 Phase1 禁止另训头的约束分开。

|表征臂|Phase2 读出|用途|
|---|---|---|
|历史 DAOT A1＋FastTrust-RC4 E200|历史 D92、BranchRidge、BranchInteraction、MILKR|复现已有证据，保留原数据和历史身份|
|当前 PAMF 三视图 E50|同一冻结 MILKR 公式|当前完整系统主行；新增 Phase2 证据，不能继承旧成绩|
|当前 PAMF 三视图 E50|NCM、k-NN、linear ridge、BranchRidge、BranchInteraction|隔离读出贡献|
|PAMF-Sup／PAMF-PL／单视图|相同 MILKR|已有匹配合规权重才进行桥接；否则列缺项及源训练成本，不能自动训练|

完整系统中的 PAMF 使用源 U_s，若外部方法仅使用 L_s，应披露源标签量、无标签量及曝光量。监督公平性面板优先增加已有 PAMF-Sup＋MILKR，而不是把外部监督方法伪称同一种 SSL。

## 3. 任务状态及严格配对

|状态|可用于拟合的数据|输出与含义|
|---|---|---|
|S：源训练|与PAMF相同source RX/day/old TX及L_s/U_s/V物理角色；U_s TX隐藏，V不更新状态|source-only checkpoint及合法L_s派生的冻结原型／enrollment；任何target训练访问为0|
|A：适应前评估|不拟合；冻结源模型只读前向目标query|A_old；该前向属于测试，不能反过来训练、校准或选模|
|B：旧类适应|仅当前row旧类有标签target support；外部方法可用明确source replay|同一旧query的B_old；需要target的UDA对齐此时才启用|
|C：新类注册|同一旧support，才加入合法新类support；按方法权限使用历史状态|每个query面对全部6＋N_new类，C_old/C_new/H|

S不允许提前读取target无标签IQ、support、query或其BN／统计／伪标签／指标；作者默认source-target联合训练器必须拆成source-only初始化与B适应。所有方法使用同一源域，不等于必须使用相同网络、损失或U_s；每行的标签权限、增强与计算披露清楚，不解封U_s标签。

源阶段需要类原型／enrollment的ProtoNet、LoRa、Tweak，只能从L_s构建并冻结A参考；PAMF使用原联合训练默认头，不另训Phase1头。source V不用于这些持久状态。没有合法A来源时记录缺口，不能借target support构造A。

MILKR的B和C在同一冻结表示上独立解析拟合；C不继承B的系数、带宽或中心化状态。外部原生C从自己的B状态复制，ISSL的B10是明确项目扩展，原生A→C另列。闭集DA不伪造C。

配对键增加`dataset_id`：`source_contract/model_seed/dataset_id/receiver/support_draw/K/class_set/query_physical_id`。K和新增类数变化时旧support及query可追溯配对；按opaque ID连接，不按数组位置。适应状态不得跨dataset、RX、draw或K继承。

## 4. 数据与场景面板

### 4.1 综合practical residual为主，单场景与历史复现分开

综合场景包含六种practical并作为主面板；独立单场景仅为用户最新指定的low郊区、mid城区、low城区。科学协议已同步本轮覆盖条款，不再以三种leo_weak作为本轮主面板。

|面板|地位|数据与处理|
|---|---|---|
|P-Mixed|唯一主结论面板|MIX01–08，固定8套，相同概率、互异物理记录；每条样本只抽取一个practical场景和一次信道。capsule尚未构建／核实|
|P-Diagnostic|单场景解释|low郊区、mid城区、low城区各自独立物理池；与综合8套互斥，不能等权加入主平均|
|P-Legacy|历史复现|既有三景及7RX capsule `residual-noeq-ba667eee4fb061055e4c08b5`、原D92四RX两capsule保持原身份，不能冒充新综合数据|
|P-Confirm|额外独立确认（若另行固定）|新接收机／物理记录，仍按冻结综合概率；换support seed或复用query不叫独立确认|

practical处理固定`processing_route=residual, mode=post_sync, equalization_enabled=false, fs_hz=25e6`。模型高度600km、地球半径6371km；城区／郊区各0.5为中性参考先验，仰角按可见覆盖球面面积计算。综合分档low10°–30°、mid30°–45°、high45°–80°；对应六概率依次为high suburban 0.04369141011349172、mid suburban 0.07317984658235889、low suburban 0.38312874330414937、high urban 0.04369141011349172、mid urban 0.07317984658235889、low urban 0.38312874330414937。

综合条件仰角按球面面积采样并传给已有ResidualChannel geometry接口，不继续对角度均匀取样。单场景诊断保留原生各区间均匀采样（mid20°–45°）。外层六场景概率不替换内部LOS/PARTIAL/BLOCKED规则。这是明确物理假设下的参考分布，不是实际星座／业务实测频率。

详见[主表设计第1–2节](PAMF_PHASE2_MAIN_TABLES_20261028.md)、[可复算概率配置](../configs/pamf_phase2_practical_mixture_20261028.json)。本轮新增冻结DURA输入对照，资格另核；不增加clean无LEO新类主行，不将三单景成绩加权冒充混合support下的适应结果。

### 4.2 源域、旧类与新类

当前 Main split 名义源 RX 索引 `[1,3,4,6,8]`、日期 D1–D3，目标 RX `[0,2,5,7,9,10,11]`、日期 D0–D3，`L_s/U_s/V=6300/56700/27000`。加载前以真实物理记录／RX 映射核对，而不是只比索引列表或“ManySig”名称。

历史 D92 旧类 TX 为 `14-10,14-7,20-15,20-19,6-15,8-20`。历史新类固定顺序为：`11-4,3-13,12-20,5-20,20-12,17-11,14-14,2-17,4-1,19-13,19-8,2-19,8-1,14-20,13-3,1-18,20-7,7-7,2-3,18-12`。正式面板采用同一身份集合仅在当前 Phase1 类映射及未见性确证后成立；否则记录具体不匹配，不静默重标 TX。

Phase1 出现过的任何 TX 只能是 old。新类身份及全部物理记录不能进入对应源模型及其祖先训练、验证或选模。输入保持 FP32 IQ `[2,256]`、25 MHz、原有表示与归一化，不把 equalized1 数据描述成未均衡原始 WiSig。

### 4.3 单次观测及 query 隔离

- 每个物理 IQ 在一个数据面板中仅绑定一个场景、一个随机信道实例和一份 received IQ；场景之间物理 ID 不交。
- support 与 query 物理 ID 不交；K 为每类独立物理记录数。三视图签名、FFT 和数学计算 view 均不增加 K。
- 本方案 MILKR 只用原 received 单观测，不生成第二次信道、不恢复 clean 参考、不读取 source 数据或源样本级特征。
- query 不参与训练、BN、核带宽、中心化、阈值、温度、候选排序或停步；逐样本面对全部注册类。
- 原生外部方法允许的 source、历史原型、Fisher、回放或多观测行为按原生面板披露；不能借外部方法例外放宽 MILKR 的输入权限。
- 数据未变化时复用 `p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`，模型与参数变化不重复验证数据。

## 5. 主矩阵、随机性与任务数量

### 5.1 固定主综合矩阵

|维度|主设置|
|---|---|
|旧类／新增类数|old6；N_new0、2、5、10、20，总注册类6、8、11、16、26|
|每类有标签support|K1、5、10、20|
|综合数据|MIX01–08，共8套，固定同一概率，各套物理ID互斥|
|目标RX|Main split全部7个；历史D92原四RX结果独立列出|
|主模型seed|2026092701至04，所有方法相同；PAMF05–08作额外稳定性补充|
|support draw|2026092711至15，按dataset_id/RX/TX命名随机流|
|每套每RX每类|36条support候选池＋30条query；每row仅开放自己的K条|
|源角色|Ls6300、Us56700（TX隐藏）、V27000，与PAMF精确物理角色匹配|
|目标日期|D0–D3，分别报告cross_rx与cross_rx+unseen_day；D0不是未来日期|

模型seed、物理池划分、场景抽样、信道、support、evaluation、class-order分别记录；同套输入和split在方法及model seed间完全相同。class-order敏感性seed为2026102841/42/43，不混成数据套数。综合receiver_seed=2027固定RX/day硬件，不随TX或dataset生成新硬件。

数据36条候选池按四日期各9条；30条query按7/8条分日轮换，使8套合计每日期各60条／RX／类。实际缺日时在任何模型目标评分前按可用性固定统一替代规则并披露。K前缀嵌套，新增类集合2⊂5⊂10⊂20，query池不随K/draw变化。实际scene独立于TX／old-new／support-query采样，不按类别强制凑场景配额。

### 5.2 主综合数量

每个支持完整B/C的方法：A=`4×8×7=224`；B=`4×8×7×4×5=4480`；C=`4×8×7×4×5×4=17920`。共**22400个fit row／22624个阶段预测**。闭集DA只有A＋B，共**4704**。这些是预登记row数，实际不同support集合数另报，不把相同draw当独立重复。

三单场景诊断另有A84、B1680、C6720，共8400个fit row／8484个阶段预测。主综合＋三单景共30800个fit row／31108个阶段预测，但不能混成一个准确率。PAMF8模型主综合为44800个fit row／45248个阶段预测，前4模型复用，不重复。

综合8套需96096个物理观测，三单景另需36036，合计132132；每RX／类至少726条互异原记录。IQ数组理论载荷270606336字节，约258.07MiB，不含ID／容器／元数据。实际覆盖尚未核实；不足时报告缺项，不重放同一IQ构成多套。历史capsule保持原池大小与观测，不为匹配新设计而重建。

K1仍有C个support，但没有独立同类持出；不得报告虚构的K1监督持出准确率。综合8套全部报告，不按成绩挑套或追加到预期均值。

## 6. 所有已找到完整代码的外部方法

完整源码存在、正式 runner 已接入、当前矩阵已有评分是三个不同状态。具体路径与版本见配套[代码清单](PAMF_PHASE2_METHOD_CODE_INVENTORY_20261028.md)。下表为必须覆盖的实验位置；尚缺接入的完整方法列入实现工作量，不因接入工作量而从设计清单删除。

|方法家族／版本|A/B 域适应|C 新类注册|主面板要求|
|---|---|---|---|
|POSTER Fine-Tuning|原生监督微调|显式扩头 SFT 可作项目扩展|保留真正 FT，不用 NCM 代理|
|RadioNet ADA＋k-NN|ADA→冻结表示→target kNN|追加新类 memory 的项目扩展|与 ADA＋support FT＋kNN 分列；核对作者完整路径|
|RadioNet ADA＋support FT＋k-NN|作者 tuning 路径|明确的追加注册扩展|与现有简化 ADA→kNN 路径不能共用名字或成绩|
|RadioNet DF Fine-Tuning|监督 FT 辅助路线|扩头属于项目扩展|不能代表 RadioNet ADA 主方法|
|RadioNet Triplet＋k-NN|作者辅助迁移路线|按实际 enrollment 规则追加|完整作者代码存在，256 点接口须接入|
|ProtoNet CDA|独立 episode 表征＋target prototype|追加 prototype 的项目扩展|native 行与 PAMF-NCM 分开|
|MRIOR-SDA|GAD、DV-KL＋target CE|闭集，N/A|原 UDA 的有标签 support 扩展；CPL 开关明确|
|DADDA-SDA|source/target CE＋动态 MMD/LMMD|闭集，N/A|LMMD 用合法 support 标签，不读 query|
|Receiver-Agnostic Two-stage UDA＋FT|DANN→LMMD→FT|闭集，N/A|不能与 MRIOR 或 RadioNet ADA 合并|
|Feature Separation（Hu 2024）|完整特征分离＋TX few-shot FT|原生闭集，N/A|旧复现与 Hu 独立完整复现是同一论文的不同实现；版本分列|
|RA-Collab 单节点表征＋FT|单节点迁移扩展|若接注册头，明确项目扩展|重新隔离 support/query；不加入多节点融合|
|Tweak|原生多域 calibration|按其已知类校准定义，不称原生 CIL|M=1 单 query 扩展参加匹配表；原生 M=10 单列多观测面板|
|CSIL corefix|old-support SFT 为 B 扩展|通道扩展／隔离、KD、有效经验 Fisher EWC|与作者 MATLAB 原版及旧 adapter 区分|
|MoPC-HR paper semantic|old-support SFT 为 B 扩展|原型校正、原型增强、HR、序贯传递|论文语义与公开代码语义分别命名|
|Orthogonal Incremental SEI／OSC-FSCIL-SEI|B 权重校准为扩展|冻结 encoder、旧权重保持、新权重校准|保留源阶段正交几何训练|
|ISSL fixed_v1|无原生 B 则记 N/A；扩展 B 单列|base→扩头→SSL→CE transfer；另有 CE＋KD 分支|正式主比较用明确修复版；作者行为保留版作版本诊断|
|LoRa scalable/channel-robust triplet＋KNN|旧类 enrollment 的 B|冻结 extractor、追加 enrollment|短 IQ STFT 扩展、Torch 移植差异、KNN15 小 K 规则明确|

MoPC-HR、ISSL 和 RadioNet 的不同实现版本不是不同论文家族。外部 C 的完整主清单为 **7 个家族**：CSIL、MoPC-HR、Orthogonal、ISSL、LoRa、ProtoNet 注册扩展、RadioNet ADA 注册扩展；闭集方法不硬塞入此表。

通用机制与通用 DA 另设面板：Frozen DG、NCM、k-NN、linear、ridge、SFT、AdaBN、CORAL＋SFT，以及完整通用代码已存在但尚未接 RFFI 的 DANN、MCD。它们不能冒充新增 RFFI 原论文。若所选 PAMF 路径没有 BN，AdaBN 在该路径记 `N/A_no_BN`，不要制造无作用的对照。

仅有论文／名称而未找到完整算法实现的 CSCNet、Self-training＋Prototype Augmentation、DAN、DSAN、CDAN、WD、SHOT、独立 LwF/EWC/iCaRL 进入缺失表，不伪称已运行或由其他方法替代。范围是本项目实际代码，不宣称穷尽 RFFI 文献。

## 7. 原生预算、权限与公平性

### 7.1 source-only时间边界与两类原生比较

源训练S阶段只用与PAMF相同的source角色，任何target输入与统计为0。原生DADDA/MRIOR/Two-stage/RadioNet训练器若默认从起点加载target，不能直接沿用；source初始化和target对齐须拆开，后者只能在B开启。允许原生source loss，不强制改成PAMF网络；source replay在B/C明确披露。

主表 `Native-matched-input`：相同 target physical support/query、K、类别、场景和随机性，各方法保留自己的训练目标与原生更新机制。源训练预算与适应预算逐项披露；epoch 相同不等于 optimizer step、样本曝光或计算相同。

辅助表 `Budget-matched`：只在需要区分算法与计算预算时，对有迭代适应的方法补预定更新量或墙钟预算对照。不能因为某方法慢而静默缩短其预算后继续叫完整原生默认。MILKR 零 optimizer step，其特征提取和解析求解成本仍计入。

### 7.2 可直接继承的历史项目预算锚点

以下数值是已有项目实现的固定预算，不全部等同论文默认；新正式矩阵原样采用时必须使用对应版本名。

|方法|源模型／适应预算锚点|需要单列的差异|
|---|---|---|
|当前 PAMF＋MILKR|既有 E50/11100；B/C 各一次解析 fit|不重训地面模型；不将旧 E200 成绩归入此行|
|九类 native 历史系列|source E200/最后一轮；各方法实际 step 与 episode 不同|只在数据祖先契约匹配时复用；否则缺项，不回退旧模型|
|Feature Separation Hu新版主行|B100epochs，Adam0.005，batch≤256，TX路径微调／RX冻结|该B预算是项目设置；旧matched版与原论文25samples/TX分别披露|
|DADDA-SDA|B 100 次更新（100 epochs×1）|source replay、动态 LMMD 权重及 target CE 启用需实际记录|
|MRIOR-SDA|B 100 次交替更新|GAD/DV-KL、CPL 关闭的旧版本明确标记|
|Two-stage|DANN100＋LMMD100＋FT100|分阶段总更新量及各阶段 source/support 曝光分别记录|
|RadioNet ADA|历史项目1000 iterations；作者默认10000|完整作者预算10000作为预定原生完整版；1000版保留预算受限对照|
|RadioNet ADA＋FT|ADA 后 support FT30 epochs，再 kNN|新增 target CE 和成本不能隐去|
|RadioNet Triplet|作者辅助实现30 epochs，batch128，Adam1e-3|source/target 三元组采样来源必须绑定|
|RA-Collab 单节点 FT|SGD20 epochs，lr1e-5，batch32|当前旧入口需修正 support 与评估池重叠|
|CSIL matched corefix|B10 epochs；C3 epochs 掩码 SGDM|经验 Fisher、teacher、mask、旧块更新及源模型成本全计入|
|MoPC-HR matched paper|B10 epochs；每 session20 epochs，最多5新类/session|最终 prototype 校正未必直接用于当前分类器预测|
|Orthogonal matched|B10 epochs；C50 epochs|源正交几何训练不可省去|
|ISSL fixed_v1＋B10|source base300；B10epochs Adam1e-4、batch=min(32,6K)；C SSL2000／transfer300|B是项目扩展；C保留原replay与queue。原生无B及作者行为版另列，不用1epoch诊断代表完整方法|
|LoRa|作者1000epoch／source-only早停规则；现有 WiSig 仅短预算诊断|triplet、STFT、KNN memory 与本地预处理扩展必须披露|
|Tweak|现有100epoch训练与固定多域calibration接口|学习率搜索只看合法source V；M10与M1不同观测预算|

对于 source-only 选模，遵守各方法合法 source V 或预先固定轮次；不得照搬用 target test loss 保存 best 的作者默认。source 原生早停若需独立 source validation，应在既有 V 中按同一方法使用，不拆造新的方法角色。

### 7.3 权限列是主表必备字段

每个方法保存：是否需要 source IQ／源样本级 embedding／历史 exemplar／类原型／Fisher／teacher／生成原型特征／额外 target 无标签池／多个 query 观测。无额外 target 池时，UDA 只能用合法 support 的 IQ 计算对齐；不得拿 query 充当无标签训练池。DADDA/MRIOR 的 target CE 为本项目 SDA 扩展。

ISSL 的原流程包含旧训练样本重复与 replay，不能标为 support-only。CSIL 的源 Fisher、MoPC 的历史原型、LoRa 和 RadioNet 的 enrollment memory 都计入状态成本。Tweak M10 使用十个观测的平均表示，单列外部原生多观测面板，不能与单 query MILKR 直接作等输入结论。

Tweak M10 需要另行固定真实可获得的同一推理事件／观测组；不能在正式预测时读取TX真值、将同类query事后凑成10包。现有单包capsule不自动提供这种分组，缺少合法组信息时M10记N/A，只运行明示M1扩展。其物理观测数量、组间重叠和构造依据单独披露。

## 8. 共享 PAMF 表示的比较

固定同一 checkpoint 与同一 received 特征：

|ID|方法|更新范围／作用|
|---|---|---|
|S0|原生冻结默认头|A；无 target fit|
|S1|PAMF-NCM|support 类均值，冻结表示|
|S2|PAMF-kNN|support feature memory；distance、k规则预定|
|S3|PAMF-linear ridge|相同监督目标与解析正则，冻结表示|
|S4|PAMF-linear CE|仅 target classifier，固定步数与初始化|
|S5|PAMF-BranchRidge|五块表示、无乘性交互、无局部核|
|S6|PAMF-BranchInteraction|保留交互、无线性到径向的局部化|
|S7|PAMF-MILKR|完整冻结方法|
|S8|PAMF-head SFT|默认头 target CE；明确 Phase2 更新|
|S9|PAMF-tail SFT＋相同 MILKR|固定尾部模块、步数与optimizer；表示适应扩展，不叫冻结MILKR|
|S10|PAMF-full SFT＋相同 MILKR|全网更新的资源／效果参照，非主方法|

S9/S10 的 B→C 继承与从原基座 reset 对照必须分开，总更新量匹配。已有 NativeLocalRidge／Crossfit／FullCrossfit 原生尾部接口可以用于明确版本的复现，不能偷偷混入 S7。

外部算法移植到 PAMF 时另命名 `on-PAMF`。冻结 PAMF 的 MoPC 会失去 encoder drift 的原含义；不做正交源训练的头部校准不能叫完整 Orthogonal；PAMF＋kNN 不能叫完整 LoRa 或 RadioNet ADA；PAMF-NCM 不能替代独立 episode 训练的 ProtoNet。

## 9. MILKR 消融：先回答机制，再扩大组合

主消融均保留原 source checkpoint、全部合法 support、默认物理权重、target encoding、FP64求解、同一query；所有派生公式单独命名冻结配置。现有 `FROZEN_CONFIG` 会拒绝任意改参，因此下述新消融需要新配置入口，不能伪称当前 runner 已支持。

|ID|变更|主要比较／解释|
|---|---|---|
|M0|完整 MILKR|主参照|
|M1|原 z_id＋FFT96 的既有 D92|历史方法整体参照，涉及多项变化，不作单因素解释|
|M2|BranchRidge|M2→M3隔离乘性交互|
|M3|BranchInteraction|M3→M0隔离局部径向核|
|M4|去 z_id，其他公式预先固定|联合身份块的条件贡献|
|M5|去 FFT96|频谱背景信息的条件贡献|
|M6|去 t_emb|时域辅助块的条件贡献|
|M7|去 f_emb|频域辅助块的条件贡献|
|M8|去 pa_local|PA辅助块的条件贡献|
|M9|trace matching 关闭|尺度／有效正则的贡献，不能只称几何变化|
|M10|带宽改为不使用类别关系的 pairwise-distance 中位数|最近异类标签规则的作用；分类监督仍保留|
|M11|仅 z_id／仅 FFT96 的匹配局部核|单独信息来源参照，不与原容量声称完全相同|

删除块采用显式 active-block 公式，保留块重新按预定规则归一化，并记录总范数、Gram trace、维度。另留 norm-matched 零块控制用于区分范数变化；不得在看 query 之后选择哪种删除定义。M6–M8 为读出输入消融，联合 `z_id` 仍可能包含被删分支信息，不能宣称网络彻底移除了该分支。真正网络分支消融需相应 source 从零训练，另计预算，不在本轮默认追加。

不开展所有删除项的笛卡尔积。先在预定MIX01/02、固定K=1/5/20、N_new=0/5/20的机制切片上完整比较，再将核心 M0/M2/M3 和预登记必要项扩到主矩阵；不能用该切片 query 分数挑选“有效消融”后只报它们。

## 10. 标签预算、参数与可靠性

### 10.1 标签预算

K=1/5/10/20 全部真实标签可见，分别报告旧类、新类、H、floor、support总数。固定新增类数时比较 K 曲线；固定 K 时报告新类数与 support总量，不能把样本增多的收益全部归为类别变化。

现有核心要求每类相同且正的 K。`K_old≠K_new`、缺类 support 和任意类别不平衡不属于当前方法可直接运行条件。它们进入“适用边界／拒绝输入”说明，不制造一个回退算法后仍标为原 MILKR。

### 10.2 冻结敏感性点

|参数|主值|预定扫描|约束|
|---|---:|---|---|
|ridge λ|1|0.01、0.1、1、10、100|保持求和形式目标；均值目标要换算等价λ，不能混淆K依赖|
|带宽倍率 mτ|1|0.25、0.5、1、2、4|τ仍只由support估计；零τ仍走精确定义|
|FFT相对权重 w|4|0、1、2、4、8|w=0与去FFT消融去重；报告各块范数|
|trace matching|开|开／关|与λ交互解释，不随测试重新调λ补偿|

每次只扫一个参数；共享默认点不重复运行。默认值始终作为主行，不能从 target 曲线挑最高点替换主方法。若需要选参，仅用合法source研究或support内折；几何、带宽及标准化在每个train fold重新估计，held数据不能进入拟合。K=1不做伪造交叉验证。

### 10.3 support可靠性与输入质量

可预定四类有标签压力实验：

1. K曲线及5个support draw的方差、最差draw、失败率。
2. 类别对称的support标签交换噪声；在K10/20以配对交换保持每类计数，报告实际交换比例与随机性，不按具体TX加规则。
3. received IQ上预定的幅度／整体相位／轻微频偏数学变换，保留同物理ID并标为计算扰动诊断，不生成第二份LEO信道，不将其当新增独立样本。
4. support和query质量分层，只在独立评分后按既定观测质量指标报告，不用其真值选择参数。

本轮不把不稳定样本、困难RX或K1负收益从统计中排除。

## 11. 新类注册规模、顺序与旧类下降分解

### 11.1 规模

必须保留完整K×N_new表，不能只展示N_new=2。N_new=0只报告A/B；新类准确率与H为N/A，不填0。新增类按固定身份顺序前缀，另做类顺序seed `2026102841/42/43` 的敏感性，不能把support seed冒充class-set seed。

### 11.2 连续注册

在同一最终20个新类、同一最终support上比较一次加入20、10＋10、5×4、2×10。主序贯切片固定 K5/20、4 model seeds、5 support draws、7 RX、8套综合数据；方法按原生状态继承，逐session保存旧类、历史新cohort和本轮新cohort表现及累计成本。

MILKR每次都用累计合法target support重新拟合。最终support集合和特征完全相同时，其最终解应与一次联合拟合相同（仅允许已说明浮点误差）；因此最终顺序实验主要检验实现一致性，科学差异来自中间session表现和重复重拟合成本，不能将顺序不敏感包装成新的持续学习机制。

CSIL、MoPC、Orthogonal、ISSL按其原生session状态前传，禁止每个session重新从B开始冒充持续学习。允许保留的target旧support与历史exemplar、prototype和source replay分别计量。若原生方法不保留旧support，不为了“公平”偷偷增加该权限；在权限匹配扩展表另比较。

MoPC-HR的原生matched主行固定最多5新类/session，最终20类采用5×4。连续注册面板中的一次20和10＋10分别命名为`MoPC-HR-session20 extension`与`MoPC-HR-session10 extension`，真实扩大每次输入的新类数量，每个实际session仍训练20epochs，分别累计20／40epochs；5×4为80epochs，2×10为200epochs，阶段数与实际更新量一并报告。不得把输入20类后在内部偷偷拆成4个5类session，却称“一次20”。这些扩展不替换原生主行；另外的总更新量匹配比较须另列预算，不能用结果选择哪个版本代表MoPC。

### 11.3 旧类下降的可核算分解

固定全部预测后，由独立分析器计算：

\[
B_{old}-C_{old,all}
=(B_{old}-C_{old,restricted})
+(C_{old,restricted}-C_{old,all}).
\]

第一项是C头在旧类内部的决策变化，第二项是新增类竞争损失。`restricted`只在离线分析中删除C分数的新类列，绝不让正式预测器依据query真身份缩小类别集合。统一tie-break下第二项非负。

同时保存B正确→C错误、B错误→C正确、old→new误判、new→old误判及逐TX混淆。不能只用净准确率变化掩盖纠错与伤害抵消；该分解不是某模块的因果证明。

## 12. 指标、统计与预定判读

### 12.1 必报指标

- A_old、B_old、C_old、C_new、overall accuracy、Macro-F1、每类recall、全部注册类最差recall及10%分位recall。
- \(H=2C_{old}C_{new}/(C_{old}+C_{new})\)，逐单元先计算再平均；分母0时H=0。
- 适应提升B−A、注册后旧类下降B−C_old、新旧绝对差、逐阶段净纠错与错误迁移。
- RX×场景×K×新类数完整分层；日期分层仅在合法元数据可提供时进行，预测器不读取query角色。
- 每个方法、每K×N的技术失败、缺失、数值退化与耗时；不把未运行当0分，也不把失败默默删除后重算均值。

单cell最差类recall只有每类30个query时分辨率为3.33个百分点，需给样本数与二项不确定性，避免将floor微小差异过度解释。

### 12.2 聚合与不确定性

主表只来自8套综合数据：先在每model_seed×dataset_id×RX×support_draw×K×N单元形成指标与方法配对差，再对固定8套、7RX、5draw等权汇总为每个model seed结果，最后报告4模型均值±样本标准差。H逐cell计算再平均；N0独立列出。T3如需聚合正新增类数，四档等权并注明，不与N0混合。

另列8套逐套均值、套间标准差／范围、模型方差和support draw波动；实际场景频率及逐scene诊断由独立scorer在预测固定后报告。不能把8×5×4当160次独立模型训练，更不能把成千上万query当独立源训练重复。四对模型seed双侧精确符号翻转检验最小p=0.125；不依赖support伪重复宣称p<0.05。8数据套上的条件性bootstrap另列，明确共享模型／RX的边界。

三单景、历史三景、综合8套分开汇总。三单景加权准确率不代表混合support下的适应。所有8套使用同一概率，不从结果换概率、删除困难套或追加到显著为止。覆盖数据独立不等于真实星座／任务独立；当前PAMF选择背景仍披露。

### 12.3 判读目标与失败定义

历史理想目标为旧类适应提升≥10个百分点、注册后旧类下降≤1个百分点、新旧准确率绝对差≤3个百分点。它们是对结果的目标解释，不是已有达成事实，不要求K1或每个cell同时满足，也不能以低分触发技术停机或删方法。

是否支持“优于对比”分别报告：整体均值、逐K、逐N、逐场景、逐RX及资源代价。不能用H上升掩盖旧类下降，也不能将局部胜出写成所有条件领先。

## 13. 资源与部署测量

每个方法分开计量：源训练成本；Phase2加载；support特征提取；适应／注册fit；单query特征提取；单query判决；端到端received-IQ→预测；状态保存与新增注册状态。缓存读出计时另列，不能叫完整部署时延。

固定同一硬件／CPU线程／精度，warm-up 50次，固定1000次同步测量或覆盖全部实际query；报告B1的median、p95、均值，batch32只作吞吐补充。GPU同步与CPU计时范围写清，不把累计进程时间当墙钟时间。

报告总参数、实际可更新参数、mask允许更新数、训练peak GPU allocated/reserved、CPU RSS、持久状态与临时峰值、模型包／最小推理包、source/history载荷、support memory、Fisher／teacher／queue、首次部署与增量传输。未测值记N/A。

MILKR理论状态可按 `8*N*(736+C+2)+16+8*(tau存在)+8*(gamma存在)` 字节记录，最大N520/C26时为3178272字节（约3.03MiB）；Gram为2163200字节、alpha为108160字节。仅为数组层面的设计计算，不含encoder、Python对象、日志、缓存及临时QR等开销，不代替峰值实测。

适应任务均以各自全部合法support拟合；query可以流式逐样本，禁止用整batch类别配额、Hungarian、OT或query间重排提高准确率。Tweak M10等外部原生多观测成本按实际物理包数披露。

## 14. 当前代码缺口与实施顺序

### 14.1 已确认需要处理的直接问题

1. 根目录与Git发布镜像组件不齐：根目录部分DADDA/MRIOR/Two-stage/CSIL包缺失，而根runner直接import；应从完整Git承载面构造单一发布版本，不能从多个CWD临时拼运行路径。
2. 当前PAMF reader与旧D92分支导出接口需只读兼容适配及实际checkpoint无query smoke；不能宣布已兼容。
3. CSIL/MoPC历史native C在N607因`kaiming_uniform_(generator=...)`失败；确定性修复须保留版本、方法和seed，先核实既有修复是否已完成，不能重复重跑健康／完成任务。
4. Hu完整复现、Tweak、RadioNet Triplet、RA单节点、ISSL、LoRa接入正式source/capsule/arbitrary-K接口；保留算法，删除测试选模／support-query重叠等非法默认。
5. LoRa现有A已用了target enrollment；正式A需要合法source-enrollment参考，否则N/A。KNN15在K1/C6只有6个support，预定`min(15,Nsupport)`为预算兼容扩展并披露，不能运行时报错或静默换NCM。
6. ISSL诊断旧3/新3/同RX/无LEO不能填正式6＋20类成绩；原行为版与fixed_v1要分开。Tweak M10不能直接接单物理query主表。
7. 新综合practical residual面板的capsule、当前checkpoint实际物理契约、完整外部方法预算与环境路径仍需在真正发布前绑定。本设计不猜这些值。

这些属于真实运行正确性问题，不增加报告SHA、封印、签名、receipt、额外审批或重复审查。

### 14.2 执行分期（本轮仅设计）

|阶段|内容|完成证据|
|---|---|---|
|E0|登记完整方法、版本、权限、source祖先和适配缺口；选择单一完整Git承载面|逐方法代码清单与实际配置；不启动GPU|
|E1|最小同row验证：seed01、固定第一个合法RX、MIX01、K1/5、N0/5/20|聚焦负测＋真实checkpoint无query smoke；验证合法输出与数值残差|
|E2|已固定方法的8套综合主A/B/C矩阵，逐row完成后自动测试／独立评分|原生系统、共享表示主表、全部缺项和资源|
|E3|MILKR单因素机制切片与冻结敏感性；所有预定点保留|机制表、完整曲线、搜索成本与负结果|
|E4|连续注册、类顺序与压力诊断|逐session cohort曲线及累计资源|
|E5|三单景诊断、PAMF8seed补充、历史复现及独立确认|分面板报告，不混合新旧基座与场景|

对新研发候选只允许先小型source/support可证伪研究，科学阈值须在query前预定；本轮固定MILKR及完整方法比较不按小型target分数删掉弱对手或改预算。低性能是结果。

启动前按最小流程使用一个代码／配置Git提交、一个最小run登记、一次直接P0/P1审查、一次无query真实checkpoint smoke、单一不可覆盖root、一次N607 preflight及单归档传输比较。smoke通过立即继续；各row预登记预测固定后独立truth-last评分。不得增加任何白名单外发布门槛。

N607执行使用普通账户，GPU选择在发布时依据实际空闲容量固定；默认每卡一个本run训练／预测worker，计入其他owner容量，独立CPU scorer≤1。不得沿用历史“GPU2/3空闲”当作当前事实，不停止／重启其他健康任务。当前没有分配GPU、创建远程run或修改队列。

## 15. 交付表图与验收

主表的最终行／列／版本及数据访问规则以[主表详细设计](PAMF_PHASE2_MAIN_TABLES_20261028.md)为准：T0 source公平边界；T1旧类A与B@K1/5/10/20；T2各K、N_new20的A/B/C_old/C_new/H；T3注册规模曲线；T4资源。

旧类主表包含PAMF＋MILKR和9个外部DA家族；注册主表包含PAMF＋MILKR及7个外部注册家族。其余完整路线、版本诊断、共享PAMF机制、MILKR消融、8套逐套波动、三单景、连续注册与旧类损伤分解全部保留在固定附表，不能以主表去重为由漏掉已有完整方法。

保存row级CSV/JSON、混淆矩阵、opaque-ID预测与独立评分、resolved config、source lineage、实际更新／曝光及资源记录。必须含dataset_id，明确source阶段target访问为0；原始IQ、checkpoint、预测与训练日志不提交Git。对外结论限定于此参考综合分布及地面代理，不声称真实在轨验证。

设计完成的验收标准是：方法代码清单可追溯、主矩阵和适用边界明确、预算与权限不混用、指标可计算、缺口与实施顺序可直接交接。**它不代表实验已运行、外部方法已全部接入或MILKR已在当前PAMF基座上达到任何性能目标。**
