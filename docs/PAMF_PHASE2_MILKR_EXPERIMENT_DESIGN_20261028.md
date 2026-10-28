# PAMF Phase2：MILKR 有标签域适应与新类注册完整实验设计

日期：2026-10-28。状态：**DESIGN_ONLY，尚未启动任何训练、预测或评分**。

本设计落实用户本轮范围：**仅设计现有有标签 MILKR 的完整 Phase2 实验**。不加入无标签适应、匿名新类发现、Phase3 unknown 拒识、多节点协同或联邦学习。K-shot 标签预算曲线属于现有有标签任务，不把它改称无标签研究。

配套文件：

- [方法代码清单与版本边界](PAMF_PHASE2_METHOD_CODE_INVENTORY_20261028.md)：逐方法源码、入口、权限、现有完成度和未接入项。
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

|状态|可用于拟合的目标信息|输出与含义|
|---|---|---|
|A：适应前|不使用本 row 的 target support|冻结源模型对全部 6 个旧类的预测，得到 A_old|
|B：旧类适应后|仅该 row 旧类有标签 support|同一旧类 query，得到 B_old|
|C：新类注册后|保持同一旧类 support，加入当前注册新类 support|每个 query 对全部 6＋N_new 类竞争，得到 C_old、C_new、H|

MILKR 的 B 和 C 是从同一冻结表示分别拟合的两个解析头。C 不继承 B 的系数、带宽或核中心化状态，不能称为神经网络持续微调或已经实现抗参数遗忘。

配对键为 `source_contract/model_seed/receiver/scenario/support_draw/K/class_set/query_physical_id`。K 或新增类数改变时，只要属于同一预登记切片，旧类 support 与旧类 query 物理身份保持可追溯匹配。按 opaque ID 连接，不按数组位置相减。

原生方法没有 A 的现成可用源判决器时，记 `N/A_no_source_decision`；不要用 target support 构建一个分类器再叫 A。没有旧类适应阶段的原生方法 B 可记 N/A；另跑显式 `+target-SFT` 扩展时单列，并同时保存原生 A→C 结果。

闭集 DA 方法只参加 A/B 主表，C 为 `N/A_closed_set_method`。若另接通用注册头，名称必须是 `方法名 + explicit registration extension`，不能填成原论文新类注册成绩。

## 4. 数据与场景面板

### 4.1 正式协议面板与历史复现分开

当前协议正文将 Phase2 单次观测场景列为 `leo_clear_weak / leo_low_elev_weak / leo_rain_weak`；历史 D92 已验证 capsule 则使用 `practical_high / practical_mid / practical_low_urban`。两者不能改名、拼接均值或继承成绩。

本设计采用：

|数据面板|地位|场景与数据状态|处理|
|---|---|---|---|
|P-Formal|当前 PAMF＋MILKR 的正式协议主面板|三种 leo_weak；**本次未读取或验证到可直接使用的匹配 capsule**|执行前先查既有有效 capsule；存在即复用，否则对新增接收观测一次性构建验证。不得拿 practical capsule 改标签替代|
|P-Legacy|历史方法复现与连续性面板|既有 practical 三景；7 RX capsule `residual-noeq-ba667eee4fb061055e4c08b5`；D92 原始 4 RX 两 capsule 另保留|原 received IQ、split 和历史许可原样复用，标为透明重复基准。当前模型能否加载须另核对 checkpoint 数据契约|
|P-Confirm|独立泛化补充|未用于开发／选模的合法 receiver 或物理记录，仍为三种 leo_weak|先按覆盖而非性能规则固定，不把换 support seed、复用 query 叫独立验证|

这项安排不修改协议，不改变任何既有实验场景。Phase1 的 clean＋六 practical 默认不自动扩展为 Phase2 七景；本轮不设计无 LEO 新类正式成绩，也不新增 DURA 路线。若将六 practical 作为未来正式主面板，应先明确对应科学协议修订，不能在启动器中静默扩大。

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

### 5.1 固定主矩阵

|维度|主设置|
|---|---|
|旧类数|6|
|每类有标签 support K|1、5、10、20|
|新增类数|0、2、5、10、20，总注册类数 6、8、11、16、26|
|目标 RX|Main split 全部 7 个；历史 D92 原始 4 RX 结果独立列出|
|场景|每个面板自身三景，不跨面板合并|
|主比较模型 seed|2026092701、02、03、04；各方法相同列表，不按性能筛 seed|
|PAMF 稳定性补充|既有 05–08，合成 8-seed PAMF-only 结果；不能与外部 4-seed 当作完全配对|
|support draw seed|2026092711、12、13、14、15|
|物理 query 数|目标每类 30，沿用既有有效 split；实际不足必须报缺项，不能换掉低分 RX／TX|
|正式面板 support 候选池|每 RX／场景／类固定36个物理记录，供5个draw独立排列取前K；每row只可用自己的K个，其余候选池记录不可参与拟合|
|采样规则|同类一次排列取前 K，K 前缀嵌套；新增类 2⊂5⊂10⊂20；query 池固定|
|日期口径|D1–D3 cross_rx 与 D0 cross_rx+unseen_day 分开报告；D0 不是未来日期|

数据／信道／support／model／class-order／evaluation 六类随机性分别记录。旧 capsule 的 data/augmentation/evaluation seed 为 `2026092705/2026092707/2026092706`，只属于旧 capsule；正式新 capsule 不凭数字相同假定相同划分。新类集合前缀固定，不按识别难度选类。

### 5.2 数量与复用

每个完整方法、每个数据面板：

- A：`4×7×3=84` 个唯一模型／RX／场景参考预测。可复用，不随 K 和 support seed 重做。
- B：`4×7×3×4×5=1680` 个 old-only support fit。
- C：`4×7×3×4×5×4=6720` 个正新增类数 fit。
- B＋C 共 **8400** 个预登记适应／注册row；A＋B＋C 共 **8484** 个去除A重复后的阶段预测记录。support draw允许重叠，实际不同support集合数须另报；若两个draw恰好相同，不把它们解释为独立support重复。序列化文件若重复A，须另外报告去重数量。
- 闭集 DA 方法只有 A＋B，共 **1764** 个唯一阶段预测记录，不伪造 C。
- 4-RX 历史面板的 B＋C 为 `4×4×3×4×5×5=4800`，不能与 7-RX 主矩阵混为同一覆盖。
- PAMF 的 8-seed 补充为 B=3360、C=13440、A=168，总 16968；其中前 4 seed 复用主实验，不重复。

新正式 capsule 采用每类36个support候选池＋30个query，以允许5个draw在K20仍有抽样变化；沿用既有builder的池式思路，不要求draw间support互斥。26类×7RX×3场景×66条，共 **36036** 个固定物理观测；每RX／类至少需198条可用记录。IQ数组理论载荷为 **73801728** 字节，约70.4MiB。每row仍只暴露K个有标签support，不得跨draw拟合或继承适配状态。这是不含ID、容器及其他元数据的设计估计，不是已构建产物或实测部署大小。历史capsule的池大小以自身manifest／builder记录为准，不重建来套本设计。

K=1 时整个任务仍有 C 个 support，不是总共一个样本；没有独立同类持出数据，不能报告 K=1 的监督持出准确率。

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

### 7.1 两类原生比较

主表 `Native-matched-input`：相同 target physical support/query、K、类别、场景和随机性，各方法保留自己的训练目标与原生更新机制。源训练预算与适应预算逐项披露；epoch 相同不等于 optimizer step、样本曝光或计算相同。

辅助表 `Budget-matched`：只在需要区分算法与计算预算时，对有迭代适应的方法补预定更新量或墙钟预算对照。不能因为某方法慢而静默缩短其预算后继续叫完整原生默认。MILKR 零 optimizer step，其特征提取和解析求解成本仍计入。

### 7.2 可直接继承的历史项目预算锚点

以下数值是已有项目实现的固定预算，不全部等同论文默认；新正式矩阵原样采用时必须使用对应版本名。

|方法|源模型／适应预算锚点|需要单列的差异|
|---|---|---|
|当前 PAMF＋MILKR|既有 E50/11100；B/C 各一次解析 fit|不重训地面模型；不将旧 E200 成绩归入此行|
|九类 native 历史系列|source E200/最后一轮；各方法实际 step 与 episode 不同|只在数据祖先契约匹配时复用；否则缺项，不回退旧模型|
|Feature Separation 旧 matched 版|B 100 epochs|Hu strict 版原论文每 TX25 FT 样本与 K=1/5/10/20 是数据预算扩展|
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
|ISSL|已有原预算模板 base300／SSL2000／transfer300；诊断仅1epoch|正式 arbitrary-K 接口和具体阶段预算需落入配置；不能用1epoch诊断代表完整方法|
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

不开展所有删除项的笛卡尔积。先在固定 K=1/5/20、N_new=0/5/20 的机制切片上完整比较，再将核心 M0/M2/M3 和预登记必要项扩到主矩阵；不能用该切片 query 分数挑选“有效消融”后只报它们。

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

必须保留完整K×N_new表，不能只展示N_new=2。N_new=0只报告A/B；新类准确率与H为N/A，不填0。新增类按固定身份顺序前缀，另做类顺序seed `2026102801/02/03` 的敏感性，不能把support seed冒充class-set seed。

### 11.2 连续注册

在同一最终20个新类、同一最终support上比较一次加入20、10＋10、5×4、2×10。主序贯切片固定 K5/20、4 model seeds、5 support draws、7 RX、3 scenes；方法按原生状态继承，逐session保存旧类、历史新cohort和本轮新cohort表现及累计成本。

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

主指标先在每个model seed内，对固定RX、场景、support draw及正新增类规模等权平均，再报告4个model seed的均值±样本标准差。正新增类四档等权；N0独立列出。保留按实际query数合并的总体表，但标注为另一口径。

方法差值先按完整配对键计算，再汇总model-seed差值。给出四seed配对差与t区间时明确样本小、共享测试集及条件性解释；描述性hierarchical／多维cluster bootstrap可另列，不把成千上万support重采样任务当作成千上万独立训练重复。四对模型seed的双侧精确符号翻转检验最小p为0.125，不能据此声称p<0.05。需要更强检验时，先固定扩seed矩阵再执行，不按结果追加到显著为止。

类排列、嵌套K、复用query会产生相关性。独立确认必须在未用于研发或选模的数据上完成，换support draw不构成新盲测。所有预定对比与负分层保留，探索性多重比较注明性质。

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
7. 新正式LEO面板的capsule、当前checkpoint实际物理契约、完整外部方法预算与环境路径仍需在真正发布前绑定。本设计不猜这些值。

这些属于真实运行正确性问题，不增加报告SHA、封印、签名、receipt、额外审批或重复审查。

### 14.2 执行分期（本轮仅设计）

|阶段|内容|完成证据|
|---|---|---|
|E0|登记完整方法、版本、权限、source祖先和适配缺口；选择单一完整Git承载面|逐方法代码清单与实际配置；不启动GPU|
|E1|最小同row验证：seed01、固定第一个合法RX、clear场景、K1/5、N0/5/20|聚焦负测＋真实checkpoint无query smoke；验证合法输出与数值残差|
|E2|已固定方法的主A/B/C矩阵，逐row完成后自动测试／独立评分|原生系统、共享表示主表、全部缺项和资源|
|E3|MILKR单因素机制切片与冻结敏感性；所有预定点保留|机制表、完整曲线、搜索成本与负结果|
|E4|连续注册、类顺序与压力诊断|逐session cohort曲线及累计资源|
|E5|PAMF8seed补充、历史复现及独立确认|分面板报告，不混合新旧基座与场景|

对新研发候选只允许先小型source/support可证伪研究，科学阈值须在query前预定；本轮固定MILKR及完整方法比较不按小型target分数删掉弱对手或改预算。低性能是结果。

启动前按最小流程使用一个代码／配置Git提交、一个最小run登记、一次直接P0/P1审查、一次无query真实checkpoint smoke、单一不可覆盖root、一次N607 preflight及单归档传输比较。smoke通过立即继续；各row预登记预测固定后独立truth-last评分。不得增加任何白名单外发布门槛。

N607执行使用普通账户，GPU选择在发布时依据实际空闲容量固定；默认每卡一个本run训练／预测worker，计入其他owner容量，独立CPU scorer≤1。不得沿用历史“GPU2/3空闲”当作当前事实，不停止／重启其他健康任务。当前没有分配GPU、创建远程run或修改队列。

## 15. 交付表图与验收

必须输出以下完整材料：

|编号|表／图|内容|
|---|---|---|
|T1|方法与权限总表|所有完整代码方法、版本、source/history/target权限、native或扩展、完成度|
|T2|旧类适应主表|每K的A、B、B−A、Macro-F1、floor及成本|
|T3|新类注册主表|完整K×N的C_old/C_new/H、B−C_old、新旧差距|
|T4|共享表示与MILKR消融|同一PAMF、相同support与读出控制、完整负结果|
|T5|逐RX／场景／日期|主表均值背后的范围，原生权限面板分开|
|T6|资源与状态|IQ端到端、缓存读出、适应／注册、内存／状态／传输|
|F1|A→B→C三阶段图|旧类适应收益、注册竞争、原生和扩展分组|
|F2|K×N热图及预算曲线|各方法H、C_old/C_new及总support数|
|F3|机制与参数图|交互／局部核增量、λ／τ／FFT权重敏感性|
|F4|连续注册曲线|最初旧类、各历史新cohort、当前新类及累计成本|
|F5|错误变化图|纠错／损伤计数、old↔new混淆与下降分解|

保存row级CSV/JSON、完整混淆矩阵、opaque-ID预测与独立评分、resolved config、source lineage引用、实际更新／曝光日志和资源记录。原始IQ、checkpoint、预测、训练日志留在受控产物目录，不提交Git；版本化方法配置与公开汇总按项目流程交付。

设计完成的验收标准是：方法代码清单可追溯、主矩阵和适用边界明确、预算与权限不混用、指标可计算、缺口与实施顺序可直接交接。**它不代表实验已运行、外部方法已全部接入或MILKR已在当前PAMF基座上达到任何性能目标。**
