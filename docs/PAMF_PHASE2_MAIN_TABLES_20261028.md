# PAMF Phase2 主表与综合 practical residual 实验设计

r05 已合并源训练、29条完整链路、状态继承和预算，并纳入所有方法匹配的 P0 received／P1 冻结DURA双路；DURA本轮资格未核实，不微调G或自动重训F。旧文中的单输入路径计数与“不新增DURA”安排由[完整总表](PAMF_PHASE2_TOTAL_EXPERIMENT_TABLE_20261028.md)覆盖，历史结果和Phase1路线不追溯改写。

r04 新增[完整链路衔接分析](PAMF_PHASE2_DA_REGISTRATION_LINKAGE_20261028.md)：T1/T2 继续解释单阶段与原生注册，T2-J 加入闭集 DA 的原生结构扩展／NCM／JointCE 三类注册，与 PAMF 做完整 A/B/C 比较。结构保持对照保留 B 的判决收益。原 MILKR 在 C 独立重拟合，不能主张 B 核状态继承或参数协同；以下原任务量不含新增 21 个组合。

本轮 r03 总表已按用户最新指定将独立单场景收敛为 **low 郊区、mid 城区、low 城区三组**；每组全部 target support/query 使用该组同一种场景，各物理记录的随机信道实现独立且固定。综合仍为六分量、8 套，历史 MILKR 原三景不改名。完整清单、主表、22 版本、消融和修订后的工作量以[全面总实验表](PAMF_PHASE2_TOTAL_EXPERIMENT_TABLE_20261028.md)为准。

2026-10-28 修订。状态：**DESIGN_ONLY；未启动源训练、目标适应、预测或评分**。

用户进一步明确应用为**单星被动监听，不控制地面TX、不选星**。下述76.63%低仰角分布保留为宽视场空间几何参考；轨道／天线／采集条件不同可显著改变分布，不能称为实际载荷频率，也不自动改成中高主导。[完整观测分布分析](analysis/pamf_passive_elevation_20261028/REPORT.md)含单次过境、14天时间采样和视场／接收门限敏感性。本轮保持主概率不变，单场景矩阵改为用户指定三组；明确假设的参考基准仍可按既有流程开展。

本文件落实用户最新要求，覆盖前版“以三种 leo_weak 为主”的安排：**以综合 practical residual 场景为主，固定8套数据抑制随机波动，三个指定单场景只作诊断；所有方法先在相同训练源域建立源模型，到目标适应阶段才接触合法target support。**现有有标签MILKR仍是主方法，不新增无标签方法研发。

配套：[完整机制设计](PAMF_PHASE2_MILKR_EXPERIMENT_DESIGN_20261028.md)、[代码清单](PAMF_PHASE2_METHOD_CODE_INVENTORY_20261028.md)、[完整矩阵](../configs/pamf_phase2_milkr_experiment_design_20261028.json)、[综合概率与8套数据配置](../configs/pamf_phase2_practical_mixture_20261028.json)、[科学协议](PROJECT_PROTOCOL.md)。

## 1. 主场景概率：来自明确几何假设，不能伪称实测频率

现有practical模型固定`h=600km`、地球半径6371km、载频2.45GHz，本轮IQ采样率25MHz，使用`residual/post_sync/noEQ`。原生六场景仰角范围为low10°–30°、mid20°–45°、high45°–80°，城区／郊区各一组。

采用“在10°–80°可见覆盖环内，地面位置按球面面积均匀”的参考几何。设地面仰角为e，星下点地心夹角为ψ：

\[
\psi(e)=\arccos\!\left(\frac{R}{R+h}\cos e\right)-e,
\quad
P(a\le e<b)=\frac{\cos\psi(b)-\cos\psi(a)}{\cos\psi(80^\circ)-\cos\psi(10^\circ)}.
\]

这使低仰角占比明显较高。为避免原生low/mid区间20°–30°重叠造成重复计数，综合采样采用互斥分档：low10°–30°、mid30°–45°、high45°–80°；全部落在对应原生场景允许范围内。城区／郊区各0.5是缺少部署地理与业务统计时的**中性覆盖先验**，不是从模型性能反推或从人口比例推算。

|综合场景分量|抽样概率|综合条件仰角范围|
|---|---:|---|
|high suburban／`practical_high`|4.369141%|45°–80°|
|mid suburban／`practical_mid`|7.317985%|30°–45°|
|low suburban／`practical_low_suburban`|38.312874%|10°–30°|
|high urban／`practical_high_urban`|4.369141%|45°–80°|
|mid urban／`practical_mid_urban`|7.317985%|30°–45°|
|low urban／`practical_low_urban`|38.312874%|10°–30°|

表中百分比仅展示，配置使用未舍入值。总低／中／高仰角分别为76.625749%／14.635969%／8.738282%。这套权重有可复算的几何基础，但不等同于某星座的时间占比、服务卫星调度分布或真实流量分布。仰角统计会随轨道倾角、站点纬度与高度变化，参见[原始仰角统计研究](https://www.mdpi.com/2076-3417/13/7/4405)；本文球面覆盖假设和数值是明确的设计选择，不声称照搬该论文结论。

### 1.1 每条样本怎样生成

1. 用固定概率向量为物理样本抽取一个场景，独立于TX、old/new、support/query、方法、模型seed与分类难度。
2. 在该互斥仰角档内按球面面积条件分布采样，而非继续均匀采样角度。具体为在两个边界的`cos(ψ)`之间均匀取值，再反推e、斜距和既有圆轨道range-rate。
3. 沿用`receiver_for_session`及固定receiver_seed=2027；硬件只绑定真实RX/day，不绑定TX或方法。用`ResidualChannel(..., geometry=...)`对该物理IQ只处理一次，其他practical衰落、噪声、同步残差、IQ补偿和AGC参数保持原版。
4. 六场景的外层概率与practical内部LOS/PARTIAL/BLOCKED三状态概率分开。内部仍按既有`state_probabilities(e, environment)`计算，不再乘一套人为阻塞率。
5. 生成后封存received IQ。实际scene、geometry和state只供builder／独立scorer诊断，不能作为预测器的场景真值或oracle分支选择输入。

原生单场景诊断保留原有均匀角度采样（mid仍20°–45°）；综合采样是明确定义的几何条件采样版本。不能用三个指定单场景准确率按上述比例加权，冒充真正混合support和混合query的适应结果。

现有`ResidualChannel`接受外部geometry，但batch入口尚未提供逐样本综合geometry接口，需要实施时接入；本轮只完成概率计算与设计，没有生成IQ。

## 2. 8套综合数据怎样避免随机波动

固定`MIX01–MIX08`，各套使用相同概率向量，独立物理记录分区、场景抽样和信道随机流。**8套之间以及三个指定单场景诊断之间，物理ID均互斥**。不对同一原始IQ生成8份不同信道观测。

|项目|固定设计|
|---|---|
|综合数据套数|8，全部进入主表，不挑最好套，不按成绩继续追加到“稳定”为止|
|物理记录划分seed|2026102800；先进行与方法无关的全局不相交分配|
|各套场景seed|2026102811至2026102818|
|各套信道seed|2026102821至2026102828|
|各套query排列seed|2026102831至2026102838|
|虚拟接收机硬件seed|2027，各套保持同一RX/day硬件，避免凭空制造8套硬件域|
|主模型seed|2026092701至2026092704|
|support draw seed|2026092711至2026092715，按dataset_id/RX/TX命名随机流|
|每套每RX每类|36条support候选池＋30条query；每row只开放自己的K条support|
|共享范围|同套IQ／split在所有方法与模型seed间相同；改变模型seed不改变信道|

36条support候选池按D0–D3各9条构造；query的30条按7/8条分日轮换，8套合计每个日期各60条／RX／类。没有完整日期覆盖时，只能在任何模型目标评分之前按可用性固定统一替代分配，并如实披露；不能因某日期表现差而换记录。随机信道分量在日期和类别内仍按同一全局概率独立采样，不对每类强行凑六场景配额。

三个指定单场景采用各自独立物理池，用来解释失效位置；主表不把“8套综合＋3个单景”等权平均成一个分数。旧三景capsule保留原值，不能复制到新8套综合，不能把同一target support状态跨套继承。

主综合池需`8×7×26×66=96096`条固定物理观测；三单景另需36036条，合计132132条。对应IQ数组理论载荷为270606336字节（约258.07MiB），每RX／类合计需726条互异原记录。覆盖不足应报具体缺项，不能静默重用物理ID。历史数据复用时须在分配前排除其物理ID；尚未核实这项完整覆盖，本文件不宣称capsule已准备好。

## 3. 所有方法统一的S→A→B→C时间边界

|阶段|允许数据|禁止行为|状态与输出|
|---|---|---|---|
|S：地面源训练|与PAMF相同源RX、源日期、6旧TX、相同L_s/U_s/V物理角色|任何target IQ、target BN／统计／伪标签／support、query及指标用于训练或选模|仅source训练并冻结source checkpoint；ground原型／enrollment只从L_s产生|
|A：适应前测试|冻结源模型只读前向目标query|更新模型／统计／头，或用A评分选源模型与B超参|A_old。A的目标前向只属于评估，不是源训练访问|
|B：旧类域适应|当前row旧类有标签target support；外部方法可按声明使用合法source replay|target new support、query拟合、额外target U池、跨dataset状态继承|B_old；同一旧query与A配对|
|C：新类注册|保留当前旧support，才加入新类有标签support；原生允许的历史状态|query角色路由、真类别配额、query更新、把未见TX提前放入S|C_old/C_new/H；全部注册类共同竞争|

S阶段source RX=[1,3,4,6,8]、day=D1–D3、L_s=6300、U_s=56700（TX隐藏）、V=27000；target RX=[0,2,5,7,9,10,11]、day=D0–D3。实际复用核对物理ID与完整祖先，不只核对名义索引。

“同一源域”不强制相同网络或损失。ProtoNet可做source episode，LoRa/Tweak可做source triplet，CSIL/Orthogonal可保留source几何目标；但episode内所谓query只能是source L_s记录，不能偷换成部署query。source U_s是否使用、增强配方、source更新量和额外源状态逐行披露，U_s标签不因外部方法而解封。PAMF-Sup＋MILKR是监督权限匹配桥接，完整PAMF仍保留其源SSL身份。

原生UDA默认从头联合source/target训练的代码必须拆开：S仅source初始化；DADDA/MRIOR/Two-stage/RadioNet的真实source-target对齐到B才启用。可以在B内部将合法target support的IQ当无标签输入计算对齐项，但不能从query池补齐target batch。source replay仅来自同一合法L_s，不增加源标签量。

`comparison_suite/source.py`已有source-only训练与`target_access=False`记录；其9家族源训练入口与独立pipeline结构可复用。它不证明所有独立作者入口均满足此边界。独立UDA模板的target-test选模、RA旧support/query重叠、ISSL诊断内用target训练base等行为必须先去除／重绑，才可进入本表。

若底层原始pickle同时装有source和target，独立角色导出层应先准备仅含source L_s/U_s/V的训练材料；S训练器不持有target数组。现有source.py的角色训练结构不等于已证明其底层loader不会解包全库。检查实际loader与源材料边界后，才能确认“源阶段未接触target”，不能只相信一个布尔字段。

## 4. 主表T1：旧类域适应

每行都由自己的source-only模型开始。表头固定为：

`方法／版本 | 源训练方式 | B是否回放source | A_old | B@1 | B@5 | B@10 | B@20`

另配同顺序的增益表`Δ@K=B@K−A_old`，逐K报告；共享PAMF的A应复用，原生系统不能用PAMF的A代替自己的A。所有值先在相同dataset/RX/draw内配对，再汇总。

|行ID|正式方法行|S阶段输入／目标|B阶段开放target后做什么|
|---|---|---|---|
|D0|当前PAMF＋MILKR|同一源域，Tri-SIG-TIME SSL E50／11100，冻结encoder|旧support一次解析局部核岭回归；不回读source|
|D1|POSTER Fine-Tuning|同一L_s的source监督训练；冻结合法源模型|support-only训练原方法声明的Dense层，10epochs|
|D2|RadioNet ADA＋support FT＋kNN|source-only训练作者ADA专用encoder/classifier，不做target域对齐|source L_s＋target support进行ADA10000iterations；support FT30epochs，再kNN(k=K)|
|D3|ProtoNet CDA|仅source L_s的独立episode训练；源类原型冻结|旧support类均值＋平方欧氏距离，0梯度更新|
|D4|MRIOR-SDA（CPL-off）|source-only GAD网络初始化，任何需要真实target的目标关闭|source L_s＋旧support；保留GAD/DV-KL及support CE，100次交替更新|
|D5|DADDA-SDA|source-only多尺度网络监督初始化|source L_s＋旧support CE＋动态MMD/LMMD，100更新|
|D6|Two-stage SDA＋FT|source-only初始化；DANN/LMMD target分支不提前启用|source/support DANN100＋LMMD100＋support FT100|
|D7|Hu2024 Feature Separation|同一source L_s，TX/RX分离；新完整复现版本固定|合法旧support TX标签微调，冻结RX路径；预定100epochs、Adam0.005、batch≤256；该预算为项目固定设置|
|D8|RA-Collab single-node FT|同一source，保留单节点source表征目标|support-only全网SGD20epochs、lr1e-5、batch32；关闭对抗，不加多节点融合|
|D9|Tweak M1 extension|同一source L_s做triplet；source centroid/radius冻结|旧support校准centroid/radius，M1逐包判决；不使用原生M10组包|

原生论文版本、source预算和source U_s权限另列T0；相同source角色不等于相同训练计算。历史RadioNet1000iteration、ADA-only、DF-FT、Triplet、Feature Separation旧版本、Tweak M10等完整路线在预定辅助表保留，不借主表版本去重而删除它们。

source源决策补齐：ProtoNet用S末L_s类原型；LoRa用S末L_s enrollment；Tweak用S末source calibration；其他方法用自己的联合训练默认source头。不能用target support制造A；PAMF不额外训练地面分类头。若独立代码缺少合法A来源且不能从允许的source状态构建，只能报N/A和缺口，不能猜数。

## 5. 主表T2：域适应后新类注册

主表包含以下8行，统一最终旧6类＋新20类，在K=1/5/10/20分别成子表；**K5是预定少样本主展示点**，其他K完整展示，不根据结果改主点。

`方法／版本 | A_old | B_old | C_old | C_new | H | B−A | B−C_old`

新旧绝对差、Macro-F1、最差类recall与逐类混淆另配同序表，不挤成无法阅读的宽表。

|行ID|方法|B定义|C定义及继承|
|---|---|---|---|
|R0|PAMF＋MILKR|old-only解析fit|用old＋new全部合法support独立重拟合；不继承B的核状态|
|R1|CSIL empirical-Fisher corefix|原matched B10epochs target SFT|从自己的B复制状态，保留通道隔离／mask／KD／合法source Fisher；C3epochs，版本差异明示|
|R2|MoPC-HR paper-semantic|原matched B10epochs，并进行声明的原型校正|继承自己的B；新20类按5×4原生session，每session20epochs；prototype augmentation＋HR与校正向后传递|
|R3|Orthogonal Incremental SEI|原matched B10epochs权重校准|继承B的旧权重；冻结encoder，新权重均值初始化并校准50epochs|
|R4|ISSL fixed_v1＋B10 extension|source base6类冻结后，旧support全网Adam1e-4微调10epochs，batch=min(32,6K)，不以query早停|以此B为当前base／teacher进入扩头→SSL→CE transfer；保留合法旧训练数据replay，完整固定阶段预算与queue。该B是明确项目扩展|
|R5|LoRa triplet＋KNN enrollment|冻结source triplet encoder，以旧target support替换旧类enrollment|继承旧target enrollment，追加new support；KNN15，N不足15时k=min(15,N)为明确预算兼容扩展|
|R6|ProtoNet注册扩展|D3自己的B|旧原型不变，追加新类support原型，全部类竞争|
|R7|RadioNet ADA＋FT＋kNN注册扩展|D2自己的B|冻结B encoder，继承旧target kNN memory，只追加new memory，不重做ADA|

ISSL原生A→C（没有B）与作者行为保留版都在辅助表保留；主表R4明确包含B扩展，不能借此声称原论文有旧类专属适应。C的SSL／transfer为现有模板2000／300epochs，source base模板300epochs；按同一合法训练池确定每epoch实际step，披露replay与曝光成本，不能只跑1epoch诊断后标完整方法。

LoRa在K1的KNN15兼容规则可能造成投票退化，应按实报告，并预定增加`k=min(15,K)`的K-adaptive辅助控制；不能因该控制分数高就替换主行或把主行低分当实现故障。

闭集MRIOR/DADDA/Two-stage/Feature Separation等不具有原生C的，**不填入T2**。如另接注册头，单列扩展，不用它们的B成绩代替新类结果。

## 6. 主表T3／T4与完整辅助材料

|表|固定内容|目的|
|---|---|---|
|T0|逐方法source角色、源更新／epoch、U_s权限、增强、参数与source历史载荷；S期间target访问为0|说明公平数据边界与预算差异|
|T1|旧类A与B@K1/5/10/20；配对增益表|检验真实旧类适应|
|T2|每K、N_new20的A/B/C_old/C_new/H及旧类损伤|比较完整联合注册|
|T3|K5／20下N_new2/5/10/20的H；配套C_old/C_new矩阵，完整4K×5N放附表|展示规模变化，避免只选容易的新类数量|
|T4|K5／20、N_new20的适应／注册成本、B1端到端median/p95、状态／内存／传输|展示部署代价；source成本另列|
|A1|RadioNet各完整辅助路线、旧Feature Separation、CSIL/MoPC/ISSL不同版本、Tweak M10适用性|覆盖已有完整代码，不把版本当新论文|
|A2|同PAMF的NCM/kNN/Linear/Ridge/SFT/AdaBN/CORAL-SFT，以及DANN/MCD移植|分离表征与适应器贡献；无BN时AdaBN为N/A|
|A3|MILKR分支／交互／局部核／带宽／trace／正则消融|机制归因，默认方法不因target扫描改变|
|A4|8套逐套值、实际场景频率、逐RX／日期／实际scene分层|显示随机波动及失效位置|
|A5|三单场景原生采样诊断、连续注册日程、旧类损伤分解|解释综合结果，不替代综合主表|

每个主表数值先在`model_seed×dataset_id×RX×support_draw×K×N`内形成同row指标，再对8套、7RX、5draw等权汇总成每个model seed的值，最后报告4个模型seed的均值±样本标准差。H先在cell计算再平均，不从两个平均准确率倒算。

另报告8套的逐套均值、套间标准差／范围和数据集不确定性；模型seed方差、dataset方差与support draw方差分别说明。8套×5draw×4模型不是160次独立模型训练；共享RX、模型与query的相关性必须保留。四模型配对差不能通过重复query把p值人为压低。

综合主表的得分直接来自混合IQ与混合support；三单景平均、按概率重加权三单景结果、以及同一原IQ的8次信道重放，都不能替代它。

环境占比敏感性预定为urban20%／50%／80%。先对固定预测做明确的query分布重加权，只解释“参考support不变时的评价分布变化”，不冒称已经完成另一support分布下的重新适应；如需完整适应压力实验，必须另行固定物理池、概率与run，不能由结果临时选择。

## 7. 主矩阵、任务量与实施边界

|维度|主综合矩阵|
|---|---|
|旧类／新增类|old6；N_new0/2/5/10/20|
|标签预算|K1/5/10/20，全部为合法真实target support标签|
|模型seed|4个公共固定seed；PAMF05–08作为额外稳定性补充|
|dataset|固定8套综合，每套概率相同且物理记录独立|
|RX／support draw|7RX×5draw|
|每类query|每套每RX30条；同K与draw复用其冻结query|

每个完整方法：A=`4×8×7=224`个唯一源参考；B=`4×8×7×4×5=4480`；C=`4×8×7×4×5×4=17920`；B＋C合计**22400个row**，A＋B＋C合计**22624个阶段预测**。闭集DA只做A＋B，共**4704**。这是逻辑row数量，实际不同support集合数另报。

三单景诊断每完整方法另有B1680、C6720、A84，共8484个阶段预测；主综合与单景合计30800个fit row／31108个阶段预测，但不得合并成主准确率。当前PAMF8模型的综合稳定性总量为44800个fit row／45248个阶段预测，前4模型复用。

先用一个模型、MIX01、一个固定RX、K1/5、N0/5/20完成小型合法性与数值验证，再运行已固定方法完整矩阵。不得按预检target分数删除弱对手、改变概率、重挑数据套或缩短某个方法的预算。源S模型可在不同dataset间复用；B/C状态绝不能跨dataset、RX、draw或K继承。

实施顺序为：①source-only资格核对与源模型冻结；②独立数据builder一次性构建综合／单景received pools；③A/B/C隔离适应与冻结预测；④独立truth-last评分；⑤主表、逐套波动和资源报告。builder可以独立准备target benchmark，但source训练器不能读取它的IQ、标签、统计或报告；这不授权在S阶段适应target。

本轮只修订设计／协议／概率配置。综合builder、当前PAMF exporter、完整外部方法接口和各运行时路径仍需在真正实施时完成；不创建远程run、不自动新训练任何源模型、不恢复已暂停队列。

## 8. 计算复用与实施优化

保留完整预登记矩阵，不靠删困难套、减弱对手或隐去原生预算省计算。按以下方式复用真正等价的工作：

1. 每方法／model seed只建立一次合规source checkpoint，跨8综合与3单景复用；source训练、L_s原型／enrollment／Fisher只做一次。source材料先与target物理隔离。
2. received IQ与split只构建一次，供全部方法共用。对冻结encoder，特征可逐样本计算并缓存；query特征不能用于拟合任何共享统计。对target微调后的encoder重新提取，禁止误用源模型缓存。
3. A按model/dataset/RX缓存一次，共224个主综合参考；B按model/dataset/RX/K/draw只做一次，供四档新增类数的C独立分叉。不能从C2继续训练C5/C10冒充独立配对任务。
4. MILKR在各row自身权限内复用原始特征／距离；每次仍只用该row support重算带宽、中心化、迹尺度与解析解。子矩阵不能读取额外类别的标签或状态；不能切片已拟合alpha替代新fit。
5. 同模型权重可常驻设备，但每个B从封存source状态恢复，包含BN和所有buffer；optimizer、teacher、prototype与queue按该方法规定初始化，不继承上个dataset的适应状态。C只继承本row自己的B（MILKR除外）。
6. CPU解析fit与GPU特征提取／迭代适应分队列。未来发布时按实际CPU/GPU容量固定并发，计入其他owner；不根据当前占用猜倍数加速，不操作健康其他任务。
7. 真正执行前用source／合法support的小型无query计时测每阶段step成本，再按本文已固定row与完整预算估算总成本。ISSL的2000epochs SSL、RadioNet的10000次ADA等应单独列预算，不能把其执行成本按MILKR一次fit估计。若另设计budget-limited版本，必须提前命名并与完整预算分表，不能运行中缩短后继续叫原生完整版。

上述优化不改变物理数据、标签数量、场景概率、科学公式或完整方法机制。原始池覆盖、当前PAMF分支接口与外部方法source-only接入仍属于尚待实施的具体工作。
