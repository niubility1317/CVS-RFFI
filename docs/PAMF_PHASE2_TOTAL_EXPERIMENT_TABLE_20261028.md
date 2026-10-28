# PAMF Phase2 完整总实验表：源训练、适应、注册与 DURA 接入

2026-10-28，r05，覆盖此前 r03 总表与 r04 衔接补充。**DESIGN_ONLY：本轮只更新设计和预算；没有训练、加载 checkpoint、读取目标 IQ/truth、生成恢复缓存或启动任务。**

核心比较固定为 15 个源系统、29 条完整 A/B/C 链路（主文15＋附表14），每条都分 P0 received 与 P1 冻结 DURA 两路。14 个外部 RFFI 家族／22 个原代码版本继续完整列入，不能与29条组合链路混为同一统计。DURA v1.0 是候选前端，其当前 Phase2 资格尚未核实，不把历史 Phase1 成功当作本轮可用或已有效。

## 1. 标记、阶段与共同权限

- **N**：保留原论文的核心算法机制；不表示本任务数据／训练预算与原论文逐位相同。
- **X**：本项目明确的阶段、监督、短窗、实现或串联扩展；不能追认为原论文原生能力。
- **F**：明确命名的修复／语义版本，例如 CSIL empirical-Fisher corefix、ISSL fixed_v1。
- 每个 source 权重先核实实际物理角色与完整祖先；源码／预算已知不证明现成权重合规。所有新增接口状态均为设计待实施。

|阶段|允许输入|输出与限制|
|---|---|---|
|S 源训练|与PAMF相同source RX13468、D1–D3、old6；Ls6300/Us56700/V27000|Us TX始终隐藏；V只验证；目标IQ/统计/BN/指标访问0；需要对齐的UDA到B才启动|
|A 适应前|同一source冻结状态，只读当前路径的旧query|自己的A_old；不更新状态；source enrollment仅由Ls冻结构建|
|B 旧类适应|当前row的6K条旧target support；外部方法按明确例外用source replay|得到实际B状态；新类support尚不可见；无额外target U/query训练|
|C 新类注册|才开放N_new×K条新support；旧support/历史状态按方法权限使用|面对全部6+N_new类；原生方法继承本row的B；MILKR独立重拟合|

同一旧query在A/B/C配对。所有N_new从同一个B独立分叉，不以C2→C5→C10代替独立规模实验。源原始容器混有target时先导出source-only材料，不以一个布尔字段代替真正隔离。Phase2新增TX不能出现在F、G或其祖先训练／选模中。

## 2. 场景与随机矩阵

|场景组|received生成规则|套数|地位|
|---|---|---:|---|
|G10综合|每物理记录按固定六分量概率抽一种，叠加一次practical residual|8，MIX01–08|主结论与稳定性|
|D-LS|全部target support/query为low郊区，仰角均匀10°–30°|1|指定单景组1|
|D-MU|全部为mid城区，仰角均匀20°–45°|1|指定单景组2|
|D-LU|全部为low城区，仰角均匀10°–30°|1|指定单景组3|

核心11个物理池互斥，每RX／类36 support候选＋30 query，合计132132个物理观测，每RX／类需726条互异原记录。P0/P1来自同一记录，不变成22个独立池；三个单景不是1/3混合。组内同场景不等于共享相同随机信道系数。历史MILKR原high郊区/mid郊区/low城区原值保留，不改名填本轮三景。

|G10分量|low郊区|mid郊区|high郊区|low城区|mid城区|high城区|
|---|---:|---:|---:|---:|---:|---:|
|概率|38.312874%|7.317985%|4.369141%|38.312874%|7.317985%|4.369141%|

G10互斥档为10°–30°/30°–45°/45°–80°，档内按覆盖球面面积条件采样，配置用未舍入值。它是600km宽视场空间参考，不称实际单星采集频率；原生D-MU仍为20°–45°均匀角度。源训练增强不因本轮目标单景变更而静默重写。

|矩阵维度|固定值|
|---|---|
|模型种子|2026092701–04，共4；P0/P1复用同一F源模型|
|目标RX/日期|[0,2,5,7,9,10,11]；D0–D3；D0是未见日，不是未来日|
|support draw|2026092711–15，共5；K前缀嵌套，按dataset/RX/TX命名随机流|
|K|1/5/10/20；K5主展示点|
|新增类数|0/2/5/10/20；类别按固定顺序嵌套|
|输入路径|P0/P1；状态和特征cache键必须包含input_path与DURA版本|
|query|每类30；同K/draw复用冻结query；新类未注册前不访问其输入|
|信道/硬件|residual/post_sync/no额外EQ，25MHz，FP32[2,256]；receiver_seed2027只绑定RX/day|

旧6类固定身份与新20类顺序沿用机器设计，加载时核实当前F/G未见性，不凭同名split确认。K1/5自然缺景，不为各类强凑六景或额外补标签。K1/5/10/20下，B support总量为6/30/60/120；N_new20时C总量为26/130/260/520。N0复用B，C_new/H为N/A。

## 3. DURA怎样接入

|路径|输入与S|A/B/C|状态与解释|
|---|---|---|---|
|P0|received y；原有source F|A看y，B拟合S_o，C看当前新旧received|无G，完整基准|
|P1|同一y先经冻结G；F与P0完全同权重，S不改|A看G(y)，B用G(S_o)，C用同路G(S_o/S_n)|所有方法共享同一G；各自独立B/C；测部署前端收益|
|未来P2-SG|source也经G后训练匹配F|对应G路径A/B/C|后续源匹配扩展，不在当前主预算，不自动重训F|

候选G固定DURA v1.0/R69 stage2，266951参数，2 prior/4 CG/27 FFT-IFFT，FP32/TF32off，Graph-RR，`reuse=True, graph=True, native=False`。源增量E20/4940是现有学生训练预算，不是完整祖先总成本；本轮新增G训练0。名义记录为原source六TX、RX13468/D123/L6300+U56700、TX隐藏/F-free；当前物理ID、教师祖先、表示和选择背景仍待对应本轮核对。若失败，P1明确缺项，不能换权重或静默训练。

1. G仅读取当前received，不能读取真实clean、信道真参数、F50输出或query标签；输出是估计视图，不能称已恢复真实TX细节。
2. G冻结eval/无梯度，无target微调、query BN或跨query统计；图池仅缓存执行形状，不能保留样本适应状态。
3. PAMF五块特征和FFT96都从同一路径计算；raw FFT＋restored网络特征等融合另记新算法，不能混入默认。
4. 所有方法的数学预处理/STFT在G之后按自己的合法定义执行；外部方法用G-only引擎，不继承旧F50 reader/logits。
5. P1不能继承P0的B头/原型/核；B后encoder改变就重算对应support特征。query始终逐样本面对全部注册类。
6. 相同G供所有F种子使用，G种子数是1，不是4；P1全管线包含G从source U学到的先验，外部F本身不用U也不能把整个P1说成未使用U。
7. 可共享确定性恢复缓存保证方法输入一致，但部署时延从received开始计入G；历史G+F50的4.531972ms不能填入本轮PAMF/外部方法时延。

同一132132个received只需一次G缓存恢复；额外恢复数组270606336字节，raw+restored总541212672字节，约516.14MiB，不含模型/特征/索引。来源定义见[DURA版本](../experiments/pamf_stage2_inference/frozen_version.json)和[机制](DURA_V1.md)。

## 4. 逐方法S源训练表

源预算保留各自机制，不能用相同epoch宣称同计算。下表区分当前用户固定版、项目匹配预算、作者上限和复现未公开默认；不得把它们统称原论文默认。现有合规权重可复用，未找到合规权重时本轮也不自动启动新训练。除PAMF外下表F源训练仅使用Ls；G的共享source L/U成本另列。

|源ID/方法|S核心目标|S预算|源选模/预算身份|
|---|---|---|---|
|PAMF 三视图 SSL|Tri-SIG-TIME SSL；固定局部排列；原默认头联合训练|E50／11100 更新|固定官方 E50；当前用户固定版本|
|RadioNet ADA＋FT|作者 ADA 架构的 source-only TX 监督；域对齐到 B 才开|E200；Adam 0.001／batch128|固定末轮；既有项目匹配源配方，不是作者全部预算|
|ProtoNet CDA|自己的 source episode，平方欧氏原型目标|E200×50 episode／E＝10000 更新；6-way，source support3/query2|固定末轮；既有项目 episode 配方|
|CSIL corefix|原生 encoder／fingerprint 监督；L_s 经验 Fisher 预计算|E200；Adam 0.001／batch128|固定末轮；既有项目匹配源配方＋经验 Fisher 修复|
|MoPC-HR 论文语义|自己的 base CE；L_s 原型／分类器预计算|E200；Adam 0.001／batch128|固定末轮；既有项目匹配源配方|
|Orthogonal SEI|自己的正交／pseudo-target base 目标，保留几何结构|E200；Adam 0.001／batch128|固定末轮；既有项目匹配源配方|
|ISSL fixed_v1|自己的旧类 base 预训练；不提前加载目标|source 模板 E300|固定已绑定 source-only 规则；既有 ISSL 模板预算，正式接口待绑定|
|LoRa triplet＋KNN|channel-independent STFT→triplet；只用 L_s；冻结源 enrollment|最多 E1000；source V val_loss 早停 patience20／LR patience10|仅合法 source V；作者训练上限；Torch／256点为项目扩展|
|POSTER FT|自己的 source CNN＋Dense，TX 监督|E200|固定末轮；既有匹配 runner 要求源 E200，不冒称作者默认|
|MRIOR-SDA|自己的 GAD 网络先做 source-only TX 监督；target 相关目标关闭|E200；Adam 0.001／batch128|固定末轮；既有项目匹配源配方|
|DADDA-SDA|自己的多尺度架构做 source CE；LMMD target 分支到 B 再开|E200；Adam 0.001／batch128|固定末轮；既有项目匹配源配方|
|Two-stage SDA＋FT|自己的 source CE 初始化；DANN/LMMD target 分支到 B 再开|E200；Adam 0.001／batch128|固定末轮；既有项目匹配源配方|
|Hu2024 新完整实现|TX/RX 特征分离；保留三项机制损失和自身增强|最多 E200；Adam 0.005；LR×0.1/patience10；早停 patience20|仅合法 source V；当前复现的未公开默认，不冒称论文精确参数|
|RA-Collab 单节点|source TX/RX 对抗学习；单节点；关闭 source PL/U 分支|最多 E100；SGD0.001／momentum0.9／batch64；早停 patience20|仅合法 source V；当前单节点源入口默认，禁止协同和 target 选模|
|Tweak M1|自己的 triplet encoder；source calibration 由 L_s 冻结|E100|预定末轮；源 LR 规则固定；项目已有完整 Tweak 训练预算；M1 是单包扩展|


S200匹配组沿用已有source-only配方，其源增强与sat CE剂量必须从实际resolved config披露（当前入口包含source practical增强、sat CE0.68及E80启用），不能替换成目标观测或称与PAMF SSL相同；所有UDA target对齐关闭到B。Hu等自身增强/预处理的25MHz与短窗适配同样事前固定。所有源原型/enrollment/Fisher仅由合法Ls产生，V不更新这些状态。

## 5. 主文完整链路：15行逐项S/B/C与继承

表内S预算指上表，避免在每个C后缀重复训练源模型。N/X/F为机制身份，不伪称原任务逐位复现。

|ID/方法|S预算|B及预算|C及预算|继承/身份|
|---|---|---|---|---|
|J00 PAMF＋MILKR|E50／11100 更新|MILKR(S_o)，冻结 encoder；目标核状态拟合；1 次解析求解；0 optimizer step；λ1／FP64|MILKR(S_o∪S_n)，重新估计全部核状态；1次解析求解；0 optimizer step|同一冻结 encoder；不继承 B 的α/τ/均值/trace；保留旧 support 信息；当前完整方法|
|J01 RadioNet ADA＋FT＋kNN 注册扩展|E200；Adam 0.001／batch128|source L_s＋旧 support 做 ADA→FT→kNN；ADA10000 iterations＋FT30epochs；k=K|冻结 B encoder，追加新 target kNN memory；0 optimizer step|B encoder＋旧 memory＋距离/投票规则；原生B机制＋项目C追加扩展|
|J02 ProtoNet CDA 注册扩展|E200×50 episode／E＝10000 更新；6-way，source support3/query2|冻结自身 episode encoder，旧 support 类均值；0 optimizer step|追加新类 prototype，全部类平方欧氏竞争；0 optimizer step|B encoder＋旧 target prototype；原生episode/B机制＋项目C扩展|
|J03 CSIL corefix＋B扩展|E200；Adam 0.001／batch128|旧 target support SFT；明确 B 扩展；10epochs／Adam0.001／batch≤128|通道隔离/mask/KD/Fisher；C仅新support及合法历史；3epochs／masked SGDM0.01／momentum0.9；source经验Fisher固定|实际B模型/参考＋固定Fisher；不恢复另一个源base；原生C核心＋数值修复版＋B任务扩展|
|J04 MoPC-HR论文语义＋B扩展|E200；Adam 0.001／batch128|旧 support SFT＋声明的原型校正；B 扩展；10epochs／Adam0.001／batch≤128|最多5新类/session；原型增强/HR/漂移校正；每session20epochs；SGD0.01；N20=4session共80epochs|B权重/已校正旧prototype/参考；session继续前传；论文语义复现＋匹配B/数据扩展|
|J05 Orthogonal SEI＋B扩展|E200；Adam 0.001／batch128|冻结 encoder，校准旧分类权重；B 扩展；10epochs／SGD0.08|冻结encoder/旧权重，新权重初始化与校准；50epochs／SGD0.08；一次N类是matched joint任务映射|B旧权重＋源正交结构；不替换encoder；原生C机制＋B/任务扩展|
|J06 ISSL fixed_v1＋B10|source 模板 E300|旧 support 全网 SFT；B10 扩展；10epochs／Adam1e-4／batch=min(32,6K)|扩头→SSL→CE transfer，保留合法旧训练样本replay；SSL2000epochs＋transfer300epochs；不是1epoch诊断版|实际B→teacher/key等；按原生C初始化/更新queue，不沿用另一run；原生C流程修复版＋B10扩展|
|J07 LoRa triplet＋KNN enrollment|最多 E1000；source V val_loss 早停 patience20／LR patience10|自己的冻结 triplet 表示，旧 target enrollment；0 optimizer step；KNN15，k=min(15,Nsupport)|冻结表示，追加新 enrollment；保持KNN规则；0 optimizer step；另列K-adaptive k=min(15,K)控制|B encoder＋旧 target enrollment；原生追加enrollment行为；Torch/256点/阶段映射为扩展|
|J08 POSTER FT→原生输出扩展FT|E200|冻结卷积，训练声明的 Dense；10epochs／Adam0.001／batch≤256|复制B旧输出权重/bias，新增类输出后联合target CE；H100：冻结前置表示/buffer，只更新扩展输出层100次|B全部前置层/旧输出权重/bias；新行合法support初始化；原生DA机制＋明确项目C扩展|
|J09 MRIOR-SDA→原生输出扩展FT|E200；Adam 0.001／batch128|GAD/DV-KL＋support CE；CPL关闭，SDA扩展；100外层更新；每步估计器7次；lr0.0006|复制B旧输出权重/bias，新增类输出后联合target CE；H100：冻结前置表示/buffer，只更新扩展输出层100次|B全部前置层/旧输出权重/bias；新行合法support初始化；原生DA机制＋明确项目C扩展|
|J10 DADDA-SDA→原生输出扩展FT|E200；Adam 0.001／batch128|source/support CE＋动态MMD/LMMD；SDA扩展；100更新；条件对齐只用 support 真标签|复制B旧输出权重/bias，新增类输出后联合target CE；H100：冻结前置表示/buffer，只更新扩展输出层100次|B全部前置层/旧输出权重/bias；新行合法support初始化；原生DA机制＋明确项目C扩展|
|J11 Two-stage SDA＋FT→原生输出扩展FT|E200；Adam 0.001／batch128|source/support DANN→LMMD→support FT；100＋100＋100更新|复制B旧输出权重/bias，新增类输出后联合target CE；H100：冻结前置表示/buffer，只更新扩展输出层100次|B全部前置层/旧输出权重/bias；新行合法support初始化；原生DA机制＋明确项目C扩展|
|J12 Hu2024 新完整实现→原生输出扩展FT|最多 E200；Adam 0.005；LR×0.1/patience10；早停 patience20|更新共享 encoder/TX 路径，RX 路径冻结；100epochs／Adam0.005／batch≤256；项目固定预算|复制B旧输出权重/bias，新增类输出后联合target CE；H100：冻结前置表示/buffer，只更新扩展输出层100次|B全部前置层/旧输出权重/bias；新行合法support初始化；原生DA机制＋明确项目C扩展|
|J13 RA-Collab 单节点→原生输出扩展FT|最多 E100；SGD0.001／momentum0.9／batch64；早停 patience20|单节点全网旧 support FT；GRL关闭；20epochs／SGD1e-5／momentum0.9／batch≤32|复制B旧输出权重/bias，新增类输出后联合target CE；H100：冻结前置表示/buffer，只更新扩展输出层100次|B全部前置层/旧输出权重/bias；新行合法support初始化；原生DA机制＋明确项目C扩展|
|J14 Tweak M1→追加类校准|E100|旧 support centroid/mean-L2-radius 校准；0 optimizer step；M=1|保留旧centroid/radius，追加新类校准并用原closed-set判决；0 optimizer step；K1半径0，无query调半径|B encoder＋旧centroid/radius；原生DA机制＋明确项目C扩展|


MILKR的B/C都是同一冻结φ上的当前support拟合，C不继承B的α/τ/均值/trace。相同最终输入时，直接C与先B后C应等价，不宣称两阶段参数协同。它仍需保留旧support信息、重建核并求解，不能称零拟合或常数时间追加。

H100结构保持版：复制B旧输出权重/bias，只扩最后可兼容类输出；新增权重方向取其B坐标support均值，norm取B旧权重范数中位数，新bias0；零均值行置零。冻结前置层与buffer，保留原生cosine/温度定义，只训练扩展输出层；新旧每条support等权/full-batch CE/Adam0.001/100更新/WD0。不兼容头明确接口缺口，不用新建头冒充继承。K相同时是各类等权，不是old/new两组各半。

MoPC主N2/5/10/20分别为1个2类短session、1/2/4个5类session，短session为任务扩展。Orthogonal主表一次N类为matched joint；ISSL B10是扩展，C保留完整SSL/transfer/replay，不拿诊断1epoch替代。LoRa的KNN15小样本与K-adaptive控制分列。

## 6. 附表统一读出：14条组合

下表每一行B完全复用J08–J14对应前缀，S也复用，不重复训练；P0/P1各自复用自己的实际B。

|ID/组合|S/B复用|C预算|状态继承与身份|
|---|---|---|---|
|J15 POSTER FT→NCM|poster_ft|B表示下重新编码全部新旧support，归一化类均值cosine判决；0 optimizer step|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J16 MRIOR-SDA→NCM|mrior_sda|B表示下重新编码全部新旧support，归一化类均值cosine判决；0 optimizer step|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J17 DADDA-SDA→NCM|dadda_sda|B表示下重新编码全部新旧support，归一化类均值cosine判决；0 optimizer step|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J18 Two-stage SDA＋FT→NCM|twostage_sda|B表示下重新编码全部新旧support，归一化类均值cosine判决；0 optimizer step|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J19 Hu2024 新完整实现→NCM|feature_separation_hu2024|B表示下重新编码全部新旧support，归一化类均值cosine判决；0 optimizer step|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J20 RA-Collab 单节点→NCM|ra_collab_single_ft|B表示下重新编码全部新旧support，归一化类均值cosine判决；0 optimizer step|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J21 Tweak M1→NCM|tweak_m1|B表示下重新编码全部新旧support，归一化类均值cosine判决；0 optimizer step|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J22 POSTER FT→JointCE|poster_ft|B表示下新建全部类共享线性头；全部新旧support CE；H100：均值初始化所有类，100更新|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J23 MRIOR-SDA→JointCE|mrior_sda|B表示下新建全部类共享线性头；全部新旧support CE；H100：均值初始化所有类，100更新|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J24 DADDA-SDA→JointCE|dadda_sda|B表示下新建全部类共享线性头；全部新旧support CE；H100：均值初始化所有类，100更新|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J25 Two-stage SDA＋FT→JointCE|twostage_sda|B表示下新建全部类共享线性头；全部新旧support CE；H100：均值初始化所有类，100更新|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J26 Hu2024 新完整实现→JointCE|feature_separation_hu2024|B表示下新建全部类共享线性头；全部新旧support CE；H100：均值初始化所有类，100更新|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J27 RA-Collab 单节点→JointCE|ra_collab_single_ft|B表示下新建全部类共享线性头；全部新旧support CE；H100：均值初始化所有类，100更新|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|
|J28 Tweak M1→JointCE|tweak_m1|B表示下新建全部类共享线性头；全部新旧support CE；H100：均值初始化所有类，100更新|只保留B表示/buffer；替换原B判决，不继承其分类权重；X组合|


NCM：实际B表示重新编码全部新旧support，逐样本unit后求类均值再unit，cosine判决，0优化器。JointCE：冻结同一B表示，所有类用target均值unit初始化、bias0，新建线性头，H100预算。二者都替换B读出，因此不能独自代表DA方法的全部能力；主文保留原生结构扩展正为防止丢掉B的头部适应收益。不依据query为每个前缀择优后缀。

## 7. 主表/附表总索引

|表|内容/行|范围|主要列|
|---|---|---|---|
|T0|15源系统＋共享G；全部版本权限|所有计划|源角色/U/源预算/增强/target首次访问/source replay/Fisher/teacher/状态|
|T1|旧类DA专项：PAMF＋9外部DA|11池，4K，P0/P1|A_old、B_old、B−A、标签/成本|
|T2-N|原生注册核心及明确B扩展：J00–07|11池，4K，N20主文|A/B/C_old/C_new/H与原生状态|
|T2-J|完整链路主文：J00–14，固定15行|11池，4K×5N，P0/P1|同上＋C_old−A_old、B−C_old、source/history、成本|
|A-J|统一读出：J15–28，固定14行|同完整矩阵|NCM/JointCE两后缀全部报告|
|T3|注册规模/标签曲线|全4K×5N；主文K5/20|C_old/C_new/H/总support；N0复用B|
|T4|适应/注册/部署成本|全row计时；固定464配置细测|G单独/G+F+读出、B1 median/p95、capture、host I/O、RSS/VRAM/state|
|T5|共享PAMF表示控制|11池，P0/P1|同source分离读出/梯度适应贡献|
|T6|MILKR消融与参数|核心对照全矩阵，其余固定切片|分支/交互/局部核/尺度/标签带宽|
|T7|序贯注册|主文15系统，8MIX，K5/20，P0/P1|逐session旧/新/H与累计成本|
|T8|DURA与可靠性|各K/N/场景、8套/4F种子/5draw|ΔA/ΔB/ΔC_old/ΔC_new/ΔH，实际scene/RX/day/TX/净纠错|
|A-V|22原代码版本覆盖/版本差异|固定5池切片；见下表|保留机制/修复/原行为与可运行边界|
|A-G|DANN/MCD等通用控制|固定2MIX切片，P0/P1|明确通用算法，不冒称RFFI新论文|
|A-D|有/无B、数据匹配/原生直接C|固定2MIX、K5/20、N20，P0/P1|阶段收益与终态效应分开|
|A-E|G20/G25、环境比例、源匹配G|条件扩展/重加权|不与当前核心预算或support分布混淆|


## 8. 全22原代码版本逐项覆盖

14个主代表版本就是上面的14个外部源系统；与PAMF合为15源绑定。其余8版本如下，未接通项不记完成，也不把N/A作0分。A-V固定MIX01/02＋三单景、4模型、7RX、5draw、K1/5/20、N0/5/20、P0/P1；新增源版本不擅自共享不匹配权重。

|额外版本|S|B|C/继承|范围/状态|
|---|---|---|---|---|
|RadioNet ADA-only→kNN|复用RadioNet S200|只做ADA10000，不含FT30|kNN追加为X；继承其实际ADA B|可从主B前FT快照精确复用；历史1000次版另命名|
|RadioNet DF-FT|自己的DF source E200|末层FT30epochs，Adam0.001/batch≤128|原生闭集C=N/A|A/B辅助路线，不能代表ADA|
|RadioNet Triplet→kNN|自己source triplet E30，Adam0.001/batch128|旧target kNN enrollment|追加memory为X，冻结自身encoder|256点正式接口待接；A用合法source enrollment|
|Feature Separation legacy|自己旧复现S200|旧matched FT100epochs|原生闭集C=N/A|与Hu新版不同实现，不混成绩或权重|
|Tweak native M10|复用Tweak S100|同校准；query原生10包聚合|无当前合法事件组，不接当前C主表|当前单包capsule未提供合法10包组：0活跃槽位/N/A|
|CSIL作者MATLAB|原源训练行为；当前数据绑定未完成|无原生B；若加B需新标X|原作者C行为，正式256点接口未接|当前0活跃槽位；不以corefix冒充原作者版|
|MoPC公开代码语义|同source/B训练确证相同才复用|明确matched B10；原型校正保留公开语义差异|raw-dot/softmax等原行为；每session20epochs|只复用相同权重部分，原型/参考/HR语义分别绑定|
|ISSL作者行为保留|source模板E300；是否与fixed base相同待核对|原生无B，A→C单列|原queue/KD/梯度行为；SSL2000＋transfer300模板|标签使用与fixed+B不同，不能解释为纯B效应|


A-V单版本每路径A140、B2100、正N的C4200。上限按4条新增B路线＋4条C路线、最多4个不同源版本计；source/A/B快照实际可证明相同时复用。MoPC/Radio的共用状态不能只按名字推定。作者CSIL与M10当前不具备合法输入绑定，不伪造数字；它们仍完整列在清单中。缺完整独立算法的CSCNet/SHOT/独立DAN/DSAN/CDAN/WD/Deep-CORAL/LwF/EWC/iCaRL等仍是缺失表，不用现有loss组件顶替。

## 9. 共享PAMF表示与通用DA

|ID|控制|B→C与预算|计数|
|---|---|---|---|
|S0|默认源头|仅A，源头不另训|复用主A，不新增fit|
|S1|NCM|B旧均值；C追加新均值；冻结表示|全矩阵新增|
|S2|cosine distance-weighted kNN，k=K|B旧memory；C追加新memory；冻结表示|全矩阵新增|
|S3|linear ridge λ1、截距不罚|B/C当前support独立求解；目标/中心化与MILKR不同明示|全矩阵新增|
|S4|linear CE20|B20epochs Adam0.001；C新建全类头20epochs；冻结表示|全矩阵新增；不是H100同预算|
|S5|BranchRidge|B/C独立解析|复用M2|
|S6|BranchInteraction|B/C独立解析|复用M3|
|S7|MILKR|B/C独立解析|复用J00/M0|
|S8|head SFT|B默认头20epochs Adam0.001；C继承旧头扩展H100|全矩阵新增；Phase2头更新|
|S9|tail SFT＋MILKR|B末级block+默认头20epochs Adam0.001；C冻结实际B，用MILKR重拟合|全矩阵新增；模块名/参数mask在接口绑定时列清|
|S10|full SFT＋MILKR|B全网20epochs Adam0.001；C冻结实际B，用MILKR重拟合|全矩阵新增；不用B前特征cache|
|S11|JointCE100|B旧类H100；C新建全类H100；同冻结PAMF|全矩阵新增|


8个新增共享控制为S1/2/3/4/8/9/10/11；S5/6在消融计，S7主行计，S0复用A。尾部和全网控制是明确表示适应扩展，不能继续叫冻结原MILKR。P1所有块从G输出重算，P0/P1不混cache。

通用控制另列：DANN/MCD采用独立已声明RFFI骨干/分类器source-only warmup（E200为本轮项目控制，不沿用MNIST target-test-best）；DANN B100更新，MCD B100个原生三步outer、每outer4次feature-discrepancy更新，Adam0.001、discrepancy权重1为明确项目设置，target只用合法旧support IQ。各自C接NCM/JointCE，source和B只做一次。接口尚未接通，不称原论文RFFI完整版。原生实现的其他优化细节在source/support无query验证后按这个版本固定。

AdaBN仅在存在BN时以旧support做20次整池统计更新，无optimizer；C若接统一读出另名，PAMF无BN时N/A，不制造空对照。CORAL+SFT单列监督机制扩展：source/support CE+covariance loss1，B100更新Adam0.001；C接统一读出时另列。它们不计29个RFFI链路。本轮预算表对DANN/MCD给固定切片槽位；AdaBN与CORAL的接口/适用性先列条件储备，不冒充已含活跃量。

## 10. MILKR机制与参数预算

|ID|变更|范围|
|---|---|---|
|M0|完整MILKR|主行复用|
|M1|原D92整体历史控制|权限/当前表征绑定后才进条件储备，不称单因素|
|M2|BranchRidge|核心11池全矩阵/P0/P1|
|M3|BranchInteraction|核心11池全矩阵/P0/P1|
|M4|去z_id|固定切片|
|M5|去FFT96|固定切片|
|M6|去t_emb|固定切片|
|M7|去f_emb|固定切片|
|M8|去pa_local|固定切片|
|M9|关闭trace matching|固定切片|
|M10|类盲pairwise距离中位数带宽|固定切片；分类监督仍保留|
|M11a|仅z_id匹配局部核|固定切片|
|M11b|仅FFT96匹配局部核|固定切片|


固定机制切片为MIX01/02×4模型×7RX×5draw×K1/5/20×N0/5/20×P0/P1，每变体每路径B840+C1680=2520fit。删块的active-block归一化与norm控制预先固定；读出删t/f/PA不等于网络彻底删分支；现有frozen runner不接受这些新参数，需独立配置入口。

参数敏感性仅P0：λ={0.01,0.1,1,10,100}、τ倍率={0.25,0.5,1,2,4}、FFT权重={0,1,2,4,8}，单因素；默认1/1/4复用主行，合计12个非默认点×2520=30240fit。P1使用固定默认，不自动乘完整参数笛卡尔积。所有点完整报告，不从query选默认或筛消融。

## 11. 衔接与持续注册实验

|实验|对象/范围|标签/状态规则|
|---|---|---|
|等数据直接C|J00＋7闭集前缀×3类C，共22路径；2MIX/K5/20/N20/P0-P1|无中间B更新，C仍得相同S_o/S_n；同后缀/预算；校准法在C建立旧target校准再追加新类；MILKR应终态等价|
|原生A→C|CSIL/MoPC/Orthogonal/ISSL四法；同切片|保持原生输入，可能不用S_o；与B+C标签使用量不同，不能归因纯参数轨迹|
|持续注册|主文15系统，8MIX/K5/20，P0/P1；日程1×20/2×10/4×5/10×2|每session只开放当前新增support，真实状态前传；MILKR全累积support重拟合；每次成本计入|
|类顺序|默认顺序外3种，seed2026102841/42/43；15系统、8MIX/K5/20，仅4×5，P0/P1|同物理池，不重生成信道；类映射/未来输入访问正确；默认序已计主日程|

序贯需要累计4/6/8/12/14/15/16/18等额外registered-class split，由独立builder对同一received池预构建opaque索引，不能让预测器读query真类筛选未来样本。完全相同状态/输入/seed的主矩阵端点可复用；不以同一query多次评分当独立重复。MoPC一次20/两次10是显式扩大session扩展，不能内部拆成4×5后仍称一次；每实际session20epochs，分别20/40/80/200epochs。其他方法的每session原预算和累计成本同样披露。

## 12. 指标与证明优势的主表列

`系统/版本 | input_path | S/B/C性质 | source/history权限 | A_old | B_old | C_old | C_new | H | ΔDA | Dreg | Gretain | G/B/C/端到端成本 | 状态`

ΔDA=B−A；Dreg=B−C_old；Gretain=C_old−A=ΔDA−Dreg，均为相同旧query上的有符号百分点差。H逐cell计算后平均，不由平均旧/新准确率倒算。另报Macro-F1、最差类/recall-P10、净纠错、old→new/new→old错误与混淆。N0没有新类/H。

旧类下降离线拆成(B_old−C_old,restricted)+(C_old,restricted−C_old,all)。restricted仅独立scorer屏蔽新列，不用于正式预测路由；第一项是旧类内部判决变化，第二项是新增类竞争损失，不把全部下降叫encoder遗忘。

DURA配对报告δA/δB/δC_old/δC_new/δH，以及δΔDA=(B_P1−A_P1)−(B_P0−A_P0)。DURA提高A后B−A变小，不自动表示B更差。波形NMSE/外观不能替代真实身份识别；目标runtime不能打开对应clean来算恢复误差。以A→B→C图、新旧Pareto、成本/H、新类规模和4条件分面展示，不能仅以闭集方法C=N/A声称胜利。

G10先在model×dataset×RX×draw×K×N×path内配对，再汇总为每模型结果，主报4模型均值±样本标准差；8套逐套值与套间波动另列。单景各自独立；P0/P1配对不当新数据套，G只有1个种子。所有预测固定后独立truth-last；失败/N/A/缺seed不删，不按分数追加、择优后缀或换G。

## 13. 全部实验预算账本

下表是逻辑fit/阶段槽位，不是optimizer step、独立样本、实际GPU小时或立即全量启动指令。每条原生算法内部多session/更新成本另报。重复端点/中间状态精确可复用，未接通或不合规项保留缺口。原单路径806960fit/811580阶段在本轮双路径核心变为1613920/1623160。

|预算ID/实验|范围|A槽位|B槽位|C槽位|B+C fit槽位|全部阶段槽位|
|---|---|---|---|---|---|---|
|E01 29完整链路|11池×4模型×7RX×5draw×4K×4正N×P0/P1|9240|184800|1429120|1613920|1623160|
|E02 8个新增共享PAMF控制|核心完整矩阵，P0/P1|0|98560|394240|492800|492800|
|E03 MILKR机制消融|M2/M3核心全矩阵；9个删块/尺度/带宽切片，P0/P1|0|39760|128800|168560|168560|
|E04 12个非默认参数点|MIX01/02，K1/5/20，N0/5/20，仅P0|0|10080|20160|30240|30240|
|E05 等数据直接C对照|22路径×MIX01/02×K5/20×N20，P0/P1|0|0|24640|24640|24640|
|E06 原生A→C对照|CSIL/MoPC/Orthogonal/ISSL；同固定切片，P0/P1|0|0|4480|4480|4480|
|E07 分次注册|15主文系统，8MIX，K5/20，1×20/2×10/4×5/10×2，P0/P1|0|0|1142400|1142400|1142400|
|E08 3个额外类顺序|15主文系统，8MIX，K5/20，仅4×5日程，P0/P1|0|0|806400|806400|806400|
|E09 8个原代码辅助版本|MIX01/02＋三单景；K1/5/20、N0/5/20，P0/P1|1120|16800|33600|50400|51520|
|E10 DANN/MCD通用控制|MIX01/02，K1/5/20、N0/5/20；各接NCM/JointCE；P0/P1|224|3360|13440|16800|17024|


以上全部预定行的复用前上界为 **4350640 个fit槽位、4361224 个阶段槽位**。其中主比较E01为核心；其余是明确的补充预算，不与主方法计数混淆，也不以每条槽位当作独立训练重复。


|预算类别|数量/规则|不计入上表的条件储备|
|---|---|---|
|核心source F|15×4=至多60个模型绑定|已有合规权重复用，不表示新训练60次|
|辅助source F|A-V至多16个不同源绑定；DANN/MCD另8个|常规上限合计84个F绑定；实际新训练数量未确定，本轮0|
|共享G|1个DURA候选权重；target训练0|P1资格未核实；旧学生E20之外的祖先成本不得省略|
|细测T4|29链路×4模型×K5/20×P0/P1=464配置|MIX01/首RX/首draw/N20；每配置50warm+200measure，共116000forward；无需新fit|
|DURA缓存|132132次唯一received恢复上界|细测P1另有58000次G调用；capture/host I/O单列，不用cache输入冒充在线耗时|
|PAMF额外4种子|仅已有合规05–08权重|双路61600fit/62216阶段，前4不重复|
|PAMF-Sup/PL桥接|仅已有匹配合法权重，额外2×4绑定|双路123200fit/124432阶段；不自动新训|
|M1原D92|权限与当前输入绑定后才做|双路固定切片5040fit；历史整体对照，不是单因素|
|G20/G25|每条件另8套，全部29链路、双输入|另2347520fit/2360960阶段；额外192192物理记录；不是主8套拆分|
|环境比例|城区20/50/80%的固定预测重加权|新增fit/预测0，只解释support不变的评价分布变化|
|source匹配DURA的F|后续单独设计，不激活|至多另60个source F绑定＋806960fit/811580阶段；不能与当前P1混同|

G20/G25低/中/高的空间参考条件概率分别为49.8946/31.3739/18.7315%与28.6931/44.6494/26.6575%，环境各半。低分量实际范围改为20°–30°或25°–30°，不只改外层权重。必须为完整适应另配support；只加权原六单景/原query分数不等价。

AdaBN适用时的固定2MIX切片双路储备为A0/B1680/C0；它仅B，不硬造C。CORAL+SFT加NCM/JointCE的同切片储备为A112/B1680/C6720，合计8400fit/8512阶段，source映射确认后才纳入。source源模型/预处理/增强/优化器完整绑定仍是具体实现工作，不能用设计表当launch-ready配置。

预算很大，尤其ISSL的2300个C阶段epoch不能按MILKR一次求解估算。执行准备先做一个seed/一套/一个RX的合法无query成本测量，按真实step/曝光/复制/缓存开销估算资源；完整原生预算不静默缩短。另设budget-limited版必须提前命名，不得跑到一半降预算后仍叫原生完整版。当前没有实际GPU ID、远程run或墙钟承诺。

## 14. 状态、实现边界与资料

所有设计都保留合法source/target时间边界、单物理观测、版本身份、固定prediction后独立评分。当前仍需处理：G当前契约祖先/表示资格、当前PAMF五块导出、外部方法source-only材料与接口、原生扩头映射、序贯split、真实物理池覆盖及464配置实测。既有target结果不能回流调参/选G/挑方法；不宣称已完成Phase2或独立盲测。

资料：[机器设计](../configs/pamf_phase2_milkr_experiment_design_20261028.json)、[信道概率/三单景](../configs/pamf_phase2_practical_mixture_20261028.json)、[衔接论证](PAMF_PHASE2_DA_REGISTRATION_LINKAGE_20261028.md)、[代码盘点](PAMF_PHASE2_METHOD_CODE_INVENTORY_20261028.md)、[DURA版本机制](DURA_V1.md)、[科学协议](PROJECT_PROTOCOL.md)。
