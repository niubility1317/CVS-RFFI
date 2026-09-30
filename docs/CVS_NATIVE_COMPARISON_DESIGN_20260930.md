# CVS Phase1/Phase2缺失对比实验设计（2026-09-30）

本轮执行用户点名及已确认的缺失方法，补齐域适应和新类注册证据。新建9个源模型系列，每个系列固定5个模型seed，共45个从零训练的Phase1任务；随后执行45个对应的Phase1目标域评估与Phase2适应/注册流水线。**本文件是已实现方案及固定实验契约，不表示GPU训练或最终评分已经完成。**实际状态以两个run的登记、进程及产物读回为准。

方法查找、论文出处与历史状态见[方法总清单](CVS_COMPARISON_METHOD_CATALOG_20260930.md)。本轮不扩展到协同、多节点融合、联邦或未知拒识。已有CVS、POSTER Fine-Tuning、RadioNet-DF Fine-tuning及其他9月27日结果保留，不重复启动或改写；它们与本轮不同训练预算、独立拟合/状态继承口径均须分列。

## 1. 数据、信道与随机性

|项目|本轮固定口径|
|---|---|
|Phase1数据|既有ManySig source契约；source接收机`[1,3,4,6,8]`、日期`[1,2,3]`|
|训练角色|`L_s/U_s/V=0.07/0.63/0.30`，物理ID精确复用；这些外部方法只用`L_s`的TX标签，`U_s`标签隐藏且本轮不使用，`V`仅评估|
|Phase1预算|scratch、200epochs、最后一轮`last.pt`；不按source-V或target成绩选权重；记录真实optimizer步数与样本暴露量|
|输入|WiSig `equalized1`、中心256点、单位RMS；对既有接收IQ施加practical LEO残差，不声称使用未均衡原始WiSig|
|信道|`residual_noeq`：`processing_route=residual`、`mode=post_sync`、`equalization_enabled=false`、`fs_hz=25e6`；不增加后续均衡|
|Phase1增强|clean＋satellite拼接；卫星视图仅TX监督CE，权重0.68，E80起计入辅助CE；E1–40高仰角0.30，E41–90中仰角/城区0.60，E91–200三场景0.80|
|目标接收机|`[0,2,5,7,9,10,11]`；目标日期`[0,1,2,3]`，source与target接收机互斥|
|Phase1测试|既有168000个目标物理记录的clean和固定satellite视图；分别汇总`practical_high`、`practical_mid`、`practical_low_urban`|
|Phase2数据|复用`p2_min_v1/VALIDATED_ONCE` capsule `residual-noeq-ba667eee4fb061055e4c08b5`，不重新生成LEO观测|
|模型seed|`392005,2026092701,2026092702,2026092703,2026092704`|
|Phase1 split seed|`392005`；实际角色物理ID与既有source_contract逐项匹配|
|Phase1数据/增强seed|数据采样/episode使用每个模型对应的data seed；增强seed和receiver seed固定`2027`，分别记录而非推断相同|
|Phase2 seed|既有capsule data/augmentation/evaluation seed分别为`2026092705/2026092707/2026092706`；support seed固定`2026092711`至`2026092715`|

`392005`是历史优化暴露过的seed，单列披露。主结果使用另外4个预先固定的新seed均值与seed间标准差；再列历史seed及全部5个seed的敏感性结果。所有已登记seed均执行，不按已见目标成绩挑选、删去或重跑seed。未知seed字段记N/A，不能从model seed推断split、support或信道随机性。

新方法的模型与随附状态均从本轮source契约从零产生，不加载历史“成熟基座”、teacher、EMA、旧原型或旧Fisher。加载时核对checkpoint内的数据角色/物理ID、类顺序、来源链、固定最后一轮及方法配置。外部方法的source访问例外不解除测试污染禁令。

## 2. 方法与固定预算

所有方法使用独立模型、完整方法机制和相同数据切片。Phase1固定200epochs是本项目的统一源训练预算，不宣称等同原论文的全部训练轮数或原始数据集复现。ProtoNet以episode训练，其他源模型的batch/optimizer步数不同，不能仅凭epoch相同声称计算量相同。

|稳定方法ID|Phase1源模型/训练区别|Phase2 B旧类适应|Phase2 C新类注册与边界|
|---|---|---|---|
|`protonet`|作者派生Conv2D编码器，独立episode训练，不能由任意CE骨干＋NCM代理|冻结编码器，用旧类support均值和平方欧氏距离分类；0个参数更新|追加新类support原型，继承旧类原型，所有注册类统一竞争；明确标记注册扩展|
|`feature_separation`|TX/RX特征分离模型及源训练机制|100epochs固定few-shot监督适应，保留论文特征分离结构|闭集方法，C记N/A；不伪造为原生类增量|
|`dadda`|论文多尺度2D结构；Phase1只在source训练，目标对齐发生在B|DADDA-SDA：source CE＋target support CE＋动态MMD/LMMD；100epochs，每epoch固定1步|闭集方法，C记N/A|
|`mrior`|ReceiverImpact GAD模型，保留估计网络/特征结构|MRIOR-SDA：source与target support监督，保留GAD和DV-KL交替更新；100epochs，每epoch1步；CPL关闭并披露|闭集方法，C记N/A|
|`twostage`|Receiver-Agnostic Two-stage模型|DANN阶段100epochs＋LMMD阶段100epochs＋support Fine-tuning阶段100epochs；各预算固定|闭集方法，C记N/A；不与MRIOR混同|
|`csil`|zero-bias指纹层和投影；六块IQ编码器为显式CVS输入扩展|仅旧类target support的10epochs SFT，标记旧类域适应扩展；source Fisher固定在B前|3epochs掩码SGDM；通道扩展、零块隔离、KD、有效EWC；使用明确的`official_repo_corefix_empirical_fisher`版本|
|`mopc_hr`|原生线性分类器＋源类原型；六块IQ编码器为显式CVS输入扩展|仅旧类support的10epochs SFT；用support估计的特征漂移执行原型校正，标记B扩展|按固定类顺序每阶段最多5个新类、每阶段20epochs；CE＋原型增强＋论文层次正则，原型校正传入下一阶段|
|`orthogonal`|六块Conv1D编码器＋固定正交/simplex伪目标，源阶段训练CE、对比与类中心分离机制|冻结编码器，旧类分类权重10epochs margin/原型对齐校准；标记B扩展|冻结编码器及旧类权重，新权重以support均值初始化，50epochs margin＋原型对齐校准|
|`radionet_ada`|作者ADA编码器/分类器拓扑，从零source训练；与已有DF Fine-tuning权重分开|作者交替ADA更新，source分类监督＋source/support域判别；固定1000iterations，之后冻结编码器，以target support构建k-NN|追加新类support近邻样本，编码器和旧近邻状态继承B；标记k-NN注册扩展|

RadioNet主路线是ADA＋冻结编码器＋k-NN，不用Fine-tuning替代ADA，也不用通用GRL替代作者交替更新。k-NN距离为cosine、按distance加权，近邻数采用作者`nShot`规则，绑定本row的K。作者代码默认ADA预算10000iterations；本轮固定1000iterations是预登记的预算缩减，须在论文结果中披露为**预算受限的匹配扩展**，不能宣称执行了作者默认完整预算。现有DF Fine-tuning单独保留；Triplet不是本轮自动新增任务。

DADDA/MRIOR/Two-stage原方法属于UDA。本轮只有固定、已标注的合法target support训练池，没有独立额外target无标签池，因此采用明示SDA/少样本扩展。support可以按原域对齐目标暂时不使用标签，但不能把query加入“无标签训练池”。MRIOR的CPL未执行，日志记为关闭，不能把未执行机制写成贡献。

CSIL作者`exp(gradient²)`近似在有界source测试中即使FP64也溢出；冻结旧指纹块还会使EWC恒为0。主版本明确修正为源域`L_s`逐样本监督CE梯度平方的均值（标准经验Fisher，FP64），允许旧×旧指纹块更新，使EWC有效。旧编码器、旧投影及新旧交叉零块保持冻结/隔离。保留KD和EWC损失逐步记录。不得静默指数截断、用单位矩阵或support Fisher回退，也不得称为作者代码逐位等价复现。

MoPC-HR主版本遵循论文的cosine动量校正和分层平方L2正则，不与公开代码的raw-dot/softmax校正及逐参数非平方L2语义混用。校正原型用于后续阶段的原型增强；最终预测仍使用训练后的完整分类器。只有一个新类阶段时，末次校正保留为持久状态，不伪称它替换最终分类器或改善了当次推理。新增2类是原论文类别日程的明示扩展；多阶段到达顺序在读取query前固定。

## 3. 配对A/B/C矩阵

Phase2固定旧类数为6，`K∈{1,5,10,20}`，新增类数`N∈{0,2,5,10,20}`。7个接收机×3个场景×4个K×5个support draw×5个N，共2100个split。每类query固定30条物理记录。复用既有收到的IQ、物理ID及support/query划分，不调用新的信道模拟器，不用query拟合BN、阈值、温度、原型或参数。

每个源模型有420个仅旧类support任务。A在适应前对该row旧类query预测；B只用该row旧类target support适应，并再次预测同一旧类物理query。原论文需要的source回放仅限外部方法显式例外的`L_s`，单独披露，不将含source回放的B描述为“全部训练输入只有旧类support”。CSIL/MoPC的B SFT实际只使用旧类target support；它们继承的source摘要/Fisher在B前冻结。

支持新类注册的5个方法系列，各N>0的C从对应B状态复制，加入新类support；不重新从source基座独立拟合C，不覆盖B对象。C中的旧类query和旧类support分别与A/B完全配对，旧新类在全部注册类上竞争。预测不读取query真值、old/new角色、真实query类数、配额或全局重排信息。物理旧类query集合一致性由独立scorer核对；数据capsule已通过一次性验证，不因方法变化重复建库验证。

|方法类型|每模型预测记录|45个流水线中的任务量|
|---|---:|---:|
|4个闭集DA系列|420个A＋420个B＝840|20个模型，共16800条预测记录|
|5个注册系列|420个A＋420个B＋1680个C＝2520|25个模型，共63000条预测记录|
|合计|固定完整coverage|79800条Phase2预测记录|

N=0列报告A/B和适应提升，C、H及注册后下降记N/A，不能复制B成绩伪造C。闭集DA方法的C明确记`N/A_closed_set_method`。本轮C继承B与既有CVS-D92联合注册的独立重新拟合具有不同状态口径，均单列说明，不能混写为相同算法流程。

## 4. 评分、报告与资源

训练、预测和独立scorer使用不同入口。所有模型完成对应阶段的预测后，scorer先核验全矩阵coverage、物理ID、capsule/split、类顺序、预测维度与范围，再读取truth。任何不完整row或不匹配预测使该阶段truth保持关闭。评分不会回流训练、选择方法版本、选seed、调整超参或触发选择性重跑。

Phase1报告目标clean准确率、固定satellite总体准确率及3种场景的准确率、macro-F1、接收机分层。ManySig Phase1与ManyTx Phase2采用不同测试池，不直接相减准确率作为适应提升。

Phase2每个K×N报告A旧类准确率、B旧类准确率、C旧类/新类/总体准确率、macro-F1、混淆矩阵及H。配对报告`B−A`适应提升、`B−C_old`注册后旧类下降、`|C_old−C_new|`旧新差距，单位为百分点。H使用`2×C_old×C_new/(C_old+C_new)`，无新类阶段记N/A。保存完整row、逐seed、接收机/场景分层及主4-seed/历史seed/5-seed敏感性表，不只展示最终H或最优K。

理想目标为适应提升≥10个百分点、注册后旧类下降≤1个百分点、旧新绝对差≤3个百分点；它们是解释结果的目标，不是阻止低分结果进入报告或自动停机的门槛。

资源报告必须来自实际测量：训练中可更新参数（掩码允许更新数与autograd参数面分别记录）、训练及推理耗时、峰值显存/内存、常驻状态字节、新增传输字节、硬件及测量范围。source样本回放、Fisher、原型、k-NN记忆、教师和判别器都计入相应状态/开销。未测量项记N/A，不能由参数量推算省算力或把未测传输写成收益。当前接口将本地已有support、无新增跨节点通信记为0新增通信字节；地面模型上传、协议之外通信及没有仪表证据的端到端传输均不由此推断，另行披露。

日志保存实际生效配置、来源/角色、各seed、版本、命令、设备、损失分量及权重、学习率、梯度、组件执行状态、source-V评估和耗时。Phase2不执行source-V选模，不可用项写null并说明原因。详细文本日志与结构化逐步记录并存，同时保存去除大数组的epoch汇总JSONL/CSV。

## 5. 启动、登记与当前交付边界

两个固定run为：

- `20260930-phase1-native-baselines-practical-manysig-m5-r01`：45个scratch源训练任务。
- `20260930-phase12-native-baselines-practical-m5-r01`：45个依赖对应源权重的Phase1/Phase2流水线。

统一组为`native-residual-noeq-exact-manysig-source-manytx-target-20260930`。每row拥有独立config/output/log；`comparison_suite.launch`按依赖完成证据调度，只有一个launch owner。先本地测试与Git交付，再同步固定release到N607。每GPU最多2个训练任务，既有健康任务继续运行；没有空位则等待，不停止健康任务或根据低分干预。

源任务输出`last.pt`、`base_state.pt`、`source_contract.json`、`resolved_config.json`、完整日志与完成标记。流水线输出Phase1预测、Phase2逐row预测、拟合日志、资源记录和来源核验。scorer独立输出Phase1/Phase2结果、seed汇总及配对A/B/C表。已存在产物和失败任务保留，不覆盖、删除或无证据重复启动。

实现入口为`comparison_suite/models.py`、`source.py`、`adaptation.py`、`registration.py`、`pipeline.py`、`score.py`和`launch.py`；固定配置由`tools/prepare_native_comparison_matrix.py`产生。运行状态、提交、远端PID/GPU及评分完成证据写回既有实验登记，不用本设计文档替代实时证据。

CVS对照固定为DAOT A1＋FastTrust-RC4源模型和D92 E0去RFRF32，即identity160＋FFT96的`P2-256-FULL`版本；已有结果来自对应登记run。POSTER和RadioNet DF Fine-tuning原结果作为辅助比较保留。其源训练步数、适应预算、是否独立重拟合及资源未测项在最终合表中披露，本轮新方法结果不能补填它们缺失的C继承或资源记录。

## 6. 论文来源与可声明范围

RadioNet采用[作者论文](https://homepages.uc.edu/~wang2ba/files/pub/cns22_haipeng.pdf)及[作者仓库](https://github.com/UCdasec/RadioNet)；ProtoNet采用[高校论文PDF](https://www.eng.auburn.edu/~szm0001/papers/sensys22.pdf)与[作者代码](https://github.com/stevester94/csc500-main)。MRIOR来源于[作者论文](https://arxiv.org/abs/2404.08566)，DADDA文献定位为[IoTJ论文DOI](https://doi.org/10.1109/JIOT.2025.3573713)，Feature Separation来源于[出版社论文](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cmu2.12841)。Two-stage UDA＋Fine-tuning的具体DOI仍按方法清单记为未核实，不猜填。

CSIL来源于[作者论文](https://arxiv.org/abs/2105.06381)与[作者MATLAB代码](https://github.com/pcwhy/CSIL)，MoPC-HR文献定位为[论文DOI](https://doi.org/10.1109/TITS.2025.3559174)，Orthogonal方法见[通信学报正文入口](https://www.joconline.com.cn/zh/article/doi/10.11959/j.issn.1000-436x.TXXB260021/)。本轮取已核查的论文机制，不将存在的公开仓库自动等同论文版本。

允许的结论是固定ManySig/ManyTx、跨接收机、物理启发practical LEO残差信道下的匹配实验结果。原论文跨天/ADS-B/FSCIL任务、输入编码器替换、SDA和注册扩展、预算缩减均披露；不声称原始数据集复现、真实在轨验证、完全计算量公平或所有原方法原生支持新类。
