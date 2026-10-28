# PAMF Phase2 对比方法代码盘点

日期：2026-10-28。范围：项目当前源目录、Git 发布镜像、已登记历史代码和固定外部源码的定向只读核查。**完整代码存在不等于当前 CWD 可运行，也不等于同契约成绩已完成。**本轮未读取训练数据／query truth，未执行任何方法。

配套：[完整实验设计](PAMF_PHASE2_MILKR_EXPERIMENT_DESIGN_20261028.md)、[机器可读清单](../configs/pamf_phase2_milkr_experiment_design_20261028.json)。论文题名和版本来自项目已有来源记录及原始代码；本轮没有声称重新逐篇复核全部论文全文。

## 本轮主表资格与目标访问时点修订

本轮以[综合practical residual的8套主表](PAMF_PHASE2_MAIN_TABLES_20261028.md)为准。以下“完整代码存在”不表示作者默认输入权限可以直接沿用：每个主表方法都先在匹配PAMF的L_s/U_s/V源角色训练与冻结source模型，S阶段target IQ／统计／BN／伪标签／指标访问为0；仅B开放旧类target support，C才开放新类support。UDA需要source-target联合更新的目标到B再开启，不允许先在target上预训练再把该模型当A。

`comparison_suite/source.py:139`已有source-only结构及target_access=False字段，`pipeline.py:142`独立加载源checkpoint后才进入target路径；这只是对应9家族路径的源码证据，不证明所有独立作者入口均已合规接入。ProtoNet/LoRa/Tweak的A只能从S末L_s原型／enrollment／calibration构建；不再接受使用target enrollment的假A。源标签、source U_s使用和native计算预算分别披露，不把“相同源域”解释为把U_s标签全部开放。

主表版本选择：RadioNet为完整ADA＋FT＋kNN；Hu2024用独立完整新版，B100epochs是项目固定预算；ISSL为fixed_v1＋明确B10扩展，原生无B版本在辅助表。原生Tweak M10在缺少合法观测组时仍为N/A，主表使用M1扩展。所有source/history权限只在其来源合法的前提下保留，不取消S阶段source-only边界。

## 1. 覆盖与去重

纳入 **14 个外部 RFFI 方法家族**：POSTER、RadioNet、ProtoNet CDA、MRIOR、DADDA、Two-stage UDA＋FT、Feature Separation、RA-Collab 单节点扩展、Tweak、CSIL、MoPC-HR、Orthogonal Incremental SEI、ISSL、LoRa scalable/channel-robust RFFI。

RadioNet 的 ADA-only、ADA＋FT、DF-FT、Triplet 是同一论文家族的不同实际路线；Feature Separation 两个目录为同一 Hu 2024 论文的不同复现；CSIL、MoPC、ISSL 的版本差异不计成新论文。新类注册完整清单为 CSIL、MoPC、Orthogonal、ISSL、LoRa 五个家族，加 ProtoNet、RadioNet 的显式注册扩展，共7个。

路径缩写只用于此表：

- `ROOT`：`E:/type10-7/`。
- `PUB`：`E:/type10-7/github_publish/CVS-RFFI-repo/`，完整Git发布承载面。
- `RADIO`：`E:/type10-7/external_sources/rffi_labeled_da_20260915/RadioNet/`。
- `OLD`：`E:/type10-7/code/snapshots/cvs_practical_baselines_20260927_wt/`。

## 2. 域适应方法：核心、入口、权限与版本

|家族／路线|实际核心与入口（路径后为核查行）|算法与权限|完成度／设计动作|
|---|---|---|---|
|POSTER Fine-Tuning，WiSec2021|`OLD/baselines/common/li_backbones.py:27,46`；`OLD/tools/run_practical_phase2_baseline.py:94`；作者`external_sources/rffi_labeled_da_20260915/RadioFingerprinting/fine-tuning/finetune.py:95`|冻结卷积层，重建并训练3层Dense；Adam0.001、10epochs、batch256；只用target support标签|完整源码／runner／旧评分存在；重新绑定当前source/capsule，不以NCM代理|
|RadioNet ADA→kNN，CNS2022|`RADIO/ADA/af_classifier.py:30,98`；`ROOT/comparison_suite/adaptation.py:186,413`|source监督＋target support域对齐，交替E/C/D；随后冻结encoder，cosine距离加权kNN，k=K|完整匹配runner；历史1000iterations是预算受限版，作者10000iterations；5模型旧预测不等于已评分|
|RadioNet ADA→FT→kNN|`RADIO/ADA/af_classifier.py:202`传`tuning=True`；`ADA/af_test.py:28,48,77`；`ROOT/comparison_suite/adaptation.py:279,437`|ADA后换头并联合support FT30epochs，再kNN；额外target CE与更新成本|作者实际nShot路径含FT；正式原生完整版与ADA-only简化路径分列|
|RadioNet DF-FT|`RADIO/finetune.py:177`；`OLD/baselines/common/li_backbones.py:57,84`；同旧runner`:94`|冻结DF encoder，训练末层；Adam0.001、30epochs、batch128；support-only|完整旧评分；作者辅助路线，不代表ADA|
|RadioNet Triplet→kNN|`RADIO/triplet_network/triplet_training.py:171,250`；`triplet_test.py:27`|独立三元组训练30epochs、batch128、Adam0.001；target support近邻注册|完整作者训练／测试源；正式256点capsule接入缺失，不排除出设计|
|ProtoNet CDA／PTN，SenSys2022 poster|`ROOT/paper_reproduction/protonet_cda/train.py:594`；`model.py:13,124,141`；`run_wisig_suite.py:48`；`ROOT/comparison_suite/source.py:207`及`registration.py:352`|独立episode源训练，冻结encoder，support类均值＋平方欧氏距离；无source回放和额外target U|完整原生及匹配runner；不能由PAMF-NCM替代；C为项目prototype追加扩展|
|MRIOR-SDA，IoTJ2024|`PUB/paper_reproduction/mitigating_receiver_impact_da/train.py:253,946`、`algorithm.py:322`；`PUB/paper_reproduction/cvs_aligned/supervised_da.py:31,94`；`ROOT/comparison_suite/adaptation.py:480`|GAD/DV-KL，估计器每步7次再更新E/C；source L_s回放＋target CE；旧版100更新、Adam0.0006、CPL关闭|原论文UDA，SDA是明确监督扩展；闭集C=N/A；完整包主要在PUB|
|DADDA-SDA，IoTJ2025|`PUB/paper_reproduction/DADDA/train.py:265,364`、`losses.py`；`PUB/paper_reproduction/cvs_aligned/supervised_da.py:168,198,221`|source CE＋support CE＋动态MMD/LMMD；LMMD使用support真标签；旧版100更新|完整核心／匹配runner；原UDA的SDA扩展，C=N/A；不能把TableII方法列表当其他算法已实现|
|Receiver-Agnostic Two-stage UDA＋FT，GLOBECOM2023|`PUB/paper_reproduction/receiver_agnostic_twostage_uda/train.py:327,938`；`steps.py:49,73,146,175`；`ROOT/comparison_suite/adaptation.py:466`|DANN→LMMD→FT；前两步使用source与support IQ，末步target标签；旧版100＋100＋100更新|完整代码；与MRIOR完全不同；闭集C=N/A|
|Feature Separation／Hu2024，IET Communications|旧`ROOT/paper_reproduction/feature_separation_crossrx/`；新`ROOT/paper_reproduction/hu_feature_separation_2024/training.py:35`、`finetune.py:12,21`、`evaluation.py:22`|source TX/RX分离；target只更新共享encoder与TX路径，冻结RX路径；新实现Adam0.005、batch256；原论文25samples/TX FT|同一DOI`10.1049/cmu2.12841`的两套复现；Hu新实现更完整，不能无说明接替旧矩阵|
|RA-Collab单节点FT扩展|`ROOT/baselines/ra_collab/finetune_cvs.py:26,50,73,75,102`；`train_cvs.py`、`model.py`|全网SGD20epochs、lr1e-5、momentum0.9、batch32；FT时grl_lambda=0|完整单节点代码；旧入口从named_test抽support后仍评完整test，需严格split重绑；不加入协同融合|
|Tweak，IEEE Access2023|`ROOT/paper_reproduction/gaskin_tweak_2023/training.py:53`、`calibration.py:22,47,65,73`；完整真实LoRa runner为`ROOT/external_sources/rffi_labeled_da_20260915/Tweak_project_reconstruction/paper_reproduction/gaskin_tweak_2023/official_lora_experiment.py:328,348,462`|共享encoder triplet100epochs、SGD momentum0.9、margin0.1；target centroid/radius校准；原生M10聚合query embedding|完整算法和runner；原生M10另列多观测面板，M1为明确项目扩展。题名为Deep Learning Model Portability for Domain-Agnostic Device Fingerprinting|

## 3. 新类注册：保留完整方法而非统一类均值代理

|家族|核心与运行入口|注册状态与输入|版本／正式接入边界|
|---|---|---|---|
|CSIL，IoTJ2021|作者`ROOT/local_artifacts/external_refs/pcwhy_CSIL/ContinualLearning/WorkStage/CSIL.m`；`PUB/paper_reproduction/CSIL/{model,losses,protocol,train}.py`；`ROOT/comparison_suite/registration.py:211,228,414,497`|通道扩展、旧projection隔离、fingerprint mask、KD、source Fisher EWC；当前C只收新support，B为target SFT扩展|matched版本用六块IQ encoder；标准经验Fisher代替溢出的exp(grad²)，允许旧fingerprint块更新使EWC有效。不能叫作者逐位等价|
|MoPC-HR，TITS2025|作者`PUB/local_artifacts/external_refs/MoPC-HR/MoPC_HR_trainer.py:23,270,353`；`PUB/paper_reproduction/mopc_hr_non_exemplar_cil_sei/algorithm.py:25,51,79,119,145`；`ROOT/comparison_suite/registration.py:506,526`|历史prototype＋参考模型；新support CE、prototype augmentation、HR；漂移校正原型传入后续session|论文cosine校正／平方L2 HR与公开代码raw-dot/softmax等差异分列；最终使用完整分类器，最后校正不自动改变当次推理|
|Orthogonal Incremental SEI／OSC-FSCIL-SEI，通信学报2026|`PUB/paper_reproduction/orthogonal_incremental_sei/train.py:540,641,995`；`ROOT/comparison_suite/registration.py:570`|源正交/simplex目标、CE＋对比＋中心分离；C冻结encoder和旧权重，均值初始化新权重，margin＋prototype alignment|完整多session源码；matched joint版是显式扩展；未经正交源训练的PAMF头不能叫完整原方法|
|ISSL，TCCN2025|固定作者`ROOT/local_artifacts/external_refs/ISSL_20260930/{main_pre-train.py,main_self_supervised.py,main_transfer_learningt.py,main_increment.py}`；`ROOT/paper_reproduction/newclass_registration_20260930/acceptance.py:93,123`、`issl_fixed.py:168,194`|base→扩头→SSL→CE transfer；teacher/key/queue＋旧训练样本重复；另有CE＋KD分支|原行为版与fixed_v1分列；fixed修复KD图、zero_grad、queue、teacher/key BN。真实WiSig短预算闭环已验，正式arbitrary-K/capsule未接；没有原文全文不冒称逐公式复现|
|LoRa scalable/channel-robust RFFI，TIFS路线|作者`ROOT/local_artifacts/external_refs/LoRa_RFFI_20260930/Openset_RFFI_TIFS/main.py:22,121,184`；Torch源`ROOT/local_artifacts/external_refs/LoRa_RFFI_Torch_20260930/`；`ROOT/paper_reproduction/newclass_registration_20260930/acceptance.py:162,210`|channel-independent spectrogram→triplet embedding→KNN15；冻结encoder，追加enrollment；无需Fisher/teacher|8192→256点为明确预处理扩展，窗64／重叠32；Torch与TF2.1数值不等价；现有诊断A已用target support，应改叫B0，不能充正式A|
|ProtoNet注册扩展|`ROOT/comparison_suite/registration.py:631`，native episode见上表|继承B旧原型，追加new support类均值，全部类共同竞争|项目C扩展，保留自己的episode encoder；完整pipeline已有预测|
|RadioNet ADA注册扩展|`ROOT/comparison_suite/pipeline.py:249`；B见上表|复制B encoder及旧support kNN memory，追加新类memory|项目C扩展；ADA-only与ADA-FT两路径应分别绑定，不共用成绩|

固定外部源提交与下载记录见 [source_manifest.json](../paper_reproduction/newclass_registration_20260930/source_manifest.json)。CSIL、ISSL等原仓库只读保留；无许可证的ISSL源不重发，本设计只引用位置。

## 4. 现有通用实现与机制控制

|方法|真正实现|性质与限制|
|---|---|---|
|DANN|`PUB/paper_reproduction/receiver_agnostic_twostage_uda/train.py:327`可独立选dann，`steps.py:49`；通用`ROOT/external_sources/Pytorch-Template/trainers/dann_trainer.py:51,129`|完整通用算法；已有RFFI Two-stage分支可复用；不重复计为两篇RFFI论文|
|MCD|`ROOT/external_sources/Pytorch-Template/trainers/mcd_trainer.py:20,98`及`main.py`、`configs/mcd_m_mm.toml`|完整三步max/min discrepancy；默认MNIST→MNIST-M，未接RFFI。默认45epochs、discrepancy_steps4、weight15属于该配置，不是RFFI论文预算|
|AdaBN|`ROOT/comparison_suite/adaptation.py:393`|0梯度参数但更新BN状态；仅support，PAMF没有BN则N/A|
|CORAL＋SFT|`ROOT/comparison_suite/adaptation.py:501`|source CE＋support CE＋协方差对齐；项目监督机制扩展，不能冒称完整Deep CORAL复现|
|SFT|`ROOT/comparison_suite/adaptation.py:317,498`|support-only全网CE；已有默认20epochs、Adam0.001|
|Linear|`ROOT/comparison_suite/adaptation.py:381`|冻结encoder、新Linear头；已有默认20epochs、Adam0.001|
|Ridge|`ROOT/comparison_suite/adaptation.py:372`|闭式解、λ1、截距不惩罚；与MILKR的中心化目标定义不同，需单列|
|NCM|`ROOT/comparison_suite/adaptation.py:361`|support均值、归一化cosine；非独立ProtoNet|
|Frozen DG|`ROOT/comparison_suite/pipeline.py:219`|只作A，不计DA方法|

通用模板 `trainers/_trainer_base.py:171,217` 使用target test loss选best；接入时不能继承该行为。`baselines/paper_resnet/train.py`的source-RX DANN与DIFEX中的source-group CORAL属于Phase1 DG，不计Phase2 target DA。

## 5. 代码存在性与已完成实验分层

1. 根目录的DADDA、MRIOR、Two-stage、CSIL包不齐，MoPC根目录主要为README；`comparison_suite`直接import这些包。完整代码在PUB／历史镜像中，不代表当前根目录可直接启动。
2. `20260930-phase12-native-baselines-practical-m5-r01`的9家族共45源模型完成；35条Phase2流水线固定预测，CSIL/MoPC各5条C因`kaiming_uniform_(generator=...)`兼容性失败。该状态来自2026-10-01记录，**不是本轮实时远端核查**。见[历史状态报告](results/cvs_native_comparison_20261001/实验进展与Phase1结果.md)。
3. ProtoNet、Feature Separation、DADDA、MRIOR、TwoStage、Orthogonal、RadioNet各5条完整预测，不等于该批独立Phase2已评分；不能拿Phase1分数填B/C。
4. CSIL/MoPC另有更早400cell/1200场景行的完整适配结果，POSTER/RadioNet DF也有旧评分；契约、版本和输入不同，不补填本次缺项。
5. ISSL与LoRa已有真实WiSig训练／预测／保存／独立评分诊断，但old3/new3、K128、同RX同日、无LEO、短预算。它只证明链路，不证明本次正式性能。

## 6. 内部 D92／MILKR 变体及底层模块

主论文只把MILKR视为本方法；内部历史候选不是新增外部论文。下列家族仍完整进入设计清单，按完整runner可用性复现，匹配旧产物优先复用，不根据已知query分数只挑赢家。

|组|现有方法／实现家族|比较用途|
|---|---|---|
|早期决策路线|原D92 P2-256-FULL、SFHead、SCV、SGJoint、MVKME、BNNA、OSC／OrbitShared、MVRidge|历史方法发展与输入／摘要权限参照；不全部当MILKR单因素消融|
|分支与核|BranchRidge、BranchMetric、BranchOrbitCE、BranchInteraction、BranchLocalRidge、BranchLocalMargin、WithinClassMetric|机制主消融与平方间隔／CE辅助对照|
|适配与注册路径|SequentialResidual、JointChannel、JointSpectral、PrototypeTransport、MCResidual8、FunctionCoordinateResidual8|监督adapter与状态继承参照，明确相对冻结MILKR增加训练|
|旧函数与校准约束|AnchorJoint、AffineJoint、ConditionalJoint、MarginJoint、GroupBarrierJoint、GroupBalancedSupportMetricJoint、SupportMetricJoint、ProtoFrameJoint|旧新竞争、函数保护与support度量历史家族|
|Fisher族|FisherSubspaceJoint、AdaptiveFisherJoint、CalibratedFisherJoint、IteratedAdaptiveFisherJoint|保留完整Fisher定义、support输入及优化预算，不从名字推断source Fisher|
|原生神经适应|NativeSupportTail、NativeLocalRidge、NativeCrossfitRidge、NativeFullCrossfitRidge|native尾部／交叉拟合及读出配对，不能称冻结MILKR原版|
|底层组件|ground_summary、ground_classifier_a、positive/adaptive_score_calibration、group_gate、metric_step/basis、conditional_affine_kernel、proto_frame primitives、fisher primitives等|单独组件不计独立方法；是否完整算法以caller＋配置＋runner为准|

机器清单将逐文件记录当前`code/cvsrffi/d92*.py`的公开函数、字面方法ID、关联配置及工具入口；这是静态存在性盘点，不实例化模型、不证明所有导入依赖或数值运行成功。旧方法全部先列存在性／权限，主完整矩阵优先原D92、BranchRidge、BranchInteraction、MILKR；其他家族有合法source/support研究依据时按预登记诊断面板覆盖，不能无条件扩大为数十个新训练候选。

## 7. 已定位论文但未找到完整算法的缺失表

|名称|本轮结论|不允许的替代|
|---|---|---|
|CSCNet|有Sensors2025论文定位，未找到完整本项目算法／runner|不能用SHOT或普通SFT代表|
|Self-training＋Prototype Augmentation|TIFS2024，DOI`10.1109/TIFS.2023.3343193`；清单和文献记录存在，未找到对应完整实现|不是ISSL，也不能用MoPC原型增强顶替|
|SHOT|在MRIOR对比清单中提及，未找到独立训练实现|不能因伪标签或冻结头存在就计为SHOT|
|DAN、DSAN、CDAN、WD、独立Deep CORAL|DADDA计划／TableII名称存在，未找到对应完整独立算法训练实现|CORAL-SFT只是一项明确项目扩展|
|独立LwF、EWC、iCaRL|没有当前独立完整RFFI Python Phase2 runner；CSIL MATLAB有KD/EWC消融|CSIL内部loss不计新论文基线；MATLAB消融不冒充已集成结果|

主设计没有因这些缺失项而停下，也没有为了“方法数量”捏造代码。后续真补齐的实现可增补本表，但必须新绑定来源、版本、权限及runner，不能追认本次已完整。
