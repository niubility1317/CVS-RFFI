# CVS Phase1/Phase2对比方法总清单（2026-09-30）

本次先完成方法查找与归类，不启动新训练。范围为Phase1域泛化、Phase2旧类域适应和新类注册；协同、unknown拒识及联邦实验不纳入本轮执行清单。这里的“找全”指恢复原方案和本轮用户点名的方法，并补列已有实现及相关候选，不宣称穷尽整个RFFI领域。

**原方案曾明确列出ProtoNet CDA、MRIOR-SDA、DADDA-SDA、CSIL、MoPC-HR和Orthogonal Incremental SEI。9月27日批次尚未覆盖这6项，不能称为论文所需全部对比已完成。**NCM属于统一分类器对照，不能代替这些方法。

## 1. Phase1：源域训练与域泛化

|方法/稳定名称|任务与区别|已有代码定位|9月27日当前契约结果|
|---|---|---|---|
|CVCNN-CE|基础复值CNN＋交叉熵；结构对照，无独立对应论文|`baselines/cvcnn_ce`|有；另有PL扩展|
|RIEI-FD|接收机/发射机特征解耦；MI/IE/CE，不能与RA-Collab混同|`baselines/riei_fd`|有；CE与PL分列|
|DRIFT|特征解耦、域对抗、receiver center与negative MSE|`baselines/drift`|有；CE与PL分列；论文版本须绑定实际配置|
|POSTER源模型|Fine-Tuning论文的源模型；零适应测试属于DG评估，不能由此声称原论文提出DG算法|A镜像`baselines/common/li_backbones.py`中的`PosterHomegrown`|有|
|RadioNet-DF源模型|RadioNet的DF骨干；DF是架构选择，不能代替ADA/Triplet算法|A镜像`baselines/common/li_backbones.py`中的`RadioNetDF`|有|
|ProtoNet/PTN源模型|独立episode训练的度量学习骨干，后接CDA；必须与任意CE骨干＋类均值分开|`paper_reproduction/protonet_cda`；历史PTN episode诊断|本批缺失|
|Feature Separation|跨接收机特征分离和few-shot路线，不应只保留一个普通CNN代理|`paper_reproduction/feature_separation_crossrx`|本批缺失|
|RA-Collab单节点分支|接收机无关对抗表征及单节点微调可作候选；多节点融合归协同另文|`baselines/ra_collab`|本批缺失；补充候选|
|CVS DAOT A1＋FastTrust-RC4|本轮主方法，固定最终轮权重|本轮CVS登记run|有|

CE、PL以及历史augmentation-consistency是训练路线/项目扩展，不分别冒充独立论文方法。历史半监督协议的比例、信道、seed与best_by_val规定不覆盖本轮用户冻结的契约。

## 2. Phase2：旧类域适应

|方法/稳定名称|原论文与本项目扩展的区别|已有实现/来源|本批状态|
|---|---|---|---|
|ProtoNet CDA|作者PTN编码器＋support类原型和距离分类；共享其他骨干的CDA另标“共享骨干扩展”|`paper_reproduction/protonet_cda`；C镜像`cvs_aligned/supervised_da.py`及runner|历史代码/实验定位已找到；当前residual_noeq矩阵缺失|
|MRIOR-SDA|对应Mitigating Receiver Impact论文；原方法为UDA，本项目SDA加入target support监督，保留GAD/DV-KL；CPL是否执行单列|C镜像`mitigating_receiver_impact_da`、`cvs_aligned/supervised_da.py`|历史代码/实验定位已找到；本批缺失|
|DADDA-SDA|原DADDA为闭集UDA；本项目监督扩展加入target CE，LMMD使用support真标签；不能称为原生新类注册方法|C镜像`DADDA`、`cvs_aligned/supervised_da.py`|历史代码/实验定位已找到；本批缺失|
|POSTER Fine-Tuning|用户点名的监督微调对比；当前为项目数据/信道下的作者实现移植|A镜像`tools/run_practical_phase2_baseline.py`|已有；不能用POSTER/NCM替代|
|RadioNet ADA＋k-NN（正式主对比）|用户于2026-09-30要求使用论文推荐方法；正文III节提出ADA训练后冻结特征提取器，以target training/support特征构建k-NN，替代source分类器|作者仓库`ADA/af_classifier.py`；须核对完整ADA＋k-NN调用链而非只取GRL|本批缺失；后续RadioNet默认采用此路线|
|RadioNet Fine-Tuning（辅助对照）|普通监督微调，记录DF/homegrown架构；不得再代表RadioNet论文提出方法|A镜像当前runner；作者仓库`finetune.py`|已有DF路线，保留原结果|
|RadioNet Triplet（候选对照）|三元组迁移对照；论文主方法选为ADA＋k-NN后，不默认扩大为必跑三条路线|作者仓库`Triplet_network/triplet_training.py`|本批缺失；不替代主方法|
|Feature Separation few-shot|特征分离论文的完整few-shot流程，区分源训练与目标适应所用标签|`paper_reproduction/feature_separation_crossrx`|代码定位已找到；本批缺失|
|Receiver-Agnostic Two-stage UDA＋Fine-tuning|Bao等GLOBECOM 2023；两阶段UDA和后续微调，不能与MRIOR-SDA混同|C镜像`receiver_agnostic_twostage_uda`|组件定位已找到；本批缺失|
|冻结DG、NCM、线性探测、全网SFT/PEFT|零适应、分类器和更新范围的基础对照；不视为4篇原生RFFI算法论文|DG/NCM已有；其余需按具体配置分列|DG/NCM有，其余缺失|
|CVS-D92 E0去RF32|本轮主方法：P2-256-FULL，identity160＋FFT96|恢复run `20260928-phase2-cvs-d92-practical-manytx-m5-r02`|本批有|

用户所说“另外一个SDA”在原始对比协议中对应**MRIOR-SDA**。SDA在这里表示监督域适应的项目扩展，不是另一个与MRIOR无关的论文标题。

## 3. Phase2：类增量与新类注册

|方法/稳定名称|应保留的核心流程|已有实现/来源|本批状态|
|---|---|---|---|
|CSIL|Channel Separation Enabled Incremental Learning；通道扩展、旧通道隔离、KD/EWC；原ADS-B协议与ManyTx扩展分列|C镜像`CSIL`、`cvs_aligned/adv3b02_official_repo_ci.py`；作者MATLAB代码|历史代码/实验定位已找到；本批缺失|
|MoPC-HR|动量原型校正、原型增强、层次正则；不能仅用类均值头代理|C镜像`mopc_hr_non_exemplar_cil_sei`及`cvs_aligned`；公开作者仓库|历史代码/实验定位已找到；本批缺失；公开代码语义差异需记录|
|Orthogonal Incremental SEI|正交空间约束FSCIL-SEI；基类几何训练、冻结编码器、新类权重初始化与增量校准|`paper_reproduction/orthogonal_incremental_sei`；C镜像`cvs_aligned/adv3b02_paper_full_ci.py`|历史代码/实验定位已找到；本批缺失|
|ProtoNet新类注册扩展|在独立PTN编码器上追加新类support原型，旧新类统一竞争；另标CVS注册扩展|ProtoNet组件＋待绑定当前契约接口|本批缺失|
|NCM/权重印刻/线性头/SFT|解释类竞争和更新范围的基础对照，逐项说明冻结/可训练参数和旧类保持方式|NCM及两条FT已有；其余待具体实现绑定|已有部分；不能代替CSIL/MoPC-HR/正交方法|
|CVS-D92联合注册|主方法旧/新support联合建模；E0去RF32版本保持明确|本轮恢复run|有；现有B与C为独立重新拟合，不宣称B→C状态继承|

DADDA/MRIOR的闭集DA结果不能自动填入新类注册表。若扩展其分类器处理新类，必须分别说明扩展公式与是否继承B阶段模型；不能与独立CIL方法混为同一项。

## 4. 相关补充候选，未自动列为新增训练任务

这些方法用于完善候选池与论文定位，不用其数量冒充执行完成，也不在本次查找中扩大训练矩阵。

|类别|候选|依据/待核对项|
|---|---|---|
|源数据可用的通用DA|DANN、DAN、DSAN、Deep CORAL（DCORAL）、CDAN、WD|DADDA的Table II复现清单明确列出；具体WD论文/版本及各实现仍需绑定，不能只按缩写启动|
|源数据不可用的RFFI适应|CSCNet|2025年Sensors的Cross-Receiver Radio Frequency Fingerprint Identification: A Source-Free Adaptation Approach；已定位论文，代码与当前K-shot监督扩展尚未确认|
|RFFI类增量|A Class-Incremental Approach With Self-Training and Prototype Augmentation for Specific Emitter Identification|TIFS 2024，DOI定位`10.1109/TIFS.2023.3343193`；论文/算法逐项确认及代码入口待补，不擅自创造缩写|
|通用遗忘控制|LwF、EWC、iCaRL|相关候选；原生replay、teacher、Fisher及内存权限需披露，CSIL内部EWC不能算作独立EWC实验|
|适应机制对照|AdaBN、CORAL、线性探测、SFT、PEFT|可解释适应来源及资源代价；优先级与预算在后续实验配置确定|

## 5. 论文与版本证据

|方法|核查来源|版本/限制|
|---|---|---|
|RIEI|[作者预印本](https://arxiv.org/abs/2411.03636)|Domain Generalization for Cross-Receiver Radio Frequency Fingerprint Identification；RIEI与FedRIEI分开，本轮不加联邦|
|DRIFT|[作者预印本](https://arxiv.org/abs/2510.09405)|v1为2025-10-10，v2为2026-05-26；本地历史审计引用过v1，不凭目录名断言当前实现完全匹配v2|
|POSTER Fine-Tuning|[作者所在高校PDF](https://cse.unl.edu/~nghose/pubs/conf/papers/LI_Wisec21.pdf)|WiSec 2021；PDF模板中的占位DOI不能作为正式DOI引用|
|RadioNet|[作者仓库](https://github.com/UCdasec/RadioNet)、[高校论文PDF](https://cse.unl.edu/~nghose/pubs/conf/papers/LI_CNS22.pdf)|CNS 2022；仓库明确分Fine-tuning、ADA与Triplet，不以DF骨干代称全部路线|
|ProtoNet CDA|[高校论文PDF](https://www.eng.auburn.edu/~szm0001/papers/sensys22.pdf)、[作者代码定位](https://github.com/stevester94/csc500-main)|SenSys 2022 poster；独立episode训练与共享骨干CDA为两种实现口径；目标loss选模不能沿用到干净source-only对比|
|MRIOR|[作者论文](https://arxiv.org/abs/2404.08566)|Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation；2024 IoTJ接收稿；MRIOR-SDA为本项目监督扩展|
|DADDA|[论文DOI定位](https://doi.org/10.1109/JIOT.2025.3573713)、C镜像`DADDA/paper_checklist.md`|Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment；2025 IoTJ；当前无已核实作者完整代码等价性|
|Feature Separation|[出版社全文](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cmu2.12841)|Few-shot Cross-Receiver Radio Frequency Fingerprinting Identification Based on Feature Separation；IET Communications 2024|
|Two-stage UDA＋FT|C镜像`paper_reproduction/README.md`及配置；GLOBECOM 2023目录定位|Receiver-Agnostic Radio Frequency Fingerprinting Based on Two-stage Unsupervised Domain Adaptation and Fine-tuning；具体论文DOI待核，不猜填|
|CSIL|[作者论文](https://arxiv.org/abs/2105.06381)、[作者代码](https://github.com/pcwhy/CSIL)|Class-Incremental Learning for Wireless Device Identification in IoT；2021；MATLAB原版与PyTorch移植区分|
|MoPC-HR|[论文DOI定位](https://doi.org/10.1109/TITS.2025.3559174)、[公开仓库](https://github.com/xmuLdz/MoPC-HR)、C镜像README/PDF定位|Non-Exemplar Class-Incremental Learning via Prototype Correction and Hierarchical Regularization for Specific Emitter Identification；2025 TITS。公开README描述SSCL/SCW/curriculum，与本地论文记录的原型校正/层次正则不同；仓库存在不证明版本完全一致|
|Orthogonal Incremental SEI|[期刊正文入口](https://www.joconline.com.cn/zh/article/doi/10.11959/j.issn.1000-436x.TXXB260021/)|正式题名《正交空间约束的特定辐射源识别小样本类增量学习方法》，通信学报2026；本地较早PDF题名略异，不重复计数|
|CSCNet|[出版社论文入口](https://www.mdpi.com/1424-8220/25/14/4451)|已定位公开论文；未确认当前实验接口/作者代码|

RadioNet主方法选择依据为作者正文III节的“RadioNet: Our Proposed Method”，见[作者论文镜像](https://homepages.uc.edu/~wang2ba/files/pub/cns22_haipeng.pdf)。ADA源/目标对齐训练与随后冻结编码器的k-NN步骤都要保留；k-NN不是类均值NCM。近邻数、距离、骨干、源域访问及训练轮数在实现时按论文/作者版本绑定，不能用目标query成绩选参数。原论文为跨天闭集适应；用于跨接收机LEO或新类注册时仍应标明项目扩展，不能声称论文原生处理全部新类任务。

查找使用现有实验索引、原协议、论文/作者仓库及出版社入口。学术检索MCP工具本会话不可用，使用web核查替代；Crossref请求返回HTTP 429，IEEE部分DOI入口无法读取，相关DOI保留为文献定位，不写成全文复核成功。未读取目标成绩来选算法、修改超参或选择seed。

## 6. 当前执行范围与后续绑定

“已找到代码”“有历史产物”“本批已完成”必须分开。历史记录只证明存在定位入口，不证明当前运行状态、checkpoint可继承或本轮科学契约合规。原始方案见`docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md`；历史索引见[实验总索引](../experiment_registry/README.md)，特别是`by_method/protonet.md`、`csil.md`、`mopc.md`。历史报告的旧信道、旧split与旧权重不能补填本批缺项。

代码定位：A=`E:/type10-7/code/snapshots/cvs_practical_baselines_20260927_wt`；C=`E:/type10-7/github_publish/rffi-labeled-da-three-20260915`。未标A/C的代码路径指当前主Git承载工作树；根目录镜像可能只保留部分模块，应按真实路径取实现。

后续每个执行方法单独绑定论文版本、实现commit、骨干、数据权限、checkpoint来源、训练预算、6类seed和日志路径。继续采用本轮practical LEO `residual_noeq`、固定5个模型seed及最后一轮权重。不得按已见目标成绩挑选seed或删去低分方法。外部方法按当前`项目.md`Phase2章节的显式例外披露所需source/历史状态；原生UDA需要的额外目标无标签训练池与query测试池必须明确区分。不存在独立训练池时不能静默拿query替代并称为干净归纳评估。

结果按同一row报告A：适应前旧类准确率；B：仅旧类support适应后旧类准确率；C：注册新类后旧类、新类、总体准确率和H。旧类support/query保持配对，明确独立重新拟合或B→C继承；再报告B−A、B−C旧类下降、旧新类绝对差及完整K×新增类数表。缺失项记N/A。资源指标只填写实测值。
