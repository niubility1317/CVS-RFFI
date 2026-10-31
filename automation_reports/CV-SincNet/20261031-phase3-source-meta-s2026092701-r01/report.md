# 源接收机留一二阶联合泛化

ANALYZED，执行VERIFIED；科学NO_PROMOTION，完整目标ACTIVE。

# 源接收机留一的联合身份与开放表征训练

Run `20261031-phase3-source-meta-s2026092701-r01`，唯一owner `codex-phase3-source-meta-20261031`。本轮为纯源域泛化候选，不含新目标端适应、发现或注册；原PCD研究基线、DURA默认、历史失败结果与完整目标ACTIVE保持。源Us重放/PL与统计混合已有负结果，原生身份自训练未达到联合目标；新增RX34的恢复前后配对又显示原F0在raw星地闭集39.689884%、DURA后41.721059%，因此本轮检验源端跨RX优化的训练目标，而非重调当前目标阈值。

方法参考[Li等的MLDG原论文](https://arxiv.org/abs/1710.03463)：用源域内部的训练/留出划分，优化一次虚拟源训练更新后的留出损失。本项目此前的TX成员集合episode和一阶冲突投影不含该虚拟更新。本行是自选损失、预算与超参数的RFFI变体，不声称复现论文图像实验或已获得泛化收益。

## 两臂、原生参数与数据

ERM_NATIVE与META_NATIVE均独立初始化为完整原F0 reader/default6（232993参数）加合规纯源r03 projection（102688），总335681、冻结Sinc48后可训练335633。原F0的完整身份初值须由真实合成smoke确认logits一致，初始开放z不承诺等价r03；不加载目标适应PCD初始化。原F0/r03/PCD/DURA及学生buffers不更新，学生reader非Sinc参数、原默认头和projection由外层优化器联合训练。不部署reader外667729参数、虚拟模型或新分类头。

只用source392005/VALIDATED_ONCE的既有星地DURA P1：L6300六旧类真标签，OE6134仅outside-six成员监督及公共RX/day、无OE真TX/K。源RX为1/3/4/6/8（1-19/18-2/19-2/2-19/3-19），日期2021-03-08/15/23，eq1；六旧TX为14-10/14-7/20-15/20-19/6-15/8-20。Us56700只保留Phase1祖先，本轮不读；目标U值/U IDs/Q、V/NV及truth训练读取均0。旧PCD仅作冻结比较，有原RX34 U576祖先，不向新学生提供初始化或教师。

每步held RX按[1,3,4,6,8]循环。其余四RX提供L128（各32）与OE64（各16）；held RX提供L64/OE32。合法L按源y/RX/day分层、OE只按公共RX/day覆盖源三日，防止原跨RX/day候选为空；不按未知真身份采样。四独立固定采样流2026103180/81/82/83，两个arm逐步全部索引完全匹配。重复曝光不计为新增物理样本；held只是当前训练划分，原F0与父r03已见过源五RX，不能声称严格未见接收机盲测。

## 确切元目标与计算

源训练F在原参数θ上计算L128/OE64，保持旧共同目标：crossset_metric＋OE_topology_KL＋10×balanced_anchor＋CE_L＋KD_L。anchor为0.5×L均值＋0.5×OE均值，父z来自冻结纯源r03，KD来自原F0缓存同P1 logits（T2、KL×4），CE/KD权重1。

计算θ′＝θ−α∇F。ERM的α=0，META的α=0.001；外层目标均F＋0.5G(θ′)，beta=0.5固定。META使用`create_graph=True`与stateless functional_call，保留二阶链式梯度，不detach虚拟梯度或候选表征，不把临时权重写回原模型。Sinc不参与虚拟更新，buffers保持eval冻结；不存在目标query内更新或额外部署参数。

G只对held L64/OE32的anchor平均，但旧跨RX/day关系损失需要其他接收机候选。因此在θ′下重新前向[trainL128,heldL64,trainOE64,heldOE32]共288行，held anchors的关系候选由另外四RX的train bank提供，沿用原不同RX且不同day掩码；G的CE/KD及balanced_anchor只取held角色。新core实现anchor行选择，all-anchor形式应与旧损失数值/梯度一致；不把单RX关系损失置零或静默改成同RX。source train bank和held都使用θ′，不能混用θ的候选。

两个arm都实际执行原参数192行＋虚拟参数288行，共480行/步，固定各444步/合计888；外层AdamW lr1e-4/wd1e-4/eps1e-8、普通normclip5，模型eval但保留梯度。数学上unused参数的虚拟梯度若为None应显式记录并保持该参数，不造梯度；真实浅层time_fuse/中层t2/defaulthead/projection路径仍须有限且有梯度。保存每步源/held各分量、外层总loss、虚拟step范数、二阶状态、梯度/clip、源RX划分与完整索引，不根据Q修改alpha/beta或预算。

新增学生训练前向426240＝2×444×480，其中base170496、virtual255744；父r03源L/OE缓存12434，原F0 L教师6300精确复用既有cache，训练新F0/G真实0。合成smoke单列。评估新学生126120，包含两臂各42584源行＋两Q各10238；Q共享原F0/G各20476。新增信号前向含父cache共564794，继承原F0/r03/PCD/G训练与旧缓存成本保留；墙钟不是B1延迟。

## 冻结评估和判定

每臂冻结后只用源L/OE拟合自己的known90/OE32密度及normalizer，V27000仅校准两臂pooled/predicted四组六τ（24新值）。与旧E0/OLD_PCD的24值合计48个FP64/384B；旧前四读出精确复用。新学生各自默认头决定closed，并路由自身密度/τ；保护基准仍原F0同DURA闭集，另存学生自身闭集。

完整8方法×3DURA条件：E0/OLD_PCD/ERM_NATIVE/META_NATIVE各pooled/predicted，唯一主META_NATIVE_predicted_ratio。三类为源已知RX星地、跨RX clean＋DURA、跨RX星地＋DURA，clean只冻结测试。源NV3150三设备仅独立评分；目标开发RX3-2/index34的future03-15/23共10238条（known4508、unknown5730），仍为既有canonical clean/star配对，不读取早日U576。目标四未知5-5/10-7/2-19/7-11名单仅builder/scorer，后者早U未见；03-23缺旧20-19/8-20，完整宏AK/CK/H/闭集保持N/A。正式RX24-18不读，已有NV/Q曝光、单seed/RX与global physical history UNKNOWN保留。

神经和密度先于V冻结，所有τ先于NV/Q冻结；全部8×3预测固定后独立CPU scorer才连接NV/clean/star三truth各一次。主报逐真实TX宏A_K/C_K/R_U/H、原F0和学生闭集、RX/day/scene/TX/day×scene、AUROC/OSCR/CI及相同ID配对得失。源known12与42诊断分别报告。主clean RU≥95%、star RU≥90%、CK≥90%、原F0同G闭集−AK≤2pp且各必需主景完整；主star AK须高于ERM和OLD_PCD，RU不低于E0/ERM/OLD_PCD，全部满足才source_meta_success=True；不通过NO_PROMOTION、不扩seed或换主。即使源DG通过，adaptation_success=None/new_target_DA=False，AN/H_OW N/A，完整目标仍未完成。

固定DURA v1.0 R69、reuse/graph开/native关、FP32/TF32off；G10为600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、input RMS一次/output RMS无，源scene3001/channel4001/future3133/3134/receiver2027继承reference。没有重抽信道或clean参考恢复。唯一实际checkpoint无query smoke为launcher首步，验证原生logits初值、虚拟状态不写回、真实二阶反向与Sinc/buffers保持，PASS立即继续源训练。

实际commit/命令/环境CWD/路径/GPU见本run登记。默认N607、卡0至7实际free≥4096MiB、每卡新增最多2本任务worker，其他owner保持。单不可覆盖run/output/release，单包一次SHA比较、remote compile、PID/CWD/argv/GPU/log-growth读回。停止限协议/祖先/输出冲突、非有限、无法执行或无法合法产生prediction，不按科学低分停止；技术失败保留root后一次针对修复与独立新run，不盲重启。预期train两CP与全888日志、models/τ/source与targets全部预测、独立score、pipeline/stdout与同row报告及登记Git交付。


实际code `e4a035b6f01a7ece98ec0dff2a6ea44867b8d536`；环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-source-meta-s2026092701-r01-release`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.source_meta_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-source-meta-s2026092701-r01-release/runtime.json`。


# Phase3 源 RX 留一的二阶联合身份／开放训练结果

**执行完成，科学结论为 NO_PROMOTION。** 唯一主方法 `META_NATIVE_predicted_ratio` 的星地 A_K/C_K/R_U/H 为22.781346%/41.166133%/50.548954%/31.407841%。A_K 较匹配 ERM 低0.173455个百分点、较 OLD_PCD 低5.170376个百分点；绝对保护、拒识与接收率门槛均失败。保留原 PCD 研究基线与 DURA 默认，完整目标 ACTIVE；本轮无新目标适应，`adaptation_success=None`、`new_target_DA=False`，AN/H_OW为N/A。

Run `20261031-phase3-source-meta-s2026092701-r01`，实际代码 `e4a035b6f01a7ece98ec0dff2a6ea44867b8d536`。依据[计划](PAMF_PHASE3_SOURCE_META_PLAN_20261031.md)、[完整888步分析](../local_artifacts/pamf_phase3_source_meta_20261031_r01/analysis.json)和[固定评分](../local_artifacts/pamf_phase3_source_meta_20261031_r01/score/results.json)；只读取既有JSON/完整日志，不重读IQ、权重、数组或真值，不新增测试、拟合、评分或阈值扫描。

## 方法、初始化与数据权限

方法参考[Li等的MLDG原论文（arXiv:1710.03463）](https://arxiv.org/abs/1710.03463)，但本行是自选共同损失、源RX划分、计算预算和超参数的RF变体，不声称复现原图像实验。ERM_NATIVE与META_NATIVE均独立构造完整原F0 E50/11100 reader/default6，加纯源r03 E_SIGNAL/2220 projection；真实合成烟测的初始身份logits与原F0最大绝对差为0。body来自F0，因此初始开放z不承诺等价r03，不以target-adapted PCD初始化。

学生reader232993＋projection102688＝335681参数，冻结Sinc48后训练335633，默认头W960随body/projection联合训练。原F0、r03、PCD、DURA和学生buffers保持；不部署wrapper外667729参数、虚拟模型或额外身份头。ERM仅指本同预算F＋0.5G框架的α0控制，不等同于所有普通源ERM配方。

每步held RX按1/3/4/6/8循环，其余四RX提供L128（各32）/OE64（各16），held提供L64/OE32，覆盖源三日。F在θ上计算crossset_metric＋OE_topology_KL＋10×balanced_anchor＋CE＋KD；anchor平衡L/OE各半，父z为纯源r03，KD为缓存原F0同P1的T2 KL×4，CE/KD权重1。

θ′＝θ−α∇F，ERM α=0、META α=.001，外层均F＋.5G(θ′)。META使用create_graph与stateless functional_call保留完整二阶链，不detach虚拟梯度或候选，不把虚拟状态写回模型。G仅以held L64/OE32作anchors，但在θ′下重新前向trainL128＋heldL64＋trainOE64＋heldOE32共288行，其他四RX候选也来自θ′；跨RX且跨day掩码保持，不能用θ候选或将单RX关系损失置零。G的CE/KD和anchor只取held角色。

每臂444步，外层AdamW lr1e−4/wd1e−4/eps1e−8、ordinary normclip5，eval数值状态但梯度开启。四采样流seed2026103180/81/82/83，两臂全部索引匹配。held仅当前源划分，原F0/r03已经见过这五RX，不是严格未见RX验证；OE只有成员监督及公共RX/day，无真实TX/K。

源合同392005/VALIDATED_ONCE：L6300、OE6134、Us56700祖先；新训练只读L/OE星地P1、L标签与缓存教师，Us/U值/U IDs/V/NV/Q/真值训练读取均0。源RX为1-19、18-2、19-2、2-19、3-19，日期2021-03-08/15/23、eq1；旧TX为14-10、14-7、20-15、20-19、6-15、8-20。V27000只校准新阈值，不是独立已知测试；NV3150含19-15、19-16、7-1各1050条，只冻结后评分。

目标开发RX3-2/index34的future Q为03-15/23的5525/4713条，共10238，旧4508／四未知5730。旧六类容量1491/1182/150/406/1186/93；5-5、10-7、2-19、7-11为1454/1439/1427/1410，后来7-11整类不在早日U。旧PCD比较保留U576祖先，新两学生和新源密度无目标U祖先。源/U/Q角色互斥、clean/star同canonical顺序；03-23缺20-19/8-20，完整闭集/AK/CK/H为N/A，不能删类认证。

全部三条件经固定DURA v1.0 R69 E20/4940（266951参数、FP32/TF32off、reuse/graph开、native关）。G10为600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、一次input RMS/无output RMS，既有源/future/receiver种子继承，不重抽信道；clean仅冻结测试。NV/Q历史曝光、单RX/seed、源日期重合、global physical history UNKNOWN、正式RX24-18未读保持，不能称新盲测。

## 全量训练与二阶证据

两臂各1至444步连续、有限，共888步；四组索引逐步匹配、train/held索引集合互斥，held循环次数为89/89/89/89/88。分析独立核对索引计数、范围、集合和循环；源RX归属、类别/日期分层及RNG到记录的分配由producer验证，未重新打开源标签/RX元数据重建采样器。

所有F/G分量及F＋.5G重建最大残差为ERM3.73e−7、META2.68e−7；未重算逐样本CE/KL或Hessian。首步两臂F同为4.065773，G为4.197691/3.854949，外层已分歧。下表为全记录三等段均值；每格“F/G”，anchor未乘10，KD已含T²=4。

| 臂、步段 | crossset F/G | OE_KL F/G | anchor F/G | CE F/G | KD F/G | F/G总损失 | 外层F+.5G |
|---|---|---|---|---|---|---|---:|
| ERM 1–148 | .799138/.804816 | .093493/.091185 | .088147/.087104 | .252570/.244462 | .030338/.031202 | 2.057013/2.042705 | 3.078366 |
| ERM 149–296 | .665660/.667391 | .085660/.087447 | .043035/.043039 | .201953/.209151 | .045260/.044658 | 1.428879/1.439037 | 2.148398 |
| ERM 297–444 | .648210/.638427 | .081925/.082197 | .029835/.029902 | .202194/.194081 | .056463/.055974 | 1.287138/1.269698 | 1.921987 |
| META 1–148 | .793609/.795568 | .093928/.089912 | .086335/.084281 | .251017/.242593 | .031043/.031112 | 2.032944/2.001996 | 3.033943 |
| META 149–296 | .657567/.655051 | .085151/.085106 | .040831/.040655 | .199578/.207000 | .047105/.045396 | 1.397711/1.399101 | 2.097262 |
| META 297–444 | .638296/.623355 | .081243/.078927 | .027942/.027713 | .199142/.190307 | .058672/.058017 | 1.256769/1.227731 | 1.870634 |

ERM/META外层首末6.164619→1.620091、5.993248→1.562675，范围1.228370–6.164619/1.180734–5.993248。META的F和G训练均值更低不等于目标泛化成功。META虚拟步范数全程非零，范围.003224–.020953、均值.006339，三段.006448/.006073/.006496；与α×inner_gradient_norm重建残差0。ERM虚拟步全程0；两臂都请求完整计算图，但α0控制的二阶贡献不活跃，META才有活跃二阶链，均无virtual update提交。

76个可训练参数tensor在全部内层/外层记录均无unused项，2个冻结tensor为Sinc；候选detach标志888步全为False。head/浅层/中层/body/projection每臂444步均有非零梯度，ERM/META均值分别为.866077/.693334、3.905554/3.491975、1.503024/1.321570、8.173544/6.975189、1.910682/1.668299；投影hidden/last亦有梯度。日志不包含每候选独立VJP，不能把未detach和整体梯度夸成每条候选每步梯度都非零；真实CP smoke与核心测试验证二阶和候选路径。

ordinary梯度裁剪ERM440步、META426步；裁剪前范数范围4.654882–31.665478/3.985956–27.801956，均值8.485479/7.252059。Sinc梯度/optimizer步和physical clip全为0，buffers、原父保持，学生body/head/projection实际改变。首步骤smoke身份等价0、真实二阶反向、虚拟状态不写回及Sinc冻结均PASS；不能用频繁普通clip推定科学失败的唯一原因。

每臂采样L192/OE96，曝光L85248/OE42624；trainL/trainOE/heldL/heldOE分别56832/28416/28416/14208。其unique分别6300/5938/6227/5127，L联集6300、OE联集6082。实际前向每步L320/OE160＝480行（θ192＋θ′288），每臂base85248＋virtual127872＝213120行；两个arm426240行，候选重复计算不增加物理样本。

## 完整24个运行点

下表为百分比；A_K/C_K/R_U按所需真实TX等权宏平均，C_K为已知接收率，H为A_K/R_U调和均值。原闭集固定为同DURA原F0，学生闭集不含拒识；新学生自己的closed同时路由自身源密度与τ，不能用较低学生闭集替换原F0保护基准。

| 方法 | 条件 | 原闭集 | 学生闭集 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 pooled | source | 90.981481 | 90.981481 | 89.770370 | 97.107407 | 60.888889 | 72.561330 |
| E0 predicted | source | 90.981481 | 90.981481 | 89.744444 | 96.759259 | 61.936508 | 73.291437 |
| OLD_PCD pooled | source | 90.981481 | 90.981481 | 89.770370 | 97.081481 | 63.619048 | 74.465443 |
| OLD_PCD predicted | source | 90.981481 | 90.981481 | 89.733333 | 96.748148 | 64.285714 | 74.907247 |
| ERM_NATIVE pooled | source | 90.981481 | 91.559259 | 89.896296 | 96.474074 | 68.730159 | 77.901088 |
| ERM_NATIVE predicted | source | 90.981481 | 91.559259 | 89.881481 | 96.422222 | 69.015873 | 78.078693 |
| META_NATIVE pooled | source | 90.981481 | 91.596296 | 89.844444 | 96.318519 | 68.158730 | 77.513420 |
| **META_NATIVE predicted** | source | 90.981481 | 91.596296 | 89.825926 | 96.274074 | 68.603175 | 77.793078 |
| E0 pooled | clean | 48.567015 | 48.567015 | 31.187288 | 43.537692 | 62.980616 | 41.716859 |
| E0 predicted | clean | 48.567015 | 48.567015 | 30.999690 | 42.654045 | 64.187968 | 41.808091 |
| OLD_PCD pooled | clean | 48.567015 | 48.567015 | 32.123647 | 44.791196 | 60.790839 | 42.034854 |
| OLD_PCD predicted | clean | 48.567015 | 48.567015 | 31.661804 | 43.761391 | 61.979647 | 41.912794 |
| ERM_NATIVE pooled | clean | 48.567015 | 45.450907 | 24.559780 | 35.294750 | 66.905430 | 35.930222 |
| ERM_NATIVE predicted | clean | 48.567015 | 45.450907 | 24.130903 | 34.854790 | 67.755678 | 35.587474 |
| META_NATIVE pooled | clean | 48.567015 | 45.134730 | 23.797165 | 34.627494 | 66.569330 | 35.060811 |
| **META_NATIVE predicted** | clean | 48.567015 | 45.134730 | 23.871772 | 34.733272 | 67.296960 | 35.242295 |
| E0 pooled | star | 41.721059 | 41.721059 | 26.867357 | 47.141720 | 51.837083 | 35.391279 |
| E0 predicted | star | 41.721059 | 41.721059 | 26.440597 | 46.197372 | 52.658525 | 35.204508 |
| OLD_PCD pooled | star | 41.721059 | 41.721059 | 29.175335 | 50.230381 | 48.454372 | 36.420916 |
| OLD_PCD predicted | star | 41.721059 | 41.721059 | 27.951722 | 48.582412 | 49.307698 | 35.678111 |
| ERM_NATIVE pooled | star | 41.721059 | 39.429149 | 23.393654 | 41.009449 | 50.438791 | 31.962848 |
| ERM_NATIVE predicted | star | 41.721059 | 39.429149 | 22.954801 | 40.542538 | 50.837654 | 31.628389 |
| META_NATIVE pooled | star | 41.721059 | 39.252284 | 22.817803 | 41.191411 | 50.184478 | 31.371610 |
| **META_NATIVE predicted** | star | 41.721059 | 39.252284 | 22.781346 | 41.166133 | 50.548954 | 31.407841 |

## 三控制配对与分组

同ID“增/失/净”为条数，不是宏百分点。新主与E0、OLD_PCD、ERM的完整配对如下。

| 主相对控制 | 条件 | 闭集正确增/失/净 | 正确接收增/失/净 | 未知拒对增/失/净 | ΔA_K / ΔR_U（pp） |
|---|---|---|---|---|---|
| E0 predicted | source | 358/192/+166 | 331/309/+22 | 301/91/+210 | +.081481 / +6.666667 |
| E0 predicted | clean | 103/217/−114 | 24/152/−128 | 407/225/+182 | −7.127919 / +3.108992 |
| E0 predicted | star | 127/153/−26 | 75/117/−42 | 278/395/−117 | −3.659251 / −2.109571 |
| OLD_PCD predicted | source | 358/192/+166 | 345/320/+25 | 238/102/+136 | +.092593 / +4.317460 |
| OLD_PCD predicted | clean | 103/217/−114 | 32/151/−119 | 445/138/+307 | −7.790032 / +5.317313 |
| OLD_PCD predicted | star | 127/153/−26 | 66/114/−48 | 310/237/+73 | −5.170376 / +1.241255 |
| ERM_NATIVE predicted | source | 27/17/+10 | 41/56/−15 | 43/56/−13 | −.055556 / −.412698 |
| ERM_NATIVE predicted | clean | 13/11/+2 | 15/15/0 | 108/133/−25 | −.259131 / −.458718 |
| ERM_NATIVE predicted | star | 14/9/+5 | 23/10/+13 | 75/91/−16 | −.173455 / −.288700 |

相对ERM，星地正确接收净增13、闭集正确净增5，但宏AK/闭集反而下降.173455/.176865 pp，因为各TX容量不同且逐类等权；不能混成总正确条数下降。主星地闭集正确1918、正确接收1516，正确但拒绝402、错误接收980；相对OLD正确接收少48、误接收少89、未知拒对多73，不构成联合成功。相对ERM则接收净增35，其中正确+13、错误+22，未知拒对−16。

| 主星地主场景 | 已知/未知数 | 学生闭集 | A_K | C_K | R_U | H |
|---|---:|---:|---:|---:|---:|---:|
| practical_high | 206/256 | 45.005176 | 22.325453 | 34.075125 | 59.259071 | 32.432269 |
| practical_mid | 293/443 | 45.101108 | 18.917131 | 31.167109 | 58.760264 | 28.620312 |
| practical_low_suburban | 1715/2167 | 43.759276 | 23.771985 | 37.633358 | 52.712298 | 32.766888 |
| practical_high_urban | 197/244 | 46.083294 | 37.223370 | 51.892736 | 55.023573 | 44.406085 |
| practical_mid_urban | 315/397 | 42.270479 | 24.674970 | 37.998485 | 50.043755 | 33.052709 |
| practical_low_urban | 1782/2223 | 31.674349 | 21.670864 | 47.316766 | 45.462673 | 29.350916 |

六场景合并两日后均覆盖旧六类/未知四类，但全部联合约束失败。主星地逐旧TX AK/CK依次为45.606975/61.904762、10.659898/64.382403、2/22.666667、1.477833/7.142857、57.588533/60.792580、19.354839/30.107527%；clean AK为47.216633、8.206430、.666667、.246305、58.937605、27.956989%。四未知clean/star RU为44.291609/36.588721、86.379430/66.851981、70.077085/44.428872、68.439716/54.326241%，不能用某一未知收益代替完整宏指标。

目标仅RX34；03-15 clean/star AK23.587545/22.679025%、RU66.475275/50.854448%；03-23因缺两个旧类，完整闭集/AK/CK/H为N/A，RU可报68.129913/50.283022%。源五RX主AK83.481481%–92.777778%、RU53.809524%–79.206349%；源三日AK89.555556%–90.166667%、RU65.619048%–73.523810%，不能称未见RX确认。NV三类RU70.571429/61.333333/73.904762%与目标四未知分开报告。

主source known12通过；source42有6交叉组失败：RX3的high/mid/mid_urban，RX4的high_urban，RX8的low_suburban/low_urban。已存下一common-tail候选令RX8原F0−AK达2.018519 pp被拒，本轮不再扫描。source总体损伤1.155556 pp；clean/star原F0−AK达24.695244/18.939712 pp，CK34.733272/41.166133%低于90%，RU67.296960/50.548954%低于clean95%/star90%。学生clean/star闭集本身较原F0低3.432286/2.468775 pp，不能用学生基准降低保护要求。

| 条件 | E0 predicted AUROC/OSCR | OLD_PCD predicted | ERM predicted | 主META predicted |
|---|---|---|---|---|
| source | 94.734250/87.937558 | 94.730673/87.940433 | 94.934060/88.335437 | 94.906499/88.368534 |
| clean | 53.862200/33.089825 | 54.364758/33.693180 | 49.224888/28.374682 | 49.374335/28.215853 |
| star | 51.242710/26.543484 | 50.997190/27.001261 | 46.735427/23.092658 | 47.177506/23.161420 |

曲线只描述固定margin，未选新运行点；逐真实类Wilson95%区间已保留，例如星地20-15 AK3/150，区间约.682475%–5.714668%。完整TX/RX/day/scene/day×scene/42组与配对见评分artifact；缺类保持N/A，不以present-only宏值认证。主AK未胜ERM/OLD，RU也未保持E0/ERM全部控制，source_meta_success/scientific_success/development_gate_passed均False，不按pooled或较弱控制换主。

## 冻结、成本与执行边界

两新学生冻结后各仅用源L/OE拟合known90/OE32和normalizer，共2套新源密度；再用V27000校准4组六τ24新值，旧E0/PCD24值精确复用，合计48 FP64/384B、单主六τ48B。网络/密度先V、全部τ先NV/Q冻结，全8×3预测后独立CPU连接NV/clean/star三truth各一次，Utruth0；旧12个overall与fresh控制逐值一致。新学生自己的closed路由自己的密度/τ，原F0同G闭集单存。

新增真实signal为父r03源缓存12434＋学生训练426240（base170496、virtual255744）＋冻结评估126120＝564794。评估源两臂85168＋future40952，Q共享F0/G各20476；训练真实F0/G/PCD0，L6300教师logits为缓存。合成训练F0/r03/PCD/G为4/2/2/4行、新学生10行；预测F0/G各2、新学生4行另计。主部署335681＋G266951＝602632参数，wrapper667729、虚拟参数和source/U行bank不部署；原F0 E50/11100、r03/2220、旧PCD444/U576、G E20/4940和缓存成本继承。

实际train/predict/CPUscore耗时394.032707/54.733757/24.364951 s，pipeline473.733402 s，三阶段return0、终态ANALYZED_DEVELOPMENT；非B1或受控加速证据。8条观测独立捕获driver1978836、train1978870、predict1983751的CWD/完整argv，训练/预测CUDA5、UUID `ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331`；训练374/1258 MiB、预测580 MiB均为离散样本非峰值。CPU1984444只有pipeline/gpuNone、CUDAempty实现及完成产物，未独立捕获live。终态owned空、foreign0。

完整四stdout185行/9188B，warnings/tracebacks/错误标记0；训练和预测日志均有独立增长，score只捕到终态41B，不能伪称其live增长。888条steps全部解析，不以stdout稀疏进度代替曲线。真实CP首烟测PASS、一次独立P0/P1 PASS；12唯一测试分为train5＋9sub（同case另1＋3定点不增unique）、eval3＋20sub、launch4＋17sub，9AST/实际reference接口通过。初期模块/fixture RED及定点修复日志保留，不说首轮全绿；eval输出1409条pytest警告与生产warnings0分开。

私有分析器对model identities键及零成本控制条目的两处schema消费修正后FULL_ARTIFACT_ANALYSIS PASS，没有修改生产实验代码/产物或重评分，不是实际run失败。单包853文件/data0，SHA `ef2d73742fb86feecafab977518b64209e74c9460af822053a08254e9f0a3569`，一次本地/远端比较、compile/runtime VERIFIED。证据见[训练完成](../local_artifacts/pamf_phase3_source_meta_20261031_r01/train/complete.json)、[训练smoke](../local_artifacts/pamf_phase3_source_meta_20261031_r01/train/smoke.json)、[预测完成](../local_artifacts/pamf_phase3_source_meta_20261031_r01/predict/complete.json)、[模型冻结](../local_artifacts/pamf_phase3_source_meta_20261031_r01/predict/models/complete.json)、[校准](../local_artifacts/pamf_phase3_source_meta_20261031_r01/predict/calibration/complete.json)、[流水线](../local_artifacts/pamf_phase3_source_meta_20261031_r01/pipeline_state.json)、[观测](../local_artifacts/pamf_phase3_source_meta_20261031_r01/observations.jsonl)、[测试](../local_artifacts/pamf_phase3_source_meta_20261031_r01/test_evidence.json)、[审查](../local_artifacts/pamf_phase3_source_meta_20261031_r01/review.json)与[发布](../local_artifacts/pamf_phase3_source_meta_20261031_r01/release_verification.json)。

本轮只否定当前固定RF二阶源RX留一方案的晋级资格，不推广为所有MLDG或源元学习无效，也不将目标退化归因单一层位/裁剪/接收机。目标Q不回流选择α、β、阈值或种子；原PCD与DURA保持，完整目标ACTIVE，无新增目标UDA、发现或注册。
