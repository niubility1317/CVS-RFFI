# PAMF P0开放表征与P1身份路径：匹配训练单候选

Run `20261031-phase3-star-raw-open-s2026092701-r01`，唯一新arm `E_RAW`，唯一主 `P0_known12`。保留原F0/default6身份路径P1不变，研究开放分支从同一星地received P0学习是否优于已完成的P1开放分支。P0不是clean；不能由已有拒识结果预先断言DURA损坏未知身份。

## 固定研究因子与矩阵

旧P1信号分支来自signal-open r03：同原F0初始化、headless232033＋zero-last MLP102688，2220更新、L192/OE64、crossset KL＋OE软结构权1。新P0分支使用相同初始化、结构、预算、采样和目标，改变开放分支及其冻结原F0 OE teacher的输入路径为P0。两路径teacher均取该路径下原冻结F0的u，不将P1 teacher用于P0蒸馏。新分支不能从已训练的P1分支warmstart。

四列固定为 `P1_known12`（精确复用macro12主控制）、`P1_ratio12`（父固定P1 ratio仅新known12校准）、`P0_known12`（唯一主）、`P0_ratio12`（诊断）。四列均按相同known12规则校准／复用，旧身份一律原F0 P1默认head；不使用P0较低闭集准确率作为保护基线。P1两分数来自同一已冻结P1 NN，P0两分数来自同一新NN；path×readout对照固定，不按novel结果换主。

## 数据、观测与来源

已有star-representation缓存根`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache`，四role为L6300/OE6134/V27000/novel3150。`received.npy`为FP32[N,2,256]的P0，`restored.npy`是相同ID顺序的P1。新训练/新投影明确只读received值，不把P0重命名为restored，不读clean/P1实值，不重抽信道、额外RMS或EQ，不运行G。旧P1身份与控制复用已完成的P1预测，不是把clean当对照。

L旧六TX各1050、每TX×RX×day70，真实旧标签可见；OE来自21辅助设备，算法只见outside-six成员角色与公共RX/day，无真实OE TX/K。source RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19；日期2021-03-08/15/23均已见。四角色物理互斥共42584既有记录，没有新增物理观测。novel为19-15/19-16/7-1各1050，有历史开发曝光；不能称新RX/TX盲测。

训练只打开L/OE；V旧六各4500（每TX×RX×day300）只在NN/全部密度冻结后用于known风险阈值；novel只在全部新旧τ冻结后预测，独立scorer最后连接trueTX。原Us56700、target U/Q、正式RX24-18不读，不回收历史query。

全部沿固定G10MIX01：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、郊区/城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001，输入一次unitRMS、输出不再RMS。P1是同received经DURA v1.0输出。新P0分支和旧P1控制共享同一物理记录／信道实现，只改变合法计算路径。

原F0为Tri-SIG-TIME SSL E50/11100/232993，sourcecontract392005与DURA祖先、既有VALIDATED_ONCE角色结论复用。P1训练控制config/checkpoint来源固定signal-open r03 actualcode `dc01a402582f93e452dbaa3ef59a9fbb44151818`；其原L/OE steps.jsonl用于逐步匹配采样。P1 known12控制取macro12-risk r01 actualcode `26fbecff1df7e5476bcc5f459d8367c6fa49fd36` 的signal_known12列3；P1 ratio固定state/raw取r03 signal_density_ratio列3，同ID/模型来源核对。没有新目标训练、其它父模型或更高控制替代。

## 教师与真正可训练的分支

首先通过现有make_trainable_model严格重建合法原F0，实际checkpoint两行synthetic G10无query smoke PASS；随后先从该可微reader导出独立学生，保留真实SequenceReference/features/components和全部信号模块，仅删W并旁路被丢弃的内部logits，沿用已验证HeadlessSignal/SignalOpen。原身份模型未修改。先完成学生8行合成梯度probe再读取真实L/OE，避免复制冻结部署reader或把teacher缓存误带成断梯度路径。

学生导出后，原F0 eval/冻结，分批对OE6134原received计算teacher u160（raw fused FP64单位化→FP32），参数/buffers保持；teacher stopgrad、无V/NV/目标，保存私有训练teacher及其原source身份、P0路径、opaque IDs/公共metadata。不得复用P1 teacher或把student更新后的输出当teacher。原F0新增真实teacher前向6134单列；teacher逐行bank不写入部署checkpoint或Git。

学生结构与旧P1一致：信号232033＋MLP102688＝334721参数，MLP640→128GELU→160、seed2701、末层精确0。每次P0 forward捕获raw fused160及原四贡献，FP64规范单位化后FP32为u_theta/c_theta，z=unit(u_theta+g(c_theta))，学生u/c不detach。NoExplicitCFO坐标27保持0。全部BN/Dropout eval、buffers不更新，原F0/default6身份仍独立P1。

新信号祖先11100＋新增2220＝13320、新MLP2220与原识别器native0/E50分别报告。fresh AdamW base lr1e−4/wd1e−4/eps1e−8，原CENTER_UNITS scale25000/multiplier1000/Sinc lr.1/eps1e−11/group decay0及物理衰减/global clip5不变；记录实际2220 Sinc steps/clipcalls。原F0＋单个P0开放分支活跃567714，加既有G算术834665；不由参数量推定B1时延。

## 完全匹配的目标、采样与预算

复用Sampler L RNG2026103017/OE RNG2026103018；L192＝六类×15域各2＋每类两轮转格，OE64＝15域各4＋四轮转格，batch无重复。全部2220步L/OE索引与父P1 r03保存trace逐值一致，不重新采样或改变配额。

T=.1 crossset KL仅Lanchor，候选须跨RX且跨day，L/OE各0.5先验且各自公共有效域等权／域内均分；positive仅L同旧TX。另加OE frozen-original-P0-u软邻域KL权1，teacher stopgrad，同跨RX/day与域均衡。没有OE–OE硬同/异类、未知K、P0/P1一致性、VICReg、logit KD、EMA、新augmentation或目标状态。固定2220终点，无V/NV早停、追加epoch或根据其它row结果中断。

首两步记录学生L/OE VJP、Sinc/浅中层/四贡献/MLP梯度，后续保存全局物理norm、实际索引、loss和optimizer计数/有限性；未记录的逐层漂移不补造。所有输入值路径明确P0，不通过P1字段别名改变旧helper语义。

## 终点冻结、四列预测与评分

新checkpoint保存完整学生state/source身份/P0 teacher来源，不携teacher逐行bank；eval严格重载及两行synthetic signal smoke后，先对L/OE received投影，拟合同族D160 known90/background32/25Lloydseed3019/covshrink.1。新P0 known8和ratio11完全共享known统计。旧P1 known8/ratio11从两个已完成控制root精确复用，所有统计在V前冻结。

V读取后一次P0投影，计算P0 known/ratio两个raw，连同父固定P1 ratio raw分别执行相同known12 common-tail校准；P1 known12旧τ/model/raw/margin逐值复用。known12保overall＋5RX＋6scene的宏C_K90和原F0 P1闭集−open≤2pp；30RX×scene完整报告，不能把known12通过写成known42通过。真实已知y只作源风险，部署按原F0 P1预测类路由六τ。

四套τ全部固定后才打开novel received及两个父novel预测；保存新P0 unit160，并精确保留父P1 signal_features/defaulthead/ID/publicmetadata。独立CUDAempty CPU scorer从四聚合state、两unit160和τ重建全部raw/margin，验证父known12控制和P1 ratio已冻结来源、P0统计共享及路径身份后才唯一读novel trueTX，不重载NN、重fit或调τ。主与P1_known12、主与P0_ratio12配对，报告完整TX/RX/day/scene、12硬组和30诊断、AUROC/OSCR、真实新teacher/学生前向/训练/参数/祖先成本。

唯一主P0_known12通过known12及source整体宏R_U≥90%才另登记跨未见RX星地验证，控制不能替主。source是历史开发池早筛，不是最终总体/每主景C_K90/R_U90/A_K损失2pp证书。完整零目标TX标签DA、未知发现和未来匿名注册（A_N80/旧类损失2pp/后续未知保护）继续，A_N/H_OW本轮N/A。

N607单GPU teacher/train随后单GPU eval、一个独立CPU scorer；每次启动actualfree≥4096MiB、每卡新增本任务≤2，允许与其他owner共卡，不停止/修改/重启其进程。一次实际代码commit、聚焦路径/teacher冻结/采样/真实合成CP/真值隔离负测、一次直接P0/P1审查、最小登记、不可覆盖root/单preflight/archive比较/remotecompile、PID/CWD/argv/GPU/log增长及终态。非法来源/角色、非有限、碰撞或无法合法prediction保留，不评分partial、不自动重启；UNKNOWN先核原PID/root，科学低分不停止健康任务。

## 实际发布预登记

Actualcode `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`，唯一owner `codex-phase3-raw-open-20261031`；四列单row矩阵在experiment.json。环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-star-raw-open-s2026092701-r01-release`，不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-star-raw-open-s2026092701-r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_raw_open_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-star-raw-open-s2026092701-r01-release/runtime.json`。原始F0新raw学生初始化，P1控制仅预测／统计复用，不用P1训练后权重warmstart。

Focused验证：{"train": {"tests": 4, "subtests": 4, "seconds": 26.31, "stdout": "local_artifacts/pamf_phase3_star_raw_open_20261031_r01/train_tests.log"}, "eval_score": {"tests": 7, "subtests": 12, "seconds": 9.84, "stdout": "local_artifacts/pamf_phase3_star_raw_open_20261031_r01/eval_score_tests.log", "existing_numpy_trapz_deprecations": 1304, "local_pytest_cache_permission_warning": 1}, "launcher": {"tests": 8, "seconds": 3.04, "stdout_evidence": "Observed one local tool result: 8 passed in 3.04s; no rerun for logging"}, "test_items_total": 19, "additional_subtests": 16, "AST_files": 8, "mirror_matches": 9, "preview_runtime_validate": true, "registry_fields_validate": true, "real_data_weights_truth_read_in_local_tests": 0}；一次直接P0/P1审查PASS。单次N607preflight核既有角色handles、两父预测／CP完成与来源元数据、文件存在、实际free显存，IQ/private truth实值0。

首步实际原F0 synthetic2 smoke，学生probe8，原F0 P0 OEteacher6134后固定2220训练，随后严格重载CP synthetic2及固定四列source预测、独立CPUtruthlast。保存PID/CWD/argv/GPU/log-growth、teacher状态、完整2220索引/finite/physicaloptimizer和终态；rawIQ/权重/逐行teacher/原始日志/逐行预测不入Git。当前未启动／无新性能，sourceNV历史曝光和已见RX边界保留。


## 实际启动与独立观察

VERIFIED／RUNNING（保存snapshot时刻），actualcode `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`；单archive SHA `8cdaa414e8ba4e1a383a67c8181f1f51446fe10c44df01174c1c7e6c7829c9d4` 比较、remotecompile/runtime PASS。所属进程PID/CWD/argv/CUDA：[{"pid": 922922, "argv": "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_raw_open_launch driver --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-star-raw-open-s2026092701-r01-release/runtime.json ", "cwd": "/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-star-raw-open-s2026092701-r01-release", "CUDA_VISIBLE_DEVICES": ""}, {"pid": 923148, "argv": "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_raw_open_train --config /home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-star-raw-open-s2026092701-r01/configs/E_RAW_train_config.json --device cuda:0 ", "cwd": "/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-star-raw-open-s2026092701-r01-release", "CUDA_VISIBLE_DEVICES": "7"}]，离散GPU ["923148, GPU-9c4eb614-ca19-a390-aaf4-801f8980b6a9, 656"]，日志实际增长；不是峰值。新训练只有合法L/OE P0，原P1身份固定，训练／全部密度／τ依次freeze后自动source预测和独立truthlast。未访问目标、未停止／修改／重启其他owner，本时刻不声称新性能或完整目标完成。


# PAMF P0开放分支与P1身份分工：匹配训练结果

**执行 VERIFIED，科学结论 NO_PROMOTION。** 唯一主 P0_known12：宏 A_K 90.048148%、C_K 98.503704%、R_U 3.936508%、H-score 7.543258%；相同原F0/P1闭集损失 0.933333 个百分点。相较固定P1_known12，R_U变化 -42.952381 个百分点、A_K变化 0.100000 个百分点。hard12通过=True，原42组全部通过=False，source早筛通过=False。

Run `20261031-phase3-star-raw-open-s2026092701-r01`，actualcode `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`。本轮新训练完整信号级开放分支；已知身份保持原F0的P1默认六类head。P0是星地received，P1是同一received经过冻结DURA v1.0，两者都不是clean。实验检验开放路径的整体差异，不预设哪条更优。

## 数据、权限与评估边界

| 角色 | 数量与划分 | 本轮使用 |
| --- | --- | --- |
| source L | 6300；旧六TX各1050，每TX×RX×day70 | 合法真实旧标签，P0信号级训练和已知统计拟合 |
| source OE | 6134；来自21辅助设备 | 只见outside-six成员角色/公共RX/day；原F0 P0 teacher、新分支训练和background统计，不见真实TX/K |
| source V | 27000；旧六TX各4500，每TX×RX×day300 | NN/全部统计冻结后仅校准风险阈值，同时报告该V上的已知风险 |
| source novel/NV | 3150；19-15、19-16、7-1各1050，每TX×RX×day70 | 所有阈值固定后才预测，独立CPU scorer最后读取trueTX |
| 原Us | 56700祖先记录 | 本轮不读取 |
| target U/Q与正式RX24-18 | 本轮0 | 未适应、发现或未来注册 |

四角色物理记录互斥，共42584条既有观测，没有新物理记录。source RX索引1/3/4/6/8分别为1-19、18-2、19-2、2-19、3-19；三天为2021-03-08、03-15、03-23，接收机和日期均已见。旧六TX为14-10、14-7、20-15、20-19、6-15、8-20。OE设备真实身份及类数没有进入算法。

**该结果是源侧开发验证。** 已知风险在用于阈值校准的同一个V上衡量；NV未参与本候选训练/阈值，但已有历史开发曝光。因此既不是独立已知测试，也不是新接收机或未知设备盲测，更不能代替用户要求的跨未见接收机星地评估。

固定G10MIX01信道：600km、25MHz、2.45GHz、residual/post_sync/noEQ、refSNR30dB、reflect；suburban/urban各0.5及原六面积权重，receiver2027/scene2026103001/channel2026104001。输入一次unitRMS，输出不再RMS。新分支只读received.npy，不读restored实值/P1 teacher/clean，不重抽信道、不运行G。旧P1身份和控制来自相同ID顺序的已完成预测。

## 实际优化设计

原F0为Tri-SIG-TIME SSL E50/11100、232993参数，sourcecontract392005与原DURA合法来源保持。先严格重建可微原F0并执行实际checkpoint synthetic2无query smoke，再导出独立学生，然后冻结原teacher；学生synthetic8梯度probe在真实L/OE读取前通过。复用已验证HeadlessSignal，保留真实SequenceReference和四贡献信号数学，仅删除W并旁路被丢弃内部logits。

新分支232033信号参数＋102688零末层MLP（640→128GELU→160），共334721。z=unit(u_theta+g(c_theta))，原四贡献及fused特征均可微；NoExplicitCFO坐标27保持0，BN/Dropout eval、buffers固定。原F0对OE6134的P0重新计算单位化u作为soft-topology teacher，冻结且状态不变；无P1 teacher复用、更新学生充当teacher、OE硬身份或未知K。逐行teacher仅保留私有训练目录，不进入checkpoint/部署/Git。

与已完成P1 signal-open r03匹配：同原F0独立初始化、2220步、L192/OE64、RNG3017/3018，全部2220步索引逐值一致；T=.1的L跨RX且跨day crossset KL＋OE frozen-u软结构KL权1，L/OE候选各.5并按公共域均衡。无P0P1一致性、VICReg、KD、EMA或新augmentation。教师与学生的输入路径一同从P1变为P0，不能单独把差异归因于恢复器。

fresh AdamW lr1e−4/wd1e−4/eps1e−8；Sinc沿原CENTER_UNITS scale25000/multiplier1000/lr.1/eps1e−11/物理衰减与globalclip5。全部2220实际Sinc steps/clipcalls、索引、loss和全局norm有限。首两步Sinc梯度范数0.016991/0.008324，L/OE全部行VJP均非零；零末层使MLP hidden首步0，第二步0.001577，符合初始化。首两步另保存VJP/模块梯度；未声称逐层梯度在所有后续步均被记录，未补造特征漂移。固定终点，无V/NV早停。

## 四列、阈值与真值隔离

P1_known12精确复用macro12 r01 col3模型/τ/raw/margin；P1_ratio12取signal-open r03 col3固定模型/raw，仅新C12校准。P0_known12为唯一主，P0_ratio12为同新NN诊断。各路径known8与ratio11共享相同known统计：D160、known90/background32、25 Lloyd/seed3019、covshrink.1；先冻结所有NN与统计，才读V。

known12对overall、5RX、6scene施加真实旧TX宏C_K≥90%及原F0 P1 closed−open≤2pp。六阈值按原F0预测类路由，使用exactFraction complete-tie common-tail最大可行q。其余30RX×scene全部报告为诊断，不把12通过写成42通过。全部四τ固定后才打开novel received和任一父novel预测；scorer先重建四raw/margin与控制，再唯一读未知trueTX，不重新fit/校准。

| 方法 | A_K% | C_K% | R_U% | H% | 原P1闭集损失pp | known12 | known42 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1_known12 | 89.948148 | 97.522222 | 46.888889 | 61.643672 | 1.033333 | True | False |
| P1_ratio12 | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | True | False |
| P0_known12 | 90.048148 | 98.503704 | 3.936508 | 7.543258 | 0.933333 | True | False |
| P0_ratio12 | 89.940741 | 98.185185 | 4.761905 | 9.044927 | 1.040741 | True | False |

| 方法 | AUROC% | OSCR% |
| --- | --- | --- |
| P1_known12 | 94.493082 | 87.505278 |
| P1_ratio12 | 94.985255 | 87.944738 |
| P0_known12 | 78.865888 | 74.733706 |
| P0_ratio12 | 78.480956 | 74.519424 |

所有列身份预测及原P1闭集基线完全相同，原闭集90.981481%。A_K为含误拒计错的旧六TX宏正确接收，C_K为宏接收，R_U为未知三TX宏拒识，H=2A_KR_U/(A_K+R_U)。

| 方法 | q | q% | 六预测类拒绝前缀 | 下一q失败硬组 |
| --- | --- | --- | --- | --- |
| P1_known12 | 56/2253 | 2.485575 | [105, 105, 111, 123, 112, 113] | rx/3 |
| P1_ratio12 | 72/2479 | 2.904397 | [123, 123, 129, 144, 130, 132] | rx/3 |
| P0_known12 | 64/4241 | 1.509078 | [64, 64, 67, 74, 67, 68] | scene/practical_low_urban |
| P0_ratio12 | 41/2253 | 1.819796 | [77, 77, 81, 90, 82, 83] | scene/practical_low_urban |

## 配对与完整分组

| 主−参考 | 已知正确gain | 已知正确loss | 已知正确net | 已知接受net | 未知拒对gain | 未知拒对loss | 未知拒对net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1_known12 | 255 | 228 | 27 | 265 | 71 | 1424 | -1353 |
| P0_ratio12 | 84 | 55 | 29 | 86 | 17 | 43 | -26 |

主计数：{"known_rejected": 404, "rejected_original_correct": 252, "rejected_original_wrong": 152, "accepted_wrong": 2283, "original_closed_correct": 24565, "known_correct_accepted": 24313, "unknown_rejected": 124}。两条路径的开放分支独立训练、所有阈值均已冻结，配对差异不用于本轮再调参。

| 旧TX | n | 原闭集% | A_K% | C_K% | 失去原正确数 |
| --- | --- | --- | --- | --- | --- |
| 14-10 | 4500 | 89.044444 | 88.200000 | 98.355556 | 38 |
| 14-7 | 4500 | 80.333333 | 79.666667 | 98.288889 | 30 |
| 20-15 | 4500 | 95.911111 | 94.755556 | 98.644444 | 52 |
| 20-19 | 4500 | 85.466667 | 84.777778 | 98.733333 | 31 |
| 6-15 | 4500 | 96.333333 | 95.177778 | 98.200000 | 52 |
| 8-20 | 4500 | 98.800000 | 97.711111 | 98.800000 | 49 |

| 未知TX | n | 拒对 | R_U% |
| --- | --- | --- | --- |
| 19-15 | 1050 | 28 | 2.666667 |
| 19-16 | 1050 | 28 | 2.666667 |
| 7-1 | 1050 | 68 | 6.476190 |

| 组 | known n | novel n | A_K% | C_K% | R_U% | H% | 原闭集损失pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 27000 | 3150 | 90.048148 | 98.503704 | 3.936508 | 7.543258 | 0.933333 |
| rx/1 | 5400 | 630 | 92.129630 | 98.666667 | 4.920635 | 9.342299 | 0.962963 |
| rx/3 | 5400 | 630 | 84.370370 | 97.851852 | 4.603175 | 8.730045 | 0.777778 |
| rx/4 | 5400 | 630 | 90.574074 | 98.629630 | 2.857143 | 5.539542 | 1.055556 |
| rx/6 | 5400 | 630 | 91.851852 | 98.666667 | 1.587302 | 3.120674 | 0.981481 |
| rx/8 | 5400 | 630 | 91.314815 | 98.703704 | 5.714286 | 10.755514 | 0.888889 |
| scene/practical_high | 1103 | 133 | 97.912879 | 100.000000 | 2.773792 | 5.394756 | 0.000000 |
| scene/practical_mid | 1983 | 256 | 97.300618 | 99.641379 | 2.315156 | 4.522699 | 0.142045 |
| scene/practical_low_suburban | 10304 | 1203 | 95.313304 | 99.405409 | 3.401858 | 6.569251 | 0.291614 |
| scene/practical_high_urban | 1171 | 130 | 96.691217 | 99.138942 | 0.724638 | 1.438495 | 0.261185 |
| scene/practical_mid_urban | 1979 | 236 | 91.356788 | 99.333282 | 3.344017 | 6.451870 | 0.357043 |
| scene/practical_low_urban | 10460 | 1192 | 81.628994 | 97.012704 | 5.504180 | 10.312966 | 1.993565 |
| rxscene/1/practical_high | 202 | 20 | 98.569157 | 100.000000 | 4.761905 | 9.084915 | 0.000000 |
| rxscene/1/practical_mid | 437 | 46 | 98.569227 | 100.000000 | 3.030303 | 5.879843 | 0.000000 |
| rxscene/1/practical_low_suburban | 2042 | 245 | 97.016991 | 99.707869 | 3.529412 | 6.811042 | 0.242081 |
| rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 100.000000 | 3.333333 | 6.450127 | 0.000000 |
| rxscene/1/practical_mid_urban | 401 | 44 | 93.397791 | 99.451872 | 4.524887 | 8.631595 | 0.548128 |
| rxscene/1/practical_low_urban | 2082 | 253 | 84.391250 | 96.928276 | 6.300228 | 11.725118 | 2.172988 |
| rxscene/3/practical_high | 208 | 25 | 93.802944 | 100.000000 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/3/practical_mid | 407 | 54 | 93.338106 | 98.128631 | 3.837719 | 7.372316 | 0.833333 |
| rxscene/3/practical_low_suburban | 2007 | 239 | 91.556256 | 98.667791 | 3.971398 | 7.612588 | 0.339553 |
| rxscene/3/practical_high_urban | 241 | 29 | 91.829085 | 97.242148 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/3/practical_mid_urban | 388 | 44 | 87.843317 | 99.066577 | 0.000000 | 0.000000 | 0.292398 |
| rxscene/3/practical_low_urban | 2149 | 239 | 73.437104 | 96.646586 | 7.611552 | 13.793451 | 1.442202 |
| rxscene/4/practical_high | 213 | 26 | 97.848482 | 100.000000 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/4/practical_mid | 403 | 48 | 98.149588 | 100.000000 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/4/practical_low_suburban | 2049 | 248 | 95.864232 | 99.660587 | 2.517227 | 4.905640 | 0.190341 |
| rxscene/4/practical_high_urban | 238 | 30 | 97.883914 | 99.603175 | 0.000000 | 0.000000 | 0.396825 |
| rxscene/4/practical_mid_urban | 392 | 40 | 92.300852 | 99.794239 | 3.703704 | 7.121641 | 0.205761 |
| rxscene/4/practical_low_urban | 2105 | 238 | 82.134306 | 96.923247 | 4.168812 | 7.934880 | 2.411593 |
| rxscene/6/practical_high | 240 | 24 | 100.000000 | 100.000000 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/6/practical_mid | 365 | 47 | 98.543783 | 100.000000 | 2.380952 | 4.649565 | 0.000000 |
| rxscene/6/practical_low_suburban | 2104 | 240 | 96.508712 | 99.720679 | 1.821465 | 3.575449 | 0.236804 |
| rxscene/6/practical_high_urban | 233 | 27 | 98.341523 | 99.170762 | 0.000000 | 0.000000 | 0.450450 |
| rxscene/6/practical_mid_urban | 408 | 57 | 92.441828 | 98.994311 | 0.000000 | 0.000000 | 0.518838 |
| rxscene/6/practical_low_urban | 2050 | 235 | 83.898712 | 97.049299 | 1.882176 | 3.681755 | 2.193726 |
| rxscene/8/practical_high | 240 | 38 | 98.413143 | 100.000000 | 6.267806 | 11.785039 | 0.000000 |
| rxscene/8/practical_mid | 371 | 61 | 97.533251 | 100.000000 | 1.587302 | 3.123766 | 0.000000 |
| rxscene/8/practical_low_suburban | 2102 | 231 | 95.529064 | 99.270199 | 4.942894 | 9.399439 | 0.439971 |
| rxscene/8/practical_high_urban | 223 | 22 | 95.911730 | 99.612403 | 0.000000 | 0.000000 | 0.387597 |
| rxscene/8/practical_mid_urban | 390 | 51 | 90.892673 | 99.303832 | 7.843137 | 14.440226 | 0.268817 |
| rxscene/8/practical_low_urban | 2074 | 227 | 84.611238 | 97.540164 | 7.381381 | 13.578215 | 1.778513 |
| day/1 | 9000 | 1050 | 90.033333 | 98.455556 | 4.285714 | 8.181956 | 1.000000 |
| day/2 | 9000 | 1050 | 89.855556 | 98.455556 | 4.380952 | 8.354573 | 1.044444 |
| day/3 | 9000 | 1050 | 90.255556 | 98.600000 | 3.142857 | 6.074200 | 0.755556 |

主30交叉组超限／缺失明细：{"rxscene/1/practical_low_urban": {"known_count": 2082, "coverage": {"known_complete": true, "unknown_complete": true, "complete": true}, "C_K": 0.9692827564030649, "A_K": 0.8439124954247269, "original_closed_minus_new_open": 0.021729883327704203, "reasons": ["original_F0_loss_above_2pp"]}, "rxscene/4/practical_low_urban": {"known_count": 2105, "coverage": {"known_complete": true, "unknown_complete": true, "complete": true}, "C_K": 0.9692324694050445, "A_K": 0.8213430564885952, "original_closed_minus_new_open": 0.024115926746147043, "reasons": ["original_F0_loss_above_2pp"]}, "rxscene/6/practical_low_urban": {"known_count": 2050, "coverage": {"known_complete": true, "unknown_complete": true, "complete": true}, "C_K": 0.9704929867484561, "A_K": 0.8389871234517837, "original_closed_minus_new_open": 0.0219372575614748, "reasons": ["original_F0_loss_above_2pp"]}}。各类Wilson区间在独立scorer完整结果，不能当成跨独立RX的宏置信证据。

## 成本与执行证据

| 新增计算 | 真实处理行数 | 合成处理行数 |
| --- | --- | --- |
| 原F0/P0 OEteacher | 6134 | 2 |
| 新信号分支训练 | 568320 | 8 |
| 新信号分支评估 | 42584 | 2 |
| DURA | 0 | 0 |

真实新网络处理共617038行＝teacher6134＋signal训练568320＋signal评估42584，是重复曝光/不同网络处理数，不是新增物理样本或F0同成本推理次数。原识别器native新增0/E50，复制信号特征祖先11100＋新增2220＝13320，新MLP2220。F0＋单个新开放分支567714参数，加既有G算术834665；不据参数量推定时延。祖先P1分支、历史失败及恢复器成本保持，不计为本轮新增。

聚合统计存储：{"P0_known12": {"fields": 8, "bytes": 322868}, "P0_ratio12": {"fields": 11, "bytes": 364340}, "P1_known12": {"fields": 8, "bytes": 322868}, "P1_ratio12": {"fields": 11, "bytes": 364340}}；四套六FP64阈值192B，source逐行银行部署0。Pipeline 461.656523s；阶段{"train": 403.4138779640198, "eval": 29.72833514213562, "score": 28.340956687927246}，不是B1或部署吞吐测量。

Actualcode `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`，单552文件/data0 archive SHA `8cdaa414e8ba4e1a383a67c8181f1f51446fe10c44df01174c1c7e6c7829c9d4` 比较、remotecompile/runtime PASS；19项focused测试/16子测试与一次直接P0/P1审查PASS。完整2220 trace与全部四stdout读取；stdout统计[{"name": "driver.log", "lines": 1, "warnings": 0, "errors": 0, "nonfinite": 0}, {"name": "eval.log", "lines": 3, "warnings": 0, "errors": 0, "nonfinite": 0}, {"name": "score.log", "lines": 1, "warnings": 0, "errors": 0, "nonfinite": 0}, {"name": "train.log", "lines": 26, "warnings": 0, "errors": 0, "nonfinite": 0}]。本地测试既有np.trapz弃用及pytest缓存warning单列在verification，不隐去。

四raw由聚合状态及两路径unit160重建：{"known": {"max_abs_error_by_method": {"P1_known12": 2.2737367544323206e-13, "P1_ratio12": 3.410605131648481e-13, "P0_known12": 1.7053025658242404e-13, "P0_ratio12": 1.7053025658242404e-13}, "P1_known_raw_margin_exact": true, "P1_ratio_raw_exact": true, "parent_fields_exact": true}, "novel": {"max_abs_error_by_method": {"P1_known12": 3.410605131648481e-13, "P1_ratio12": 3.979039320256561e-13, "P0_known12": 1.7053025658242404e-13, "P0_ratio12": 2.2737367544323206e-13}, "P1_known_raw_margin_exact": true, "P1_ratio_raw_exact": true, "parent_fields_exact": true}}；原P1 known12 raw/margin、P1 ratio raw和其他publicmetadata/defaultidentity均精确一致，没有重跑NN、fit、τ扫描或scorer。

独立观察记录实际PID/CWD/argv/CUDA/GPU及日志增长，终态本run live/GPU均空；GPU为离散观察，未测峰值。其它owner停止/修改/重启0。训练/预测/评分所有stage COMPLETE，不覆盖、不重启原root。

## 结论与下一步

本轮不支持直接把开放分支迁到P0。主P0known12较P1known12少拒对1353个未知、只净增27个正确已知接收；宏R_U从46.888889%降到3.936508%，A_K只从89.948148%到90.048148%。P0ratio12同样只有4.761905%拒识；这不是换一种密度读出就已解决的问题。

P1ratio12诊断列A_K89.770370%、C_K97.107407%、R_U60.888889%、H72.561330%，较同P1known12的R_U高14个百分点、A_K低0.177778个百分点。它仍未达90%，也不是本轮唯一主，不能替代主晋级或追溯改主。该结果支持后续在P1上改进开放分离和拒识读出，而不能证明某个未实现的新loss会成功。

下一轮保持原P1身份保护，优先检验P1开放表征的训练目标与源OE使用方式；先核对已有条件支持、局部支持和表征消融，避免重复已失败路线。新目标、对照、阈值规则和预算须另行预登记。P0可以作为待研究的辅助证据，当前没有使用其替换主开放输入的证据；不追加这次训练预算、调本轮阈值或从NV真值构造身份关系。

本候选的预登记判断为 `NO_PROMOTION`，不以控制列替换唯一主。它仅决定source早筛；总体和预登记主场景的跨未见RX星地C_K90/R_U90/A_K较同输入原闭集损失≤2pp仍须独立验证。完整无标签域适应、未知类发现、未来匿名注册继续ACTIVE，A_N/H_OW本轮N/A。不得从这一次输入路径差异推断所有恢复器会去除设备指纹，也不按NV真值更改本轮阈值、loss或主方法。
