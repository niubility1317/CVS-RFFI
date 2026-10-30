# PAMF 独立信号级开放分支：固定单候选

Run `20261030-phase3-star-signal-open-s2026092701-r03`，唯一新arm `E_SIGNAL`。唯一主readout为`signal_known_support`。匹配冻结前端控制直接复用已完成crossset-metric的`S_SEP_TOPO`；不重训或按结果挑选控制。完整Phase3目标保持，不把source筛查当跨未见RX、无标签适应或匿名注册成功。

## 要检验的改变

上一轮只在冻结原F0输出后训练102688参数残差网络。本轮保持crossset KL＋OE软结构、L/OE样本随机流、2220更新和known-support主读出，改变开放损失的梯度范围：从最终特征后的MLP推进至原PAMF的独立信号特征分支。它从同一个原F0严格初始化，保留时频、PA、参考和三视图SIG-TIME；不改变原F0默认六类身份路径。

这与历史representation/physical-open不同：不更新原部署分类器，不做logit KD、相位增益双视图、P0/P1 VICReg、成员关系头或另训身份头。它检验训练位置，而不是新的未知标签或多个损失超参数组合。信号分支有效性尚未验证。

## 数据与固定星地观测

使用既有star-representation缓存根`/home/szu2070436088/2510044040/CV-SincNet/runs/20261029-phase3-star-representation-s2026092701-r01/cache`。L6300、OE6134、V27000、novel3150各自独立role目录，`restored.npy`为FP32[N,2,256]的P1。只使用P1值；role loader可以映射P0文件但不读取P0值，不重抽信道、不运行G、不再次RMS或EQ。

训练只打开L/OE。L旧六设备各1050、每TX×RX×day70，真旧TX可见；OE为21辅助设备，算法只见outside-six角色和公共RX/day，真实TX/K隐藏。源RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23均已见。

V旧六设备各4500、每TX×RX×day300，仅在全部NN和密度统计冻结后用于known42/common-tail六predicted-class阈值校准。novel为19-15/19-16/7-1各1050，只在全部阈值固定后预测；所有本row预测完整后独立truth-last。四角色互斥42584条既有物理记录，无新增物理数据。novel有历史开发曝光，不是新RX/TX盲测。

全部沿原G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、郊区/城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001。P0为received星地IQ，P1为同一received经固定DURA v1输出，没有输出RMS。原Us56700不新读，target U/Q、正式RX24-18不读，历史query不转训练。

原F0为Tri-SIG-TIME SSL E50/11100/232993/default6head，sourcecontract392005与DURA v1祖先、未变VALIDATED_ONCE角色匹配结论直接复用。OE teacher、L漂移参考及sampler metadata读取已完成crossset-metric code`fb016284e6c1f4715da4c690e2333efce89103b5`的原F0派生feature cache；该cache原数据来自protected7dd原F0，不含学习后的S_SEP_TOPO特征。参考模型/阈值/预测为该run `rows/S_SEP_TOPO/predict`，只作冻结控制，不用于初始化新MLP或优化signal分支。

## 网络、可微路径与优化器

先调用现有`source_open_embedding_train.make_trainable_model`核对原source祖先、重建真正可微的PAMF并严格加载原F0。禁止复制冻结部署reader后训练，其Sinc冻结路径不保梯度。复制该实际`SequenceReference`完整特征提取器并保留其原features方法，仅删除六类W[6,160]，得到232033参数的独立信号分支。原六类身份W没有进入新分支，也不创建新身份头。 原return_aux仍计算随后被丢弃的分类logits，独立copy仅把该内部classify输出槽显式置零，保留原components/joint及特征前向；这些零logits不作为身份预测，不修改原F0分类器，不携带旧Wbuffer。

新分支同次P1 forward捕获原raw fused160和原四贡献（time-frequency、tanh(gain)×PA、tanh(response_gain)×reference、signature）。参考29维的NoExplicitCFO坐标27仍为0。四贡献各FP64单位化、拼接整体单位化再FP32为c_theta，raw fused FP64单位化再FP32为u_theta；与冻结特征预处理保持一致。新102688参数MLP与前轮同seed2701、640→128GELU→160、最后层精确0。`z=unit(u_theta+g(c_theta))`，此处禁止detach学生u/c；OE teacher为固定原F0 u的stopgrad，不能让teacher随student更新。

新信号分支、MLP均可训练，全部BN/Dropout eval、运行buffers不更新，Sinc保持CENTER_25K可微坐标。参数232033＋102688＝334721；原身份识别器仍232993，合计活跃567714（加既有G算术为834665）。不把headless分支冒称新六类分类器，不由参数量推定时延。

fresh AdamW base lr1e−4/wd1e−4/eps1e−8、固定2220步。Sinc使用现有CENTER_UNITS物理策略：scale25000、sample rate25000000、坐标lr倍率1000、eps除1000、坐标group decay0，原物理衰减和global physical clip5复用`f_optimizer.attach/clip_physical`。新headless分支不调用强制训练六类head的旧specification；新模块只建立该分支所需的同一物理policy，不改共享optimizer源代码。记录实际Sinc group lr/clip/更新计数，及首两步Sinc、浅/中层、PA/reference/signature、MLP梯度。其他参数只有常规AdamW设置。

原识别器仍E50/11100、native更新0。独立signal分支继承该特征祖先11100步，再新增2220步，其特征祖先累计13320，明确披露，不写成原F0未经历的续训或只报增量成本。新MLP独立2220步。无早停、V/NV选模、额外epoch、EMA或目标更新。

## 固定目标与采样

完整复用前轮Sampler，L RNG2026103017/OE RNG2026103018；每步L192、OE64，全部2220步须与冻结前端控制的保存索引一致。L六类×15域各2＋每类两轮转格，OE15域各4＋四轮转格，格内无放回、batch无重复。

crossset监督KL仅L anchor，T=.1；候选L/OE各0.5先验，各集合按跨RX且跨day的有效公共域等权/域内均分；positive仅L同旧TX、正域等权。两侧学生信号特征均参与反向。主另加权重1的OE frozen-u软邻域KL，同样跨RX且跨day/域均衡。没有OE–OE硬异类、真实未知身份/类别数、P0/P1对比或新augmentation。新学习的位置是唯一研究因子；native梯度和物理optimizer是执行该因子所需，不另扫候选。

## 冻结预测、评分与推进

launcher第一步对实际原F0 checkpoint重建出的可微模型执行现有两行synthetic G10 no-query smoke，PASS立即导出新分支并继续合成梯度probe，然后才读L/OE P1。烟测不更新optimizer，不算入2220；真实部署默认身份路径不被修改。训练终点保存完整headless特征分支/MLP状态与来源，eval严格重载真实新checkpoint、两行合成推理smoke后开始源拟合。

eval先映射L/OE全部P1到z160，拟合同族D160 known90/background32/25Lloydseed3019/covshrink.1，全部统计冻结后才打开V。固定四列为`frozen_known_support`（父col1原样复用）、`signal_known_support`（唯一主）、`frozen_density_ratio`（父col2原样复用）、`signal_density_ratio`（诊断）。两新读出使用同一次signal投影和完全共享的known8统计；原冻结控制的模型、τ与预测不重fit/校准。两新τ均按同一V known42规则独立冻结，所有τ固定后才打开novel IQ与父novel参考预测。

旧身份一律为同输入原F0默认head的既有argmax，V与baseline_closed[:,1]核对，novel默认预测按父参考复制；新分支不输出身份。每条source IQ与参考ID/metadata对齐。保存新z和冻结前端z以供独立CPU scorer从聚合状态重建四列raw/margin，不在scorer重载NN/拟合/调τ；全部预测完整后独立truth侧唯一读novel真实TX。主报宏A_K/R_U/H、C_K、同P1原F0闭集损失、AUROC/OSCR、完整TX/RX/day/scene/42风险组，主与冻结known控制、主与同signal ratio作配对。

唯一主signal_known_support满足known42及source整体宏R_U≥90%才另登记跨未见RX星地测试，不因诊断较好换主或晋级。source通过也不是最终目标。完整目标为跨RX星地宏C_K/R_U≥90%、A_K距同输入闭集≤2pp，继续争取95%拒识及无标签适应、未知发现和未来独立观测匿名注册；本轮A_N/H_OW=N/A。

N607唯一owner、独占不可覆盖root，单GPU train/eval（每卡新增本任务≤2、启动free≥4096MiB），独立CUDAempty CPU scorer一次1个。允许与其他用户共卡，不停止/修改/重启其任务。一次实际代码commit、聚焦协议负测/合成真实checkpoint贯通、唯一直接P0/P1审查、最小登记、单preflight/archive SHA比较/remotecompile、PID/CWD/argv/GPU/log增长及终态读回。固定训练后自动eval/score，不因科学负结果停健康任务；非法来源/角色、非有限、碰撞或无法产生合法预测保留、不自动重启，UNKNOWN先核原PID/root。

## 实际发布预登记

Actualcode `dc01a402582f93e452dbaa3ef59a9fbb44151818`，唯一owner `codex-phase3-signal-open-20261030`，唯一row矩阵在experiment.json。环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-signal-open-s2026092701-r03-release`，独占runroot `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-signal-open-s2026092701-r03`，train/eval/score输出rows/E_SIGNAL，日志logs。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_signal_open_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-signal-open-s2026092701-r03-release/runtime.json`。

本地focused验证：{"model_train": {"unique_tests": 8, "subtests": 7, "seconds": 6.43, "stdout": "local_artifacts/pamf_phase3_star_signal_open_20261030_r01/core_train_tests.log"}, "eval_score": {"unique_tests": 7, "subtests": 18, "initial_unaffected_passed": 3, "initial_unaffected_subtests": 8, "calibration_directory_fix_targeted_passed": 4, "targeted_subtests": 10, "targeted_seconds": 7.51, "legacy_trapz_deprecations": 872}, "launcher": {"tests": 7, "seconds": 3.09, "prior_unfinished_eval_syntax_import_failure_resolved": true}, "AST_files": 10, "mirror_bytewise_matches": 10, "original_candidate_unique_tests": 22, "r02_export_fix": {"targeted_unique_tests": 2, "regression_passed": 2, "regression_seconds": 5.26, "production_export_checkpoint_targeted_passed": 1, "checkpoint_seconds": 4.96, "overlapping_case_not_double_counted": true, "full_suite_rerun": false, "stdout": "local_artifacts/pamf_phase3_star_signal_open_20261030_r02/export_fix_tests.log"}, "r03_return_aux_fix": {"targeted_tests": 2, "seconds": 5.36, "actual_historical_forward_contract_reproduced": true, "features_and_non_W_gradients_exact": true, "original_identity_unchanged": true, "production_two_steps_and_strict_CP": true, "full_suite_rerun": false, "stdout": "local_artifacts/pamf_phase3_star_signal_open_20261030_r03/return_aux_fix_tests.log"}, "real_data_weights_truth_read_in_local_tests": 0}；唯一直接P0/P1审查PASS。单次N607preflight确认固定四角色P1文件/祖先与source元数据、原F0 checkpoint、冻结前端控制及实际剩余显存；IQ/私有truth实值读取0。尚未本轮实际训练/取得性能。

预期产物：actualcheckpoint syntheticG10烟测/梯度probe、2220完整loss/indices/梯度/physicaloptimizer日志与headless+MLP final.pt；严格重载smoke、known/background fit/四model/四τ/known风险/四列预测、独立CPUtruthlast；PID/CWD/argv/GPU/log-growth与终态。IQ/NN权重/原始日志/逐行预测/source feature bank不入Git。完整跨RX星地与无标签DA/发现/未来REG目标ACTIVE。


## 实际启动与独立观察

VERIFIED／RUNNING（所存snapshot时刻）。actualcodedc01a402582f93e452dbaa3ef59a9fbb44151818；单archive SHA5c79cd8db374185c17cabdc89655209eaea06c73d17179e586ca5ee055bfaef6比较、remotecompile/runtime PASS。所属进程实际PID/CWD/argv/CUDA及GPU/日志增长核实：[{"pid": 845456, "argv": "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_signal_open_launch driver --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-signal-open-s2026092701-r03-release/runtime.json ", "cwd": "/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-signal-open-s2026092701-r03-release", "CUDA_VISIBLE_DEVICES": ""}, {"pid": 845479, "argv": "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_signal_open_train --config /home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-signal-open-s2026092701-r03/configs/E_SIGNAL_train_config.json --device cuda:0 ", "cwd": "/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-signal-open-s2026092701-r03-release", "CUDA_VISIBLE_DEVICES": "4"}]；离散GPU ["845479, GPU-95d925a1-7218-ca7b-adef-164241f4f491, 656"]不是峰值。sourceL/OE固定P1训练，模型/阈值冻结后自动预测与独立truthlast。无target、无foreign停止/修改/重启，本时刻不声称性能或完成完整目标。


## 已完成实际结果

# PAMF 独立信号级开放分支：完整源侧结果

**执行VERIFIED，科学结论NO_PROMOTION。** 唯一主signal_known_support的A_K90.388889%、C_K98.388889%、R_U31.301587%、H46.500199%；相对匹配冻结前端的R_U23.015873%提高8.285714pp，H提高9.801909pp，A_K降低0.107407pp。同P1原F0闭集90.981481%，主损失0.592593pp，known42通过，但未知拒识仍未达到90%。不换诊断主、不晋级正式目标或扩seed。

Run `20261030-phase3-star-signal-open-s2026092701-r03`，actualcode `dc01a402582f93e452dbaa3ef59a9fbb44151818`。固定设计 `docs/PAMF_PHASE3_SIGNAL_OPEN_PLAN_20261030.md`，唯一E_SIGNAL矩阵在experiment.json；r01/r02技术失败另存，不当科学负结果。原Tri-SIG-TIME SSL/default6head与DURA v1.0默认不变，完整Phase3继续ACTIVE。

## 数据、跨域范围与标签

| 角色 | 物理记录 | 实际用途 |
| --- | --- | --- |
| source L | 6300，旧六TX各1050 | P1 IQ训练crossset KL、冻结后known统计 |
| source OE | 6134，21辅助设备 | P1 IQ作outside-six候选与软结构；真TX/K隐藏；冻结后background统计 |
| known V | 27000，旧六TX各4500 | 模型与全部统计冻结后，仅固定known42风险和新六阈值 |
| source novel | 3150，19-15/19-16/7-1各1050 | 全部τ固定后P1推理，全部预测完成后独立truth-last |
| 原Us | 56700祖先记录 | 本轮新读取0、真TX不解封 |
| target U/Q与正式RX24-18 | 本轮0 | 无目标适应/发现/注册或跨未见RX结果 |

四source角色物理互斥，共42584既有记录；训练曝光和重复forward不计新记录。源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23均已见。L/novel每TX×RX×day70，V300；OE不按隐藏设备均衡。source novel有历史开发曝光，不是新RX/TX盲测或未来注册Q。

全部使用既有固定G10MIX01/P1 restored FP32[N,2,256] IQ：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect，郊区/城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001。P0为叠加星地信道的received，P1为同一P0经固定DURA输出，没有第二次RMS。本轮不读P0/clean实值、不重抽信道、不运行G。

IQ来自20261029 star-representation固定role缓存；原F0 u/metadata teacher来自crossset-metric codefb016284e6c1f4715da4c690e2333efce89103b5原F0派生cache，原数据protected7dd；冻结前端控制精确复用该run S_SEP_TOPO完整model/τ/predictions。控制的学习后NN权重不用于新信号或MLP初始化。原F0 E50/11100/sourcecontract392005及DURA祖先与角色匹配核验沿用，不将VALIDATED_ONCE解释为任意权重通行。

## 实际网络、梯度与训练

先重建真正可微的原F0并严格加载原权重，实际checkpoint synthetic G10烟测PASS后，复制真实SequenceReference特征提取器，仅删六类W[6,160]，保留完整features/components/时频/PA/reference/三视图SIG-TIME。native return_aux仍计算之后被丢弃的logits，独立副本只把该内部classify输出槽显式置零；这些零值不输出为身份，不修改原F0，不保留旧Wbuffer。实际历史方法定义的回归已验证原/副本特征和非W梯度逐值一致。

信号分支232033参数＋zero-last MLP102688＝334721。MLP为640→128GELU→160、seed2701；同次forward提取raw fused160与四贡献，FP64规范单位化再FP32得到u_theta/c_theta，z=unit(u_theta+g(c_theta))。学生两侧没有detach；OE teacher为旧冻结F0的u，不随student改变。NoExplicitCFO reference29坐标27始终0；BN/Dropout eval，运行buffers不变。原身份一律原F0默认head argmax，新分支没有身份输出。

L/OE每步192/64，Sampler和RNG3017/3018、全部2220步与冻结前端控制一致。L六类×15域各2＋每类两轮转格，OE15域各4＋四轮转格，同batch无重复。Lanchor跨RX且跨day，L/OE候选集合各0.5先验、公共域均衡，positive仅L同旧TX；T.1 crossset KL对两侧反向，再加权重1的OE frozen-u软邻域KL。没有OE–OE硬身份、未知TX/K、P0/P1 VICReg、logit KD、augmentation或EMA。

fresh AdamW lr1e−4/wd1e−4/eps1e−8，Sinc CENTER_25K采用既有CENTER_UNITS物理策略，scale25000/sample rate25000000、坐标lr0.1/eps1e−11/group decay0，原物理衰减与global physical clip5。全部2220 optimizer/Sinc step/physical clip调用一一对应，48步preclip物理norm>5；不把原始坐标模块梯度与Hz物理全局norm混为同一单位。原F0 native更新0、仍E50/11100；复制信号特征祖先11100+新增2220=13320，新MLP2220，分别披露。

| trace量 | 全2220均值 | 首222均值 | 末222均值 | 最小 | 最大 |
| --- | --- | --- | --- | --- | --- |
| loss | 0.567839841 | 0.900396093 | 0.385140704 | 0.234901920 | 1.670074701 |
| crossset_metric | 0.508338280 | 0.849452748 | 0.327585445 | 0.198779300 | 1.651134133 |
| OE_KL | 0.059501562 | 0.050943345 | 0.057555260 | -0.000000017 | 0.154919282 |
| gradient_norm_before_physical_clip | 2.530540128 | 1.884687735 | 3.752497541 | 0.854851961 | 6.762005806 |

| 实际step | 模块 | L2梯度 |
| --- | --- | --- |
| 1 | sinc | 0.010766686 |
| 1 | shallow | 1.166096704 |
| 1 | middle | 0.949786040 |
| 1 | time_frequency | 0.549797104 |
| 1 | PA | 0.054295282 |
| 1 | reference | 2.561023963 |
| 1 | signature | 0.415494207 |
| 1 | hidden | 0.000000000 |
| 1 | last | 0.975875670 |
| 2 | sinc | 0.020588392 |
| 2 | shallow | 0.781878932 |
| 2 | middle | 0.508006484 |
| 2 | time_frequency | 0.544059014 |
| 2 | PA | 0.052847363 |
| 2 | reference | 2.135864516 |
| 2 | signature | 0.388869228 |
| 2 | hidden | 0.002639145 |
| 2 | last | 0.712497393 |

只有首两步保存了模块梯度/VJP；Sinc、浅中层、时频、PA、参考、signature和末层在这两步非零，hidden第1步0/第2步非零符合zero-last初始化。后2218步只保存全局物理norm并执行finite检查，不能宣称已逐步观测每个模块非零。没有记录L/OE训练feature drift，不补造漂移结论。全部2220步连续有限、L/OE索引逐步与父控制精确相同，独立metadata重放/192和64配额/unique/batch无重复均通过；实际unique L6300/OE6134。OE_KL末段未趋零，不由loss下降宣称开放结构收敛。

## 冻结预测与独立主指标

真实新checkpoint严格重载与两条synthetic signal smoke后，全部L/OE P1映射到z160，再拟合同族D160 Gaussian known90/background32/25Lloydseed3019/covshrink.1。新known8与ratio11共享known字段和同一次signal投影，25Lloyd215.964945→191.020093、全有限单调、32mode无空。全部统计冻结后才V，两个新六阈值按固定known42/common-tail独立校准；原两控制不重fit/重校准。所有τ固定后才novel IQ/父novel参考，独立CPU scorer从保存features和聚合统计重建四列raw/margin后才唯一连接trueTX。

A_K为拒绝计错的旧六TX宏平均正确接收，C_K为宏平均接收，R_U为三未知设备宏平均拒识，H=2A_KR_U/(A_K+R_U)。原默认身份及同P1闭集24565/27000=90.981481%逐条不变；AUROC/OSCR只作固定分数诊断，不用于选τ。

| readout | A_K% | C_K% | R_U% | H% | 原F0损失pp | AUROC% | OSCR% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_known_support | 90.496296 | 98.685185 | 23.015873 | 36.698290 | 0.485185 | 89.584163 | 83.137141 |
| signal_known_support | 90.388889 | 98.388889 | 31.301587 | 46.500199 | 0.592593 | 93.115866 | 86.509364 |
| frozen_density_ratio | 90.418519 | 98.325926 | 32.952381 | 48.301593 | 0.562963 | 91.787741 | 84.893828 |
| signal_density_ratio | 90.462963 | 98.448148 | 31.111111 | 46.299399 | 0.518519 | 93.827297 | 87.177584 |

| 主−参考 | 正确gain | 正确loss | 正确net | 接收net | 未知拒对gain | 未知拒对loss | 未知拒对net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_known_support | 59 | 88 | -29 | -80 | 439 | 178 | 261 |
| signal_density_ratio | 18 | 38 | -20 | -16 | 193 | 187 | 6 |

主已知正确接收24405、接收26565、拒绝435，拒绝者160个原身份正确、275个原身份错误；接收中2160身份错误。未知拒对986/3150=31.301587%，距90%的2835条还少1849条。A_K变化仅由拒绝集合产生，不能称原分类器准确率改善。四列known42均通过、整体未知90均失败，唯一主NO_PROMOTION。

## 设备、日期和完整42分组

| 未知TX | 分母 | 拒对 | R_U% |
| --- | --- | --- | --- |
| 19-15 | 1050 | 481 | 45.809524 |
| 19-16 | 1050 | 199 | 18.952381 |
| 7-1 | 1050 | 306 | 29.142857 |

三个未知设备R_U分别45.809524%、18.952381%、29.142857%，相对冻结known控制净拒对+167/−14/+108；不是三类一致改善。

| 旧TX | 分母 | 闭集% | A_K% | C_K% | 额外失去正确 |
| --- | --- | --- | --- | --- | --- |
| 14-10 | 4500 | 89.044444 | 88.377778 | 98.111111 | 30 |
| 14-7 | 4500 | 80.333333 | 79.733333 | 98.066667 | 27 |
| 20-15 | 4500 | 95.911111 | 95.422222 | 98.866667 | 22 |
| 20-19 | 4500 | 85.466667 | 84.644444 | 98.200000 | 37 |
| 6-15 | 4500 | 96.333333 | 95.844444 | 98.000000 | 22 |
| 8-20 | 4500 | 98.800000 | 98.311111 | 99.088889 | 22 |

RX/day为已见source，scene为同MIX的固定切片，不是重抽六份观测。全部required TX覆盖完整，没有剔除困难设备或把缺类填0。四列完整分组均保存在原独立score与私有analysis；下表列唯一主。

| 组 | known n | novel n | A_K% | C_K% | R_U% | H% | 原F0损失pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 27000 | 3150 | 90.388889 | 98.388889 | 31.301587 | 46.500199 | 0.592593 |
| rx/1 | 5400 | 630 | 92.870370 | 99.611111 | 30.000000 | 45.350414 | 0.222222 |
| rx/3 | 5400 | 630 | 83.962963 | 95.796296 | 30.952381 | 45.230750 | 1.185185 |
| rx/4 | 5400 | 630 | 91.203704 | 99.000000 | 30.476190 | 45.686125 | 0.425926 |
| rx/6 | 5400 | 630 | 92.703704 | 99.685185 | 37.142857 | 53.036144 | 0.129630 |
| rx/8 | 5400 | 630 | 91.203704 | 97.851852 | 27.936508 | 42.771671 | 1.000000 |
| scene/practical_high | 1103 | 133 | 97.183854 | 98.810782 | 51.458632 | 67.288274 | 0.729026 |
| scene/practical_mid | 1983 | 256 | 97.143533 | 98.971555 | 31.387167 | 47.444856 | 0.299130 |
| scene/practical_low_suburban | 10304 | 1203 | 95.069417 | 98.655925 | 37.174266 | 53.448841 | 0.535501 |
| scene/practical_high_urban | 1171 | 130 | 96.114331 | 98.161706 | 43.993559 | 60.359363 | 0.838072 |
| scene/practical_mid_urban | 1979 | 236 | 91.204948 | 98.660928 | 30.053419 | 45.209590 | 0.508883 |
| scene/practical_low_urban | 10460 | 1192 | 82.941793 | 97.938625 | 21.824640 | 34.556389 | 0.680766 |
| rxscene/1/practical_high | 202 | 20 | 98.569157 | 99.549550 | 49.206349 | 65.643197 | 0.000000 |
| rxscene/1/practical_mid | 437 | 46 | 98.334486 | 99.765258 | 41.189203 | 58.059232 | 0.234742 |
| rxscene/1/practical_low_suburban | 2042 | 245 | 97.162454 | 99.755419 | 36.936275 | 53.525028 | 0.096618 |
| rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 99.645390 | 30.952381 | 47.193051 | 0.000000 |
| rxscene/1/practical_mid_urban | 401 | 44 | 93.700821 | 99.754902 | 38.978668 | 55.054979 | 0.245098 |
| rxscene/1/practical_low_urban | 2082 | 253 | 86.178021 | 99.424590 | 18.780700 | 30.840382 | 0.386217 |
| rxscene/3/practical_high | 208 | 25 | 91.990324 | 97.374372 | 47.222222 | 62.407989 | 1.812619 |
| rxscene/3/practical_mid | 407 | 54 | 92.829171 | 95.815906 | 19.956140 | 32.850235 | 1.342269 |
| rxscene/3/practical_low_suburban | 2007 | 239 | 90.870183 | 96.571747 | 39.767998 | 55.324182 | 1.025625 |
| rxscene/3/practical_high_urban | 241 | 29 | 90.580353 | 94.833185 | 50.925926 | 65.196942 | 1.248732 |
| rxscene/3/practical_mid_urban | 388 | 44 | 86.556417 | 96.756673 | 16.666667 | 27.951247 | 1.579297 |
| rxscene/3/practical_low_urban | 2149 | 239 | 73.735547 | 94.905542 | 23.456568 | 35.591012 | 1.143758 |
| rxscene/4/practical_high | 213 | 26 | 96.744493 | 98.254986 | 58.796296 | 73.141173 | 1.103989 |
| rxscene/4/practical_mid | 403 | 48 | 98.149588 | 100.000000 | 28.888889 | 44.638957 | 0.000000 |
| rxscene/4/practical_low_suburban | 2049 | 248 | 95.721102 | 98.882792 | 35.908699 | 52.225564 | 0.333471 |
| rxscene/4/practical_high_urban | 238 | 30 | 96.603251 | 97.925685 | 51.111111 | 66.851989 | 1.677489 |
| rxscene/4/practical_mid_urban | 392 | 40 | 92.506614 | 99.722222 | 34.814815 | 50.590080 | 0.000000 |
| rxscene/4/practical_low_urban | 2105 | 238 | 84.083300 | 98.991199 | 18.216967 | 29.946016 | 0.462598 |
| rxscene/6/practical_high | 240 | 24 | 99.593496 | 99.593496 | 59.629630 | 74.596240 | 0.406504 |
| rxscene/6/practical_mid | 365 | 47 | 98.543783 | 100.000000 | 50.695971 | 66.949625 | 0.000000 |
| rxscene/6/practical_low_suburban | 2104 | 240 | 96.697624 | 99.807870 | 41.965542 | 58.529865 | 0.047893 |
| rxscene/6/practical_high_urban | 233 | 27 | 98.375307 | 99.583333 | 42.407407 | 59.266391 | 0.416667 |
| rxscene/6/practical_mid_urban | 408 | 57 | 92.960665 | 99.712644 | 35.247209 | 51.113927 | 0.000000 |
| rxscene/6/practical_low_urban | 2050 | 235 | 85.904376 | 99.517420 | 26.872765 | 40.938936 | 0.188062 |
| rxscene/8/practical_high | 240 | 38 | 98.016317 | 99.164578 | 30.687831 | 46.741433 | 0.396825 |
| rxscene/8/practical_mid | 371 | 61 | 97.533251 | 99.152542 | 28.510379 | 44.122974 | 0.000000 |
| rxscene/8/practical_low_suburban | 2102 | 231 | 94.810734 | 98.211798 | 30.802715 | 46.498652 | 1.158301 |
| rxscene/8/practical_high_urban | 223 | 22 | 95.484380 | 98.757702 | 35.353535 | 51.601409 | 0.814947 |
| rxscene/8/practical_mid_urban | 390 | 51 | 90.429642 | 97.276033 | 30.134159 | 45.204633 | 0.731848 |
| rxscene/8/practical_low_urban | 2074 | 227 | 85.195777 | 97.044078 | 21.791817 | 34.706282 | 1.193975 |
| day/1 | 9000 | 1050 | 90.566667 | 98.733333 | 37.714286 | 53.252756 | 0.466667 |
| day/2 | 9000 | 1050 | 90.333333 | 98.522222 | 31.238095 | 46.422771 | 0.566667 |
| day/3 | 9000 | 1050 | 90.266667 | 97.911111 | 24.952381 | 39.097151 | 0.744444 |

主下一更严格共同尾率会绑定RX3/practical_high（208条），接收率仍96.986775%，但原F0至open损失2.200216pp超过2；signalratio下一绑定RX4/practical_high_urban损失2.074315pp。该证据来自既存V校准轨迹，不是重新扫描未知阈值。当前总体主损失只有0.592593pp，但局部分组已限制继续提高同一共同尾率，不能仅从C_K98.39%推断存在安全拒识空间。

## 成本与执行证据

| 读出 | 聚合状态B |
| --- | --- |
| frozen_known_support | 322868 |
| signal_known_support | 322868 |
| frozen_density_ratio | 364340 |
| signal_density_ratio | 364340 |

四套条件τ192B。主部署可只带signal known聚合322868B及对应六τ，ratio与冻结前端为诊断状态，不携带source逐行bank。独立signal334721＋原F0232993＝567714活跃参数，加既有G266951的算术总数834665；不能由参数数推定B1速度。原特征祖先11100、新signal2220/MLP2220和既有G/缓存成本不抹去。

r03真实signal forward610904行＝训练568320（L426240/OE142080）＋eval42584；signal合成8+2，原F0真实0/合成2、G0。pipeline实际421.430972s，train380.661747/eval22.344126/score18.257967s；这不是端到端B1时延。没有测峰值显存。4份完整stdout31行，无warning/error/nonfinite。

actualcode `dc01a402582f93e452dbaa3ef59a9fbb44151818`；534文件/data0 archive SHA5c79cd8db374185c17cabdc89655209eaea06c73d17179e586ca5ee055bfaef6单次比较/remotecompile/runtime PASS，原候选22项focused测试与两个明确缺陷的定向回归/各自一次定点复审保留；未反复跑全套。独立live观察driver845456/train845479 CUDA4/656MiB及四次log增长，eval851281/GPU7与CPUscore851440只按jobs/artifacts记录，不补称live。终态全部train/eval/score COMPLETE、live/GPU空、foreign停止/修改/重启0。

r01在旧x3 exporter类型guard ValueError退出；r02在合成return_aux对被删除W访问AttributeError退出；均未到真实IQ/optimizer/预测/评分，各自root/完整log/状态与Git记录保留。两失败没有落盘smoke.json，烟测forward成本记UNKNOWN，不能填0或假定2后合并；只读CPU类诊断的IQ/forward/optimizer0有独立记录。r03分别修实际SequenceReference导出与副本内部discardedlogits依赖，保持科学数据/预算/损失/控制不变。

## 研究判断与后续

相同目标进入可训练信号层后，主R_U与H改善，AUROC89.584163→93.115866%，支持“训练位置影响开放分数”的有限结论。但一个未知设备下降，固定ratio控制R_U32.952381%仍高于本主，且四列均远低90%；不能宣称已解决未知泛化，也不能按较高控制替换主。

后续先用合法V证据检查：当前42组共同尾率校准的局部保护，是否让总体宏A_K的2pp预算只用到0.592593pp；比较校准方式必须另预登记固定规则、只用V确定阈值，保持用户总体及主场景约束，不用NV寻找新τ。本轮不改变现有阈值或历史判定，也不据单个未知设备结果调整loss/频带/训练预算。更强表征与无标签目标证据仍需独立检验。

完整目标继续ACTIVE：跨未见RX且postPhase1全为星地观测；宏C_K/R_U≥90%、A_K距同输入同路径原闭集≤2pp，总体和预登记主场景验收，主报A_K/R_U/H；继续提高拒识和接收。零目标TX标签DA、未知发现、未来独立观测匿名注册（A_N≥80%、旧类下降≤2pp、后续未知拒识保护）均未由本轮完成。A_N/H_OW=N/A，正式RX24-18新访问0。
