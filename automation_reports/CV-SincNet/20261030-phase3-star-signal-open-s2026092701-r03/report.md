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
