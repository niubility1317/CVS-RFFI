# additive As：固定 seed02 的独立源域确认

状态：PLANNED，尚未启动。本轮只检验同一候选的独立种子可复现性，不增加 IQ 算子、损失、U 规则或模型分支。

首轮 20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01 的三组 E20、26460 个预测和全部13320步已独立复算，原两对照源条件全通过。As 相对 D00 clean +3.333333pp、low urban +5.714286pp、urban均值 +9.708995pp；相对D_raw也通过。它仍有TX标签2/3在六practical的明显负收益，lowurban仅33.809524%，suburban自身clean差25.396825/28.809524/32.063492pp。首轮没有证明 A_g、跨seed或完整联合目标，也不能证明raw权重变化是旧normalized75/25失败的唯一原因。

唯一假设：配方完全不变，modelseed2026092702仍同时满足原来对D00和D_raw的源规则。此seed在本次确认登记前固定，不跳过旧其他配方曾出现负结果的seed，不按target选择。失败停止固定候选扩展，不做seed03、RX、系数、场景概率或额外预算救跑；结果与首轮共同报告，保留负确认。

## 模型、实际种子和数据契约

F＋跨RX同TX保持三路SIG_TIME、统一Sinc、joint160、NoExplicitCFO及既有必要内部补偿、默认cosine头随骨干正常联合训练。原ordinary两前向的time_down/t2.pool SupCon、temperature.2、weight.1×clip((epoch−5)/10)、同TX异RX正对等全部沿用。无新部署模块、头替换、冻结骨干后重训或诊断分类头。

三组scratchE20、每组4440更新，native args.seed和实际F model_config.model_seed均为2026092702。split392005、channel20261016101、receiver20261016102、eval20261016103保持；其余native参数只允许run_id/candidate_id/output_dir变化。模型scratch状态三组相同且不同于首轮seed01，ownEMA从自己的学生复制，无初始化/resume/teacher父权重。首轮D00 checkpoint只用于dispatcher的真实CP无query合成smoke，严格匹配源数据契约；它不进入训练初始化。parent_run_ids和训练checkpoint sources均为空。

仍为同一个sourceRX1-held契约：fitRX3/4/6/8×day1/2/3×6TX，L5040/U45360/V21600；held RX1原L1260只在final20冻结后源读出，U TX始终隐藏。不是fullL6300或L_train5580/L_dev720，无target访问或目标选择，不重新构建、哈希或重验已验证数据。

## 原样三组矩阵与损失

每个物理L的共同base=.50R(raw)+.25R(ordinary)+.25R(第三raw)+w SupCon。R是同一生成IQ的两次默认头CE均值＋原RDrop1双向KL，label_smoothing=.01。每个pair按物理样本均值，KL求类别和后取样本均值，不按新增视图数重新除共同base。

| row | 第四对 | 额外系数 | 总pair系数 | 相对raw/ordinary/As份额 | L前向/包 |
|---|---|---:|---:|---|---:|
| D00-RX1-s2026092702 | raw | 0 | 1 | .75/.25/0 | 8 |
| D_raw-RX1-s2026092702 | raw | .25 | 1.25 | .80/.20/0 | 8 |
| D_As-RX1-s2026092702 | 完整practical(raw) | .25 | 1.25 | .60/.20/.20 | 8 |

原raw绝对系数.75与ordinary.25均保留，新增项改变相对份额，如表明确披露；不能由权重推出clean保证。D00控制额外前向/dropout/RNG，D_raw控制重复监督和1.25总尺度，当前F trackedBN数0仍记录其计数。没有新对齐、clean-sat KL、门控或整个特征等同clean的要求。As只从raw生成一次完整practical residual，保留既有传播/噪声/接收质量/补偿依赖，全部L监督，无urban高置信过滤，不把ordinary通用AWGN/multipath再送进As，不改变正式received IQ/SNR/信道或删困难包。

每步一份原nativeordinary，DC/taper/循环移位/gain/phase/CFO/Wiener/multipath/AWGN/DAC/PA顺序、min(epoch,13)实际日程保持，PA normal开启但不宣称标签保持，DAC p=0、class_signature=false。原13L+7U、旧门控/ownEMA、AdamW lr.0002/weight_decay.0001、cosine和物理Sinc修正全部保持，无额外U/teacher As。

## 固定判断与输出

固定final20，不按V/held/target选epoch，每row完成即自动源clean＋六practical，每视图全部1260包，完成固定prediction后独立全量复算。D_As相对D00 clean accuracy/F1均≥0pp，lowurban≥+5pp、urban均值≥+3pp；相对D_raw clean accuracy/F1均≥−.5pp，lowurban≥+5pp、urban均值≥+3pp。相对两对照每个practical与seenV均≥−.5pp。每组4440更新/4546560LNN/extra1136640/SupCon3330有效步；最后222步共同base完整目标≤.5；As568320只在D_As、无As U/teacher、target/头拟合或非有限跳步。所有条件原样，不按首轮类别代价修改本轮门槛。

native final_L_CE为第一ordinary前向平滑CE，与完整base CE＋RDrop＋SupCon目标及extra目标分别报告。硬预测无logits，只独立复算accuracy/F1/CM/配对修复新错；runtime读出CE保留，不宣称独立重算CE。完整TX/day负面分层、三suburban绝对值和自身clean差距均报告，不只给总均值。

本轮追加60modelepochs/13320更新，每组4546560LNN、其中extra1136640，另实测原U学生/ownEMA/V/held8820/scratchsmoke，扣除EMA继承eval计数2；每组568320nativeordinary生成，D_As另568320完整practical生成。dispatcher真实CP smoke另2合成前向。首轮成本60modelepochs/13320更新/17469956真实NN包单独保留，不当本轮父权重或续训。两个seed完成后累计120modelepochs/26640更新，NN以实测汇总，不预造数值。新局部profile验证24负例、0IQ/0NN/0优化；原机制20负例/3858合成NN/3合成更新验证复用，不重复。无新数据验证或模型头训练。

E20为用户明确指定的压缩探索预算，历史clean基准已暴露；本轮source-only选择，不能声称新的独立盲测。source PASS只支持后续合法研究，不自动target或完整联合成功。A_g及统一B00/B10/B01/B11、交互项、正式target clean＋六practical、lowurban>60%与suburban接近自身clean均仍未完成。

## 发布与技术停止

唯一owner codex/root/F-CROSS_RX-additiveAs-confirm-E20-20261016。普通N607账号，dispatcher CPU2线程/loader0，原共享capacity lock，所有owner每GPU最多3进程且启动时≥12000MiBfree。GPU由实际空闲选择，启动后独立核实PID/CWD/argv/CUDA/GPU及log增长。不得停止、重启或热改其他健康任务。

CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_confirm_sourceRX1_e20_20261016_r01。dispatcher命令 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_additive_as20/runner.py；worker追加 --worker <row_id>。不可变runtime.json profile=confirmation_seed02在dispatcher/worker构造配置和模型前生效，旧seed01行及未知profile拒绝。实际代码身份以Git commit/release_commit/resolved为准，发布后回填。

新增profile接线做过一次独立P0/P1定向审查PASS，仅覆盖本次新增profile→真实seed/输出/拒绝路径，未完整复审原机制。一次N607 preflight、一份release单次本远端SHA对比、remote compile，dispatcher第一步真实CP noquery smoke PASS即继续。协议/角色/初始化/预算/输出碰撞/系统性非有限或无法合法prediction为技术失败：保存partial、取消pending、健康worker自然完成，无自动重试。科学负分不停止或改训练。

保留实际args/resolved/roles/init/frozen、私有CP、全stdout/20epoch/4440native-view-unit轨迹/physical序列/As覆盖/七视图prediction/result/completion或failure。IQ、权重、原始日志、逐包轨迹、机器私有状态不入Git；小型聚合及同row报告提交推送。
