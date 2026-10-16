# additive As：固定 seed02 的独立源域确认

状态：ANALYZED／VERIFIED。三组 E20 和全部七源视图已完成独立复算。FAIL_FIXED_ADDITIVE_AS_CLOSED_AFTER_CONFIRMATION。

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


## 实际发布与启动读回

执行commit 05984a1229ebac7025a95081dc5fae4108276b28 已推送，远端OID与本地HEAD独立比对一致。归档单次本远端SHA比对、remote compile VERIFIED；匹配真实CP noquery smoke PASS。dispatcher和三worker的PID/CWD/argv/CUDA环境、实际GPU进程以及日志增长独立核实VERIFIED。三行真实seed2026092702，20/13/7，sourceRX1 fold，无baseline/teacher父权重；resolved为8L前向、保留base raw.75/ordinary.25及extra raw0/raw.25/As.25，updates4440。

- D00-RX1-s2026092702：GPU0，观察完整step315/4440，E2/20；stdout11753字节。仅为运行进度，无最终识别结论。

- D_raw-RX1-s2026092702：GPU1，观察完整step372/4440，E2/20；stdout11771字节。仅为运行进度，无最终识别结论。

- D_As-RX1-s2026092702：GPU2，观察完整step214/4440，E1/20；stdout6580字节。仅为运行进度，无最终识别结论。

仅修改本轮本地报告和状态，不改已发布训练release，不重复启动同run。完整联合目标仍未完成。


远端runtime.json的confirmation_seed02已独立读回；三row实际native与resolved model_seed均2026092702，initialization均scratch_only且checkpoint_sources/ancestors为空、无target_contact。实际F构造配置seed02的本地state验证与已提交runtime接线一致，未将首轮smoke权重用于训练。


## 已完成行的独立源读出：D_raw-RX1-s2026092702

此行已完成E20/4440更新、七源视图及8820固定预测的完整独立复算，进程自然结束。矩阵其他行仍运行，尚无完整As比较，不据此改变训练或选择配方。

| 场景 | Accuracy % | Macro-F1 % |
|---|---:|---:|
| clean | 81.190476 | 81.080657 |
| practical_high | 37.698413 | 31.713206 |
| practical_mid | 37.142857 | 31.601767 |
| practical_low_suburban | 34.920635 | 29.868069 |
| practical_high_urban | 38.492063 | 33.810567 |
| practical_mid_urban | 30.158730 | 26.469637 |
| practical_low_urban | 27.380952 | 24.317261 |

实际L NN 4546560，extra L NN 1136640，含原U/EMA/V/held的实际NN 5823318。第一ordinary平滑CE 0.3171610871145317，共同base完整目标 0.3372762059574729；二者口径不同。未访问target或拟合任何头。


## 已完成行的独立源读出：D00-RX1-s2026092702

此行已完成E20/4440更新、七源视图及8820固定预测的完整独立复算，进程自然结束。矩阵其他行仍运行，尚无完整As比较，不据此改变训练或选择配方。

| 场景 | Accuracy % | Macro-F1 % |
|---|---:|---:|
| clean | 82.619048 | 82.627195 |
| practical_high | 37.380952 | 31.501339 |
| practical_mid | 37.380952 | 32.055336 |
| practical_low_suburban | 35.555556 | 30.498361 |
| practical_high_urban | 37.857143 | 32.985774 |
| practical_mid_urban | 31.031746 | 27.294329 |
| practical_low_urban | 27.380952 | 24.317713 |

实际L NN 4546560，extra L NN 1136640，含原U/EMA/V/held的实际NN 5823318。第一ordinary平滑CE 0.31106840960077337，共同base完整目标 0.3340985238686338；二者口径不同。未访问target或拟合任何头。


## seed02 两个控制组的完整匹配比较

两行均已自然退出并完成全部七源视图，完整 8880 步与 17640 个硬预测已独立复算。L／U 物理训练序列完全相同，实际 native 参数仅 candidate_id／output_dir 不同。每行实际 NN 为 5823318，L NN 为 4546560（其中额外前向 1136640），4440 次更新，无 As／额外 teacher 调用。

| 场景 | D_raw−D00 accuracy pp | 修复 | 新错 |
|---|---:|---:|---:|
| clean | -1.428571 | 20 | 38 |
| practical_high | 0.317460 | 30 | 26 |
| practical_mid | -0.238095 | 24 | 27 |
| practical_low_suburban | -0.634921 | 26 | 34 |
| practical_high_urban | 0.634921 | 31 | 23 |
| practical_mid_urban | -0.873016 | 19 | 30 |
| practical_low_urban | 0.000000 | 30 | 30 |

D_raw 的额外 raw pair 使用 CE＋原 RDrop，且总 pair 系数从 1 增至 1.25；这是额外监督和损失尺度控制，不能把结果归因为单独 CE 或 RDrop。此 seed 的 clean 下降 1.428571 个百分点、low urban 无差异，未形成 clean／low urban 联合提升。As 行仍运行，原固定晋级条件不变；本比较没有候选晋级或目标访问。详见 controls_comparison.json。

## seed02 完整结果

全部三 worker 和 dispatcher 自然结束，无技术失败。完整读取 13320 步 native/view/Sinc 修正记录、3×20 轮、全部 L/U 物理序列与 26460 个固定硬预测，三组物理训练序列相同。以下仅为同一个源 RX1-held 契约的 seed02，不是正式 target 或完整三种子结果。

| 场景 | D00 % | D_raw % | D_As % | As−D00 pp | As−D_raw pp |
|---|---:|---:|---:|---:|---:|
| clean | 82.619048 | 81.190476 | 74.365079 | -8.253968 | -6.825397 |
| practical_high | 37.380952 | 37.698413 | 57.698413 | +20.317460 | +20.000000 |
| practical_mid | 37.380952 | 37.142857 | 53.968254 | +16.587302 | +16.825397 |
| practical_low_suburban | 35.555556 | 34.920635 | 53.095238 | +17.539683 | +18.174603 |
| practical_high_urban | 37.857143 | 38.492063 | 54.920635 | +17.063492 | +16.428571 |
| practical_mid_urban | 31.031746 | 30.158730 | 42.301587 | +11.269841 | +12.142857 |
| practical_low_urban | 27.380952 | 27.380952 | 36.111111 | +8.730159 | +8.730159 |

| 对照 | clean F1 差 pp | low urban 修复／新错 | urban 均值差 pp | seen V 差 pp |
|---|---:|---|---:|---:|
| D_As−D00 | -8.892667 | 199/89 | +12.354497 | +0.055556 |
| D_As−D_raw | -7.346130 | 202/92 | +12.433862 | -0.013889 |

| 原固定条件 | 结果 |
|---|---|
| D00_clean | FAIL |
| D00_clean_macroF1 | FAIL |
| D00_low_urban | PASS |
| D00_urban_mean | PASS |
| D00_practical_protection | PASS |
| D00_seen_V | PASS |
| D_raw_clean | FAIL |
| D_raw_clean_macroF1 | FAIL |
| D_raw_low_urban | PASS |
| D_raw_urban_mean | PASS |
| D_raw_practical_protection | PASS |
| D_raw_seen_V | PASS |
| healthy | PASS |

| row | 第一 ordinary 平滑 CE | 共同 base 完整目标 | 未加权 extra 目标 | 总 pair 系数 | 实际 NN 包 | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|
| D00 | 0.31106841 | 0.33409852 | 0.10483400 | 1.0 | 5823318 | 1556.230 |
| D_raw | 0.31716109 | 0.33727621 | 0.10136575 | 1.25 | 5823318 | 1432.415 |
| D_As | 0.31783310 | 0.34827843 | 0.57095288 | 1.25 | 5823318 | 2259.272 |

各目标为最后 222 步均值，native CE 不是完整训练目标。硬预测缺少 logits，runtime 读出 CE 未独立复算。每组 4440 更新、4546560 L NN，其中 extra 1136640；每组 568320 次 ordinary，只有 D_As 另生成 568320 次完整 practical，无额外 U／teacher As，tracked BN 和额外 BN 更新实际均为 0。新局部 profile 检查 24 个负例，0 IQ／0 NN／0 更新；原机制 3858 合成 NN／3 合成更新复用未重跑。

| D_As 场景 | 绝对 accuracy % | 与自身 clean 的差距 pp |
|---|---:|---:|
| practical_high | 57.698413 | 16.666667 |
| practical_mid | 53.968254 | 20.396825 |
| practical_low_suburban | 53.095238 | 21.269841 |

全部 TX／day 分层、accuracy／Macro-F1、混淆矩阵和相对两个对照修复／新错均保留于 source_analysis.json。不得用总均值掩盖类别负收益。

第二种子未通过原规则，关闭固定 additive As 候选扩展。不做 seed03、换 RX、系数、场景概率、阈值或额外预算救跑；与首轮正结果共同保留。其他健康任务不变。

本轮实测总 NN 17469956（包含 dispatcher 真实 CP 的合成 smoke 2），60 model epochs／13320 更新。首轮不作训练父权重，累计成本在两种子报告中分别汇总。E20 为用户指定的压缩探索预算，历史 clean 基准已暴露，本轮仅源域判断，不声称新的独立盲测。尚无有效新增 Ag、完整因子矩阵及交互项、正式 target clean＋六 practical 或跨种子联合成功。

完整训练曲线：[complete_training_curves.png](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_additive_As_confirm20_20261016/final/complete_training_curves.png)。

七源视图与自身 clean 差距：[held_source_results.png](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_additive_As_confirm20_20261016/final/held_source_results.png)。

全部 TX／day 分层：[source_strata.png](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_additive_As_confirm20_20261016/final/source_strata.png)。


## 同源折两种子合并结果

FAIL_FIXED_ADDITIVE_AS_CLOSED_AFTER_CONFIRMATION。两种子均值、样本标准差、每种子的原条件、全部 TX／day 代价、suburban 差距和累计成本见 [两种子确认报告](two_seed_source_confirmation.md) 与 [完整聚合](two_seed_source_confirmation.json)。n=2／同一 RX1-held 源折，联合目标仍未完成。


## 确认失败的具体含义

第二种子相对 D00 的 clean −8.253968pp、clean Macro-F1 −8.892667pp；相对 D_raw 的 clean −6.825397pp、Macro-F1 −7.346130pp，均违反原保护条件。urban 均值及 low urban 对两个控制均提高，六 practical、seen V 与共同 base 健康条件通过，但这不能补偿 clean 失败。

绝对 raw 监督系数 .75 保留这一实现事实，仍不足以保护跨 RX clean；不再将它当作充分条件，也不能据此断言旧失败只由监督稀释造成。此固定候选关闭扩展，首轮正结果完整保留。没有 seed03、换 RX、改系数或延长 E20 救跑。其他 owner 的健康实验不变。

第二种子 D_As 的 suburban 自身 clean 差距为 16.666667／20.396825／21.269841pp，仍远超探索 2pp；这些差距变小也包含 clean 下降 8.253968pp 的贡献，不能声称通过保护 clean 达到接近。其 clean 三个 day 均退化，TX 2 的 clean 相对 D00 下降 36.190476pp；首轮与第二轮受损 TX／场景并不完全相同，不能把均值当作各身份稳定获益。

[固定预测错误转移](error_transitions.md) 及 [全部转移聚合](error_transitions.json) 只解释同一源物理样本的描述性变化。完整 20 轮训练曲线、七视图绝对结果及全部 TX／day 图已生成并逐张查看，未新增前向。

累计六模型 120 model epochs／26640 更新，实测 NN 34939912。学生 train／eval 为 29657568／2851212；teacher train 为 0，teacher eval 仪表值 2378220，扣除继承计数 12 后为 2378208。另 held 52920 和 dispatcher smoke 4。旧本地工程 3858 合成 NN／3 合成更新另计，确认 profile 新检查 0 IQ／0 NN／0 更新，未重复旧工程测试。
