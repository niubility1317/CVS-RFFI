# F＋跨 RX 同 TX：固定 75/25 practical 的独立 seed02 确认

- 当前状态：ANALYZED_SOURCE_ONLY；两组完整 E20 和七视图独立复算结束，固定 75/25 候选确认失败并关闭。正式测试及联合目标未完成。
- 唯一配置与逐行矩阵：[experiment.json](experiment.json)。
- 执行代码 commit：由本次 Git 提交及 release_commit.txt 确认，训练解析时保存。

## 假设与最小矩阵

已完成 run `20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01` 通过全部预登记源阈值，完整结果见其 source_analysis.json。此处只确认固定配方的收益能否在独立模型 seed2026092702 重现，不搜索比率或新机制，不使用目标分数。第一种子的 clean+3.095238pp、low urban+7.539683pp、urban均值+13.095238pp 是已有源证据，不作为本次结果。第一种子 TX2/TX3 的所有卫星场景下降、suburban 与 clean 的大差距原样保留。

| row | 同一 F＋CROSS_RX 底座 | raw RDrop 对 | 旧普通增强 RDrop 对 | S 槽 RDrop 对 |
|---|---|---:|---:|---|
| B00-RX1-s2026092702 | F 三路 SIG_TIME＋统一完整 Sinc＋默认 cos 头联合训练 | 50% | 25% | 25% raw |
| B01-RX1-s2026092702 | 完全相同 | 50% | 25% | 25% 完整 practical(raw) |

每物理 L 包为三对六次前向；每对均为平均 LS0.01 CE＋alpha1 对称 RDrop KL，同对使用同一份已生成 IQ，不重新抽样。总监督权重为 1，原始 raw 固定 50%；原 L（raw＋旧普通增强）占 B01 的 75%。B00 无新增 Ag/As，仍有共同旧增强。CROSS_RX 仅使用共有普通增强对的 time_down、t2.pool，同 TX 异物理 RX 正对，同 TX 同 RX 排除分母；温度0.2、原权重0.1×clip((epoch−5)/10,0,1)，不随 CE 份额缩小。新增 raw/satellite/U/teacher/eval 无对比目标。没有新部署结构、损失或分类头。

## 数据、初始化与 seed

固定 split392005、原6TX、day1/2/3。训练仅 RX3/4/6/8 的 L5040/U45360，U 标签隐藏；held RX1 原 L1260 只固定最终读出；已见 V21600 只健康诊断。不是完整 L6300 或 L5580/Ldev720。两个模型均从零 E20，固定13L＋7U、222step/epoch、4440更新；原 oldgate/ownEMA/cos 保持。

只有模型 seed 从2026092701变为2026092702（同时控制原 native init/dropout/sampler/ordinary augmentation 流）；数据划分不变，新 practical channel seed20261016101、receiver seed20261016102、源评估 seed20261016103 均不变。不可变 runtime.json 的 confirmation_seed02 profile 在 dispatcher 和每个 worker 的配置／模型载入前，绑定当前 design 与底层 F/timing design 的 model seed。原生参数独立比较只变 seed/run_id/candidate_id/output_dir。

不加载第一种子、E50、旧 F-only 或完整源权重作初始化。dispatcher 唯一真实 checkpoint smoke 复用已核实同四 RX 角色的自有 timing B00 fixedfinal20，只严格加载后跑两个合成包，0 源/目标 IQ、0 更新；PASS 后自动继续。每个新行仍有自身 scratch checkpoint smoke，默认头随骨干更新，head-fit=0。

## 完整 practical 与普通增强

As 仅作用原始 x_l，不作用旧增强后的 IQ；不重复普通 AWGN/multipath，不连续叠完整星地信道。既有完整 residual/post_sync、EQ关闭G=1、Fs25MHz/fc2.462GHz、reflect、质量适配/相位跟踪/IQ补偿保持，reference_snr_db30。六场景每包每步均匀一次抽样；RX 硬件 session 仅 RX/day/eq，不依 TX。全部合法 satellite L 样本监督，无高置信门槛，不丢困难样本。保留 NoExplicitCFO 的特征约束与内部必要补偿。

共同旧普通增强保留实际已运行配置（含 PA-normal、CFO；DAC0、class-signature关闭），这不证明 PA 是身份保持的新 Ag。每行保存 actual_native_args、ordinary_augmentation_actual、actual_channel 和输入变化，后续不据默认值猜测。新确认不改变任何正式测试 IQ、SNR 或信道。

## 源评估、晋级及放弃条件

每行第20轮固定最终 checkpoint 后，自动生成同一 held L1260 的 clean＋六 practical 默认头预测，共8820决策；两组复用同一命名物理信道。所有预登记源视图齐全后，独立核对全部预测与完整日志，报告 accuracy、macro-F1、混淆矩阵、修复／新增错误、TX/day分层和 suburban 与各自 clean 的绝对差。源 L 标签本来可见，无 target truth-side scorer，也无目标访问。

确认必须独立同时满足原条件：low urban≥+5pp、三个urban均值≥+3pp；clean、每个suburban、seenV均≥−0.5pp；两组完整有限4440更新、3409920L前向、3330非零对比调用及最终LCE≤0.5。任一不满足即不晋级；保留两个种子的结果，不改seed/RX/比例或选择性重跑。suburban差距2pp仍是探索目标，本次保护条件不代替联合目标。

即使通过，本次仍仅源确认，不能证明Ag、B10/B11交互项、跨目标RX收益或最终lowurban>60%。不自动启动完整源或目标预测。后续正式四因子仍需独立固定，冻结后按clean＋六practical、prediction先于truth收尾。

## 成本、命令、资源和产物

本次新增两模型各E20，合计40模型轮、8880更新，无续训。每模型L网络包前向3409920；原单对RDrop为1136640，新增2273280完全匹配。原U、ownEMA、V、smoke、源评估分别按实际计数。B01生成568320训练practical IQ，B00为0；每行源评估另生成7560。第一种子历史成本40模型轮/8880更新/9373358实际NN包前向单列，不隐去、不串接。记录每模型训练墙钟，不能把并行墙钟相加当经过时间。

唯一owner `codex/root/F-crossRX-satmix25-confirm-20261016`，普通N607账号。CWD `PROJECT/releases/cvs_F_crossRX_satmix25_confirm_source_e20_20261016_r01`，命令 `CVS-RFFI/bin/python -u experiments/cvs_crossrx_satmix20/runner.py`。新不可覆盖run root为 `PROJECT/runs/20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01`，所有row/log/config精确路径见experiment.json。CPU dispatcher；共享容量锁，每GPU最多3个所有owner进程、至少12000MiB空闲后启动。预计优先GPU0/1，实际按共享锁记录。不得修改、终止或重启健康旧任务。

技术停止仅限错误角色/初始化/输出、非有限、预算违例或无法合法预测；保留全部产物、停止新pending、健康worker继续，无自动重试。单次归档比对与远端编译后，启动独立核实PID/CWD/argv/GPU/log增长；观察超时先对原run只读核实，不重复提交。

实际产物包括：release_commit/runtime与启动记录、匹配真实CP smoke、每行scratch/final checkpoint（私有）、完整原生step/epoch与stdout（私有）、配置/初始化/物理序列/视图step/practical覆盖、七视图源预测（私有）、独立全日志／预测分析、三类图和汇总报告。训练完成不足以标ANALYZED；必须收齐声明视图与独立复算。仅报告汇总和代码/config入Git，数据、权重、预测、完整日志与本机状态保持私有。

## 本地验证

已通过8个固定配置负例及2个未知runtime/publish profile负例、两个arm原生参数差异检查、底层F与native seed传递及幂等profile绑定；0IQ、0模型前向、0更新。复用已验证的六前向、SupCon、完整practical与真实checkpoint smoke实现；这次未重复跑不变训练模型。一次限定新profile入口的独立P0/P1审查PASS，无发现；不增加审批或旧机制重复审查。普通账号N607只读preflight已完成，实际传输/编译/smoke/启动状态待发布后按证据追加。


## 实际发布与运行

执行 commit `d71531d53c1d9ec46e3915963e05bff96760e80e` 已提交推送并独立核对远端OID。单份release archive传输比对、远端编译均VERIFIED，真实匹配checkpoint的两个合成包no-query smoke PASS后直接进入训练。独立读回两个原PID的CWD/argv、GPU0/1归属和stdout增长；两行的actual_native_args.seed与resolved config.model_seed均为2026092702。B00已完成E2/20、444更新；B01已完成E1/20、222更新、28416份实际practical监督；没有failure、target访问或重启。摘要见launch_verification.json。该读数只是进度，不是收益或跨seed确认。


## 实际普通增强补充审计

独立只读代码解析已完成，覆盖实际DataAugmentation、导入的build_aug_base_cfg/configure_augmentor、resolved args、11个操作的顺序/概率/范围/标签依赖/归一化/边界，以及同IQ RDrop与raw practical调用。完整解释见analysis/cvs_joint_ordinary_IQ_audit_20261016.md，逐轮汇总见ordinary_augmentation_audit.json。0IQ/模型前向/训练更新；没有修改本run。概率缩放不等于增益/相位/AWGN范围缩放，前7轮taper恒等，PA保留未验证身份保持限制。此审计不构成Ag有效证据。


## B00 完成，配对尚未齐全

B00 已自然完成固定E20与held RX1 clean＋六practical，worker正常退出。独立重算全部4440step、20轮原生CSV/JSONL与compact记录、800行完整stdout、5994个L/U批次及8820个默认头决策；没有非有限、跳步、警告或U真值使用。最大L尺度重构误差为4.47034836e-07，CROSS_RX层位／前向目标13320项。

L观察568320/U396368，唯一训练物理样本L5040/U45360，全部排除RX1。每模型实际NN前向4686678，训练墙钟1294.776291秒；已扣teacher复制带入counter2，student自身smoke2已包含，公共dispatcher smoke2留全矩阵另计。

B00源clean为83.095238%，lowurban为28.492063%；这只是新seed的对照，无候选增益。B01继续原进程，整个run仍为RUNNING。完整汇总见B00_source_analysis.json，不进行候选晋级或target访问。


## 最终确认：固定 75/25 候选关闭

**VERIFIED：独立 seed2026092702 未通过原定 clean 保持条件。** clean 从83.095238%降到70.000000%，下降13.095238pp，远超允许0.5pp。其余原定源条件通过：low urban+7.539683pp，urban均值+11.825397pp，各suburban与seenV非劣条件及完整健康预算均通过。固定候选仍判FAIL，不以多数条件或两种子平均分替代同时通过要求；不追加seed03、换RX、调比例或延长训练救跑。

| seed02 源 RX1 场景 | B00 正确率 | B01 正确率 | 差值（百分点） | B00 / B01 macro-F1 | 修复 / 新增错误 |
|---|---:|---:|---:|---:|---:|
| clean | 83.095238% | 70.000000% | -13.095238 | 83.008860% / 69.138052% | 35 / 200 |
| high suburban | 38.174603% | 59.047619% | +20.873016 | 32.121756% / 56.919488% | 325 / 62 |
| mid suburban | 38.809524% | 55.714286% | +16.904762 | 33.463651% / 53.314909% | 299 / 86 |
| low suburban | 36.587302% | 54.206349% | +17.619048 | 31.577605% / 52.066552% | 295 / 73 |
| high urban | 38.809524% | 55.000000% | +16.190476 | 33.946829% / 52.773223% | 296 / 92 |
| mid urban | 30.873016% | 42.619048% | +11.746032 | 27.668161% / 41.232710% | 246 / 98 |
| low urban | 28.492063% | 36.031746% | +7.539683 | 25.805816% / 35.005376% | 207 / 112 |

每行每场景均为同一held RX1原L1260、6TX各210包、相同命名信道。clean修复35包、新增200包错误，净损失165包；lowurban修复207包、新增112包错误，净增95包。macro-F1的变化与accuracy方向一致，不能把星地增益写成clean身份提取提升。

### 两个固定模型种子的完整汇总

| 场景 | seed01 配对差 | seed02 配对差 | 两种子均值 ± 样本 SD（pp） |
|---|---:|---:|---:|
| clean | +3.095238 | -13.095238 | -5.000000 ± 11.448396 |
| high suburban | +23.253968 | +20.873016 | +22.063492 ± 1.683588 |
| mid suburban | +19.206349 | +16.904762 | +18.055556 ± 1.627468 |
| low suburban | +20.238095 | +17.619048 | +18.928571 ± 1.851946 |
| high urban | +18.412698 | +16.190476 | +17.301587 ± 1.571348 |
| mid urban | +13.333333 | +11.746032 | +12.539683 ± 1.122392 |
| low urban | +7.539683 | +7.539683 | +7.539683 ± 0.000000 |

三个urban均值的配对差为+12.460317±0.897913pp；lowurban两次均+7.539683pp，但两次B01绝对正确率只为36.111111%、36.031746%。clean配对差的均值为−5.000000pp，SD为11.448396pp。源星地适应有重复的净收益，clean保持没有复现，固定组织不能作为已验证的联合底座。

SD为两个模型种子的无偏样本SD，不是置信区间。两个种子复用同一1260个物理包、同一RX1留出与同一channel/receiver/eval seed，不能当作2520个独立物理样本，也不能代表跨RX、独立信道随机性或正式target结果。所有四行的role IDs相同、原生参数差异仅seed/run/candidate/output，共有SupCon与视图配置相同；每个种子内部L/U物理批次顺序完全匹配。

### suburban 差距与错误重分配

| suburban | seed02 B00 与其 clean 绝对差 | seed02 B01 与其 clean 绝对差 | 差距缩小量 | 其中 clean 下跌贡献 |
|---|---:|---:|---:|---:|
| high suburban | 44.920635 pp | 10.952381 pp | 33.968254 pp | 13.095238 pp |
| mid suburban | 44.285714 pp | 14.285714 pp | 30.000000 pp | 13.095238 pp |
| low suburban | 46.507937 pp | 15.793651 pp | 30.714286 pp | 13.095238 pp |

差距变化满足 `Δ(clean−suburban)=Δclean−Δsuburban`。seed02 high suburban差距缩小33.968254pp，其中13.095238pp来自clean下跌；另20.873016pp来自suburban本身提高。三个suburban实际均提高，但clean下降使“更接近clean”的表象被进一步放大，仍未接近探索2pp目标。不能把该表象作为联合成功。

| TX（原映射索引） | seed02 clean 差 | seed02 low urban 差 |
|---|---:|---:|
| 0 | -13.809524 pp | +0.476190 pp |
| 1 | +9.047619 pp | +37.619048 pp |
| 2 | -49.047619 pp | +4.285714 pp |
| 3 | +0.000000 pp | +0.000000 pp |
| 4 | +0.000000 pp | -1.904762 pp |
| 5 | -24.761905 pp | +4.761905 pp |

| day | seed02 clean 差 | seed02 low urban 差 |
|---|---:|---:|
| 1 | -7.380952 pp | +8.333333 pp |
| 2 | -5.000000 pp | +10.476190 pp |
| 3 | -26.904762 pp | +3.809524 pp |

seed02 clean下降集中在TX2（−49.047619pp）、TX5（−24.761905pp）、TX0（−13.809524pp）；day3下降26.904762pp，另外两day也下降。没有按类别或日期删除、重权或选择性重跑。seed01的类间代价继续保留，完整每场景CM与全部TX/day统计见source_analysis.json及two_seed_source_summary.json；源均值不代表所有身份均改善。

### 完整日志、默认头与实际成本

每行全部4440原生step、4440视图记录、4440物理裁剪记录、20轮原生CSV/JSONL与compact日志、800行完整stdout、5994个L/U批次、8820个默认头决策均独立重算。没有警告、非有限值、跳步、U真值使用、额外诊断头或checkpoint继承。实际损失尺度最大误差B00为4.470348e−7、B01为3.874302e−7；每行SupCon为4440调用、3330非零、13320层位／前向目标，额外视图无残留hook。默认W与骨干联合更新，最终只取epoch20。

| 实际成本／状态 | seed02 B00 | seed02 B01 |
|---|---:|---:|
| 从零 epoch / 更新 | 20 / 4440 | 20 / 4440 |
| L 网络包前向 | 3409920 | 3409920 |
| U student 包前向 | 396368 | 396368 |
| ownEMA 实际包前向 | 396368 | 396368 |
| student V 与自身初始 smoke 包前向 | 475202 | 475202 |
| held-source 七视图包前向 | 8820 | 8820 |
| 每模型 NN 包前向合计 | 4686678 | 4686678 |
| 训练 practical IQ 生成 | 0 | 568320 |
| 源评估 practical IQ 生成 | 7560 | 7560 |
| 训练墙钟（秒） | 1294.776291 | 2145.872858 |
| final L CE | 0.318492099 | 0.320248424 |
| seen V 健康正确率 | 98.134259% | 98.194444% |

seed02两模型合计40模型轮、8880更新、9373358实际NN包前向，含公共dispatcher smoke2；teacher复制带入的counter2已扣除，student自身smoke2已包含。两种子四模型累计80模型轮、17760更新、18746716实际NN包前向、1136640份训练practical IQ；没有续训或外部教师。NN成本两arm匹配，CPU信道生成和训练墙钟单列；不能把并行训练墙钟相加当经过时间。

seenV由98.134259%到98.194444%，健康条件通过，而heldclean明显下降。该实例直接说明源健康正常或训练loss正常不能替代跨RX识别证据。两worker与dispatcher均自然退出，输出保留，无技术重启、停止或修改其他健康任务。

### 结论与下一项可放弃的检验

已证实：完整practical监督在此源折、两个模型种子都有星地整体识别净收益。已否定：本固定75/25组织能稳定保护clean并同时形成联合收益。该否定只关闭本配方，不把它扩大成所有As均无效，也不宣称F+CROSS_RX本身有问题。

当前A_g仍无通过候选。下一唯一待检验假设为：当前F+CROSS_RX默认头仍对轻度共同相位变化过敏，存在可通过纯IQ观测视图研究的缺口。先做一个不更新模型的最小source检验：固定自有合法B00/seed01/final20 checkpoint，只对fit源L5040评估identity、+0.37rad、−0.37rad三个IQ视图，共15120默认头决策；不读heldRX1、U、V或target，不新增分类器。该检验尚未登记、实现或启动。

相位幅度固定沿用已登记公共物理诊断的0.37rad，不从目标或本轮分数选幅度；单位复乘保持IQ功率、时间位置和频率，不施加PA、量化、剪裁、噪声或多径，也不等于更换真实RX。旧CNN全局相位训练未获晋级、旧E200接收相位敏感性结果均保留；不能把旧架构结果当作当前F收益。仅当两个方向均造成至少2pp源默认头准确率下降，且每方向至少3/4个fitRX下降至少1pp，才支持另行考虑同底座B00/B10的E20；否则关闭该相位视图方向，不扫描幅度救跑。即使敏感性成立，也不证明Ag有效，仍须后续真实clean身份收益验证。

仍缺：有依据且有效的Ag、B10/B11及交互项、正式完整源L6300矩阵与clean＋六practical独立truth-last、最终clean净收益/lowurban>60%/suburban接近clean。源ANALYZED不是正式测试或联合目标完成。

完整20轮曲线、七场景结果、全部TX/day分层和两种子对照四张图已生成并逐图检查，保持私有local_artifacts。小型可审查汇总为[source_analysis.json](source_analysis.json)、[two_seed_source_summary.json](two_seed_source_summary.json)。训练release保持原执行commit，后续分析代码未热改运行中的训练。
