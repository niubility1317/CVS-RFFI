# F＋跨 RX 同 TX：固定 75/25 practical 的独立 seed02 确认

- 当前状态：RUNNING；两个原进程已独立核实，尚无最终确认预测。
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
