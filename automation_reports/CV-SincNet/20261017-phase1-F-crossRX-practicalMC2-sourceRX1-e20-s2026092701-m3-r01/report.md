# F＋跨 RX 同 TX：完整 practical 双实现期望监督

状态：RUNNING／VERIFIED，三组已实际从零启动，尚未完成训练或源评分。当前唯一候选为 MC2 采样组织，联合目标未完成。执行代码以本次 Git commit 与 release_commit.txt 为准，发布后写入 execution_summary.json。

## 问题与假设

旧 normalized/additive As 跨 seed 的 clean 失败保留且关闭；同身份实测 IQmix 已在三组 E20 完成源反证后关闭，不调整这些旧配方。当前新假设：完整 practical 随机实现的有限采样可能使 clean 保持不稳定；在相同 expected objective、相同前向数及监督权重下，两个独立完整 practical 视图的损失均值，可能比重复一个视图更好地保留 clean，同时保留 urban 适应。旧失败只提供动机，没有证明采样方差是原因。

本轮是 A_s 训练组织研究，没有新增 A_g。RDrop 仍是随机子网络正则，不称为 IQ 泛化增强。不预告任何绝对成绩；只有固定默认头的源留出识别结果决定此配方是否继续。

## 固定基座与数据

F＋跨 RX 同 TX 对比学习，三路 SIG_TIME、统一 Sinc、joint160、既有默认 cosine 分类头与骨干联合训练；NoExplicitCFO 与必要内部补偿保留。原浅层／中层 CROSS_RX SupCon 只使用两次旧普通增强前向，不扩展至新增 As 视图。没有新损失类型、跨视图一致性、门控条件、诊断头或部署分支。

原已验证载体为 L6300/U56700/V27000。RX1 在模型、教师、训练、校准前排除，实际训练为 RX3/4/6/8、day1/2/3、6 TX 的 L5040/U45360/V21600；RX1 原 L1260 仅作 final20 源留出评分。source_contract.counts 可保留原载体计数，role_ids 定义实际角色。不能称为完整 L6300 或 L_train5580/L_dev720 训练。U 的 TX 真值始终隐藏，原 13L＋7U、旧伪标签规则、自身 EMA 均不改。

所有模型从零 E20，seed2026092701、split392005，无训练父权重、无续训。烟测只读原匹配 B00 scratchE20/final20 checkpoint，不作为初始化。nominal F50 与 legacy E200 仅为旧构造／历史元数据；实际 args 和预算断言为 20/13/7、4440 更新。

## 最小三行与损失

| 行 | 两个额外 IQ 对输入 | 独立 As 生成数／L包 |
|---|---|---:|
| M00 | raw、raw | 0 |
| M_repeat | As0、同一个 As0 | 1 |
| M_independent | As0、独立 As1 | 2 |

原共同目标完整保留：raw .75＋ordinary .25＋原 SupCon。额外目标为 .125 R(v0)＋.125 R(v1)；R 是原 smoothing .01 CE 的两前向均值加同 IQ RDrop1 KL。各对按物理包均值归一化，整组 pair 权重和均为 1.25，不把整个目标再除以五对。原始监督的共同绝对权重 .75 不变；M00 的额外 raw 使其 raw 总权重为1.0，As 行为.75，分别相当于总 pair 权重的80%／60%，明确披露而不隐式稀释。不能由绝对系数保留推出 clean 必然安全。

每个物理 L 包共五对、10 次有标签 NN 前向。RDrop 每对的两次前向复用同一个已生成 IQ；重复行的两个额外对还复用同一个 As0。两个独立视图只平均损失，绝不平均 IQ 或通过平均提高接收 SNR。三个控制的初始化规则、更新数、总损失尺度、前向数及原 U 规则相同；M_repeat／M_independent 是采样组织的直接匹配对照，不用旧八前向实验冒充。

## 完整信道与接收依赖

直接使用固定后端的原 PairedChannel：practical residual、post_sync、EQ=false（G=1）、25 Msps、2.462 GHz，六场景均匀先验，保留传播、噪声、质量、补偿的依赖。As 输入为原始 L IQ；不先施加旧普通增强，也不串联 As(As(x))。所有合法有标签 L 包的两对 As 均受真实标签监督，无 urban 学生置信度或质量过滤，不删困难包。

MC0 namespace 为 train/epochE/stepS/mc0，MC1 为对应 mc1。M_repeat 与 M_independent 的 MC0 输入／随机流精确匹配；MC1 只改变命名随机流。旧 CHANNEL_SEED=20261016101、RECEIVER_SEED=20261016102、source EVAL_SEED=20261016103 保持。receiver_for_session(cfg, receiver_seed, rx/day/eq) 不依赖 TX 或 mc namespace，接收机 session 参数稳定；配置内场景依赖继续使用原实现。没有修改正式测试 IQ、SNR、信道或接收处理。

原普通增强为三行共有，按实际 build_aug_base_cfg/args 记录。保留旧 ordinary 不等于其每个物理损伤都获新身份保持证明，也不把该共有增强称为本轮 A_g。

## 预算与固定评分

每行 E20／222 step每轮／4440 更新，实际 lr .0002、weight decay .0001、batch128、原 cosine 与 Sinc 物理单位优化保持。合计60 model-epochs／13320 更新，无预算延长。每行 LNN=5683200，其中 extra=2273280，ordinary 生成568320；三行 As 生成分别0／568320／1136640。As 监督的 pair-observations 为0／1136640／1136640，区分生成视图、网络前向、物理包重复和损失配对。

原 U／自身 EMA／V／held／初始化 NN 实测另计，不预测总耗时或隐去教师成本。dispatcher 仅一次匹配真实 checkpoint 无 query 烟测，2 个合成包；本地工程已消耗4626合成NN／3合成联合更新／384完整As生成／128ordinary，0真实IQ／0target／0另训头。历史训练与本轮不混算。

每行 final20 冻结后立即自动源 RX1 clean＋六 practical，每场景1260个同物理包，共8820行决策、三行26460；是重复观察同一1260物理样本，不冒充26460独立样本。固定硬预测后独立复算 accuracy、MacroF1、混淆、双方控制修复／新增错误、所有 TX/day、suburban 绝对值及与自身 clean 的差距。完整20轮、每行4440条native/view/Sinc记录、全部L/U序列、MC0共享draw和MC1覆盖／损失／成本一起分析。

本次明确为源域先行反证，不开放 formal target。完整 B00/B10/B01/B11、交互项、有效 A_g、跨 seed 与正式 target clean＋六 practical 仍缺，不能把单源折／单种子通过写成联合目标。最终 clean 净正收益、low urban>60%、suburban 各场景接近自身 clean（首轮探索2pp）保持，不靠降低clean缩小差距。

## 固定晋级与停止

M_independent 相对 M00：clean accuracy／F1 不下降、low urban≥+5pp、urban均值≥+3pp、各 suburban≥−.5pp、seen V≥−.5pp。相对 M_repeat：clean≥+1pp、clean F1≥0、每 practical≥−.5pp、seen V≥−.5pp。全部预算、默认头／U隔离／生成计数正确，final共同base目标≤.5是健康条件，健康本身不等于识别收益。

任一科学条件失败即关闭固定 MC2，不扩 M4、不改系数／RX／seed／预算救跑。PASS只支持独立源种子确认，不构成 Ag、formal target 或联合完成。系统性协议／角色／初始化／预算／输出冲突／非有限／无法启动／非法预测失败时保留产物、取消未启动行，健康 worker 继续，不自动重试；不改动其他 owner 的健康任务。

## 执行与验证

run_id：20261017-phase1-F-crossRX-practicalMC2-sourceRX1-e20-s2026092701-m3-r01。唯一 owner：codex/root/F-CROSS_RX-practicalMC2-E20-20261017。

普通 N607 CVS-RFFI／FP32；CPU dispatcher 2 threads、loader0。GPU由既有共享容量规则分配：每GPU最多3个所有owner总实验进程、空闲显存至少12000MiB。CWD为 PROJECT/releases/cvs_F_crossRX_practicalMC2_sourceRX1_e20_20261017_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_practical_mc2_e20/runner.py。不可变输出为 PROJECT/runs/上述run_id，各行source与logs；绝对路径见experiment.json。

本地43负例与真实F结构合成测试PASS：共同目标全部参数梯度精确匹配、三行scratch状态与CPU/Python RNG一致、10前向、每对同IQ、MC0输入／draw记录跨As行完全一致、receiver session稳定、默认头联合更新、完整joint160导出／状态键不变。一次独立P0/P1执行审查PASS，无待修问题；未追加未变代码或报告审查。

预期产物：真实checkpoint smoke；实际args/config/角色/init/frozen；全部步与L/U序列；MC2_progress/audit及覆盖；七视图预测／独立评分；完整曲线、分层、错误转移、实际成本、同row报告和登记。只作一次发布归档本远端SHA比较与远端compile；不增加数据重验、签名、receipt链或审批层。


## 实际发布与独立启动读回

执行 commit 8c55f5f542a6a28b699156859f31345b9d15f76a 已推送，远端 OID 与本地 HEAD 独立核对一致；一次归档 SHA 比较、remote compile VERIFIED，匹配真实 checkpoint 无 query smoke PASS。dispatcher 与三个 worker 的 PID/CWD/argv/CUDA环境、实际 GPU 和逐步日志增长 VERIFIED。实际 seed2026092701、20/13/7、scratch、无 baseline/teacher 父权重；resolved 为原 F+CROSS_RX、L5040/U45360/V21600、10次L前向、两额外对各.125、总权重1.25，原U规则保持。

- M00-RX1-s2026092701：GPU 0，观察完整 step 368/4440，E2/20。仅为运行证据，无最终识别结论。

- M_repeat-RX1-s2026092701：GPU 1，观察完整 step 209/4440，E1/20。仅为运行证据，无最终识别结论。

- M_independent-RX1-s2026092701：GPU 2，观察完整 step 157/4440，E1/20。仅为运行证据，无最终识别结论。


## 首行完成：M00

M00 raw/raw 对照已自然退出，final20 冻结后源 RX1-held 的 clean＋六 practical 全部完成。独立复算全部20轮、4440步、L/U物理序列、8820个固定硬预测和成本，0新增NN／0更新／0target。另两条As行仍运行，保持原参数、预算和判据；本段不作候选晋级或假设成败判断。

| 源视图 | accuracy % | MacroF1 % |
|---|---:|---:|
| clean | 79.841270 | 79.275233 |
| practical_high | 37.063492 | 26.286933 |
| practical_mid | 36.666667 | 26.902714 |
| practical_low_suburban | 33.968254 | 25.832379 |
| practical_high_urban | 37.380952 | 27.534082 |
| practical_mid_urban | 30.952381 | 25.649411 |
| practical_low_urban | 27.063492 | 23.452034 |

实际NN包前向6959958（含学生／自身EMA／seenV／held读出，扣除teacher继承仪表offset2），L5683200／extra2273280，As生成0；一次dispatcher合成smoke2尚不并入单行。final共同base目标0.30881574，仅为健康信息。每视图同1260物理样本，不能将8820次决策视为独立样本。完整聚合见source_partial_analysis.json，完整三行结论待两条As行完成。
