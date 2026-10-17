# 固定平滑 IQ 滤波首次训练效用：全源 E30

状态：IMPLEMENTED_NOT_LAUNCHED。run_id：`20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01`。实际矩阵与 seed 见 [experiment.json](experiment.json)，本地检查及限定独立审查见 [verification.json](verification.json)。

用户最新预算为 E30，基线为 F＋跨 RX 同 TX 对比学习。全部源 RX1/3/4/6/8、day1/2/3、L6300／U56700 隐藏 TX／V27000、split392005 保持，没有训练域留出或 L 内部再划分。使用 F 三路 SIG_TIME、统一完整 Sinc 修正、真实 joint160、既有默认 cosine 头，骨干和头按原流程联合训练。保留 NoExplicitCFO 和必要逐包补偿。

## 问题与历史负结果

唯一假设：既有源 RX 关联支持的轻度有界平滑观测变化，在相同训练份额下首次训练能改善跨 RX clean，并保留普通 practical 适应。这里检验学习效用，不宣称数字滤波替换真实 RX。

旧源研究 `20261016-diagnostic-F-crossRX-spectralIQ-fitL-s2026092701-m1-r01` 仍为 FAIL_FIXED_FILTER_CLOSED：跨 TX/day 的 RX 谱关联及默认头身份保护通过，但有限窗 PSD 与循环参考的保真条件失败；没有训练过 FIR 模型。本轮不改判旧结果，不重算或放宽旧阈值，不更换核、幅度、边界、统计拟合、RX 子集或随机种子。原 L5040 只用于拟合源输入统计，包含于当前 L6300；不继承旧四 RX 分类模型权重。新模型实际使用全部五 RX。

旧普通 practical 拼接已显示 urban 适应收益与 clean 代价；时钟、接收校准刷新、有限窗质量、PDP 形状、MMSE 混合及等均值 N0 的固定配方失败均保留。最新包络统计假设亦失败，不重复这些训练路线。

RDrop 是既有随机子网络正则，不记作 IQ A_g。历史 SIG_TIME 两前向与 RDrop 的正收益来自不同 clean 配方；旧 ORIGINAL／TWO_CE／RDrop 完整匹配登记尚未发布，当前 F＋跨 RX 配方也没有完整三者 E30 星地比较。该证据缺口如实保留，不新增阻断闸门、不将不同实验增益相加。

## 最小矩阵与模型来源

| cell | 新增 A_g | A_s | 来源与增量 |
|---|---|---|---|
| B00 | 无 | 无 | 复用已冻结匹配 E20＋10 的 B00 |
| B10 | 固定 FIR，仅 E21 至 E30 | 无 | 从同 B00 的全源 E20 父独立追加 10 轮 |
| B01 | 无 | 原 practical | 复用已冻结匹配 E20＋10 的 B01 |
| B11 | 固定 FIR，仅 E21 至 E30 | 原 practical | 从同 B01 的全源 E20 父独立追加 10 轮 |

新增行仅 FIR_B10-s2026092701、FIR_B11-s2026092701。匹配对照来自 `20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01` 的 nonAg B00/B01，不使用旧 clock Ag 父模型。B10 父为完整源拼接参照的 REPEAT/source-final20；B11 父为完整源四因子的 B01/source-final20。均按固定末轮产生、从零初始化且实际源角色一致；继承自身 EMA，重置 AdamW 与原生 cosine10。不是恢复完整 optimizer 轨迹，也不是从零完整 E30 Ag。加载时只复用已有匹配来源证据与数据句柄，拒绝来源不明、契约不匹配及目标参与选模的权重。

每行父成本 E20／4440 更新，新增 E10／2220 更新，累计 E30／6660 更新；两行新增 20 model-epochs／4440 更新。原 U 规则、教师、门控、SupCon 层位及权重不变；不使用 U TX 真值、不新增 U 增强。测试结果不影响父模型、核、幅度、损失、轮次、seed、行顺序或选择性重跑。

## 实际 IQ 与监督路径

保留旧 ordinary 的实际父参数和 E13 日程；不是完全无增强对照。每步共有 ordinary128 的同 IQ 两次 RDrop 前向占监督 .5，extra128 的同 IQ 两次前向占 .5；总 L 尺度为 1，另加原 SupCon .1。extra 中按既有交替 mask 分 64 A_g 槽和 64 A_s/control 槽。

A_g 仅在对应 ordinary 槽之后施加一次原 structured FIR：33 tap Hann、reflect16、去除数值中心延迟、原包 RMS 恢复；原核响应最大绝对幅度不超过 .5 dB。所有原 12 个 structured 核等概率，物理五元组＋累计 epoch/step＋独立 seed20261017991 决定选择；不依赖 TX 标签，不消耗原全局随机流。RMS 恢复可消除纯增益，但保留相对频率响应；不增加 PA、DAC、剪裁、CFO、AWGN 或多径。边界延拓及有限窗与循环参考差异继续明确保留。

A_s 在另一槽从原 L raw 输入调用原完整 practical residual/post_sync 处理一次，不先经过 ordinary，不连续施加完整星地信道。传播、噪声、接收质量及补偿依赖保持；全部合法有标签 A_s 参与监督，不设高置信前置条件。本轮不构造 A_s(A_g(x))，组合视图留待独立问题。共同原监督份额和四次 L 前向保持，对照控制重复前向。

每行新增 L 神经网络包前向 1136640，A_g 观测 142080；A_s 为 B10 的 0 或 B11 的 142080。FIR 额外物理计算与 host enqueue 时间单列，不冒称同 FLOPs；原 U、教师、源评估、测试、wall 各自记录。FULL_AS 的 A_s 剂量 .5、高于四因子的 .25，复用作普通强拼接背景参照，差异不归因于纯 A_g。

## 输入检查、执行与停止

本地聚焦检查 PASS：41 个协议负例、128 个合成 FIR 包、64 次原公式复算，最大误差 4.7684e−7；原 RNG／输入未变、包重排不变；实际 F 参数 900722，合成 NN4，真实 IQ／CP／更新均 0，编译 PASS。一次限定 P0/P1 审查发现自身 final 重载问题；已接入父缓存／自身 final 原生加载／其他来源拒绝策略，一次定向复核 PASS。

启动首步 CPU2、CUDA 空：加载两个匹配全源父 checkpoint，完成真实 checkpoint 无 query 合成 smoke4，并在原 L6300 对 raw＋12 原核共 163800 个默认头决策。总 163804 NN、2 CP、1 原缓存、0 更新／target。沿用原身份保护要求：每核整体下降不超过 .5 pp，各 RX／TX／day 不超过 1 pp。该检查只约束 raw 源输入，不证明旧 ordinary＋FIR 交互无害或训练有益，也不恢复旧 PSD 保真结论。失败保存 SOURCE_INPUT_REJECTED、零训练、零 target；通过则同 dispatcher 自动训练两行，不再申请审批。

本地发布：`python -m experiments.cvs_spectral_train30.publish`；远端 CWD：`PROJECT/releases/cvs_F_spectralIQ_firsttrain_E30_20261017_r01`；入口：`CVS-RFFI/bin/python -u experiments/cvs_spectral_train30/runner.py`。一次 N607 preflight、一次 release archive 传输比对、远端编译／合成检查、启动后 PID/CWD/argv/GPU/日志增长读回。执行 commit 以同 release 的 release_commit.txt 为准；本地登记占位由发布时写入实际 commit。

不可变输出：`PROJECT/runs/20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01`；独占 row/source、prediction、scores、logs。普通账户，CPU2；所有 owner 合计每 GPU 不超过 3 实验进程且预留 12 GB。没有自动重试、换 seed 或追加预算。源身份保护失败科学关闭；启动后低分不技术停止。仅协议越界、来源/数据错误、非有限值、不可执行、输出碰撞、正式 prediction/scorer 无法闭合等所属技术故障停止本 run，保留产物且不干预健康旧任务。

每行固定 final10／累计30 后自动预测 clean＋六 practical，各 168000 原固定接收 IQ；本行七视图全部完成并固定 prediction/logp 后独立 CPU truth-last 评分。无需等待另一行，不改测试 SNR、信道、received IQ 或样本覆盖。预期 source/final checkpoint、实际 args／训练与增强计数、source_guard、七视图 prediction／score／CM／配对错误、成本、completion、result 等产物；未完成评分不能记为实验完成。

## 预定判据与边界

固定单种子反证条件：B10 clean−B00 至少 +.2 pp；B11 clean−B00 非负；B11 low urban−B01 至少 +.5 pp、mid urban 不降；各 suburban 相对 B01 不低于 −.25 pp。任一失败即关闭此固定训练配方，不改核/幅度/父/seed/预算救跑。全部通过仍仅为本配方单种子结果，不自动晋级或扩种子。

必须报告 B10−B00、B11−B00、B11−B10、B11 low−B01、各 suburban 绝对值及相对自身 clean 的差，以及每视图交互 B11−B10−B01＋B00、修复/新增错误与 clean/场景共同正确变化。完整联合目标仍要求 clean 正净收益、low urban>60%、每 suburban 距 clean 的首轮探索目标 2 pp；不能靠 clean 下跌缩小差距。本轮尚无训练或正式评分结果，联合研究保持 ACTIVE。
