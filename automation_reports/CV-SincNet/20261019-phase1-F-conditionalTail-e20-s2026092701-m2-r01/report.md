# 纯 F：逐包双完整信道条件尾风险 E20

状态：LOCAL_VERIFIED，未发布。主目标仍为 low urban>60%、clean 相对同预算 NoAs 可重复正收益、逐 suburban 接近自身 clean。本轮是单 seed E20 机制比较，不能据源观察或单 seed 微增声称完成。

## 唯一问题及既有源证据

问题：对同一个合法物理 L 包，在两个独立完整 practical 观测之间保留固定监督总量、稍偏重当前较难观测，能否比均匀双 As 和原普通 As 提高信道稳定性。它并不预设所有高 CE 信道都可恢复，也不拒绝难包。

既有纯 F B1 FINAL20 的 V360 两 draw cached logits 已完整重计，mid/low 分别有 111/360、156/360 包只在其中一个 draw 答对；两 draw argmax 不同分别为 133/360、199/360。它证明同包跨 draw 的不稳定性，不能证明 max 风险训练有益。仅用原声明源诊断缓存，无新 NN/IQ/拟合；配方先固定再重计。source_basis.json 保留全部计数。

同时完整复盘原 B0/B1 各20 epoch/4440步骤和全部机制日志。B1 As CE 后段仅小幅波动，E14 .699661、E20 .694575；clean V E14 97.740741%、E20 97.762963%，没有足够同步改善证据据此直接延长 E50。不得因训练残余损失或已有目标成绩而延长。

已关闭的 nontrue/Jacobian/sceneRisk/capability/raw-lookahead 保持关闭。旧场景 GroupDRO 在跨包场景组分配；旧 withinTXHard 在不同物理包之间重新分配。这里每包都保留相同总质量1，只在自己的两个完整信道 draw 间分配，不让任何 TX/物理包因别人更难而丢失监督。旧 quality bank、recoverable proposal、Latin-hypercube、概率边缘化、公共传播/噪声对照和 raw-As mask 不重启。其他对话的 U 学生路由、HQ raw 配对、暂未发布 noise KL 不纳入本轮。先前双独立完整 As 的曝光结果提示必须配双 draw 均匀 CE 控制；不将新增曝光收益归给条件尾风险。

## 固定训练与控制

experiment.json 维护两新行及路径/seed/config。B0、B1 原纯 F 同 seed scratch E20 冻结读出直接复用；不加载为训练初值。新 CE2 和 TAIL2 都从同 seed 随机初始化，累计20轮=13有标签+7原U，4440更新。F=SIG_TIME_X3+CENTER_25K+CENTER_UNITS，实际joint160/900722参数，现有 cosine6×160 scale30 与 Sinc/骨干正常联合更新，无 CROSS_RX/RDrop、新分支、诊断头或身份重训。

记同一包两 draw 的真实标签 CE01 为 c0、c1。CE2 的 As 项为 (c0+c1)/2。TAIL2 为 .5×(c0+c1)/2+.5×max(c0,c1)，按128包平均。总 L=.5×ordinary CE+.5×As 项；原 U 损失另按原流程保留。TAIL2 的两 draw 权重为 .75/.25（精确 tie .5/.5），CE2 为 .5/.5；每包权重和1，所有合法 As 都保留正真实标签 CE。最大项索引 detach，无学生正确/高置信门。相对总 L，普通份额 .5；CE2 每 draw .25；TAIL2 较难 .375、较易 .125。系数、K、阈值不扫参。

ordinary 原增强保持；两个 As 都直接从原 native raw IQ 单次完整引擎生成，不叠 ordinary，不重归一化已生成信号。draw0 原 namespace/scene先验/seed/step 完全保持；draw1 保留实际 epoch，将 namespace step 固定映射为4441−s，使用原稳定种子算法。全部4440步两个 key 在当前epoch互斥，未按预测重抽或挑场景。receiver session 固定 RX/day/eq、不含TX，原15个会话硬件两 draw 相同。原传播、噪声、接收状态、锁定、同步补偿、AGC 和 noEQ 依赖保留；新增噪声比不叫总 SNR。

额外第二 As 的 Dropout 使用 private fork seed20261019631+s，恢复原 CPU/CUDA RNG；两新行相同，原 ordinary/firstAs/U 抽样流不因额外 Dropout 前向推进。额外前向仍有随机 Dropout，非 DET。每新行每步3L384学生NN，两 draw 全部反向；CE2与TAIL2完全匹配。原 B1 是2L及冻结CSIL shadow前向，不声称其学生反向成本相同；新两行不加载或前向该外部教师。

完整 L6300/U56700 hiddenTX/V27000、RX1/3/4/6/8、day1/2/3、split392005、eq1/IQ256/Fs25MHz/NoExplicitCFO不变。U原资格、weak/strong、ownEMA与更新/分母不变；不读U标签。当前聊天明确固定该基础配方，覆盖通用外部对比的普通PL/As.68默认，不据此静默改训练契约。V仅无更新健康/机制读出，不拟合阈值、增强幅度或头。

## 冻结、源判定与完整收尾

固定 FINAL20 student。预定源判定：同固定 V1800×2draw 的 mid/low urban 平均，TAIL2−CE2≥.5pp，TAIL2−B1≥.5pp；完整 V27000 clean，TAIL2−B0≥−.2pp。三项合取，决定写入后才读新正式数值；FAIL关闭该 K2/half-mean-half-max固定方案，不扫 mixture/K/seed 救回。PASS只是另行规划 ownE20+30 的源资格，需要同预算对照及优化器/EMA/U缓存/采样器/信道衔接；本run不自动E50。源结果不能证明未见RX性能。

每新行训练冻结后立即进入原七环境预测，固定clean+六practical，每view168000，正式receivedIQ/SNR/困难包/物理ID不变。两新行预测全部固定且源决定已写后，分别独立CPU exec truth-last scorer；不因科学负结果取消已登记评价。不用正式数值调参、选种子、重排或选择性重跑。原B0/B1预测/评分直接复用，benchmark已暴露，不称新独立盲测。

最终报告 all7 accuracy/macroF1/最差RX、RX/day/TX、修复与新增错误、各suburban绝对差距。分开列TAIL2−CE2（条件尾风险）、CE2−B1（多draw曝光）、TAIL2−B1（总变化）、TAIL2−B0（clean净变化/星地总作用）。仅1个模型seed，两个信道draw/两个条件不是独立seed。固定真实joint及Sinc/各骨干/默认头实际更新证据与全4440权重/损失复算，不能只报非零梯度或低loss。

## 验证、成本、执行身份

本地数学公式/两 draw 精确梯度/逐包正权与总和/tie/交换对称/11个协议负例、全4440 namespace互斥通过。实际 F 的共享训练hook两模式均跑通：768合成NN、2个仅本地丢弃更新，初始化逐参数相同，真实Sinc/骨干/默认头联合梯度和更新、第二Dropout恢复nativeRNG通过。开发早期测试误用optimizer函数签名，在384合成NN/0更新时失败；按实际接口修正后通过。本地累计1152合成NN/2临时更新，0真实IQ/CP/目标。此测试不替代远端真实来源/物理执行证据。

新增共40模型轮/8880更新；每行 L1704960学生NN，As1136640完整物理模拟，原U学生/ownEMA教师各396900NN。原生验证、final特征读出、源面板12600NN/行、正式1176002NN/行、CPU/墙钟/模拟与拷贝另据实际计数。首步共享一次原B1真实训练checkpoint无query smoke2NN，只读且不继承；每行原生scratch init保存/重载smoke仍按原流程保持。原B0/B1及外部shadow历史成本单列，不能说新机制零成本或三路相同反向预算。

N607普通账号既有环境，FP32/TF32off/CPU2，无安装。唯一owner codex/root/F-conditionalTail20；CWD PROJECT/releases/cvs_F_conditionalTail_E20_20261019_r01；输出PROJECT/runs/20261019-phase1-F-conditionalTail-e20-s2026092701-m2-r01；入口CVS-RFFI/python -u experiments/cvs_conditional_tail20/runner.py，逐row source/prediction/scoring/log独占。每GPU所有owner最多2计算进程，free≥12000MB，CUDA初始化位于共享容量锁内，不干预健康旧任务。

一次限定P0/P1独立审查；代码/config commit后发布一次不可覆盖release/run，一次普通preflight/单archive比对/compile。真实CP无query smoke是launcher首步，PASS立即继续。启动后核对PID/CWD/argv/GPU/日志增长。只实际数据/头/非有限/接口/输出/合法预测等所属系统技术故障停止自身并保留产物，不科学低分停止，不自动重试。没有额外审批/数据重验/文件哈希链。

预期首步smoke、launch/queue、每行source完整state/config/init/ownEMA/原生step/sequence/units/mechanism/两draw统计、source_qualification、源决定、all7 prediction/separate score、analysis/complete。IQ、权重、原始日志、机器状态与私有runtime不入Git。

实际 code/config commit `1e3cdcca4481c2f30c864e452ab5f69c8db46aea`。唯一独立限定 P0/P1 审查 PASS，无直接使本次运行错误、越权、覆盖、不可启动或无法合法预测的问题。发布前完整本地检查已完成；尚未以此宣称远端训练或识别收益。
