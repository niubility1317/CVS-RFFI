# F＋跨 RX 同 TX：As 两次 CE 与 RDrop 的匹配 E20

状态：RUNNING／原进程身份与输入权限 VERIFIED；训练、预测和独立分析未完成。只新增一行从零 E20；两个已完成控制复用结果，不加载其权重。固定 F／三路 SIG_TIME／统一 Sinc／NoExplicitCFO／真实 joint160／默认 cosine 头正常联合训练，原 13L＋7U、oldgate、ownEMA 与隐藏 U TX 保持。

## 唯一假设与已核实依据

当前额外 As 对沿用 mean(CE1,CE2)＋对称 KL。原 clean TWO_CE/RDrop 的阳性结果不能证明该 KL 在 practical 下有收益。已完整读取自有 Q_ord 全20轮记录：As CE 均值0.6143249255、As KL均值0.0794714871，E20分别0.4984014025／0.0561134103。KL不是近零，但这些数值不证明它有害，也不作为识别增益。

唯一待检验假设：仅移除额外 As 对的随机子网络 KL、保留两次 CE 与全部原监督，能否比相同 As/RDrop 增加 urban 收益并保护 clean/suburban。这是 As 配方消融，不是 IQ 泛化增强 Ag；不新增损失、门控、网络或教师。

其他候选边界：共享微弱加性方向已 FAIL_CLOSED，不重开；F没有BN，运行统计污染解释不适用；另一对话当前随机窄频带遮蔽不在本任务范围。复共轭/IQ反射未进入源实验：可逆性不能证明TX身份语义保持，[Wang等原文](https://arxiv.org/pdf/2304.14795)讨论了翻转可能改变设备特有星座偏差的限制，不凭调制识别的翻转经验晋级。该文仅提供此限制，不为当前As收益背书。

## 最小三行矩阵与真实控制

唯一维护矩阵、完整参数和路径见 [experiment.json](experiment.json)。

|逻辑行|额外 IQ 对|额外对损失|执行|
|---|---|---|---|
|Q0|raw/raw|mean两CE＋KL|复用原 sceneCycle B00/final20|
|Q_ord|同一fullAs(raw)的两次随机网络前向|mean两CE＋KL|复用原 sceneCycle C_IID/final20|
|C_CE（唯一primary）|同一fullAs(raw)的两次随机网络前向|mean两CE，无KL梯度|新增scratch E20|

三行均8L前向，完整base raw总权重0.75＋ordinary0.25＋原浅/中层跨RX同TX SupCon，额外对权重0.25，总对权重1.25；没有除以新视图数稀释raw。原base三对RDrop完全保留，只改额外As对。两个As前向复用一次已生成IQ，不重抽practical。KL只在no_grad下记录诊断值，不连入目标或梯度，不把“权重为0的旧一致性图”冒充纯CE。该改变减少一个损失图的反向，不声称总计算时间完全相同；完整NN数与墙钟分别报告。

控制是20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01的B00与C_IID，执行commit c9a74172e5f81bb4f04ce12cd4386ad4dd953ee9。既有loader在首smoke后核对实际resolved config、完整L/U/V角色、scratch/noancestor/final20/4440、8L/1.25、头正常更新、原U与无target，不凭名称复用；不读取控制模型权重初始化候选。其源RX1 clean分别80.158730%／84.444444%，lowurban27.777778%／33.888889%，只是已知控制描述，不能预测新行。

## 数据、算子及训练边界

原载体split392005/VALIDATED_ONCE；训练只fit RX3/4/6/8、day1/2/3，L5040/U45360 hiddenTX/V21600。RX1 L1260在任何训练输入、教师、V、历史缓存继承中排除，冻结后才作七视图源评估。这不是全L6300或L_train5580/L_dev720，源RX1已有历史暴露，不声称新盲测。无checkpoint训练继承；首个真实CP仅匹配原fit角色的兼容smoke后丢弃，候选从零。

旧ordinary共有配置保留，已核实含PA enable/.14、CFO等ramp参数、循环位移、增益及AWGN/multipath；这里不新启用它们、不宣称已证明标签保持。ordinary在独立对中；practical只作用原raw，避免ordinary与完整practical重复损伤。NoExplicitCFO与必要内部补偿不删。As保持原IID六场景及传播/噪声/状态/接收处理/补偿依赖、residual/post_sync/EQoff/G=1/reference30/reflect；全部合法L As监督，无置信门槛，U不增加As或KL变体。正式测试场景/SNR/receivedIQ完全不变，本sourcepilot无target访问。

## 预定识别判据与放弃条件

只读固定final20，不按V择轮或救跑。相对Q_ord：lowurban准确率至少＋1pp、urban三场景均值至少＋1pp、midurban不降、lowurban Macro-F1不降；clean准确率和F1各最多下降0.5pp，每个suburban最多下降0.5pp，seenV最多下降0.5pp。相对Q0：clean准确率/F1不降、lowurban至少＋5pp、urban均值至少＋3pp、midurban及low F1不降；各suburban和seenV同样最多下降0.5pp。每行须完整预算/默认头/无target、final base L objective≤0.5；新行4440次As纯两CE且KL梯度项0。

绝对suburban准确率、各自clean差距、全部配对修复/破坏与共同clean正确集合均报告；本轮通过也不代表suburban接近clean或联合目标。任一原条件失败，关闭固定As KL移除，不试中间KL权重、改CE权重/场景/日历/RX/seed/预算救援。通过只允许另行固定独立源seed确认，不自动扩预算或访问target。

## 成本、发布与停止

新增20 model-epoch／4440更新，L前向4546560、其中额外1136640，fullAs生成568320；U学生/教师、seenV与held8820前向单列。按相同执行路径预期总NN约5823318包，实际以计数器复算，另计dispatcher真实CP合成smoke2包。复用两控制40历史model-epoch／8880更新／11646636 NN，不把历史成本写成0或重复训练。没有新增教师、辅助头或U变体。

ordinary N607/CVS-RFFI，dispatcher CPU2/CUDA隐藏；新worker CPU2/loader0，使用原共享capacity锁计入所有owner，每GPU总进程≤3、空闲≥12000MiB才启动。一个owner codex/root/F-AsPairCE-E20-20261017；CWD PROJECT/releases/cvs_F_AsPairCE_sourceRX1_e20_20261017_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_as_pair_ce20/runner.py。新独占root PROJECT/runs/20261017-phase1-F-crossRX-AsPairCE-sourceRX1-e20-s2026092701-m3-r01，日志logs/C_CE-RX1-s2026092701.stdout.log。实际发布HEAD由publisher写release_commit/resolved；精确私有运行位置留本地runtime和原远端记录，不写Git。

一次普通账户preflight、一个release archive/一次SHA比较、remotecompile；launcher首步真实匹配checkpoint no-query smoke PASS即继续检查控制并scratch训练。只读核对原PID/CWD/argv/GPU/日志增长。角色、来源、非有限、预算、输出冲突、不能启动或非法预测等系统技术错误保存失败并取消未启动行，无自动重试；已健康运行和其他owner不动。科学失败不提前停止，完成固定final和源七视图后判定。

预期产物：原始/末态CP、effective roles、actualnativeargs/resolved、完整step/epoch/物理LUV观察、As纯CE审计、scene覆盖、ordinary实际配置、8820个源预测、rootsmoke/controlreuse/launch/queue/result/completion或failure及完整stdout，独立全量损失/角色/计数/配对错误复算、曲线与分层报告。无正式target预测或truth-side评分，不能标为联合目标完成。

## 已完成的本地验证

19协议/配置/入口反例＋15固定科学反例PASS；真实F初始state、8实际IQ输入、global/dropout RNG逐位相同，74参数梯度差只等于旧额外0.25KL项，最大误差1.1920929e−7；原base KL/SupCon保留，默认头与骨干正常更新，导出160与实际特征一致。合计2054合成NN包前向、256 fullAs合成生成、1合成优化更新，0真实IQ/CP/target。首反例误将已有As=true当成变更，0NN即发现并修正；Torch2.1/NumPy2旧桥警告保留，没有改环境。唯一P0/P1静态审查PASS，无额外报告审批或哈希层。

完整目标仍ACTIVE：本轮不新增Ag，不证明As稳定跨seed、同契约四因子交互、clean净正收益／lowurban>60%／suburban接近自身clean；这些须后续实际身份识别结果证明。


实际启动：执行commit 89505b966a853e9126f372c89581f4c2b1c8eb16，已push并独立核对remote OID；一次archive传输比较及remotecompile VERIFIED。真实匹配CP合成no-query smoke PASS后直接继续，两个原控制完整metadata核对PASS，控制权重没有初始化新行。唯一worker从零，原PID/CWD/argv/GPU0/CPU2一致，进程CPU ticks持续增长、已创建实际step记录；角色独立读回L5040/U45360/V21600，初始化无祖先/no target/own scratchEMA。当前尚无完整epoch结果，不宣称收益；原健康进程不热改、不重启。


首轮执行进展：原worker已完成E1／222实际更新，L前向227328、额外对前向56832、完整As监督28416，训练已覆盖全部5040可见L。stdout由6500增至11665字节，原PID/CWD/argv/GPU保持一致，failure不存在；epoch observer已通过逐步As meanCE/零KL梯度预算检查。原SupCon按既定E6启用，当前尚未启用，不提前更改。真实性能读出仍等待final20及源七视图，当前不能报告新增accuracy/F1/clean代价。


离线工具进展：已实现只读完整产物收集、全4440步base/extra目标公式、原U边界、完整scene序列、8820默认头预测、paired repair/break与共同clean正确集合、TX/day及18个TX×day单元、成本和完整曲线工具。40个独立读者反例PASS，含纯CE不受诊断KL改变影响、旧控制仍使用CE＋KL、非法heldRX/U真值/预算拒绝。新读者已逐行完整复算两条旧控制（各20轮/4440步/8820预测）；没有重做NN/训练、读取IQ或权重。新行原进程已完成E5/1110更新，仍live，无final20/七视图新结果，不将前缀进度当全日志结论。离线代码未热改远端不可变发布。


## 已有普通 As 的独立控制错误归因

此次仅对固定Q_ord与Q0的17,640个已有七视图硬判决完整复算，无新IQ、checkpoint、NN或更新，不改当前C_CE/阈值/训练。该对照与新行同8L/1.25/E20/seed01/sourceRX1，原RDrop/SupCon/U保持；只能说明这个已暴露源折/seed的结果，不证明跨seed稳定或正式目标泛化。

Q_ord相对Q0的clean accuracy＋4.285714pp／F1＋4.289331pp，lowurban accuracy＋6.111111pp／F1＋8.545529pp，urban三场景均值＋10.846561pp。lowurban整体修复201、新错124，净增加77个正确判决；clean修复74、新错20，净增加54。

|按两模型clean判决形成的固定分组|包数|lowurban修复／新错|净正确变化|总体准确率贡献 pp|
|---|---:|---:|---:|---:|
|both_clean_correct|990|186/65|121|9.603175|
|As_clean_repaired|74|5/14|-9|-0.714286|
|As_clean_broken|20|1/7|-6|-0.476190|
|both_clean_wrong|176|9/38|-29|-2.301587|

共同clean正确集合内净＋121，而其他组净−44；因此不能把总体＋77全说成无代价修复，也不能把As收益全归为改变clean正确集合。这里是两个训练后默认头的描述性分组，不是传播状态、观测质量或因果分解，不作门控或删除样本的依据。新C_CE完成后仍按预定总体accuracy/F1与保护条件判断，同时报告同集合修复与新错。

Q_ord三个suburban与自身clean仍差25.079365／28.730159／31.825397pp；相对Q0的各suburban绝对正确率增加19.444444／15.476190／16.269841pp，clean同时提高，故当前控制的差距缩小没有靠降低clean，但远未接近探索性2pp。上述仍是已有控制结果，不是新C_CE成绩。完整私有聚合保存在local_artifacts/F_As_pair_CE_E20_20261017/final/control_As_effect.json。
