# Phase3 原生开放集特征增量

2026-10-29。候选 `20261029-phase3-source-embedding-increment-s2026092701-r01`。冻结表示的源拒识开发最高观测到85.7460%，21台辅助设备与域配重的组合反而降为75.0794%，全部未过源95%门槛。本次检验让原生编码器学习未知类边界是否有效；不把这些负结果当作表示的绝对能力上限。

两个匹配行分别为 `supervised_only`（原生L监督增量）和 `uniform_oe`（相同监督＋源OE均匀目标）。监督骨架包括原T1配对KL、As真实标签CE和跨RX对比，因此不称纯CE或PAMF-CE。两行都独立从已核实完整六类三视图／固定时间排列的E50原始权重初始化，不从对照行继续。原11100步后固定新增222步，累计11322，按原生epoch计为51；这是独立Phase3增量，不改Phase1/2历史E50或默认模型。

先沿原训练构造器重建完整六类wrapper，再strict加载验证后的payload model。不能对推理reader直接解冻，因为其Sinc预计算路径会切断原梯度。首步用合法L两条真实IQ检查Sinc参数集合与默认头至少有非零有限梯度；OE行再用两条OE检查梯度连通。不执行优化器step，恢复模型参数／buffers、CPU/CUDA RNG、模式和sampler状态，清空梯度后立即进行222次正式更新。模拟smoke不增加优化预算。

保留原六类默认头与骨干联合训练，232993参数，原CENTER_UNITS物理优化器冷启动，不继承父moments；lr恒1e-4，weight decay继承已核实父args，betas(.9,.999)、eps1e-8、物理clip5。原native L监督六次完整前向保持，global step11101–11322使跨RX权重恒.1；ordinary增强沿原续段epoch13，As使用epoch51/global step，satellite_concat关闭而原半批As practical混合真实监督保持。原U_s／REPL联动、EMA／teacher训练全部关闭，本增量不称重做完整Tri-SIG-TIME SSL。

L只读取固定split392005的6300合法记录，不重新划分；batch128、222步，共28416采样曝光，170496包前向视图，其中生成14208条As视图。OE行只读已验证source21的6134条public IQ／域元数据；每次先均匀抽15个RX/day域，再域内均匀抽记录，batch64，共14208次曝光。额外目标为`0.1*KL(uniform6 || softmax(原完整六类logits))`，不需要辅助设备逐TX身份，也不将它们伪标为旧六类。OE前向临时eval但启用梯度，随后恢复模式及随机流；不回写BN运行统计。L采样seed2026092701、OE独立seed2026102907；监督对照根本不接OE路径。

固定checkpoint后，每行新前向L6300／V27000／novel3150及两条自有checkpoint L smoke，共36452，不复用旧表示冒充新编码器输出。只用新L特征形成六个原型；五个静态分数为MSP、maxlogit、margin、energy、cosine，不新增拒识读出。新V仅作4%宏FRR／1.5pp自闭集正确样本拒绝损失的最大单全局阈值校准，总体及每RX约束。阈值落盘后才读取novel IQ／旧novel baseline缓存，30150×5预测全部固定后才独立连接私有TX。

原F6在相同V IQ上的冻结预测作为保护基准，并由scorer对照旧缓存logits独立核验。门槛同时要求新闭集宏准确率相对原F损失≤0.5pp、新开放宏准确率相对原F损失≤2pp、FRR≤5%、未知宏拒识≥95%，总体及各源RX均满足。新闭集损伤超限仍完成预测和评分，作为科学失败。预先只允许`uniform_oe::msp`推进；其余四分数和监督行仅作诊断／对照，不挑最高分规则。

两行完整后才判推进并输出MSP配对差。两行可并行，每行同卡依次train→eval，score使用CPU；优先两张不同较空闲卡，仅一卡有容量时可新增2进程，不停止任何他人任务。一行技术失败保留产物并让另一健康行完成，整体PARTIAL不晋级，不自动重启。所有输出新root不可覆盖，代码／配置提交、一次整个候选P0/P1、一次N607 preflight／归档比较／编译与实际PID/CWD/cmdline/GPU/log证据写同名最小登记。

源成员监督仍是额外标签权限；novel3是重复源开发，known是同批经验校准风险，不宣称新盲测。正式RX24 U/Q不读，未执行目标适应、发现或匿名注册。原实现及合成验证77项测试／52subtests通过；r02现已实际完整训练、预测和评分，VERIFIED/ANALYZED_SOURCE_ONLY、NO_PROMOTION，Phase3目标未完成。

发布前状态：一次完整候选独立P0/P1审查PASS，独立运行同77项／52subtests通过。N607一次preflight PASS，原source parent、权重、cache及远端已验证OE6134 public数据在位；新run/release不存在。source_only协议字段在命名调整后的最终集成中已验证，控制行正式名称supervised_only。归档只传代码/配置/runtime，不重复传输数据；实际checkpoint梯度smoke为每row训练首步，r02两行原L2及OE行OE2均PASS，已完成各222次更新与独立评分。

## 首次技术失败与定向恢复

r01已实际发布，两行均在新增父兼容检查处退出，未到梯度smoke、优化或评估；没有新模型性能。原native E50祖先检查已通过，新增代码误把split_seed放在完成config中，并把原cache路径与验证后的复用路径相等当作身份。实际split_seed392005在source contract，两个cache路径不同，物理角色由既有contract和精确storage ID绑定确定。失败进程均已退出、foreignstop0，r01产物保留。

r02仅修这一个兼容层：从真实contract严格核对seed，保留native来源校验和exact role_ids；不要求缓存路径一致，不重读raw或做新hash／数据验证。真实布局回归由旧代码失败到修复后2项／3subtests通过，并完成一次该P1的定向复核PASS，未重复完整77套。两行科学配置、222步、初始权重、数据、种子、阈值和95%门槛均不变。技术恢复使用独立不可覆盖run/release `20261029-phase3-source-embedding-increment-s2026092701-r02`，新release一次preflight已PASS，现已通过真实checkpoint梯度smoke并完整执行同矩阵，VERIFIED/ANALYZED_SOURCE_ONLY、NO_PROMOTION。

## r02 实际完成（2026-10-29）

执行commit28d53ea9b38c19f642c59e31f178a3f65f40894d，两行各222/累计11322步，真实L2/OE2梯度及最终checkpoint smoke PASS；GPU2监督、GPU1 OE，foreignstop/target/U训练0。固定MSP监督未知47.8095%、OE59.9683%（配对+12.1587pp），OE closed98.4963%、open97.9593%、FRR1.5185%、对原F总损伤0.5074pp。总体及每RX已知保护全部通过，但unknown全部RX未95，eligible空、NO_PROMOTION；其他4分数不替换规则。完整两trace、两行评分与10份新旧日志见[实际结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_EMBEDDING_INCREMENT_RESULTS_20261029.md)。本轮不调参重跑，下一组合读出候选尚未实施。
