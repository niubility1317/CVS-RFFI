# 真实同 TX 观测配对：源域预登记

REGISTERED_NOT_RUN。实际矩阵／commit／command／environment／CWD／paths／GPU／owner／stop／artifacts见experiment.json。

# 真实同 TX 观测配对：独立源域检验

当前唯一假设：固定 F＋跨 RX 同 TX 对比学习后，保持一半原样本监督，将另一半交给同 TX、同 day、不同源 RX 的真实有标签 IQ，能否比原 C0 和同 RX 独立观测都更好地识别未见源 RX。这个候选研究真实观测的训练组织，尚不称为已成立的 A_g；它不是对同一物理样本做数字扰动，不是接收机硬件替换、TX 反演、波形混合或新的特征对齐。

依据：原点抖动 OJ24、WHITE30、幅度响应 EMP/ROT、相位统计、NoPA 等已按各自规则失败，不重开强度或概率扫描。普通 practical 在当前底座首源折明显提高 urban，并保留 clean，但 low urban 仍约38%，BLOCKED/unlocked改善小，且 TX2 有负收益；冻结控制不支持仅增加某一种噪声／多径约束即可恢复 clean。新方法不再假定 PA、IQ 失衡、相位或量化变化能保持身份，而使用已有真实 L 标签保证配对观测属于同一 TX。不能从这个标签事实推断训练一定有泛化收益。

最小新矩阵：`SAME_RX` 和 `CROSS_RX` 两个 scratch E20，model seed2026092701。结果对照复用已完成当前 C0 的报告与完整原 L/U 顺序，不加载其任何权重、EMA、optimizer 或统计。C0来源为 `20261016-phase1-FcrossRX-practical-source-RX1-e20-s2026092701-m2-r01/C0-RX1-s2026092701/source`，其训练代码8235c71cc、fit RX3/4/6/8 L5040／隐藏U45360／V21600、FINAL20／4440更新，来源在新训练开始前核对。新模型不从 E50/E200 或失败候选继承状态。

两新行第一视图完全相同：原 native L128 和原 ordinary private stream。第二视图从同一个当前 filtered train_loader.dataset 的原 L5040 中选取同 TX／同 day、不同物理样本，SAME_RX 保持原 RX，CROSS_RX 在另外三个 fit RX 中均匀取一个。每个 TX×RX×day cell 原有70包；SAME_RX 排除本包后69包，CROSS_RX 原cell70包，均匀采样。每条 partner 保留其自身实际 RX/day domain，全部使用真实 L 标签。不能用 U 伪标签配对，不读 U TX、不用 V 或 held RX1 L/U。source bank 只在自身 scratch checkpoint smoke 通过后构造，5040 个原 IQ 读取一次，保留 native 原 packet RMS，不另归一化、改写、拼接或合成信号。

第二观测复用相同旧 ordinary 配方，用独立流20261016712；partner采样流20261016711，每更新刷新，二者不推进 primary／loader／dropout／U 随机流。TX只用于合法样本分组，不定义人工硬件参数或随机session。两行同样每步两次 ordinary 生成；C0原本只生成一次再复用，故CROSS_RX−C0包含独立观测和独立ordinary的共同作用。CROSS_RX−SAME_RX才检验跨 RX 组织的独立增量，不能省去这一控制或事后改主对照。

每物理槽位目标为 `.5 RD(primary ordinary)+.5 RD(partner ordinary)`，每个RD内部严格复用同一个已生成IQ做两次dropout前向，原LS=.01／RDrop alpha1不改。没有两真实观测之间的 KL／间隔／embedding 对齐。跨 RX SupCon仍只复用第一对原RDrop的 time_down／t2.pool，温度.2、上限.1、前5轮0随后10轮增加；第二对不加新特征约束。默认joint160/cos30、三路SIG_TIME、CENTER_25K／CENTER_UNITS和原物理梯度clip继续联合更新；NoExplicitCFO、原U/ownEMA/旧门控/cos保持。

新增每模型总 E20=13L+7U，4440更新、2273280 L网络前向；每primary L568320次观测各配一个partner，总计568320次secondary物理曝光。每行可见的独立有标签ID仍只有5040，不能说标签数量翻倍，也不能把两新行看成两个模型seed。两个新模型共40模型轮／8880更新；C0历史E20另列。额外5040 bank读取及568320 partner视图构造的墙钟单列，后者包括采样、原ordinary、设备传输、波形统计及日志／cache，不含骨干前向；原U／EMA／source eval／heldV／六practical网络费用按实际计数。四网络前向与C0匹配，但独立物理曝光故意不同并明确披露。

固定源RX1留出，heldV5400仅FINAL；seenV21600只健康诊断。每新行自动完成clean和六原practical的源V预测，独立结果统计；不构造正式target。正式测试IQ、reference30、25MHz／IQ256、residual/post_sync／EQoff G1／reflect及原接收质量、相位跟踪、IQ补偿均不改。As在本训练为0，不能将源practical评价当As训练、正式性能或联合交互证据。

事前放弃条件：CROSS_RX相对C0的held clean accuracy至少+1pp、相对SAME_RX至少+.5pp，且对两个对照macro-F1均不降、seenV损失均≤.5pp，全部更新有限、原始L/U全5994批顺序与C0及另一新行一致、全合法pair cell／5040 partner覆盖。任一失败关闭“跨RX真实配对具有独立泛化价值”的本固定假设，不调比例、换源RX／seed或把SAME_RX事后改为获胜候选。低分不停止健康行。通过只允许另登记剩余源折确认，不自动进入target或Ag+As组合。五源折不是五模型seed；后续仍需匹配四因子及交互、正式clean＋六practical truth-last和跨seed重现。

运行矩阵／命令／环境／CWD／路径／GPU／owner／stop rule／预期artifact以新run experiment.json为准。新输出根不可覆盖；共两个独占worker，所有owner/preCUDA每GPU≤3且free≥12000MiB。只停止自身协议／来源／非有限数／错误预算／缓存或输出技术错误，不自动重试、不动健康五源折确认或其他任务。原source数据VALIDATED_ONCE复用，不引入新校验链或审批。

本地1024合成网络包／2次optimizer、1024 partner观测／1280 ordinary IQ、14协议负例、206初态、首两步primary逐值匹配、原默认头／SIG3／Sinc梯度、ownEMA与四前向均PASS。隔离发布导入／两行实际E20参数／真实C0元数据及一次污染拒绝PASS。唯一P0/P1发现raw domain未按native map转换，已修复；5040合成bank条目与实际native domain_from_extra逐域相同、128个额外partner及空map／identity map两负例PASS，0额外网络／更新。累计partner1152，0真实IQ／CP。原审查者仅对此P1做一次定向读回PASS。N607普通账户preflight于2026-10-16 15:19:03 CST核实一次。尚未启动，无实际新训练成绩。

## 实际启动

VERIFIED / RUNNING，实际代码 `9b2eebcfa809c8d5cfee6523d35b84aa10ded1a4`。两行自身scratch CP no-query smoke、source L5040 bank、raw→compact domain、实际E20/13+7与原U、PID/CWD/argv/GPU/log增长、默认头／三个SIG／Sinc更新、原物理clip、primary/secondary真实覆盖已独立读回。单archive传输对照／remotecompile通过。旧ordinary包含的原物理处理保持共有条件，配对算子本身不新增硬件合成。不改其他健康确认或原practical测试，尚无新FINAL性能。

独立离线读取器已实现，仅读取已保存的完整source预测、日志、F/SupCon轨迹和真实pair记录，不导入训练／IQ／checkpoint。完成后将逐条复核三行primary L/U和每个partner的合法ID、同TX/day、RX关系、所有cell/5040 ID覆盖，独立重算两主比较、F1、seenV及默认头分层。13个纯元数据／指标负例及有效same/cross和资格fixture通过，0网络／IQ／CP；未完成的行在读取前拒绝。运行release仍为9b2eebcfa，未热改训练。
