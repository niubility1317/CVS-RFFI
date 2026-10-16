# F＋跨 RX 同 TX：同抽样普通 PA 旁路 E20 源对照

状态：ANALYZED_SOURCE_ONLY／VERIFIED。两组 scratch E20、全部七个源视图及独立完整分析已完成。固定 PA 旁路候选未通过源条件，关闭固定配方。当前固定基线F＋CROSS_RX，E20从零，默认cosine头正常联合训练；不把源诊断或RDrop/SupCon当作IQ泛化完成。

## 唯一假设及证据边界

唯一待检验假设：只旁路普通有标签增强的PA输出，保留其余同次原生抽样和完整PA计算，能改善训练后未见RX clean，并保护practical。已有冻结fitL5040同抽样诊断直接请求20轮日程，实际变动3054观测修复640、新错27，支持存在可避免源身份损伤；不证明训练收益。实际native trainer把增强epoch截为min(training_epoch,label_epochs=13)。前探针E14至20直接函数调用属于更强诊断外推，已在原报告及日程更正文档明确，保留全部结果及原判定。前13轮描述性重算1475变动、254修复、7新错，净恢复16.745763pp；没有新增IQ或重跑。此次训练保持原生截断，PA概率最高0.035084047，未选择更强日程。

## 固定最小矩阵

| row | ordinary送入模型 | raw/S槽 | 预算 | 训练后自动读出 |
|---|---|---|---|---|
| B00-RX1-s2026092701 | 原native完整PA输出 | 两槽均raw | scratchE20/4440更新 | heldRX1 clean＋六practical |
| B10-RX1-s2026092701 | 同次native调用PA前prefix | 两槽均raw | scratchE20/4440更新 | 相同物理样本和信道抽样 |

两组真实原L5040/U45360/V21600契约相同，fitRX3/4/6/8×day1/2/3×全部6TX；原RX1 L1260只在fixedfinal20后读出。U真值隐藏，原13L＋7U oldgate/自有EMA流程保持，V只健康诊断不选checkpoint。不能冒充完整L6300或L5580/Ldev720；不调用target，source-only预登记结束于源证伪。复用现有carrier/角色过滤，不重复数据验证。

两组同模型seed2026092701、split392005，原训练增强全局随机流程、source practical信道/receiver/eval seeds均不变。只用不变backend0ae代码，正式训练不加载任何父模型。dispatcher真实checkpoint smoke仅使用自有匹配scratchE20/fixedfinal20/B00做2个合成包无query兼容检查，PASS直接继续；来源无祖先/目标选模，真实训练另从零开始。不继承E50/E200或诊断模型。

F三路SIG_TIME、统一完整Sinc修正、joint160、NoExplicitCFO与必要逐包补偿、默认cosine头和原联合训练保留。仍是三对共六L前向，.5raw＋.25ordinary＋.25raw，每对RDrop alpha1/LS.01，浅中层time_down/t2.pool CROSS_RX只取ordinary原两次前向，温度.2/最大权重.1/5轮warm＋10轮ramp不变；同TX同RX不作负对，U没有新对比或新增视图。

不改变任何概率或算子范围。native只调用一次ordinary augmentor，捕获post-general-channel的PA前prefix，原生PA仍完整计算。B00返回full，B10返回prefix；两组均记录全PA生成及捕获成本，保留原native Python/Torch随机消耗。DACp0、class_signature=false，PA选中子链仍按原抽样，未触发包full/prefix逐位相同。native n.configure按min(epoch,13)配置；记录真实training_epoch/configure_epoch/状态和每step有效PA概率，不能只凭默认值称启用。无新增RMS、参考窗移动、CFO身份输入、噪声/多径级联或As训练。新变体是旧ordinary配方消融，不直接冒称新增A_g已有效。

## 原定数值判断与成本

fixedfinal20后比较B10−B00：heldRX1 clean准确率≥+1pp，clean MacroF1≥0，六个practical各≥−0.5pp，seenV≥−0.5pp；两组完整20轮/4440更新/3409920LNN、3330非零SupCon调用、最终LCE≤.5、0As/头拟合/target。全部同时通过才支持下一独立源确认。任一失败关闭固定PA-bypass训练配方，不改seed/RX/强度/阈值救跑。源确认不等于正式target或多seed泛化，原联合目标及B00/B10/B01/B11矩阵仍未完成；不把clean降低造成gap变小当改善。

每组E20＝13L＋7U、4440更新；每组3409920L网络包前向，另实际U/ownEMA/V/2合成smoke/8820held；此前同配方参照每组4686678NN，当前按实际计数，不用预期冒充实际。每组568320个ordinary完整PA生成及prefix捕获，NN视图数不增加，PA实际触发/变动单列；0As训练生成/0诊断头拟合。两个模型40模型轮/8880更新；dispatcher真实CP smoke2另列。局部验证13056合成增强包、2312NN、2合成更新、0真实IQ，与正式预算分开。

## 发布与停止

唯一launch owner codex/root/F-CROSS_RX-PAbypass-E20-20261016；普通N607账号，CPUdispatcher2线程，GPU动态共享容量锁、每卡最多3个所有owner进程且启动前≥12000MiB空闲。健康旧任务不改/不停/不重启。CWD为PROJECT/releases/cvs_F_crossRX_PAbypass_sourceRX1_e20_20261016_r01，命令CVS-RFFI/bin/python -u experiments/cvs_pa_bypass20/runner.py，worker带固定--worker row_id。路径、六类seed、逐行完整config见experiment.json；Git实际commit及release_commit/resolved是代码身份。

一个不可覆盖run/output root、一个N607 preflight、一个归档单次SHA比对和远端compile；launcher第一步真实CP no-query smoke，PASS继续，不另建许可。启动后独立核实PID/CWD/argv/GPU/log增长。角色/初始化/预算/非有限/输出冲突/不可执行或不能合法完整预测为技术失败：保留产物、停pending、不干预健康worker、无自动重试。科学低分不停止、改配方或追加预算。

每row final20后自动完成clean及六practical源默认头读出；全部prediction固定后本地独立复算完整epoch/step/PA记录、实际日程和NN成本、物理L/U序列、accuracy/F1/CM/CE、配对修复/新错和TX/day分层及原判断。IQ/CP/逐包预测与完整日志私有，只代码/config/登记/汇总入Git。一次限定P0/P1审查及14个负例、原生输出/随机流精确匹配、同初始化/B00损失和头梯度精确对照、六前向同IQ/SupCon域、默认头联合更新和export160检查通过；这些不是识别收益。


## 实际发布与启动读回

执行commit `bdb3733de36b2d0e01515c91b7c9df8462b8c39b`，本地与远端分支OID已独立一致。归档单次比对、远端编译、匹配真实CP smoke PASS；dispatcher和两worker的PID/CWD/argv/GPU及日志增长VERIFIED。两行实际seed均2026092701，epochs/label/pseudo=20/13/7，source fold RX1，无baseline/teacher权重，实际PA旁路/Ag/As标志与登记一致。

- B00-RX1-s2026092701：GPU0，已观察完整step532/4440、E3/20；stdout16814字节。此为运行进度，不是最终性能。
- B10-RX1-s2026092701：GPU1，已观察完整step465/4440、E3/20；stdout16815字节。此为运行进度，不是最终性能。

独立全量分析程序已准备并编译，只有全部声明预测完成后才执行。不修改已发布训练代码，不重复提交同run，不基于中间或目标分数改变配方。联合目标仍未完成。

独立评分边界：当前继承源读出保存hard predictions及运行时mean CE，没有logits。完成后独立复算accuracy/F1/CM/repair-break和全部训练/PA成本；mean CE仅引用冻结前向的记录，不声称从hard predictions独立重算CE。该限制不改变预登记识别准确率/F1及practical保护判断。


## 完整源结果与原定判断

两worker和dispatcher自然退出、completion齐全、无failure；无重启/热改/选择性重跑。每组E20/4440更新/3409920LNN、3330非零SupCon调用/13320层位前向项，全部17640个默认头源决策、完整epoch/step、两组物理L/U序列、PA抽样和实际13截断日程独立复算。mean CE引用冻结运行记录，hard predictions无logits，未冒称独立重算CE。

| 固定 held RX1 源视图 | B00 准确率 | B10 准确率 | 差值 | B00 Macro-F1 | B10 Macro-F1 | 修复/新错 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 81.507937% | 80.238095% | -1.269841pp | 81.132673% | 79.887119% | 2/18 |
| practical_high | 37.857143% | 39.444444% | +1.587302pp | 27.877176% | 28.453225% | 45/25 |
| practical_mid | 38.095238% | 39.603175% | +1.507937pp | 29.135008% | 29.039771% | 45/26 |
| practical_low_suburban | 35.000000% | 36.031746% | +1.031746pp | 27.678789% | 27.401562% | 37/24 |
| practical_high_urban | 37.698413% | 39.285714% | +1.587302pp | 29.105684% | 29.133153% | 44/24 |
| practical_mid_urban | 31.190476% | 32.063492% | +0.873016pp | 26.168089% | 26.193247% | 34/23 |
| practical_low_urban | 27.857143% | 29.285714% | +1.428571pp | 24.115680% | 25.243003% | 39/21 |

| suburban 与自身 clean 绝对差 | B00 | B10 |
|---|---:|---:|
| practical_high | 43.650794pp | 40.793651pp |
| practical_mid | 43.412698pp | 40.634921pp |
| practical_low_suburban | 46.507937pp | 44.206349pp |

原预登记条件：{"healthy": true, "clean": false, "clean_macroF1": false, "practical_protection": true, "seen_V": true}。全部同时通过才晋级；六practical每场景≥−.5pp属于原保护条件，不能事后删去urban或suburban失败项。

- B00：最终LCE 0.277130948，seenV 98.041667%；实际NN 4686678，墙钟1267.253秒，完整stdout 800行。完整ordinary管线覆盖568320包、调用4440，真实触发15144、IQ改变15124，旁路返回0；PA捕获和实际ordinary输入MSE重算最大差2.41212547e-07。
- B10：最终LCE 0.251672544，seenV 98.050926%；实际NN 4686678，墙钟1231.857秒，完整stdout 800行。完整ordinary管线覆盖568320包、调用4440，真实触发15144、IQ改变15124，旁路返回568320；PA捕获和实际ordinary输入MSE重算最大差2.29105353e-07。

两组PA mask/subchain/strength及有效概率/shift逐step匹配：{"triggered_bits": true, "changed_bits": true, "subchains": true, "strength_max": true, "effective_p_pa": true, "effective_max_time_shift": true}；U TX真值可用和非有限跳步均0。所有同IQ RDrop对、六L前向和原损失尺度逐step复算；0As训练、0头拟合，未新增部署分支或加载父权重。

本实验40模型轮/8880更新，含dispatcher真实CP smoke2的总实际NN包前向9373358；原L、U学生/教师、V/学生评估和held8820分列，EMA继承计数2已扣除。局部合成13056增强/2312NN/2更新和先前源诊断206642NN/0更新另列，不并入真实E20。source-only，无target预测或调参，联合目标未完成。

本次否定固定PA旁路配方的训练晋级依据；源冻结诊断的修复并未保留为全部训练收益。关闭此配方，不换seed/RX/强度/阈值或追加预算救跑，不扩seed。suburban与clean差距缩小同时包含clean下降的贡献，不能记为联合成功；practical平均准确率的正差亦不能覆盖类别负结果。

| 分层 | clean | high sub | mid sub | low sub | high urban | mid urban | low urban |
|---|---:|---:|---:|---:|---:|---:|---:|
| TX0 | -1.904762pp | +0.000000pp | +0.000000pp | +0.476190pp | -0.952381pp | -0.952381pp | +2.380952pp |
| TX1 | -0.476190pp | +0.000000pp | +0.000000pp | -0.952381pp | -0.952381pp | -0.476190pp | +1.904762pp |
| TX2 | -1.904762pp | +19.047619pp | +16.190476pp | +14.761905pp | +19.047619pp | +11.428571pp | +9.523810pp |
| TX3 | +0.000000pp | -3.809524pp | -0.476190pp | -1.428571pp | -1.904762pp | -3.333333pp | -1.428571pp |
| TX4 | +0.000000pp | -1.428571pp | -0.952381pp | -0.952381pp | -0.952381pp | +0.952381pp | -1.428571pp |
| TX5 | -3.333333pp | -4.285714pp | -5.714286pp | -5.714286pp | -4.761905pp | -2.380952pp | -2.380952pp |
| day1 | -1.428571pp | +1.904762pp | +2.619048pp | +3.333333pp | +2.619048pp | -0.476190pp | +1.666667pp |
| day2 | -0.714286pp | +0.238095pp | -1.666667pp | -0.952381pp | +0.476190pp | +0.714286pp | +1.428571pp |
| day3 | -1.666667pp | +2.619048pp | +3.571429pp | +0.714286pp | +1.666667pp | +2.380952pp | +1.190476pp |

下一唯一拟检验：温和的包内常数观测增益视图x→a·x，是否在保留全部旧ordinary及原U的条件下，带来未见RX clean净收益。a先固定在已核实旧增益范围0.9至1.1，不从本次practical或target分数选幅度；它是线性缩放已接收IQ，不等于改变TX PA或替换真实RX。此方向区别于已失败的包内缓变斜坡，也不删除PA。先核对实际输入/后续RMS归一化是否将其消除；若只成identity，不启动训练。最小训练仍应为固定源折/同seed/同六前向的scratchE20 B00/B10，原raw.5/ordinary.25份额保持，第三槽raw或独立轻增益raw；在任何新增源计算前独立登记。若clean无预定净收益或practical保护失败即关闭，不靠源fit敏感性晋级。当前仅提出，尚未实现、登记或启动。

成本用语：PA_native_L_packets=568320统计有资格进入原生PA门控的ordinary L包数，不表示每包都执行非线性子链。两组真正选中PA的包数均为15144，其中15124包IQ实际改变；PA子链调用按私有PA_steps逐项保存。两组门控/子链和完整生成工作相同，无新增网络前向。
