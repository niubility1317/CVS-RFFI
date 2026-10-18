# LRaw 固定配方总 E50 匹配延长预登记

状态：PLANNED；实现、唯一独立 P0/P1、代码提交及发布待完成。E50 结果尚未产生。

唯一假设：已由固定源域两项判据支持的 LRaw30 配方，在不改参数、数据和视图组织的前提下追加20轮，能否改善 practical，同时保留相对同预算匹配NoAs的clean（下降不超过0.1个绝对百分点）。这只是总预算验证，不新增损失、门控或网络。

## 晋级依据与冻结时间

父run 20261018-phase1-FcrossRX-LRaw-e30-s2026092701-m2-r01；代码 d14450bf1b0d12cc281df1f4d0e79e8760e48282。主线程先保存 source_decision.json（source archive 2188eb7864e49d75940239842b4ae463e1c9af7c3c059aedf9002d40e96307f2），newAs−newNoAs fullVclean−0.02962963pp、固定V1800两draw mid/low均值newAs−originalAs＋1.13888889pp，预登记两规则均PASS。随后登记本E50矩阵，再读取父E30 formal。E50资格和配方不依据该formal修改，不是目标分数选择性救跑。既有benchmark已暴露的背景保留，不声称新独立盲测。

## 矩阵与继承

仅两条新训练：B00-s2026092701 / B01-s2026092701，分别对应 LRAW_NOAS50 / LRAW_AS50。各严格从自己LRaw FINAL30学生和ownEMA追加20/4440更新；累计20＋10＋20=E50/11100更新，不是额外50。默认cos6×160 scale30、SIG_TIME三路统一Sinc/joint160、CROSS_RX sameTX、RDrop及正常联合训练保持。严格核对parent30 run/row/累计计数/源角色/无目标训练选模。

旧原NOAS50/AS50复用run 20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01 两B00/B01完整预测与评分；实际冻结发布commit0f399c910324e21a8b1a60346d879be5c2964a9c，非后续报告commit115dc。它们也是own20＋10＋20。旧原NOAS30/AS30、LRaw30两条件用于30→50预算配对差，均复用，无重新训练/预测/评分。8个逻辑条件中本次只新增2条。

## 固定机制

前20轮原ordinary，后30轮L ordinary返回调用前raw。原ordinary仍在独立clone上完整执行一次，弃用输出，保留这一局部call的随机消耗/计算控制；不声称整个训练随机轨迹相同，不声称省去增强计算。返回precallraw，连同ordinary归一化尾部旁路，无新RMS/crop/shift/prefix。U weak/strong、教师和EMA保留原P43，不包装U。legacy ORD128 anchor实际raw128，层位/温度/系数保持。原As本就practical(raw64)，不称修复重复星地损伤。

原6L=768L样本NN/step、Lscale1.25=.5RDprimary+.5RDmixed+.25RDraw、As64/step保持；newAs cleanraw1.0/As.25，newNoAs raw1.25。三场景IID完整practical、传播/接收噪声/质量/lock/tracking/AGC/内部补偿依赖保持，NoExplicitCFO/原noEQ/refSNR30保持。未改变标签或PA/DAC/classsignature/CFO。

新20轮按原E50匹配规则重置AdamW lr2e−4 WD1e−4、cos20/min1e−6；不是恢复原optimizer连续50轮。channel epochoffset30/stepoffset6660，SupCon累计+30。discardedordinary保持原own20 args/frozen13，不因父30 args改变旧增强算法。

## 数据、冻结与评分

源RX[1,3,4,6,8]/day[1,2,3]，full L6300/U56700隐藏TX/V27000，split392005、eq1、IQ256、Fs25MHz。无训练RX留出或L5580/Ldev，复用VALIDATED_ONCE载体和原role_ids，不做重复数据重验。U累计7＋10＋20=37原U轮，不读取TX真值。

固定FINAL50，不挑epoch。本次source50只如实描述固定预算变化，不新增晋级或关闭阈值，不额外扩seed/预算。源域检查仅真实标签/数据/数值/执行故障。每row冻结后自动source及clean＋六practical预测，七视图prediction固定后独立exec truth-side scorer才连接truth。全部负结果保留，逐row完成不等待另一健康行。主线程先收源汇总再收formal，formal不回流。

固定测试每视图168000；RX[0,2,5,7,9,10,11]/days[0,1,2,3]；clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。receivedIQ/测试SNR/信道/困难包均保持。

## 指标与成本

逐seed（本次n1）报告all7 accuracy/macroF1/最差RX，重点low/midurban、urban3均值、RX/day/TX分层、修复/新增错误/净变化、suburban绝对值和自身clean差；同时对newNoAs50、oldAs50及对应30报告配对差/交互。阶段目标clean newAs50−newNoAs50≥−.1绝对pp；不把弱NoAs对照下降写成机制clean获益。n1不宣称可重复双提升，<.1pp微增只作观察。

本次新增40modelE/8880updates；每row预期LNN3409920、U学生1134000、U教师eval1134000；nativeeval及source12600/formal1176002按实际记录。每row568320 discardedordinary输出，As行284160 As包。clone/增强/物理信道/日志CPU时间不为0。父30、旧50和各旧readout分别列复用成本，不与新计费混合。首步真实双父4NN＋两actualfactory6L1536=1540NN，syntheticordinary生成512包另报，无真实IQ/query/更新；PASS直接train。

## 发布与停止规则

代码/config身份为一次Git commit，代码冻结后回填，无额外SHA/签章链。一个run/outputroot与owner codex/root/practical-LRaw50；N607普通账号、现有CVS-RFFI/PyTorch fullFP32、CPU2，无安装/管理员操作。CWD=PROJECT/releases/cvs_FcrossRX_LRaw_E50_20261018_r01；输出=PROJECT/runs/20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01；local报告=automation_reports/CV-SincNet/20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01。

发布命令：verified native Python -m experiments.cvs_lraw_tail50.publish；worker：CVS-RFFI/python -u experiments/cvs_lraw_tail50/runner.py --mode worker --row {B00|B01}-s2026092701。private runtime保留P43 reference_release/reference_commit，并显式parent_release=LRaw30冻结release/parent_commit=d14450bf1b0d12cc281df1f4d0e79e8760e48282，不把liveworktree当依赖。

动态GPU容量：每GPU所有owner compute进程≤3、free≥12000MB；不动健康任务。一个N607preflight、一releasearchive本地/远端SHA对比、remotecompile、launcherfirst实际CPsmoke、一次PID/CWD/argv/GPU/log增长核验。唯一阻止真实run的是协议/角色/父模型/数值/输出/预测关闭的系统性技术失败；仅停止明确own PID并保留产物，无自动换root/重试。科学负结果不停止formal。

预期产物：resolved_config/actual_native_args、initialization/EMA/optimizer/source_contract、final_ssdg/checkpoint/source_frozen/completion、reference/unit/shape/lraw/ordinary_draw两完整日志流、source_qualification、每row7prediction/scoring/complete；0NNanalysis source_summary/training_logs/formal_summary/costs/complete，完整报告与登记。分析仅新4440全日志，旧audit复用；8×7重计预期1008CM/168partition。source/final按5/11文件单archive取回。

不得把PLANNED、smokePASS、sourcePASS或训练完成写成正式性能成功。


## 实际启动

代码 1cf3bf1c9232110b4199e01628b682b74a47d3a9；transfer/remotecompile VERIFIED，双真实父4NN+actualfactory1536NN首步PASS，syntheticordinary生成512包、0 realIQ/query/更新。dispatcher1465885，双worker1465929/1466063在GPU0/1，0NNanalysis1466559 WAIT_SOURCE。PID/CWD/argv/GPU/loggrowth、ownLRaw30CP/EMA/数据角色/实际20U轮开关均VERIFIED。私有观察器沿用旧脚本名的期望已改正，实际任务未改动。当前无E50正式结果。
