# P64 结题：satellite CE 温度与减半权重的 E30 匹配对照

本轮已 ANALYZED：两条新增训练、逐行 clean＋六种 practical 预测及独立 truth-last 评分、完整日志分析和 5 组 PNG/PDF 均完成，无缺失产物。代码 90fe8ac0e2577ce0038a8e5a7855a4d2d9c5efc6；起点 0bcfc6f3962285afd23e29cdac150167799a8b99。CVS、内部门控与默认 cosine30 头保持原联合训练。

**主要判断：两种新目标均未超过原 As30。** HALF 改善 clean，却削弱六种 practical；T2 追回部分 HALF 损失，仍弱于原 As30。源域四项判据已在负责人读取正式分数前固定为 FAIL，候选关闭，不追加 E50、扩种子或扫描温度。此前源域表现好的配方可按其独立计划总计 E50，本轮不修改其他任务。

唯一假设是 satellite CE 的概率形状是否比统一减弱 CE 更有效。合法源 logits 不支持“urban 错误主要是高置信错判”：mid/low 错样本 pmax 中位数约 .376/.359，pmax≥.95 仅 .306%/.204%；T2 错样本的 logit 梯度与半 CE 几乎同向。温度 2 在描述统计前已固定，没有扫描。这是 logit 导数，不是实测骨干梯度。

| 固定配置 | 实际执行 |
|---|---|
| 矩阵 | CE_HALF=.5CE(z,y)；CE_T2=CE(z/2,y)，无T²；原smoothing=.01 |
| 初始化/预算 | 同一 P43 B01 scratchAs20 student＋ownEMA，严格加载，分别重置 AdamW/cos10/cache 追加E10；每行累计E30/6660更新；原As30/NoAs30复用 |
| 数据 | L6300/U56700 TX隐藏/V27000，RX1/3/4/6/8、day1/2/3、sixTX，split392005；无留出训练RX/Ldev |
| 原训练 | sixL128，共768 labeled NN/step；ordinary SupCon .1、原RDrop对称KL、raw clean槽及native U/ownEMA保持；全部As64监督 |
| 精确替换 | .25×(两次新CE64均值−两次原CE64均值)，各自算两次CE，不先平均logits；非As CE与全部KL不变 |
| 尺度解释 | 1.25为原槽权重，HALF有效CE系数和1.125、KL不变；两arm只匹配显式链式1/2因子，不匹配实际参数梯度范数 |
| 实际信道 | 导入adv3b02_xuc/code/leo_practical；25MHz/IQ256/RMS center=false；residual/post_sync/EQfalse⇒G=1；接收质量适配、相位跟踪、IQ补偿开启；ref30dB、reflect；训练fc2.462GHz，正式2.45GHz保持 |
| 训练场景 | lowSub/midUrban/lowUrban各1/3；完整物理与接收依赖保留，不在As前叠普通PA/噪声 |

相同model seed2026092701、数据角色、更新数、前向数和命名随机流。每条与原As30的4440条L/U序列及逐步ordinary MSE指纹均无差异；未直接保存比较全部IQ字节或完整RNG状态。没有新增参数、部署模块、分类头、教师或诊断头。

## 全部正式结果

每格为准确率／宏平均F1／最差RX准确率，单位%。每视图168000个固定物理样本。最终truth连接前验证全部四行七视图；504次混淆矩阵重数和84组分区求和一致。

| 视图 | NoAs30 | 原As30 | HALF | T2 |
|---|---:|---:|---:|---:|
| clean | 85.247619／84.979636／78.450000 | 85.341071／85.044474／78.362500 | 85.808333／85.533163／78.512500 | 85.388095／85.095240／78.062500 |
| practical_high | 42.685119／39.311416／34.304167 | 81.405357／81.084230／71.908333 | 80.770238／80.440633／69.816667 | 81.094048／80.682125／71.145833 |
| practical_mid | 40.995238／37.954813／33.520833 | 78.661310／78.359706／70.212500 | 77.885714／77.573271／68.129167 | 78.214881／77.829851／69.262500 |
| practical_low_suburban | 39.472619／36.758731／32.054167 | 75.861905／75.572741／67.741667 | 75.007143／74.710396／65.729167 | 75.521429／75.156042／67.133333 |
| practical_high_urban | 41.439881／38.481030／34.000000 | 77.767262／77.423208／69.108333 | 77.161310／76.791902／67.216667 | 77.377381／76.952679／68.066667 |
| practical_mid_urban | 35.132738／33.446603／29.495833 | 62.806548／62.450463／56.458333 | 61.823214／61.443702／54.395833 | 62.141071／61.754747／55.479167 |
| practical_low_urban | 30.139881／29.184589／25.750000 | 51.244048／50.887035／46.379167 | 50.150595／49.851231／44.733333 | 50.565476／50.248062／45.675000 |
| urban三场景准确率均值 | 35.570833 | 63.939286 | 63.045040 | 63.361310 |

两新模型clean最差RX均为1-1，六种practical均为14-7。相对原As30，两新模型mid/low urban全部7个RX、4个日期均下降。整体clean提升不代表每层保持：相对NoAs30，两者仍各有4/7个RX的clean下降。

| 对比 | clean Δpp | mid urban Δpp | low urban Δpp | urban均值 Δpp |
|---|---:|---:|---:|---:|
| CE_HALF−AS30 | +0.467262 | -0.983333 | -1.093452 | -0.894246 |
| CE_T2−AS30 | +0.047024 | -0.665476 | -0.678571 | -0.577976 |
| CE_HALF−NOAS30 | +0.560714 | +26.690476 | +20.010714 | +27.474206 |
| CE_T2−NOAS30 | +0.140476 | +27.008333 | +20.425595 | +27.790476 |
| CE_T2−CE_HALF | -0.420238 | +0.317857 | +0.414881 | +0.316270 |

HALF与T2相对匹配NoAs30的正式整体clean分别增加0.560714、0.140476个百分点，均满足不下降超过0.1个百分点的阶段约束。相对原As30，T2 clean仅增加0.047024个百分点，urban均值下降0.577976；HALF以urban均值下降0.894246换得clean增加0.467262。只有一个model seed，没有配对跨seed均值/波动可报告，不宣称可重复双提升。low urban为50.150595%／50.565476%，未达60%。

## 修复与新增错误

每格为修复／新增错误，固定168000样本；五组比较全部保留。

| 视图 | HALF−As | T2−As | HALF−NoAs | T2−NoAs | T2−HALF |
|---|---:|---:|---:|---:|---:|
| clean | 1071／286 | 947／868 | 4171／3229 | 3985／3749 | 429／1135 |
| practical_high | 898／1965 | 1279／1802 | 67394／3411 | 68110／3583 | 2021／1477 |
| practical_mid | 1012／2315 | 1419／2169 | 65944／3968 | 66790／4261 | 2300／1747 |
| practical_low_suburban | 1092／2528 | 1645／2217 | 64066／4368 | 65230／4668 | 2640／1776 |
| practical_high_urban | 1033／2051 | 1412／2067 | 64012／4000 | 64666／4291 | 2049／1686 |
| practical_mid_urban | 1505／3157 | 2041／3159 | 51342／6502 | 52346／6972 | 2681／2147 |
| practical_low_urban | 1864／3701 | 2442／3582 | 42036／8418 | 43256／8941 | 3052／2355 |

## 源域判定与退化分层

固定源条件：T2完整V clean−NoAs≥−.1pp；panel mid/low−As≥+.5pp；low−As≥0；mid/low−HALF≥+.5pp。四项全部FAIL：−.148148／−.805556／−1.138889／+.375000pp。HALF仅作归因控制。该决定保存于source_decision.json，正式分数没有回流。

源panel是同一1800合法V物理包×2draw；每practical3600观测，draw不是模型seed。完整V27000与panel不能混用。

| 指标% | NoAs30 | 原As30 | HALF | T2 |
|---|---:|---:|---:|---:|
| full V clean | 98.359259 | 98.255556 | 98.262963 | 98.211111 |
| panel practical_low_suburban | 49.861111 | 88.666667 | 87.722222 | 88.111111 |
| panel practical_mid_urban | 43.555556 | 72.805556 | 72.000000 | 72.333333 |
| panel practical_low_urban | 35.611111 | 59.083333 | 57.527778 | 57.944444 |

相对原As30，源mid全体T2净损失17包（32修复／49新增），low净损失41包（31／72）；HALF为mid23／52、low25／81。T2唯一正state组合是mid PARTIAL/locked，n306净修复1包（3／2，+.326797pp），不足以认定部分遮挡得到缓解。low BLOCKED/degraded n414，T2下降2.898551pp（3／15），比HALF还差0.724638pp。两个draw的mid/low变化均负。

T2相对As在mid/low所有新增噪声比区间均下降：

| 新增噪声比dB | mid n／Δpp／修复新增 | low n／Δpp／修复新增 |
|---|---|---|
| <3 | 383／−.522193／7:9 | 1127／−.177462／19:21 |
| 3至8 | 718／−.417827／14:17 | 794／−3.022670／5:29 |
| 8至15 | 293／−1.706485／4:9 | 303／−3.300330／2:12 |
| ≥15 | 2206／−.317316／7:14 | 1376／−.363372／5:10 |

该比率不是录音总SNR。质量代理区间与锁定组对应，不能算独立重复证据；state/lock/quality是完整信道的关联分层，不是移除损伤的受控反事实。本轮没有证据确认多径、CFO、噪声或某类遮挡获得额外缓解。

## 默认头保持率和间隔

固定NoAs30 clean正确集合143216包，不是实际As20父模型集合。实际As20正式clean正确集合未保存，不能冒称已交付该父模型队列。

| 视图 | As保持率% | HALF保持率% | T2保持率% | As margin/30 | HALF margin/30 | T2 margin/30 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 97.408809 | 97.745364 | 97.382276 | 0.146124229 | 0.145108774 | 0.157917097 |
| practical_high | 90.110044 | 89.420875 | 89.755335 | 0.099391870 | 0.091715835 | 0.139971107 |
| practical_mid | 86.772428 | 85.908697 | 86.282957 | 0.089297004 | 0.081985503 | 0.126782626 |
| practical_low_suburban | 83.456457 | 82.516618 | 83.105938 | 0.079503775 | 0.072634257 | 0.114044666 |
| practical_high_urban | 85.932438 | 85.271199 | 85.502318 | 0.090724602 | 0.083422922 | 0.127550393 |
| practical_mid_urban | 68.782119 | 67.767568 | 68.166965 | 0.053809803 | 0.047853757 | 0.075508498 |
| practical_low_urban | 55.632751 | 54.464585 | 55.000838 | 0.028075527 | 0.023135012 | 0.039307609 |

T2平均默认头间隔在固定/30尺度下升高，但六种practical准确率、F1、保持率均低于As。间隔上涨不能作为身份识别改善证据。没有保存本轮独立CE参数梯度或完整内部门控轨迹；只能报告native/global/Sinc梯度健康与已有默认头结果。

## 完整训练与成本

两条尾段各2220次完整连续更新，共4440次；全部CE修正有活梯度，最大L分解误差分别1.278e−7、1.371e−7，没有非有限跳步。epoch骨干梯度范数HALF 1.090146至1.550461，T2 1.222550至1.682388。native loss首尾HALF .528366→.512031，T2 .648824→.618137；目标尺度不同不可据此排序。预算用尽不代表收敛。

每行1704960个L前向、142080个完整As；native U观察567000，原门控选中8717/8817，U真值未读。两行channel覆盖一致：lowSub47525/midUrban47116/lowUrban47439；6TX×5RX×3scene的90组合都有支持；15个RX/day硬件session不由TX定义；810展开格中515非空，空格保留；LOS/PARTIAL/BLOCKED=71872/22794/47414，locked/degraded/unlocked=86115/23865/32100。

| 训练成本 | HALF | T2 |
|---|---:|---:|
| 新增轮次／更新 | 10／2220 | 10／2220 |
| 学生train NN | 2271960 | 2271960 |
| 学生eval NN | 324002 | 324002 |
| 教师eval NN | 567000 | 567000 |
| 教师train NN | 0 | 0 |
| 新训练native NN | 3162962 | 3162962 |
| 新训练模型墙钟秒 | 1040.527079 | 1053.862350 |
| source读出NN／秒 | 12600／1.720002 | 12600／1.771662 |
| formal预测NN／秒 | 1176002／47.285043 | 1176002／48.435044 |

共享As20父成本：20E/4440更新/4797722nativeNN/1810.249311s，只在unique历史计一次。每候选累计30E/6660更新；新增训练模型墙钟合计2094.389429s，两行实际并行，合计不是端到端等待时间。新增remote已知总NN8703130（训练6325924＋source25200＋formal2352004＋一次CPsmoke2）。本地实际F验证1536NN单列；CE额外NN、额外教师NN均0。CPU分析29.546655s／0NN／0IQ／0CP／0更新。

历史NoAs30+As30=60模型E/13320更新；加本轮两尾段，unique80E/17760更新、已知remote27001702NN。共享As20未重复计费；历史六row共享smoke未分摊，明确保留未知。对照重新预测/评分次数均0。

## 验证、命令与实际文件

37协议负例＋12目标函数张量检查、4流程检查通过；两arm实际F共1536NN一次执行，默认6×160/cos30头与Sinc反传、206状态键/900722参数保持；分析12toy通过。唯一独立P0/P1 PASS。一次真实As20 checkpoint smoke1CP/2NN/0IQ/0query/0更新PASS；单archive SHA比较PASS；22py远端编译PASS。没有额外重跑模型测试或正式预测。

本地已执行命令，CWD=E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014：

    E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 experiments/cvs_satellite_ce64/local_synthetic_entry.py --dependencies E:/type10-7/code/worktrees/cvs_iq_timing_20261016
    E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 experiments/cvs_satellite_ce64/local_synthetic_entry.py --dependencies E:/type10-7/code/worktrees/cvs_iq_timing_20261016 --flow-only
    E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 experiments/cvs_satellite_ce64/local_synthetic_entry.py --dependencies E:/type10-7/code/worktrees/cvs_iq_timing_20261016 --synthetic

本地实际读数保留于 [local_verification.json](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/local_verification.json)。初次source toy因patchBASE导致config路径不符而失败（0NN），修正fixture后通过；实际F没有重跑。预检首次本地wrapper缺PATH，未SSH，恢复显式native环境后有效预检VERIFIED。这些不是训练失败。

实际运行命令（普通N607现有环境，CWD=PROJECT/releases/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01）：

    CVS-RFFI/python -u experiments/cvs_satellite_ce64/runner.py

自动CPU分析命令：

    CVS-RFFI/python -u analysis/cvs_satellite_ce64_analysis_20261018.py --project PROJECT --out PROJECT/runs/20261018-phase1-FcrossRX-satelliteCE-temperature-e30-s2026092701-m2-r01/analysis_r01 --local-nn 1536 --code-commit 90fe8ac0e2577ce0038a8e5a7855a4d2d9c5efc6

Python3.10.19/Torch2.1.0+cu121/NumPy2.2.5；fullFP32/CENTER_UNITS/CPU2。唯一owner codex/root/satellite-ce64；dispatcher1340924，HALF1341087/GPU0、T21341161/GPU1，均自然退出。独占runroot保留。实际模块路径及运行值见actual_runtime.json；resolved config、initialization、complete及scoring原产物位于local_artifacts取回目录。

实际新增实现与配置文件：
- [experiments/cvs_satellite_ce64/channel.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/channel.py)
- [experiments/cvs_satellite_ce64/checks.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/checks.py)
- [experiments/cvs_satellite_ce64/design.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/design.py)
- [experiments/cvs_satellite_ce64/dispatch.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/dispatch.py)
- [experiments/cvs_satellite_ce64/evaluate.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/evaluate.py)
- [experiments/cvs_satellite_ce64/flow_checks.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/flow_checks.py)
- [experiments/cvs_satellite_ce64/local_synthetic_entry.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/local_synthetic_entry.py)
- [experiments/cvs_satellite_ce64/model.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/model.py)
- [experiments/cvs_satellite_ce64/objective.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/objective.py)
- [experiments/cvs_satellite_ce64/objective_checks.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/objective_checks.py)
- [experiments/cvs_satellite_ce64/observer.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/observer.py)
- [experiments/cvs_satellite_ce64/preflight.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/preflight.py)
- [experiments/cvs_satellite_ce64/protocol.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/protocol.py)
- [experiments/cvs_satellite_ce64/qualification.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/qualification.py)
- [experiments/cvs_satellite_ce64/quality.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/quality.py)
- [experiments/cvs_satellite_ce64/README.md](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/README.md)
- [experiments/cvs_satellite_ce64/report.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/report.py)
- [experiments/cvs_satellite_ce64/runner.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/runner.py)
- [experiments/cvs_satellite_ce64/source.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/source.py)
- [experiments/cvs_satellite_ce64/synthetic.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/synthetic.py)
- [experiments/cvs_satellite_ce64/__init__.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_satellite_ce64/__init__.py)
- [analysis/cvs_satellite_ce64_analysis_20261018.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_satellite_ce64_analysis_20261018.py)
- [analysis/cvs_satellite_ce64_analysis_checks_20261018.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_satellite_ce64_analysis_checks_20261018.py)

数据、权重、原始日志和预测没有提交Git；代码、公共报告与登记提交。

## 图表与原始汇总

五组PNG/PDF都已实际生成并打开核查：
- [training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.pdf)
- [satellite_ce_training PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/satellite_ce_training.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/satellite_ce_training.pdf)
- [practical_mid_urban_source_states PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/practical_mid_urban_source_states.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/practical_mid_urban_source_states.pdf)
- [practical_low_urban_source_states PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/practical_low_urban_source_states.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/practical_low_urban_source_states.pdf)
- [formal_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/formal_accuracy.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/formal_accuracy.pdf)

全部细分、完整曲线与原始汇总：
- [complete.json](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/complete.json)
- [formal_summary.json](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/formal_summary.json)
- [formal_metrics.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv)
- [formal_pair_summary.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv)
- [formal_paired_effects.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv)
- [formal_confusions.json](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/formal_confusions.json)
- [source_summary.json](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/source_summary.json)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv)
- [source_paired_effects.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)
- [source_confusions.json](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/source_confusions.json)
- [new_tail_training.json](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/new_tail_training.json)
- [training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.csv)
- [training_coverage.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/training_coverage.csv)
- [CE_HALF_satellite_ce_steps.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/CE_HALF_satellite_ce_steps.csv)
- [CE_T2_satellite_ce_steps.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/CE_T2_satellite_ce_steps.csv)
- [costs.json](E:/type10-7/local_artifacts/cvs_FcrossRX_satellite_CE_temperature_E30_20261018_r01/retrieved_r01/analysis_r01/costs.json)

未完成的科学目标：没有超过原As30，没有low urban>60%，没有跨seed可重复双提升，也没有新的物理退化因果归因。实际As20父模型formal clean正确集合、独立CE参数梯度与本轮完整内部门控轨迹未保存。本矩阵训练/评价/图表产物完整，不将科学失败写成技术缺项。
