# P70 完整 practical：固定质量组风险的 E30 匹配结果

**结论：仅改变质量分层估计，尚未建立有价值、可重复的新增星地收益。** M2 相对同池 M1 的 urban 均值 +0.059722 个百分点，low/mid urban 各约 +0.101 个百分点；相对原 As30，urban 均值 −0.022024、clean −0.086310 个百分点。M2 相对同 seed NoAs30 的 clean +0.007143 个百分点，本 seed 满足 clean 下降不超过 0.1 个百分点；这只有净多12次正确，不能称为可重复双提升。源两项条件均 FAIL，固定候选关闭，不扩 E50、不挑新 seed。

本轮已 ANALYZED，所有必需 row/视图/评分/曲线齐全；总体目标仍 ACTIVE，low urban 60% 及跨 seed 双提升尚未完成。

## 实际配置与方法

运行 ID：20261018-phase1-FcrossRX-qualityStrat-e30-s2026092701-m2-r01；代码提交：000a34f52dc4240c8d078254ada17e7761a27a84。M1=POOL_RANDOM、M2=QUALITY_STRAT，仅模型 seed2026092701。两条新尾段均严格加载原 P43 B01 scratchAs20 学生和自身 EMA，独立 RESET AdamW＋原 cosine10/CENTER_UNITS，追加 E10/2220 更新，总 E30/6660。原 P43 NoAs30/As30 是既有完整轨迹参照，未重训、未重预测或重评分。NoAs 对照有自己的 NoAs20 父，并不共享 As20 父。

固定 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、joint160、默认6×160 cosine头/scale30，900722参数/206state；默认头与骨干正常联合更新。完整L6300/U56700（TX隐藏）/V27000，RX1/3/4/6/8，源day1/2/3，无held RX/L_dev，split392005。没有新增部署模块、诊断分类器、损失堆叠或U规则。

实际模块由运行日志确认来自 pinned20261015_r02 的 experiments/adv3b02_xuc/code/leo_practical，而非顶层code同名实现。fs25MHz、IQ2×256、逐包RMS/center=false后走原practical，沿原网络输入策略。训练fc2.462GHz和固定正式fc2.45GHz的历史差异保持。真实运行是 residual/post_sync、EQ=false（G=1）、reference_snr_db30、reflect；接收质量适配、相位跟踪、IQ补偿、AGC均启用，AGC±30dB，quality_reference_snr_db20。配置中的equalizer129tap/delay64因EQ关闭而不代表已均衡。正式received IQ和全部六种practical不变。

共享池依据原 P43 As30 尾段 L 物理序列与原 epoch/step/ID 命名流，从raw L逐次生成完整practical，未经过ordinary预损伤。生成142080次observations（出现次数），保留15个按RX/day/eq定义的硬件session；不按TX定义session，不去重原物理出现次数。IQ、真实标签、compact domain、物理ID和接收元数据共同移动；每行恰消费每个池索引一次。两行 native L/U 物理序列与原参照逐项一致，As顺序单独重排。

M1全池随机排列、每步64、普通均值CE。M2每步四组floor/ceil配额，固定p_g=N_g/N，AsCE=.25×Σp_g mean_g CE。原同IQ RDrop KL仍等权.25；ordinary/raw、ORD128 SupCon和U不变。L每步768次样本前向，CE总系数1.25，其中As .25。继承config中少数4L/cost_claim文字描述不是执行值，实际六次前向、L_NN与日志分解已经逐步核对。

四组新增噪声比（不是总SNR）的N_g/p_g依次为21825/.15361064、21572/.15182995、12517/.08809825、86166/.60646115。所有组正且足够，无proposal、丢弃或fallback。M2对有限nominal池的固定参数组风险估计无偏；不声称对连续物理总体或训练轨迹条件梯度精确无偏。缓存跨原epoch重排，不宣称每步仍是同一clean包的成对观测。

## 正式结果

以下均为准确率%，每视图168000包；每row全部7预测固定后，独立truth-last评分。四行共504次混淆矩阵复算、84次分区核对通过。

| 环境 | NoAs30 | 原As30 | M1同池随机 | M2质量分层 | M2−M1（百分点） |
|---|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.183333 | 85.254762 | +0.071429 |
| practical_high | 42.685119 | 81.405357 | 81.280952 | 81.274405 | -0.006548 |
| practical_mid | 40.995238 | 78.661310 | 78.545833 | 78.532143 | -0.013690 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.804167 | 75.806548 | +0.002381 |
| practical_high_urban | 41.439881 | 77.767262 | 77.679167 | 77.655357 | -0.023810 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.674405 | 62.776190 | +0.101786 |
| practical_low_urban | 30.139881 | 51.244048 | 51.219048 | 51.320238 | +0.101190 |
| urban三场景均值 | 35.570833 | 63.939286 | 63.857540 | 63.917262 | +0.059722 |

| M2环境 | Macro-F1（%） | 最差RX准确率（%） | 最差RX |
|---|---:|---:|---|
| clean | 84.950956 | 78.333333 | 1-1 |
| practical_high | 80.929941 | 71.220833 | 14-7 |
| practical_mid | 78.203072 | 69.608333 | 14-7 |
| practical_low_suburban | 75.478604 | 67.204167 | 14-7 |
| practical_high_urban | 77.282360 | 68.316667 | 14-7 |
| practical_mid_urban | 62.349728 | 56.112500 | 14-7 |
| practical_low_urban | 50.858866 | 46.458333 | 14-7 |

相对原As30，M2 clean −0.086310、mid urban −0.030357、low urban +0.076190、urban均值 −0.022024 个百分点。新增机制没有超过原配方的整体urban表现。相对NoAs的大幅收益主要属于已有完整As：M2 low +21.180357、mid +27.643452、urban均值 +28.346429 个百分点，不能归给质量分层。

相对M1，clean修复431/新错311，净+120；mid urban修复1069/新错898，净+171；low urban修复1350/新错1180，净+170。urban合计修复3092/新错2791，净+301/504000。low准确率微涨同时Macro-F1 −0.020114个百分点；mid最差RX −0.054167个百分点，高urban最差RX −0.337500个百分点。low七个RX均正向，但属于同一模型的切片，不能替代跨模型seed复现。

M2 clean与三个suburban的差距为3.980357、6.722619、9.448214个百分点；仍未达到建议的2个百分点内探索目标。low urban51.320238%，距60%还差8.679762个百分点。

## 源域证据与归因

源规则在读取本轮正式成绩前固定封存：全V27000 M2 clean−NoAs=−0.125926个百分点（FAIL）；两draw pooled midlow M2−M1=−0.027778个百分点（FAIL，要求≥+.5）。源low本身+0.166667，但mid −0.222222个百分点。源FAIL不取消正式测试，也不由正式微涨推翻。

| 源low urban分组，两draw合计 | N | M1正确 | M2正确 | 变化（百分点） |
|---|---:|---:|---:|---:|
| <3 dB | 1127 | 307 | 305 | -0.177462 |
| 3-8 dB | 794 | 313 | 320 | +0.881612 |
| 8-15 dB | 303 | 218 | 219 | +0.330033 |
| >=15 dB | 1376 | 1284 | 1284 | +0.000000 |

源low BLOCKED共1979包，仅净+4（+.202122个百分点）；BLOCKED/unlocked共1565包，仅净+2（+.127796个百分点）。源mid BLOCKED净−4（−.299850个百分点）。可描述的微小改善集中在low的3–8dB/degraded观测，不能声称最差噪声或所有遮挡退化已被修复。正式逐包状态标签不可用，不能把这些源标签投射为正式收益的因果归因。

四组批次数方差从[8.09624,8.26052,5.46691,16.17333]降至[.14039,.20286,.23088,.15171]，但As训练CE均值仅从.738595变为.737791，源身份识别并未随之推进。这支持“批次组成变稳”，不支持“低质量身份证据显著增强”。两行均覆盖90/90 TX×RX×训练场景，channel状态由原物理过程联动生成。

固定实际P43 As20 clean正确集合为1765/1800。M2相对M1的low条件保持率+0.169972、mid−0.226629、clean−0.056657个百分点；clean/mid/low的margin÷30均略降，分别−.000468567/−.000467608/−.000094879。该集合不是源NoAs30正确集合1772，更不是正式low准确率或模型seed。

旧U规则不变，两行各观察567000次U，接受M1=8710/M2=8724；不解封U真值，不把接受数当伪标签精度。未新增逐内部门控反事实或分支诊断前向，因此不作门控因果解释。梯度健康与完整训练曲线保留。

## 成本、验证和交付

两新尾段共20模型epoch/4440更新；每行自身累计30epoch/6660更新。新native NN6325924、source25200、formal2352004、共享真实checkpoint smoke4，总8703132新远端样本前向。本地验证/分析均0真实CVS NN。共享As20父native4797722 NN/1810.249秒已训练并复用；原NoAs20父亦4797722 NN/1329.394秒。包含原参照的唯一父训练40模型epoch、四尾段40epoch，共80唯一模型epoch，不能把每条声明E30简单相加为独立成本。

M1/M2原生阶段墙钟1156.040/1117.213秒（包含pool生成或等待），源读数1.907/1.802秒，正式49.297/47.551秒。共享pool生成287.656秒、142080次原信道处理、IQ290979840字节；这笔生成只发生一次，等待不能重复当模拟成本。CPU分析33.631秒；提交至分析完成1254.797秒（20.913分钟），并行训练wall之和不是总耗时。预算耗尽不等于收敛。

本轮35项训练协议/池/接线检查、9项CPU分析toy和最终零权组定点检查通过；20个新Python文件及23文件远端发布编译通过；唯一独立P0/P1 PASS；真实父student/ownEMA无query smoke4NN/0更新PASS。一份代码归档只做一次local/remote SHA比较，PID/CWD/argv/GPU/log增长及退出后状态独立核实。无技术失败或重跑，其他健康实验未改动。

实际执行入口（既有CVS-RFFI环境、cwd=PROJECT/releases/cvs_FcrossRX_quality_strat_E30_20261018_r01）：
```text
python -u experiments/cvs_quality_risk70/runner.py
```
本地合同检查入口：
```text
E:/type10-7/local_envs/ssr-gpu/python.exe experiments/cvs_quality_risk70/local_synthetic_entry.py --dependencies E:/type10-7/code/worktrees/cvs_iq_timing_20261016 --native-source E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/adv3b02_xuc/code/SSDG/train_ssdg.py
```

实现文件位于[实验目录](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_quality_risk70)；[逐row/seed正式指标](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv)、[配对错误](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv)、[源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv)、[完整曲线表](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.csv)、[成本](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/costs.json)。原始日志、预测、IQ和权重未提交Git，保留在原run和本地取回目录。

[完整训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/full_training.png)；[状态/质量分层图](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.png)；[正式配对图](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired.png)；[训练池分配图](E:/type10-7/local_artifacts/cvs_FcrossRX_quality_strat_E30_20261018_r01/retrieved_r01/analysis_r01/pool_allocation_readout.png)。四PNG已实际查看，四PDF已导出并核实存在。

必需项无缺失。仅1模型seed，未运行M3、有界困难风险、U practical路由或新E50。NoAs20自身clean正确cohort及正式parent20 cohort的历史缓存不可用，未为此补充NN；本轮使用已存在合法As20源缓存。当前固定分层候选关闭；下一项可独立验证的假设是固定As总权重下改变困难组内部风险，而不是再次扫描分层seed或增加视图。后续晋级仍只看预登记源证据，不依据本轮正式微涨调参。

**最终回答：** 原完整As仍提供显著星地适应；本轮分层只见low的3–8dB源观测微改善，未修复<3dB主难点。默认头相对同池控制low +.101190、urban均值+.059722、clean+.071429个百分点；相对原As，urban略降。相对匹配NoAs，clean+.007143个百分点，本seed通过0.1个百分点阶段要求。没有跨seed可重复双提升证据，也未达到low urban60%。
