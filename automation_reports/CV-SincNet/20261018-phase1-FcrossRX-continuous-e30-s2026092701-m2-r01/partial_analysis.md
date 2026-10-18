# 连续 E30：技术失败后的实际交付

状态：PARTIAL_TECHNICAL_FAILURE。No-As B00 完成 E30、原七视图及独立 truth-last；As B01 完成 E20 后在 E21 被旧包装器边界拒绝。此矩阵没有新的 As 配对收益，也不能判断 0.1 pp 目标。不是科学负结果。

实际运行代码：`04ac1098c8c23ec7dab655a1a1317e7363ad580c`；本地修复提交 `e0b5b91cf6591e9b028c14df8ebeccd021ffa3a3`、接口兼容及本报告工具提交 `231702b5a32a30f0839ac9361c0fc6ac1c9489ad`。原远端 R01 不变。

## 已完成的固定测试

B00、model seed2026092701、split392005、固定 ManySig 168000/视图、网络默认头。所有七视图 prediction 固定后独立 CPU scorer 连接 truth，126 个分层结果齐全。以下只是无 As 对照，不能视为增强收益。

|视图|准确率 %|宏 F1 %|最差 RX %|
|---|---:|---:|---:|
|clean|84.883333|84.597034|76.875000|
|practical_high|43.560119|40.083272|33.295833|
|practical_mid|41.729167|38.727962|32.325000|
|practical_low_suburban|40.026786|37.398752|31.179167|
|practical_high_urban|42.319643|39.293892|32.883333|
|practical_mid_urban|35.608333|34.020499|28.775000|
|practical_low_urban|30.417262|29.674297|25.170833|

urban 三场景均值 36.115079%。B01 没有 final checkpoint、预测或评分；不拿 E20 对 E30 作匹配结果，不替代缺失成绩。

## 实际训练与失败成本

两行均 scratch，父训练成本0。模型 F/SIG_TIME_X3、统一 Sinc、原内部门控/参考、joint160/defaultcos30 与联合训练不改；L6300/U56700 TX隐藏/V27000，全部源RX1/3/4/6/8，day1/2/3，eq1、IQ256/25MHz/RMS centerfalse。没有训练 RX 留出。

|行|完整 epoch|成功更新|已记录完整步 L 前向|As IQ|epoch 时间合计 s|
|---|---:|---:|---:|---:|---:|
|B00|30|6660|5114880|0|2113.316|
|B01|20|4440|3409920|284160|1795.009|

B00 实际 student_train/eval=6078780/864002，teacher_train/eval=0/963900；冻结源读出12600NN、七视图及预测smoke1176002NN，合计9095284NN。原生 source wall=2132.725s，prediction wall=47.367s。epoch时间合计与source总wall口径不同，未把并行两行时间相加冒充矩阵墙钟。

失败 B01 的 L sequence 为4441条/568448物理样本曝光，含失败步额外128；U为1554批/396900样本。两行记录的唯一训练ID均为L6300/U56700，U ID只作不透明计数，无TX真值。B01已完成4440次更新的L前向为3409920；失败步部分前向和未持久化总NN计数不完整，total_NN记为UNKNOWN。含B00全部NN、B01完整步L和共同smoke1538，本次已知NN下界12506742，不是总NN估计。完整模型epoch合计50、成功更新11100；失败成本不能从后续预算报告中删除。

训练实际完整As为原adv3b02_xuc practical residual/post_sync、EQoff G1、reference30、25MHz、fc2.462GHz、reflect；接收质量、相位追踪、IQ补偿、AGC保持。正式原received IQ（原fc2.45GHz及既有manifest）和SNR不变。失败由外层预算guard造成，不是数值发散、标签变化或测试条件变化。

## 完整已执行曲线与缺项

已解析所有50条native epoch、11100条完整成功step，以及两行所有L/U sequence；无抽样代替完整曲线。源V、loss、gradient、LR、原U接纳和epoch wall均画出；B01 E21以后的数值没有补画。实际PNG/PDF已取回并目视检查。

- [训练曲线 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_E30_20261018/retrieved_partial_r01/partial_training_curves.png)
- [训练曲线 PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_E30_20261018/retrieved_partial_r01/partial_training_curves.pdf)
- [完整 epoch 表](E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_E30_20261018/retrieved_partial_r01/full_completed_epoch_curves.csv)
- [NoAs 七视图表](E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_E30_20261018/retrieved_partial_r01/completed_NoAs_views.csv)
- [实际成本与缺项 JSON](E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_E30_20261018/retrieved_partial_r01/summary.json)

原四行独立分析等待源输入后因已知failure按原规则退出，0NN/0新IQ/0checkpoint/0truth，没有生成完整源判据、四行物理分层、收益归因或新机制配对图。该缺项保留。新生成的部分报告只读取已经保存的日志和独立scorer输出，0NN/0truth读取，不改变原训练或评分。

## 新目标下仍然成立的证据

当前最值得保留的完整As参照仍为旧三seed匹配的clean份额配方。以下不是本次不完整新矩阵的结果：

|seed|clean Δ pp|low urban Δ pp|mid urban Δ pp|urban均值 Δ pp|clean下降≤0.1 pp|
|---|---:|---:|---:|---:|---|
|2026092701|0.093452|21.104167|27.673810|28.368452|满足|
|2026092702|-0.153571|14.517857|17.533929|17.629167|不满足|
|2026092703|-0.902976|16.232738|20.626786|20.926190|不满足|

三seed均值：low +17.284921 pp、mid +21.944841 pp、urban +22.307937 pp，clean −0.321032 pp，clean样本SD0.518893 pp。均值不满足0.1 pp阶段目标；seed01的+0.093452 pp不能作稳定双提升证据。原三seed全部六practical、宏F1/最差RX、修复/新增错误、源state/lock/quality分层与完整曲线见同run系列的[原完整报告](../20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02/final_analysis.md)。

旧源分层的As收益在LOS/locked与PARTIAL可恢复观测更大，BLOCKED/unlocked仍困难。这是源固定面板的状态关联，不是正式目标逐包物理因果结论，不能归因到某一种多径、CFO、噪声或门控。新增噪声比不等于总SNR。

## 验证、命令与后续边界

原运行命令和环境/CWD/GPU/停止规则见report.md；原独立实际PID/CWD/cmdline/GPU/log增长已核实。两个worker、dispatcher和旧四行分析现均退出，原失败文件存在；健康B00未被停止、重启或热改。

后续边界修复：10项本地检查PASS；真实引擎合成输入验证E20旧新完全等价、E21/30 finite和命名流刷新、越界拒绝、EMA deepcopy；真实F工厂B00 NoAs0包/B01 Practical2包；原tensor_iq桥保留并实测。定向P1复核PASS；修复验证全为0NN/0真实IQ/0checkpoint。原7项launcher检查和实际checkpoint2NN＋1536合成梯度NN也已PASS，但未覆盖E21，记录该测试缺口。

本地检查：`F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_continuous52.checks`（10项）；`F:/App/miniconda3/python.exe -X utf8 analysis/check_satellite_priority_assessment.py`（9项）。部分分析命令：`CVS-RFFI/bin/python cvs_continuous52_partial_20261018.py --run RUN_ROOT --output RUN_ROOT/analysis_partial_r01`；只读已有产物，不再推理或评分。

唯一未完成的匹配假设仍为原连续优化轨迹；没有科学结论。另一聊天的satellite-only RDrop KL消融由其独立推进，本对话不重复。原预登记“技术失败不自动重试”及用户“不主动扩预算”继续有效；已向用户明确额外从零As E30的成本，待预算答复，不静默覆盖或重跑。本报告未用正式分数调参、选seed、晋级或扩预算。

最终判断：旧完整As的星地收益跨3seed复现；clean代价均值0.321 pp，不满足新0.1 pp目标；尚无星地增强同时改善clean的可重复证据。本次技术失败不增加支持或反对该科学目标的证据。
