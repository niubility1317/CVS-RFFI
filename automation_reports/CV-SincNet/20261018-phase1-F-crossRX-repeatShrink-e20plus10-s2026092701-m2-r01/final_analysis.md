# 重复残差 IQ 增强：E30 B00/B10 完整结果

状态：ANALYZED／VERIFIED。固定算子已按源条件失败关闭。两行均完成固定 E20＋10、默认头 clean＋六种 practical 预测、逐 row 独立 truth-last 评分与全部复算。联合目标仍 ACTIVE。

|固定视图|B00 准确率|B10 准确率|B10−B00（百分点）|
|---|---:|---:|---:|
|clean|85.357738%|85.333333%|-0.024405|
|practical_high|43.461905%|43.538095%|+0.076190|
|practical_mid|41.784524%|41.885119%|+0.100595|
|practical_low_suburban|39.940476%|40.064881%|+0.124405|
|practical_high_urban|42.236310%|42.299405%|+0.063095|
|practical_mid_urban|35.521429%|35.567857%|+0.046429|
|practical_low_urban|30.353571%|30.423214%|+0.069643|

源 V 判据已先冻结：整体差 +0.000000pp，条件 {"fullV_clean_gain": false, "each_RX_protected": true}，结论 CLOSED_FIXED_REPEAT_SHRINK_SOURCE_GAIN_FAILED。正式结果不回流改写这一决定。

|视图|修复原错误|新增错误|clean 与该场景同时正确的净变化（包）|
|---|---:|---:|---:|
|clean|107|148|-41|
|practical_high|534|406|143|
|practical_mid|567|398|183|
|practical_low_suburban|600|391|207|
|practical_high_urban|544|438|127|
|practical_mid_urban|604|526|100|
|practical_low_urban|686|569|101|

该同时正确量比较的是两个仅研究 Ag 的模型，不是 Ag＋As 联合结果。所有源与正式 RX/day/TX 结果、macro-F1、最差 RX 和共同 clean 正确集合上的修复/新错均保留在聚合结果中。

|模型|high suburban 与自身 clean 差|mid suburban 差|low suburban 差|
|---|---:|---:|---:|
|B00|41.895833pp|43.573214pp|45.417262pp|
|B10|41.795238pp|43.448214pp|45.268452pp|

2pp 仅为事先固定的探索目标。需同时检查 suburban 绝对准确率与 clean，不能把 clean 下降造成的差距缩小当作保护。

本矩阵没有 As 训练，也没有 B01/B11，Ag–As 交互项未定义。仅一个模型 seed，不宣称跨 seed 重现。不能将本矩阵的 clean 差与另一个 As 实验的收益相加预测联合成绩。

两行来自同一个合法 fullsource REPEAT final20 学生及 own EMA，各重置 AdamW/cos10、续训 2220 次，累计 E30/6660 更新。F/SIG_TIME 三路、统一 Sinc、joint160、默认 cosine 头与骨干共同训练；NoExplicitCFO/原补偿和 U 规则保持。全部源 RX 参与训练，未使用 L5580/dev720 或长 IQ。

监督为 .5RD(ordinary)＋.25RD(raw 锚点)＋.25RD(raw 或 Ag(raw))，总 L 尺度1，原 SupCon/U 另按原配方。每包6次有标签前向且每个 RDrop 对复用同一 IQ；0新损失/部署参数/头拟合/新教师/VJP/As 训练。固定 Ag 仅将四周期偏离自身均值部分按私有 Uniform[0,.1] 收缩，不能视为真实接收机替换或纯 RX 噪声去除。

成本：共享 E20/4440 父成本单列，新增两行各 E10/2220，唯一累计40 model-epoch，非两个 scratch E30。每行有标签 NN1704960（比旧4前向多568320），原 U/教师/验证及正式预测另计。全部精确前向和 CPU bridge/operator 时间见结果 cost；sourceguard 先前14404 NN/0更新另列。

独立复算全部252正式混淆矩阵、42组分区求和、7组同物理包配对，最大度量误差 0.0。源全部步骤/序列/lambda/损失尺度和 native CSV/JSONL 逐值核对；仅读已保存日志/源聚合/正式计分与正确性向量，0 IQ/CP/NN/更新/raw truth。

测试 received IQ、信道/SNR、困难样本与全部168000物理包保持，全部7视图固定后才独立计分。代码执行身份与资源核实见原 report.md；完整私有日志/预测/物理ID/权重保留，不提交 Git。

联合目标尚未完成：有效 Ag 跨 seed 证据、同契约完整四格与交互项、clean 净正、low urban>60%、suburban 保持仍需按完整证据判断。

独立Ag结果：正式clean −0.024405pp（修复107、新增148，净少41包），low urban +0.069643pp（修复686、新增569，净多117包）。六个practical均有小幅单种子增加，但没有独立clean净收益；这不支持通过新Ag增强跨RX身份提取。因源门槛已在target读取前失败，不扩种子或进入该固定算子的联合训练。

精确前向：新增两模型训练共6325924 NN、正式预测2352004 NN（含各2包预测兼容smoke）、共享真父smoke2 NN，当前矩阵新增共8677930 NN；共同E20父3661082 NN单列，含共同父共12339012 NN。先前源保护14404 NN另列；未新增teacher模型或VJP。operator/bridge调用墙钟B00 13.334535秒、B10 13.970596秒，含可能GPU等待，不能作为纯CPU核耗时或速度收益。模型训练墙钟B00 818.841166秒、B10 811.528637秒；预测各约41.4秒，完整计分各约3.3秒。

原E20压缩预算保留此前已暴露基准的历史选择背景，本轮不声称新的独立盲测；本算子和源判据由合法源信息冻结，正式测试没有参与强度、父权重或晋级。

[固定源决定](source_decision.md)、[全部聚合结果](final_analysis.json)、[准确率与配对差图](repeat_shrink_E30_accuracy.png)、[PDF图](repeat_shrink_E30_accuracy.pdf)、[执行摘要](execution_summary.json)。
