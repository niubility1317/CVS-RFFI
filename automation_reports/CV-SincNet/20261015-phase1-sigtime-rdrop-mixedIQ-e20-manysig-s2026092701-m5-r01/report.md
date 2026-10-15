# SIG_TIME＋RDrop：固定 E20 混合 IQ 五组结果

状态：ANALYZED，五个row／一个seed已完成，科学目标未达到；当前候选不扩seed。训练、预测和评分进程均自然结束，远端终点与本地取回产物核验VERIFIED。

**结论：本轮没有实现clean和urban双突破。** B11默认头clean80.2732%、low urban51.5262%、mid urban62.2244%、urban均值63.3119%。相对仅practical的B01，clean只+0.0131pp、low只+0.0393pp、urban均值只+0.0500pp；相对原RDrop参照clean−2.7280pp、low+15.2851pp。low仍低于60%，suburban的high/mid/low距自身clean为0.6637/3.1024/5.3280pp，只有high满足2pp探索目标。当前A_g独立源收益已在读取正式分数前判失败，见[source决定](source_decision.md)，对应提交e7745dbdf1e898709633c3d8b42e32d7751e1eca。

收益主要来自A_s。与前向匹配的B00比较，B01的low+20.1708pp、urban均值+24.0486pp，但clean−2.8488pp。A_g单独使正式clean+0.2089pp，但联合后该clean增量没有保留；clean/low/mid/urban的交互项依次为−0.1958/−0.1179/−0.2310/−0.1879pp。这是竞争而非双增强协同的证据。

源固定V720的low urban分层显示，A_s相对B00将LOS从62.05%提高到95.38%、PARTIAL从52.94%到85.29%、BLOCKED从18.44%到34.99%；unlocked仅从17.87%到32.60%，新增噪声SNR<0dB组仅从17.53%到21.65%。这支持“适应仍有可用证据的受损观测”，不证明已恢复不可观察身份，也不能从本因子实验分离噪声与多径的各自因果贡献。B11相对B01在该源low panel反而净少对2条；按各自分层汇总，净损失落在BLOCKED层和degraded层，不能据净计数推断逐样本纠错流向。平均头间隔略升并未变成正确率提升。

后续唯一源问题仍是读取本轮正式分数前已固定的问题：TX平衡后源RX/day的观测响应差异是否跨TX重复，当前单一形状轻度响应是否覆盖不足。先开展无额外诊断头的合法源统计／冻结默认头干预，再决定下一种IQ算子；本轮目标数字不用来选择幅度、采样或预算。当前未启动下一模型。

执行代码：`1fe3f00bdddfdb0af494f74c6570db65490ddf72`。一个模型种子 2026092701，五个独立 scratch E20；每个模型 4,440 次更新，full L6300/U56700/V27000。原 SIG_TIME/joint160/NoExplicitCFO、默认 cosine 头联合训练，旧 U 门控与 ownEMA 不变。

所有 row 在原 clean＋六种 practical 上固定预测后独立 truth-last 评分。以下是已暴露固定基准上的单种子探索，不是新的独立盲测或跨种子复现。

## 七视图默认头准确率

|row|Clean %|High S %|Mid S %|Low S %|High U %|Mid U %|Low U %|
|---|---:|---:|---:|---:|---:|---:|---:|
|R_LEGACY|83.0012|58.7024|55.6101|52.5476|56.1321|44.5857|36.2411|
|B00|83.1089|50.4548|47.8018|45.1506|48.0643|38.2595|31.3161|
|B10|83.3179|50.7750|48.1304|45.5589|48.3476|38.5327|31.4732|
|B01|80.2601|79.5196|77.1131|74.9006|76.1167|62.1821|51.4869|
|B11|80.2732|79.6095|77.1708|74.9452|76.1851|62.2244|51.5262|

|row|urban 均值 %|urban 宏 F1 %|最差 urban RX %|
|---|---:|---:|---:|
|R_LEGACY|45.6530|43.1090|36.2681|
|B00|39.2133|36.9720|30.1153|
|B10|39.4512|37.2146|29.5181|
|B01|63.2619|62.9518|56.2250|
|B11|63.3119|63.0008|56.3569|

## 配对差和交互项

|比较|clean Δpp|low urban Δpp|mid urban Δpp|urban 均值 Δpp|
|---|---:|---:|---:|---:|
|B10-B00|+0.2089|+0.1571|+0.2732|+0.2379|
|B11-B00|-2.8357|+20.2101|+23.9649|+24.0986|
|B11-R_LEGACY|-2.7280|+15.2851|+17.6387|+17.6589|
|B11-B10|-3.0446|+20.0530|+23.6917|+23.8607|
|B11-B01|+0.0131|+0.0393|+0.0423|+0.0500|
|B01-B00|-2.8488|+20.1708|+23.9226|+24.0486|
|B10-R_LEGACY|+0.3167|-4.7679|-6.0530|-6.2018|
|B01-R_LEGACY|-2.7411|+15.2458|+17.5964|+17.6089|
|B11−B10−B01＋B00|-0.1958|-0.1179|-0.2310|-0.1879|

B10−B00 检验新增响应变化；B01−B00 检验 practical；B11−B01 检验响应变化对 practical 的增量。B00−R_LEGACY 同时改变了旧普通 L 增强和前向数量，不能单独归因为某个算子。R_LEGACY 不能被较弱的重组对照替代。

## 源规则与正式探索条件

源资格：False；后续扩种子资格只来自该固定源规则，目标成绩不能覆盖。

|源条件|通过|
|---|---|
|B10_clean_gain|False|
|B11_clean_gain|False|
|B11_low_gain|False|
|B11_urban_gain|True|
|clean_health|True|
|worst_RX_health|True|

|正式探索目标|通过|
|---|---|
|clean_gain_over_both|False|
|low_urban_over60|False|
|suburban_near_own_clean|False|
|suburban_absolute_protection|True|

正式状态：`EXPLORATION_TARGETS_NOT_MET`。一个种子不足以证明跨 seed 复现。固定 E20 结束不等于收敛。

## 源域机制与观察边界

|row|full V clean %|V 最差 RX %|固定 V720 low urban %|low 保持率 %|low 默认头间隔/30|
|---|---:|---:|---:|---:|---:|
|R_LEGACY|97.8630|94.2778|40.9722|41.5954|-0.013189|
|B00|98.2333|95.2963|35.1389|35.8974|-0.031979|
|B10|98.2481|95.2407|34.5833|35.0427|-0.031717|
|B01|97.8370|94.1852|58.4722|59.2593|0.057284|
|B11|97.8667|94.3333|58.1944|59.1168|0.057736|

保持率始终基于冻结参照模型在同一 V720 上的 clean 正确集合；分母固定，不按每个新模型重选。分层图只描述合法源域相同受损观测，不把源状态差异解释成目标域因果结论。源新增噪声 SNR 分箱也不等于总 SNR。

## 实际成本

|row|L 前向|student U 前向|EMA 前向|student 源评估前向|源训练秒|全 row 墙钟秒|
|---|---:|---:|---:|---:|---:|---:|
|R_LEGACY|1136640|396900|396900|594002|1188.863|1243.850|
|B00|3409920|396900|396900|594002|1506.779|1561.312|
|B10|3409920|396900|396900|594002|1578.036|1631.326|
|B01|3409920|396900|396900|594002|2285.043|2338.531|
|B11|3409920|396900|396900|594002|2231.107|2283.239|

每 row 另有冻结源 panel 5,040 前向及正式预测 1,176,002 前向（含 2 条 smoke）；五个模型累计 100 model-epoch、22,200 更新。共同源准备 722 前向另计。历史冻结源安全实验 135,002 前向／74.915770 秒、0 更新；其 E20 原参照历史成本 4,440 更新／923.560061 秒，不是本轮任何模型的初始化或继承成本。

当前矩阵墙钟 2358.224 秒，按已保存文件时间戳计算；每 row 墙钟从已登记启动时间到 row 完成，包含各阶段与开销。GPU 与其他健康任务共享，墙钟不作为纯算法速度归因。

## 图和可检查数据

- [formal_metrics](E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/analysis_r01/formal_metrics.png)
- [factorial_effects](E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/analysis_r01/factorial_effects.png)
- [training_curves](E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/analysis_r01/training_curves.png)
- [source_state_quality](E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/analysis_r01/source_state_quality.png)
- [training_coverage](E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/analysis_r01/training_coverage.png)
- [measured_cost](E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/analysis_r01/measured_cost.png)

同目录 CSV 包含七视图逐 row／seed 结果、配对差、交互项、全 20 轮曲线、源分层、实际覆盖与成本。分析程序只读取已保存日志、源诊断和独立评分，不访问 IQ、模型权重或 truth。

整体目标保持未完成，不能把本轮完成或局部正差写成 clean 与 urban 双突破。新方案若继续研究，须依据正式分数读取前已冻结的源问题另行登记，不用本轮目标分数选择幅度、采样或训练预算。

## 执行路径、测试和未完成项

actual channel `__file__`：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py`，residual在同目录。实际25MHz／IQ256、residual/post_sync、EQfalse即G=1、referenceSNR30、reflect、质量适配／相位跟踪／IQ补偿开启。源原RMS后进入模拟器unitRMS，模拟输出不额外RMS；训练G/S分开，S前无普通AWGN／多径。原正式received IQ及信道未改。

完整解析五组100个epoch／22200条step／完整stdout：无技术错误、关键损失／梯度非有限值或跳过更新，U truth标记全为0，有标签原CE替换与登记相符。75项实际源聚合核对通过。完整读取1136640条实际训练信道记录：B01/B11各568320条、180个TX×RX×场景单元、15个session且每session6TX；全部568320对除学生correct字段外元数据完全一致。各有43次AGC限幅，残余核长62，质量代理范围−20.4743至35.2169dB；这些为训练观测统计，不是总SNR或正式预测指标。

原合成检查8项、五真实native图3328个合成前向／10负例、13项协议负测、observer774个合成前向、唯一发布竞态2项针对性检查与限定P0/P1复核通过；所有本地合成检查无真实训练更新。发布首步真实checkpoint无query smoke通过，单归档比较、远端编译、运行PID/CWD/argv/GPU/log增长和最终产物均有实际记录。每row126个独立评分记录，合计630个；七视图正式决策总数5880000，额外10条预测smoke另计。六张图已逐张检查。

实际launcher命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_sigtime_rdrop_mixedIQ_e20_20261015_r01/experiments/cvs_mixed_iq20/train_runner.py
```

launcher自动执行source smoke、五worker、逐row predict和fresh CUDA隐藏CPU score。逐row命令、CWD、GPU和resolved config见原experiment.json及launch.json。离线分析实际命令：

```text
F:/App/miniconda3/python.exe -X utf8 experiments/cvs_mixed_iq20/analyze.py --input E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/final --output E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/analysis_r01
F:/App/miniconda3/python.exe -X utf8 experiments/cvs_mixed_iq20/analyze.py --input E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/final --output E:/type10-7/local_artifacts/cvs_mixedIQ_train_20261015/analysis_r01 --trace-only
```

本轮未完成的科学目标：low>60%、clean净提升、全部suburban接近clean、跨seed复现。没有技术缺失测试，没有加载E50/E200或追加预算；当前源规则失败，未扩种子。整体goal保持active。历史报告与产物保持，当前早期过程见progress_history.md。

门控只作机制记录，不能替代识别收益；以下为E1完成后至E20的变动，不冒称初始化差。全部逐row向量和默认头／SIG参数相对初始化的移动保存在原training_mechanism_progress.json。

|row|记录门控|E1均值|E20均值|E1→E20 L2|
|---|---|---:|---:|---:|
|R_LEGACY|PA_gain|0.255829|0.259984|0.479844|
|R_LEGACY|reference_gain|0.000574|0.002958|0.846786|
|R_LEGACY|signature_gain|0.056207|0.078955|0.296271|
|B00|PA_gain|0.255726|0.263912|0.468444|
|B00|reference_gain|0.000404|-0.001722|0.868864|
|B00|signature_gain|0.056887|0.078727|0.287793|
|B10|PA_gain|0.255675|0.264228|0.470648|
|B10|reference_gain|0.000423|-0.001129|0.868081|
|B10|signature_gain|0.056820|0.078655|0.287539|
|B01|PA_gain|0.254713|0.258431|0.555330|
|B01|reference_gain|0.000648|-0.000453|0.713576|
|B01|signature_gain|0.056233|0.082736|0.343538|
|B11|PA_gain|0.254729|0.258810|0.545719|
|B11|reference_gain|0.000449|-0.004509|0.732034|
|B11|signature_gain|0.056229|0.082540|0.341288|
