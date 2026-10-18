# P65：固定同龄 clean 教师监督完整 practical U，E30 最终分析

四条新增训练及各自 clean＋六种固定 practical 预测、独立 truth-last 评分已完成，状态为 **ANALYZED，missing 为空**。附加 U 教学有小幅星地收益，但 clean 代价超过阶段界限；固定同龄 NoAs 教师来源没有表现出可靠的独立优势。

主候选 FC 的 urban 三场景均值为 **64.603175%**，相对原 As30 增加 **0.663889 个百分点**，mid/low urban 分别增加 0.691071/0.797024 个百分点。clean 为 **85.098214%**，相对 As30 下降 0.242857 个百分点，相对同预算同 seed 无星地增强 NoAs30 下降 **0.149405 个百分点**，**未满足 clean 下降不超过 0.1 个百分点的阶段目标**。全部本轮方法 low urban 仍低于 60%；仅一个模型 seed，没有“星地增强同时改善 clean”的可重复证据。

源域判定已在正式成绩读取前固定为 FAIL。保持该关闭结论，不追加 E50、扩 seed，也不扫描 q、教师年龄或系数挽救本路线。整体研发目标仍未完成。

## 1. 配方与执行基线

缩写：ZERO＝ZERO_KL；EA＝EVOLVING_AS_EMA；FA＝FROZEN_AS20_EMA；FC＝FROZEN_NOAS20_EMA。NoAs30/As30 复用 P43 B00/B01 已冻结 E30 默认头预测及独立评分，没有重新训练、预测或评分。

| 行 | 学生起点与预算 | 本轮额外 U 监督 | 解释 |
|---|---|---|---|
| NoAs30 | P43 B00，从头 20＋10 | 无 | 无星地增强历史参考 |
| As30 | P43 B01，从头 20＋10 | 无 | 完整 practical 拼接历史参考 |
| ZERO | 同一 P43 B01 As20 student＋own EMA，新增 10 | EA 目标被计算，系数 0 | 额外前向成本匹配对照 |
| EA | 同一 As20，新增 10，学生累计 30 | 演化中的原 own EMA | 附加教学 |
| FA | 同一 As20，新增 10，学生累计 30 | P43 B01 As20 冻结 EMA | 固定教师作用 |
| FC | 同一 As20，新增 10，学生累计 30 | P43 B00 NoAs20 冻结 EMA | 固定同龄训练来源作用 |

固定结构为 F＋跨 RX 同 TX 对比学习：SIG_TIME_X3、unified Sinc/CENTER_UNITS、160 维联合表示、既有内部门控及参考响应、6×160 cosine 默认头（scale30）。学生 900,722 参数/206 state keys，默认头与网络联合更新。正式和 source panel 均严格加载 FINAL30 的 **student model** 及其默认头，完整 V27,000 末轮健康指标也来自 student；两条冻结外部教师才加载历史 E20 的 ema_model。原 native U 仍使用学生自己的演化 EMA。这些角色不能用旧 HQ 的“冻结 EMA 默认头”称呼混写。

两个外部教师各 900,722 参数，eval、stop-gradient，参数和持久 buffer 不变；不进入学生、优化器、原 own EMA 或部署 state_dict。没有训练新教师或新分类头。原 6 次有标签前向共 768 样本/更新、L 权重 1.25、ordinary 增强、RDrop 对称 KL、ordinary 上的 CROSS_RX SupCon 权重 0.1，以及原 native U 规则均保留。全部合法有标签 As 仍参与 CE。

每个 native U 批次为 256、末批 124；额外教学取前 min(128,n)，不按 TX/q/学生置信度选择，不 padding。四行都执行同一 full-practical U 学生和两个冻结 raw-U 教师前向，包括 ZERO。唯一新增损失为 teacher→urban student KL：q＝clip((teacher pmax−0.95)/0.05,0,1)，温度 1，subset 完整 batch mean；新增前五个 local epoch 升至 0.16。q 仅调节附加 KL，原 mask 不变，没有 urban 学生先高置信的条件，也没有 P39 全局探针禁用门。教师软标签和 q 覆盖共同变化，FC−FA 不能解释成仅标签内容改变的等权实验。

数据保持 L_s/U_s/V＝6,300/56,700/27,000，源 RX 索引 1/3/4/6/8、day1/2/3、六 TX、split392005；没有留出源 RX 或新 Ldev，U TX 真值始终隐藏。25 MHz、IQ256，信道入口 RMS 归一化且 center=false；不反演原录音未知 TX 硬件。训练 low suburban/mid urban/low urban 各 1/3，L/U 流独立，RX/day/eq 硬件 session 稳定且不由 TX 定义。

实际导入旧 evidence-responsibility release 的 experiments/adv3b02_xuc/code/leo_practical/channel.py 及 residual.py，不是仓库根 code 的同名实现。实际路径见 [运行配置](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanTeacher-U-e30-s2026092701-m4-r02/runtime_configuration.json)。完整 residual/post_sync，EQ=false 所以 G=1，质量适配、相位跟踪、IQ 补偿开启，reference_snr_db=30、reflect；训练载频 2.462 GHz，正式 2.45 GHz。正式 IQ、信道、SNR 和样本集合均未改变。

## 2. 正式六行×七视图结果

每个视图 n＝168,000：六 TX×七正式 RX×四日期，每个物理单元 1,000 包。以下均为冻结 FINAL30 学生自身默认分类头，准确率、宏平均 F1、最差 RX 准确率均以百分数表示。

### 准确率（%）

| 视图 | NoAs30 | As30 | ZERO | EA | FA | FC |
|---|---|---|---|---|---|---|
| clean | 85.247619 | 85.341071 | 85.345238 | 85.045833 | 85.052976 | 85.098214 |
| practical_high | 42.685119 | 81.405357 | 81.375595 | 81.797024 | 81.707738 | 81.872024 |
| practical_mid | 40.995238 | 78.661310 | 78.630952 | 79.176786 | 79.070238 | 79.205357 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.820833 | 76.444048 | 76.363095 | 76.500000 |
| practical_high_urban | 41.439881 | 77.767262 | 77.764286 | 78.204167 | 78.127976 | 78.270833 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.769048 | 63.490476 | 63.441667 | 63.497619 |
| practical_low_urban | 30.139881 | 51.244048 | 51.247024 | 52.092857 | 52.037500 | 52.041071 |
| urban 三场景均值 | 35.570833 | 63.939286 | 63.926786 | 64.595833 | 64.535714 | 64.603175 |

### 宏平均 F1（%）

| 视图 | NoAs30 | As30 | ZERO | EA | FA | FC |
|---|---|---|---|---|---|---|
| clean | 84.979636 | 85.044474 | 85.050204 | 84.739161 | 84.743737 | 84.795577 |
| practical_high | 39.311416 | 81.084230 | 81.055992 | 81.503820 | 81.407000 | 81.580485 |
| practical_mid | 37.954813 | 78.359706 | 78.329212 | 78.885350 | 78.765349 | 78.912780 |
| practical_low_suburban | 36.758731 | 75.572741 | 75.532249 | 76.133499 | 76.037901 | 76.190297 |
| practical_high_urban | 38.481030 | 77.423208 | 77.420816 | 77.873620 | 77.781397 | 77.939669 |
| practical_mid_urban | 33.446603 | 62.450463 | 62.410538 | 63.048469 | 62.955176 | 63.058067 |
| practical_low_urban | 29.184589 | 50.887035 | 50.884214 | 51.537337 | 51.418946 | 51.493743 |

### 最差 RX 准确率（%）

| 视图 | NoAs30 | As30 | ZERO | EA | FA | FC |
|---|---|---|---|---|---|---|
| clean | 78.450000 | 78.362500 | 78.545833 | 78.291667 | 78.295833 | 78.404167 |
| practical_high | 34.304167 | 71.908333 | 71.758333 | 72.545833 | 72.187500 | 72.616667 |
| practical_mid | 33.520833 | 70.212500 | 70.075000 | 71.116667 | 70.712500 | 71.091667 |
| practical_low_suburban | 32.054167 | 67.741667 | 67.604167 | 68.637500 | 68.241667 | 68.737500 |
| practical_high_urban | 34.000000 | 69.108333 | 68.950000 | 69.787500 | 69.416667 | 69.850000 |
| practical_mid_urban | 29.495833 | 56.458333 | 56.325000 | 57.341667 | 57.083333 | 57.387500 |
| practical_low_urban | 25.750000 | 46.379167 | 46.341667 | 47.433333 | 47.287500 | 47.391667 |

全部 practical 最差 RX 均为 14-7。clean 的 NoAs30 最差 RX 是 14-7，其余行均为 1-1。完整逐 TX/RX/day 准确率、F1 及 CM 见 [formal_metrics.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/formal_metrics.csv)、[formal_confusions.json](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/formal_confusions.json)。756 次独立混淆矩阵重计数、126 次分区合并一致；全部 6 个逻辑行×7 视图先冻结检查，再进行本次 truth join。

### 配对收益、clean 代价与机制归因

| 新行−对照 | clean Δpp | mid urban Δpp | low urban Δpp | urban 均值 Δpp |
|---|---|---|---|---|
| FC−ZERO | -0.247024 | +0.728571 | +0.794048 | +0.676389 |
| FC−FA | +0.045238 | +0.055952 | +0.003571 | +0.067460 |
| FC−EA | +0.052381 | +0.007143 | -0.051786 | +0.007341 |
| FA−EA | +0.007143 | -0.048810 | -0.055357 | -0.060119 |
| EA−ZERO | -0.299405 | +0.721429 | +0.845833 | +0.669048 |
| FA−ZERO | -0.292262 | +0.672619 | +0.790476 | +0.608929 |
| FC−As30 | -0.242857 | +0.691071 | +0.797024 | +0.663889 |
| FC−NoAs30 | -0.149405 | +28.364881 | +21.901190 | +29.032341 |
| ZERO−As30 | +0.004167 | -0.037500 | +0.002976 | -0.012500 |

FC−ZERO 说明附加 U 教学提高 urban 并损伤 clean。FC−FA 才是固定同龄训练来源的直接对比：urban 仅+0.067460 个百分点，low 仅+0.003571 个百分点；FC−EA 的 low 反而−0.051786 个百分点。FA−EA 的 urban 为−0.060119 个百分点，不能证明冻结本身更好。EA/FA/FC 相对 ZERO 的收益接近，主要可识别作用来自附加 full-practical U 教学，不能归为“固定 clean 教师更准”。单 seed 的小差异不构成稳定优势。

原 As30/NoAs30 仅匹配学生 epoch/seed/角色，没有本轮额外 U 前向和固定教师系统成本。相对 NoAs 的大幅增量主要包含既有 As 训练的收益，不能全归为本轮教师机制。ZERO−As30 的 urban−0.012500、clean+0.004167 个百分点也不是增加零损失前向有效的证据。

| 对照→FC | 视图 | 修复 | 新增错误 | 净变化 | Δpp |
|---|---|---|---|---|---|
| ZERO→FC | clean | 227 | 642 | -415 | -0.247024 |
| ZERO→FC | practical_mid_urban | 2206 | 982 | 1224 | +0.728571 |
| ZERO→FC | practical_low_urban | 2597 | 1263 | 1334 | +0.794048 |
| FA→FC | clean | 195 | 119 | 76 | +0.045238 |
| FA→FC | practical_mid_urban | 541 | 447 | 94 | +0.055952 |
| FA→FC | practical_low_urban | 603 | 597 | 6 | +0.003571 |
| As30→FC | clean | 201 | 609 | -408 | -0.242857 |
| As30→FC | practical_mid_urban | 2176 | 1015 | 1161 | +0.691071 |
| As30→FC | practical_low_urban | 2608 | 1269 | 1339 | +0.797024 |
| NoAs30→FC | clean | 3777 | 4028 | -251 | -0.149405 |
| NoAs30→FC | practical_mid_urban | 54485 | 6832 | 47653 | +28.364881 |
| NoAs30→FC | practical_low_urban | 45760 | 8966 | 36794 | +21.901190 |

FC−EA 总体差来自指标直接相减，既定产物未单独保存这组逐包修复/新增，不能将 FC−FA 和 FA−EA 修复数相加。其余 13 组既定配对完整保留于 [formal_paired_effects.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/formal_paired_effects.csv)。

### 全部正式 RX 与日期

| 分层 | 每视图 n | FC−ZERO clean | mid | low | FC−FA clean | mid | low |
|---|---|---|---|---|---|---|---|
| receiver/1-1 | 24000 | -0.141667 | +0.695833 | +0.870833 | +0.108333 | -0.020833 | -0.100000 |
| receiver/14-7 | 24000 | -0.100000 | +1.062500 | +1.050000 | +0.208333 | +0.304167 | +0.104167 |
| receiver/2-1 | 24000 | -0.291667 | +0.462500 | +0.641667 | -0.008333 | +0.000000 | +0.054167 |
| receiver/20-1 | 24000 | -0.275000 | +0.887500 | +0.883333 | -0.045833 | +0.129167 | +0.133333 |
| receiver/7-14 | 24000 | -0.200000 | +0.787500 | +0.775000 | +0.100000 | +0.008333 | -0.116667 |
| receiver/7-7 | 24000 | -0.108333 | +0.683333 | +0.750000 | +0.079167 | -0.045833 | -0.104167 |
| receiver/8-8 | 24000 | -0.612500 | +0.520833 | +0.587500 | -0.125000 | +0.016667 | +0.054167 |
| day/2021_03_01 | 42000 | -0.097619 | +0.823810 | +0.664286 | +0.104762 | +0.080952 | +0.000000 |
| day/2021_03_08 | 42000 | -0.464286 | +0.740476 | +0.988095 | +0.002381 | +0.107143 | +0.059524 |
| day/2021_03_15 | 42000 | -0.147619 | +0.669048 | +0.750000 | +0.033333 | +0.000000 | +0.007143 |
| day/2021_03_23 | 42000 | -0.278571 | +0.680952 | +0.773810 | +0.040476 | +0.035714 | -0.052381 |

FC−ZERO 的 mid/low 在全部 7 RX、4 日期均正，clean 均负。FC−FA 的 low 在 3/7 RX 下降、4/7 RX 上升，四日期有零、正、负；整体净增六包不能称为跨 RX 稳定提升。六行全部 TX/RX/day 结果没有选择性删除，完整 CSV 可核查。

### 固定 clean 正确集合与默认头间隔

可用固定集合为 **NoAs30 clean 正确 143,216 包**，不是实际 As20 父模型的正确集合。明确保留这个替代标签，不冒称父模型保持率。

| 行 | clean 保持% | mid 保持% | low 保持% | clean margin/30 | mid margin/30 | low margin/30 |
|---|---|---|---|---|---|---|
| NoAs30 | 100.000000 | 38.619288 | 32.807089 | 0.140333742 | -0.033064462 | -0.041066073 |
| As30 | 97.408809 | 68.782119 | 55.632751 | 0.146124229 | 0.053809803 | 0.028075527 |
| ZERO | 97.414395 | 68.743716 | 55.638336 | 0.145083576 | 0.053696424 | 0.028059963 |
| EA | 97.149760 | 69.551586 | 56.606105 | 0.146132603 | 0.058990031 | 0.032208782 |
| FA | 97.148363 | 69.513881 | 56.585158 | 0.146840483 | 0.058988757 | 0.032152452 |
| FC | 97.187465 | 69.554379 | 56.539074 | 0.146935239 | 0.059285197 | 0.032347668 |

FC 对 As30 的平均 margin/30 上升，clean 准确率与固定集合 clean 保持率仍下降。FC 对 FA 的 low 总体净增六包，但该固定集合 low 保持率反略低。平均间隔改善不能替代分类收益或原身份集合保持。

## 3. 正式成绩读取前固定的源判定

源域使用完整 V27,000 clean 指标，以及同一 V1,800 物理样本×两个命名信道 draw。每个 practical pooled n＝3,600；draw 不是模型 seed。仅 FC 为候选，ZERO/EA/FA 为控制。生产者原判定先于 formal 读取，分析未重算晋级，正式成绩没有回流。

| 预登记条件 | 实测 | 结果 |
|---|---|---|
| FC fullV clean−NoAs30 ≥−0.1 pp | −0.170370 pp | FAIL |
| FC mid/low 均值−As30 ≥+0.5 pp | +0.750000 pp | PASS |
| FC low−As30 ≥0 | +0.750000 pp | PASS |
| FC mid/low 均值−FA ≥+0.5 pp | 0.000000 pp | FAIL |
| FC mid/low 均值−ZERO ≥+0.5 pp | +0.694444 pp | PASS |
| FC low−FA ≥0 | −0.083333 pp | FAIL |

总判定 **SOURCE_FAIL，3/6 条件未通过**。[原源判定](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanTeacher-U-e30-s2026092701-m4-r02/source_matrix_decision.json)保存 formal_scores_read=false、target_scores_read=false，正式评价不改变关闭结论。

| 源指标 | NoAs30 | As30 | ZERO | EA | FA | FC |
|---|---|---|---|---|---|---|
| full V clean% | 98.359259 | 98.255556 | 98.237037 | 98.192593 | 98.196296 | 98.188889 |
| panel clean% | 98.444444 | 98.333333 | 98.277778 | 98.277778 | 98.277778 | 98.277778 |
| panel low suburban% | 49.861111 | 88.666667 | 88.694444 | 89.027778 | 89.055556 | 88.972222 |
| panel mid urban% | 43.555556 | 72.805556 | 72.916667 | 73.500000 | 73.472222 | 73.555556 |
| panel low urban% | 35.611111 | 59.083333 | 59.083333 | 59.916667 | 59.916667 | 59.833333 |

FC−As30 源 mid/low 均为+0.75 个百分点；FC−FA 的 mid+0.083333、low−0.083333，合并为 0。full V 最差源 RX 均为索引 3，FC 为 95.148148%，NoAs30 为 95.611111%、As30 为 95.333333%。小 panel clean 持平不能抵消完整 V 上的 clean 损伤。

### 状态、锁定与质量分层

以下完整列出非空 state×lock 格及质量区间，格式为“Δpp（修复/新增）”。全部原始 acc/F1/margin、独立 state/lock、空格和 TX/RX/day 见 [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/source_strata.csv)、[source_confusions.json](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/source_confusions.json)、[source_paired_effects.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/source_paired_effects.csv)。

#### practical_mid_urban

| 分层 | n | FC−As30 | FC−ZERO | FC−FA |
|---|---|---|---|---|
| state_lock/LOS/locked | 1899 | +0.315956（7/1） | +0.157978（6/3） | +0.157978（4/1） |
| state_lock/PARTIAL/locked | 306 | +0.000000（1/1） | +0.000000（1/1） | -0.326797（0/1） |
| state_lock/PARTIAL/degraded | 59 | +5.084746（3/0） | +5.084746（3/0） | +0.000000（0/0） |
| state_lock/PARTIAL/unlocked | 2 | +0.000000（0/0） | +0.000000（0/0） | +0.000000（0/0） |
| state_lock/BLOCKED/degraded | 656 | +1.829268（19/7） | +1.829268（19/7） | +0.000000（4/4） |
| state_lock/BLOCKED/unlocked | 678 | +0.884956（13/7） | +0.737463（13/8） | +0.147493（4/3） |
| signal_to_added_noise_db/<3 | 383 | +1.827676（9/2） | +1.827676（9/2） | -0.261097（2/3） |
| signal_to_added_noise_db/3-8 | 718 | +0.557103（14/10） | +0.417827（14/11） | +0.557103（5/1） |
| signal_to_added_noise_db/8-15 | 293 | +3.412969（12/2） | +3.412969（12/2） | -0.682594（1/3） |
| signal_to_added_noise_db/>=15 | 2206 | +0.271985（8/2） | +0.135993（7/4） | +0.090662（4/2） |
| receiver_quality_proxy_db/<5 | 680 | +0.882353（13/7） | +0.735294（13/8） | +0.147059（4/3） |
| receiver_quality_proxy_db/5-15 | 715 | +2.097902（22/7） | +2.097902（22/7） | +0.000000（4/4） |
| receiver_quality_proxy_db/>=15 | 2205 | +0.272109（8/2） | +0.136054（7/4） | +0.090703（4/2） |

#### practical_low_urban

| 分层 | n | FC−As30 | FC−ZERO | FC−FA |
|---|---|---|---|---|
| state_lock/LOS/locked | 1060 | +0.377358（5/1） | +0.377358（5/1） | +0.188679（2/0） |
| state_lock/PARTIAL/locked | 307 | +0.651466（3/1） | +0.651466（3/1） | +0.000000（0/0） |
| state_lock/PARTIAL/degraded | 253 | +0.000000（2/2） | +0.395257（3/2） | +0.000000（0/0） |
| state_lock/PARTIAL/unlocked | 1 | +0.000000（0/0） | +0.000000（0/0） | +0.000000（0/0） |
| state_lock/BLOCKED/degraded | 414 | +1.207729（7/2） | +1.690821（9/2） | +0.000000（1/1） |
| state_lock/BLOCKED/unlocked | 1565 | +1.022364（25/9） | +0.830671（28/15） | -0.319489（5/10） |
| signal_to_added_noise_db/<3 | 1127 | +1.242236（21/7） | +0.887311（22/12） | -0.443656（4/9） |
| signal_to_added_noise_db/3-8 | 794 | +0.755668（10/4） | +1.133501（14/5） | +0.000000（2/2） |
| signal_to_added_noise_db/8-15 | 303 | +0.330033（3/2） | +0.660066（4/2） | +0.000000（0/0） |
| signal_to_added_noise_db/>=15 | 1376 | +0.436047（8/2） | +0.436047（8/2） | +0.145349（2/0） |
| receiver_quality_proxy_db/<5 | 1566 | +1.021711（25/9） | +0.830140（28/15） | -0.319285（5/10） |
| receiver_quality_proxy_db/5-15 | 667 | +0.749625（9/4） | +1.199400（12/4） | +0.000000（1/1） |
| receiver_quality_proxy_db/>=15 | 1367 | +0.438917（8/2） | +0.438917（8/2） | +0.146306（2/0） |

聚合 state 的 FC−As30：mid LOS/PARTIAL/BLOCKED 为+0.315956/+0.817439/+1.349325 pp，low 为+0.377358/+0.356506/+1.061142 pp。聚合 lock 的 mid locked/degraded/unlocked 为+0.272109/+2.097902/+0.882353 pp，low 为+0.438917/+0.749625/+1.021711 pp。源域 BLOCKED、degraded/unlocked 确有部分错误修复关联：mid BLOCKED/degraded n656 修复 19/新增 7，low BLOCKED/unlocked n1,565 修复 25/新增 9。

这些改善大多 FA 也有：FC−FA 的 mid PARTIAL/degraded、BLOCKED/degraded 均为 0，low BLOCKED/unlocked 反−0.319489 pp（5 修复/10 新增）。这支持附加 urban 教学适应了部分受损观测的观察，不证明 NoAs 来源专门解决某一物理损伤。mid PARTIAL/degraded +5.084746 pp 仅净修复 3 包、n59；PARTIAL/unlocked 仅 2/1 包，不作强结论。

新增噪声比率不是原录音总 SNR。质量代理<5、5–15、≥15 dB 对应 unlocked/degraded/locked，不能当作独立重复证据。没有执行本轮固定接收状态的移除损伤反事实；正式缺少逐包 state/lock 真值。因此不能把源状态关联直接宣称为正式环境中的多径、Doppler、相位、RX IQ 或边界因果改善。

### 全部源 RX 与 day

| 源分层 | 每 scene n | FC−As mid | low | FC−FA mid | low | FC−ZERO mid | low |
|---|---|---|---|---|---|---|---|
| RX/1 | 720 | +0.277778 | +0.416667 | -0.555556 | +0.277778 | +0.000000 | +0.000000 |
| RX/3 | 720 | -0.138889 | +0.416667 | +0.694444 | -0.416667 | -0.138889 | +0.277778 |
| RX/4 | 720 | +1.250000 | +1.388889 | -0.138889 | +0.277778 | +1.388889 | +1.527778 |
| RX/6 | 720 | +0.833333 | +0.833333 | +0.138889 | -0.138889 | +0.833333 | +0.833333 |
| RX/8 | 720 | +1.527778 | +0.694444 | +0.277778 | -0.416667 | +1.111111 | +1.111111 |
| day/1 | 1200 | +0.416667 | +0.916667 | -0.083333 | -0.083333 | +0.083333 | +1.000000 |
| day/2 | 1200 | +1.166667 | +0.250000 | +0.166667 | -0.166667 | +1.166667 | +0.166667 |
| day/3 | 1200 | +0.666667 | +1.083333 | +0.166667 | +0.000000 | +0.666667 | +1.083333 |

这里没有留出任何训练域，也不要求先独立证明额外泛化。源角色与正式角色不可互换。源 panel 未包含 high urban，full V 没有保存 F1；缺项不以其他范围指标冒充。

## 4. 完整日志、数值健康和覆盖

四条新增尾段各 E21 至 E30、2,220 更新，共 8,880 更新，全部 step/epoch 已解析。每行原 L 前向 1,704,960、L satellite IQ142,080、原 U 曝光 567,000。4,440 条 L/U 采样记录与 As30 一致，ordinary 逐步 MSE 指纹差异 0；四行额外 U 物理前缀和信道 summary 相同，private dropout RNG 恢复。不将此扩大为未保存 IQ 字节或所有内部 RNG 状态的逐字比较。

| 行 | native loss 首→尾 | epoch 骨干梯度范围 | 原 U 选中/567000 | 额外 q 均值% | 额外 q>0 比例% | qKL/step | 加权 qKL/step |
|---|---|---|---|---|---|---|---|
| ZERO | 0.623339→0.604291 | 1.221769 至 1.684289 | 8723 | 83.137927 | 91.749613 | 0.529427 | 0.000000 |
| EA | 0.639593→0.684849 | 1.248955 至 1.702434 | 8722 | 83.198687 | 91.759468 | 0.515338 | 0.065895 |
| FA | 0.639689→0.675789 | 1.241193 至 1.700180 | 8718 | 79.057991 | 90.094678 | 0.481233 | 0.060791 |
| FC | 0.640813→0.682408 | 1.250324 至 1.707070 | 8715 | 83.525801 | 92.439462 | 0.520770 | 0.065797 |

原 native U mask 约选中 1.54%，附加 q>0 覆盖约 90%至 92%，二者不同；旧 mask 低覆盖不代表新增教学未运行。冻结 FA/FC 在相同 U 上的预测分歧为 2,948/284,120＝1.037590%，FC 的 q 均值 83.525801%高于 FA 的 79.057991%。这是置信、覆盖和教学强度的来源差异，U 真值隐藏，不能声称 FC 标签更准。

全部梯度健康记录有限，没有非有限损失/梯度跳步；原 L 分解最大误差不超过 1.602×10⁻⁷。冻结教师参数和持久 buffer 不变。KL 权重前五轮爬升，native loss 首尾不能作为相同目标下的直接收敛比较。[完整 100 条 epoch 曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/training_curves.csv)保留日志中的 source V TX accuracy 标签：epoch 验证日志没有明确 student/EMA 身份，不擅自将该曲线称为 EMA；末轮 completion、source panel 与 formal 则已确认 student。没有分损失参数梯度或分支/门控完整轨迹，不推造其独立贡献，预算耗尽不等于收敛。

EA/FA/FC 各 2,220 步加权 KL 均非零，成熟权重 0.16 覆盖 1,332 步、170,472 个额外 U 观测。额外 U 每行实际 284,120，尾批 124，不是理论满批 284,160；双冻结教师各 284,120，共额外 852,360 NN。训练时外部参考模型合计 1,801,444 参数，部署仍 900,722 参数。

| 每行覆盖 | L satellite | 额外 U satellite |
|---|---|---|
| 总 IQ | 142080 | 284120 |
| lowSub / midUrban / lowUrban | 47525 / 47116 / 47439 | 94660 / 95245 / 94215 |
| LOS / PARTIAL / BLOCKED | 71872 / 22794 / 47414 | 144184 / 44836 / 95100 |
| locked / degraded / unlocked | 86115 / 23865 / 32100 | 171931 / 47829 / 64360 |
| 硬件 session | 15 | 15 |
| 状态格 | 810 展开格，515 非空 | 268 个有观测 RX/day/scene/state/lock 格 |

L 全部 6TX×5 源 RX×3 场景 90 组合有支持，物理不可达格不要求非空。U 覆盖全部 45 个 RX×day×scene 组合、15 个 RX/day session，以及 56,643 个唯一额外 U 物理样本。U 覆盖无 TX 真值，不能将伪标签直方图冒充真实 TX 覆盖。见 [L 覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/training_coverage.csv)、[无 TX 的 U 覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/training_U_coverage.csv) 和四份*_clean_teacher_steps.csv。

## 5. 实际成本与技术恢复

| 尾段 | 新增 E/更新 | native NN | 额外 student U | 额外 frozen teacher | 训练总 NN | 训练墙钟 s |
|---|---|---|---|---|---|---|
| ZERO | 10 / 2220 | 3162962 | 284120 | 568240 | 4015322 | 1636.524130 |
| EA | 10 / 2220 | 3162962 | 284120 | 568240 | 4015322 | 1674.320338 |
| FA | 10 / 2220 | 3162962 | 284120 | 568240 | 4015322 | 1700.893306 |
| FC | 10 / 2220 | 3162962 | 284120 | 568240 | 4015322 | 1611.396438 |

四行训练墙钟相加 6,623.134211 s，不是并行总历时。本轮 artifact envelope 为 1,824.828845 s（dispatcher metadata 至 analysis complete 的 mtime 差，包含队列、初始化、训练、源 readout、预测、评分及分析），不等同纯训练墙钟。native encoder 计数先扣除额外 student，再单列加回；extra_U_NN 为 extra_student_U_NN 别名，未二次相加。

| 成本项 | 实际值 |
|---|---|
| r02 新增训练 | 40 model E / 8880 更新 / 16061288 NN |
| r02 源 readout | 50400 NN；7.306966 s |
| r02 正式预测 | 4704008 NN；194.420766 s |
| r02 真实 checkpoint no-query smoke | 6 NN / 2 unique CP / 0 IQ / 0 更新 |
| r02 新 remote 合计 | 20815702 NN |
| r01 技术失败 | 重建初始化eval296＋独立smoke6＝302 NN；0训练E/优化更新/L或U训练前向/正式预测/truth评分 |
| 本次新增 remote，含 r01 失败 | 20816004 NN |
| 共享一次本地 actualF | 6656 NN，函数 12.045978 s；不混入 remote |
| CPU 最终分析 | 54.282939 s；0 NN/IQ/CP/更新 |
| 复用历史两 E30 参考 | 60 model E / 13320 更新 / 18298572 已知 remote NN |
| 历史＋本轮唯一总量，含 r01 失败 | 100 model E / 22200 更新 / 39114576 已知 remote NN |

历史共享六行 smoke 未分配，未计入已知历史总量，保留这一不完备范围。原参考新增重复预测/评分为 0。每学生预算 As20＋10＝E30/6,660 更新；系统唯一前提 As20 及 NoAs20 各 4,797,722 NN，墙钟分别 1,810.249311/1,329.394069 s，共 40 历史 model E/8,880 更新。As20 兼学生/own EMA/冻结教师只计一次，NoAs20 已在 NoAs30 历史中，不重复计。单行系统前提 40＋新增 10＝50 唯一 model E/11,100 更新，不等于学生 E50；四行共享历史后矩阵 100 model E。

r01 在第 1 更新前因安装后 native 回调内调用原 builder 构造教师而递归，失败层为教师初始化，四行自然退出且原产物保留。修复用已构造同构学生 deepcopy，清除额外 U 回调，严格加载各自 EMA，外部 holder/eval/stop-gradient 和参数不共享保持。新不可变 r02 执行原同 seed/同四行/同 E30 矩阵，不是基于科学成绩的选择性重跑。r01 额外 302 NN 单列（根据完整失败栈及固定源码重建，并非运行时counter），本地 actualF 不重复。

## 6. 实际图表、文件与命令

以下 5 组真实 PNG/PDF 均已生成，主执行者已逐张完成全部 5 个 PNG 视觉 QA：可读、无空白，legend 和源空格标注保留正确。

- 完整训练曲线：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/training_curves.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/training_curves.pdf)
- U 教学 q/KL/教师分歧/NN 曲线：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/clean_teacher_training.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/clean_teacher_training.pdf)
- 源 mid 状态分层：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/practical_mid_urban_source_states.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/practical_mid_urban_source_states.pdf)
- 源 low 状态分层：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/practical_low_urban_source_states.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/practical_low_urban_source_states.pdf)
- 正式准确率：[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/formal_accuracy.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/formal_accuracy.pdf)

主要机读证据：[最终状态](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/complete.json)、[生产者成本](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/costs.json)、[正式汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/formal_summary.json)、[配对汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/formal_pair_summary.csv)、[源汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/source_summary.json)、[完整尾段解析](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02/retrieved_r01/analysis_r01/new_tail_training.json)。本报告在生产者 costs 之外补入 r01 失败 302 NN（37次完成的两包初始化eval/row×4＝296，加独立smoke6；完整栈与固定源码重建），未改生产者文件；上级 retrieved 及远端原 run 保留四行完整 source/result/completion/epoch/step 日志和独立评分。

PROJECT 指既定 N607 项目根，CVS_PYTHON 指现有 CVS-RFFI 环境 Python；使用公开别名，不将私有账户路径写入 Git。实际环境 Python3.10.19/Torch2.1.0+cu121/NumPy2.2.5、full FP32、CPU2；CWD 为 PROJECT/releases/cvs_FcrossRX_clean_teacher_U_E30_20261018_r02，四行 GPU0/1/2/3，未停止或修改其他健康任务。实际 argv 在 retrieved/launch.json 及 analysis_launch.json。

```text
CVS_PYTHON -u experiments/cvs_clean_teacher65/runner.py --mode worker --row ZERO_KL-s2026092701
CVS_PYTHON -u experiments/cvs_clean_teacher65/runner.py --mode worker --row EVOLVING_AS_EMA-s2026092701
CVS_PYTHON -u experiments/cvs_clean_teacher65/runner.py --mode worker --row FROZEN_AS20_EMA-s2026092701
CVS_PYTHON -u experiments/cvs_clean_teacher65/runner.py --mode worker --row FROZEN_NOAS20_EMA-s2026092701
CVS_PYTHON -u analysis/cvs_clean_teacher65_analysis_20261018.py --project PROJECT --out PROJECT/runs/20261018-phase1-FcrossRX-cleanTeacher-U-e30-s2026092701-m4-r02/analysis_r01 --local-nn 6656 --code-commit 603843ae58912b68ed9a8ff48fdd418c64a24231
```

worker 自动串联训练、final-only 冻结、七环境预测、独立 truth-last 评分；分析没有启动新模型或 scorer、无新增前向/IQ/CP 读取。此为已暴露固定基准，不声称新的独立盲测。

### 实现、检查与 commit

实现目录 [experiments/cvs_clean_teacher65](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_clean_teacher65) 包含：__init__.py、design.py、teaching.py、model.py、channel.py、rng.py、source.py、observer.py、protocol.py、preflight.py、quality.py、qualification.py、evaluate.py、dispatch.py、runner.py、report.py、checks.py、teaching_checks.py、flow_checks.py、synthetic.py、local_synthetic_entry.py。分析为 [analysis 脚本](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_clean_teacher65_analysis_20261018.py) 与 [analysis 检查](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_clean_teacher65_analysis_checks_20261018.py)。

已执行 59 协议负例、8 教学 toy、4 流程检查、一次四 arm 真实 F6656 NN。原 U256mask/extra128、RNG 边界、外部教师冻结及学生默认头/Sinc 梯度通过。随机未训练合成教师 q 全 0，actualF 新增 KL 梯度 0 符合公式；阳性 q 的非零教学梯度由 toy 验证，未伪称 actualF 阳性。唯一 P0/P1 发现双 checkpoint 元数据被旧分析拒绝，修复后新增 1 个 0 NN 元数据 toy 及一次定向复核通过；初始 12 分析 toy 3.444 s，元数据 toy0.251 s，原 actualF 未重跑。

- [59/8/4 及 actualF 证据](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r01/local_verification.json)
- [12toy 实际命令/耗时](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r01/analysis_checks_execution.json)
- [12toy 输出](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r01/analysis_checks.stdout.txt)
- [元数据修复命令/耗时](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r01/analysis_metadata_fix_execution.json)
- [元数据修复输出](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r01/analysis_metadata_fix.stdout.txt)
- [构造修复检查](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r01/constructor_fix_checks.json)
- [构造修复输出](E:/type10-7/local_artifacts/cvs_FcrossRX_clean_teacher_U_E30_20261018_r01/constructor_fix.stdout.txt)

构造递归修复回归 1 项：0 NN/CP/IQ/更新，helper0.238351 s/process2.772195 s；独立定向复核同项 0 NN，helper0.285698 s/process2.781684 s，builder 重入与 forward 均 0，没有重复完整审查或 actualF。r02 一次 archive 本地/远端比较、23py compile、真实 2CP/6 NN no-query smoke 及 PID/CWD/argv/GPU/日志增长核实见 [发布核实](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanTeacher-U-e30-s2026092701-m4-r02/launch_verification.json)、[修复证据](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanTeacher-U-e30-s2026092701-m4-r02/verification.json)。

r01 原实现 commit：8bfc5f53883933a6e069526dc05793340f805370；r02 实际运行代码 commit：603843ae58912b68ed9a8ff48fdd418c64a24231。失败产物保留，未覆盖。最终结果和登记与本报告一并提交；交付消息给出实际元数据 commit，不构造自引用 commit 链。

## 7. 限制与最终回答

实际 As20 父模型的正式 clean 正确集合未测；143,216 包明确来自 NoAs30。正式无逐包信道 state/lock truth，源诊断未做本轮固定接收状态的移除损伤反事实；无分支/门控完整因果轨迹或分损失参数梯度。U 教师准确率未知，源 high urban 及 full V F1 未记录。只有一个模型 seed，两个 draw、多 RX 或日期均不能替代跨 seed 重现。全部限制保留，平均 margin、q 或单 seed 微涨不能替代目标。

固定 CVS 与固定 practical 测试下，附加完整 practical U 教学相对 ZERO 改善 mid/low；源 BLOCKED、degraded/unlocked 存在错误修复关联，但不能定位为某种物理损伤被单独缓解。主候选学生默认头对 As30 的 urban 均值+0.663889、mid+0.691071、low+0.797024 pp，clean 代价−0.242857 pp；对同预算同 seed NoAs30 的 clean 代价−0.149405 pp，**未满足 0.1 pp 阶段目标**。固定同龄 NoAs 教师对固定 As 教师的 low 仅净增六包，尚无来源优势的重复证据；**未获得星地增强同时改善 clean 的可重复证据，low urban>60%也未达到**。

本次固定教师来源方案按预登记 SOURCE_FAIL 关闭，不扩预算/seed，不用正式成绩调参或选择性重跑。整体研发目标仍 active，尚未完成。

成本更正证据：先前6NN仅覆盖独立preflight，漏记失败递归内已完成的初始化eval。四份完整栈各37个checked_model103，均已完成两包forward及finite检查；压缩块与最终异常仅在纯builder重入，因此重建初始化296NN，加独立6NN＝302NN。该修正无新模型运行。详见[更正记录](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanTeacher-U-e30-s2026092701-m4-r02/initialization_cost_correction.json)。历史共享smoke未分配限制保持。
