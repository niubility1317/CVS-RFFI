# 逐物理包六场景均衡：三组匹配 E20 结果

**ANALYZED／VERIFIED，FAIL_FIXED_PER_PHYSICAL_PACKET_SIX_SCENE_SCHEDULING_CLOSED。** [固定预登记](preregistration.md)、[实际配置](experiment.json)和[完整复算](source_analysis.json)保留。F＋跨 RX 同 TX、三 SIG_TIME／统一 Sinc／joint160／NoExplicitCFO／默认 cosine30 头正常联合训练；三组 scratchE20／4440 更新／原13L＋7U，没有训练父权重、另训头或部署分支。

结果只属于 sourceRX1-held／seed2026092701。fit RX3/4/6/8，L5040／U45360（TX真值隐藏）／V21600；held原L1260／day1/2/3／6TX，仅FINAL20源读出，不混同fullL6300或L_train5580/L_dev720。E20来自既有测试基准暴露后的压缩探索，当前没有正式target访问，不能声称新独立盲测。

## 默认头识别结果

| 场景 | B00 accuracy % | C_IID accuracy % | B01 accuracy % | B01−B00 pp | B01−C_IID pp |
|---|---:|---:|---:|---:|---:|
| clean | 80.158730 | 84.444444 | 84.682540 | +4.523810 | +0.238095 |
| practical_high | 39.920635 | 59.365079 | 60.555556 | +20.634921 | +1.190476 |
| practical_mid | 40.238095 | 55.714286 | 56.825397 | +16.587302 | +1.111111 |
| practical_low_suburban | 36.349206 | 52.619048 | 54.444444 | +18.095238 | +1.825397 |
| practical_high_urban | 38.968254 | 54.365079 | 55.317460 | +16.349206 | +0.952381 |
| practical_mid_urban | 32.460317 | 43.492063 | 43.888889 | +11.428571 | +0.396825 |
| practical_low_urban | 27.777778 | 33.888889 | 34.047619 | +6.269841 | +0.158730 |

七场景重复同一1260物理包，不称8820个独立样本；全部26460硬判决、macro-F1、混淆矩阵、TX/day及修复／新错已独立复算。硬预测不含logits与传播状态，因此运行时CE单列，不冒称重算CE、margin或识别机制。

| 对照 | clean macro-F1变化 pp | low urban macro-F1变化 pp | urban均值变化 pp | seenV变化 pp | clean修复／新错 | low修复／新错 |
|---|---:|---:|---:|---:|---|---|
| B01−B00 | +4.542045 | +8.826562 | +11.349206 | +0.004630 | 75/18 | 201/122 |
| B01−C_IID | +0.252714 | +0.281033 | +0.502646 | -0.055556 | 6/3 | 20/18 |

| 原固定条件 | 结果 |
|---|---|
| B00_clean | PASS |
| B00_clean_macroF1 | PASS |
| B00_low_urban | PASS |
| B00_low_urban_macroF1 | PASS |
| B00_mid_urban | PASS |
| B00_urban_mean | PASS |
| B00_each_practical | PASS |
| B00_seen_V | PASS |
| C_IID_clean | PASS |
| C_IID_clean_macroF1 | PASS |
| C_IID_low_urban | FAIL |
| C_IID_low_urban_macroF1 | PASS |
| C_IID_mid_urban | PASS |
| C_IID_urban_mean | FAIL |
| C_IID_each_practical | PASS |
| C_IID_seen_V | PASS |
| healthy | PASS |

B01相对C_IID须clean和cleanF1不降、lowurban≥＋1pp且F1不降、midurban不降、urban均值≥＋1pp；相对B00须clean及F1不降、lowurban≥＋5pp且F1不降、midurban不降、urban均值≥＋3pp；两个对照下每practical及seenV≥−.5pp。判据没有根据结果改变。

| 行 | high suburban距ownclean pp | mid suburban距ownclean pp | low suburban距ownclean pp |
|---|---:|---:|---:|
| B00 | 40.238095 | 39.920635 | 43.809524 |
| C_IID | 25.079365 | 28.730159 | 31.825397 |
| B01 | 24.126984 | 27.857143 | 30.238095 |

每个suburban距自身clean≤2pp仍为探索目标；上表与绝对accuracy同时报告，不能降低clean来制造接近。

## 唯一变化、覆盖和成本

共同完整base保留raw.75＋ordinary.25＋原ordinary-only浅／中层跨RX同TX SupCon。另加.25R(view)，R为原LS.01的CE均值＋同一已生成IQ两次网络前向的RDrop1 KL。B00额外raw、C_IID额外随机单As(raw)、B01额外逐物理包均衡单As(raw)；三者8L前向／总pairweight1.25，未稀释原base绝对份额，raw相对占比差异明确，不添加跨视图损失。原U／oldgate／ownEMA保持，无新U或teacher视图，全部合法L As直接监督。

B01同包连续6次出现恰好覆盖六scene，随机排列seed202610175001与物理pid／cycle绑定，不用标签／预测／质量／难度。保留原IID私有抽样工作、原传播noise seed、receiver session、归一化、质量／补偿／AGC及reflect边界；同scene同packet同step沿原fullpractical。只改变同一包跨出现的scene相关性，边际概率均匀；没有提高urban概率、连续叠加两次完整信道或改变测试SNR/receivedIQ。source_eval委托原实现，训练scheduler不参与测试。

旧IID历史4440步／568320曝光精确复算：所有5040包均见过全部六scene、缺失为0，次数差中位11／最大27；因此否定漏场景解释。当前独立reader将两As行全部1136640个实际scene决策与原L物理顺序逐一重建，最终5040逐包计数与运行时完全一致；0新IQ/CP/NN/更新。覆盖更均匀本身不作为科学成功。

| 行 | scene调度 | As训练包数 | 同包六scene次数差：中位／最大 | 完整base最终目标 | 实际NN包前向 | 训练秒 | channel生成/传输秒 |
|---|---|---:|---|---:|---:|---:|---:|
| B00 | none | 0 | 无As | 0.31048761 | 5823318 | 1403.311 | 0.000 |
| C_IID | iid | 568320 | 11.0/27 | 0.32166711 | 5823318 | 2310.321 | 871.221 |
| B01 | balanced | 568320 | 1.0/1 | 0.32069408 | 5823318 | 2322.832 | 901.205 |

全部13320 native／view／Sinc unit记录与60个epoch行、L/U物理序列、loss公式／系数／BN／原普通增强配置已核对。三组实际nativeargs仅candidate_id/output_dir不同；每组LNN4546560、其中extra1136640、ordinary568320。两As行各568320次完整信道，B00为0；U学生／teacher／V／8820held每行分别记录，teacher初始继承counter2按既有实际计量扣除，不算额外前向。总NN 17469956 含dispatcher真实匹配CP两合成包smoke2；本地工程合成NN4114／更新3／fullpractical520／调度1921／ordinary128单列，历史训练不混入本轮scratch成本。

真实同契约CP仅用于launcher第一步合成兼容smoke，PASS立即继续，从未训练初始化。一次P0/P1 PASS、一次普通N607预检、一份archive传输比较及remotecompile；实际PID/CWD/argv/GPU/日志增长曾独立核实。所有训练worker与dispatcher自然退出，无训练技术failure或重跑，健康旧任务未改。独立读取器曾误读训练序列中不存在的step字段；按原RecordingLoader的role/IDs格式修正后，使用同一固定预测和日志全量复算，0新增NN，未改模型或判据。执行commit c9a74172e5f81bb4f04ce12cd4386ad4dd953ee9。

全部曲线、TX/day分层与七场景/gap图保留在local_artifacts/F_physical_scene_cycle_E20_20261017/final/；原IQ、checkpoint、逐包physicalID／预测和全日志不提交Git。

## 科学判定与剩余目标

固定逐包六scene循环未通过预登记识别条件，关闭；不改顺序seed／cycle长度／边际概率／系数／RX／modelseed／预算救跑。更均衡的计数未充分转化为规定的识别收益，不以loss下降或健康检查替代。

完整联合目标保持ACTIVE：仍需有依据且有效的Ag、稳定As、同完整训练契约B00/B10/B01/B11及交互项、跨seed与正式clean＋六practical独立truth-last，并同时满足clean净收益／lowurban>60%／suburban接近自身clean。本轮为As组织消融，不构造虚假的Ag×As交互项。


### 错误转移与图表

相对随机As，均衡组lowurban修复20／新错18，净增加2个正确样本；两组clean均正确的1061个样本内，lowurban修复17／新错18，净减少1个。该分组仅为描述性统计，不是新增选模门槛或传播机制因果证明。完整组别及预测流见[错误转移](error_transitions.md)与[聚合数据](error_transitions.json)，TX/day负结果全部保留。

三张PNG／PDF均已实际查看：完整20轮曲线、七场景与suburban差距、全部TX/day分层。图仅基于独立聚合结果生成，0源IQ／NN／更新；记录目录见上文。Git中保存本轮analyze.py与recount_schedule.py的独立读取逻辑；相对于实际执行副本，只调整默认本地产物路径，不改变算法、条件或数据。

公开聚合JSON保留全部识别、混淆、TX/day、配对指标；训练覆盖公开各维计数与非零联合格数，完整联合训练计数和逐包记录留在原私有产物，避免将大批运行输出提交Git。
