# MILKR 源跨RX对比有界残差编码

R41在新拟合的R40相关性加权CCA上增加源域神经残差编码器。新增NN使用合法L标签监督训练，不能称整个方法无神经更新；Tri-SIG-TIME SSL基座、DURA、原默认头及buffers保持冻结。NN输出每block64维表示，不另训六类分类头，原C4 head概率函数保持，最终MILKR判决会改变。

每orbit4/8各有raw/G两路径、八block MLP：2d→64 tanh→64 tanh，最后线性层零初始化。delta=.25*MLP/8，每向量L2≤.25；output=unit(baseCorrCCA64+delta)，归一化前范数至少.75。Source训练FP32，权重导出FP32只读数组；source冻结校准与target均用同一NumPy逐记录FP64算术，加载不训练、无跨query状态。源CCA仍557440FP64，新增NN684032FP32，总1241472数值参数/7195648未压缩字节；checkpoint只留模型参数，不携源IQ、例样本、cov或类库。

训练复用source Episodes的物理ID去重分组，每step取四TX和两个不同RX，sourceSUP K按1/5/20循环、另一RX每TX8条，分别36/52/112物理记录；相同索引提供raw/G两处理视图。SupCon正对仅同TX不同RX，同TX同RX从分母排除，异TX作负对，temperature=.1。总loss=SupCon+mean||e_raw−e_G||²+.1mean||e−base||²。每orbit固定600步AdamW lr.001/wd.0001，无早停/最佳步挑选，每orbit40000配对记录曝光。五fold验证模型加最终模型，共6×2×600=7200 NN optimizer步；最终部署模型为每orbit600步、两orbit合1200步，不把验证训练步数或原F训练步数混称为部署训练预算。

源L6300、ep91/92共12600配对。五RX/两TX留出同时排除两个CCA及两个NN的held RX/held TX，fit只接当前合法source L的y/RX/物理ID；component seed=source_proxy_seed+heldRX*10000+303+orbit*100003，最终无heldRX项。源F已见六TX，proxy old4/new2、源3景与目标old6/new20/6景、已曝光开发集边界保持。held-new head列softmax前裁除，NN不接head payload。源120K1规则只按Q0/QF宏H选择、ref6类数缩放；源只合法/数值门槛后固定验证，不重开旧失败gate。

target只加载同一冻结源encoder，每K/N/S仅本格SUP的R31成员gate/双RF核/残差KRR物理OOF，所有带宽/gate/KRR仅trainSUP。没有目标NN更新、query统计/角色/配额/真值。support严格纯S0raw/S1G，query raw/G-only/同记录融合完整；R38RF成员gate、R39类方向prior和R36密度均不引入。F每路径8次、G原记录一次，新增encoder推理开销实际记录。

首RX1-1/P01/MIX01/seed2026092701全部16KN，K1/5/10/20×N2/5/10/20，B24+C96=120预测、40fit/40fusion全部固定后独立truth-last。两S每KN raw较匹配原raw≥5pp、QF>raw及raw旧新均值guard≥原版−1pp才同冻结法扩其余六RX；最终七RX等权all16目标不变。G-only同输入配对和融合跨输入差分开。

sourceGPU4/targetGPU0实际容量核实并保护其他owner。真实F/G无query smoke为launcher首步PASS继续；唯一独立P0/P1、一次归档传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。实际commit、命令、环境CWD、来源、数据句柄、矩阵和技术停止条件见experiment.json。

## 发布前核验

33项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增4项+worker alias共5项PASS：两epoch特征和y/RX/IDs同train索引过滤、两orbit CCA/rho/NN同步heldRX/TX；Episodes物理ID去重/4类跨两RX、SupCon正负与排除mask正确；零初/残差L2≤.25；FP32导出/重载及源冻结评估/target共用NumPy路径；参数/schema/head列映射正确；旧R30收益gate仅匹配旧method，新route合法数值后固定验证；worker实际加载新NN双桥且无target fit。真实F/G无query smoke在launcher首步PASS继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、120个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `af6d04fa181820d8b33ca44d11baf2b1efc29fae`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'single'], 'ridge': 0.001, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1', 'orbit_ensemble_weight': 0.5}；源流程350.929秒。

target worker [227.1561293718405]秒，raw每行score/distance临时缓存[91408448]字节，最终缓存0行；40个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；R31多维ridge gate仅当前SUP拟合，组数/ridge.1核实；两路核独立SUP带宽后各半权，第二路RF实际存储每物理SUP8576bytes，没有R38/R39 RFgate额外状态，全部物理OOF train-only。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 59.4842 | +10.7074 | +3.2566 | -1.4258 |
| S0Q1 | 58.0585 | +15.8152 | +3.2566 | -1.4258 |
| S0Q2 | 62.7408 | 不适用：融合新路径 | +3.2566 | -1.4258 |
| S1Q0 | 53.4846 | +40.3934 | +13.3248 | +15.8826 |
| S1Q1 | 69.3672 | +7.2986 | +13.3248 | +15.8826 |
| S1Q2 | 66.8094 | 不适用：融合新路径 | +13.3248 | +15.8826 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐KN验收

- S0：raw较原raw至少+5为14/16，QF>raw为16/16，全部角点通过False。
- S1：raw较原raw至少+5为16/16，QF>raw为16/16，全部角点通过True。

严格晋级 False；是否扩其余六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 50.643 (+19.523) | +3.158 | +2.546 | 47.507 (+10.915) | +10.110 | +13.529 |
| 1 | 5 | 39.934 (+13.995) | +1.604 | +1.949 | 37.692 (+22.054) | +8.385 | +10.855 |
| 1 | 10 | 38.025 (+7.434) | +1.377 | -0.524 | 34.900 (+10.895) | +6.955 | +10.061 |
| 1 | 20 | 31.113 (+5.138) | +1.677 | +0.479 | 28.988 (+13.082) | +6.615 | +10.066 |
| 5 | 2 | 61.255 (+13.527) | +3.523 | +0.347 | 62.463 (+47.199) | +12.344 | +14.708 |
| 5 | 5 | 60.808 (+17.917) | +2.654 | -6.094 | 52.252 (+48.718) | +14.570 | +18.587 |
| 5 | 10 | 59.328 (+10.892) | +3.538 | -1.337 | 48.053 (+33.457) | +15.693 | +20.283 |
| 5 | 20 | 56.727 (+13.251) | +3.780 | -1.170 | 49.456 (+35.771) | +13.093 | +15.871 |
| 10 | 2 | 62.847 (+4.365) | +3.729 | -2.029 | 59.373 (+57.936) | +18.935 | +21.800 |
| 10 | 5 | 68.674 (+13.982) | +3.778 | -3.634 | 55.996 (+55.387) | +16.413 | +19.269 |
| 10 | 10 | 68.419 (+10.513) | +4.237 | -1.180 | 61.768 (+48.511) | +14.623 | +16.409 |
| 10 | 20 | 64.947 (+9.759) | +4.095 | -1.380 | 60.203 (+42.018) | +13.960 | +15.649 |
| 20 | 2 | 75.128 (+9.568) | +2.726 | -1.718 | 61.584 (+57.592) | +17.189 | +18.768 |
| 20 | 5 | 70.309 (+4.836) | +3.955 | -6.617 | 66.958 (+66.142) | +13.683 | +14.844 |
| 20 | 10 | 72.863 (+7.713) | +4.259 | -1.280 | 66.864 (+50.516) | +14.270 | +15.134 |
| 20 | 20 | 70.726 (+8.906) | +4.015 | -1.172 | 61.696 (+46.102) | +16.359 | +18.288 |

实际失败组合：[{'S': 'S0', 'K': 10, 'N': 2, 'raw_gain_pp': 4.365037257552196, 'fusion_gain_pp': 3.729269009027014}, {'S': 'S0', 'K': 20, 'N': 5, 'raw_gain_pp': 4.836417069276388, 'fusion_gain_pp': 3.955103107146263}]。角点晋级 False；不扩其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
