# MILKR 当前support RF旧类先验方向校准

R39保留R38双C4/C8核、R31head成员gate、R38RF成员gate与R26融合，检验RF对“哪一个旧类”的证据能否改善旧类先验方向。A=.5RBF4+.5RBF8，使用当前trainSUP各自tau和所选single/multi尺度，只在likelihood A去掉线性成分以保持非负。a_c(x)=mean_{S_j属于旧c} A(x,S_j)，pcal=normalize(original_head_p*a)。旧类RF似然均匀时还原原p，无新增混合权重。head成员gate始终输入原p，不将校准p喂回旧gate。prior旧槽=.5*g_head(originalp)*(pcal−1/m)*R38_RFfactor，新槽0，因此prior零和及幅度界保持。

每物理OOF的old类均值/labels、两tau、两原gate和KRR只使用trainSUP，held/query逐条到固定trainSUP算affinity，无query统计/真值/角色/配额；不同K/N/S独立。源m4、目标m6依合法标签表对齐原head列；新SUP不进入旧类likelihood均值，但保持R38成员gate/KRR训练。RFclass状态为int64物理SUP labels及m个counts，额外8*(physicalSUP+m)bytes。不是神经头训练，Tri-SIG-TIME SSL、DURA、原head及buffers冻结，最终MILKR判决改变。非正likelihood normalizer明确报错，无静默替代。

source L6300/ep91、92共12600配对，fresh双CCA64合557440FP64；五源RX/两TX留出同时保护两个编码器，held-new head列在softmax前裁去。F已见源六TX、源3景/new2与目标6景/new20及已曝光开发集边界保留。源120K1规则只按源Q0/QF宏H选择，ref6类数缩放，K>1当前SUP物理OOF；源合法/数值后固定验证，不用target选择参数。

本新候选首RX1-1/P01/MIX01/seed2026092701覆盖全部K1/5/10/20×N2/5/10/20，B24+C96共120行、40fit/40fusion。它仍是单capsule小矩阵，补充中间K/N证据；旧角点未通过run不重开、不改gate。全部预测固定后独立truth-last。两S每格raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才同冻结方法扩其余六RX；首RX已覆盖all16，无剩余12KN阶段。最终七RX等权每KN要求保持，首RX全过也不代表最终达成。

support严格纯S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整保留。G-only同输入配对与QF跨输入差分开。sourceGPU4/targetGPU0实际容量检查并保护其他owner。真实F/G无query smoke在launcher首步PASS继续；唯一P0/P1审查、单归档传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。完整commit、命令、环境CWD、数据句柄、来源、逐行矩阵和技术停止条件见experiment.json。

## 发布前核验

36项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增4项+worker alias共5项PASS：source old4/target old6的旧类列序正确；RF likelihood去linear且context尺度一致；新pcal仅替换先验方向，原headgate仍接original p，R38 factor有界，prior新槽0/零和/幅度保持；每OOF仅TRAIN全部likelihood/tau/两gate/KRR，held/query逐条同cross规则；source grid两marker和实际双CCA/fitter同路。单capsule16KN对应40fit/40fusion/120row动态计数闭合，无density/center串入。真实F/G无query smoke为launcher首步PASS继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、120个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `5819bc89b519c9abb4580c0d353eed1bba7bbb5e`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'multi'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1', 'orbit_ensemble_weight': 0.5, 'rf_prior_gate_rule': 'support_balanced_kernel_evidence_v1', 'rf_class_prior_rule': 'support_rf_old_class_product_v1'}；源流程77.177秒。

target worker [311.7873685453087]秒，raw每行score/distance临时缓存[91408448]字节，最终缓存0行；40个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；R31多维ridge gate仅当前SUP拟合，组数/ridge.1核实；两路核独立SUP带宽后各半权，第二路RF实际存储每物理SUP8576bytes，RFgate额外状态按实际SUP计数核实，全部物理OOF train-only。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 57.4202 | +8.6434 | +2.9327 | -2.5242 |
| S0Q1 | 54.8961 | +12.6529 | +2.9327 | -2.5242 |
| S0Q2 | 60.3530 | 不适用：融合新路径 | +2.9327 | -2.5242 |
| S1Q0 | 51.7813 | +38.6901 | +14.5367 | +17.2205 |
| S1Q1 | 69.0018 | +6.9332 | +14.5367 | +17.2205 |
| S1Q2 | 66.3180 | 不适用：融合新路径 | +14.5367 | +17.2205 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐KN验收

- S0：raw较原raw至少+5为11/16，QF>raw为16/16，全部角点通过False。
- S1：raw较原raw至少+5为16/16，QF>raw为16/16，全部角点通过True。

严格晋级 False；是否扩其余六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 50.575 (+19.455) | +3.355 | +1.962 | 46.408 (+9.815) | +10.737 | +15.177 |
| 1 | 5 | 36.687 (+10.748) | +1.159 | -0.671 | 36.131 (+20.494) | +9.505 | +12.508 |
| 1 | 10 | 36.594 (+6.002) | +0.952 | -1.241 | 34.595 (+10.590) | +7.778 | +10.651 |
| 1 | 20 | 30.926 (+4.951) | +1.387 | -0.566 | 28.355 (+12.449) | +7.320 | +10.394 |
| 5 | 2 | 59.608 (+11.881) | +3.123 | +1.745 | 55.297 (+40.033) | +14.827 | +18.663 |
| 5 | 5 | 57.077 (+14.186) | +2.608 | -6.571 | 51.183 (+47.649) | +16.056 | +20.156 |
| 5 | 10 | 59.195 (+10.759) | +3.447 | -1.630 | 48.717 (+34.121) | +16.600 | +20.288 |
| 5 | 20 | 55.439 (+11.962) | +3.500 | -0.792 | 44.100 (+30.414) | +15.955 | +19.558 |
| 10 | 2 | 63.025 (+4.543) | +3.739 | +0.747 | 59.484 (+58.047) | +19.031 | +21.585 |
| 10 | 5 | 62.035 (+7.342) | +1.451 | -9.797 | 56.743 (+56.134) | +16.761 | +18.771 |
| 10 | 10 | 67.277 (+9.371) | +4.250 | -0.906 | 56.566 (+43.308) | +17.249 | +19.172 |
| 10 | 20 | 62.349 (+7.161) | +3.726 | -4.140 | 57.472 (+39.287) | +15.000 | +17.216 |
| 20 | 2 | 70.328 (+4.767) | +2.966 | -3.695 | 60.846 (+56.855) | +18.673 | +20.832 |
| 20 | 5 | 69.319 (+3.846) | +3.730 | -6.153 | 64.775 (+63.959) | +15.553 | +16.683 |
| 20 | 10 | 69.025 (+3.875) | +3.588 | -7.376 | 65.402 (+49.053) | +15.286 | +16.378 |
| 20 | 20 | 69.265 (+7.445) | +3.944 | -1.303 | 62.426 (+46.832) | +16.256 | +17.495 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 4.951270096769211, 'fusion_gain_pp': 1.3870375975507665}, {'S': 'S0', 'K': 10, 'N': 2, 'raw_gain_pp': 4.542752733491517, 'fusion_gain_pp': 3.738796792809951}, {'S': 'S0', 'K': 20, 'N': 2, 'raw_gain_pp': 4.767061039023912, 'fusion_gain_pp': 2.9658397354685118}, {'S': 'S0', 'K': 20, 'N': 5, 'raw_gain_pp': 3.846181281562122, 'fusion_gain_pp': 3.7295036166338558}, {'S': 'S0', 'K': 20, 'N': 10, 'raw_gain_pp': 3.8747501861795874, 'fusion_gain_pp': 3.587759402317392}]。角点晋级 False；不扩其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
