# MILKR 当前support选择源残差强度

R42复用R41唯一固定600步/每orbit的纯源NN和源K1规则，不重新训练网络，也不继承目标SUP适应状态。新假设是多样本SUP能评估源残差的可靠性，避免所有case固定完整残差。K1没有类内物理留出，保留原fullNN源先验；K>1在固定强度0/.25/.5/.75/1上，联合24核×5ridge作当前SUP物理OOF选择，共600候选。优先old/new宏H，其次较小平方误差、较大ridge、精确tie时较弱残差。相同物理fold用于全部强度，每fold带宽/gate/KRR仅trainSUP；选择后只保留本格一个模型，查询不作选择或五模型集成。

base和full分别是同一处理输入的原CorrCCA和有界NN特征，计算base+s*(full−base)后沿原RF各branch单位化。端点0/1直接返回原特征避免改变端点数值；head概率不混合。S0的两个表示都来自raw，S1都来自DURA输出，故不是把raw混入DURA support，也不增加K/物理记录/标签。全部强度同样遵守逐记录计算、跨K/N/S独立，query无批统计、角色、配额、标签或状态更新。600候选OOF用于选择，不能当独立泛化证据。

源NN checkpoint只含源CCA/NN参数，固定在R41目标预测前保存，原target SUP KRR/prediction/truth/optimizer从不加载。NN加载前核对其源数据/F/G角色与源训练零target/query/Utruth读入；同一次launcher首步无query smoke检查真实Tri-SIG-TIME SSL、DURA及已加载NN，对2条合法source L的raw/G输出检查base/full和五强度逐记录独立性。PASS直接继续，模型对象在进程内传给源冻结保存，不引入smoke票据或审批。

新源拟合记录为空、新NN更新0、model_fitting_rows0；父训练6次模型/7200 NN更新仅作继承历史，部署权重原600步/每orbit。源K1 120grid及规则可继承，因为K1 full端点不变；父K20验证只标继承，不冒充新选择器已验证。CCA557440FP64+NN684032FP32、合1241472数值参数/7195648未压缩字节不变。额外成本为多样本SUP的600候选拟合与同路径双表示缓存，实际时间报告；查询只用本格已选模型。

Tri-SIG-TIME SSL、DURA、原C4head及buffers冻结，R31headgate/prior和R26 query融合保持，不引入R38/R39 RFgate/prior。源L6300/12600配对祖先，F曾见源六TX，代理old4/new2及源3景对目标6景/已曝光开发基准边界保留。源encoder不由target拟合，SUP只用于本格KRR/表示强度选择。

首RX1-1/P01/MIX01/seed2026092701，全部K1/5/10/20×N2/5/10/20，B24+C96共120预测、40fit/40fusion全部固定后独立truth-last。两S每KN raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才冻结同法扩其余六RX；最终七RX等权all16目标不变。G-only同输入配对与融合跨输入差分开。

sourceGPU4只smoke/复用、targetGPU0，启动前检查余量并保护其他owner。最小流程的一次独立P0/P1、针对性检查、一次传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。实际commit、命令、环境CWD、来源、数据句柄、矩阵与技术停止条件见experiment.json。

## 发布前核验

36项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增6项+worker alias共7项PASS：NN加载前parent_metadata核对源数据/F/G角色、固定600步源审计及纯源checkpoint；smoke返回同一模型上下文给freeze，无新训练或父target状态；25字段为同处理路径full/base各12RF和同一original headp，S1无raw混入；K1 full精确端点；K>1五强度物理TRAIN-only OOF600候选排序正确，查询同blend；旧K20源验证明确仅继承；worker新fit/桥及120行计数闭合。一次合并真实F/G/NN无query smoke为launcher首步PASS继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、120个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `5dbe2c46d83e0b083152f474de037405844bb923`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'single'], 'ridge': 0.001, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1', 'orbit_ensemble_weight': 0.5, 'support_residual_strengths': [0.0, 0.25, 0.5, 0.75, 1.0], 'K1_residual_strength': 1.0}；源流程0.258秒。

target worker [312.30534499976784]秒，raw每行score/distance临时缓存[91408448]字节，最终缓存0行；40个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；R31多维ridge gate仅当前SUP拟合，组数/ridge.1核实；两路核独立SUP带宽后各半权，第二路RF实际存储每物理SUP8576bytes，没有R38/R39 RFgate额外状态，全部物理OOF train-only。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 59.5259 | +10.7490 | +3.2552 | -1.3680 |
| S0Q1 | 58.1579 | +15.9147 | +3.2552 | -1.3680 |
| S0Q2 | 62.7811 | 不适用：融合新路径 | +3.2552 | -1.3680 |
| S1Q0 | 53.7677 | +40.6765 | +13.2358 | +15.7468 |
| S1Q1 | 69.5145 | +7.4459 | +13.2358 | +15.7468 |
| S1Q2 | 67.0035 | 不适用：融合新路径 | +13.2358 | +15.7468 |

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
| 5 | 5 | 61.341 (+18.450) | +2.766 | -5.811 | 52.512 (+48.978) | +14.708 | +18.321 |
| 5 | 10 | 59.328 (+10.892) | +3.538 | -1.337 | 48.053 (+33.457) | +15.693 | +20.283 |
| 5 | 20 | 56.407 (+12.931) | +3.726 | -1.039 | 49.608 (+35.922) | +12.912 | +15.760 |
| 10 | 2 | 62.847 (+4.365) | +3.729 | -2.029 | 59.373 (+57.936) | +18.935 | +21.800 |
| 10 | 5 | 69.088 (+14.396) | +4.032 | -3.209 | 56.195 (+55.586) | +16.144 | +18.924 |
| 10 | 10 | 68.404 (+10.498) | +4.201 | -1.203 | 63.086 (+49.829) | +14.278 | +15.739 |
| 10 | 20 | 65.040 (+9.852) | +4.039 | -1.157 | 60.203 (+42.018) | +13.960 | +15.649 |
| 20 | 2 | 75.362 (+9.801) | +2.810 | -1.772 | 61.957 (+57.966) | +17.407 | +19.845 |
| 20 | 5 | 70.216 (+4.744) | +3.686 | -6.592 | 66.865 (+66.049) | +13.858 | +15.042 |
| 20 | 10 | 72.863 (+7.713) | +4.259 | -1.280 | 66.906 (+50.557) | +14.293 | +15.177 |
| 20 | 20 | 70.548 (+8.728) | +3.959 | -1.257 | 63.976 (+48.382) | +15.175 | +16.189 |

实际失败组合：[{'S': 'S0', 'K': 10, 'N': 2, 'raw_gain_pp': 4.365037257552196, 'fusion_gain_pp': 3.729269009027014}, {'S': 'S0', 'K': 20, 'N': 5, 'raw_gain_pp': 4.743612515503626, 'fusion_gain_pp': 3.6856378337208784}]。角点晋级 False；不扩其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
