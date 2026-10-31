# MILKR 源 CCA 正交余量特征保留

R37 从 R31 的 C4/R31成员prior/R26融合独立分叉。假设是 CCA 压缩可能遗漏新设备特征，而普通 native PCA 与 CCA 重复。新编码器先去除 raw/restored 两个 CCA 投影矩阵的联合列空间，再对源数据的正交余量做 PCA，固定保留 d−64 维。它不同于 R28 的普通 native PCA；旧负结果和停止门槛保持，不重开旧 run。

除FFT与频率嵌入外的六个2d维C4均值/标准差block先按原规则单位化，源CCA64得到Px/Py。对[Px,Py]作SVD，仅用机器精度eps*max(shape)*最大奇异值确定数值秩。两路源数据共用pooled mean，从中心数据去掉联合span，再取余量协方差最大d−64个正特征方向。raw/G使用相同余量mean/basis；CCA按原路径各自编码。最终concat单位CCA64和单位余量d−64，各能量1/2，输出d维不变。非正余量方差明确报错，没有截断或静默回退。编码器合404768个FP64参数（约3.088MiB），仅保留参数，不携源样本或协方差到target。

合法源L6300、ep91/92共12600配对重新拟合。五RX/两TX留出对CCA和余量PCA同时生效，源held-new head列先裁除再softmax。F曾见源六TX，源3景/new2与目标6景/new20及已曝光开发集边界保持。源120K1规则仅用源Q0/QF宏H/ref6类数缩放，K>1当前SUP物理OOF；源合法/数值后直接固定角点证伪，无sourceH收益否决。

Tri-SIG-TIME SSL、DURA v1.0、原C4 head及buffers冻结；每路径4次F/G原记录一次。R31当前SUP成员ridge .1、prior.5与R26融合不变，无dualC4C8/density/classprior/SUPcenter。每物理OOF gate/tau/KRR仅trainSUP；query逐条冻结，无批统计、真值old/new角色或配额。support严格纯S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整，G-only同输入对照与融合跨输入差分开。

首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。每角点两S raw较匹配原raw≥5pp、QF>raw、raw旧新均值guard≥原版−1pp才同冻结法补剩余12KN，再六RX。最终七RX等权的all16逐格目标不变。

sourceGPU4、targetGPU0启动前核实余量，保护其他owner。实际commit、命令、环境CWD、数据句柄、来源、矩阵及技术停止条件见experiment.json。真实F/G无query smoke为launcher首步PASS继续；唯一P0/P1审查、单包传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。

## 发布前P1修复

独立审查指出真实C4共享FFT，FFT标准差半块恒零，两份CCA64可能耗尽其96维有效空间。原实现相对近零余量最大值判秩会误收舍入误差。尚未远端发布或读取目标。本候选现在事前明确：FFT只保留原CCA输出（能量1），不存余量mean/basis；其余七块CCA/正交余量各半能量。余量正方差按原源数据协方差trace与机器精度判断，完全投影掉的子空间明确报错，不用数值噪声填维。参数修正为435808FP64；真实FFT零std及完全span舍入噪声回归通过。仅对该P1定点复审，不重复全审。

源checkpoint只读核对还确认f_emb标准差半块投影范数严格0，与FFT同属C4不变量。因此最终事前定义为FFT/f_emb两块原CCA-only，其余六块各半余量；保存参数404768FP64。测试同时将FFT后96及f_emb后160坐标置零，两路两块输出与原CCA逐位一致，保存无两块native字段。首次FFT修复定点复审PASS后，此同结构扩展由对应回归检查核实，不再增设重复审查门槛；未发布或读取目标。

## 发布前验证结论

28项相关检查与4项非法协议输入检查通过；低秩修正后新增3项定向回归通过，compile/diff通过。唯一独立P0/P1审查发现真实FFT低秩P1，已修复并经唯一一次定点复审PASS（原复现及新增3项独立通过）。频率嵌入同结构扩展按实际源checkpoint证据及同一回归覆盖，本地PASS。最终FFT/f_emb原CCA-only、其余六块各半正交余量，404768FP64，无数值噪声填余量；没有重复全审或报告审查。真实F/G无query smoke为launcher首步PASS继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `93ac7f55eb514e53fe6cd687be52579b01eaa013`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'single'], 'ridge': 0.001, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1'}；源流程58.482秒。

target worker [81.48423352418467]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；每个多维平衡ridge gate仅从本组SUP拟合，old_rows/new_rows、statebytes、ridge_fraction=.1核实。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 52.7515 | +6.6325 | +0.0689 | -11.5912 |
| S0Q1 | 41.1602 | -1.2324 | +0.0689 | -11.5912 |
| S0Q2 | 52.8203 | 不适用：融合新路径 | +0.0689 | -11.5912 |
| S1Q0 | 39.3677 | +21.3467 | +20.9933 | +22.9582 |
| S1Q1 | 62.3259 | +4.0219 | +20.9933 | +22.9582 |
| S1Q2 | 60.3610 | 不适用：融合新路径 | +20.9933 | +22.9582 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为2/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为4/4，QF>raw为4/4，全部角点通过True。

严格晋级 False；是否补其余12KN/六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 42.106 (+10.986) | +0.121 | -1.487 | 44.182 (+7.590) | +8.224 | +8.066 |
| 1 | 20 | 30.165 (+4.190) | +0.087 | -3.098 | 22.237 (+6.332) | +12.207 | +16.151 |
| 20 | 2 | 69.176 (+3.615) | +0.054 | -29.933 | 39.097 (+35.106) | +39.217 | +40.594 |
| 20 | 20 | 69.558 (+7.738) | +0.013 | -11.847 | 51.954 (+36.360) | +24.326 | +27.021 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 4.190412441485291, 'fusion_gain_pp': 0.08721451450550699}, {'S': 'S0', 'K': 20, 'N': 2, 'raw_gain_pp': 3.6149159952980847, 'fusion_gain_pp': 0.054222622747379656}]。角点晋级 False；不扩剩余12KN/其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
