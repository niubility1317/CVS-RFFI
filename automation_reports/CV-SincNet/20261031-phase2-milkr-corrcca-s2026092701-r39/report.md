# MILKR 源CCA典型相关强度加权

R40独立检验源表示中的方向可靠性，沿用固定R34双C4/C8核、R31当前SUP成员先验及R26融合，不混入R38成员核gate或R39旧类方向校准。原CCA白化后的64个方向等权；新源编码器固定对第i个坐标乘sqrt(rho_i)，rho_i为源raw/G配对CCA的典型相关系数。相应线性几何对方向按rho_i加权，rank64、投影维度和参数量不变，没有新增混合强度或目标参数搜索。

每block输入仍为原单位化2d均值/标准差。源Wx/Wy保持原.01trace/d正则inverseRoot，SVD WxSxyWy=U diag(rho)Vᵀ，Px=WxU64diag(sqrt(rho64))、Py=WyV64diag(sqrt(rho64))，再按原规则逐record单位化、pad回d。rho理论0..1，数值舍入clip到此区间，超过1+1e−8或非有限值明确报错，无旧权重回退。FFT/f_emb的真实零std半块仍由原CCA处理，不新增余量。权重完全由合法源配对拟合；target不访问源样本/cov/classstatbank，只加载557440个FP64投影参数。新dual checkpoint schema明确为source_C4_C8_correlation_weighted_CCA_v1，不将普通CCA静默认作新版。

源L6300、ep91/92共12600配对，五源RX/两TX留出同时保护两个encoder及rho权重；held-new head列先裁除再softmax。F已见源六TX，源3景/new2与目标6景/new20及曝光开发集边界保留。源120K1规则由Q0/QF宏H选择，ref6类数缩放；K>1当前SUP物理OOF。源仅合法/数值门槛，固定新机制直接验证，不用target选rank/power或回改旧run。

Tri-SIG-TIME SSL、DURA v1.0、原C4 head及buffers冻结，每路径8次F/G原记录一次。R31 gate输入原p、prior.5/R26融合保持。每物理OOF带宽/gate/KRR仅trainSUP；源encoder不由target拟合。support严格纯S0raw/S1G，query三路逐条独立，无真值/角色/配额/批统计。G-only同输入对照与融合跨输入差分开。

首RX1-1/P01/MIX01/seed2026092701全部K1/5/10/20×N2/5/10/20，B24+C96共120预测、40fit/40fusion，全部固定后独立truth-last。两S每KN raw较匹配原raw≥5pp、QF>raw及raw旧新均值guard≥原版−1pp才同冻结方法扩其余六RX；最终七RX等权all16目标不变，单RX不代表完成。

sourceGPU4/targetGPU0启动前核实容量并保护其他owner。真实F/G无query smoke为launcher首步PASS继续；唯一P0/P1审查、一次归档传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。实际commit、命令、环境CWD、来源、数据句柄、矩阵及技术停止条件见experiment.json。

## 发布前核验

32项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增3项+worker alias共4项PASS：Px/Py列乘sqrt(rho)与公式一致；真实FFT/f_emb零std结构可fit/reload，557440参数；新weighteddual schema与普通旧schema隔离，普通默认保持；sourcefold同时保护C4/C8及rho，target加载新桥；原head裁剪/映射及R31gate保持；rho摘要读取finalfit而非load后对象；无RFgate/RFclass/density/center串入；16KN/120row沿通用worker计数。真实F/G无query smoke为launcher首步PASS继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、120个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `bfa5e9c61e20aab9b9bcda911cc892106818a6b2`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'single'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1', 'orbit_ensemble_weight': 0.5}；源流程73.910秒。

target worker [220.06967119686306]秒，raw每行score/distance临时缓存[91408448]字节，最终缓存0行；40个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；R31多维ridge gate仅当前SUP拟合，组数/ridge.1核实；两路核独立SUP带宽后各半权，第二路RF实际存储每物理SUP8576bytes，没有R38/R39 RFgate额外状态，全部物理OOF train-only。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 59.1319 | +10.3550 | +3.1892 | -1.4769 |
| S0Q1 | 57.6550 | +15.4118 | +3.1892 | -1.4769 |
| S0Q2 | 62.3210 | 不适用：融合新路径 | +3.1892 | -1.4769 |
| S1Q0 | 53.8519 | +40.7607 | +13.3116 | +15.6949 |
| S1Q1 | 69.5468 | +7.4782 | +13.3116 | +15.6949 |
| S1Q2 | 67.1635 | 不适用：融合新路径 | +13.3116 | +15.6949 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐KN验收

- S0：raw较原raw至少+5为13/16，QF>raw为16/16，全部角点通过False。
- S1：raw较原raw至少+5为16/16，QF>raw为16/16，全部角点通过True。

严格晋级 False；是否扩其余六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 50.855 (+19.735) | +3.274 | +2.504 | 47.480 (+10.888) | +10.052 | +13.581 |
| 1 | 5 | 38.543 (+12.603) | +1.487 | +1.264 | 37.198 (+21.560) | +8.733 | +11.104 |
| 1 | 10 | 36.958 (+6.366) | +1.272 | -0.424 | 35.023 (+11.017) | +7.285 | +9.851 |
| 1 | 20 | 30.774 (+4.799) | +1.615 | +0.415 | 28.815 (+12.909) | +6.950 | +9.967 |
| 5 | 2 | 61.530 (+13.802) | +3.672 | +0.699 | 63.662 (+48.398) | +12.156 | +14.060 |
| 5 | 5 | 59.614 (+16.722) | +2.324 | -6.066 | 52.224 (+48.690) | +14.917 | +18.734 |
| 5 | 10 | 58.994 (+10.558) | +3.491 | -1.121 | 49.551 (+34.955) | +15.534 | +19.516 |
| 5 | 20 | 56.393 (+12.916) | +3.677 | -1.000 | 49.797 (+36.111) | +13.027 | +15.714 |
| 10 | 2 | 63.468 (+4.986) | +3.813 | +0.275 | 59.813 (+58.376) | +18.893 | +21.443 |
| 10 | 5 | 68.694 (+14.002) | +3.855 | -4.233 | 56.196 (+55.586) | +16.058 | +18.965 |
| 10 | 10 | 68.340 (+10.434) | +4.190 | -1.356 | 63.086 (+49.829) | +14.278 | +15.739 |
| 10 | 20 | 64.997 (+9.809) | +3.990 | -1.156 | 59.283 (+41.098) | +14.141 | +16.002 |
| 20 | 2 | 75.455 (+9.894) | +3.009 | -0.827 | 61.957 (+57.966) | +17.407 | +19.845 |
| 20 | 5 | 70.176 (+4.704) | +3.660 | -6.367 | 66.865 (+66.049) | +13.858 | +15.042 |
| 20 | 10 | 70.573 (+5.422) | +3.716 | -5.319 | 66.843 (+50.494) | +14.360 | +15.233 |
| 20 | 20 | 70.747 (+8.927) | +3.982 | -0.917 | 63.839 (+48.245) | +15.335 | +16.322 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 4.7993016847533205, 'fusion_gain_pp': 1.6151794290816979}, {'S': 'S0', 'K': 10, 'N': 2, 'raw_gain_pp': 4.986189782680784, 'fusion_gain_pp': 3.8128501672164528}, {'S': 'S0', 'K': 20, 'N': 5, 'raw_gain_pp': 4.703615703764131, 'fusion_gain_pp': 3.6600810159622243}]。角点晋级 False；不扩其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
