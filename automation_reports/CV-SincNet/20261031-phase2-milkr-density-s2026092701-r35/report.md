# MILKR 当前 support 核密度归一化

R36 在 R34 的等权 C4/C8 核上检验 support hub 影响。固定 Tri-SIG-TIME SSL、DURA v1.0、原 C4 head、R31 成员 ridge 先验和 R26 同记录融合；不采用 R35 的类别校准，不回调旧 run 参数。

在当前训练 support S 上，A(x,S)=两路独立带宽后的 RBF 各半权，沿用所选 single/multi scales，只在密度 A 中去除线性分量以保证非负。d(x)=mean_j A(x,S_j)，support 自项包含在内；dbar=mean_i d(S_i)。新核为 dbar*K_parent(x,y)/sqrt(d(x)*d(y))。它是正对角合同变换，保留正半定；常数密度时严格还原父核。没有新增可调强度、截断底值或静默回退；非法或非正数值报错。核密度只降低 support 中相似度普遍偏高的记录权重，不用 query 真值、old/new 角色、配额或批次统计。

每个物理 OOF 的 tau、degree、dbar、成员 gate 和 KRR 只由 train support 拟合。held/query 逐记录对固定 train support 计算自身 d(x)，不写回状态；support 路径始终纯 S0 raw 或纯 S1 DURA，不增加物理 K。密度额外状态为 (physicalSUP+1)*8 bytes；F/G/head 参数和 buffers 冻结，最终 MILKR 判决有意改变。RF 最近距离及融合门控保持原定义。

双 source CCA64 重新拟合，源 L6300/ep91、92共12600配对，557440 FP64参数；五源RX/两TX留出同时保护两编码器，源held-new head列先移除再softmax。F已见源六TX，源3景/new2与目标6景/new20、已曝光开发基准边界如实保留。源120规则只选K1 Q0/QF宏H，ref6类别数缩放，K>1物理OOF。源合法/数值后直接固定角点证伪，无源H收益否决。

首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行，Q0raw/QG仅DURA/QF同记录融合完整；全部预测固定后独立truth-last。两S每角点raw较匹配原raw≥5pp、QF>raw及raw旧新均值guard≥原版−1pp才冻结同法补12KN，再六RX。最终K1/5/10/20×N2/5/10/20的七RX等权逐格目标不变；G-only同输入配对和融合跨输入差分开。

source GPU4、target GPU0，启动前核实容量并保护其他owner。实际commit、命令、环境CWD、合法数据句柄、路径、矩阵和技术停止条件见experiment.json。真实F/G无query smoke为launcher第一步PASS继续；唯一P0/P1审查、单归档传输校验、remotecompile、PID/CWD/argv/GPU/log增长及独立评分保持。

## 发布前核验

29项相关检查及4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增4项加density worker alias共5项通过：非负A/mix0、正对角合同PSD、训练和held/query degree公式一致；每物理OOF只trainSUP的tau/degree/dbar/gate/KRR；源grid携带density rule，source/worker实际双CCA/fitter同路；R31 prior/R26 fusion/strictSUP保持。真实F/G无query smoke在launcher首步，PASS继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `c9623d56c9dc262b16631992f9e7c12d5fafdf34`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'nearest', 0.5, 'single'], 'ridge': 0.001, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1', 'orbit_ensemble_weight': 0.5, 'kernel_density_rule': 'support_rbf_degree_v1'}；源流程108.556秒。

target worker [146.33028256194666]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；R31多维ridge gate仅当前SUP拟合，组数/ridge.1核实；两路核独立SUP带宽后各半权，第二路RF实际存储每物理SUP8576bytes，核密度状态另计(physicalSUP+1)*8bytes，全部物理OOF train-only。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 55.5923 | +9.4734 | +2.8146 | -0.5526 |
| S0Q1 | 55.0397 | +12.6470 | +2.8146 | -0.5526 |
| S0Q2 | 58.4069 | 不适用：融合新路径 | +2.8146 | -0.5526 |
| S1Q0 | 48.7588 | +30.7379 | +13.5447 | +16.4026 |
| S1Q1 | 65.1614 | +6.8574 | +13.5447 | +16.4026 |
| S1Q2 | 62.3035 | 不适用：融合新路径 | +13.5447 | +16.4026 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为3/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为4/4，QF>raw为4/4，全部角点通过True。

严格晋级 False；是否补其余12KN/六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 50.944 (+19.824) | +3.018 | +2.890 | 45.743 (+9.150) | +11.557 | +15.354 |
| 1 | 20 | 30.700 (+4.725) | +1.367 | -0.396 | 28.141 (+12.235) | +7.011 | +10.386 |
| 20 | 2 | 71.421 (+5.860) | +2.959 | -3.347 | 60.479 (+56.488) | +18.672 | +20.997 |
| 20 | 20 | 69.305 (+7.485) | +3.914 | -1.358 | 60.672 (+45.078) | +16.939 | +18.874 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 4.725180215955088, 'fusion_gain_pp': 1.3670789974981945}]。角点晋级 False；不扩剩余12KN/其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
