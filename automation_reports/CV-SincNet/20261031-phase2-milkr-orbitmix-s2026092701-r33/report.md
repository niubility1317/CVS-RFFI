# MILKR C4与C8等权局部核组合

R34 固定C4与C8两种RF分辨率，不按目标分数选择混合权重。每条记录仍只做八次F：全八次生成C8矩，偶数四次生成原C4矩，head只用原四次。G只在原received处理一次后再旋转；两种RF表征不增加物理观测、标签或额外F/G调用。Tri-SIG-TIME SSL、DURA v1.0和默认神经头保持冻结。

备选是继续加密相位或做support中心化，已有独立负结果保留。本轮只检验两种分辨率互补，等权不保证收益。每个RF分辨率在合法源L上重新拟合一个普通CCA64，不复用target-adapted状态。两编码器共557440个FP64参数，约4.25MiB，另有当前SUP第二路RF特征存储；公开报告实际时间和额外RAM。新checkpoint双CCA schema只含source投影/均值偏置，不含样本、协方差、标签/ID或类原型库。

当前SUP中分别计算tau4与tau8，再分别构造原linear/RBF/multiscale核，最终K=.5K4+.5K8。是两个核的算术平均，不是把相似度先平均后过RBF。只解一套Y−prior的KRR及无惩罚截距，R31多维平衡ridge gate(.1)、priorcap.5保持。raw/G融合仍用R26条件规则，RF距离定义为先对同一物理SUP的两路距离等权，再取最近SUP；不能分别挑两路不同原型形成虚假接近。没有按query真实old/new角色分支或类别配额限制。

K1源五RX/TX-held、120核/正则规则按Q0/QF宏H选择/ref6classscale；sourceK20及target K>1真实物理三折OOF，两个带宽、gate、核系数均只trainSUP。最终当前全SUP重拟合，query逐条冻结应用，无query统计/拟合/选择，不跨K/N/S借SUP。默认head概率只来自C4，源held新类head列在softmax前移除；两编码器都排除held RX/两TX。F已见源六TX、source3景/new2与target6景/new20、已曝光开发基准限制如实保留。源L6300、ep91/92共12600配对包，只源合法/数值门槛，无源H收益否决。

support严格S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整。无R32中心或R30学习度量。首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。两S每角点raw较匹配原raw≥5pp、QF>raw及raw旧新均值guard≥原版−1pp才同冻结方法补12KN；all16后补六RX。最终七RX等权逐KN验收不变，G-only匹配控制与QF跨输入差分开。

sourceGPU4/targetGPU0实际容量核实，其他owner保持。commit/命令/环境CWD/来源/data句柄/输出/技术停止条件见experiment.json；真实F/G无query smoke为launcher首步，包含C4/C8组件、原四相位head和单包检查，PASS继续。唯一P0/P1审查、单包传输/remotecompile、实际PID/CWD/argv/GPU/log增长及独立truth评分保持。

## 发布前核验

18项相关检查与4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS：两路各SUP带宽后等权，单残差KRR；RF距离先同一SUP合并再min；source/worker双提取器/双CCA schema/新fitter，物理OOF两尺度和gate均只trainSUP。实际F/G无query smoke在launcher首步，包含两个RF组件/原head/单包检查，PASS直接继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `156d494190ebd7a172057becaa02adec1ed7050b`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'nearest', 0.5, 'single'], 'ridge': 0.001, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1', 'orbit_ensemble_weight': 0.5}；源流程132.627秒。

target worker [139.21275397622958]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；R31多维ridge gate仅当前SUP拟合，组数/ridge.1核实；两路核独立SUP带宽后各半权，第二路RF实际存储每物理SUP8576bytes，全部物理OOF train-only。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 55.4954 | +9.3764 | +2.8832 | -0.7234 |
| S0Q1 | 54.7720 | +12.3793 | +2.8832 | -0.7234 |
| S0Q2 | 58.3786 | 不适用：融合新路径 | +2.8832 | -0.7234 |
| S1Q0 | 48.8641 | +30.8432 | +13.4135 | +16.4118 |
| S1Q1 | 65.2759 | +6.9719 | +13.4135 | +16.4118 |
| S1Q2 | 62.2776 | 不适用：融合新路径 | +13.4135 | +16.4118 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为3/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为4/4，QF>raw为4/4，全部角点通过True。

严格晋级 False；是否补其余12KN/六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 50.510 (+19.390) | +3.323 | +2.428 | 46.073 (+9.481) | +11.021 | +15.429 |
| 1 | 20 | 30.734 (+4.759) | +1.396 | -0.517 | 28.198 (+12.292) | +7.137 | +10.323 |
| 20 | 2 | 71.439 (+5.878) | +2.925 | -3.461 | 60.570 (+56.578) | +18.537 | +20.923 |
| 20 | 20 | 69.298 (+7.478) | +3.889 | -1.343 | 60.616 (+45.022) | +16.958 | +18.973 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 4.7588828060873, 'fusion_gain_pp': 1.3959961256561737}]。角点晋级 False；不扩剩余12KN/其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
