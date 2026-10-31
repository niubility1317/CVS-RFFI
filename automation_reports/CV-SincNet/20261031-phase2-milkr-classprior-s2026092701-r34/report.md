# MILKR 当前support类别校准先验

R35 保留R34的C4/C8等权核、源双CCA64、8次F/原4相位head/G原记录一次，Tri-SIG-TIME SSL和DURA及神经head权重不变。新假设是只校准old/new成员可信度，仍可能保留跨RX的旧类混淆，因此在MILKR先验内增加当前SUP的闭式类别校准。它不改神经head输出，但会改变MILKR先验和最终判决；不能称整个分类器不变。

q与R31相同，包含sqrt(m)(p−1/m)、熵置信度和top2间隔。SUP旧/新组分别总权重.5（B只有旧类时总1），联合目标为[old成员指示，类概率目标]。旧SUP类目标为其旧类one-hot，新SUP类目标为旧m列上的均匀向量。以当前SUP加权mean/cov、lambda=.1trace(cov)/d进行一次多输出岭回归，截距不惩罚。成员分量在两组齐全时与R31二组ridge相同，B全开；类分量逐record做欧氏单纯形投影得到pcal。prior旧槽=.5·g(p)·(pcal−1/m)，新槽0，故仍有固定幅度界和每行prior和0。零方差明确constant_unseparated/old_only_constant模式，非法数值报错，不静默回退。

这是当前SUP的闭式先验状态，不新增神经身份头，不更新F/G/buffers，不写回源checkpoint。每物理OOF只用trainSUP拟合mean/cov/classmatrix/tau/KRR，held/query只冻结应用；不读取query真实old/new身份/类别数/配额或批统计，不跨K/N/S借数据。target m6时q维8、输出7，mean8+coef56+targetmean7共71个FP64/568bytes；公开审计只计数/维度/范数/mode，不提交适配系数。

RF仍两种源CCA64新拟合，合557440参数，source L6300、ep91/92共12600配对；五RX/TX-held两编码器均排除held RX/两TX。源old4/new2先移除held新类head列再softmax，类校准只对源旧4列，新增2类目标均匀。F已见源六TX、source3景/new2与target6景/new20、已曝光开发基准限制保留。source120规则选K1 Q0/QF宏H/ref6scale，K20物理OOF，源合法/数值后直接角点证伪，无源H收益否决；不改R34已完成规则或输出。

support纯S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整。RF两带宽后各半权、单残差KRR与R26融合条件保持；RF距离先同一SUP两分辨率平均再最近，head仍原C4。首RX1-1/P01/MIX01/seed2026092701 K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。每角点两S raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才同冻结方法补12KN；all16后补六RX。最终七RX等权逐KN要求不变，G-only同输入控制与QF跨输入差分开。

sourceGPU4/targetGPU0实际容量核实，不动其他owner。commit/命令/环境CWD/来源/data句柄/输出/技术停止条件见experiment.json；真实F/G无query smoke在launcher首步，检查原head/双RF/单包独立，PASS继续。唯一P0/P1审查、单包传输/remotecompile、实际PID/CWD/argv/GPU/log增长和独立truth评分保持。

## 发布前核验

24项相关检查与4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS，独立执行新增4项测试通过：旧类onehot/new uniform列与当前旧类概率顺序一致；prior只写旧槽、逐record simplex后零和且幅度受限；OOF gate/mean/cov/tau/KRR仅trainSUP；source/worker class rule及双CCA/fitter路由正确。神经head不更新，MILKR prior/最终决策有意改变。实际F/G无query smoke在launcher首步，PASS继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `a2867c3427c0e4476a2b31caeabc1172b43ea0ca`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'single'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_class_ridge_v1', 'orbit_ensemble_weight': 0.5}；源流程106.374秒。

target worker [150.62550117215142]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；当前SUP类别校准ridge状态568bytes/输出7/组数/ridge.1核实，原神经head不更新但MILKR prior改变；两路核独立SUP带宽后各半权，第二路RF实际存储每物理SUP8576bytes，全部物理OOF train-only。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 55.6625 | +9.5435 | +2.8282 | -0.9518 |
| S0Q1 | 54.7106 | +12.3179 | +2.8282 | -0.9518 |
| S0Q2 | 58.4906 | 不适用：融合新路径 | +2.8282 | -0.9518 |
| S1Q0 | 50.2185 | +32.1975 | +12.8349 | +15.3794 |
| S1Q1 | 65.5979 | +7.2939 | +12.8349 | +15.3794 |
| S1Q2 | 63.0534 | 不适用：融合新路径 | +12.8349 | +15.3794 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为3/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为4/4，QF>raw为4/4，全部角点通过True。

严格晋级 False；是否补其余12KN/六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 50.679 (+19.559) | +2.935 | +1.168 | 47.904 (+11.312) | +10.517 | +13.928 |
| 1 | 20 | 30.560 (+4.585) | +1.389 | -1.074 | 28.130 (+12.225) | +7.174 | +10.454 |
| 20 | 2 | 71.615 (+6.054) | +3.028 | -3.059 | 62.325 (+58.333) | +17.449 | +19.495 |
| 20 | 20 | 69.795 (+7.975) | +3.960 | -0.842 | 62.515 (+46.920) | +16.200 | +17.641 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 4.585342794301766, 'fusion_gain_pp': 1.3888308036236918}]。角点晋级 False；不扩剩余12KN/其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
