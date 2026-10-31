# MILKR 当前support多维平衡先验校准

R31 固定Tri-SIG-TIME SSL、DURA v1.0、原默认神经头、C4moments/sourceCCA64、先验幅度上限.5和R26融合规则。新假设是仅用熵置信度的old/new均值会丢失概率形状，也会在均值差很小时形成陡峭gate。因此只在当前SUP上拟合多维平衡岭回归可信度，不改R27已完成参数，不使用R30的512权重。

q(p)=[sqrt(m)(p−1/m), c(p), margin(p)]，m是合法已知头列数，c为1−H(p)/logm，margin为最大两概率差。SUP已知/新增角色来自已授权标签，t=1/0；旧组、新组分别总权重.5。加权mean q，centered cov和rhs，ridge=.1·trace(cov)/d，beta=(cov+ridge I)^−1rhs，g(p)=clip(.5+(q−mean)beta,0,1)。这是带无惩罚截距的平衡二组岭回归软权重，不宣称概率校准。B无新类g=1；C covtrace≤float64eps时明确constant_unseparated/g=.5并审计。非法输入和数值求解失败报错。当前SUP旧类prior=.5·g(p)·(p−1/m)，新增槽0；KRR拟合Y−prior保持。只增加至多8维mean/beta，不增加观测、标签或神经身份头。

K>1真实三物理折OOF，每折gate均值/协方差/系数、RF带宽和KRR都只用trainSUP，held逐条应用冻结参数；最终当前全SUP重拟合。query不提供真实old/new角色，不参与统计/拟合/选择，也不跨K/N/S借SUP。每条query只用冻结映射概率。公开fit审计仅计数、mode、ridge、系数范数和statebytes，不提交适配权重。

source L6300、ep91/92共12600配对，CCA64五RX/TX-held只选120个K1核/正则规则（Q0/QF宏H、ref6 classscale），K20保持物理OOF；held新类head两列在softmax前剔除。F仍见源6TX、源三景/目标六景、源new2/目标new20的限制保留。本轮是新固定SUP机制，沿R26/R27的直接四角点证伪政策，源阶段只作合法/数值检查及K1选择，无源H收益否决门槛；不放宽R30门槛或复用其被拒权重，不因源小代理替代目标验证。

support严格S0raw/S1G；Q0raw/QG仅DURA/QF同记录融合完整保留。原received经G一次再C4，融合仍只在G到SUP最近RF距离更小且全类margin更大时半权，否则raw。原F/G/head不更新。首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。每角点两S raw较匹配原raw≥5pp、QF>raw、raw旧新均值guard≥原版−1pp才冻结方法补12KN；全部16格后再补另6RX。最终七RX等权逐KN目标不变，QG匹配控制和QF跨路径差明确区分。

sourceGPU4/targetGPU0启动前核实容量，其他owner不动。commit/命令/环境CWD/来源/data句柄/输出/技术停止规则见experiment.json；真实F/G noquery smoke为launcher首步PASS继续，唯一独立P0/P1审查、单包传输/remotecompile及实际PID/CWD/argv/GPU/log增长核实保持。

## 发布前核验

21项相关检查与4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS：加权中心化ridge与0.5截距一致，B全开/零方差0.5均有明确模式；OOF gate/cov/核/系数仅trainSUP，query冻结应用；普通CCA路由，不加载R30 gains或进入其源收益门槛。实际F/G noquery smoke将在launcher首步运行，PASS直接继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `c0aa35cc871432ad857981b52fa7dbcd1093b186`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'single'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1'}；源流程42.955秒。

target worker [84.6971016060561]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；每个多维平衡ridge gate仅从本组SUP拟合，old_rows/new_rows、statebytes、ridge_fraction=.1核实。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 54.6420 | +8.5231 | +3.1900 | -0.6078 |
| S0Q1 | 54.0342 | +11.6415 | +3.1900 | -0.6078 |
| S0Q2 | 57.8320 | 不适用：融合新路径 | +3.1900 | -0.6078 |
| S1Q0 | 46.9921 | +28.9712 | +14.1279 | +17.2994 |
| S1Q1 | 64.2916 | +5.9876 | +14.1279 | +17.2994 |
| S1Q2 | 61.1201 | 不适用：融合新路径 | +14.1279 | +17.2994 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为3/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为4/4，QF>raw为4/4，全部角点通过True。

严格晋级 False；是否补其余12KN/六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 48.559 (+17.439) | +3.144 | +0.379 | 43.501 (+6.909) | +11.103 | +16.221 |
| 1 | 20 | 30.071 (+4.096) | +1.547 | -0.356 | 27.096 (+11.190) | +7.797 | +10.870 |
| 20 | 2 | 72.481 (+6.920) | +4.117 | -0.149 | 58.190 (+54.199) | +19.261 | +22.037 |
| 20 | 20 | 67.457 (+5.637) | +3.952 | -2.304 | 59.181 (+43.587) | +18.350 | +20.069 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 4.095765023241099, 'fusion_gain_pp': 1.5472148207863023}]。角点晋级 False；不扩剩余12KN/其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
