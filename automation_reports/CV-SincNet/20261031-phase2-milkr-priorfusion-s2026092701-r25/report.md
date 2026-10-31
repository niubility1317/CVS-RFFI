# MILKR 冻结函数先验与同记录query融合先导

用户已明确允许query同记录raw与DURA输出融合，并保留仅DURA输出对照；support仍严格S0 raw／S1 G，不混入另一处理视图或增加物理K。科学协议已先加入本轮新run例外，旧实验/其他owner定义保持。新版Q0=raw、QG（机器Q1）=仅G、QF（机器Q2）=同记录融合。最终逐格目标仍为两S Q0较匹配原版Q0≥+5个百分点，并按用户新口径要求QF>Q0；QG完整单列，QF与原QG差注明路径改变，不冒充同输入配对。

R26 保持完整Tri-SIG-TIME SSL、DURA v1.0、原分类头冻结，C4moments/sourceCCA64固定。新raw读出直接使用冻结head的函数先验b0(x)：old列为0.5·(meanC4 softmax旧head−1/oldcount)，new列0，MILKR拟合当前SUP残差Y−b0并使用不惩罚的残差截距。不是R25的概率几何：核仅用原RF signature几何，head知识直接保留到输出函数。新神经训练/改头为0。源old4/new2先去掉heldnew两head列再softmax，F曾见源6的边界披露；target用sourcecontract.classes与合法registeredold表映射head顺序，实际六旧类顺序已核实相同。

融合规则事前固定：同一冻结SUP状态分别算raw/G全注册类分数、到SUP的最近RF距离及top2分数间隔。只有G距离严格更小且G间隔严格更大时，QF为两路分数等权均值；否则为raw。没有query真值、真实old/new角色、批统计、权重/阈值更新或类配额。权重只取0/0.5，raw始终保留。QF计算需要同记录8次F和一次G（两路各C4），Q0和QG对照也保留；G只对原received执行一次，无重抽信道/第二次观测。

source合法L6300物理记录、既有ep91/92共12600配对；五RX/TXheldCCA均排除heldRX及两TX全部RX/两epoch。源K1规则120个按Q0/QF四主路径H、更大ridge/固定序选择，QG完整记录；ref6 lambda随当前注册类数/6缩放。K>1真实三物理折OOF选120个RF核/正则配置，各折只用训练SUP估尺度/系数，再全SUP重拟合。F仍见源六TX，源old4/new2/三景不等于目标old6/new20/六景，不称新盲测。最终checkpoint只普通CCA权重，无source样本/特征库/类条件统计上传。

source合法/数值/实际checkpoint无query smoke后，首RX1-1/P01/MIX01/seed2026092701四角点K1/20×N2/20。B12+C24=36行，所有SUP状态在读取query前固定；每个state只读一种SUP路径。raw每行分数/距离临时缓存，仅与同物理ID、同case的G分数融合，不改变模型；全部36预测固定后独立truth-last。原版Q0/QG同case历史复用。每corner两S raw≥原版+5点、QF>raw、raw旧新均值guard≥原版−1点才补剩余12KN，最终all16后才六RX；均值不能替代逐格。

sourceGPU4/targetGPU0，启动前核实实际显存、最多每卡两个本owner新训练/预测，不动其他owner。完整commit/命令/环境/CWD、来源/data句柄、路径和停止条件见experiment.json。最小独立P0/P1审查、真实checkpointsmoke、一次archiveSHA/remotecompile、实际PID/GPU/log增长和独立评分保持；技术失败保留，不盲重试。

唯一独立P0/P1审查PASS；函数prior残差正规方程/截距、类别表映射、融合无标签/逐条独立、真实物理OOF、mockworker三path行对齐/严格SUP/缓存清空及既有teacher/orbit/非法输入共18项相关检查通过。真实F/G/head checkpoint无query smoke为launcher第一步，PASS后直接源校准及36行先导。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source进程PID/GPU已读取，最终源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `a656856c978334cbc2dce7027599b1f96d4ecfed`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'single'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1'}；源流程53.902秒。

target worker [85.74730308167636]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 53.8908 | +7.7718 | +2.9507 | -2.0680 |
| S0Q1 | 51.8228 | +9.4301 | +2.9507 | -2.0680 |
| S0Q2 | 56.8414 | 不适用：融合新路径 | +2.9507 | -2.0680 |
| S1Q0 | 45.3046 | +27.2837 | +14.2197 | +18.2062 |
| S1Q1 | 63.5108 | +5.2068 | +14.2197 | +18.2062 |
| S1Q2 | 59.5243 | 不适用：融合新路径 | +14.2197 | +18.2062 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为3/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为3/4，QF>raw为4/4，全部角点通过False。

严格晋级 False；raw尚有失败格，不扩其余12KN或其他六RX。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 47.259 (+16.139) | +2.595 | -2.377 | 40.945 (+4.352) | +10.762 | +17.242 |
| 1 | 20 | 29.766 (+3.791) | +1.368 | -1.499 | 26.567 (+10.661) | +6.898 | +10.400 |
| 20 | 2 | 70.629 (+5.068) | +3.804 | -0.903 | 56.472 (+52.480) | +19.905 | +23.369 |
| 20 | 20 | 67.909 (+6.089) | +4.035 | -3.493 | 57.235 (+41.641) | +19.313 | +21.813 |

raw失败为S0 K1/N20（+3.7906pp）及S1 K1/N2（+4.3524pp）。融合在全部8个角点/S组合均提升，但不能代替raw每格+5的要求。只是一RX/单seed/P01MIX01的已曝光开发证据，不代表全部16KN或七RX，不称新盲测。源F见过六TX/源三景代理限制保持，goal ACTIVE。
