# MILKR 八相位RF轨道与原四相位先验

R33 从固定R31独立分叉，不带R32的support中心。假设是原C4只覆盖四个相位，RF矩特征对中间相位仍可能敏感。固定RF相位为0°、45°、90°、135°、180°、225°、270°、315°，不扫描数量或角度。每个block先逐视图单位化，再取八视图均值/标准差；输入维度仍2d，source普通CCA64保持。偶数位置直接使用原C4的符号/交换实现，原分类头先验只收集这四个偶数位置logits，故原C4概率函数保持；新增四视图head输出不用。原Tri-SIG-TIME SSL、DURA v1.0和神经头权重均冻结。

另一选择是时间位置轨道或support中心，已有独立负结果；本轮只检验固定相位密度变化，不能把增加视图先验当作保证收益。G只在原received上处理一次，再做八相位，FFT只在原输出计算一次后供恒相位视图共享。没有新增观测、信道重抽、query批统计或time shift。RF分量具有C8轨道不变性检查；整个分类器保留C4先验，因此不声称整个分类器严格C8不变。RF每路径从4次F增加到8次F，原head四次已包含在八次中，无另加四次F；计算增量和实际时间如实报告。

source L6300、ep91/92共12600raw/G配对，CCA64五RX/TX-held排除held RX/两TX。源old4/new2先移除held新类head列再softmax，F已见源六TX的限制保持。120核/正则选K1 Q0/QF H、ref6 classscale，K20物理OOF。源3景/new2与目标6景/new20、已曝光开发集边界保持。沿R31固定新机制直接角点证伪政策，源只用于合法/数值检查和核规则选择，无sourceH收益否决；不重开旧run，不按本run目标结果调相位。

target纯S0raw/S1G，R31多维平衡ridge gate(.1)、priorcap.5、残差KRR及R26融合公式不变。所有OOF每折gate/带宽/核系数仅trainSUP，当前全SUP重拟合；query逐条冻结计算，全部注册类竞争。无R30的学习度量，也无R32的RF中心。只普通源CCA参数进入target，无源样本/统计库。Q0raw/QG仅DURA/QF同记录融合完整保留，G-only匹配控制和QF对原G的输入改变分开。

首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行全部预测固定后独立truth-last。每角点两S raw较匹配原raw≥5pp、QF>raw、raw旧新均值guard≥原版−1pp才同冻结方法补剩余12KN；all16后补六RX。最终七RX等权逐KN目标不变，角点不替代完整验收。

sourceGPU4/targetGPU0实际容量核实，其他owner保持。commit/命令/环境CWD/来源与data句柄/输出/技术停止条件见experiment.json；真实F/G无query smoke为launcher首步PASS继续，检查RF45°轨道不变性、原C4 head一致性和单包独立性。唯一独立P0/P1审查、单包传输/remotecompile及实际PID/CWD/argv/GPU/log增长核实保持。

## 发布前核验

21项相关检查与4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS：RF八相位，head仅索引0/2/4/6对应原C4，维度顺序正确；G在旋转前一次，source/worker均C8；默认C4/time保持，不带R32中心/R30gains。实际F/G noquery smoke在launcher首步运行，包含RF C8/原C4head/单包检查，PASS直接继续。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `033ce2ea7f9dc1b8e867369da861a57c364a71a9`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'all', 0.5, 'single'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_balanced_ridge_v1'}；源流程63.523秒。

target worker [137.14631445333362]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；每个多维平衡ridge gate仅从本组SUP拟合，old_rows/new_rows、statebytes、ridge_fraction=.1核实；无SUP中心，RF8/head4元数据和实际smoke一致。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 55.0244 | +8.9055 | +2.8388 | -1.1562 |
| S0Q1 | 53.8682 | +11.4755 | +2.8388 | -1.1562 |
| S0Q2 | 57.8632 | 不适用：融合新路径 | +2.8388 | -1.1562 |
| S1Q0 | 49.7440 | +31.7230 | +12.7843 | +15.2963 |
| S1Q1 | 65.0403 | +6.7363 | +12.7843 | +15.2963 |
| S1Q2 | 62.5282 | 不适用：融合新路径 | +12.7843 | +15.2963 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为2/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为4/4，QF>raw为4/4，全部角点通过True。

严格晋级 False；是否补其余12KN/六RX由严格逐格晋级决定，不以均值代替。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 50.926 (+19.806) | +3.668 | +3.749 | 47.994 (+11.402) | +10.937 | +13.607 |
| 1 | 20 | 29.469 (+3.494) | +1.460 | -1.372 | 27.001 (+11.095) | +6.895 | +10.534 |
| 20 | 2 | 70.427 (+4.866) | +2.488 | -5.422 | 64.138 (+60.147) | +15.704 | +17.264 |
| 20 | 20 | 69.275 (+7.455) | +3.740 | -1.580 | 59.842 (+44.248) | +17.601 | +19.781 |

实际失败组合：[{'S': 'S0', 'K': 1, 'N': 20, 'raw_gain_pp': 3.4939674498645767, 'fusion_gain_pp': 1.4602964475262492}, {'S': 'S0', 'K': 20, 'N': 2, 'raw_gain_pp': 4.866037333333717, 'fusion_gain_pp': 2.48762460114772}]。角点晋级 False；不扩剩余12KN/其他六RX，保留本负结果。 源partial-novelty、单RX/单seed/P01MIX01已曝光开发边界保持，最终all16目标仍ACTIVE。
