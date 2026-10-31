# MILKR support置信度校准先验与query融合

R27 固定完整Tri-SIG-TIME SSL、DURA和原神经分类头冻结，C4moments/sourceCCA64、先验最大幅度0.5及R26已验证的query融合规则均保持。新假设是固定先验容易偏向旧类，应只从当前support校准其可信度；不修改R26已完成产物或按query成绩调其参数。

置信度c(p)=1−H(p)/log(oldcount)，逐record计算。当前SUP中旧类置信度均值a、新类均值b：B无新类时g=1；C只有a−b超过float64机器epsilon才g=clip((c−b)/(a−b),0,1)，否则g=0，并记录no_positive_gap模式、a/b及实际组数。新prior旧类槽为0.5·g(p)·(p−1/oldcount)，新类槽为0，当前SUP仍拟合Y−prior的KRR残差和无惩罚截距。只增加当前SUP的两个标量统计，不增加神经头、物理K、标签或其他K/N/S信息。query只逐条应用冻结a/b；没有query真实old/new角色、类别数、配额、批统计或参数更新。

K>1的真实三物理折OOF中，gate均值、RF带宽和系数均只由训练SUP拟合，held使用训练gate；选120核/正则规则后当前全SUP重拟合。K1仍由源五RX/TX-held、old4/new2的120规则按Q0/QF四主路径H、较大ridge/固定序选择，ref6 lambda随当前注册类数/6缩放。源old4head先去掉heldnew两列再softmax，骨干仍见过源六TX的限制保留。sourceL6300、既有ep91/92共12600配对，sourceCCA不读headpayload，最终只普通CCA参数进入target；不带source样本/特征库/类条件统计，NN/F/G/head更新0。源三景与目标六景、old4new2与old6new20差异不称新盲测。

用户已允许同记录query raw/G融合并保留G-only，support仍pureS0/S1。Q0=raw、QG(机器Q1)=仅G、QF(机器Q2)=融合。R26规则原样保持：只有G到SUP的最近RF距离更小且全类top2分数间隔更大才两路半权，否则raw；规则不读query标签。G对原received只执行一次再C4，QF使用两路既有四次F前向，未重抽信道或增加观测。原始分数/距离cache只按同case同物理query对齐融合，不改变任何SUP状态，全部三路预测后清空。

source合法/数值/真实checkpoint noquery smoke后，首RX1-1/P01/MIX01/seed2026092701同四角点K1/20×N2/20，B12+C24=36行全部固定后独立truth-last。每corner两S Q0较匹配原Q0≥5个百分点且QF>Q0、raw旧新均值guard≥原版−1点才补剩余12KN，最终all16后才其他六RX。QG对原QG同输入配对完整保留，QF对原QG标明路径改变，不用融合代替raw+5要求。

sourceGPU4/targetGPU0启动前核实容量，不动其他owner。完整commit、命令、环境/CWD、来源/data句柄、路径、停止条件见experiment.json；最小独立P0/P1审查、实际checkpointsmoke、一次archiveSHA/compile及PID/GPU/log增长核实保持。技术失败保留，不盲重试。

唯一独立P0/P1审查PASS；support熵置信度/旧新均值/B全开/非正差关闭、校准prior残差方程与截距、query逐条独立/均值不变、物理折gate只读训练SUP、旧三路worker/head crop/非法输入等19项相关检查通过。真实F/G/head checkpoint无query smoke为launcher第一步，PASS后继续源校准及36行先导。

## 实际完成与结果

**VERIFIED**：release archive传输/remotecompile通过，发布后实际source与target进程PID/CWD/argv/GPU均已读取，源/target日志增长、36个prediction与SCORED_COMPLETE及全部所属进程退出核实。真实F/G/head无query smoke PASS。代码commit `32a49c989a6e5d5c53a5e966a2de26a6775734c6`。source固定Q0/QF选择K1规则：{'config': ['signature_identity', 'nearest', 0.5, 'single'], 'ridge': 0.001, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'prior_weight': 0.5, 'fusion_policy': 'rf_distance_margin_half_v1', 'confidence_rule': 'support_entropy_means_v1'}；源流程67.927秒。

target worker [110.60879214620218]秒，raw每行score/distance临时缓存[28777056]字节，最终缓存0行；12个fit记录均严格单SUP路径，head6列合法映射、prior.5、物理OOF状态与allocatedK核实；每个置信度gate仅从本组SUP拟合，old_rows/new_rows与标签数一致。融合没有额外F/G参数更新。

| 路径 | 当前角点平均宏H% | 同输入原版差pp | QF−Q0 pp | QG−Q0 pp |
|---|---:|---:|---:|---:|
| S0Q0 | 53.6974 | +7.5785 | +2.9604 | -2.7961 |
| S0Q1 | 50.9014 | +8.5087 | +2.9604 | -2.7961 |
| S0Q2 | 56.6578 | 不适用：融合新路径 | +2.9604 | -2.7961 |
| S1Q0 | 45.5434 | +27.5224 | +14.9396 | +18.4050 |
| S1Q1 | 63.9483 | +5.6444 | +14.9396 | +18.4050 |
| S1Q2 | 60.4829 | 不适用：融合新路径 | +14.9396 | +18.4050 |

Q0/Q1分别为raw/G-only同输入对照；Q2为用户允许的同记录融合，不把Q2对原G-only的差冒充同输入配对。Q2对原raw/G的跨路径差单独保留result_summary字段。

## 逐角点验收

- S0：raw较原raw至少+5为2/4，QF>raw为4/4，全部角点通过False。
- S1：raw较原raw至少+5为4/4，QF>raw为4/4，全部角点通过True。

严格晋级 False；raw尚有失败格，不扩其余12KN或其他六RX。纯raw/G四路对原版逐格+5全部通过 False，作为控制单列，不用融合隐藏原G-only下降。

| K | N | S0 raw H%（对原差pp） | S0 QF−raw pp | S0 G−raw pp | S1 raw H%（对原差pp） | S1 QF−raw pp | S1 G−raw pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 48.672 (+17.552) | +3.217 | -1.341 | 43.227 (+6.634) | +10.797 | +16.200 |
| 1 | 20 | 29.793 (+3.818) | +1.630 | -1.480 | 26.096 (+10.190) | +8.106 | +11.315 |
| 20 | 2 | 68.427 (+2.866) | +2.531 | -4.567 | 57.447 (+53.456) | +19.792 | +22.579 |
| 20 | 20 | 67.898 (+6.078) | +4.463 | -3.796 | 55.404 (+39.809) | +21.063 | +23.525 |

raw失败为S0 K1/N20（+3.8179pp）及S0 K20/N2（+2.8660pp）。融合在全部8个角点/S组合均提升，但不能代替raw每格+5的要求。只是一RX/单seed/P01MIX01的已曝光开发证据，不代表全部16KN或七RX，不称新盲测。源F见过六TX/源三景代理限制保持，goal ACTIVE。
