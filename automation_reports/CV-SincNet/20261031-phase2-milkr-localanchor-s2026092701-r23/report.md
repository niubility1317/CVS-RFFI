# MILKR 局部support核特征角点验证

R24 固定完整Tri-SIG-TIME SSL、DURA v1.0和原神经分类头冻结，保留C4同记录moment/source pairedCCA64。此前全局tau范围改变收益小，新方法检验当前support的局部类间尺度与密度归一化，不从目标结果挑旧候选重跑。

每条当前物理support记录作为一个centre，tau_j取它到其他类support距离的nearest/median；显式RBF特征r_j(x)=exp(-d(x,s_j)/(tau_j·scale))，scale固定single1或multi[0.25,1,4]。只用SUP计算每个特征列范数，再将每个record的行单位化。query只用固定SUPcentre/tau/列norm和本record的行norm，log-row-max仅为避免指数下溢的等价数值运算，无query跨样本统计。K=(1-mix)phi(x)phi(y)^T+mix原signature几何，mix0/0.5，显式特征保证PSD。

K1源域120个核/正则规则按四路宏H、更大ridge/固定序选择，ref6 lambda按当前注册类数/6缩放。K>1以真实三物理折OOF选择120规则，centre/tau/列norm/系数都只在训练折拟合，留出物理record不参与；按宏H、较小MSE、较大ridge选择，再用当前全SUP重拟合。没有Fisher strength网格，也没有用完整SUP几何代替物理OOF。每条记录仍只占一个核行/标签，K不变。预合并localW=phi(S)^T alpha，预测等价完整双核式，避免query×support²；额外W/tau/列norm状态字节实际记录。

source合法L6300物理记录、既有ep91/92共12600配对；五RX/TX-held sourceCCA均排除heldRX及两TX全部RX/两epoch，最终encoder全source拟合一次。F本身见源六类，不把adapter-held当完整基座未知类；源old4/new2/三景与目标old6/new20/六景差异保留。target只载冻结投影参数，无source样本/特征库/类条件统计。NN训练0，F/G/head更新0。

这版固定结构先导只要求source合法/数值/真实checkpoint无query smoke，不设源相对收益门槛；旧R22/R23停止及产物完整保留。首RX1-1/P01/MIX01/seed2026092701，四角点K1/20×N2/20，严格S0/S1×Q0/Q1，共B8+C16=24行，全query补集。Q1仍仅G输出，G在原包上只执行一次再C4，raw不调用G，同一supportstate共享Q0/Q1。24预测全部固定后独立truth-last。每corner两S Q0≥原版+5点且Q1>Q0、旧新均值guard≥原版−1点才补其余12KN；最终16KN全部过后才其他六RX，角点不代替最终目标。

sourceGPU4/targetGPU0，启动前实测容量，最多每卡两个本owner新训练/预测，不动其他owner。完整commit、命令、环境/CWD、checkpoint/data来源、路径与停止条件见experiment.json；一次最小P0/P1审查、实际checkpointsmoke、archiveSHA/remotecompile、PID/GPU/log增长核实保持。技术失败保留，不盲重试。

唯一独立P0/P1审查PASS；局部核PSD/合并预测等价、query独立、远query数值稳定、每centre尺度方向与排除同类、SUP列norm、真实物理OOF隔离、K1物理数/缩放/route及旧orbit/source隔离/非法输入共16项相关检查通过。真实checkpoint无query smoke为launcher第一步。

## 实际完成证据

**VERIFIED**：真实 F/G no-query smoke PASS；C4 不变性、独立逐包、G先于orbit核实。archive传输/compile、PID/CWD/argv/GPU与日志增长核实。24行预测固定后独立truth-last评分完成，所属进程均退出。代码commit `ff1f605efb56a956bebe46e8853efa6312d7c4ed`。

源K1唯一规则：{'config': ['signature_historical', 'nearest', 0.5, 'single'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'local_anchor_policy': 'column_norm_row_unit_v1'}。source L6300物理记录/12600raw-G配对，C4moments/rank64固定，局部centre尺度/列norm/行unit，K>1真实物理OOF、120个source核规则，无classmetric/F/G/head更新。源流程48.40秒；target worker（support提取/拟合+query预测）{'1-1': 84.25764767685905}秒。新增encoder278720个FP64参数（约2.13MiB），每条四次F前向、G路径仍一次G；没有将计算开销忽略为免费。

| 路径 | 当前角点均值H | 原版匹配均值H | 配对差 | Q1−Q0 |
|---|---:|---:|---:|---:|
| S0Q0 | 54.4089% | 46.1190% | +8.2900pp | -0.8128pp |
| S1Q0 | 48.1502% | 18.0209% | +30.1292pp | +15.7128pp |
| S0Q1 | 53.5961% | 42.3927% | +11.2035pp | -0.8128pp |
| S1Q1 | 63.8630% | 58.3040% | +5.5590pp | +15.7128pp |

## 角点逐格判断

- S0：Q0≥原版+5pp为3/4，Q1>Q0为0/4，全部角点通过False。
- S1：Q0≥原版+5pp为4/4，Q1>Q0为4/4，全部角点通过True。

角点晋级：False；四路对原版逐格+5全部通过：False。未通过，不扩展其余12KN或其他六RX。

| K | N | S0Q0 H%（差pp） | S0 Q1−Q0 pp | S1Q0 H%（差pp） | S1 Q1−Q0 pp |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 48.391 (+17.271) | -0.284 | 44.747 (+8.154) | +12.983 |
| 1 | 20 | 29.033 (+3.058) | -1.670 | 27.666 (+11.761) | +9.992 |
| 20 | 2 | 72.111 (+6.550) | -0.269 | 60.741 (+56.750) | +20.030 |
| 20 | 20 | 68.101 (+6.281) | -1.028 | 59.447 (+43.852) | +19.847 |

仅首RX、单seed、P01/MIX01四角点，开发基准已曝光，不代表七RX/全部16格或新盲测。原版同case分数复用；raw仍为practical received。goal ACTIVE。
