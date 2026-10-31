# MILKR 完整相位分布核角点验证

R21 固定完整 Tri-SIG-TIME SSL 和 DURA v1.0。R20 四角点失败产物保留，不扩展、不选择其其他源配置重跑。新方法检验均值/标准差压缩是否丢失轨道分布形状，C4 和 rank64 保持预定预算。

同一 received 记录做四个确定性相位旋转；恢复路径先对原包执行一次 G，raw 路径不执行 G。每相位提取真实 F z/t/f/pa 和三路112 signature，FFT只在原包或实际G输出算一次。source raw/G同phase配对训练标准 PairedCCA64，共6300物理记录、两epoch12600配对记录、50400计算型phasepairs；无TX监督metric。source五RX/TX-held折排除两TX全部RX/epoch/phase，仅source标签选择120个K1核规则，最终全合法L拟合一次。只有普通冻结encoder权重进入target，无source样本/特征库/类条件统计。F已见源六TX，不把encoder-held TX说成完整基座未知类。

每个物理record保持四个编码特征，两个record的核为全部16个phasepair RBF/linear核值均值；不先平均特征再RBF。保留轨道在核空间的集中程度，不做self-diagonal归一化或跨query统计。每条记录仍为一个support核行、一个标签，K不变。带宽只用当前old support的平均phase距离；K1源规则和ref6类数缩放，K>1当前物理support条件LOO选择120核/正则项，无R20 Fisher/outer选择。明确带宽/几何仍在完整support上定义，非无偏nested CV。

源相对收益不设门槛，合法性/数值和真实F/G无query smoke必须通过。first RX1-1/P01/MIX01/seed2026092701 K1/20×N2/20、四S/Q，共B8+C16=24行，全query补集。S0/S1严格分开、共享对应support状态的Q0/Q1，所有注册类竞争。全部24预测固定后独立truth-last。每角点两S Q0≥原版+5个百分点且Q1>Q0、旧新均值≥原版−1点才同模型补12KN；最终16KN全部通过才其余六RX。角点均值不与历史16格均值混比。

源GPU4/targetGPU0，启动前核实容量、最多每卡两个本owner训练/预测，不动其他owner。完整commit/命令/环境/CWD、checkpoint/data契约、路径/停止条件见experiment.json；技术失败保留产物，不盲重试。

唯一独立 P0/P1 审查 PASS。新核 PSD/相位顺序不变/逐query独立/K1物理标签数、phase encoder配对保存重载、K2正则方程一致及旧orbit、K1/源隔离、非法输入共15项相关检查通过。真实checkpoint无query smoke为发布第一步，PASS后继续。

## 实际完成证据

**VERIFIED**：真实 F/G no-query smoke PASS；C4 不变性、独立逐包、G先于orbit核实。archive传输/compile、PID/CWD/argv/GPU与日志增长核实。24行预测固定后独立truth-last评分完成，所属进程均退出。代码commit `bc2dc0976cf8922d376c18e272ed76e7bf726b11`。

源K1唯一规则：{'config': ['signature_historical', 'all', 0.0, 'multi'], 'ridge': 10.0, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6}。source L6300物理记录/12600raw-G配对/50400计算型phasepairs，rank64固定、120个source核规则，无classmetric/F/G/head更新；K>1物理support条件LOO，无Fisher/nested。源流程71.47秒；target worker（support提取/拟合+query预测）{'1-1': 166.76214851764962}秒。新增encoder139360个FP64参数（约1.06MiB），每条四次F前向、G路径仍一次G；没有将计算开销忽略为免费。

| 路径 | 当前角点均值H | 原版匹配均值H | 配对差 | Q1−Q0 |
|---|---:|---:|---:|---:|
| S0Q0 | 51.9768% | 46.1190% | +5.8579pp | -9.0958pp |
| S1Q0 | 44.4843% | 18.0209% | +26.4634pp | +19.3658pp |
| S0Q1 | 42.8810% | 42.3927% | +0.4883pp | -9.0958pp |
| S1Q1 | 63.8501% | 58.3040% | +5.5462pp | +19.3658pp |

## 角点逐格判断

- S0：Q0≥原版+5pp为2/4，Q1>Q0为1/4，全部角点通过False。
- S1：Q0≥原版+5pp为4/4，Q1>Q0为4/4，全部角点通过True。

角点晋级：False；四路对原版逐格+5全部通过：False。未通过，不扩展其余12KN或其他六RX。

| K | N | S0Q0 H%（差pp） | S0 Q1−Q0 pp | S1Q0 H%（差pp） | S1 Q1−Q0 pp |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 43.950 (+12.830) | -4.625 | 42.180 (+5.588) | +13.264 |
| 1 | 20 | 27.231 (+1.256) | +0.207 | 26.730 (+10.825) | +12.609 |
| 20 | 2 | 69.673 (+4.113) | -20.004 | 55.602 (+51.610) | +26.229 |
| 20 | 20 | 67.052 (+5.232) | -11.961 | 53.425 (+37.831) | +25.362 |

仅首RX、单seed、P01/MIX01四角点，开发基准已曝光，不代表七RX/全部16格或新盲测。原版同case分数复用；raw仍为practical received。goal ACTIVE。
