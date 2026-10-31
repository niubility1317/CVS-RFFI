# MILKR 四相位特征角点验证

R20 固定使用完整 Tri-SIG-TIME SSL、DURA v1.0。此前 R19 首 RX 未逐格通过，不扩展。新方法检验绝对相位敏感性是否影响跨 RX 识别；不根据本轮目标结果选择相位数、rank 或模型。

每个既有 received 记录仅做 0°、90°、180°、270° 的确定性复相位旋转。DURA 路径先对原记录执行一次冻结 G，再旋转输出；raw 路径不执行 G。冻结 F 的 z/t/f/pa 和真实三路112维 signature 分别每相位归一化，再按每条记录求均值/标准差。FFT 仅在 base/raw 或实际 G 输出计算一次，标准差为0。无新物理观测、信道生成、标签、query 跨样本统计。

输入宽度加倍，预定 source paired CCA rank64，分别编码 raw/G 轨道特征到原 MILKR 特征 schema；无 source 监督协方差度量。只用合法 L6300 和既有 ep91/92 模拟缓存。五 source RX/TX-held folds（两 TX 全 RX/两 epoch 排除）选择唯一 K1 kernel/ridge，120 个规则按四路源宏 H、较大 ridge、固定序打破平分；K>1 原 nested support 校准，K1 reference6 随当前 support 类数缩放。最终 encoder 全 source L 拟合一次，只上传 bias/projection 权重，不上传源样本、特征库、类均值/协方差。源 proxy 只对新 encoder 隔离 TX/RX，F 已见源六类，不称新盲测。

这是预定固定新表征的首次角点证伪，源分数只选 K1 规则并诊断，不设相对收益门槛；合法性、数值和真实 F/G 无 query smoke 必须通过。首 RX1-1、P01/MIX01、seed2026092701，K1/20×N2/20 四角点，B8+C16=24 行。S0/S1 严格分离，同一 support 状态共享 Q0/Q1，所有已注册类别竞争，不读 query 真值/角色/配额。24 行预测全部固定后独立 truth-last。

每角点两 S 的 Q0 较原版至少+5个百分点且 Q1>Q0，并保持旧/新均值 guard≥原版−1点，才补齐其余12个K/N（冻结同一模型/规则，新不可覆盖root）；16格全部通过后才六 RX 确认。最终用户目标完全不变，角点通过本身不算成功。完整命令、环境/CWD、checkpoint 来源、GPU、停止条件和输出见 experiment.json。源 GPU4/target GPU0，启动前核实容量，不动其他 owner。

唯一独立 P0/P1 审查 PASS，未发现阻断下一真实 run 的问题；相位不变性、分批/G顺序、配对encoder拟合/序列化/逐query独立、method routing、K1缩放/源隔离及4项非法协议输入共11项相关检查通过。实际checkpoint无query smoke由launcher第一步执行，PASS后直接继续。

## 实际完成证据

**VERIFIED**：真实 F/G no-query smoke PASS；C4 不变性、独立逐包、G先于orbit核实。archive传输/compile、PID/CWD/argv/GPU与日志增长核实。24行预测固定后独立truth-last评分完成，所属进程均退出。代码commit `30bdb9f73b2fc2274275e65a34bdc4f6ca9be0e5`。

源K1唯一规则：{'config': ['signature_historical', 'all', 0.0, 'single'], 'ridge': 0.1, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6}。source L6300物理记录/12600raw-G配对，rank64固定、120个source核规则，无classmetric/F/G/head更新。源流程45.30秒；target worker（support提取/拟合+query预测）{'1-1': 93.67933013103902}秒。新增encoder278720个FP64参数（约2.13MiB），每条四次F前向、G路径仍一次G；没有将计算开销忽略为免费。

| 路径 | 当前角点均值H | 原版匹配均值H | 配对差 | Q1−Q0 |
|---|---:|---:|---:|---:|
| S0Q0 | 54.3788% | 46.1190% | +8.2599pp | -0.1536pp |
| S1Q0 | 48.3730% | 18.0209% | +30.3521pp | +15.9682pp |
| S0Q1 | 54.2252% | 42.3927% | +11.8325pp | -0.1536pp |
| S1Q1 | 64.3411% | 58.3040% | +6.0372pp | +15.9682pp |

## 角点逐格判断

- S0：Q0≥原版+5pp为3/4，Q1>Q0为1/4，全部角点通过False。
- S1：Q0≥原版+5pp为4/4，Q1>Q0为4/4，全部角点通过True。

角点晋级：False；四路对原版逐格+5全部通过：False。未通过，不扩展其余12KN或其他六RX。

| K | N | S0Q0 H%（差pp） | S0 Q1−Q0 pp | S1Q0 H%（差pp） | S1 Q1−Q0 pp |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 48.632 (+17.512) | -0.418 | 44.677 (+8.085) | +13.080 |
| 1 | 20 | 29.122 (+3.147) | -1.091 | 28.377 (+12.471) | +9.974 |
| 20 | 2 | 71.671 (+6.110) | +1.441 | 63.315 (+59.324) | +18.875 |
| 20 | 20 | 68.091 (+6.271) | -0.546 | 57.123 (+41.529) | +21.944 |

仅首RX、单seed、P01/MIX01四角点，开发基准已曝光，不代表七RX/全部16格或新盲测。原版同case分数复用；raw仍为practical received。goal ACTIVE。
