# MILKR 冻结分类头概率核角点验证

R25 检验当前压缩特征核忽略原分类头信息是否影响旧类识别，完整Tri-SIG-TIME SSL、DURA v1.0、原六类神经分类头全部冻结。新几何固定为0.75原signature几何+0.25原分类头概率余弦，不拟合或重训任何神经分类头。不同于早期semantic-residual的固定分数先验，本版概率只作为PSD核几何的一部分，与当前C4moments/CCA64连续特征联合由合法support拟合MILKR。

每记录四次既有F前向的六类logits opt-in保留，T=1逐相位softmax、四相位概率平均再单位化，保持C4不变；旧extract默认仍只返回原phase0logits。源fold模拟old4/new2时，必须先选old_tx四列再softmax/平均，heldnew两列完全排除，不能用六类softmax后mask来获取新类置信度。F骨干本身仍见过源六TX，明确仅新encoder TX/RX-held，不称完整F未知类。target使用实际old6的六列，概率余弦对列置换不变，不依赖真实query角色。

source合法L6300物理记录、既有ep91/92共12600配对，五sourceRX/TX-held CCA拟合均排除heldRX及两TX全部RX/两epoch。TeacherBridge拟合CCA前剔除headpayload，transform后加冻结概率；最终checkpoint仍只有普通OrbitCCA64 bias/projection，无source样本、特征库、类条件统计或head概率上传。原F权重已包含冻结head，没有新NN训练、F/G/head更新0。

K1源域120核/正则规则按四路宏H、更大ridge/固定序选择，ref6 lambda按当前注册类数/6缩放。K>1真实三物理折OOF选120规则，各折tau与系数仅由训练SUP估计，最终当前全SUP重拟合；概率是逐记录冻结特征，无训练态统计。采用全注册SUP全局尺度，不叠加R24局部basis或Fisher网格。源三景/old4new2与目标六景/old6new20差异保留。

源只设合法、数值和真实F/G checkpoint noquery smoke门槛，不设相对收益门槛；固定新机制首次角点证伪，旧负run不改写、不重开、不按目标成绩挑旧源候选。firstRX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B8+C16=24行。S0/S1严格分开、对应SUPstate共享Q0/Q1；Q1仍仅G输出，G在原IQ上只执行一次再C4，raw不调用G。所有注册类别竞争，query逐条只读。24预测全部固定后独立truth-last，原版同case复用。每corner两S Q0≥原版+5点且Q1>Q0、旧新均值guard≥原版−1点才补剩余12KN；最终all16才其余六RX，不以角点平均代替最终目标。

sourceGPU4、targetGPU0启动前核实容量，不动其他owner。完整commit/命令/环境/CWD、来源/data句柄、路径、停止条件见experiment.json；最小一次P0/P1审查、真实checkpointsmoke、archive传输SHA/compile和PID/GPU/log增长核实保持。技术失败保留，不盲重试。

唯一独立P0/P1审查PASS；head先crop再softmax/排除列零影响/C4均值不变、实际extract opt-in phasepayload、CCA与headpayload隔离/普通参数存档、PSD/query独立/K1缩放、真实物理OOF及既有orbit/source隔离/非法输入共17项相关检查通过。真实F/G/head checkpoint无query smoke为launcher第一步。

## 实际完成证据

**VERIFIED**：真实 F/G no-query smoke PASS；C4 不变性、独立逐包、G先于orbit核实。archive传输/compile、PID/CWD/argv/GPU与日志增长核实。24行预测固定后独立truth-last评分完成，所属进程均退出。代码commit `c47ff1ffedf0e4319d135f1bb11d934c18f68882`。

源K1唯一规则：{'config': ['signature_identity', 'nearest', 0.0, 'single'], 'ridge': 0.01, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'teacher_weight': 0.25}。source L6300物理记录/12600raw-G配对，C4moments/rank64固定，冻结head概率权重.25，源old4先crop，K>1真实物理OOF、120个source核规则，无classmetric/F/G/head更新。源流程49.85秒；target worker（support提取/拟合+query预测）{'1-1': 81.83299064822495}秒。新增encoder278720个FP64参数（约2.13MiB），每条四次F前向、G路径仍一次G；没有将计算开销忽略为免费。

| 路径 | 当前角点均值H | 原版匹配均值H | 配对差 | Q1−Q0 |
|---|---:|---:|---:|---:|
| S0Q0 | 54.5087% | 46.1190% | +8.3897pp | -1.2586pp |
| S1Q0 | 47.6905% | 18.0209% | +29.6695pp | +16.1940pp |
| S0Q1 | 53.2501% | 42.3927% | +10.8575pp | -1.2586pp |
| S1Q1 | 63.8845% | 58.3040% | +5.5805pp | +16.1940pp |

## 角点逐格判断

- S0：Q0≥原版+5pp为3/4，Q1>Q0为1/4，全部角点通过False。
- S1：Q0≥原版+5pp为4/4，Q1>Q0为4/4，全部角点通过True。

角点晋级：False；四路对原版逐格+5全部通过：False。未通过，不扩展其余12KN或其他六RX。

| K | N | S0Q0 H%（差pp） | S0 Q1−Q0 pp | S1Q0 H%（差pp） | S1 Q1−Q0 pp |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 48.181 (+17.061) | +0.239 | 44.514 (+7.922) | +14.777 |
| 1 | 20 | 28.776 (+2.801) | -0.367 | 27.433 (+11.527) | +9.509 |
| 20 | 2 | 73.101 (+7.540) | -2.994 | 60.746 (+56.755) | +19.459 |
| 20 | 20 | 67.978 (+6.158) | -1.912 | 58.068 (+42.474) | +21.032 |

仅首RX、单seed、P01/MIX01四角点，开发基准已曝光，不代表七RX/全部16格或新盲测。原版同case分数复用；raw仍为practical received。goal ACTIVE。
