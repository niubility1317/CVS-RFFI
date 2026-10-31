# MILKR 全注册support带宽角点验证

R23 固定带宽范围为当前全部已注册物理support。此前实现只用old_count旧类support估计tau，即使C已注册20新类也不让新类参与尺度估计；本版检验该假设。旧新宏H目标、注册类别、支持标签和物理K不变。nearest/all两种源候选公式只改变参与估计的support范围；query只读评分，不更新带宽。

保留完整Tri-SIG-TIME SSL、DURA、原神经头冻结；C4同记录moments、source pairedCCA64固定，重新只用合法sourceL/ep91/92拟合，没有R22非线性网络。source五RX/TXheld折排除heldRX及两TX全部RX/两epoch，120个K1核规则按四路源宏H选择（更大ridge/固定序平分），K>1原nested physicalsupport-fold规则；所有折仅训练support估尺度，最终用全部合法当前support重拟合。源F仍见过源六类，不称完整基座未知类验证；源old4/new2/三景与目标old6/new20/六景差异保留。

旧方法默认oldscope不改变。同一固定核配置下B公式完全相同，因为B所有注册类都是旧类；本版重新源校准K1规则，实际B结果可能变化，照实报告，不假定预测相同。source/target冻结规则均含bandwidth_scope=registered，fit记录scope和实际参与物理行数。

这是预先固定的结构假设证伪，source只作合法性、数值、真实checkpoint无query smoke门槛和源核规则选择，不设相对收益门槛；不修改R22失败门槛或重新测试其NN。最终source全L6300物理记录、12600配对重新拟合encoder一次，只冻结bias/projection参数，不带source样本、特征库、类条件统计。F/G/head更新0，query真值/角色/配额/跨样本统计0。

首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20四角点，严格S0/S1×Q0/Q1，共B8+C16=24行。Q1仍仅DURA输出，待询问的双路融合口径不影响本run。恢复路径G仅在原包上先执行一次再C4，raw路径不执行G。对应supportstate共享两query路径。全部24预测固定后独立truth-last，原版同case历史分数复用。每corner两S Q0≥原版+5个百分点且Q1>Q0、raw旧新均值guard≥原版−1点才同冻结模型补12KN；最终16KN全部过后才其余六RX，角点不代替最终目标。

sourceGPU4/targetGPU0，启动前核实实际容量，不动其他owner。最小源码/配置commit、独立P0/P1审查、实际checkpointsmoke、一次archive传输/remotecompile、PID/CWD/argv/GPU/log增长核实与独立评分保持；不新增数据重验证/哈希链。完整命令、环境/CWD、来源、路径、停止条件见experiment.json。

唯一独立P0/P1审查PASS；注册带宽公式、新support影响/旧默认保持、同配置B等价、非法scope拒绝、nested仅本折训练support估尺度、query独立及既有KRR方程/K1缩放/source隔离/非法输入共14项相关检查通过。真实F/G checkpoint noquery smoke为launcher第一步。

## 实际完成证据

**VERIFIED**：真实 F/G no-query smoke PASS；C4 不变性、独立逐包、G先于orbit核实。archive传输/compile、PID/CWD/argv/GPU与日志增长核实。24行预测固定后独立truth-last评分完成，所属进程均退出。代码commit `bae4e91c9a0089156c9691608a03cac0a16e871d`。

源K1唯一规则：{'config': ['signature_historical', 'all', 0.0, 'single'], 'ridge': 0.1, 'outer_folds': 3, 'outer_strengths': [0.0, 0.25, 0.5, 0.75], 'ridge_reference_classes': 6, 'bandwidth_scope': 'registered'}。source L6300物理记录/12600raw-G配对，C4moments/rank64固定，bandwidth_scope=registered、120个source核规则，无classmetric/F/G/head更新。源流程45.06秒；target worker（support提取/拟合+query预测）{'1-1': 95.34192574815825}秒。新增encoder278720个FP64参数（约2.13MiB），每条四次F前向、G路径仍一次G；没有将计算开销忽略为免费。

| 路径 | 当前角点均值H | 原版匹配均值H | 配对差 | Q1−Q0 |
|---|---:|---:|---:|---:|
| S0Q0 | 54.4109% | 46.1190% | +8.2919pp | -0.1827pp |
| S1Q0 | 48.7540% | 18.0209% | +30.7330pp | +15.6229pp |
| S0Q1 | 54.2282% | 42.3927% | +11.8355pp | -0.1827pp |
| S1Q1 | 64.3769% | 58.3040% | +6.0729pp | +15.6229pp |

## 角点逐格判断

- S0：Q0≥原版+5pp为3/4，Q1>Q0为1/4，全部角点通过False。
- S1：Q0≥原版+5pp为4/4，Q1>Q0为4/4，全部角点通过True。

角点晋级：False；四路对原版逐格+5全部通过：False。未通过，不扩展其余12KN或其他六RX。

| K | N | S0Q0 H%（差pp） | S0 Q1−Q0 pp | S1Q0 H%（差pp） | S1 Q1−Q0 pp |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 48.627 (+17.507) | -0.413 | 44.655 (+8.063) | +13.101 |
| 1 | 20 | 29.125 (+3.150) | -1.094 | 28.400 (+12.494) | +9.949 |
| 20 | 2 | 71.860 (+6.299) | +1.312 | 63.646 (+59.655) | +18.737 |
| 20 | 20 | 68.031 (+6.211) | -0.536 | 58.314 (+42.720) | +20.705 |

仅首RX、单seed、P01/MIX01四角点，开发基准已曝光，不代表七RX/全部16格或新盲测。原版同case分数复用；raw仍为practical received。goal ACTIVE。
