# 源 RX 身份风险约束下的 AdamW 真实特征位移

本轮检验一个独立训练机制：默认头继续由原主 CE 联合学习，额外约束只修正实际部署特征参数的 AdamW 位移。最近 PA 完整交换源诊断 0/8 达标，原 E20 loss-gap 三种子对照亦未确认正收益，均按既定规则关闭。该源证据提示必须区分非零梯度、真实位移、特征变化和身份收益，不能用放大损失替代不同训练方向。

RAW 是当前 L90 的原 CE→clip5→AdamW。GRAD_CONE 先将同一 CE 的 feature 下降梯度投影到五个源 RX 风险半空间，再用原 AdamW。UPDATE_CONE 则投影 RAW 真正的 FP32 参数前后差。目标是最小欧氏修正，使 `g_RX·delta<=0`；欧氏距离仅定义在当前固定 CENTER25K 坐标，不声称单位不变。W、PA/reference/SIG 三增益不属于投影范围；原主 CE 仍保留其梯度。

投影类方法已有 PCGrad 和 CAGrad。这里的具体研究区别是约束**真实 AdamW 提议位移**，而不是假设梯度满足条件后，动量／裁剪／weight decay 的实际更新也会满足条件。不能宣称投影概念原创或已证明泛化收益。[PCGrad](https://arxiv.org/abs/2001.06782)，[CAGrad](https://arxiv.org/abs/2110.14048)。

固定矩阵为自有 BASE_CE 第一 seed 2026092701 的 INITIAL 与 FINAL20，各四个 L90（ordinal0/17/35/52），比较三控制。风险 L180 是各90cell的 ordinal66/67，独立评价 L180 是 ordinal68/69，与当前360包和彼此物理ID不交。它们均属于已训练过的合法源 L，未改变数据契约或持久角色；V、U IQ、U真值、target均不读取。每 RX 的风险／评价包为36，6TX×3day均衡。

风险 CE 使用真实四项 joint（base、PA、reference、SIG_TIME），原 W／三 gain detached。静态 feature 范围为部署 encoder 原参数，保留浅层、中层、PA、参考响应、原归一化等，排除 W／gain；不以当前非零梯度选择参数。INITIAL 明确是原配方、空矩、LR2e-4的假想一步，FINAL20 使用实际保存状态及 LR1e-6。两者是 CPU eval-mode 诊断，不冒充原 CUDA 训练同时发生的步骤，也没有中期证据。

FP64 解算只用5×5 Gram、最多32个 active set和KKT检查。UPDATE 的参数加法重新转换至实际dtype，逐 RX 记录理想风险、实际风险和 `sum(abs(g)*abs(actualDelta-idealDelta))` 舍入界；不使用全模型最大绝对值扩大容差。后续源网络评价恢复更新前 W 和全部增益，以直接比较 joint160 的有限步变化。零位移始终可行，因此必须同时核验保留位移、独立 held-L CE 和实际 feature 变化，不能把“被压成零”算收益。

预登记 source gate：八单元至少两个满足 RAW 至少一个 RX 正一阶风险、UPDATE 相对 RAW 实际位移差比例 >.01、实际位移保留 ≥.5、独立 held-L 最差 RX 的 CE 增量减少 >1e-6、mean CE 相对 RAW 增加 ≤1e-6、held feature RMS 差 >1e-7、至少一个浅／中层原参数实际不同，以及实际风险在 solver＋逐RX舍入界内。同时至少两个单元 UPDATE 相对 GRAD 实际位移差比例 >.01。门槛只判学习空间和有限步方向，不是 DG 性能、V选模或原创性证明。失败时关闭该固定变体，不扫描阈值、权重或 step scale。

实际预算：720 unique L／6300 metadata／10444 packet 主网络前向／16临时CPU AdamW／0真实训练、头拟合、EMA、U/V/target。任何来源／数据角色／状态不匹配、非有限、无合法 solver解或技术错误均保存产物并停止本 run，不影响其他 owner；科学失败不重试。

唯一 run `20261015-phase1-SIGTIME-RX-update-projection-sourceL-s2026092701-r01`，owner `codex/root/RX-update-source-20261015`。发布时实际 Git HEAD 写入 `release_commit.txt`；父 E20 执行版本 `6693e4480ef990bc23c5021ff1440b547fe5fcaf`，父本轮源码版本以当前 Git archive 固定。N607 普通账户，CVS-RFFI Python，CPU2线程／GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01/experiments/cvs_rx_update_projection20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01/resolved_request.json`。唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-RX-update-projection-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01/source_probe.stdout.log`。完整矩阵／数据／checkpoint来源／技术停止条件／预期产物见 [experiment.json](experiment.json)。入口两份真实 checkpoint 无 query smoke PASS 后立即继续固定源诊断。

状态 PLANNED，无实际源结果。只有固定源判定通过后才另行固定新 E20 或 E20+20 训练；原默认头、隐藏 U、匹配纯 CE、冻结后 clean truth-last流程保持。SIG_TIME、固定 CENTER25K、COMPACT_LINEAR 和 LONG_SILU 历史正收益独立保留，不能相加不同预算／结构的数值。84% 默认头 clean 三种子目标仍 active、未达成。


## 完整源结果：固定变体关闭

**ANALYZED / VERIFIED，source gate 0/8**。INITIAL 0/4，FINAL20 0/4；UPDATE 与 GRAD 的独立位移条件在四个 FINAL20 单元通过，仍不足以晋级。执行代码 `846a539adeeec9a0d76d41e4d89e14bd3ce0705e` 已推送并独立核对 remote OID。归档 SHA、远端编译、进程/CWD/命令、CPU/GPU 与日志增长核实；两份真实 checkpoint 无 query smoke 通过，独立读回完整八记录、summary、自然终止与无 failure。原模型／优化器／梯度／RNG 保持，不影响其他 owner。

| 状态 | ordinal | RAW 正风险 RX | 位移保留 | 最差 RX 增量改善 | 平均 CE 相对 RAW 增加 | held feature RMS 差 |
|---|---:|---:|---:|---:|---:|---:|
| INITIAL | 0 | 0 | 1.000000 | +0.000e+00 | +0.000e+00 | 0.000e+00 |
| INITIAL | 17 | 0 | 1.000000 | +0.000e+00 | +0.000e+00 | 0.000e+00 |
| INITIAL | 35 | 0 | 1.000000 | +0.000e+00 | +0.000e+00 | 0.000e+00 |
| INITIAL | 52 | 0 | 1.000000 | +0.000e+00 | +0.000e+00 | 0.000e+00 |
| FINAL20 | 0 | 3 | 0.916051 | +1.080e-06 | +1.298e-06 | 4.938e-05 |
| FINAL20 | 17 | 3 | 0.925732 | +9.026e-07 | +1.053e-06 | 4.048e-05 |
| FINAL20 | 35 | 3 | 0.926354 | +1.142e-06 | +1.220e-06 | 4.734e-05 |
| FINAL20 | 52 | 3 | 0.919643 | +9.678e-07 | +1.136e-06 | 4.334e-05 |

INITIAL 四个原 AdamW 位移全部五 RX 风险均为负，GRAD 与 UPDATE 都原样返回。这不是机制未执行，而是该状态没有需要修正的方向。

FINAL20 四单元 RAW 都有三个源 RX 正一阶风险；GRAD_CONE 即使下降梯度已在风险半空间内，经过保存动量、clip 与 AdamW 后仍有三个正风险 RX。这次真实 checkpoint 诊断确认了“梯度可行不等于 AdamW 位移可行”的实现区别，不能进一步推断整个训练期的梯度冲突或泛化失败原因。

UPDATE 保留 91.6051% 至 92.6354% 原 feature 位移，与 RAW 的相对位移差为 .3775 至 .4020，与 GRAD 为 .3493 至 .3882。每单元有 16 个真实浅／中层参数张量不同，held joint160 RMS 差为4.05e-5至4.94e-5；primary Sinc 额外改变1/0/1/3个核。因此本次不属于完全零更新、仅修改分类头或无特征变化。

逐 RX 实际参数舍入均在记录的 solver＋实测误差界内。理想 FP64 可行不代表实际 FP32 位移严格非正：末态实际最大正风险仍为1.93e-9至2.90e-9，不能隐去这些残差或用全模型原始 Hz 尺度扩大容差。完整 source_RX_metrics 保留有限步风险／held CE，不能用一阶条件替代有限步观察。

source gate 失败来自独立 held-L 对照：FINAL20 的 UPDATE 平均 CE 比 RAW 高1.0533e-6至1.2979e-6，四单元均超出固定1e-6；最差 RX 增量改善只有两个单元超过1e-6。此处“平均 CE 高”是**相对 RAW**。相对更新前，RAW 的平均 held CE 降低1.8782e-6至2.6954e-6，UPDATE 也降低8.2489e-7至1.4750e-6，只是改善幅度较小。不能将本结果表述为 UPDATE 相对更新前一概变坏，也不能为使两个单元通过而放宽阈值。

实际消耗720独立L／6300metadata／10444packet前向／16临时CPU AdamW，约15.68 s source进程运行；0真实训练、头拟合、EMA、U/V IQ、U真值或target。240条 perRX/role/mode/cell 记录齐全。两个固定eval状态没有中期或真实U更新证据；末态LR1e-6且实际保存矩，初始LR2e-4且为空矩假想原配方。结论只关闭这个固定 Rx-average风险投影版本，不证明所有梯度方法无效。没有新预测/评分，clean性能没有更新。

下一独立假设转为可信身份监督下的主路径证据补偿：适度减弱实际PA/ref增量，将辅助梯度定向给a/SIG_TIME，并在目标概率达到要求时停止。只写出设计，未实现/发布/训练；先普查活跃机会与其源代理可靠性，再检验真实定向梯度和实际AdamW，而非继续扫描投影距离或微调科学阈值。见 [下一机制设计](../../../analysis/phase1_evidence_compensation_20261015.md)。原SIG_TIME、固定CENTER25K、COMPACT_LINEAR、LONG_SILU正收益分别保留。84%默认头clean三seed目标仍active、未达成。

机器可读分析见 [八单元表](source_cells.csv)、[240条RX记录](source_RX_metrics.csv)、[完整标量证据](result_evidence.json)及[发布终态证据](launch_evidence.json)。原私有不可变run、模型和原日志保留。
