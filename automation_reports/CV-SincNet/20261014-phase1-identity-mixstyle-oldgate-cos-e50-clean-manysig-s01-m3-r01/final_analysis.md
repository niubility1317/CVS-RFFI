# 身份保护浅中层 MixStyle：完整首种子结果

run：20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01；训练代码 b2a4146b7571d8713ddbd98cd096fb98b2b4c00c；固定 model seed 2026092701。两新模型 scratch50/11100，33 label＋17 pseudo；旧门控＋EMA＋默认 cosine，分类头正常联合训练，没有单独拟合。

## 结果与边界

MixStyle 尚未证明性能提升：clean 80.6095%，匹配控制80.8500%，配对差−0.2405个百分点；相对旧门控固定基线80.6685%为−0.0589个百分点。只是一模型seed，SD不定义，不冒充三seed均值或84%目标完成。测试为固定168000/row、旧6TX、目标7RX/4day、clean，不声称新TX或卫星部署。

| arm | clean准确率 | Macro-F1 | 最差RX | 最差day |
|---|---:|---:|---:|---:|
| OLD_GATE_COS50 | 80.6685% | 80.1734% | 70.6167% | 75.4524% |
| MATCHED_CONTROL | 80.8500% | 80.3734% | 70.9083% | 75.9429% |
| IDENTITY_MIXSTYLE | 80.6095% | 80.0385% | 66.9833% | 75.5548% |

对 MATCHED_CONTROL，候选修正2588个错误、引入2992个错误，净−404/168000。相对旧基线修正2715、引入2814，净−99。最差RX下降3.9250个百分点，最差day下降0.3881个百分点，负结果完整保留，不用平均差掩盖组别退化。控制相对历史+0.1815个百分点；本次两新模型初始191个state tensors逐元素相同，但实际GPU配方并非逐位确定。没有据此宣称控制与历史训练轨迹完全相同或为微小差异指定未经证明的原因。

## 实际接线与训练

候选只在主L_s监督前向的t1/t2混合mean/std，p=.5、alpha=.1、strength=1，同真实TX/day、不同源RX且不同真实物理ID；无配对保持identity。私有NumPy随机流，不新建模块参数/分类器/额外训练前向。不改U排序、原邻包门控、EMA、NoExplicitCFO或内部逐包补偿，不叠加PAIRS/Sinc/TIME。默认头6×160/scale30保持正常联合训练。

每层11100次主L调用、1420800包次，t1改变690849（48.6239%），t2改变695484（48.9502%）；未登记角色不混合。50轮及全22200新step、成功EMA/optimizer更新、source角色、17轮963900 U包/模型、U真值隐藏均完整重算。实际U接收：control15211/963900，candidate15001/963900，旧基线15214/963900；覆盖不是原生accepted-mean CE的loss乘数。没有解封U真值或使用目标统计训练。

L6300/U56700/V27000、10%训练池标注率、split392005及原物理RX/day/TX不变。V仅eval健康诊断，不求导/选模；固定final50，不按V或target选checkpoint。源终轮CE control0.287843、candidate0.289247，二者训练输入分布不同，不能把该差全部解释为拟合能力；V终轮97.8630%和97.3852%。source目标/标签阶段33轮U取批与前向0，后17轮沿原配方运行。

## 实际可迁移特征边界

实际fθ176065只删除类别W960，保留base_norm、pa_norm、gain及参考响应。两模型feature-only导出与原joint160逐元素一致；临时置零W不改变z；观察期间θ/buffers完全不变，eval、无拟合/优化。初始→终轮fθ改变174741/175128个标量分量；候选与控制终轮175127个fθ分量不同。固定L720 paired joint160平均L2=6.010110、cos=0.883807，全部115200特征分量不同。此证明机制进入实际导出特征及训练改变骨干，不证明新TX迁移或全部原头成绩来自骨干。普通换头不执行，服从用户明确禁止单独训练分类头。

两新checkpoint原生optimizer183参数顺序/shape与保存moments匹配，全部step11100。Sinc未叠加坐标修复；student真实核relativeL2均约1.17e−8，不把浮点极小变化写成有效滤波器学习。

## 完整L6300浅中层源机制

新两模型各50批、12600包次的冻结干预，旧基线同checkpoint/同shuffle源诊断直接复用。新25200包次、旧12600单列复用，CPU8.39＋8.33秒；V共享cache materialize后丢弃，无V前向/求导，无U/target读取。p/alpha/strength和同TX/day跨RX规则固定，不更新W或θ。

| 条件mean/std描述量 | t1 control→candidate | t2 control→candidate |
|---|---:|---:|
| RX条件变化占中心平方和 | 32.6890%→30.8785% | 28.2204%→24.4771% |
| TX条件变化占中心平方和 | 64.4704%→69.5496% | 66.9048%→73.8025% |
| 绝对RX条件平方和 | 2258.50→2934.35 | 4080.83→5465.15 |
| 总中心平方和 | 6909.05→9502.90 | 14460.57→22327.57 |

RX比例下降时绝对RX变化反而增加，因此不能解释为删除RX信息或因果域不变性。两个条件比例不是互斥方差分解，也不是特征秩。全RX已参与骨干训练，源分层不是留一RX泛化。候选默认头clean正确6232/6300，固定冻结干预后6250（修正22、破坏4）；control6264→6266（修正10、破坏8）。这是观察，不是训练有效性或目标泛化证明。

## 完整验证、成本与状态

唯一P0/P1审查PASS，负例/真实adapter梯度和数值更新/teacher-U-eval隔离/实际W边界检查PASS。单release archive SHA本地远端一次一致、remotecompile、real checkpoint zeroIQ no-query smoke PASS；owned PID/CWD/cmd/GPU/loggrowth已核对。所有worker终止，root EXPERIMENT_COMPLETE，零failure。3行全部预测固定后独立truth-last，共504000默认决策、54 CM/metric；全量audit CPU9.12秒，source_state/参数/全部step和50epoch重算。没有重跑、停止、热改健康任务；source qualifier首调用用了缺Torch的系统Python，未读真值或修改训练，改用既有实验环境后PASS，属于分析环境层恢复。

新成本2×50=100model-epochs/22200updates，历史50epoch独立复用。两新模型epoch时间合计control1943.32秒、candidate2078.20秒（model-time，非并行wall time），本次约+6.94%，不承诺普遍速度比；无新增参数/额外训练前向/单独头训练。桥接33/34额外V2次/模型，0V梯度；不把完整source-cache重复物化说成未接触V字节。

源科学条件PASS：两LCE≤.5、t1/t2实际修改≥25%、实际source_z差≥1e−6。这些是接线与健康反证条件，不是性能已经提升。继续预定义seed02/03、同机制/同预算/匹配scratch控制的决定已在读取target指标前写入并推送 9e79dc413967aee78fa87e5d0f6c7408c44e4efc，本轮target负差不回流调整p/alpha/层位/父权重/seed/停轮，也不选择性重跑。另注册4个新模型、共200model-epochs；首seed不重训，最终合并真实三seed统计。当前新矩阵尚未启动，84%保持active。

产物：results.json包含完整audit、逐seed/分组CM、50epoch曲线、U RX/day计数、错误迁移和optimizer证据；source_features.json为真实fθ；source_style_mechanism.json为完整源机制；source_qualification.json与seed_expansion_decision.json为不读target的固定源决策。Checkpoint、IQ、原日志、大型prediction留既定远端run，未上传。
