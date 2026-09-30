# 注册机制诊断之后：继承状态的残差分类头候选

状态：**IMPLEMENTATION_APPROVED_NOT_RUN；已纳入 LocalMargin 完整 support 证据，获准实现及合成验证，未启动真实实验。** 本文只依据完整 160 parent 原 LocalRidge support 诊断、完整 LocalMargin support 结果、已完成的 D11 support 日志复盘及冻结方法源码。未读取 query、ABC 结果、总索引、交接或 LocalMargin partial 指标。当前其他健康实验保持原样。

## 1. 证据支持什么

当前诊断在两个冻结模型、两个 receiver、两个 practical residual 场景、一个 support seed 上完整执行。信道为 residual / post_sync / noeq，25 MHz；实际选择为 rx3 的 19-1 与 rx1 的 20-19，场景均为 practical_high、practical_low_urban。未覆盖 practical_mid、其余 receiver、其余模型或 support seed。K 为 1/5/10/20，旧类 6 个，新类 0/2/5/10/20 个；真实 K1 仅有数值身份检查，无拟合或持出准确率。

以下聚合均为新类存在的 96 parent；OOF 与 proxy 分开，单位为百分数或百分点：

| 诊断 | B0 旧 | C_old 旧 | C0 旧 | C0 新 | H | 旧内部有符号损失 | 新类竞争损失 | 总旧损失 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OOF | 71.042 | 71.259 | 63.767 | 54.846 | 58.417 | -0.217 | 7.491 | 7.274 |
| 单 anchor proxy | 52.147 | 52.949 | 44.354 | 34.108 | 36.275 | -0.802 | 8.595 | 7.793 |

B0 是旧类 support 拟合，C0 是旧新类 support 独立联合拟合；C_old 是固定 C0 分数的旧类列限制。**这不是地面 A→微调 B→注册 C 的完整结果**，真实 A、微调 B 和适应收益均为 N/A。

平均新增竞争损失明显大于净旧内部排序损失，支持把下一步重点放在完整注册竞争。它不证明旧头已经无需改善：OOF / proxy 的旧类 winner 改变比例分别约 5.017% / 9.416%，净正确性变化会抵消正反翻转。旧样本被新类获胜的比例为 18.620% / 30.804%，也不能等同于额外正确性损失，因为其中原本就有旧类错误。新类平均准确率仍低于旧类；直接给所有旧类加常数，可能以继续损伤新类为代价缩小旧类下降。

Nnew20 时，OOF 的 K5/10/20 新类竞争损失为 15.000/9.375/11.979 pp，并非严格随 K 单调；完整分层不能由一个总均值替代。完整 K×新增类数及 model/cohort/receiver/scene 数值保留于原报告和 CSV，不把其未测部分外推到所有目标域：

- [完整诊断报告](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-registration-diagnostic-m2-r01/results/support_summary/report.md)
- [全部 K×新增类数](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-registration-diagnostic-m2-r01/results/support_summary/by_k_new_count.csv)
- [总体口径](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-registration-diagnostic-m2-r01/results/support_summary/overall.csv)
- [模型与 cohort 分层](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-registration-diagnostic-m2-r01/results/support_summary/by_model_cohort.csv)
- [receiver 与场景分层](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-registration-diagnostic-m2-r01/results/support_summary/by_receiver_scene.csv)

D11 的完整 support 复盘已经表明：rank8、恒等起点、表示保持正则、冻结继承本身不足以防止新增竞争抵消适应收益；训练 loss 下降也不代表 held 准确率提高。其数据、信道、基座和日志限制与当前不同，不能把历史数值当成现行效果。见 [D11 完整 support 复盘](D92_HISTORICAL_D11_SUPPORT_RECHECK_20260930.md)。

随后完成的 LocalMargin 分析覆盖完整 4,800 parent、3,600 OOF parent、42,000 proxy anchors。相对同折 LocalRidge：

| 诊断 | parent K | Δ旧 pp | Δ新 pp | ΔH pp |
|---|---:|---:|---:|---:|
| standard | 5 | +0.632 | +0.542 | +0.660 |
| standard | 10 | +0.358 | +0.541 | +0.511 |
| standard | 20 | +0.365 | +0.642 | +0.551 |
| oneshot proxy | 5 | +0.024 | -0.020 | +0.011 |
| oneshot proxy | 10 | -0.028 | -0.023 | -0.026 |
| oneshot proxy | 20 | -0.049 | -0.033 | -0.046 |

标准 OOF 有小幅同向收益，proxy 的差异很小，不能称为显著退化；但它未满足既有两个 support screen，不推进其 query benchmark。该完整矩阵比本次机制 pilot 更广，不能把两份报告的绝对均值直接相减作方法比较，也不能把 proxy 的微小负值推断为所有真实 K1 场景更差。

成本是另一条明确经验：LocalMargin / LocalRidge 的平均单次 fit 时间在 OOF 为 1.197981 / 0.094104 s（约 12.73 倍），proxy 为 0.294636 / 0.004106 s（约 71.75 倍）；整个四 arm 诊断墙钟 11,398.950 s。该结果不支持仅为不到 1 pp 的 OOF 改善继续增加大规模迭代。它也不证明交叉熵优于平方 hinge；本文选择 CE 残差头是待验证的分类目标与输入相关修正假设，不是从 Margin 结果推导出的性能保证。

来源：[LocalMargin 完整 support 报告](E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-margin-support-m4-r01/results/support_summary/report.md)、[完整统计及 fit_stages 成本](E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-margin-support-m4-r01/results/support_summary/summary.json)、[全部 K×新增类数](E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-margin-support-m4-r01/results/support_summary/by_parent_k_newcount.csv)。

## 2. 单一候选与最小取舍

候选暂称 **LocalRidge-SequentialResidualHead8**：保留原 LocalRidge 作为冻结基线分数，在其上增加一个输入相关的小残差分类头。B 阶段用旧类 support 学习；C 阶段继承瓶颈及旧类输出列，添加零初始化的新类输出列，然后用全部合法注册 support 联合训练。C 的基准 LocalRidge 仍按原算法用旧新类 support 重拟合，能够产生全部新类分数。

关键假设是：**C 阶段直接学习旧新类竞争，加上对 B 状态的近端约束，可能比只冻结旧适应状态并加入新类更能保留旧收益，同时改善新类。** rank8 是预先选定的小预算容量，不是创新或效果证据。继承也不保证旧分数、旧排序或准确率不变。

纯 class bias/temperature 校准虽然更小，但若两个样本具有相同原分数而需要不同修正，它不能区分；全局 temperature 更不会改变 argmax。这里允许残差依赖 160D z_id，能够学习样本相关修正。其能力仍受 8 维瓶颈、固定 encoder 和合法少量 support 限制；若错误来自当前表征不可分、z_id 缺失信息或所需修正高度复杂，候选可能失败。它也可能过拟合、迁移旧类方向到新类时产生负迁移，或仅改善训练 loss。不要将这些失败事后归结为缺少更大 rank，再在同一证据上无边界扫参。

本轮不实现 source prototype 派生状态、不更新 encoder/BN、不做旧类固定加分、不增加角色条件推理、不尝试多组损失或 rank 网格。LocalMargin 的完整证据使这一候选优先采用固定小维度、固定步数，并且仍以 LocalRidge 为直接效果与成本对照；不继续放大其原 RKHS margin 迭代预算。

## 3. 精确前向与基线尺度

记当前阶段训练集为 `S={(z_i,y_i)}`，N 为真实物理 support 数，C 为本阶段注册类数，d=160、r=8。z 是原单观测缓存 z_id，按冻结分支的 norm_floor=1e-12 逐样本 L2 归一化；不新增 view，不从 source 或 query 估计统计。

先用当前阶段合法训练 support 拟合原 `fit_branch_local_ridge(..., arm='local_ridge')`，得到冻结分数向量 `f_S(x)`。B 只拟合旧类，C 重新拟合全部注册类；原 kernel 的带宽、中心化、迹匹配、物理和 ridge=1、退化分支和 float64 解法不改。其五块缓存与所有拟合状态冻结，残差训练不对 kernel 或 encoder 反向传播。

从该阶段训练 support 的冻结分数确定一个正标量：

`q_S = sqrt(mean_i (max_c f_S(z_i)_c - min_c f_S(z_i)_c)^2)`。

平方和先按最大 range 缩放，再求缩放后的均方根并乘回，避免直接平方溢出或提前除 sqrt(N) 导致下溢。仅在所有原 range 精确为零时定义 q_S=1；原 range 非零但 RMS 低于 float64 最小可表示值时明确技术失败，不能冒充零范围。非有限值同样为技术失败，不静默调温度或换核。B/C 都用这一相同规则，分别只从各自训练 support 计算。它让基线 logit 的训练集 RMS 范围为 1；正标量不改变原 argmax。它不是概率校准，不保证 B/C score 分布相同，不用于跨拟合 raw margin 相减，也不利用 held/query 数值。

残差前向为：

`h_U(z) = GELU(sqrt(d) · z U) / sqrt(r)`；

`g_S(z) = f_S(z)/q_S + h_U(z) V`，其中 `U∈R^(160×8)`，`V∈R^(8×C)`。

GELU 使用精确 erf 形式 `a·(1+erf(a/sqrt(2)))/2`，无 bias、dropout 或 BN。V 的列由真实物理 class ID 绑定。所有分数列一起 argmax，精确并列采用原物理 class ID 字典序。所有 query 必须独立面对全部已注册类，不能根据真实 old/new 角色选择 B、C_old 或不同温度。

初始化 V=0，因此初始残差为零，预测严格等于原 LocalRidge。U 使用确定性非零 DCT 列：`U[a,b]=sqrt(2/d) cos(pi·(a+0.5)·(b+1)/d)`，a=0…159，b=0…7；没有初始化随机 seed、随机 minibatch 或未登记数据依赖。若以后实现框架需要 seed 字段，记 null / deterministic，而非混用 model seed。非零 U、零 V 使首步 V 通常有非零梯度；不得以零残差为由在训练前向中走恒等 shortcut，跳过可训练分支。

## 4. B/C 状态继承与训练目标

Stage A：使用既有合规冻结地面预测器，本候选不修改或派生该状态。当前 support pilot 没有它的合法同样本预测时记 N/A；B0 support 分类器不可冒充 A。

Stage B：先拟合旧 support 的 LocalRidge 和 q_B；以 U0、V0=0 为初始参数和近端锚点，执行下述固定训练。保存 U_B、V_B、旧类列映射、原 B 基线与 q_B。B 的训练与 C 的旧 support 必须逐物理 ID、标签和缓存特征一致。

Stage C：用同一旧 support 和当前新 support 重拟合 C 基线与 q_C。初始化 `U=U_B`；将 V_B 的旧列按 class ID 放入 V，全部新增列初始化为 0。初始新类 logit 因此是完整 C LocalRidge 的新类分数除 q_C，并非零；旧类 logit 是完整 C 基线的旧列加继承残差。不存在混接 B 原核旧列和 C 原核新列。近端锚点为这个继承参数对 `(U_anchor,V_anchor)`，之后 U 与全部 V 列都允许更新。

每阶段最小化同一个目标，写作平均损失方便数值实现：

`J = (1/N) sum_i CE(g_S(z_i), y_i)`

`    + [||U-U_anchor||_F² + ||V-V_anchor||_F²] / (2N)`。

等价的物理和目标为 `sum CE + 0.5||ΔU||² + 0.5||ΔV||²`；两个近端系数均为 1，不按 train loss 或 held 表现选择。报告平均 CE、两个独立正则分量和总损失，不能同时套用 Adam weight decay 重复正则。每个物理样本权重为 1；当前每类实际 K 相同，因此全注册类具有相同总监督权重。不存在按 old/new 组、已知错误类别或结果好坏加权。

Stage B 的锚点是初始参数；Stage C 的锚点包含已学 B 状态。因此 C 的统计先验不是 old/new 角色完全交换对称，但这是明确的顺序状态继承，不是每个旧类的固定 logit 奖励。类名重命名并同步角色和列映射时，loss 与预测等变。C 的全部类参与同一 softmax 分母，新增列能从第一步得到梯度，旧列也能调整以解决实际竞争。

Nnew=0：直接复用 B 的基线、q、残差和分数为 C，不再训练；旧类下降严格为 0，新准确率/H 为 N/A。真实 full-support K1 若将来进入已冻结方法推理，则可依法执行这一监督训练，但其每类单个正样本没有类内泛化证据，不能把训练准确率当成效果。当前 pilot 的真实 K1 仍不拟合，只记录输入/数值检查；学习 K1 的行为由 parent K5/10/20 的单 anchor proxy 单独观察。

## 5. 固定优化、保护与技术失败

每次阶段训练固定 64 个全批次 Adam 更新，float64，learning rate=0.01，β1=0.9、β2=0.999、epsilon=1e-8，无 weight decay、无调度或早停。优化 J 的平均形式。合并 U/V 梯度后采用全局 L2 norm 上限 1 的确定性梯度裁剪；先记录未裁剪梯度，再执行裁剪与 Adam。所有常数是预算和稳定性假设，不是从 query 或最佳历史 epoch 选得，尚不宣称最优。

有限步数、低维头、固定原基线、物理和近端项及梯度裁剪是仅 train 的保护；不保证泛化。尤其 K1 下没有同类第二个物理样本，K5 也易记忆。不得用外层 held loss 决定 64 步中的最佳状态、挑 rank、调整正则或重跑。最终状态固定取第 64 次更新，不把较早 held 峰值替换它。

可采用 NumPy float64 与 scipy.special.erf 的解析梯度，或数学等价 autograd；验证独立有限差分后才接入。梯度不能穿过缓存、q、原 kernel 或原基线系数。C 中继承的是参数值与近端锚点，**Adam 一二阶矩在每阶段重置为零**；不将 B 的 optimizer 动量伪装为相同预算的 C 更新。完整轨迹需明确 update 前/后的记录口径。

非有限 loss/梯度/参数/分数、维度/类映射错配、原 ridge 技术失败立即记录失败上下文，保留 parent/fold/anchor、实际训练与 held IDs、当前阶段、已完成 base fits/更新、最后有限状态和原因。不静默回退 LocalRidge、不缩 learning rate、不加 jitter、不自动重试；其他健康 row 按原调度完成。全部零 z 时 h=0，残差学习无效是可预期退化，记录梯度零及确定性预测，不造随机特征；单类时同样按公式处理。

## 6. 最小可区分对照及完整 pilot

只用三个预先固定路径，不形成参数搜索：

| 路径 | B | C | 可以识别的差异 |
|---|---|---|---|
| R0 | 原 LocalRidge B0 | 原 LocalRidge C0 | 直接基线 |
| R_reset | 同一个训练好的残差 B | C 残差从 U0/V0=0 训练，锚点也是 U0/V0 | C 增加残差头及其全类监督相对 R0 的效果 |
| R_seq（候选） | 与 R_reset 共用完全相同 B | C 继承 B，近端锚点也是继承值 | 相对 R_reset 的顺序继承整体效果 |

R_seq−R_reset 同时包含初始化与近端先验的继承，不能宣称仅隔离了 warm-start。两者 C 的原 kernel、q、训练物理 ID、64 步预算、正则系数和 loss 形式完全相同。B 的同一实物状态可复用以避免重复训练。所有路径的原 B0/C0 基线 fit 也可共享不可变结果，实际 base factorization 只计一次；这不是跨 parent 拟合状态复用。Nnew0 两种残差路径都严格复用 B，不制造伪注册更新。

第一轮沿用已完成诊断的同一 160 parent、2 CPU lanes×2 BLAS：两个 model、两个 cohort 的各两场景键、单一 support seed，K1/5/10/20×Nnew0/2/5/10/20。每个外层物理 OOF fold、每个 proxy anchor 都从其 train-only 缓存重新拟合 B/C 和残差，不能复用完整 parent 已训练的 U_B。OOF 先池化 held 样本，每个物理 held ID 恰好一次；K5 的 2/2/1 不等折不能等权平均折准确率。proxy 先平均 parent 内全部 anchor，再等权 parent，重复样本不当独立重复。

基线实际拟合预算仍为 3,168 次；共享后的残差 B 训练为 1,760 次，Nnew>0 的 C 两对照各 1,408 次，总计 4,576 次阶段训练、292,864 次 Adam 更新。真实 K1 parent 不包含在这些拟合次数中。最初先做合成数值/资源验证，不要求先跑完整 4,800 parent；随后必须完成预设 160 parent 的全部路径再解释 pilot，不从部分结果择优。是否扩展依据完整证据讨论，不新建全 strata 必须正改善门槛。

报告每一 K×新增类数中的 A、B0、残差 B、C_old、全注册 C 旧/新、H、相对直接 LocalRidge 的两类变化、B−A、B−C_old/C_old−C 的分解、总注册下降和新旧绝对差；A 缺失记 N/A，B−B0 只能称 support 分类器上的适应增量。保留 old-only、模型、receiver/scene 分层和六种正确性转换。10/1/3 pp 为理想方向；不把减少旧损失但降低新准确率，或以降低旧准确率缩小新旧差距，描述为同时改善。

## 7. 资源、日志与可反驳结论

参数上限为 U 的 1,280 个加 V 的 8C 个，C≤26 时共 1,488 个；float64 的 C 残差参数为 11,904 字节，加 q 为 11,912 字节，不含类别映射/序列化开销。B 的参数为 1,328 个。部署若只需 C，可以释放 B 模型及 optimizer；A/B/C 审计所保留副本与实际部署常驻状态分别报告。此数字只是新增数值参数，不含原 LocalRidge 核状态、160D 缓存及原 encoder。

每样本残差前向约 `160×8+8C` 个乘加，C26 为 1,488 个，另有 8 次 GELU 和基线 logit 除法；原 query encoder、五分支与核评分成本仍全部存在。训练无需 encoder 反传，但每阶段还需 base fit、训练 support 的原分数、64 次残差 forward/backward 和 Adam。N520/C26 时，64 次残差前向约 4,952 万乘加，反传/更新另计；这不是实测耗时，也不据参数少宣称总算力降低。

测量同一硬件、线程数和 dtype 的训练/推理墙钟，原 base fit、base score、残差 forward/backward、Adam 分项，CPU 峰值 RSS、可用时显存、实际常驻数组字节、部署包与新增传输字节；未测量为 N/A。完整诊断原基线墙钟 109.111 s、head fit 耗时总和 47.865 s、score 耗时总和 148.122 s 只能作该既有环境的参考，并发工作量之和不等于墙钟，不能直接外推新预算。64 次全批次更新与 LocalMargin 的 row-block 更新计数不是同一个工作单位，不能据次数直接宣称更快。先对 N26/364/520 的合成合法形状测实际总成本，再按完整 pilot 同样口径报告相对 LocalRidge 的增量；若代价仍大且收益很小，应如实说明收益成本取舍，不增加轮数追逐小幅提升。

每阶段启动打印实际 U/V 形状、训练类别数/物理 K、原核 tau/trace/q、所有优化常数、继承来源与零新增列数。每一步保存完整结构化和紧凑 JSONL/CSV，同时输出 CVS 风格文本：CE、U/V 正则、总 loss、lr、裁剪前后梯度范数、参数/残差范数、相对锚点位移、梯度零状态、实际更新计数与耗时。源域验证 N/A（禁止访问），held/query 不进入逐步训练监控。训练准确率单独明确为 train-only；外层 held 只在预定最终状态评估。

合成测试至少覆盖：首步 V 梯度可识别；独立有限差分；零 residual 初始预测与原基线一致；B/C q 的 train-only 和正尺度 argmax 不变；新类初始分数来自 C 基线；C 确实继承 B 值但清空 optimizer；类别重命名/列置换等变；全部类同权 CE；Nnew0 比特级复用；零输入/精确并列/非有限技术失败；物理 train/held 隔离；K1 无虚构 held；对照共享与实际成本计数。

可反驳预测：若 R_reset 或 R_seq 在完整 pilot 中不能同时改善旧/新持出表现，或继承相对 reset 普遍损伤新类，即否定当前候选所期待的竞争修正/继承收益；训练 CE 降低、参数少、某一个 K 的 H 升高均不能替代这项判断。这里的“否定收益”是证据解释，不是新增自动 gate。若新增竞争项降低却只是把错误转移给新类，按完整两类指标明确报告。所有结论限于该 support pilot，不能宣称真实 K1、未测场景或 query 泛化改善。

LocalMargin 完整证据已纳入；下一步由 root 根据本设计进入有界实现及合成验证，或指出具体科学/实现缺口。本文不构成已实现、已训练或有效性声明。
