# SequentialResidual 完整 support 失败复盘与结构性取舍

结论：**该候选不晋级。停止围绕 residual rank、步数或 learning rate 做小幅试探。** 完整 pilot 已表明，残差头在 B 阶段就损伤原 LocalRidge 的旧类持出表现，C 阶段的完整竞争训练没有同时改善旧新类；继承相对重置主要表现为旧新权衡。本结论限于已测 support，不涉及 query、ABC 或未测场景。

本文仅修改本文件，不修改既有方法、配置或实验。下一步提出的是可反驳的结构方向及最小证据需求，**尚未冻结、实现、预登记或启动**。已有优化目标授权覆盖必要的新实验，后续按既有最小流程落实具体方案。

## 1. 完整证据与准确口径

来源为 [完整报告](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/support_summary/report.md)、[summary](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/support_summary/summary.json)、[overall.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/support_summary/overall.csv)。160 parent、4 model/cohort row、三条路径完成；实际 base fits=3,168、residual stages=4,576、Adam updates=292,864。真实 K1 不训练、不生成持出指标。

两个模型为 2026092701/2026092702；receiver 为 19-1/20-19；场景为 practical_high/practical_low_urban，沿用 residual/post_sync/noeq、25 MHz；support seed 为 2026092711。旧类固定 6 个，新增 0/2/5/10/20 个。未覆盖 practical_mid、其余 receiver/模型/support seed。proxy 来自 parent K5/10/20，不能冒充真实 K1。OOF 先池化物理 held 样本，proxy 先平均 parent 内全部 anchor。

overall 的 new_present 登记为 128 parent，其中 32 个 K1 指标 null，实际每项准确率均值来自 96 个有持出证据的 parent；old_only 为 32 个，其中实际 24 个有持出证据。不得把 null 当 0，或把 128 写成有效样本数。A 和 B−A 全部 N/A；B0 是原旧类 support 分类器，不是真实地面 A。

### 1.1 B 阶段已经失败

R_reset 与 R_seq 共享同一个 B。准确率为 %，变化为 pp。

| 诊断 | parent K | B0 旧 | 残差 B 旧 | B−B0 |
|---|---:|---:|---:|---:|
| OOF | 5 | 63.333 | 57.500 | -5.833 |
| OOF | 10 | 73.542 | 69.375 | -4.167 |
| OOF | 20 | 76.250 | 71.562 | -4.687 |
| proxy | 5 | 50.938 | 50.000 | -0.938 |
| proxy | 10 | 53.032 | 53.426 | +0.394 |
| proxy | 20 | 52.473 | 51.908 | -0.565 |

OOF 聚合 B0 71.042%→B 66.146%，下降 4.896 pp。Nnew0 也出现相同 B 退化，因其 C 严格复用 B，不能把这部分损失归因于新类竞争或 C 继承。

### 1.2 C 阶段两条路径均弱于原 LocalRidge

下表均为 new_present 的 96 个有效 parent；H 和绝对新旧差先在 parent 内计算再平均，不由表中两个总均值重新计算。

| 诊断 | 路径 | C 旧 | C 新 | H | C 旧−R0 | C 新−R0 | 平均绝对新旧差 |
|---|---|---:|---:|---:|---:|---:|---:|
| OOF | R0 | 63.767 | 54.846 | 58.417 | 0 | 0 | 10.740 |
| OOF | R_reset | 54.696 | 40.521 | 44.998 | -9.071 | -14.326 | 17.250 |
| OOF | R_seq | 55.252 | 39.169 | 44.352 | -8.516 | -15.677 | 18.372 |
| proxy | R0 | 44.354 | 34.108 | 36.275 | 0 | 0 | 15.853 |
| proxy | R_reset | 39.449 | 26.534 | 29.135 | -4.905 | -7.573 | 18.124 |
| proxy | R_seq | 41.734 | 23.218 | 27.366 | -2.620 | -10.890 | 21.205 |

R_seq−R_reset 的 OOF 旧类为 +0.556 pp、新类为 -1.352 pp、H 为 -0.646 pp；proxy 旧类为 +2.285 pp、新类为 -3.316 pp、H 为 -1.770 pp。它比较的是**初始化加近端锚点的整体继承**，不能单独归因为 warm-start。这里没有得到两类同时受益的继承证据。

| 诊断 | 路径 | B→C_old 有符号损失 | 新类竞争损失 | B→C 总旧损失 |
|---|---|---:|---:|---:|
| OOF | R0 | -0.217 | 7.491 | 7.274 |
| OOF | R_reset | -0.061 | 11.510 | 11.450 |
| OOF | R_seq | 1.293 | 9.601 | 10.894 |
| proxy | R0 | -0.802 | 8.595 | 7.793 |
| proxy | R_reset | 1.116 | 11.213 | 12.329 |
| proxy | R_seq | 1.691 | 8.353 | 10.044 |

特别是 proxy R_seq 的竞争项 8.353 pp 略低于 R0 的 8.595 pp，却伴随更差的 B、旧内部变化及新类准确率。因此“竞争项降低”不能独立证明方法改善。新增类从错误旧预测手中获胜也不是新增正确性损失；继续使用六种正确性转换，而非仅数预测为新类的旧样本。

完整 K×新增类数及分层保留在 [by_k_new_count.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/support_summary/by_k_new_count.csv)、[by_model_cohort.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/support_summary/by_model_cohort.csv)、[by_receiver_scene.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/support_summary/by_receiver_scene.csv)；正文不筛除退化 strata，也不新增“所有 strata 必须正向”的 gate。10/1/3 pp 仍只是理想方向。

## 2. 从源码能确认什么，不能确认什么

源码为 [d92_sequential_residual_head.py](../code/cvsrffi/d92_sequential_residual_head.py)。原 LocalRidge 五块核保持冻结；残差使用单独 z_id，经 `GELU(sqrt(160)·unit(z)U)V/sqrt(8)` 直接加到已缩放的原分数。B/C 都优化训练 support CE 加参数近端项。首步 V=0、U 非零；C_seq 继承 B 的 U/旧 V，C_reset 从初始参数重训；C 原核均按全部注册 support 重拟合；两者 q 相同。没有 held 早停、最佳步选择或 query 输入。

已确认的结构事实：

1. q 是正标量，V=0 时只缩放原分数，不改变原预测。新增残差会改变决策；B 的损失不能归因于“只做了温度缩放”。q 仍影响 CE 优化相对尺度，因此不能进一步断言它对最终训练无影响。
2. 近端项约束 U/V 参数变化，不约束每个输入的分数变化或原核决策 margin。DCT 非零 U、sqrt(160) 投影与非线性使“参数距离小”不等于“预测扰动小”。这是公式性质，不是当前实际扰动大小的测量。
3. 仅用 z_id 的有监督残差可以覆盖原五块核融合分数，训练没有显式要求保留局部邻域或多分支互补信息。它是否实际主导了分数，须看训练/held 上对应分数尺度，不能仅凭结构断言。
4. C 的完整 softmax 竞争目标和参数继承确实实现了，但有完整竞争训练并不保证持出旧新同时改善；本次 reset 和 seq 都退化已经反驳该预算下的收益预期。
5. B 已经变弱，C_seq 的近端锚点也来自这个变弱 B。继承可能传递不利状态；本实验隔离的是整个继承组合，不能拆出初始化与正则哪一个造成了具体退化。

当前完整汇总足以否定候选收益，尚不足以确定唯一失败原因。**不能把“训练 CE 下降、held 下降”直接写成已证明的过拟合因果**：还可能包括训练目标与判决几何不匹配、残差表征限制、尺度和有限步优化路径、近端先验不合适，以及小样本统计不确定性。参数多、K 小使记忆训练样本成为合理假设，但不是已完成的识别。

### 2.1 全部训练曲线已核查

全量 collector 状态为 `COMPLETE_TRAINING_LOG_SCAN_VERIFIED`，已完整解析 4,576 阶段、292,864 次更新，扫描四个 row 的 training.log、probe.log 与发布 run.log。本文用标准库完整读取 [training_diagnostics.json](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/training_diagnostics/training_diagnostics.json)、全部 4,576 行 [stage_summaries.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/training_diagnostics/stage_summaries.csv)、全部 [curves.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/training_diagnostics/curves.csv) 与 [snapshots.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-sequential-residual-support-m2-r01/results/training_diagnostics/snapshots.csv)，没有用抽样阶段代替全曲线。

21 个 scope×stage×train K 组，每组恰有 64 个曲线点，共 1,344 个组步点、24,192 条曲线指标行；另有 63 个首步/末步/final 快照、1,134 条快照指标行。每个点在该组内对实际训练阶段等权，不等于持出分析的 parent 权重，阶段和步也不是独立重复实验。OOF parent K5 对应 train K3/4，K10 对应 6/7，K20 对应 13/14；proxy 的 train K 始终为 1。

下表完整列出 21 组。初始是第 1 条更新日志的 **pre-update** 状态；final 是 64 次更新之后独立记录的状态。`r` 是每阶段训练 residual RMS/base-logits RMS 比值再取组均值，分母为缩放后的当前阶段 LocalRidge logits。`首超 1` 指组均 r 首次大于 1 的日志序号，序号 t 的损失和 RMS 对应 t−1 次更新后，因此 11 表示完成 10 次更新后；它不是每个阶段的首次越界时间。

| scope | 阶段 | train K | 阶段数 | CE 初始→final | 训练准确率初始→final（%） | final r | 首超 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| proxy | B | 1 | 1,400 | 1.0715→0.4562 | 100.00→100.00 | 2.190 | 11 |
| proxy | C_reset | 1 | 1,120 | 1.7887→0.7580 | 100.00→99.96 | 5.083 | 9 |
| proxy | C_seq | 1 | 1,120 | 1.4539→0.6745 | 74.25→99.87 | 6.614 | 1 |
| OOF | B | 3 | 80 | 1.0801→0.4366 | 100.00→99.31 | 3.108 | 11 |
| OOF | B | 4 | 40 | 1.0809→0.4219 | 100.00→98.96 | 3.363 | 11 |
| OOF | B | 6 | 40 | 1.0831→0.4175 | 100.00→97.92 | 3.875 | 11 |
| OOF | B | 7 | 80 | 1.0804→0.4227 | 100.00→97.92 | 3.957 | 11 |
| OOF | B | 13 | 80 | 1.0827→0.4324 | 100.00→97.84 | 4.647 | 11 |
| OOF | B | 14 | 40 | 1.0821→0.4523 | 100.00→97.77 | 4.441 | 11 |
| OOF | C_reset | 3 | 64 | 1.7986→0.8519 | 100.00→96.05 | 6.932 | 10 |
| OOF | C_reset | 4 | 32 | 1.8022→0.8754 | 100.00→95.23 | 7.719 | 10 |
| OOF | C_reset | 6 | 32 | 1.8043→0.8881 | 99.95→93.30 | 8.517 | 10 |
| OOF | C_reset | 7 | 64 | 1.8031→0.9255 | 99.99→91.70 | 8.637 | 10 |
| OOF | C_reset | 13 | 64 | 1.8073→0.9701 | 99.97→89.17 | 9.626 | 10 |
| OOF | C_reset | 14 | 32 | 1.8074→0.9818 | 99.97→90.21 | 9.897 | 10 |
| OOF | C_seq | 3 | 64 | 1.4801→0.7656 | 66.25→95.94 | 9.345 | 1 |
| OOF | C_seq | 4 | 32 | 1.4858→0.8024 | 62.63→94.93 | 10.156 | 1 |
| OOF | C_seq | 6 | 32 | 1.4926→0.8240 | 62.83→93.01 | 11.846 | 1 |
| OOF | C_seq | 7 | 64 | 1.5190→0.8588 | 62.61→90.84 | 11.925 | 1 |
| OOF | C_seq | 13 | 64 | 1.5504→0.9349 | 62.82→88.04 | 13.158 | 1 |
| OOF | C_seq | 14 | 32 | 1.5629→0.9521 | 62.96→88.34 | 13.213 | 1 |

全曲线和逐阶段端点给出以下事实：

- 21 组的 CE 与 total 均值在日志序号 1 至 64 没有向上跳变（按 float64、1e-12 比较容差）；所有 4,576 个阶段的 final CE、final total 都严格低于本阶段初始值。因此不是“训练目标根本没被优化”。这不等于每个阶段每步都单调：最后一次更新有 803 个阶段 CE 上升，5 个 proxy B 阶段 total 上升，最终仍低于初始。不能从均值曲线掩盖这些个体变化。
- B 初始训练准确率均为 100%，OOF 中 210/360 个阶段 final 训练准确率下降；C_reset OOF 中 268/288 个阶段下降。CE 优化能提高部分样本的概率而牺牲另一些样本的 argmax，近端惩罚也改变目标。因此本次并不是简单的“训练分类越来越准、只有持出变差”；已观测的是预设连续目标下降而原判决没有得到保留。该目标与判决保持不一致，是事实；它造成 held 退化的全部因果份额仍未知。
- C_seq 从继承 B 后的状态开始，初始训练准确率已低于 C_reset，OOF 组均约 62.61% 至 66.25%，proxy 为 74.25%；初始组均 r 已超过 1。这里 C 的原基线和 q 已按全注册类重拟合，旧残差列和新增零列与它相加。训练后 C_seq 能显著修复自己的初始训练错误，但不能据此认为其持出表现优于 R0 或 reset；第 1 节的实际结果相反。初始化和近端锚点仍是同时变化的因素。
- 所有组最后一步平均梯度范数仍非零，范围约 0.0292 至 0.0502。64 步是固定预算，日志没有收敛证明，也没有证据支持“继续加步就会改善持出”。阶段内近端锚点位移非零：OOF B 的 final U/V 距离组均分别为 1.812 至 3.026/1.938 至 2.385，C_reset 为 3.351 至 5.349/3.329 至 3.666，C_seq 为 2.693 至 3.853/2.748 至 3.059；C_seq 距离是相对于继承 B，而非原 DCT。
- nonfinite=0、零总梯度=0；clip 共 30/292,864 次（约 0.0102%），全部属于 OOF C_seq。文本中的 error、warning、traceback、OOM/Killed、非有限、恢复和早停匹配数均为 0。重复出现于 training.log 与 probe.log 的同一事件不重复计为独立更新。没有观测到数值崩溃、频繁裁剪或未执行优化导致退化的证据；这些检查不能证明不存在任何实现缺陷。

RMS 结果证实：**残差在训练 support 上很早就不再是相对基线的“小幅值修正”**。但 raw logits RMS 含对所有类共同的平移成分，而共同平移不改变 softmax 或 argmax；因此 r>1 不能直接解释为残差主导了类别 margin，更不能外推 held 或 query 的幅值。现有提取没有逐样本、逐步 held 分数，没有记录用于本次解释的类中心化残差 margin 分解；这些量记缺失，不从平均 RMS 推算，也不挑最佳步重算模型。

完整日志将解释收窄为：预设优化真实执行且降低目标，训练决策保持和持出泛化却没有随之改善；继承也未消除这种不一致。它支持停止当前“强核分数外加可自由改类分数的 CE 头”小修路线，不证明 PEFT 普遍无效，也不证明下一种度量结构一定有效。

## 3. 实测成本没有支持继续这条小修路线

本次 CPU-only、float64、2 lanes×BLAS2，运行环境记录 96 逻辑 CPU，未记录更具体 CPU 型号。实测墙钟 302.153 s，lane 墙钟之和 599.638 s；并发工作量不等于墙钟。

| 分项 | 实测 |
|---|---:|
| 原 base fit 总时间 | 47.932 s |
| base train-score 总时间 | 80.992 s |
| base held-score 总时间 | 148.962 s |
| residual fit 总时间 | 231.430 s |
| 其中 forward/backward | 77.616 s |
| 其中 Adam 更新 | 18.261 s |
| 其中 callback/log | 80.158 s |
| residual score 总时间 | 11.497 s |
| 最大单 lane 峰值 RSS | 346,492,928 bytes |
| 最大新增残差数值状态 | 11,912 bytes |
| 最大可训练参数 | 1,488 |
| GPU 峰值、实际部署包、增量传输 | N/A，未测量 |

新增参数小、避免 encoder 反传均属实，但不能抵消当前准确率大幅损失。也不能把移除详细日志后会更快当作方法有效性补救；日志成本已单列。既有 LocalMargin 仅得到小幅 OOF 收益、proxy 未通过且迭代成本大，结合本次结果，不应继续把原核之上的监督头优化视为必然有效的方向。

## 4. 结构性取舍：保留 LocalRidge 形式，先估计类内变化

建议停止新增千参数 CE 判决器，优先检查合法旧 support 是否含有能跨类别复用的类内变化方向。假设是：部分接收/噪声等扰动在不同旧类之间共享，从训练 support 估计并适度抑制这些方向，可能比重新学习 class logit 更适合小样本。**类内变化不等于已辨识域扰动**，它也包含硬件时变、调制、噪声和统计误差；没有 clean/source 配对就不能声称分离了真实信道。

一个具体、可反驳的结构方向如下，暂不冻结为候选：在原 interaction 特征空间做有界的类内协方差收缩，然后仍使用原 Gaussian 带宽、中心化、迹匹配和物理和 ridge=1 的分类结构。保留 b、a 与 b⊗a 三部分，即 `φ(x)=[b(x),a(x),vec(b(x)a(x)^T)]`，而非仅取 z_id。该空间维数 123,616，采用原核内积与低秩运算，不显式构造全部外积特征。

在 B 训练折的每个旧类求均值 μ_c。令 M 的列为旧类均值减全旧类均值，P_M 为其 span 的正交投影。对每个旧训练样本定义 `r_i=(I-P_M)(φ_i-μ_yi)`，令 `R=[r_1,…,r_n]`、`s=tr(RR^T)`。若 s>0，定义：

`W_B = (I + RR^T/s)^(-1)`；

`d_B²(x,y) = (φ(x)-φ(y))^T W_B (φ(x)-φ(y))`。

若 s=0，W_B=I。没有 optimizer、epoch、temperature 或可搜索 rank；所有非零方向使用同一确定性公式。因为 `0<=λ_j(RR^T)<=s`，有 `0.5 I <= W_B <= I`，从而 `0.5 d_0² <= d_B² <= d_0²`。该界限制距离扭曲，不保证准确率不下降。

投影 P_M 使旧类 interaction 均值差保持不变。必须准确解释其能力边界：这也意味着旧类最近均值分类器的排名不会因此改变；它保护的是 interaction 均值几何，**不**是 Gaussian 核类均值、LocalRidge 分数、局部邻域或最终准确率。LocalRidge 使用各个 support 点及非线性核，仍可能从收缩类内变化中获益，也可能受损。新类身份差可能恰好落在旧类类内方向中，旧均值保护并不保护未知新类，必须用完整注册 held 新类检验。

拟议 B 使用 W_B 作用下的距离走原 LocalRidge 规则；拟议 C 继承并冻结同一个 W_B，再用全部注册训练 support 估计该距离下的 tau、中心化、trace scale 与 ridge 头。C 并不冻结旧分类分数，不混接 B/C 不同尺度的 score，也不加固定旧类 bonus。类名重命名和列同步置换时公式等变，所有部署 query 仍逐样本面对全部注册类别。Nnew0 复用 B。

可以用 Woodbury 在旧训练样本空间计算：设 G=R^T R、v=R^T(φ(x)-φ(y))，则 `d_B²=d_0²-v^T(sI+G)^(-1)v`。矩阵维数 n_old≤120（完整 K20 的 6 个旧类），有效类内 rank≤n_old−6；不构造 123,616 维协方差。数值实现需复用原稳定的 interaction 差分，不能用近乎相同 Gram 对角相减破坏近重复样本；距离收缩项最多原距离的一半，使最后一步减法有明确的稳定性界。M 的低秩投影采用预先规定的 float64 相对秩阈值；明显 PSD/区间违反为技术失败，不靠自适应 jitter 补救。这些是实现前需精确规定并合成验证的数值细节，本次不写新 core。

该方向相对于 CE 头有可检验的结构优势：完整保留多分支 interaction、无直接 class-logit 自由度、距离变化有界、随无类内信息自动恢复原方法。但它没有效果保证。若类内方向几乎各向同性，λ_j/s 很小，变换会接近 I；不能因此加大强度追逐指标。若方向不跨类别复用、主要是有限样本噪声或与新类身份重叠，应否定当前解释，而不是继续换 rank。

## 5. 最小下一步与 K1 的不可辨识限制

下一步应先预登记一项有界、support-only 的机制诊断，而非修改推进条件或立即启动下一候选：保持同一 160 parent 身份，在每个已声明 OOF train fold 内估计上述量，报告类内谱的集中程度、留出旧类时方向/投影能量能否迁移到该类，以及变换前后 held-to-support 同类/异类邻域和完整旧新准确率。任何 held 只用于最终诊断，不回流估计 W_B，也不挑方向、强度或最优 fold。若只做类内统计可行性阶段，则不能声称已得到准确率收益；两类证据分开。本文件只提出诊断内容，不生成实际预登记或配置。

三点特别重要：

- 检查共享变化时，以“用其余旧类估计、描述留出旧类的类内变化”为主要证据，不以同一训练类的重建率自证泛化。类别数仅 6，必须保留所有留出类结果和不确定性，不当作独立大样本显著性检验。
- 由于旧均值差被保护，不能用变换后到旧类各均值的距离差改善作为收益论据；该差在理想算术下本来就不变。真正需要看的是点级邻域结构及原核分类规则下的持出变化。
- **train K1 的类内残差严格为零。** 当前缓存没有同一类的第二个物理样本，不能由不同类样本估计出可辨识类内方差。此结构方向在单 anchor proxy 应精确等于原 LocalRidge，真实 K1 仍无独立 held 证据。它不是解决所有 K 的完整方案；若沿用此前要求 proxy 严格正收益的两 screen，就不满足该要求，不能把精确回退写成正收益。这一事实不预先绑定未来所有候选的适用范围和评价约定，也不把 10/1/3 pp 理想方向改为硬门槛。

因此下一步应先判断是否确有可迁移的 target 类内结构；若有，只能先给出适用于物理 K≥2 的证据。要改善真实 K1，必须另外说明合法、可核验的先验信息来源及为何不会把类间差异误当域变化，不能凭空补方差、让 view 增加 K，或偷取未用的新 support/source 数据。后续方案须明确研发范围和相应证据口径，本复盘不改变原目标。

后续结构实验的直接对照保持原 LocalRidge，并分开报告 B−B0、注册前后旧类、新类、H、完整 K×新增类数以及实际成本。新状态需保存低秩旧 support 参考和小矩阵，可能超过 11,912 bytes 残差头；查询也增加投影/核计算，不能预先声称更便宜。先从公式计数、合成数值测试与资源测量开始，不宣称“无 optimizer 就没有适应成本”。

完整 support 与全部训练曲线共同支持停止 SequentialResidual 的小改路线，仍不能确定唯一失败原因。训练日志复核已完成；本文件没有训练、query 访问、方法热修改或新候选启动。
