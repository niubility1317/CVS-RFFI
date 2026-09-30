# MC-Residual8 完整训练机制发现（2026-09-30）

本轮 936 个有信息阶段全部发生参数更新，任务损失和总目标均下降。保持损失参与了梯度，但保持硬约束、方向保护 guard 和参数球投影在本轮没有实际限制任何试探。内层准确率的平均提升只有 0.20 至 0.40 个百分点，不能据此推断外层分类效果或旧类保持能力。训练确实执行；“更新发生”与“取得足够大的有效适应”需要分别判断。

## 证据范围与统计口径

run 为 `20260930-phase2-d92-mc-residual8-support-m2-r01`，release commit 为 `191d7df111a58aa8435473582db42902f9b8d3a0`。诊断状态为 `COMPLETE_MC_TRAINING_DIAGNOSTICS_DERIVED`。本文完整读取本轮派生的 4576 条阶段、17604 条曲线和 1944 条教师折记录，并读取各 CSV 的全部行；沿用独立 summary 已完成的绑定和坐标核验，没有重跑 collector、拟合或数学审查。没有读取 query、外层评分、历史结果或注册索引。

主要证据：[summary.json](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/summary.json)、[stages.jsonl](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/stages.jsonl)、[curves.jsonl](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/curves.jsonl)、[teachers.jsonl](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/teachers.jsonl)。下表的损失和准确率按有信息阶段等权平均；准确率为 `inner_training_accuracy` 的监督内层诊断，增量用百分点，不能当作独立 query 准确率。

160 个 parent 覆盖 40 个数值 K1 和 120 个物理 OOF parent。实际 B 阶段 1760 个，C_seq 和 C_reset_init 各 1408 个。3640 个 `PHYSICAL_K1` 阶段没有信息目标，其损失和内层准确率为 N/A；不能把它们算成优化失败或 0 损失。完整曲线包括 initial 936、gradient 3721、trial 4827、accepted_step 3544、final_cached 936、no_information 3640。accepted_step 重复展示已接受 trial，不能再次计费。

B 的 352 个物理绑定分别出现在 5 个新增类上下文中，共有 1408 个重复上下文。去重后 B 有信息绑定为 72 个、无信息绑定为 280 个；下面的实际阶段表保留 360 个有信息 B 上下文及其真实成本。重复 B 不能作为 360 份独立适应证据。定位：`summary.totals`、`deduplicated_B_statistics.by_mode_train_k` 和 [B_binding_groups.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/B_binding_groups.csv)。

## 初末目标与内层准确率

| 模式 | 有信息／有更新 | task 初→末 | keep 初→末 | proximal 初→末 | total 初→末 | 内层准确率初→末 | 提升（百分点） |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 360／360 | 0.367307→0.365348 | 0.005033→0.004905 | 0→0.001223 | 0.372340→0.371475 | 68.6639%→69.0603% | +0.3964 |
| C_seq | 288／288 | 0.441556→0.439337 | 0.024873→0.024179 | 0→0.001045 | 0.466429→0.464561 | 56.7609%→57.1339% | +0.3729 |
| C_reset_init | 288／288 | 0.442205→0.440648 | 0.024944→0.024375 | 0→0.000884 | 0.467149→0.465908 | 56.7421%→56.9452% | +0.2030 |

task 和 total 在全部 936 个信息阶段下降。keep 在 B 的 340 个阶段下降、20 个上升；在 C_seq 的 287 个下降、1 个上升；在 reset 的 288 个全部下降。keep 的少量上升仍在允许余量内。内层准确率提升／不变／下降分别为 B 的 60／300／0、C_seq 的 96／185／7、reset 的 71／204／13；合计 227／689／20。连续目标下降并不保证离散准确率上升。

总目标平均降幅分别为 0.000864、0.001868、0.001241。proximal 从 0 增长，抵消了部分 task 与 keep 的下降。C_seq 的平均 task、keep 和 total 下降均大于 reset，其起点也不同：它继承 B 的 U/V，reset 则从零 U、DCT V 初始化，但使用同一 B 教师。以上是实测训练差异，不能单凭模式均值断言顺序继承改善外层泛化。定位：`stages.initial/final/final_minus_initial`；分层全表见 [by_mode_train_k.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/by_mode_train_k.csv)。

### 各 train K 的差异

train K 为内折的真实训练样本数／类：3、4 对应 parent K=5，6、7 对应 parent K=10，13、14 对应 parent K=20。表中 `ΔL` 为 total 末减初，`ΔAcc` 为内层准确率百分点；这是不同 support 规模和问题上下文的描述，不能解释为单一 K 的因果效应。

| train K | 信息数 B／seq／reset | B ΔL／ΔAcc | seq ΔL／ΔAcc | reset ΔL／ΔAcc | 预算耗尽数 B／seq／reset |
|---:|---:|---:|---:|---:|---:|
| 3 | 80／64／64 | −0.000442／+0.6944 | −0.000725／+0.1630 | −0.000553／+0.0979 | 75／24／26 |
| 4 | 40／32／32 | −0.000564／0 | −0.001138／+0.2066 | −0.000824／+0.1578 | 35／5／12 |
| 6 | 40／32／32 | −0.001157／+0.3472 | −0.002083／+0.8527 | −0.001174／+0.3849 | 0／0／0 |
| 7 | 80／64／64 | −0.000821／+0.4464 | −0.002043／+0.5756 | −0.001338／+0.1832 | 0／0／0 |
| 13 | 80／64／64 | −0.001223／+0.3205 | −0.002858／+0.2894 | −0.001896／+0.2407 | 0／0／0 |
| 14 | 40／32／32 | −0.001087／+0.2976 | −0.002338／+0.2413 | −0.001600／+0.2410 | 0／0／0 |

## 教师 q 与保持信号

B_prepare 有 1080 个教师折记录，C_prepare 有 864 个。C 教师准备由 seq/reset 共享，不能翻倍。计数是实际上下文中的旧类 held-support 出现次数，包含 B 重复上下文，不是独立物理样本数。

| 教师准备 | held-support 出现数 | q 均值／最大值 | 正 margin，q>0 | q=0 | 0<q<0.25 | 0.25≤q<0.5 | 0.5≤q<0.75 | 0.75≤q<1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B_prepare | 16800 | 0.289300／0.904471 | 12000（71.43%） | 4800（28.57%） | 3820 | 3290 | 3930 | 960 |
| C_prepare | 13440 | 0.291568／0.906212 | 9648（71.79%） | 3792（28.21%） | 3056 | 2640 | 3168 | 784 |

两类教师都没有 q=1 的条目。train K 从 3 至 14，B 的 q 均值依次为 0.1495、0.1651、0.2447、0.2480、0.3377、0.3553；C 为 0.1500、0.1662、0.2463、0.2500、0.3406、0.3580。低 K 教师的正 margin 较弱，且全部教师约 28% 的 q 被截断为 0。这限制了“教师为每个旧类样本提供正保持目标”的解释。

q=0 的一般含义是 margin 非正，包括负 margin 和并列，不能直接等同于精确错误数。本轮 q=0 条目都有唯一 score winner，派生的保证错误下界分别为 4800 和 3792，恰好覆盖所有 q=0 条目；并列不确定条目为 0。未连接 held truth 的 `teacher_exact_wrong_count` 仍为 N/A。q=0 不提供正 margin 下限，但其 deficit 项仍可惩罚学生产生负旧类 margin，不能称为完全没有保持约束。定位：[teachers.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/teachers.csv) 的 q 分布、`teacher_guaranteed_wrong_count` 与并列计数。

## 梯度、保护与接受条件

`total_vs_keep` 比较总梯度与 keep 梯度；`task_vs_keep` 使用派生的 `g_task=g_total−g_keep−(theta−anchor)/N`，排除了 keep 和物理 proximal 梯度。两种冲突不能混称。所有相应余弦均有定义，没有用 N/A 当作无冲突。

| 模式 | 实际反向 | total_vs_keep 冲突 | task_vs_keep 冲突 | 两种余弦均值 total／task | 初次→末次总梯度范数均值 | 接受／拒绝 trial |
|---|---:|---:|---:|---:|---:|---:|
| B | 1425 | 210 | 125 | 0.3594／0.4493 | 0.005229→0.001428 | 1315／755 |
| C_seq | 1148 | 37 | 80 | 0.6259／0.4476 | 0.005489→0.003555 | 1119／231 |
| C_reset_init | 1148 | 48 | 59 | 0.6090／0.4539 | 0.004213→0.002472 | 1110／297 |

负内积冲突确实存在，但 guard 的激活次数为 0；存在冲突并不意味着超过带 slack 的方向保护界。球投影改变 trial 位移的次数为 0，最终参数均未触及两颗参数球边界。所有 4827 个 trial 都通过 keep 条件，接受的 3544 个还通过 Armijo 和总目标非增条件；1283 个拒绝全部为 `ARMIJO_AND_OBJECTIVE_INCREASE`，同时未通过后两项。accepted keep violation、accepted objective increase 和 rejected keep violation 均为 0。最终 keep 余量最小值分别为 B 0.014611、seq 0.014808、reset 0.014773，所有 trial/final 的 keep excess 为 0。

因此，本轮的直接接受瓶颈是目标下降条件，保持硬约束并未成为瓶颈；这不表示 keep 无效，因为其非零损失和梯度持续进入总目标。也不能从“约束未触发”推出外层旧类不会下降。定位：`curves.kind=gradient/trial/final_cached` 的冲突、guard、球投影、三 pass 标记及 keep remaining/excess；`stages.rejection_reasons`。

| 模式 | MAX_ITERATIONS | ARMIJO_OR_KEEP_BUDGET_EXHAUSTED | 接受 4／3／2 次更新的阶段数 |
|---|---:|---:|---:|
| B | 250 | 110 | 250／95／15 |
| C_seq | 259 | 29 | 259／25／4 |
| C_reset_init | 250 | 38 | 250／34／4 |

177 个耗尽阶段全部来自 train K=3、4；名称包含 keep，但本轮没有 keep 拒绝，不能把停止原因归因于保持风险。759 个阶段达到固定 4 次迭代上限。没有 ZERO_GRADIENT、ZERO_GUARDED_DIRECTION 或 ZERO_PROJECTED_STEP 停止，也没有零更新信息阶段。末次 task 梯度范数均值仍为 B 0.007060、seq 0.006048、reset 0.004721；有限预算结束不能称为已收敛。B/reset 首次 V 梯度为 0，seq 没有该情形，符合前两者从零 U 初始化的日志表现，并非 V 在全程不训练。

## 参数变化与几何解释边界

| 模式 | 最终 ‖U‖ 均值（范围） | 最终 ‖V−V_DCT‖ 均值（范围） | 相对本阶段 anchor 的总位移均值 | 每阶段改变的 U／V 坐标数均值 |
|---|---:|---:|---:|---:|
| B | 0.2811（0.0957 至 0.4074） | 0.1603（0.0136 至 0.2555） | 0.3252 | 5421.9／5421.9 |
| C_seq | 0.5456（0.1878 至 0.7590） | 0.3554（0.0560 至 0.5762） | 0.4231 | 5456.9／5456.9 |
| C_reset_init | 0.3576（0.1062 至 0.4480） | 0.1726（0.0249 至 0.2551） | 0.3979 | 5456.9／5456.9 |

每个矩阵有 5888 个坐标，变化覆盖多数坐标；这不是“只有少数坐标更新”。完整轨迹中的最大 ‖U‖ 为 0.7590，最大 ‖V−V_DCT‖ 为 0.5762，低于两个冻结的 1.0 球界。seq 的 V 相对 DCT 距离包含继承 B 的变化，其本阶段 V-anchor 位移均值为 0.2155；不能把 0.3554 全部记为 C 更新。定位：`stages.final`、`U_changed_coordinates/V_changed_coordinates` 和完整曲线的范数、anchor distance、DCT distance。

**能证明的是**参数在界内进行了广泛而非零的更新，seq 累积位移较大。当前派生数据没有输入逐样本角度、适配前后特征距离或核变化的完整测量，这些项为 N/A。冻结 κ=0.25 和参数球界是设计约束，不是实测几何变化；不能仅凭参数范数、参数数量或残差形式断言“特征几何只发生小扰动”，也不能反向断言已发生大几何改变。

## 实际资源与完整归档成本

全部计数保留实际重复上下文。B 缓存准备的 1080 个 initial inner head 已计入 inner head 总数，`initial_inner_*` 是解释分项，不能重复加；C 的 864 个 teacher head 只计一次，seq/reset 共用。

| 实际工作 | 数量 |
|---|---:|
| baseline head／factorization | 各 3168 |
| inner head／factorization（含 B 预付初始头） | 各 17289 |
| final head／factorization | 各 936 |
| teacher head／factorization | 各 864 |
| 全部 head／factorization | 各 22257 |
| task／keep 伴随三角求解 | 各 22326，合计 44652 |
| 反向／新逻辑目标／MC forward | 3721／5763／19089 |
| prepared 距离引擎调用 | 11952 |
| diagnostic fit | 0 |

硬件为 Linux x86_64，报告 CPU count=96，各 lane 的 OMP、OpenBLAS、MKL 线程设置均为 2；float64、未使用 GPU。运行实际 wall time 为 2904.920 s，lane wall time 之和为 5739.724 s。准备累计 240.284 s，candidate fit 累计 4365.680 s，candidate score 累计 391.928 s。task/keep 内层伴随分别累计 1085.694／1082.724 s；keep 伴随不是零成本。上述准备、fit、目标、archive 时间存在包含关系，不能相加重复计费；lane/阶段之和表示工作量，不是并发运行的实际耗时。

| train K | 有信息阶段平均 fit 秒 B／seq／reset |
|---:|---:|
| 3 | 0.798／1.779／1.724 |
| 4 | 0.954／2.323／2.277 |
| 6 | 1.145／3.853／3.667 |
| 7 | 1.332／4.812／4.566 |
| 13 | 2.591／13.086／12.260 |
| 14 | 2.850／14.879／13.940 |

936 个信息阶段平均 fit 秒分别为 B 1.599、seq 6.712、reset 6.332。K 增大时平均 fit 时间明显上升；同时 C 的注册类数和 support 规模也变化，不能把模式差异全部归因于优化器。单阶段 score time 未提供，为 N/A；上面的 score 累计来自已核实的全局资源字段。

最大 lane RSS 为 768847872 B，含 Python、共享缓存和记录开销，不能称为 adapter 专属峰值。最大准备数值状态 41883808 B、初始 forward cache 5253312 B、临时距离状态 2119936 B；最大拟合常驻数值状态 6608384 B，其中 adapter 为 94208 B（11776 个 float64 参数），还含 head、原始／适配 support 和继承状态。常驻范围不含冻结 Phase1 交付、共享特征归档、teacher/training cache、Python 和序列化。外置向量记录的内存字段为 0，不表示不存在教师或训练向量。峰值显存、部署包、增量传输字节和独立 kernel 时间均为 N/A。

| NPZ phase | 文件数 | 实际文件字节 | 归档累计秒 |
|---|---:|---:|---:|
| B_prepare | 1080 | 1599470 | 0.572 |
| B_MC | 7015 | 732664910 | 30.263 |
| C_prepare | 864 | 1295200 | 0.470 |
| C_MC_seq | 5314 | 694625133 | 27.061 |
| C_reset_init | 5371 | 570125630 | 23.404 |
| 合计 | 19644 | 2000310343 | 81.771 |

约 2.00 GB 为保留全部坐标的实际 NPZ 归档量，不是推理常驻状态，也不是方法传输量。本文保留原始引用，没有把 11776 维向量嵌入报告。资源定位：`summary.coverage/resources/state_archives` 和 [archive_by_phase.csv](E:/type10-7/automation_reports/CV-SincNet/20260930-phase2-d92-mc-residual8-support-m2-r01/results/training_diagnostics/archive_by_phase.csv)。实际参数数目较少仍需执行大量内头、反向和两套伴随；本轮没有配套星载硬件或部署传输测量，不能宣称已节省星载算力。

## 已证实的不足与不能推断的结果

本轮没有从这些完整诊断发现无更新信息阶段、已接受风险违反或已接受总目标上升。已证实的机制局限是：固定预算结束时仍有非零梯度，低 K 的下降条件拒绝集中，teacher 正 margin 覆盖不完整，总目标和内层准确率的平均改善幅度有限，且保持硬约束本轮未形成实际制约。以上支持分析“信号强度和有限预算下的有效变化”，不等同于诊断训练执行故障。

源域验证、逐样本预测变化、外层旧／新类效果及泛化、真实输入几何变化、星载收益均不在本次证据中，为 N/A。完整合法 support 诊断可以用于后续机制解释；本文没有依据部分结果改变当前候选、选择参数或提出下一候选。
