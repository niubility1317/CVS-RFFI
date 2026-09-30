# AJLR 唯一预发布独立 P0/P1 审查（2026-10-01）

结论：**NO_UNRESOLVED_P0_P1**。本次唯一 QUERY-BLIND 审查完成；发现的三项 P1 已修复并定点核验，未发现 P0。该结论只涵盖数学、科学权限、输入与执行正确性，不评价性能目标是否达标。

## 范围与科学权限

审查范围为 `d92_anchor_joint_local_ridge.py`、冻结配置、evaluate/summarize、prepare/run/preflight/publish/analyze、新 spec/cohorts 和对应合成测试。只读取设计第 2 章起的数学与边界，没有读取第 1 章 current outer 性能、历史 query/ABC、global registry 或 handoff。审查者只拥有本报告，不改其他工作者文件，不运行数值 import、训练、pytest、Conda、SSH 或 Git；静态 AST/UTF-8 文本读取只使用 `F:/App/miniconda3/python.exe -X utf8`。

唯一候选为 R_AJLR_seq，控制为 R0。允许当前 target support 与标签上的监督训练；Phase1 固定。inner-held 标签参与 CE，是训练监督，不能解释为独立验证。outer-held 只在阶段拟合完成后评分，不进入 H/W、prior、核统计、步长选择或接受判断。代码没有 source 样本、地面逐样本特征、query fit、query 反馈、旧/新真值路由或全局配额分配。

新 run 为 `20261001-phase2-d92-anchor-joint-support-m2-r01`，发布与输出路径独立，root 是唯一 launch owner。prepare 使用独占写入，run/output/row 使用 `exist_ok=False`，分析先要求完整声明的 4 rows/160 parents；失败保留已完成产物，不自动重跑或干预其他健康 lane。读取的预登记与 preflight 不是已启动证据。

## 已发现的 P1 与修正

| P1 | 触发条件与影响 | 修正状态 |
|---|---|---|
| 参考 pair 计数口径错误 | 非零 U 的正常 inner head 被汇总当作计数错误拒绝 | RESOLVED |
| 新 spec 仍保留旧 path 数量 | 汇总 validator 在读取产物前拒绝合法 AJLR spec | RESOLVED |
| NumPy 与原生 bool 的身份比较 | 数学接受条件正确的 trial 被汇总拒绝 | RESOLVED |

1. 原 `tools/summarize_d92_anchor_joint_probe.py:336` 将参考 pair 数写为 `m*(n+h)`；核心 `code/cvsrffi/d92_anchor_joint_local_ridge.py:249` 至 `:252` 实际按无对角的对称 train pair 与 cross pair 计数：`m*(m-1)//2+m*(n-m)+h*m`。例如 B 内折 `n=m=4,h=2`，核心为 14，旧汇总为 24。正常非零 U 的 objective 必被拒绝，与性能无关。汇总已改成实际口径；完整正常 fixture 对 B 的 `m=n` 和 C 的 `m<n` 都增加断言。参考计数是 raw 距离工作的子集，不重复相加。

2. 原 `tools/prepare_d92_anchor_joint_probe.py:58` 至 `:60` 只更新 exact counts，遗留 `expected_sequence_paths=1760`；新 K1 完整头新增 40 paths 后实际固定值为 1800。新 spec 静态读回确为 1760，而 entry `validate_spec()` 明确要求 1800，独立 summarize 因此不能执行。修正后 prepare 明确写入 1800，run validator 检查该字段，兼容旧 metadata validator 的临时 view 显式设回 1760；新 spec 读回为 1800，测试同时断言正确值与拒绝回退值。此修正未改变选择矩阵或预算。

3. 正常 fixture 实际在 `verify_candidate():477` 报 `AJLR exact two-condition acceptance mismatch`。旧 `:475` 用 NumPy 的 eps 计算 tol，比较可得到 `numpy.bool_`；随后将 JSON 原生 `bool` 与之做 `is` 比较，值相等也会失败。静态定位与实际错误相符，不是 Armijo 数学失败。修正后的 `:475` 将 eps 转为 float，`:476` 将两项比较明确转为 bool，继续使用相同容差和 Armijo/实际目标非增条件，没有放宽接受规则。

上述第 1、3 项修正后，同一正常完整 fixture 通过全部 prior/reference/CE/closed solve/outer score/workload 链；再用现有 10 种变造 fixture 确认拒绝，避免原正常链必败导致 `assertRaises` 伪通过。源码、正常与负测证据均已读回。

另收到按 `id(cf)` 查找旧 head NPZ 的生命周期疑点，仅核对当前核心 `:664` 至 `:681`：归档引用保存在缓存对象 `cf['_head_ref']`，没有永久对象 ID 映射。新缓存对象不会继承旧字段，该疑点不适用于当前版本，未列新增 P1，也未要求重复修改或测试。

## 数学及实施核对

- **实际函数 prior 与闭式头。** B prior 为零；C final prior 是当前实际 B 的完整分类函数，映射到旧类列，新列零 padding。C inner prior 在固定当前 U_B 后只拟合本折 old inner-train；nuisance 也来自这个集合，不借完整 B head 给 inner-held 形成 prior。最终 C 使用真实 final B，C 入口核对 B 类型、mode、classes、上下文、旧物理 ID、label 和 raw feature 完全相同（核心 `:409` 至 `:419`）。N_new=0 直接复用 B。
- **无 intercept 的真实目标。** `Y=onehot−1/C` 是类别维中心化，`E=Y−M` 没有再次做样本中心化。规范解是 `(K+I)α=E`，score 为固定 prior 加 kernel residual；没有隐藏 bias、温度或旧/新组偏置。设计如实说明旧参考 residual 均值为零的限制：平衡旧参考半平方误差下界为 `M*C_new/(2*C_old*C_total)`。这不保证旧类逐样本 score 或准确率不变，也不是精确顺序 Bayes。
- **移动参考与完整伴随。** 固定的是物理测度 q 及原始旧类 τ/γ，参考 embedding 每次随当前 U 重算。`K=γPq R Pqᵀ`、`L=γ(Q−1qᵀR)Pqᵀ` 与核心 `_center_train/_center_cross` 相符；`_center_vjp()` 包含参考项 `−γq(1ᵀ barL Pq)`，train/held 两端距离 VJP 都进入当前 U 梯度。prior 在 C 梯度中固定，τ/γ/q 的无梯度是设计定义，不是漏掉动态统计。
- **CE pooled class RMS。** 核心 `:522` 至 `:548` 先跨折按物理记录累计 CE sums/counts，形成各类均值和 RMS，再给每条类别 c 记录权重 `mean_c/(C*RMS*count_c)`。closed-head 伴随为 `barL=Gαᵀ`、`barK=−sym((K+I)⁻ᵀLᵀGαᵀ)`；复用当前 Cholesky 的两次三角求解。最后 `g_Z=G_U W+Z`，prox 为 `0.5||Z||²`，无重复除 N，RMS=0 不除零。
- **坐标、退化及核。** 保留完整 `φ=(b,a,b⊗a)`，123616 维隐式几何。H/√N 的 SVD 只定义当前 outer-train 优化坐标，固定 U 时不改变 head；Z=0 精确复制 anchor，未训练方向保留。正带宽的联合 Gaussian 与旧测度中心化在精确算术下 PSD，K+I 的最小特征值至少为 1。K1/rank0 不更新 adapter，但仍拟合合法完整闭式头。τ0=0 使用原始完整特征等价核、零 adapter 梯度；缺失旧尺度使用零 residual kernel 的规范 α=Y−M，不回退新集合尺度、历史方法、地面摘要或 jitter。
- **决定规则与优化。** 全部注册类逐样本统一 argmax，精确并列按 class ID 字典序。最多 4 次 normalized-gradient 更新，每轮 12 次固定 halving，接受第一个同时满足 Armijo 与实际目标非增的 trial；不用准确率选步，不挑最佳 trial。拒绝 cache 不覆盖最后接受状态，预算耗尽不是收敛。数学保证只适用于所述舍入容差和目标条件，不是 query 泛化或逐类改进保证。
- **归档、日志与实际成本。** NPZ 保留 Z/U/W、head、q、prior、E、labels、scores、梯度/方向和全部 trial；独占文件、无 pickle、严格 finite 原生 JSON，summary 重新核对 prior、闭式解、全类 score、CE、继承、接受及完整事件/文件。详细文本、完整 JSONL、紧凑 JSONL/CSV 与配置启动日志均落盘。baseline primal/EDF、prior、student、CE adjoint 和外层 prior/residual 推理分别计数；最大 5888 梯度参数不等于总决策参数。U+固定 V0 是 94208 数值字节，C 还保留实际 B 与两个 head；常驻、部署数值、训练 cache、证据归档和压缩文件口径分开，未测部署包、传输、GPU 或星载成本写 N/A。

## 输入证据与验证边界

新 spec 仅复用指定 source-only scratch final200 的冻结 received-support feature cache，选择固定 metadata seeds 和相同 capsule/physical split。`load_support()` 保留 source contract EXACT_MATCH、epoch200、empty checkpoint inheritance、target_access_before_freeze=false，以及 producer native_eval/unchanged parameters/buffers、无 source/query/truth 访问、received original-only feature 契约与物理 ID/role 检查。该身份契约检查不新增数据重验证。

已直接读取新 run 的 `evidence/anchor_joint_preflight_1790789568781255000.json`：四个 cache bindings 为 VERIFIED，每个 row 40 selected splits，practical residual/post_sync/noeq/25MHz；run、release 和 archive 均尚不存在，query/source_sample_access=false。发布脚本只携带纯代码依赖与配置，不携带历史适配状态或 checkpoint，不占 GPU、不重复 launch。

主 Agent 集中执行并提供以下证据，审查者只读 stdout/stderr，没有再次运行测试：核心 22 PASS（4.66 s，`pytest_utf8_1790789398064171500`）；资源定义影响的 3 项 PASS（2.10 s，`1790789671575009100`）；启动器 22 PASS（0.95 s，`1790789624029909800`）；path 字段修正影响的 6 项 PASS（0.32 s，`1790789874716344800`）。这些文件在 `E:/type10-7/.codex_tmp/`，通过调用 stderr 均为空。12 个相关 Python 文件 AST 静态解析通过，未 import 数值实现。

正常汇总 fixture 在 `pytest_utf8_1790789897088187900` 的布尔校验失败证据已保留并读回。修复后的正常完整链为 `pytest_utf8_1790790062262768900`，1 PASS，23.74 s；10 种变造拒绝 fixture 为 `pytest_utf8_1790790155613702700`，1 PASS，39.06 s，均已直接读取 stdout 和空 stderr。未重复核心检查或正常 fixture。

合成验证说明公式、权限、隔离、归档和计数正确性，不证明实际性能、理想 10/1/3 目标达标或计算节省。本次没有利用性能选择方法，没有增加候选、参数网格、审批、哈希链、重复数据重验或完整 125 早期门槛。
