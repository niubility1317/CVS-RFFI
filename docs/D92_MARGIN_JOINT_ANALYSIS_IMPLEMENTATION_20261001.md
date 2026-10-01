# Margin 联合方法的独立档案分析实现

状态：只新增摘要器、两组人工合成测试和本文；本 Agent 未执行数值测试，未读取真实产物或启动实验。根 Agent 的合成验证和待复测范围见末段。冻结的训练 core、QP 组件、入口和已有分析器未修改。本实现不选择参数、预算或 checkpoint，不宣称准确率提升。

## 接口和依赖

- 摘要器：`tools/summarize_d92_margin_joint_probe.py`。
- schema：`d92_margin_joint_support_summary_v1`；成功状态：`COMPLETE_MARGIN_JOINT_PROBE_VERIFIED`。
- CLI：`--spec`、`--output` 必需，`--run-root` 可选。
- Python：`summarize(*, spec, run_root=None, output)`。
- 档案链：`StateResolver`、`verify_head`、`verify_preparation`、`verify_objective`、`verify_candidate`、`verify_record`、`verify_stage_stream`。
- 独立数学接口：`verify_margin_head(data, audit=None, analysis_work=None)`、`margin_head_vjp(data, L, G, analysis_work=None, certificate=None)`。
- 失败证据：`verify_failed_fit(audit, arrays, analysis_work=None)`。它只证明可取得的末次费用和当前失败快照，不返回成功头、完整运行或普通 Jacobian 证书。

直接项目依赖为 `evaluate_d92_margin_joint_probe`、`run_d92_margin_joint_probe`、`summarize_d92_affine_joint_probe`、`summarize_d92_branch_support_probe`、`summarize_d92_registration_diagnostic`、`d92_affine_analysis_math` 和 `cvsrffi.d92_branch_local_ridge._distances`。标准库、NumPy 和由已有独立 adapter 数学使用的 SciPy 保持原依赖。

入口用于读取 schema、计数清单、物理选择和既有纯汇总函数；训练方法可以被导入为其依赖，但摘要器不调用生产 prepare/fit/objective/forward/predict/adjoint 作为 oracle。合成回归测试生成档案后，将这些生产数值函数替换成抛错函数，再运行独立验证。没有 Conditional 核、Conditional 摘要或固定双 factor/三谱公式依赖。

## 独立数学核验

摘要器从已存五块 raw、original/adapted 几何、U、固定尺度、类别和物理索引重建完整 Gaussian 或 exact-equivalence Gram。所有 old-old、old-new、new-new 和 held-to-train 元素参与检查，不物化 `n×123616` interaction 特征。距离与 adapter 反向使用已核实的独立分支数学表达式。

B 使用已验证独立 Affine 摘要数学，核对自由截距、中心化 gauge、原尺度、完整 Schur 状态和 `[C+1]` RHS。

C 不优化或重新选择 active set。它验证原归档中的完整 primal 和 dual：

`R=onehot(y)−1/C−M`，`V_old=Dᵀμ`，`(K+I)α+1b=R+V`，`1ᵀα=0`。

C 归档的 `s0` 保留纯 QP 的完整物理约束基准 slack 向量，shape 为 `(m*(C−1),)`；几何尺度保存为独立 `reference_s0`，必须是恰含一个有限值的 0D scalar。分析器分别重建并严格核验两者，不能把约束向量转换成 scalar 或将缺失向量视为正常旧 ABI。B 的 Affine 几何尺度继续使用原 `s0` scalar；纯 QP Schur `s` 继续要求 shape 为 `()`。

所有物理旧行的 true-vs-every-other 约束使用原 `min_j(M_iy−M_ij)`，保留正、零、负值。核验非负 μ、全部 slack、互补性、RKHS 正则、primal/dual 目标和 gap 恒等式。自由截距不惩罚，scores 必须为 `M+Kα+b` 或 `M_H+Lα+b`。raw K 单独进行 PSD 谱核验；奇异 K 可保留唯一函数，但 canonical α 仍须满足保存的正规方程。不能把退化系数的非唯一性当作任意改档案的理由。

独立线性求解使用一般 saddle 矩阵 `[[K+I,1],[1ᵀ,0]]`，合并 target 与物理旧选择 RHS，得到 affine 响应。active 小系统由该响应和实际全列差分构建，仅使用归档 working set；不分配完整 dual Hessian，也不枚举生产约束。

完整伴随包含 `g_b=Σ_h G_h`、自由截距响应及 active multiplier 响应。使用一般 saddle/active 求解独立重建 `T_G/t_G/eta/W_eta`、对称 barK 和一般 barL，再逐端点进入距离和 adapter VJP，最后检验 `g_Z=g_U@W`。C 不经过 center VJP，不添加 `+Z`；B 保留完整 reference 梯度。保存但未新增收费的 cached companion 仍可重验，不能算成新的训练 solve。

有 tight 行不属于独立严格互补 working set 时，普通 Jacobian 不受支持。分析器拒绝用伪逆、jitter、无约束伴随或零梯度替代。tau0、zero kernel、K1、rank0 和没有 inner-held 的分支沿用方法定义；不声称这些位置发生普通 adapter-kernel 梯度更新。

## 物理继承和独立 held

所有归档 ref 的 namespace、stage 和 physical IDs 必须属于当次 run/row/split/scope/fold。C 初始 U 和 `prior_B_*` 与当次 B 完整状态一致，旧 raw 特征及类列映射不变。内层 prior 头只使用对应 old inner-train；固定 M 可以预测合法 inner-held，但不使用其标签拟合头或尺度。每个 prepared、trial、accepted 和 final 状态都绑定同一实际预算和 anchor。

先从 final 数值档案逐单样本、全注册列重建 outer support-held scores，再与固定预测证据比较；完成后才连接合法 support-held 标签。指标按 parent 计算，再做 parent 平均；proxy anchors 先在同一 parent 内平均。H、注册下降和新旧差不能由总体均值重新拼造。真实 K1 没有独立 held，保持 N/A。new0 检查复用结构和 B/C 同一固定函数，不新增 C head。

完整 `K×new_count`、cohort、receiver/scenario 和资源分层保留既有输出形式。A 及 B−A 保持 N/A；用目标旧 support 拟合的 R0/B0 不能冒充地面 A。

## 真实费用和两个 MAX

前向和伴随的 QP ledger 分开。每次日志的 system、dimension、RHS columns、transpose 和完成状态用于重算 calls、RHS 列/元素和 `n²×RHS` 维度工作代理。成功前向还核对实际 transition/working set 对应的 factor 与 solve 系统、所有扫描、compact snapshot 和单次 raw-kernel 谱诊断。拒绝 trial、prior、final、缓存免收费范围和失败末次工作都保留各自所有权。

不能用固定次数掩盖动态 active-set 工作。`factorization_attempts`、`factorizations_completed`、condition-estimation 和实际 RHS 单独核验。两个峰值 `margin_qp_peak_factor_buffer_bytes`、`margin_qp_peak_explicit_solve_temporary_bytes` 从实际系统重建后按 MAX 聚合；其余 `COUNTERS` 按 SUM。cached gradient audit 可能保留其关联前向 factor 的高水位，这仍是 MAX，不是新 factor 收费。

`resources.actual_counters` 保留完整实际账。顶层 `qp_resources` 与逐 lane actual startup/config/spec 精确匹配，`algorithm` 为实际冻结结构，`release_commit` 来自 runtime supervisor 的 startup/complete 一致值。摘要器不选择资源数值。

分析器自身的 general solves、RHS、dense-work 代理和 PSD 谱检查记录在 `independent_analysis_work`；档案缓存 load/hit/eviction/字节记录在 `cache`。它们不进入训练计数。`analysis_wall_seconds` 只覆盖输出写入前的输入绑定、档案和数学核验；不冒充训练时间。

resident、minimum deployment、磁盘 NPZ、传输和进程峰值是不同口径。NPZ 副本不能重建原对象 alias 布局，所以摘要器核验保存口径和数组可行下界，不把 NPZ 逻辑字节伪装成独立测得的物理 resident。未知设备、GPU/能耗、序列化部署和增量传输仍为 N/A。

## 失败、缓存和验证范围

完整摘要先核对 declared matrix、supervisor/lane completion 和清单，再读取评分档案。INCOMPLETE 档案在数值读取前拒绝，保留原失败产物。失败证据 API 明示 `complete_run=False`、`optimality_certified=False`、`cumulative_work_verified=False`；不把缺失 solver 工作填成零。能取得 compact rho/V/P_old/slack 时，重建的是同一失败时刻的快照，不是上一 working solve。

档案缓存继承 64 MiB/64 entries 的有界 LRU，仅缓存不可变数值数组，所有数学检查照常执行。超过预算的单个 NPZ 不进入缓存；该边界约束缓存持有内存，不声称限制调用者暂存矩阵、解压瞬时内存或进程峰值。file-size/mtime、metadata、完整 shape/dtype/summary 和引用清单仍检查，没有新增签名或 receipt 链。

Margin core 的 recorder 返回 callback 的完整身份字段，但将每个数组的 metadata 写成 `shape/dtype/nbytes` 三字段。StateArchive manifest 还保存实测 `all_finite/nonfinite_count`。Margin resolver 只允许完整 manifest ref 或上述精确三字段 view；两者都绑定回同一个清单项。所有其他字段、数组名集合和三字段值须一致，成功清单须声明 `failed_numeric_state=False`、`all_finite=True`、`nonfinite_count=0`，然后继续原有路径、NPZ 内容、dtype、shape、字节、summary 和缓存校验。未知字段、身份缺省和其他 metadata 省略均拒绝。

合成测试包括真正 production callback→NPZ→StateResolver→独立 summary、禁用生产 oracle、继承/CE/完整截距与梯度篡改拒绝、全部约束、动态 RHS/峰值/费用篡改、缓存开关等价、K1/new0、tau0/zero kernel、实际失败部分档案和未知费用拒绝。另一组小矩阵测试使用独立 feature-primal active-set 枚举生成有限合成 oracle，检验 primal/dual、完整 K/L 差分、非零 g_b/乘子响应、singular K、正/零/负 margin、weak/singular active 拒绝及每个关键字段篡改。

根 Agent 首轮 summary/math/publish/prepare 合成验证为 54 passed、9 failed；9 项失败来自合法 ref 与 manifest metadata 的形状差异。ref 修复后为 38 passed、6 failed；剩余失败定位为 C 几何尺度覆盖 QP 向量。修复生产归档并加入新 ABI 回归后，相关 core/summary 两文件复测 70 passed、1 failed；该项失败是 root 新增回归直接把精简 final_cache 传给需要 problem 的归档函数。按实际归档调用绑定 problem 后，该项单独复测 1 passed。独立 summary 全部 49 项在上述两文件复测中通过；实际 production callback、QP 向量及 scalar/vector 篡改检查均通过。原失败记录保留，未核验任何真实 Margin 运行、性能、Ground A 配对或部署。
