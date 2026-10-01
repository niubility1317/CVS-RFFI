# D92 ConditionalJoint 上层实现

状态：独立结构实现，尚未完成端到端预登记、实验、scorer 整合或部署。schema 为 `d92_conditional_joint_local_ridge_v1`，method 为 `D92-ConditionalJointLocalRidge-v1`。不修改既有 Affine 或正在运行 release，不声明准确率提升或星载省计算。

本模块实现 [冻结数学设计](D92_POST_AFFINE_JOINT_MATH_DESIGN_20261001.md) 的 B/C 路径。B 使用现有 Affine 纯解析 solve/adjoint；C 接入独立 `d92_conditional_affine_kernel.py`，不自行替换其数学。几何、DCT8 字典、白化、完整距离及 adapter VJP 复用既有纯 helper，不改其全局配置。

## 公开接口与状态

模块：`code/cvsrffi/d92_conditional_joint_local_ridge.py`。

- `prepare_conditional_joint_training(*, z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes,inherited=None,context=None,log_callback=None,state_callback=None)`。
- `evaluate_conditional_joint_objective(prepared,Z,anchor_U=None,*,gradient=True,forward_cache=None)` 返回 `(RMSCE,g_Z,audit,ObjectiveCache)`。
- `fit_conditional_joint_local_ridge(prepared,*,mode='B'|'C_seq',baseline_state=None,log_callback=None,state_callback=None)`。
- `predict_conditional_joint_local_ridge(state,**features)`。
- `ConditionalJointState.score/score_with_audit/predict/audit_dict/to_arrays/state_records`；`ConditionalJointTraining.audit_dict()` 提供同样的安全 JSON metadata 接口，`ConditionalJointTraining` 和 `ConditionalJointProblem` 为独立类型。

五个 feature block 的维度是 160、96、160、160、160。公开 preparation 使用现有合法支持集协议：整数标签、每类相同物理 K、物理 IDs 和全注册类列表；原始 block 与标签按物理 ID 排序，class 按物理类字符串的稳定字典序规范化。推理每个样本面对所有已注册类，采用同一 argmax，空输入返回正确形状的空分数。

C 只接受本方法同 run_id/row_id/scope 以及一致 split_id/fold 的当次 B。old IDs、labels、原始五块与实际 B 一致；trial 只是 trace 坐标，不新增授权条件。C 起点为实际 U_B，prior 的 U_B 和旧头固定，新增列为零。inner C prior 只在该 fold 旧 inner-train 上求头，使用实际 U_B；final C prior 是实际 full-support B。`baseline_state` 只保留入口兼容签名，不作为 prior 或初始化来源。

## 已实现数学

`U=anchor_U+ZWᵀ`，H/W/SVD 来自当前合法 support 字典。固定 DCT8、κ=1/4，原始/适配完整 interaction 距离各占 1/2，τ/γ/s₀ 仅由原始旧 R0 几何确定。未物化 123616 维 interaction。

B 使用含自由类截距的解析 Affine LocalRidge。C 正核使用正确的 `γ exp(−d/τ)` raw PSD Gram，而不是 `γ expm1(−d/τ)`。C 的 old/new/held 核块为 A/B/D/F/E；pure kernel 返回含自由常数 rank-one 项的 K_perp/L_perp 及 α/β/v。部署 residual 为 `k_new α−k_old β+v`，所以全部旧 support 上的所有 C 列 residual 为零。

完整 C 伴随包含 A/B/D/F/E 端点、T/Lambda/X_alpha/X_T/g_b，再接完整距离及 adapter VJP。拼接对称全 train Gram 时将 B 的总上游一半放在每个交叉块，避免双计。old 代表点梯度不 detach；精确重复输入在全局 U 方向下共同移动，所以无损代表点约束求导与未重复约束函数相同。

C 的 pointwise constraint 检查覆盖实际 `old_indices` 的全部注册列，旧参考均值审计也使用这些物理旧点。任意 q 的 gauge 变化不能变成 C 的额外零均值约束；B 的 centered Affine 审计仍使用 q。外部归档 shape 通过 `audit_dict()` 的真实 JSON 序列化检查：JSON 标量 shape 是 `[]`，NPZ 标量 shape 是 `()`；内部冻结 metadata 的 tuple 不改变此契约。

外层只用跨所有 fold 合并每类 CE 后的类 RMS，温度 1，不加 `.5||Z||²`。梯度为 `g_U W`，没有 `+Z`。最多 4 次 accepted 更新、每次最多 12 次 trial，步长 `.125*.5**(trial−1)`，方向归一化，候选投影至半径 .5 球，Armijo 使用实际 delta。保留最后 accepted 状态；失败记录 budget exhaustion，不挑训练最高步骤或重跑。

## 边界与失败

- new0 在 preparation 不拟合新 C，在 fit 精确返回 actual B 对象。
- K1/rank0 不更新 adapter，但仍求完整 B/C head。rank0 不等于 raw kernel 零核。
- τ0 沿用既定 original complete-interaction equivalence 前向，不求 adapter 梯度。仅压缩完整原始输入精确相同的旧约束；同时核验整个 train kernel 行和 held cross 列精确一致，保留全部旧 IDs 与 physical→representative 映射。未使用标签、近似距离阈值或训练秩删约束。
- 正 τ 的重复组还要求原/适配完整输入精确一致；新增样本不压缩，保留每个物理样本的 SSE。
- τ=None/γ=None/s₀=0 的零核，C 的旧锚点约束强制 residual 和自由常数均为零，输出实际 B prior。B 仍使用合法解析均值截距；balanced zero-prior 的精确零保持原 tie 规则。
- 原 raw old Gram 无 ridge。病态/不可分辨 projection、非 PSD 或正规方程/constraint 残差失败由独立纯核抛出，上层记录 `TECHNICAL_FAILURE`；不 jitter、不更换带宽、不做软 fallback。

旧点错误会被保留，新旧完整核输入重合时新增方向可能不可学习。旧 support 上函数相等不能推出 query 零遗忘；自由常数还可能影响远离 support 的点。这些限制保持冻结设计的范围。

## 日志、归档和独立重算

`state_callback(key,dict[str,ndarray])->JSON mapping` 接收只读有限 numeric arrays，dtype 为 float64/int64/bool，标量 shape `[]` 保留；classes/IDs/缺失尺度原因在 metadata，不写字符串 NPZ 或 NaN。缺失 τ/γ 数组为空。无 callback 时 `records` 保存相同无损数组。

事件名是 `CONDITIONAL_JOINT_PREPARED/INITIAL/GRADIENT/TRIAL/STEP/FINAL`。PREPARED 保存 context，fit audit 的 `preparation` 完整保留 run/row/scope/split/fold/trial/parent_k/train_k。FINAL payload 是 `{mode,state_ref,audit}`；所有 scope 审计仍由入口命名空间绑定，不能假设为旧 Affine 的扁平 FINAL。

每个 objective fold 的 `head_state_ref` 归档：original/adapted train/held b/a、整数 labels、q、old/new indices、old representatives/inverse groups、原始/当前 distance、raw_train/raw_cross（未乘 γ 的径向核）、Y/residual_target/M_train/M_held、scores/train_scores、τ/γ/s₀。C 另含 pure kernel 全部 arrays：A/B/D/F/E/raw_A/raw_D、J/z/s/c_N/c_H、K_perp/L_perp、α/β/v、chol_A/chol_ridge/projection_rhs、old_residual/old_scores/train_new_scores/held_scores/R_N。B 保存 K/L、解析截距、α、Schur z/s、combined RHS 和稳定中心量。

GRADIENT fold 还保存 `adjoint_*`：完整 pure A/B/D/F/E 上游、G/T/Lambda/X_alpha/X_T/g_b/projection_adjoint_rhs，以及 raw/distance/adapter 上游。INITIAL/GRADIENT/TRIAL/STEP/FINAL 保留 Z/U/W/anchor、方向、实际 delta 和类 CE sums/counts/RMS。`final_objective_state_ref` 明确引用 aggregate record；final state 递归含 `prior_B_*` 和整数 `prior_old_class_indices`。独立 analyzer 可以从原始/适配几何及 actual B 重建，不必调用此模块 forward。

## 真实资源字段

导出 `PREPARATION_COUNTERS/STAGE_COUNTERS/AUDIT_COUNTERS`。既有 ajlr_* 只表示函数坐标联合家族的真实 prep/stage/forward 工作，schema/type/method 独立。prep 计数只在 `audit.preparation`，stage 顶层不重复累加 prep；入口聚合一次。

新增 pure counters：projection/residual/projection_adjoint/residual_adjoint 四前缀，各有 `_factorization_count/_triangular_solve_count/_triangular_rhs_count/_triangular_rhs_element_count/_triangular_dense_work_unit_count`；另有 `spectral_diagnostic_count/completed_factorization_count`。C 每正核 forward 实际有 2 次 Cholesky 和 **3 次 eigvalsh**（A、rawSchur、K_perp），因此不把纯核成本仅写作两次 factor。谱诊断的 cubic 成本包含在实测 conditional_head_seconds/forward_seconds 中，未单独隔离谱耗时。

B 的 residual 前缀表示其 ordinary Affine head，projection 为零；C residual 表示 projected ridge。head triangular 总量等于 projection+residual。derivative triangular 总量等于 projection_adjoint+residual_adjoint。`ce_adjoint_solve_count` 表示一组完整 CE 伴随：B 对应一个 SPD 系统（2 triangular），C 对应两个 SPD 系统（4 triangular）；各系统 RHS 已按其真实维度单列。

forward projection RHS 为 p+1，forward residual 为 C；B 为 C_old+1。C backward 两系统各 C。`dense_work_unit_count` 为 n²×RHS 代理，不能标成实测 FLOP。raw/reference 距离是总工作与其中旧参考子集，reference 不再相加成额外总距离。prep prior fit/score 与外部单样本 score 的 prior/residual 距离、核、adapter、dictionary 工作分别保留。失败和 rejected trial 的已发生工作计费，缓存展示不计新 fit。

`declared_coordinate_scalar_count=736r` 是表示尺寸；`trainable_parameter_count/optimizer_parameter_count` 在 no-information 为零；`active/trained_parameter_count` 仅在实际更新后非零。闭式 α/β/v/intercept 数量由 `analytic_head_scalar_count` 单列。

记录 forward/adjoint/prep/fit/score 的实际 wall time、`forward_cache_bytes/max_forward_cache_bytes/max_simultaneous_forward_cache_bytes`、`resident_numeric_state_bytes/deployment_numeric_state_bytes/retained_vector_record_bytes`。cache 字节按真实 ndarray 根缓冲去重，包含 adapter derivative cache 和 pure state/expansion 缓冲；部署集合不需要投影 A/J/factor，但仍含实际 B prior、几何和 α/β/v。这些是 numeric buffer 字节，不能冒充进程 peak RAM/VRAM；完整硬件、峰值、能耗和端到端传输由后续入口实测，当前 N/A。

## 合成验证状态

新增测试覆盖真实 736 维 branch geometry 的 B/C 完整 Z 方向差分、独立全 Q KKT oracle、actual B mapping 冻结、全部旧列约束、每类 CE 汇总、缓存真实计费、4×12/非升 CE/最后 accepted、K1/rank0/new0/零核/τ0 重复组、单样本/空输入/类行置换、无损真实 NPZ callback、readonly scalar/labels 和真实 resource identities。

编写 worker 仅做 AST/UTF-8 静态检查。root 在约定 `ssr-gpu` 环境串行完成核心与五个 ENTRY 测试：**37 passed，16.48 s**，证据前缀为 `E:/type10-7/.codex_tmp/pytest_utf8_1790819674350570000`。首次检查发现 C 的 q-gauge 审计、内部冻结 shape 与 JSON 格式口径以及 Training 的公开 `audit_dict()` 接口问题，修正后完成此联合验证；失败日志保留。该结论覆盖合成数学、实际训练回调和无损归档，不代表真实数据性能或完整 summary/report/部署链已完成。
