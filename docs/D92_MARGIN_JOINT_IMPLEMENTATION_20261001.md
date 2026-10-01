# Margin 联合 LocalRidge 结构实现草案

状态：**联合core已通过合成数值验证与限定独立数学审查，未预登记或启动该方法的真实实验。** core作者仅用源码与合成数据完成实现，没有读取真实特征、权重、cache、query、外层评分、历史结果或实验索引，没有修改现有Affine、Conditional或健康运行。root随后在项目环境串行执行数值验证。数学性质与合成证书不代表准确率提升、星载部署完成或用户理想目标已经达到。

实现依据为 [Margin 结构推导](D92_MARGIN_JOINT_STRUCTURAL_DERIVATION_20261001.md) 和已冻结纯头 [组件说明](D92_MARGIN_QP_HEAD_IMPLEMENTATION_20261001.md)。原始问题、非负 dual 和正则 active 区域完整伴随的独立数学证书见 [合成证书](../tests/test_d92_margin_joint_math_certificate.py)。本文不重述该推导。

## 1. 文件和公开接口

新 core：[d92_margin_joint_local_ridge.py](../code/cvsrffi/d92_margin_joint_local_ridge.py)。合成测试：[test_d92_margin_joint_local_ridge.py](../tests/test_d92_margin_joint_local_ridge.py)。

- schema：`d92_margin_joint_local_ridge_v1`。
- method：`D92-MarginJointLocalRidge-v1`。
- arms：`local_ridge`、`margin_joint_seq`。
- 状态：`MarginJointTraining`、`MarginJointState`；其 `audit_dict()` 返回独立安全 JSON，`state_records()` 返回内存档案的数值副本。

```python
prepare_margin_joint_training(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes,
    max_transitions, max_factor_buffer_bytes,
    inherited=None, context=None, log_callback=None, state_callback=None,
)
evaluate_margin_joint_objective(
    prepared, Z, anchor_U=None, *, gradient=True, forward_cache=None,
)  # (RMSCE, g_Z, audit, ObjectiveCache)
fit_margin_joint_local_ridge(
    prepared, *, mode='B', baseline_state=None,
    log_callback=None, state_callback=None,
)  # MarginJointState; modes B / C_seq
predict_margin_joint_local_ridge(state, **five_named_blocks)
```

`max_transitions` 和 `max_factor_buffer_bytes` 必须由调用方显式传入正整数，包括 B 准备时也要声明。没有默认值、隐藏重试或已选定的实际实验预算。每个 C head fit 使用 prepared 中冻结的相同数值。限制属于**每次 C head 调用**，不能当成整个序列或实验的总预算。`FROZEN_CONFIG` 中该两项是调用方必需的声明，prepared 的 `actual_parameters` 和 stage 的 `config` 才记录实际传入值。

纯头 ABI 为 `fit_margin_qp_head(*, K, M, labels, old_indices, max_transitions, max_factor_buffer_bytes)`、`predict_margin_qp_head(state, *, L, M)`、`margin_qp_head_vjp(state, *, L, G)`。其预测已经包含 M。联合推理则计算 residual 展开后单次加入实际 B，避免双加 prior。

## 2. B、C 和合法支持集边界

B 只使用当前旧类 support：解析头复用已验证 Affine 的 centered-kernel Schur 解与完整自由截距伴随。训练目标和优化循环采用 Conditional 的 CE-only 投影逻辑，**不调用现有 Affine 的整体 evaluate/fit**，因为该旧方法仍包含 `.5||Z||²` 和 `+Z`，且 trial 没有硬球投影。

C 只接受同一新 run、row、scope 和物理 fold 当次生成的本方法 B 状态。`run_id/row_id/scope` 必需；`split_id/fold` 的存在性和值必须一致。物理旧 IDs、旧标签与五块 raw 特征必须匹配。trial 是日志坐标，不新增继承授权条件。C 起点为实际 `U_B`，没有 adapter reset；白化坐标未覆盖的 anchor 分量保留。

内层 prior 只在 old inner-train 拟合解析头，使用冻结实际 `U_B`；原尺度也只来自该 old inner-train。它可以对当前合法 train/inner-held 特征进行预测以形成固定 M，但 head 拟合与尺度不使用 inner-held 标签。final C 使用当次实际完整 B 的函数及尺度。旧列按类 ID 映射，新增列为零。`M_train/M_held` 只读；C trial 不重训 B 或改变阈值。

C 对所有物理 old/new train 计算同一个 raw Gram；没有旧点压缩、仅 new 行 SSE、旧 residual 等式或 q 均值约束。标签为全部注册列 one-hot 减 `1/C`，残差 target 为 `Y−M`。最小真实类 margin 包括补零新列，保留负数和零的原值。Margin 保护仅作用于合法旧 train，不是独立 held/query 的零遗忘保证。

## 3. 完整梯度和外层优化

C 的 kernel 为 `gamma*exp[-(d_original+d_adapted)/(2*tau)]`。纯头返回全 Gram Frobenius 约定下的对称 `barK` 和一般 `barL`，包括自由截距 `g_b` 与 active multiplier 响应。联合 core 不对 C 调用 `center_vjp`。

完整链为：纯头 K/L 上游乘 gamma，进入 adapted-distance 的半权重导数；train Gram 的两个端点及 held-to-train 的两个端点都进入距离 VJP；累加所有 train 和 held adapter VJP 得到 `g_U`，再计算 `g_Z=g_U@W`。old-old、old-new、new-new 和 held-to-old 块均保留。原始几何、实际 B、标签和 margin 阈值冻结。

Outer 先跨所有 inner folds 汇总每类 CE sums/counts，再求类均值的 RMS；温度为 1。没有 keep、guard、近端 penalty 或 `+Z`。固定 4 次更新、每次最多 12 次 trial，步长从 `.125` 逐次减半。trial 先投影到 `||Z||_F<=.5`，Armijo 使用实际投影后的 delta。保存最后 accepted 状态；正常 trial 耗尽记录 `TRIAL_BUDGET_EXHAUSTED`，不反馈重跑或选择历史 checkpoint。

Forward cache 绑定 prepared 对象和精确 Z；复用仅免去已经完成的前向计算。再次请求梯度仍实际收费。白化球约束的是平均 pre-tangent 位移，不能直接转写成最终分类 margin 或 query 风险保证。

## 4. 退化和失败

| 边界 | adapter 更新 | final head |
|---|---|---|
| new0 | 不更新 | QP 前直接返回同一实际 B 原对象；没有新 stage 事件或 head 收费 |
| K1、没有 inner-held | 不更新 | 仍拟合完整 B 解析头或 C margin QP |
| rank0 | 不更新，保留 anchor U | 仍使用实际 raw kernel；rank0 不等于零 kernel |
| tau0 | 不声称普通 Gaussian kernel Jacobian | 原 complete-input exact equivalence kernel，保留全部物理 SSE 和标签约束 |
| tau=None / gamma=None 或 gamma=0 | 不更新，无连续 adapter-kernel 信息 | C 仍拟合受约束自由常数 b；不能套用 Conditional 的 residual=0 分支 |
| 正 tau 下 Jacobian 不受支持 | 技术失败 | 保留已完成头、最后接受状态、失败输入与真实已消耗成本；不换无约束或零梯度 |

纯头 `MarginQPFailure` 和 `UnsupportedJacobian` 转成联合 `NumericalFailure`，公开 `.audit/.audit_dict()/.records/.arrays`，原异常保留在 cause。`.arrays` 是失败 state ref 对应的只读数值快照，callback 写磁盘时也可直接读取这个接口。失败 audit 包括 `failure_code`、纯头 audit、失败 state ref 和已经完成 fold 的 refs。QP 失败数组以 `qp_failure_` 前缀保存；backward 失败还保存 G。前向、伴随和 final 失败的最后成本都累计，不以“没有返回成功头”为理由清零。没有 jitter、伪逆、软 margin、动态带宽或静默 fallback。

## 5. 状态和日志

`state_callback(key, dict[str, ndarray]) -> JSON ref` 原样保存有限数值数组；整数 labels/indices/counts 保持 int64，scalar 保持真实 shape。ID、类序、scope、预算及数组 shape/dtype/nbytes 在 audit metadata 中。无 callback 时完整数值 records 留在内存，不仅保留打印分数。

PREPARED 保存 H/W/SVD/anchor/V0。INITIAL、GRADIENT、TRIAL、STEP、FINAL 保留目标、状态 ref 和完整 head/伴随档案。每个 head 保存 original/adapted train/held b/a、原/当前距离、raw kernel、K/L、固定 M/Y、标签、old/new indices、scores、alpha/b 及 QP 的 P_old、margins/delta、slack、multipliers、working set 与两类 factors。GRADIENT 额外保留完整 g_b、T_G、W_eta、K/L/raw-distance 上游和 g_U/g_Z；B 保存解析头的中心化与 Schur 状态。

final 保存实际 B 的 `prior_B_*` 数组及 `prior_old_class_indices`，加上当前 U/Z/W/H/anchor/SVD。State 的 `score/score_with_audit/predict` 逐样本面对全部注册列；prediction 使用固定类 ID 排序的统一 argmax。没有 truth/role、配额或跨样本重排。

`log_callback` 接收完整结构化事件，并含可打印 `text`：实际 RMSCE、prox=0、步长、梯度、更新、接受状态、头/factor/RHS/transition 和时间。PREPARED 显示实际 rank、固定参数与调用方资源界限。`compact_training_record(record)` 提供纯 scalar JSON/CSV 面，并标注 `counter_ownership` 为 PREPARATION、OBJECTIVE_INCREMENT 或 STAGE_CUMULATIVE；不能机械相加不同所有权记录。源验证和未测设备/峰值为 N/A，不伪造源访问或 GPU 测量。

## 6. 实际资源口径

入口应直接导入 `PREPARATION_COUNTERS/STAGE_COUNTERS/AUDIT_COUNTERS`。家族 `ajlr_*` 字段保留工作量含义，schema/method/state 与旧方法独立。prior B 拟合/预测只归 preparation；student 和 final head 归 stage。每个 rejected trial 的 forward 全部收费。

新的稳定前缀为 `margin_qp_forward_` 和 `margin_qp_adjoint_`。每个前缀的 `QP_WORK_KEYS` 为：

```text
transitions
full_constraint_scans
spectral_checks
compact_snapshot_rebuilds
compact_snapshot_dense_work_units
spectral_cubic_dimension_units
independence_checks
factorization_attempts
factorizations_completed
condition_estimation_calls
triangular_calls
triangular_rhs_columns
triangular_rhs_elements
triangular_dense_work_units
```

该 28 项进入 STAGE/AUDIT_COUNTERS，以纯头实际 ledger 累加；prior 解析头没有 QP，所以不向 preparation 虚填 QP 工作。`inner_factorization_count+final_factorization_count` 是真实 factor attempts；`completed_factorization_count` 单列成功数。C 的 head/derivative triangular 调用、RHS 列、元素与 dense-work 字段逐项映射实际调用，不能复制 Conditional 的双 factor、三 eigen 或固定 solve 公式。B 解析头仍实际记录 `[C+1]` RHS 的两 triangular solves。伴随复用前向因素，正常情况下没有新 factor，但有真实 solves。

`spectral_cubic_dimension_units`、`triangular_dense_work_units`、`compact_snapshot_dense_work_units` 是明确的维度工作代理，不是测量 FLOPs 或时间。完整 ledger 的 solves、factors、events、条件估计及残差仍在每个 head audit；它们不是新参数或性能指标。

`margin_qp_peak_factor_buffer_bytes` 和 `margin_qp_peak_explicit_solve_temporary_bytes` **不在可相加 COUNTERS 中**，取实际最大值。前者只覆盖组件声明的自有 factor 输入/输出；后者覆盖显式 solve 输出，均不冒充进程峰值。Forward cache 与同时存在的 accepted/trial cache 数值字节另行测量。

Resident 使用实际 byte-backed ndarray 缓冲去重，包括 MappingProxyType、actual B 和 State 实际保留的内存 records。`retained_vector_record_bytes` 另外报告档案子集的物理字节，不能再与 resident 相加；callback 已写出且 State 未保留的磁盘数组不虚填为常驻数组。Deployment 数值字节只计算推理需要的 expansion/geometry/actual B 数组：常数零核分支只需要截距，tau0 不需要 adapter/dictionary，正 tau 才需要当前 adapter 和适配 train 几何。该口径排除 Python/类 ID metadata、library workspace、磁盘压缩与协议包装；并非已生成部署包的大小。序列化部署、增量传输、RSS/GPU 峰值和设备型号未测量，保持 None。

## 7. 合成验证与未覆盖项

归档字段区分几何尺度与 QP 约束向量：B 的标量几何尺度保持 `s0`；C 的标量几何尺度保存为 `reference_s0`，`s0` 则保存全部旧类物理行对所有注册类别的 QP 基线松弛向量。独立分析发现并修复了此前同名字段覆盖向量的问题。修复只改变归档，不改变求解状态、数学方法或优化参数；尚无真实 Margin run 使用旧归档。

测试只用固定人工 raw feature 表、有限小矩阵及合法合成 support 标签。覆盖 B/C 实际序列、内层 prior 隔离、固定 prior、全列 margin、解析 B 与独立 feature-primal oracle、完整 U/Z 差分（含 active table）、CE 汇总与合法 held 监督、cache 计费、4×12/硬球/最后接受、拒绝 trial、new0 原对象、K1/no-held/rank0、零核非零自由截距、tau0 保留重复物理行、unsupported Jacobian 与资源失败、真实回调 NPZ 无 pickle 读回、逐样本一致与真实数值 buffer 字节。

Active-table 和正 scale/rank-one 的失败 fixture 是数学边界测试，不冒充从生产 B 生成的注册状态。它们不进入实际序列测试，也不用于选参数。独立 primal 的 active-set 枚举仅在 4 条约束的小例子执行；较大合成序列检查全部 KKT、margin 和完整状态，没有抽样原方法的约束。

root在已验证ssr-gpu环境串行运行本测试文件：**21 passed in 4.64s**，证据`E:/type10-7/.codex_tmp/pytest_utf8_1790835764364153400.stdout`及同前缀`.stderr`。独立query-blind审查核对完整两端梯度、实际继承与inner prior、退化分支、投影接受和成本口径，限定范围未发现P0/P1；审查只读，未运行数值或读取真实结果。此前19项数学证书与30项头组件测试未重复运行。

五入口集成另通过34项合成检查，见[入口说明](D92_MARGIN_JOINT_ENTRY_20261001.md)。尚未完成新方法的独立真实训练档案摘要、独立Margin分析链、真实设备测量、Ground A三阶段配对或实际部署。实际资源限制数值和实验矩阵未选择，没有新run。不得将这些正确性检查写成联合方法性能有效。
