# AffineJoint 独立核心草案

日期：2026-10-01。状态：`STRUCTURE_CANDIDATE_DRAFT_NOT_PUBLISHED`。schema 为 `d92_affine_joint_local_ridge_v1`，method 为 `D92-AffineJointLocalRidge-v1`。

**本实现只在 AJLR 的 residual head 中增加由合法 support 解析估计的无惩罚类截距。** 尚未发布或启动实验；数学性质不构成旧类、新类或 query 准确率提升。核心 Agent 只执行 AST/UTF-8 静态检查，数值测试由主 Agent 在项目环境统一运行。

实现依据 [截距数学审计](D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md) 和 [最小实现映射](D92_AJLR_AFFINE_IMPLEMENTATION_MAP_20261001.md)。[新核心](../code/cvsrffi/d92_affine_joint_local_ridge.py) 是独立文件，未 monkeypatch 旧模块或改动 AJLR、现有配置和运行 release。复用的 FCR 数值原语只承担既有 DCT8、GELU、范数保持 adapter、函数坐标和 interaction VJP，不修改它们的全局配置。

## 公开接口与输入范围

```python
prepared = prepare_affine_joint_training(
    z_id=..., fft=..., t_emb=..., f_emb=..., pa_local=...,
    support_labels=..., support_ids=..., classes=..., old_classes=...,
    inherited=None, context=None, log_callback=None, state_callback=None)
state = fit_affine_joint_local_ridge(prepared, mode='B')  # C 仅有 C_seq
loss, g_Z, audit, cache = evaluate_affine_joint_objective(
    prepared, Z, anchor_U=None, gradient=True, forward_cache=None)
scores = state.score(**features)
scores, workload = state.score_with_audit(**features)
predictions = predict_affine_joint_local_ridge(state, **features)
```

状态类型为 `AffineJointState`，准备类型为 `AffineJointTraining`。仍接受五块当前 received support 特征、标签、物理 ID 和注册类元数据；没有 outer-held/query 拟合入口、源域逐样本输入或 encoder backward。预测逐样本面对全部注册类，精确 tie 按 class ID 字典序。

C 只接受同一新 run、同一 row/scope/物理 fold 当次合法 B 的 `AffineJointState`，并检查旧物理 ID、标签和特征完全配对。C 与 B 的 `context` 必须显式提供相同 `run_id`、`row_id`、`scope`；`split_id` 或 `fold` 出现时两边必须匹配。`trial` 是日志信息，不构成额外授权或继承门槛。合成调用使用显式 synthetic scope。AJLR/其他 run 的目标适配 state 不因 balanced B 的数学等价性成为可继承来源。

## 唯一结构变化

保留原始旧 support 估计的 τ/γ、固定旧物理参考测度 q、双完整 interaction 距离半半、DCT8 与 κ=1/4、U=anchor+ZWᵀ、跨全部 inner folds 汇总的类 RMS CE、0.5||Z||² 和 4 次更新×最多 12 次固定 Armijo 试探。B anchor 为零；C anchor 为实际本次 U_B，未覆盖 null 分量保留。没有 reset 分支、参数网格、keep、guard 或输出校准扫描。

`_solve_affine_head` 对 `[E,e]` 合并 C+1 右端执行一次 n×n Cholesky 和两次 triangular solve，得到 Schur z/s，再求截距和 canonical α。Y 仍逐行 onehot−1/C，E=Y−M；不通过单独减样本均值并丢掉常数改变模型。完整 train/held score 同时包含 prior、residual 核函数和截距。ridge 罚项只作用于 residual，不惩罚截距。

`_affine_adjoint` 复用前向 Cholesky、z/s，包含 `g_b=sum_rows(gscore)`，形成完整 saddle 伴随 T/eta。`_backward` 将 T 代入 barK，并保留完整中心 VJP 和 raw Gaussian、interaction、adapter 链。前向/伴随公式分别对应数学审计式（4）至（7）、（22）至（28）；不是继续使用旧无截距 `A^-1 L^T G`。当前 actual B prior 的核与截距一起冻结，没有 C→B 反传。

原始等价核 τ0=0、s0>0 仍执行 Schur solve，adapter 导数为零；它通常不是零核。K=L=0 时显式使用截距 mean(E)、α=E−e·intercept，0 factor/0 triangular。当前均衡、零 prior、缺尺度契约的数学截距为零，代码保留精确全类 tie，不从舍入误差生成常数类别偏好。一般非零 prior 的零核低层测试仍使用 mean(Y−M)，不会扩展当前数据/尺度权限。

K1/rank0 不更新 adapter，但保留完整 support 的解析头。Nnew=0 在任何新增 C solve 前精确复用 actual B，不重拟合 residual。均衡旧 B 在每个固定 U 上与原无截距 B 数学等价；浮点操作顺序不承诺 bitwise 相同。低层不平衡 B 一般不等价；public equal-K 契约没有放宽。

正规方程检查使用 `(K+I)alpha+e*intercept−E`，分母含当前实际 trace 与截距项；另检查样本维度 α 零和、类别维度 score/α/intercept 零和及 Schur s 的正性和实际 trace 界。residual 在旧参考上的核均值仍为零，完整参考均值为 `q^T M+intercept`。不使用 jitter、带宽/Schur floor 或结果驱动 fallback。

## 档案与状态

回调形式与旧核心一致：`state_callback(key, arrays)` 返回 JSON 引用，`log_callback` 接收 `AFFINE_JOINT_PREPARED/INITIAL/GRADIENT/TRIAL/STEP/FINAL`。数组仅为有限 numeric dtype，标签/索引/count 保留 int64。None τ/γ 为长度零的 float64 数组并由 audit null 解释。

在原始/适配 train-held geometry、q、raw kernels、K/L、Cholesky、α、Y/M/E、scores、中心量与尺度之外，每个 head 固定增加以下键：

| 键 | 形状 | 用途 |
|---|---|---|
| `intercept` | C | 完整分类函数与推理状态 |
| `schur_z` | n | 前向与伴随共享 A⁻¹e |
| `schur_s` | 标量 | 共享 Schur scalar |
| `combined_rhs` | n×（C+1） | 实际合并右端证据 |

GRADIENT 的 aggregate NPZ 还保存 `fold_j_adjoint_T`、`fold_j_adjoint_eta`、`fold_j_adjoint_g_b`、`fold_j_adjoint_rhs`。不同 forward 的 head 只归档一次；cache 展示不新增拟合。拒绝 trial 单独归档并计费，不能覆盖最后接受 cache。

FINAL 的 `prior_B_` 前缀保存本次 actual B 的完整 geometry/α/intercept/U，`prior_old_class_indices` 保存旧列映射。inner head 的 `prior_ref` 指向准备中的对应 `prior_folds[].head_state_ref`；final 使用实际完整 B。`to_arrays()` 提供完整 numeric 档案，可独立重算冻结 prior+residual+截距，而不重新拟合。

## 实际费用字段

既有 `ajlr_preparation_count`、`ajlr_stage_count`、`ajlr_forward_evaluation_count` 保留为函数坐标联合家族的工作计数，新 schema/method/API/state 独立。所有字段由 `PREPARATION_COUNTERS`、`STAGE_COUNTERS`、`AUDIT_COUNTERS` 导出；preparation 只累计一次，prior 与 student/final 分列。

新增 RHS 字段采用三个稳定后缀：`triangular_rhs_count`、`triangular_rhs_element_count`、`triangular_dense_work_unit_count`。每次真实 triangular 调用分别累计 RHS 列数 r、n·r 和 n²·r。最后一项是 dense 工作代理，**不是测得 FLOP 或耗时**。

| 所属计数集合 | 固定前缀和额外字段 | 真实行为 |
|---|---|---|
| PREPARATION | `prior_` + 三个 RHS 字段；`prior_intercept_fit_count`、`prior_intercept_addition_count` | 每个实际 prior head 的合并前向及其实际 score 加法；final actual B 复用不新增 prior head |
| STAGE | `head_` + 三个 RHS 字段；`intercept_fit_count`、`intercept_addition_count` | student/final 原解与完整 score，加上所有完成的拒绝 trial |
| STAGE | `derivative_` + 三个 RHS 字段 | 缓存前向下的单 CE 完整伴随 |

一个有信息前向为 2 次调用，每次 C+1 RHS，三个累计量为 2(C+1)、2n(C+1)、2n²(C+1)。一个有信息伴随为 2 次调用，每次 C RHS，对应 2C、2nC、2n²C。零核三者为零；解析均值截距仍是真实 head 执行，所以 `intercept_fit_count=ajlr_forward_evaluation_count`，`prior_intercept_fit_count=prior_head_fit_count`。intercept_addition_count 按实际 train/held 或推理记录数×该 head 类数记录，constant 分支也记录其 score 广播。

最终 audit 分开保存 `analytic_intercept_parameter_count=C`、`analytic_intercept_contrast_count=C−1`、`analytic_coefficient_parameter_count=nC` 和 `analytic_head_parameter_count=nC+C`。这些是 support 解析学习的决策状态；adapter 的 `trainable_parameter_count/trained_parameter_count` 仍只计 Z，不因截距增加 SGD 参数。

`persistent_state_bytes` 对返回状态实际保留的唯一 numeric buffer 计费，包括 Schur/RHS/head cache/坐标与实际 B；`deployment_numeric_state_bytes` 只列推理所需数组，包括截距，排除 Schur/RHS。共享 V0 只计一次，records 字节另列。新增完整截距数组为实际 8C bytes，不能按 C−1 个 contrast 自动减存储。峰值 RSS/显存、部署压缩包和增量上传量仍需后续实际测量，未测项为 N/A。

## 确定性合成测试

[新测试文件](../tests/test_d92_affine_joint_local_ridge.py) 不访问任何实数据、历史结果、query 或 outer 成绩。测试覆盖：

- 独立有限特征 primal 与 saddle oracle；完整等式、Schur 缓存和 canonical 样本零和。
- 非零 g_b 的完整 head 伴随有限差分与漏截距反例；B/C 的完整 Z 方向有限差分和近端。
- 固定相同 R/Q/U/τ/γ/Y/prior 时的 q gauge、α/完整 score/ridge/raw VJP/Z 梯度等价。
- 每个固定 U 下 balanced B 的旧 head/score/梯度等价，不平衡低层 B 的不等价。
- 同 run/row/scope/物理 fold actual B 绑定，拒绝旧 AJLR 类型和跨 run state；trial 记录不增加授权。
- Nnew=0 对象复用，K1/rank0 的完整解析头，τ0 非零等价核和零核均值分支，精确缺信息 tie。
- prior+residual+截距的独立 archive 复算、单样本/批切分一致性、全类 argmax 和禁止额外输入。
- 合并 RHS/伴随/prior 的实际费用、analytic/gradient 参数分列、唯一 buffer 字节、无损回调和 12 次耗尽后 accepted cache 保留。

这些是直接实现正确性测试，不验证实数据性能；未在本文中宣称已运行通过。后续独立入口、摘要、发布和实验启动由主 Agent 决定和执行。
