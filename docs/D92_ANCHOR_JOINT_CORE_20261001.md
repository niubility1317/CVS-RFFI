# AJLR_seq 核心实现

状态：已实现，核心 Agent 仅执行 AST、UTF-8 和冻结配置静态检查。数值正确性由主 Agent 在 `ssr-gpu` 环境运行合成测试确认；本文不报告实数据性能。

数学约定来自冻结设计 [D92_JOINT_AFTER_FCR8_DESIGN_20261001.md](D92_JOINT_AFTER_FCR8_DESIGN_20261001.md) 的第 2 至 7、9 节。实现只接受当前合法 support 的五块特征、标签、物理 ID、注册类表和同一 row 的实际 B 状态。函数没有 outer-held 或 query 参数，也不读取任何文件、结果索引、source 数据或 encoder 状态。

`prepare_anchor_joint_training` 按物理 ID 排序，保存 DCT8 的 GELU 字典 H、一次薄 SVD、W、谱和实际 anchor_U。B 的 anchor_U 为零；C 为本次实际 B 的 U。Z=0 精确复制 anchor_U，C 的未覆盖分量保留。K1 和 rank0 保留完整闭式 head，adapter 不更新；Nnew=0 精确复用 B，准备不新增 SVD、prior head 或 C head。

每折的 τ、γ、s0 只由旧 inner-train 原始 interaction 几何计算。nuisance 计算没有拟合头或线性求解。C 的 prior head 固定实际 U_B，只拟合该折旧 inner-train；最终 C 直接使用实际完整 B。固定 q 在旧参考位置为 1/M，其余为零。每次 trial 对 train、held 和 q 的参考集合重新映射。原始与适配后完整 interaction 距离各占一半，距离用现有 rank-two QR 非负平方和计算，不物化 123616 维映射。中心化使用 expm1 和参考差分后的 rank-one 运算，不物化 Pq 或增加 N³ 中心乘法。

标签为逐行 onehot−1/C。闭式头精确求 α=(K+I)⁻¹(Y−M)，C 的 M 是实际 B 分类函数按旧类列映射并将新列补零。没有 intercept，不再次减残差的样本均值。每个有信息头使用一次 float64 Cholesky 和原解的两次 triangular solve；缺少旧核尺度时 K=0、α=Y−M、核预测为零。τ=0 使用原始完整特征等价关系的 PSD 核，不合成小带宽，也不对该边界求导。当前实际 trace 用于正规方程残差分母，沿用 `128*eps64*max(N,C)` 容差；不做旧 FCR 的动态 trace-match 断言，不加 jitter。

内层目标先跨全部 folds 按类别汇总 CE，再计算各类 mean CE 的 RMS，温度固定 1，近端为 0.5||Z||²。每折只有一个 CE 伴随，使用缓存 Cholesky 的两次 triangular solve。完整 VJP 包括闭式解、train/held 的双方距离以及旧参考中心测度贡献。最终 g_Z=g_U W+Z。没有 keep、guard、teacher margin、accuracy 选步或组偏置。

优化最多 4 次更新。每步从 1/8 开始做最多 12 次固定二分 Armijo 试探，接受第一个同时满足 Armijo 和目标不增的试探。正常耗尽记 `TRIAL_BUDGET_EXHAUSTED`，保留最后接受的坐标和 forward cache；拒绝 cache 不能覆盖它。所有注册类在同一 argmax 中竞争，精确 tie 按物理 class ID 字典序，每个输入样本独立应用相同公式。

公共接口：

```python
prepared = prepare_anchor_joint_training(
    z_id=..., fft=..., t_emb=..., f_emb=..., pa_local=...,
    support_labels=..., support_ids=..., classes=..., old_classes=...,
    inherited=None, context=None, log_callback=None, state_callback=None)
state = fit_anchor_joint_local_ridge(prepared, mode='B')  # C 使用 C_seq
loss, g_Z, audit, cache = evaluate_anchor_joint_objective(
    prepared, Z, anchor_U=None, gradient=True, forward_cache=None)
scores = state.score(z_id=..., fft=..., t_emb=..., f_emb=..., pa_local=...)
scores, workload = state.score_with_audit(**features)
```

`state_callback(key, arrays)` 返回可 JSON 序列化引用。数组只含有限 numeric dtype，None 的 τ/γ 表示为长度零的 float64 数组，语义由 audit 的 null 明确记录。`log_callback` 接收 `AJLR_PREPARED`、`AJLR_INITIAL`、`AJLR_GRADIENT`、`AJLR_TRIAL`、`AJLR_STEP`、`AJLR_FINAL`。每个不同 forward 的 head 数组仅保存一次；梯度和最终展示复用引用，不新增 head 费用。INITIAL/GRADIENT/TRIAL/STEP 保存 Z、U、anchor、W、谱、scores、labels、class CE sums/counts/means、RMSCE、prox；梯度保存 g_Z/d_Z，trial 保存实际 delta_Z。FINAL 保存全部原始五块特征、原始/适配 geometry、闭式 head 和实际 B prior 的完整推理数组。

head 数组的固定键为 `original_train_b/a`、`original_held_b/a`、`adapted_train_b/a`、`adapted_held_b/a`、`train_labels`、`held_labels`、`q`、`raw_train/cross`、`raw_train/cross_minus_one`、`distance`、`cross_distance`、`K`、`L`、`chol`、`alpha`、`Y`、`M_train/held`、`E`、`train_scores`、`scores`、`reference_kernel/self`、`center_mean/grand`、`tau`、`gamma`、`s0`、`actual_trace`。物理 ID、classes 和旧参考物理集合在 head audit 中。C 内层 head 的 `prior_ref` 绑定准备阶段对应 `prior_folds[].head_state_ref`；最终数组的 `prior_B_` 前缀保存实际 B，`prior_old_class_indices` 指明旧列映射。摘要可从合法 outer-held 原始特征与这些冻结数组独立复算 scores，无需再次拟合。

费用由 `PREPARATION_COUNTERS` 和 `STAGE_COUNTERS` 导出。准备计数只加一次，包含 prior head、prior 原解、prior score 和其实际 distance/kernel/adapter 工作；stage 只含 student/final head 和伴随。`reference_distance_*` 是 raw distance 中涉及参考点的子集量，与 raw distance 不再次相加。score_with_audit 对 residual 和 prior 分列，并记录真实逐样本 geometry、dictionary、adapter、distance、kernel 工作及时间。cache 展示不计新的原解。`persistent_state_bytes` 对返回状态实际保留的唯一数值 buffer 计费，包含坐标和完整 head cache；`deployment_numeric_state_bytes` 单列推理需要的数组，二者都包含实际 B prior，共享 V0 只计一次。训练证据 records 的字节另列。文件压缩字节、上传字节、峰值 RSS、显存和未知硬件信息记 N/A，不能从参数量推断。

成功阶段的计数关系：

- optimizer_steps = accepted_trial_count；trial_count = accepted_trial_count + rejected_trial_count。
- 无技术异常时 trial_attempt_count = trial_count，inner objective = initial + 完成 trial。
- student heads = inner_head_fit_count + final_head_fit_count；factorization_count = inner_factorization_count + final_factorization_count。
- head_triangular_solve_count = 2 × student factorization；prior_triangular_solve_count = 2 × prior factorization。
- derivative_triangular_solve_count = 2 × ce_adjoint_solve_count。伴随与原解是不同费用。
- 有信息阶段固定 F 个 folds，inner heads = F × inner objective；总 forward = inner heads + final head。

每个有信息阶段最多 49F 个 student inner heads、4F 个 CE 伴随、8F 次伴随 triangular，另一个完整 final head。每个 C fold 最多一个 prior head，final prior 复用 actual B。全矩阵上界必须按主 Agent 的实际登记阶段数计算；退化时 head、factor 和已训练坐标数按真实分支记录。

合成测试覆盖原解/primal oracle、无二次中心化、不平衡标签、B/C 完整有限差分、参考点显式 VJP、跨折 CE 汇总、真实继承与旧核不变边界、N0/K1/rank0/缺旧尺度/τ0、PSD/SPD、单样本一致性、行/类置换、全类竞争、12 次耗尽与接受 cache、回调档案和实际费用。该测试套件不访问任何实数据，也不能证明目标域性能。
