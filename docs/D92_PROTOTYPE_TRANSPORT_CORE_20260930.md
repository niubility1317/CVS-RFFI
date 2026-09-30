# PrototypeTransport 数值核心实现说明

最新状态：IMPLEMENTED_CORE_JSON_BOUNDARY_VERIFIED_R02_PENDING。r01 已实际运行并因 JSON 类型边界故障退出，修复回归通过，r02 待准备恢复。历史[设计](D92_JOINT_NEXT_AFTER_CHANNEL_20260930.md)与[设计审查](D92_PROTOTYPE_TRANSPORT_DESIGN_REVIEW_20260930.md)保持原记录；本文描述实现合同及验证历程，不声称性能改善。

## 交付与依赖

- `code/cvsrffi/d92_prototype_transport_local_ridge.py`：原型构造、切向旋转及 VJP、双距离 LocalRidge、跨折类别 RMS 目标、有界投影与回溯、不可变部署状态。
- `configs/d92_prototype_transport_frozen_20260930.json`：外层 `algorithm` 与核心 `FROZEN_CONFIG` 完全相同。
- `tests/test_d92_prototype_transport_local_ridge.py`：只用独立合成数据的数值及协议回归。

核心复用现有 `d92_joint_channel_local_ridge` 的规范输入绑定、稳定距离 VJP、带宽并列广义导数、核中心化和基线标签验证等纯函数；没有修改 Channel、LocalRidge 或暂停的 Residual8。发布闭包必须包括 Channel 及其原有依赖。没有 encoder 加载、源样本、源逐样本特征或 query 拟合路径。

## API

```python
prepare_prototype_transport_training(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes,
    inherited=None, context=None, log_callback=None,
) -> PrototypeTransportTraining

evaluate_transport_objective(
    prepared, theta, anchor, *, gradient=True, forward_cache=None,
) -> (loss, gradient, audit, forward_cache)

fit_prototype_transport_local_ridge(
    prepared, *, mode='B', baseline_state=None, log_callback=None,
) -> PrototypeTransportState
```

模式为 `B`、`C_seq`、`C_reset`。`prepared.audit_dict()` 返回可序列化副本。state 提供 `score(**five_blocks)`、`predict(**five_blocks)`、`audit_dict()`；`theta` 为 10 个存储参数，`u` 是相同只读数组的接口别名，便于沿用入口外形。`classes/raw/labels/ids/base_state` 对应已有入口字段。原型为 `state.prototypes.p/m/nu`；数值数组只读，audit 返回独立副本。

`baseline_state` 必须是同一规范 support 切片的原 LocalRidge。复用前同时验证 ID、原 b/a、类别、当前标签的正规方程以及核统计；只按类别集合和特征匹配不足以复用。零分类器明确记录标签不影响输出的退化语义。

## 数值与数据角色

原 b/a 只规范化一次。p 是各类原子块均值，m 是单位方向均值，ν 包含训练异类原型距离中的零。每个内折独立构造自己的原型、邻居、ν、带宽与原 interaction 迹；完整外层原型不借给内折。所有内折的每类 half-squared-margin 先累加并除以该类实际物理数，再求类别 RMS。inner-held 标签参与共享参数监督，不能把内层准确率称为独立验证。

β 为五维盒约束，η 为五维零和盒约束，总共 9 个自由度。η 零和固定分支权重几何均值，不声称消除了温度冗余。β=0 直接保留原 b/a 和原 LocalRidge 行向量评分，但反向仍通过旋转及 0.5 适配距离项；η 首步梯度为零而 β 一般可训练。q 接近零时使用连续级数，不先做除零运算。

双距离始终含半份原 interaction。允许适配分量将不同原点合并；只要求原相同输入映射相同，且联合距离不小于原距离的一半。原 tau=0 的等价核与零导数保留。无法表示的正原型距离、非有限状态、范数或头残差异常报技术失败，不加隐含 epsilon 或更换核。

C_seq 继承同 row、同物理旧 support、同标签和原缓存的 θ_B。最终旧 p/m 列先按当前 support 复算验证，再逐元素采用 B 保存值；新列追加，ν_C 和全注册头重算。C 内折仍重新构造自己的旧/新原型。C_reset 使用相同 prepared 状态但零参数与零近端 anchor。Nnew=0 复用 B。平衡 K1 没有内持出监督，保持零 anchor，与 R0 精确相同；true K1 在入口只做数值检查。

部署对每个样本独立构造原型 attention，并对全部注册类输出 LocalRidge 分数。它不先按预测类分配标签路线，不读取其他 query，也不使用真实 old/new 角色。

## 优化、事件与计数

最多四次归一化投影梯度迭代，每次依次试探步长 1/8、1/16、1/32。η 使用零和盒的欧氏投影，不用“减均值再截断”。投影残差阈值为 `128*eps64*5*eta_bound`，最多 80 次二分。Armijo 系数固定 `1e-4`，比较容差固定 `128*eps64*max(1,abs(loss_before),abs(loss_trial),abs(rhs))`。两者由 float64 精度决定，不根据成绩选择。

初始前向保存每个内头的 Cholesky、alpha、双距离和映射中间量。之后反向复用接受缓存；每次 trial 使用新缓存，仅在接受时替换。最终只使用最后接受参数，不择外层最佳步。最终训练目标由该接受缓存重算类别聚合，新增前向、头拟合和反向计数均为零。

回调是单个 dict 位置参数，事件为：

| event | 内容 |
|---|---|
| `TRANSPORT_INNER_PREPARED` | 内折 train/held ID、原迹/带宽及原型来源 |
| `TRANSPORT_INITIAL` | 初始参数、anchor 与一次前向 objective |
| `TRANSPORT_GRADIENT` | iteration、当前参数、真实梯度/范数及缓存反向 objective |
| `TRANSPORT_TRIAL` | iteration/trial、试探参数/步长、前后目标、Armijo RHS/容差、接受标记及真实费用 |
| `TRANSPORT_STEP` | 接受更新编号 step、参数前后值、位移、梯度、步长和耗时 |
| `TRANSPORT_FIT` | 完整阶段审计，无 step 字段 |

`steps` 只保存接受更新；`gradients` 保存全部梯度事件；`trials` 保存全部完成的试探。`trial_attempt_count` 在试探求值前计入，`trial_count` 表示完成试探；技术失败的 `current_trial` 保留尚未完成的坐标与参数。完成阶段两者相等，失败不能把中途实际计算抹为零。

阶段主要计数如下：

- `optimizer_steps/accepted_trial_count/nonzero_projected_update_count`：实际接受更新数；`optimizer_iterations`：实际进入的迭代数。
- `inner_objective_evaluation_count`：真实新目标前向，不含接受缓存重用；`backward_evaluation_count`：目标反向调用。
- `inner_head_fit_count/inner_factorization_count`：实际内头尝试和实际 Cholesky；`final_head_fit_count/final_factorization_count`：额外全 support 头。
- `derivative_triangular_solve_count`：伴随三角求解，不包含前向 alpha 的求解。
- `transport_forward_evaluation_count`：核心 `_forward` 调用，包含内头与最终全 support 头，不含 query 评分。
- preparation 的 `prototype_construction_count/prototype_distance_evaluation_count/prepared_distance_evaluation_count`：原型构造、原型距离调用及原 train/cross interaction 距离调用；空 held 调用也按执行记录。

objective 的 `inner_folds[*].class_margin_loss_sums/class_physical_counts` 可独立重算类平均及 RMS。顶层 `class_loss_values/class_held_counts`、`loss_data/loss_proximal/loss_total` 明确区分监督目标与内头 `head_training_loss_*`。停止原因是 `MAX_ITERATIONS`、`ZERO_GRADIENT`、`ZERO_PROJECTED_STEP`、`ARMIJO_BUDGET_EXHAUSTED`，或无内层监督的 `PHYSICAL_K1/SINGLE_REGISTERED_CLASS`。预算耗尽不假称收敛，也不追加性能门槛。

## 成本、测试与尚未验证项

每三折阶段上界是 13 次目标前向、39 个内头、4 次反向、24 次伴随三角求解，另至多一个最终头。拒绝 trial 仍计费。80 B 参数不等于完整模型：`persistent_state_bytes` 分解为 head、adapter、prototype、lineage；prepared 另计内折特征、原型、距离和映射上下文。Python 容器、audit 日志及运行库内存不在数值 payload 字节中，峰值 RSS、部署包及传输需入口实测。优化状态没有 Adam 矩，不能据此推断总成本更低。

主任务统一运行：

```text
python -s -m pytest tests/test_d92_prototype_transport_local_ridge.py
```

合成检查覆盖非零 β/η 的全目标有限差分、零 β 精确原值和活跃梯度、q=0、原型并列、行/类置换、内折隔离、跨折 RMS、合法非单射和双距离下界、Decimal 近重复距离、零带宽、B/C 原型继承、K1/N0、缓存接受/拒绝、真实费用与失败保存。不可微点单独验证对称广义导数，不用单侧差分替代其定义。

主任务已串行执行完整核心合成测试：**27/27 通过，2.15 s**。证据为 `E:/type10-7/.codex_tmp/pytest_utf8_1790762141810528400.stdout` 及同名 `.stderr`。本 agent 只编写实现与测试、执行标准库语法编译；pytest 与数值诊断均由主任务在已验证环境执行。

首轮结果保留为 `E:/type10-7/.codex_tmp/pytest_utf8_1790761931958217100.stdout`：24 项通过，1 项在 support 自身逐样本评分时触发 `ORIGINAL_EQUAL_ADAPTED_UNEQUAL`。独立合成中间量诊断确认原特征、方向、范数、原型距离及 attention 均逐元素一致，首次差异位于 `mu=att@table.m`：批量 GEMM 与单样本 GEMV 的最大差为 `2.77555756e-17`。修复将 μ 统一为逐样本显式乘积求和，保留原相等输入必须映射相等的严格检查。诊断中修复后的 μ/b/a 逐元素相同；新增不同批量划分与重复样本回归通过。没有放宽相等条件、数值容差或有限差分要求。

另外补全了拒绝/失败 trial 的实际尝试上下文、缓存复用时折级 forward 计数及退化头的实际数值 payload 字节。27 项通过仅证明这些合成数值与协议行为；原型邻域噪声、困难类重加权及注册竞争的泛化效果仍待完整合法 support 实验验证。尚未真实拟合、预登记、启动或产生新性能结论。

## r01 后的 JSON 输出边界修复

上述“未启动”是首版核心交付时点的记录。之后主任务报告 r01 四行均因技术故障退出：首个完整 K5 parent 写出 JSON 时遇到 `np.bool_`，没有健康任务被修改。失败日志、输出和旧 release 均保留，本文不读取其性能指标。

核心确认的类型路径为 `_EPS` 的 NumPy 标量参与 Armijo 容差计算，容差得到 `np.float64`，比较得到 `np.bool_`，再进入 `state.audit.trials[*].accepted`。callback 已经过 JSON 类型归一化，所以逐步事件能写出；旧公开 `audit_dict()` 只走 `_plain`，未转换 NumPy 标量，完整 parent writer 因而失败。

局部修复只将容差返回值显式转换为 Python `float`、接受标志显式转换为 `bool`，并在 prepared/state 两个公开 `audit_dict()` 边界复用已有 `_safe` 递归输出原生 JSON 类型。没有改变浮点计算公式、参数、接受判据或冻结配置，也没有热修旧运行。入口的完整流式写出回归由入口负责人单独维护。

新增核心测试 `test_public_audit_tree_strict_json_native_including_trial_flags` 使用合成 K3 实际拟合，递归检查完整公开 audit 树并执行 `json.dumps(..., allow_nan=False)`，同时检查生产端冻结 audit 中的接受标志本身已是原生 bool。主任务已串行执行该新增回归：**1/1 通过，2.33 s**，证据为 `E:/type10-7/.codex_tmp/pytest_utf8_1790763972888069700.stdout` 及同名 `.stderr`。此前 27 项数学与确定性检查已通过，本次未修改数学；这一次新增验证专门覆盖此前遗漏的公开审计序列化边界。

入口负责人另完成真实 full/compact/text/CSV 写出和严格类型边界检查，主任务验证 15/15 通过，证据为 `E:/type10-7/.codex_tmp/pytest_utf8_1790764153519320100.stdout` 及同名 `.stderr`。r01 的真实失败记录保持原样；r02 待主任务准备恢复。上述验证不产生新的性能结论。
