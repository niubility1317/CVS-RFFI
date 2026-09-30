# MC-Residual8 核心实现合同

状态：IMPLEMENTED_SYNTHETIC_CHECKS_VERIFIED，核心与 driver 共 38 项不同合成检查通过。本次只实现[冻结设计](D92_JOINT_AFTER_PROTOTYPE_DESIGN_20260930.md)，尚未进行真实 N607 拟合、参数扫描或产生新运行结论；此前方法的状态与产物保持原样。

## 文件和权限

本次四个核心产物为 `code/cvsrffi/d92_margin_constrained_residual8_local_ridge.py`、`configs/d92_mc_residual8_frozen_20260930.json`、`tests/test_d92_margin_constrained_residual8_local_ridge.py` 和本文。配置外层为 `algorithm`，内容与核心 `FROZEN_CONFIG` 完全相同。

只使用已构造的五块合法目标 support 缓存，Phase1/encoder/BN 不更新。没有源样本、逐样本源特征、地面原型包或 query 拟合入口。旧 Residual8 草稿及其他方法源码未修改。当前复用 Channel/PrototypeTransport 的规范输入、原头标签绑定、稳定 interaction 距离、固定零带宽语义等函数；新残差和两条目标的反向在本核心实现。

本次表征确实训练 U/V 共 11776 个参数，LocalRidge 是唯一最终分类器；教师只提供训练常量。现有论文可以支持可微闭式头、参数高效残差和功能保持等一般思路，但**类别 RMS、κ=0.25、半份原几何以及物理 slack 的具体组合是本项目设计假设**。不能写成论文已经证明注册旧类下降≤1 pp或新类必然改善。有限差分验证导数实现，不验证泛化收益；原始论文、LoRA/adapter/ReFT 的区别及推导边界见[联合微调数学依据](D92_JOINT_FINETUNING_MATH_FOUNDATIONS_20260930.md)。

## API 与状态

```python
prepare_mc_residual8_training(
    *, z_id, fft, t_emb, f_emb, pa_local,
    support_labels, support_ids, classes, old_classes,
    inherited=None, context=None, log_callback=None, state_callback=None,
) -> MCTraining

evaluate_mc_objective(
    prepared, U, V, anchor_U, anchor_V,
    *, gradient=True, forward_cache=None,
) -> (loss, (g_U, g_V), (keep_g_U, keep_g_V), audit, forward_cache)

fit_mc_residual8_local_ridge(
    prepared, *, mode='B', baseline_state=None,
    log_callback=None, state_callback=None,
) -> MCResidual8State
```

模式为 `B`、`C_seq`、`C_reset_init`。部署 state 的 `U/V/raw/labels/ids/classes/base_state` 与物理绑定保持一致；U/V、原 support 和 raw 数组均只读。`score(**five_blocks)`、`predict(**five_blocks)` 每次对单个样本独立映射，并对全部注册类评分。`audit_dict()` 递归返回原生 JSON 类型，没有将 ndarray 或 NumPy bool 留给调用方。

`state_callback(key, arrays)` 接收只读的完整数值数组，返回入口定义的 JSON 引用。核心附上每个数组的 shape/dtype/nbytes；入口以不可覆盖 NPZ 保存，不允许 `default=str` 或 NaN。无 callback 时，核心保留只读记录，`state_records()` 返回其副本，便于合成测试和调用方自行导出。外置成功时核心不再保留重复向量；`retained_vector_record_bytes` 明确区分这种情况。

主要状态引用为：

| 引用 | 数组 |
|---|---|
| `initialization_state_ref` | U、V、anchor_U、anchor_V |
| `gradients[*].state_ref` | U、V、g_U、g_V、keep_g_U、keep_g_V、d_U、d_V |
| `trials[*].state_ref` | 该次试探 U/V，包括拒绝试探 |
| `steps[*].state_ref` | 已接受 trial 的同一引用 |
| `final_state_ref` | 最后接受 U/V 与阶段 anchor |
| preparation `teacher_folds[*].state_ref` | teacher_scores、teacher_q、held_old_mask |

梯度、方向、初始/最终及拒绝试探均可重建；JSON 日志只含标量、类别风险与引用。保存失败的状态引用和最后接受态，而不是只写错误字符串。

## 数学路径与教师隔离

原 b/a 只构造一次。各子块固定单位方向组成 x，U/V 使用 exact-erf GELU，切向残差按 κ 饱和后重新保持原块范数。U=0 的前向直接返回原 b/a bits，但保留活跃 U 导数及初始为零的 V 数据梯度。训练、教师和 score 都按同一逐样本乘加路线执行，避免批量 GEMM 与单样本 GEMV 的舍入差异。

每个头使用 `.5*d_original²+.5*d_adapted²`，带宽来自其当前训练距离，迹匹配到该头原始 interaction s0。反向包括带宽、中心化、迹缩放和闭式 ridge；只有适配距离链乘 0.5。原等价输入必须映射相等，适配分量允许将不同原点合并，联合距离保留原距离一半的下界。非有限、未分辨正带宽、切向/范数或正规方程残差异常报技术失败，保存上下文。

B 的 prepare 真正拟合首批 U=0 内头，用其分数冻结 q，并保留前向缓存。fit B 的初始目标复用该缓存，没有第二次拟合。C 的 prepare 对每个内折仅用旧 inner-train、冻结的同 row B 参数建立教师，再对旧 inner-held 产生 q；不能借 B 完整旧头。该 prepared 教师可由 C_seq/C_reset_init 共享，但两个阶段分别按自身 anchor 计算保持上限。

教师正确且正 margin 时 q 保存其截断 margin；错误或非正 margin 时 q=0，仍只要求当前 true-class margin 不为负。保持风险面对当前全部注册类的最大错误分数。类风险先跨所有内折汇总，再除以每类物理数并做 RMS，不等折不等权平均。任务风险与保持风险分别反向；近端只进入总梯度，不进入保持梯度。

C_seq 核对旧 ID、标签、原 raw 值和 row/fold/proxy 绑定，再复制 B 的 U/V 为初始化和 anchor，不修改 B。C_reset_init 仍使用同一 B 教师；它改变初始化、近端 anchor 及起点保持上限，不能称为完全无 B 信息的对照。single-class 教师没有错误类 margin，保持项整体关闭。K1 不做内层训练；当前平衡 K1 的 U=0，所以保持 R0。Nnew=0 直接复用 B，不建立 C 教师。

## 优化和审计

每阶段最多四次迭代、每次最多三次试探。方向先用真实总梯度归一化，再按保持梯度投影到当前风险预算的线性半空间；投影后不再归一化。U 与 V−V0 分别投影到 Frobenius 单位球。所有接受都同时满足总目标 Armijo、总目标不增和真实保持风险上限；线性方向检查不能替代真实试探。

独立数学审查发现球投影后的实际位移可能使 `g·Δ>0`，即使投影前 `g·d<0`；仅有 Armijo 会允许小幅总目标上升。因此接受总目标的界明确为 `min(L_current,L_current+1e-4*g·Δ)+tol_L`，保持原比较容差。核心和冻结配置已记录这一正确性修正，不改变步长、预算或目标权重；合成反例从单位球边界、合法半空间方向出发，验证原 Armijo 通过而新增不增条件拒绝。日志分别保留 `armijo_pass/objective_nonincrease_pass/keep_pass/rejection_reason`。

保持上限为阶段 anchor 风险加 `1/(2k*sqrt(C_prev))`，数值比较容差为 `128*eps64*max(1,比较量绝对值)`。这是固定训练定义，不是实验许可或外层表现门槛。它不等于最多错一条，更不保证≤1 pp遗忘。若三次试探都拒绝，保留最后接受状态并记录 `ARMIJO_OR_KEEP_BUDGET_EXHAUSTED`，没有加步数、择优外层步或切换主候选。

事件为 `MC_INNER_PREPARED/MC_INITIAL/MC_GRADIENT/MC_TRIAL/MC_STEP/MC_FIT`。每次梯度事件记录 g/a 范数、方向范数、半空间前后内积/界及状态引用；trial 记录两个独立接受标志、实际违反量、拒绝原因、步长、完整数组引用和费用。停止原因还包括 `ZERO_GRADIENT/ZERO_GUARDED_DIRECTION/ZERO_PROJECTED_STEP/MAX_ITERATIONS/PHYSICAL_K1/SINGLE_REGISTERED_CLASS`。

费用计数必须同时累计 preparation 和各 stage：

- B 的 `initial_inner_head_fit_count/initial_inner_factorization_count` 是 preparation `inner_head_fit_count/inner_factorization_count` 的解释分项，不再额外相加。它们是真实执行的首批内头，fit 初始目标复用时头计数为零。
- C 教师使用 `teacher_head_fit_count/teacher_factorization_count/teacher_score_evaluation_count/teacher_score_physical_count`；共享 prepared 只计一次。B 教师读取首批缓存也记录实际 teacher score 次数和物理数，没有额外教师头。
- stage 的 `inner_objective_evaluation_count` 统计初始目标与每次真实 trial；接受缓存反向和最终缓存汇总不新增前向。`inner_head_fit_count/inner_factorization_count` 只记录新增求解。
- `task_derivative_triangular_solve_count` 与 `keep_derivative_triangular_solve_count` 分别记录两条目标的伴随，总和为 `derivative_triangular_solve_count`。这是伴随求解，不包括前向 alpha 的三角求解；退化跳过按实际计零。
- `mc_forward_evaluation_count` 覆盖真实核心前向头调用；`prepared_distance_evaluation_count` 覆盖准备中的 student/teacher train/cross 原距离调用。`final_head_fit_count/final_factorization_count` 单列最终全 support 头。
- `trial_attempt_count` 在求值前增加；`trial_count` 只计完成试探，失败保存 `current_trial`。中途伴随失败保存已完成 task 与部分 keep 通道的实际求解数。

每三折阶段至多 39 个内头、48 次伴随三角求解、一个额外最终头。C 教师另计，B 首批预付仍属于该阶段内头预算。936 个信息阶段的设计上界仍为 42336 个含基线/教师/最终的总头、44928 次伴随三角求解；这是上界，不是运行实测。

## 资源与验证边界

参数 U/V 为 94208 B。`persistent_state_bytes` 只统计部署所需 head、U/V、原/适配 support、raw 及标签；教师、优化缓存、完整轨迹记录和 Python 对象另计。prepared 记录其数组字节、初始前向缓存及保留教师记录；objective 记录缓存数值 payload。部署包、实际峰值 RSS/显存、训练与推理耗时、增量传输由入口实测，未测项为 N/A。无 Adam 矩和少于 encoder 的参数量都不足以证明总资源更省。

建议主任务串行运行：

```text
python -s -m pytest tests/test_d92_margin_constrained_residual8_local_ridge.py
```

主任务已串行执行首轮核心与 driver 合成检查：**36/36 通过，2.69 s**，证据为 `E:/type10-7/.codex_tmp/pytest_utf8_1790770662322837500.stdout` 及同名 `.stderr`。其中已包含球投影后实际位移使梯度内积为正、必须额外拒绝总目标上升的回归。此前通过部分没有重跑。

随后只补两项明确要求的检查：类别重命名和物理行置换下的两风险/梯度/教师 q 等变；真实非零残差对近重复样本的逐样本一致性及 90 位 Decimal 显式 interaction 双距离 oracle。主任务已定点验证 **2/2 通过，0.64 s**，证据为 `E:/type10-7/.codex_tmp/pytest_utf8_1790770803798577400.stdout` 及同名 `.stderr`。核心数学没有变更，累计为 38 项不同的核心与 driver 检查，没有重复计入原 36 项。

源码和合成测试语法检查由本 agent 完成；没有由本 agent 执行数值测试、Conda、真实拟合、SSH 或 Git。测试覆盖两目标全链有限差分、恒等活梯度、块范数/角度/零块、批量划分、跨折 RMS、教师隔离、真实继承、物理 slack、方向与接受条件、完整向量回调、计数和失败状态。任何导数测试通过均不构成性能改善证据。
