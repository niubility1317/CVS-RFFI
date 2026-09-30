# PrototypeTransport-LocalRidge 核心实现审查

日期：2026-09-30。状态：`CORE_IMPLEMENTATION_REVIEW_COMPLETE`。本次是已审查设计的一次实现核对，不重复数学设计审查，不审查入口或实验编排，不产生性能结论。

## 1. 结论与证据边界

核心未发现 P0；一次实际回归暴露的 P1 确定性问题已修复并通过受影响完整核心测试关闭。失败试探与缓存前向的两项计数缺口也已修正。当前核心无未关闭 P1。

审查对象为 `code/cvsrffi/d92_prototype_transport_local_ridge.py`、`configs/d92_prototype_transport_frozen_20260930.json` 及 `tests/test_d92_prototype_transport_local_ridge.py`；必要时只读复用的 Channel 准备、绑定和伴随接口。作者确认 READY 后，仅核对新增修复段，没有重新审查已通过的数值设计。

审查者只创建本文，未修改方法、配置或测试，未真实拟合、执行 Conda/SSH/Git、读取 query 或历史评分。正式测试及合成诊断由 root 串行执行。审查者已读回实际 stdout/stderr：修复后 **27 passed in 2.15 s**，stderr 为空。该结果只证明这些核心合成检查通过；入口、编排、发布与真实 support pilot 不在此完成声明内。

## 2. 发现、修复和关闭

### 2.1 P1：相同样本的映射随批量形状产生末位差

定位：`_adapt` 的旧 `mu=att@table.m` 与 `PrototypeTransportState.score` 的逐样本路径。训练以多行矩阵计算 μ，评分以单样本计算；不同 BLAS 累加树导致同一原特征的适配特征不再逐元素相等。

root 首轮核心测试为 24 passed、1 failed。失败测试为 `test_optimizer_actual_cost_bounds_last_accepted_and_batch_independence`，在 `state.score(**p.raw)` 进入 `_joint_distances` 时触发 `ORIGINAL_EQUAL_ADAPTED_UNEQUAL`。这是确定性映射合同的真实缺陷，不能以删除等值检查、放宽断言或恢复旧双向单射限制处理。

root 的合成中间量诊断使用 seed=81：原始 x/w/rho/d/neighbors、weighted distance 与 attention 均逐元素一致；旧批量 GEMM 与逐条 GEMV 的 μ 最大差异为 `2.7755575615628914e-17`。因此问题定位到原型加权和的计算路线，而非原特征、邻域或 attention。

修复后的 `_adapt` 第 158 行统一使用

```python
for i in range(len(x)):
    mu[i] = np.sum(att[i, :, None] * table.m, axis=0)
```

训练与评分使用同一个逐样本归约；空输入由预分配空数组自然处理。`_joint_distances` 的原始相等输入确定性检查仍保留，合法适配分量非单射仍允许。新增回归覆盖批大小 1/2/3/7/全批，以及重复和置换样本。root 诊断确认补丁后的 μ、b、a 均逐元素一致；受影响完整核心测试 27/27 通过。该 P1 已关闭。

### 2.2 计数与失败试探记录

原实现只在 trial 前向成功返回后增加 `trial_count`，技术失败 trial 虽已计入 failed-stage 的目标/头预算，却没有独立的尝试次数和 trial 上下文。修复后调用前增加 `trial_attempt_count`，并保存 `current_trial` 的 iteration/trial/step_size/u_pre/u_trial；成功返回后才增加表示已完成试探的 `trial_count` 并清空 current_trial。技术失败保留未完成 trial 及已执行的部分头、分解和反向计数。新增失败试探回归覆盖这一口径。

新增 `transport_forward_evaluation_count` 时，复用缓存的折级审计一度继承原缓存的前向计数 1，而顶层计数为 0。第 372 行现已明确覆盖为 `0 if reuse else 1`，与已覆盖为零的 head/factorization 计数一致。两项修正均已局部读回，未改变损失或更新公式。

## 3. 核心实现对照

| 合同 | 实现核对 | 结果 |
|---|---|---|
| 零 β 精确原前向，β 导数活跃 | `_adapt` 保留 t/q/attention 缓存后直接返回原 x；`_backward` 不因 β=0 跳过，适配距离传 `.5*dd`、`.5*dc` | 符合 |
| 旋转小 q 和 VJP | `_rotation_vjp` 用 β·sinc 及小角 B 展开；VJP 最后一项为 t 方向乘 w 与输出伴随的内积；零块由 rho=0 消除贡献 | 符合 |
| attention、原型和并列 J | `_make_prototypes` 由该头 train 构造 p/m/ν；`_map_context` 以第二距离为阈值包含全部并列；attention 导数使用当前固定邻域，无标签路由 | 符合 |
| inner-train 隔离 | 每个 inner `_problem` 不带 inherited；完整原型只用于 full_problem，inner-held 标签只进入监督损失 | 符合 |
| 类 RMS 及近端梯度 | `evaluate_transport_objective` 先跨折累计类 sums/counts，再用 `losses/(C*data*counts)` 缩放 score 伴随；近端梯度 `(theta-anchor)/N` | 符合 |
| 同 Cholesky 伴随 | `_forward` 缓存 chol/alpha/crosskernel/径向状态；`_backward` 复用该 chol 的两次三角求解，完整经过中心化、迹缩放和带宽链 | 符合 |
| accepted cache 隔离 | trial_cache 为独立前向产物；只在 Armijo 接受后更新 theta/current；拒绝试探不替换当前缓存 | 符合 |
| 有限优化 | 最多四次归一化梯度，步长从 0.125 起最多三次试探；η 以移位截断求零和盒的欧氏投影；float64 Armijo 容差已配置 | 符合 |
| 双距离与非单射 | `_joint_distances` 组合原/适配两项并检查下界；只禁止原相等却适配不等，允许原不等而适配相等 | 符合 |
| state.score | 非恒等路径逐样本生成固定原型映射，用原/适配两份 support 计算距离，再对全部注册类输出 LocalRidge 分数 | 符合 |
| B→C 真继承 | 准备层核对旧 ID、标签、raw 特征及存在的 row 上下文；full p/m 重算相等后逐列复制 B；C_seq 复制 theta_B 为初始化和 anchor，C_reset 为零 | 符合 |
| K1、单类、N0 和零带宽 | no_information 跳过监督；N0 返回对应 B；原 tau0=0 使用原等价核且反向零；ν=0 单独令 adapter 恒等 | 符合 |

冻结配置的十个存储参数、九个约束自由度、0.5 原距离权重、RMS 目标、四迭代/三试探、投影和 Armijo 容差与当前核心一致。配置精确对照包含在通过的核心测试中。

## 4. 实际费用与资源表述

成功阶段的目标前向上界为一次初始前向加最多 12 次 trial，即 13；三折最多 39 个 inner-head，四次反向最多 24 次伴随三角求解。最终全 support 头另计。末尾取得 final_objective 时复用已接受缓存，没有新增前向、头拟合或“最好步”搜索。真实拒绝试探与技术失败的部分工作都保留费用；零梯度、零投影位移或回溯预算耗尽只保留已接受参数，不扩展预算。

`transport_forward_evaluation_count` 是折/最终头的 transport 前向次数，不等于完整三折目标调用次数；`inner_objective_evaluation_count` 才表示目标前向次数。伴随三角求解数不包含前向求 alpha 的两次三角求解。审计已将 head-fit 与实际 Cholesky factorization 分开，退化头不能冒充实际分解。

持久数值字节计数包含原/适配 support、raw/标签继承信息、alpha、中心化、theta 与原型，明确排除 Python 元数据。`optimizer_state_bytes=0` 表示无持久动量状态，不表示目标缓存占用零字节。完整训练时间、评分时间、峰值 RSS/显存、部署包及传输字节仍需后续 pilot 按实际硬件、线程和测量口径报告；核心合成测试不能支持“更省算力”或达到10/1/3 pp理想目标的声明。

## 5. 验证证据

正式测试由 root 执行，审查者未另开一次测试：

- 首次失败日志：`E:/type10-7/.codex_tmp/pytest_utf8_1790761931958217100.stdout`，24 passed / 1 failed；对应 stderr 同名。
- 修复后日志：`E:/type10-7/.codex_tmp/pytest_utf8_1790762141810528400.stdout`，27 passed in 2.15 s；对应 `.stderr` 已读回为空。
- root 合成诊断脚本：`.codex_tmp/diagnose_prototype_map_batch.py`，逐层核对批量与单样本输入、attention、旧/新 μ 和映射；上文诊断数值来自 root 的实际执行回报。

通过的核心合成检查覆盖全链有限差分、零 β/小 q/零块、类 RMS 跨折聚合、原型/内折隔离、并列与类别/行置换、稳定近重复距离、零带宽、投影、原/适配联合下界、缓存拒绝、计数失败状态、B/C继承及K1/N0。此次关闭仅适用于这些已执行的核心检查。入口/orchestration 仍由 root 完成其独立验证；未运行真实 pilot，未访问 query，无适应收益或性能结论。
