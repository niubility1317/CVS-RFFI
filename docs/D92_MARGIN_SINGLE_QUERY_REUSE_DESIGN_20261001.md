# D92 Margin 单样本 B prior 分数复用设计

## 1. 源码结论与范围

当前非 new0 推理会重复计算实际 B 分数。入口先对同一份单样本五分支输入调用 `b.score_with_audit`，再调用 `c.score_with_audit`。C 内部先计算自身 residual，然后通过 `self.prior` 再计算一次 B，最后将 B 的旧类列加入 C。安全的计算优化是复用刚返回的原始 B 分数，并保留 C residual 的完整计算；训练方法、模型状态、预测列与决策规则不变。

本文件仅由冻结源码推导未来实现条件，不实现接口，不修改既有脚本或运行产物。调查没有读取真实 query、truth、分数、缓存、权重或结果，也没有执行数值测试。本文不声称准确率改善、已测耗时收益或星载可行性。

后续实现更新：root 已按本文条件完成可选接口和 22 项合成验证，详见[实现与验证](D92_MARGIN_SINGLE_QUERY_REUSE_IMPLEMENTATION_20261001.md)。本文保留静态推导范围，源码链接已对齐新增接口后的当前位置；当前远端冻结实验没有接入该接口。

| 核实项 | 源码位置与现有行为 |
|---|---|
| 同一样本的 B→C 调用 | [query 入口](../tools/evaluate_d92_margin_joint_benchmark.py#L235)第 235 至 255 行：`one` 使用同一物理行，B 在第 245 行返回 `b_values/b_work`；非 new0 的 C 在第 250 行另行评分 |
| C 的重复 B 计算 | [MarginJointState](../code/cvsrffi/d92_margin_joint_local_ridge.py#L901)的 `_score_geometry` 先调用 C 的 `_score_residual`，第 905 行再调用 prior 的 `_score_residual`，第 906 至 907 行按类名逐列相加 |
| prior 是实际 final B | [C preparation](../code/cvsrffi/d92_margin_joint_local_ridge.py#L545)第 545 至 563 行核对继承来源、物理旧 ID、标签与原五分支特征；第 586 至 588 行使用实际 B 的 problem、U、final cache 和 final ref；[state 构造](../code/cvsrffi/d92_margin_joint_local_ridge.py#L1163)第 1163 至 1164 行保留 `prepared.inherited` |
| 入口绑定同一 B | [stage 入口](../tools/evaluate_d92_margin_joint_benchmark.py#L205)第 205 至 233 行将同 split 的 B 对象传给 C，并比较 C 的 final prior ref 与实际 B 的 final ref |
| new0 已经复用 | [query 入口](../tools/evaluate_d92_margin_joint_benchmark.py#L247)第 247 行在 `c is b` 时直接令 C 使用 B 分数；不增加 C score call |
| 分数不是重新拟合 | [固定 residual 评分](../code/cvsrffi/d92_margin_joint_local_ridge.py#L456)第 456 至 488 行只有距离、kernel、已有系数乘积与常数项；没有 QP、factorization、solve 或参数更新 |

## 2. 等价关系与数值边界

令 B 的注册列为旧类集合，C 的列为全部注册类，`P` 按类名将 B 的列嵌入 C，新增列置零。对同一物理样本 `x`，冻结模型的关系为

\[
f_C(x)=P f_B(x)+r_C(x),\qquad
f_B(x)=\gamma_B\widetilde{k}_B(x)\alpha_B+b_B,\qquad
r_C(x)=\gamma_C k_C(x)\alpha_C+b_C.
\]

`b_B` 是 B 的自由截距，`b_C` 是 C residual 的自由常数。式中的 kernel 项在对应 gamma 为 None 时取零，与源码的常数分支一致。B 使用既有中心化 cross kernel，C residual 使用原始 radial cross kernel；这两种几何不能互换。具体计算仍以 [`_score_residual`](../code/cvsrffi/d92_margin_joint_local_ridge.py#L456)和 [`_center_cross`](../code/cvsrffi/d92_affine_joint_local_ridge.py#L167)为准。C score 中重复的 B 调用使用 B 自己的 U、支持集、中心化参考与系数，属于相同函数求值，并非用 C 的 adapter 近似 B。

源码可证明上述数学恒等。若同一数值后端对固定输入与固定 B 状态的重复评分是确定的，则先前的 `b_values` 与 C 原本重算的 prior 分数逐位一致。未来实现可保持 C 当前的浮点加法顺序：先生成完整 `r_C`，再依次执行 `out[:, C_index(B_class)] += b_values[:, B_col]`。新增列保持原 residual 值，避免为零填充列额外作一次加法；不将截距、矩阵乘积与 prior 合并重排。

逐位一致仍需未来用合成输入在声明的数值环境下验证。单凭数学恒等不能保证不同 BLAS、线程策略、批量形状、精度或求和顺序之间的逐位一致。本设计不允许将已有 JSON 分数、float32 副本、四舍五入结果或仅有 argmax 的预测作为 prior。B 返回的原 float64 数组须只读消费，C 输出使用自己的新数组，不能就地修改 B 输出。

## 3. 安全复用的精确条件

未来实现宜采用单样本的成对评分路径，在内部持有 B 的对象和刚返回的分数，不建立跨样本全局 score cache，也不增加 hash、receipt 或重新验证数据的流程。以下条件是该复用自身的直接正确性要求。

| 条件 | 必须保留的含义 |
|---|---|
| 同一实际 B 状态 | 最直接的条件是 `c.prior is b`，且 B 为已完成的 old-only 状态、`b.prior is None`。若未来改为状态重载，必须证明完整冻结数值状态相同，不能仅凭相同 checkpoint、seed 或 U 判断 |
| 同一调用输入 | B 和 C 使用同一份五分支单样本数组，物理 ID、行数、顺序与内容不变。ID 只是位置绑定，不能代替输入内容相同；复用期间不能修改输入 |
| 同一上下文 | 同 run、row、split、scope 与 final state。不能从其他 cohort、fold、inner prior、试探 adapter、其他方法或其他 run 借用 B |
| 完整原分数 | 使用模型刚返回的有限 float64 `b_values`，形状为 `1 × len(b.classes)`，不是序列化预测文件。空批量或批量扩展不属于本次单样本设计 |
| 列对应完整 | B 类名唯一，全部属于 C 注册类；沿用当前按类名映射的顺序。原 B 分数需包含全部旧类列，不能按 query 的 truth、role、类别数量或配额截取 |
| C residual 不省略 | 每个样本仍计算 C 的固定 residual，包括其支持端几何、alpha 与自由常数；C 的所有注册类统一竞争 |
| 固定性不变 | 分数固定后才输出预测，后续独立 truth join 不进入复用判断。A 原 Ground 头与 A 的评分路径不受影响 |

“相同 U”不足以确认相同 B：支持物理 ID、支持值、alpha、intercept、中心化参考、tau、gamma 或类序任一差异都会改变分数。相反，C 的 U 是否等于 B 的 U 并不妨碍 prior 分数复用，因为 prior 始终使用冻结 B 自己的 U。

本次只消除 C 内的重复 prior 评分。不复用 C/B cross kernel，不合并 adapter，也不修改 batch 大小。C residual 支持端包含新增样本、列数不同且使用原 kernel，不能把 B 的 cross kernel 或 B 的预测直接当作 C residual。

## 4. new0 与退化分支

| 分支 | 复用结论与限制 |
|---|---|
| new0 | preparation/fit 精确返回实际 B，入口已在 `c is b` 时复用 B 分数。这一分支没有额外 C prior call 可消除；维持当前 C score call 为 0、`C_work=None` 的口径 |
| K1 | 无 adapter 更新不等于无 C residual。新增类存在时仍完成 full-support margin head，必须保留其 kernel 项与自由常数；只能复用其中的 B prior |
| rank0 或 no-information | 同样仍有最终完整 head。不能从零维坐标或未更新 U 推断 `r_C=0` |
| tau=0 | 使用原完整特征的严格等价 kernel，[radial helper](../code/cvsrffi/d92_branch_local_ridge.py#L99)按距离是否严格等于 0 判断。复用 prior 不改变这一规则，不添加近似阈值，也不缓存近邻结果 |
| gamma=None | 当前 residual score 仅广播自身截距/常数，不计算 cross kernel。C 的自由常数仍可能非零；不能直接令 C 分数等于 padded B |
| gamma=0 且非 None | 当前实现仍进入距离/kernel 分支，之后乘以零。复用只去掉重复 prior 调用，不另加零 gamma 快路径；实际 residual 工作继续如实计数 |
| prior 或 residual 失败 | B 尚未成功返回时没有可复用分数。C residual 失败时保留已完成 B 工作与失败 C 工作边界，不能输出完整 C，也不能回退到 B 冒充 C |

K1、rank0、tau=0、gamma=None/0 均不是跳过 C head 或省略其常数项的依据。其数学与训练状态遵循冻结 core，本调查不改变退化分支。

## 5. 计算与审计口径

对非 new0 的一个样本，设 `W_B` 为一次实际 B 评分工作，`W_r` 为一次 C residual 评分工作，`W_add` 为旧类逐列相加。当前与拟议路径的关系是

\[
W_{\mathrm{current}}=W_B+(W_r+W_B+W_{\mathrm{add}}),\qquad
W_{\mathrm{reuse}}=W_B+(W_r+W_{\mathrm{add}}).
\]

因此可消除的是一次重复 B prior 求值，而不是整个 C score，更不是一次 QP 或训练 solve。A 评分、C residual、输出文件与完整预测核对都仍需执行。只有 C 分数的部署场景也需要计算 B 一次作为 prior；该场景没有独立 B 输出可免费复用，不能照搬 A/B/C 配对基准的节省判断。

当前 [`_score_residual` 的计数](../code/cvsrffi/d92_margin_joint_local_ridge.py#L476)是执行分支计数。令 B 的物理支持行数为 `m=6K`，对单样本且 `gamma_B is not None`，被消除的重复 prior 具有

\[
d_B=1+\mathbf 1[\tau_B\ne0\ \land\ U_B\ne0],\quad
\text{raw distance pairs}=d_Bm,\quad
\text{kernel pairs}=m.
\]

相应 raw distance evaluation 为 `d_B`、kernel evaluation 为 1；reference distance 指标是 raw 的子集，不可再次相加。当前实现只要 tau 不为 0，就调用一次 query adapter/context，即使 U 为零；因此其 adapter 与 dictionary 计数仍按源码为 1。默认 H 未传入，dictionary 在 context 中实际计算，见 [dictionary/context](../code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py#L77)和 [sample-wise adapter](../code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py#L135)。若 `gamma_B is None`，上述距离/kernel/adapter 工作为 0，但重复的 B 常数输出与 `intercept_addition_count=6` 仍存在。

未来审计应分开记录新执行的工作和被复用的结果，不把 `b_work` 再计入 C 的 actual work：

- 非 new0：B score call 为 1，C residual/composed score call 仍为 1；C 内 fresh prior score call 为 0，reused prior physical count 为 1。C 的 actual geometry/kernel/adapter 计数仅来自自身 residual，旧类合并属于另列的 composition 工作。
- `B_work` 由 B 拥有一次。C 可以保留对该已完成 B 结果的复用标记与对应状态/样本位置，但不复制它为第二份已执行 prior 工作。SUM 汇总按唯一实际调用；内存 peak 仍按实际测量取 MAX。
- `score_seconds` 使用未来 paired/C 调用的实际计时。现有 C 内的 `residual`/`prior` 子计时与外层总计时不能机械相减得出新耗时；也不能用理论省去的 B 次数猜峰值 RSS、能耗、传输或百分比收益。
- 失败时先前 B 的 returned-call 工作仍保留，未完成 C 的实际工作另列或记 unknown。只有 C 完成才产生完整 C 分数。已有运行、marker、计数与归档保持原样，不追溯改写。

B 分数已经是入口为 B 输出保留的 `1 × 6` 数组，复用可以只延长这份数组在当前样本内的使用，无须新增全 query 分数缓存。不过真实内存还包含 residual scratch、immutable state、BLAS workspace、Python 对象和输出归档；不能据此宣称整进程峰值不变或下降。资源解释继续遵循[既有资源边界](D92_MARGIN_JOINT_RESOURCE_BOUND_20261001.md)。

## 6. 未来验证范围

若 root 后续授权实现，必要的合成验证应由冻结生产 core 生成 B/C 状态，并与原 `score_with_audit` 路径比较完整分数、列序与最终预测。覆盖普通新增类、new0、K1、rank0、tau=0、gamma=None/0、非零 B 截距与 C 常数；篡改 B 对象、输入行、类映射和来源上下文应拒绝复用。还应检查 B 输出未被 C 修改、失败记录保留先前 B 工作，以及实际计数仅减少一次 prior 求值。

本设计没有执行这些数值验证，也未测量实际推理收益。当前交付只完成源码位置核实、数学等价条件、退化分支与诚实审计口径的静态推导；不授权实现、参数搜索、重新训练、停止或重启既有任务。
