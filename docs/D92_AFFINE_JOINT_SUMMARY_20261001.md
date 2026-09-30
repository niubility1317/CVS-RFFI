# D92 AffineJoint 独立 support 汇总器

本文件说明汇总器的核验范围、接口和证据边界，不包含真实 run、query 或外层成绩，不冻结下一候选，也不表示候选已发布。

实现：[汇总器](../tools/summarize_d92_affine_joint_probe.py)，合成检查：[测试](../tests/test_summarize_d92_affine_joint_probe.py)。数学依据：[截距审计](D92_AJLR_INTERCEPT_MATH_AUDIT_20261001.md)和[实现映射](D92_AJLR_AFFINE_IMPLEMENTATION_MAP_20261001.md)。实际字段来自[核心](../code/cvsrffi/d92_affine_joint_local_ridge.py)和[入口](../tools/evaluate_d92_affine_joint_probe.py)。

## 输入和完成条件

命令由唯一 launch owner 在全部 parent 完成、完整输入权限与绑定核实后执行：

```text
python -X utf8 tools/summarize_d92_affine_joint_probe.py --spec <frozen-spec.json> --run-root <complete-run-root> --output <new-output-directory>
```

Python API 为 `summarize(spec=..., run_root=..., output=...)`。`spec` 可为已加载的字典；`run_root` 默认取冻结 spec。输出目录必须不存在。汇总器只读输入，只在新的输出目录创建派生小文件，不写回归档，不创建或复拟合缺失分类头。

方法和边界固定为：

| 字段 | 值 |
|---|---|
| algorithm schema | `d92_affine_joint_local_ridge_v1` |
| method | `D92-AffineJointLocalRidge-v1` |
| marker status | `AFFINE_JOINT_PROBE_COMPLETE` |
| summary status | `COMPLETE_AFFINE_JOINT_PROBE_VERIFIED` |
| scope | `SUPPORT_ONLY_AFFINE_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION` |
| paths | `R0`、`R_AFFINE_seq` |
| numeric archive | `d92_affine_joint_state_archive_v1` |
| artifact inventory | `d92_affine_joint_artifacts_v1` |

在打开第一条 outer support-held trace 或数值特征前，先检查全部 4 row 的启动记录、完成 marker、producer/capsule/checkpoint 来源绑定、无 query/source 访问、完整文件清单和 `COMPLETE` 数值归档 manifest。任一 row 缺失、不完整或绑定错误均失败。启动记录和 marker 的 `run_id`、`row_id` 必须分别等于 `spec['run_id']` 和当前 row；不增加哈希、receipt、authority 链或新数据重验。

完整覆盖为 4 row × 40 parent = 160 parent、1800 条顺序路径。旧类为 6 个；`K ∈ {1,5,10,20}`，新增类数为 `{0,2,5,10,20}`。严格保留冻结 spec 的 row、receiver、scene、model seed 和 cohort 选择。原有 `ajlr_preparation_count`、`ajlr_stage_count`、`ajlr_forward_evaluation_count` 等兼容计数名继续使用，方法 schema 和状态不能因此混用。

## 独立数值核验

汇总器不调用 AffineJoint 的拟合函数、目标函数或反向实现。它从原始归档数组重算核、得分、损失及梯度链。R0 的原闭式头保持原规则。

对每个 AffineJoint 头，令 `E = Y − M`、`Y = onehot − 1/C`、`A = I + K`、`e = ones(n)`。检查：

\[
A z=e,\quad s=e^Tz>0,\quad F=\alpha+zb^T,\quad AF=E,
\]

\[
b^T=\frac{e^TF}{s},\quad A\alpha+eb^T=E,\quad e^T\alpha=0.
\]

`intercept` 为 `(C,)`，`schur_z` 为 `(n,)`，`schur_s` 为零维标量，`combined_rhs` 为 `(n,C+1)`，最后一列严格为 `e`。还检查 `n/(1+tr(K)) ≤ s ≤ n`，采用核心既定数值容差，不放宽 normal-equation 条件。

完整函数必须为 `M + K alpha + b`，持出函数必须为 `M_held + L alpha + b`。样本维度的 `e^T alpha = 0` 与类别维度的 `alpha 1_C = 0`、`b^T 1_C = 0`、每行 score 类别零和分别核验。固定旧参考测度满足 `q^T K = 0`，但完整旧参考均值是 `q^T M + b`；不能将它要求为零。

数据损失重算为 `0.5 ||train_scores − Y||²`，ridge 损失为 `0.5 <alpha,K alpha>`，截距不受 ridge 惩罚。旧无截距头的正 SSE 下界不作为本方法正确性条件。

每个 C 内层 prior 只用当前合法旧 inner-train 的物理记录和当前实际 `U_B`。最终 C 保存当前实际 final B 的全部字段，逐数组检查 `prior_B_*`，包含 `prior_B_intercept`。新类 prior 列严格为零。不得继承其他 run、row、parent、fold 或 proxy path 的 B。

## 完整伴随和 CE/prox 梯度

所有 physical inner-held 的 CE 按类别跨 fold 累加、求类别均值，再做类别 RMS。监督标签参与适配器训练，不能称为独立验证。温度固定为 1，目标为 `RMSCE + 0.5 ||Z||²`。

对风险给 score 的导数 `G`，归档伴随必须满足：

\[
g_b=e_h^TG,\quad A T+e\eta^T=L^TG,\quad e^TT=g_b,
\]

\[
V=T+z\eta^T,\quad \eta^T=(e^TV-g_b)/s.
\]

分别核验 `fold_j_adjoint_T`、`adjoint_eta`、`adjoint_g_b`、`adjoint_rhs` 的形状、数值和实际 C 列 RHS。不能将伴随样本和错误地固定为零，也不能漏掉 `g_b = sumRows(G)`。

令 `P_q = I − e q^T`，先独立执行完整 centering VJP：

\[
\bar L=G\alpha^T,\quad\bar K=-\operatorname{sym}(T\alpha^T),\quad
\bar Q=\gamma\bar L P_q,
\]

\[
\bar R=\gamma\operatorname{sym}\{P_q^T\bar K P_q-q(e_h^T\bar L P_q)\}.
\]

再核验其在两组 affine 约束下的简化结果，继续通过固定 `tau/gamma` 的 Gaussian、混合距离中实际 adapted 权重 `0.5`、完整 interaction-distance VJP 和固定 DCT/GELU 切向归一化 adapter VJP 回到原始 U。最终独立检查 `g_Z = g_U W + Z`，没有 `/N`，不反向进入冻结 prior。

完整四次迭代、每次最多十二个试探、归一化负总梯度、固定回溯步长、Armijo 与 objective nonincrease 同时通过、首个可接受试探、last-accepted cache 和停止原因逐条核验。拒绝试探的拟合及工作量全部计入。

## 工作量和状态字节

新增 13 个计数完整累加到 stage、parent、row 和总覆盖：

| 计数组 | 字段 |
|---|---|
| head | `head_triangular_rhs_count`、`head_triangular_rhs_element_count`、`head_triangular_dense_work_unit_count` |
| derivative | `derivative_triangular_rhs_count`、`derivative_triangular_rhs_element_count`、`derivative_triangular_dense_work_unit_count` |
| prior | `prior_triangular_rhs_count`、`prior_triangular_rhs_element_count`、`prior_triangular_dense_work_unit_count` |
| intercept | `intercept_fit_count`、`intercept_addition_count`、`prior_intercept_fit_count`、`prior_intercept_addition_count` |

每次实际 triangular call 按 RHS 列数 `r`、元素数 `n r`、稠密工作单位 `n² r`计数。前向两次调用均为 `C+1` 列，完整伴随两次调用均为 `C` 列，旧类 prior 为 `6+1` 列。工作单位是计算代理，不能标为实测 FLOP。基线 primal 与 EDF 求解分别保留原计数。reference pairs 属于 raw-distance 工作的子集，不能再次相加。

adapter 的 `736 × retained_rank` optimizer 坐标与闭式分析参数分别报告。自由截距有 C 个存储参数、C−1 个类别对比自由度；分类系数为 `n C`，分析头参数总数为 `(n+1) C`。

独立按当前核心实际不可变 buffer 形状核验 persistent、deployment、训练坐标及 head 字节。`persistent_state_bytes` 包含训练坐标、Schur/RHS、头缓存和实际 B。`deployment_numeric_state_bytes` 排除训练专用 Schur/RHS/坐标，保留截距与 prior；C/B 共享 V0 只计一次。兼容字段 `deployment_C_state_bytes` 实际承载 persistent 上界，不能称为最小部署常驻字节。归档文件和冻结 Phase1 另外报告。

prior 和 residual 外层推理分别保留距离、kernel、adapter、dictionary、intercept addition 及实测时间。CPU/RSS、耗时和线程来自归档记录。GPU、部署包、新增传输字节及单独 kernel 时间未测时为 `null/N/A`，不凭小参数量宣称省算力。

## 边界和输出

`tau = 0` 且 `gamma > 0` 使用原始完整 feature-equivalence PSD kernel，仍有 Cholesky、C+1 RHS 和闭式截距，adapter 距离导数为零。`gamma = None` 为真正零 kernel，`b = mean(E)`、`alpha = E − e b^T`，没有 factor/triangular calls；不得混为同一分支。

真实 K1 和 rank0 保留合法完整头但不执行监督 adapter 更新。真实 K1 没有独立持出分数，记 N/A。proxy trainK1 是 parent 内全 anchor 诊断，不能替代真实 K1 持出。new0 精确复用当前 B，不额外拟合 C residual 头。

生成 `summary.json`、四种完整统计 CSV（overall、K×新增类数、receiver/scene、model/cohort）、四种完整资源 CSV（另含 row）、`training_objectives.jsonl`、`training_objectives.csv` 和含完整 K×新增类数表的 `report.md`。保留 accuracy、margin、winner/correctness 转换、OOF 按每个物理 held ID 一次、proxy 先 parent 内平均后 parent 等权。A 和 B−A 始终 N/A，不以 R0 代替地面 A。

读取真实 callback 流，核验 `BASE_FIT B0 → BASE_FIT C0 → AFFINE_PREPARATION B → CANDIDATE_FIT B_AFFINE → AFFINE_PREPARATION C → CANDIDATE_FIT C_AFFINE_seq`。new0 对省略的 C 使用精确复用规则。完整训练事件 `AFFINE_JOINT_*` 与各阶段归档逐条绑定，checker 不制造事件自证。

大数组继续引用原 NPZ，汇总器不复制矩阵到 CSV/JSONL。若完整 final C Z0 头没有归档，registration-only 头改变与后续 adapter 改变的 outer-held 因果分解为 N/A；inner Z0 头属于自己的监督持出集，不能替代它，也不进行新增拟合。

合成测试覆盖正常完整复算、真实 callback、截距/Schur/RHS/伴随/梯度/当前 B 绑定篡改拒绝、positive-equivalence tau0、零 kernel K1、new0 精确复用、分析参数与实际 buffer，以及不完整 run 不打开 held trace。作者只做 AST、UTF-8 和静态读回；正式测试由主 Agent 串行执行。
