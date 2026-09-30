# D92 AffineJoint 训练诊断收集器

本收集器只从完整、独立核验过的 AffineJoint 归档派生监督训练描述。它不读取真实 query、外层 held 评分、历史报告或总索引，不选择参数，不拟合、不求解、不做 SVD、adapter 前向或距离计算。本文件没有真实实验成绩，也不表示候选已晋级。

文件：[收集器](../tools/collect_d92_affine_joint_training_diagnostics.py)、[纯合成测试](../tests/test_collect_d92_affine_joint_training_diagnostics.py)。字段依据：[当前核心](../code/cvsrffi/d92_affine_joint_local_ridge.py)、[入口](../tools/evaluate_d92_affine_joint_probe.py)、[独立汇总接口](D92_AFFINE_JOINT_SUMMARY_20261001.md)。数学核验由独立汇总器完成，本收集器不重复审计。

## 完整输入条件

仅接受 `summary.json` 的状态 `COMPLETE_AFFINE_JOINT_PROBE_VERIFIED`，方法 `D92-AffineJointLocalRidge-v1`、schema `d92_affine_joint_local_ridge_v1`、scope `SUPPORT_ONLY_AFFINE_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION`。必须有四个不同 row 的训练来源，query/source 行数均为零。

冻结覆盖必须为 160 parent、40 个真实 K1、120 个 OOF parent、1400 个 proxy anchor、1800 条顺序路径、3240 个基线头、3240 个 preparation 和 3240 个候选阶段。合法阶段为 `B_AFFINE`、`C_AFFINE_seq`。新增类数为零时复用 B，因此不制造不存在的 C 训练阶段。有信息、实际更新和停止原因等实际数量从完整训练事件读取，不预设数值。

只选择 summary 的允许元数据。`statistics`、`outer_stats` 和 `resources/resource_statistics` 的外层结果区域直接跳过，不反序列化。原始归档文件数量/字节是目录清单信息；不因保留这些清单而打开其中的 outer feature NPZ。

## 两阶段只读接口

第一阶段 `snapshot(summary_root, run_root=None)` 返回 `COMPLETE_AFFINE_JOINT_TRAINING_READONLY_SNAPSHOT`。它捕获目录、阶段身份、预期事件数量和训练 NPZ 引用索引，不携带完整损失曲线，不读取任何 NPZ 或 `fit_stages` 日志。

引用索引来自 summary 的 `training_objectives.jsonl`、完整 compact 训练流，以及 full training events 中补充的 prior/preparation 引用。引用只允许当前 run、当前 row 的 `B_prepare`、`C_prepare`、`B_AFFINE`、`C_AFFINE_seq` 数值归档；`OUTER_SUPPORT_HELD` 不在允许范围。

第二阶段 `extract(snapshot, run_root=None)` 返回 `COMPLETE_AFFINE_JOINT_TRAINING_DIAGNOSTICS_DERIVED`。它读取索引声明的完整训练曲线和日志，只访问第一阶段捕获的训练 NPZ 引用。目录、阶段索引、事件覆盖或引用不匹配时拒绝继续。它不读取 `fit_trace.jsonl`、outer feature/scores、外部样本或 query。

`fit_stages` 使用字段选择器跳过 `score_seconds`、`score_workload`、`outer_stats` 和外层评分字段。训练阶段的 fit/preparation/cache 时间、训练状态字节及实际工作计数保留。完整训练事件只补回 compactor 省略的物理训练身份、prior/final-problem 引用及已测分布统计。

远端执行采用 SSH stdin 发送完整脚本、stdout 返回 JSON；不创建远端文件。由唯一 launch owner 使用已授权的显式 SSH 配置和 NumPy-capable Python 调用：

```text
python -X utf8 tools/collect_d92_affine_joint_training_diagnostics.py --ssh-host <host> --ssh-config <config> --remote-python <python> --summary-root <verified-summary> --run-root <complete-run> --snapshot-output <new-local-snapshot.json>
python -X utf8 tools/collect_d92_affine_joint_training_diagnostics.py --ssh-host <host> --ssh-config <config> --remote-python <python> --snapshot <local-snapshot.json> --output <new-local-diagnostics>
```

本机可以分别调用 `snapshot(...)` 和 `extract(...)`，再用 `write_outputs(output, result)` 写出结果；也可以使用 `collect(summary_root, run_root=None)`。CLI 的 `--snapshot-output` 与 `--snapshot --output` 保持分阶段。远端内部 `--snapshot-stdout/--extract-stdout` 分支不能写输出文件。

快照和诊断输出均为本机新建文件，目标必须不存在。原始 NPZ 保持原位置和原内容，不复制大矩阵进 JSONL/CSV，不新增 hashes、receipts、authority 链或数据重验。

## 完整诊断内容

每个实际阶段都保留 initial、每次 gradient、全部 trial、每个 accepted step 和 final cached 状态。十二次拒绝试探完整保留，并区分 Armijo 失败、objective increase 或两者同时失败。无信息阶段仍保留 initial/final 和真实完整头，缺失监督目标记 `null/N/A`。

保留 pooled class CE 的 sums/counts/means、RMSCE、prox、total、旧/新类 CE、监督训练正确率、margin、精确 winner tie 及同一训练问题上的 score/winner 变化。类别 CE 先按物理计数跨 fold 求均值，再做 RMS；不能混用宏平均和物理平均。

动态 rank 使用已有 W 的列数和已归档谱。描述 H/W/anchor U、全部 singular values、比例、保留谱条件数、已测 whitening residual/tolerance、原 U 和 Z 位移；不会重新执行 SVD 或白化。rank0 和空坐标的范数为零，min/max/cosine 等无定义量为 N/A。

风险梯度从归档直接分解为 `gCE = g_Z − Z`、`gprox = Z`，没有 `/N`。保留二者的范数、点积、夹角、冲突及归一化方向斜率。只有已归档的完整伴随才报告 `T/eta/g_b/RHS` 的形状和向量统计；缺失分支有明确 N/A 原因，不通过新求解补齐。

每个 inner/full/prior 头保留已有 `intercept b`、`Schur z/s`、combined RHS 的列数/形状/字节，以及已测 normal equation、sample sum、class sum、head loss 等 scalar audit。b 的类别参数数和 C−1 对比自由度分别报告，不能与 `736 × retained_rank` 的 adapter optimizer 坐标混为一谈。

实际函数诊断保留固定 `tau/gamma/s0/q`、actual kernel trace、移动中心、K/L、混合 distance、原始和 adapted 几何、监督得分、prior/residual norm 与已测 block angle。完整旧参考均值是 `q^T M + b`，不能要求它等于零。`scores − M_held − b` 只描述已归档 residual kernel 部分，不构造新前向。

位移和 0.5 路径上界都属于当前训练 support 的 **pre-tangent** 函数坐标。它们不等于真实分数保持，不从 U 范数推断功能变化。tangent/κ 和 adapted-only distance 没有归档时保持 N/A；不补算。

## 13 项新增工作计数

| 计数组 | 三项或两项字段 |
|---|---|
| head | `head_triangular_rhs_count`、`head_triangular_rhs_element_count`、`head_triangular_dense_work_unit_count` |
| derivative | `derivative_triangular_rhs_count`、`derivative_triangular_rhs_element_count`、`derivative_triangular_dense_work_unit_count` |
| prior | `prior_triangular_rhs_count`、`prior_triangular_rhs_element_count`、`prior_triangular_dense_work_unit_count` |
| intercept | `intercept_fit_count`、`intercept_addition_count` |
| prior intercept | `prior_intercept_fit_count`、`prior_intercept_addition_count` |

每次实际 triangular call 计 RHS 列数 r、元素 n r、工作代理 n² r。head 为 C+1 列，adjoint 为 C 列，旧 prior 为 6+1 列。拒绝试探已经消耗的工作全部保留。dense work unit 不能称为实测 FLOP。当前以及未来整数工作计数都保留；preparation 和 stage 分别累计，避免重复计入共享字段。

accepted STEP 会重复已接受 TRIAL 的显示，cached gradient/final 会重复缓存头的 forward 时间。工作费用来自最终 stage/preparation scalar audit，不将曲线显示重复求和。reference pair 是 raw-distance 工作的子集，不再相加。

训练常驻、部署数值状态、训练坐标/cache、准备状态及压缩 NPZ 字节分别保留。`persistent_state_bytes` 包含真实保留的训练 buffers；不能当作最小部署常驻状态。`deployment_numeric_state_bytes` 独立报告。package、传输、GPU、未传入的完整运行墙钟/硬件和外层 prior/residual 推理资源为 N/A，说明字段选择边界，不宣称省算力。

## 实际 B→C、分层与解释边界

C 的 initial U/anchor U 与最终归档内的 `prior_B_U` 做存储状态差值诊断，同时保留 `prior_B_intercept`、prior Schur s、prior combined RHS 形状和实际 prior ref。只描述已存状态，不重新评价 B 分类函数。

在已有 inner Z0 头上，`M_held` 与已拟合 residual+intercept 得分的差表示同一监督 inner 问题中的配对头变化；final inner 与 initial inner 的差描述其后的 adapter/闭式头联动。完整 final C Z0 头未保存时，full-support registration-only 效应与后续 adapter 变化不能独立观测，保持 N/A，禁止新增拟合。这些诊断不能唯一归因性能变化。

`tau = 0` 的原始 feature-equivalence kernel 与 `gamma = None` 的零 kernel 分开描述，二者不以 epsilon 代替。真实/proxy K1 保留合法完整闭式头，没有 adapter 监督目标；new0 精确复用 B，不制造 C 阶段。

完整输出包括 `summary.json`、`stages/curves/preparations/prior_heads/curve_statistics` 的 JSONL/CSV、五种完整训练分层 CSV、对应资源描述、原始归档 phase 清单及 `B_binding_groups.csv`。分层为 mode/trainK、scope/K/新增类数、receiver/scene、model/cohort、row。

B 按同 row、scope、parent K、train K、类别和物理训练 ID 分组，实际重复上下文仍全量计费；另给 representative 的去重描述。不因物理绑定分组就声称重复状态逐位相同，也不增加独立样本数量。

inner-held 标签参与监督训练，因此训练正确率、margin、winner 变化不是独立 query/外层验证。A、B−A 及外层成绩不属于本收集器权限。来源验证和 source validation 仍为禁止访问/N/A。

## 合成检查与交付状态

测试使用手工 NPZ、JSONL 和 summary fixture，不调用核心 fit。覆盖完整曲线与重复 B、全部拒绝试探、动态 rank/空坐标、`g_Z−Z` 分解、b/z/s/伴随、13 计数、实际 B→C 存储诊断、tau0/零 scale、快照无曲线/数组读取、不完整输入拒绝、外层字段不反序列化、引用 allowlist、独占本机输出和模拟 SSH 两阶段传输。

开发 Agent 仅做 AST、UTF-8 和静态接口读回。主 Agent 在项目环境串行运行合成测试，12 项全部通过（9.02 s，证据 `pytest_utf8_1790805054766380800`）；随后仅修正流程注释，无逻辑变化。未经主 Agent 完整 summary 确认，不运行真实收集器；不修改已运行 release 或健康任务。
