# ConditionalJoint 支持集入口

本次新增五个独立入口及其合成测试，只接受原协议允许的 support 特征缓存。方法标识为 `D92-ConditionalJointLocalRidge-v1`，schema 为 `d92_conditional_joint_local_ridge_v1`，证据范围为 `SUPPORT_ONLY_CONDITIONAL_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION`。两条路径是独立基线 `R0` 和继承当次 B 的 `R_CONDITIONAL_seq`。本文件不包含真实运行、性能结论或独立数据验证结论。

## 边界与入口

| 文件 | 职责 |
|---|---|
| `tools/prepare_d92_conditional_joint_probe.py` | 从 owner 明确提供的新 request、spec 相对路径、commit、rows、cohorts 和六类 seed 生成 spec 与 evaluator 配置；不读取历史配置或结果。所有目标必须不存在。 |
| `tools/run_d92_conditional_joint_probe.py` | 校验声明矩阵、输入身份和路径，按实际 rows 派生预算，使用两条 CPU lane、每 lane 两个 BLAS 线程调度，核对每行 marker 并汇总实测计数。 |
| `tools/evaluate_d92_conditional_joint_probe.py` | 复用原 support-only cache 权限及来源校验，执行真实 core callback，保存完整结构化状态与日志。 |
| `tools/preflight_d92_conditional_joint_probe.py` | 仅核对显式 capsule/cache 元数据、source-only provenance、选择矩阵和输出冲突；不加载 NPZ 数值或 checkpoint。 |
| `tools/publish_d92_conditional_joint_probe.py` | 维护精确依赖白名单，复用已有发布传输。缺失任一依赖时在发布前退出。 |

Prepare 的 CLI 是 `--request --commit --output-root`。request 明确包含 `run_id/group_id/spec_path/code/execution/permissions/probe/rows`；cohort 提供实际 capsule 身份和路径、producer matrix、selection 及配置相对路径；每个 row 提供 cohort、row ID、support cache、source-only checkpoint SHA256 和全部六类 seed（未知且不适用者显式为 null）。不从一个 seed 推断其他 seed。Evaluator 使用 `--support-features --capsule --output --config --expected-capsule-id --expected-checkpoint-sha256 --expected-model-seed --run-id --row-id`。

禁止 query/source 样本、源域逐样本特征、历史 target 评分、跨 run 适应状态和 summary 输入参与适应。Evaluator 复用已有 `load_support` 对 producer cache、capsule、物理 ID、checkpoint 身份及来源结论的检查；不新增 IQ 重验或 receipt 链。Preflight 只检查数值缓存文件存在，不读取其值。真实 ground A 尚未接入，`A_old_accuracy` 与 `adaptation_gain_B_minus_A` 始终为 null，不用 R0/B0 代替。

## 物理路径与继承

每个声明 receiver/scenario pair 必须覆盖 K=1/5/10/20、旧类 6、新增类 0/2/5/10/20 的完整矩阵。选择权限由显式 selection 确定，不限制为固定四个 row，也不将其他 receiver/scenario 强加到 selection。相同 K 的不同新增类数量必须复用相同旧类物理 support。

真实 K1 只执行全 support 解析头，无 OOF、held 或伪造指标。K>1 按物理 ID 排序做 `min(K,3)` 个外层 fold；同时遍历 K 个单样本 anchor，先在 parent 内求均值。训练、held ID、fold、trial、parent K 和 train K 均保留。

路径中的真实顺序为 B0/C0 基线，然后 `prepare B → fit B → prepare C → fit C`。C 接收刚完成的同 row、run、scope、fold 及物理训练集 B 对象。core 检查 B 来源与输入；inner prior 由 core 在当次实际 B adapter 上用 old-inner-train 重拟合。新增类数 0 时直接复用 B 对象和分数，不构造 C preparation/fit。入口不修改 RMSCE-only、0.5 坐标球和 4×12 固定预算。

所有 held 分数先固定成完整注册类列序的列表，再交给独立诊断函数连接合法 support-held 标签。每条证据注明 `FIXED_BEFORE_SUPPORT_TRUTH_JOIN` 或 `NO_HELD_PREDICTIONS`。没有 query 推理或评分接口。

## 归档与日志契约

`StateArchive` 接受只读 Mapping 中的有限数值数组。浮点数组必须为 float64，标量 0D 保留原 shape，整数/布尔索引原样保存，禁止 object/pickle。每次真实 core callback 写独立 NPZ；manifest 记录每个数组的 shape、dtype、字节数和已测归档耗时。namespace 包含 run/row/split、state、scope、fold、trial、parent K、train K。模型状态和 prior 的数组由 core 原样提供，入口不猜补字段。

产物包括 `fit_trace.jsonl`、`fit_stages.jsonl`、完整 `training_events.jsonl`、紧凑事件 JSONL/CSV、`compact.csv`、详细 `training.log`、`state_arrays/` 和 `state_manifest.json`。事件 envelope 明确 schema、method、split 和作用域，保留 core `FINAL` 事件的 `{mode,state_ref,audit}` 实际结构。训练 source validation 为 null，并注明 `SOURCE_ACCESS_FORBIDDEN`。

异常保留完成的 preparation、candidate state、路径、计数、原异常信息和已落盘数组；manifest 标记 `INCOMPLETE`，不生成完成 marker，不自动重试。输出目录、数组文件、marker 及 preparation 文件均采用独占创建。Supervisor 记录失败 lane，不覆盖或重新运行该 lane。

## 工作量与内存口径

预算由每个 row 的显式 selection 逐物理路径推导，再按实际 rows 求和。预算上界仅验证固定流程是否越界，不写成实际工作量。退化、K1、rank0、tau0 和 new0 的真实工作按 core counters 累计。

Candidate 分别保留 projection、residual、projection_adjoint、residual_adjoint 的 factorization、triangular solve、RHS 列数、RHS 元素数和稠密 `n²×nrhs` 工作代理。正条件核的三项谱诊断单列 `spectral_diagnostic_count`。Student factor 总计来自实际 inner/final audit，不沿用 Affine 每个头最多一次分解的假设；baseline/prior 另计。Entry 的 `candidate_preparation_count/candidate_stage_count` 是 core 既有 `ajlr_preparation_count/ajlr_stage_count` 的实测别名。

保留 baseline 的额外 effective-df solve、优化器接受/拒绝试探、前向/反向、最终推理次数与物理样本数。未测量值不从预算推断。Marker 核对实际计数的非负整数、覆盖数、总计、固定预算和 RHS 基本恒等关系；独立数学重建属于后续 summary。

为兼容已有紧凑报表，`persistent_state_bytes` 和路径 `deployment_C_state_bytes` 明确表示 **保留实际 B 的 C resident numeric state**，不是最小部署字节数；marker 带完整 scope 说明。真正的最小部署数组量另列 `minimum_deployment_numeric_state_bytes`，来自 core `deployment_numeric_state_bytes`。NPZ 文件大小、原数值数组量、prepared state、运行进程峰值 RSS 和实测耗时各自保留，不混为硬件 FLOPs 或端到端部署成本。

## 发布状态与验证

Publisher 显式包含纯条件核、Conditional 上层、实际使用的 Affine/geometry/cache helper 和独立数学证书 `tools/d92_conditional_analysis_math.py`。它也要求将来独立的 `tools/summarize_d92_conditional_joint_probe.py`；本入口交付时尚未据此宣称分析链完整或可以 launch。没有生成真实 spec/config、发布记录或 launch。

五个测试文件只构造合成数据。覆盖实际 `evaluate()` 调用新 core 后产生的 stage/event/NPZ archive、schema/scope、精确 B 对象继承及替换拒绝、new0、K1、固定分数之后的标签连接、失败保留、实际 row 数预算、marker 与传输前边界。缓存 I/O 由合成 loader 替身隔离，core callback 和 archive writer 均为 production 代码。root 已串行执行本组与核心测试：**37 passed，16.48 s**（18 个核心用例与 19 个入口用例），原始输出前缀为 `E:/type10-7/.codex_tmp/pytest_utf8_1790819674350570000`。首轮暴露的 Training `audit_dict()` ABI 缺口已补齐；未修改数学或删除失败用例。此结论不含真实评分、独立 summary、实际发布或 launch。
