# Margin 冻结 query benchmark 的独立三阶段评分

状态：只实现 scorer、reporter 和人工合成 fixtures，未运行数值测试、未读取真实 cache、packet、权重、预测或成绩，未启动实验。方法、参数、健康运行及既有报告不变。根 Agent 是唯一数值验证、Git、远端和 launch owner。

## 接口与依赖

`tools/score_d92_margin_joint_benchmark.py` 提供：

- `load_fixed_predictions(*, predictions, capsule, config, expected_binding)`：不打开 truth。
- `score_benchmark(*, spec, output, run_root=None)`：完整所有行核验后统一连接 truth。
- CLI：`--spec PATH --output PATH [--run-root ROOT]`。
- score schema：`d92_margin_joint_query_score_v1`；状态：`MARGIN_JOINT_QUERY_SCORE_COMPLETE`。

`tools/report_d92_margin_joint_benchmark.py` 提供 `report_benchmark(*, score, current_metadata=None, baseline_score=None, baseline_metadata=None)`。CLI 必需 `--score`、`--report`、`--interpretation`，可显式提供 baseline。report schema 为 `d92_margin_joint_query_report_v1`。

Scorer 仅依赖标准库和 NumPy；NumPy 只读取 capsule `received.npz` 的 `ids` 成员，仅读 opaque 物理 ID，不读取 IQ、features 或 query 标签。Reporter 仅依赖标准库和本 scorer 的纯函数。没有 core、fit、predict、encoder、torch、packet loader 或历史索引导入。

Spec 使用 `d92_margin_joint_query_benchmark_v1`：共用 `benchmark.config={algorithm, qp_resources}`，cohort 显式 `{capsule, truth, expected_capsule_id}`，row 显式 `{row_id, cohort, expected_model_seed, expected_checkpoint_sha256, row_root, branch_features, ground_packet, output_root}`。预测目录为明确的 `output_root/predictions`。`run_root` 只映射已有 remote run root 下的输出路径，不推断 run/row 身份。

实际 publisher release commit 来自 predictor 和 supervisor 首次写入的字段；它必须与 global runtime commit 一致。Preparation code commit 另行保留，不要求等于发布 commit。

## Truth-last 顺序

1. 先核对全局 startup/complete 的 spec、运行身份、全部声明行、实际 runtime 和完成状态；失败或 partial 不评分。
2. 对所有行核对 A/B/C 三流、C public alias、source/cache/ground packet 身份声明、完整 K×新增矩阵、物理 support/query 隔离、完整 class columns 及 actual B→C ref。
3. 三流逐行只允许 `split_id/query_id/classes/scores/prediction`。每条必须面对该阶段的完整列，finite scores 的 argmax 必须一致。A 保持原 native 列与 tie 顺序，B 和 C 使用 physical class ID canonical 顺序。三流每 split 的 opaque ID 集合和顺序、全局 split 顺序必须一致。
4. 只读当前 `state_manifest.json`，将 core 三字段 metadata view 精确绑定完整 manifest ref；不读取训练 NPZ 数值。Namespace 必须为当次 run/row/split，scope 为 `query_benchmark_support_training`，fold/trial 为 null。C 继承 ref 必须等于实际 B ref；new0 必须复用 B ref 和同一分数。
5. 所有行完成第一遍后，独立重新打开每行三流、alias、startup/complete 与 capsule metadata，要求内容一致，再重读全局 metadata。
6. 此后才打开明确的 opaque truth 文件。结构沿用 `score_d92_confirmation.py`：`query_id → {pool_role, receiver, scene, transmitter, old}`。Scorer 检查角色、接收机、场景、注册列与 old 标志，只用于评分，不传给 predictor。

没有模型执行、拟合、候选选择、校准、参数扫描、重跑或新数据重验。独立 readback 没有增加签名、receipt 或 hash 链。

## 三阶段与统计

每个 parent 的 A/B/C 旧类准确率来自同一物理旧 query。A 是原地面六旧类冻结头，不能由目标 support 拟合的 R0/B0 替代。C 对全注册列统一竞争。K 为 1、5、10、20；新增类数为 0、2、5、10、20；旧类数为 6。整个 capsule 的接收机、场景和 support seed 轴保留，不抽样。

Query benchmark 的 K1 有独立 query，正常报告。新增 0 的新类准确率、H 和新旧绝对差为 N/A。每个注册类须有 query 覆盖。同一 row/receiver/scenario/K/support seed 的旧 query 与旧 support 在不同新增数间保持物理配对。

先逐 parent 计算 `B−A`、`B−C_old`、`abs(C_old−C_new)` 和 `H=2*C_old*C_new/(C_old+C_new)`，再做等 parent 均值。输出完整 parent 记录、overall、K×新增数、receiver/scenario、model/row、support seed 和其完整交叉分层。Reporter 从完整 parent 重算并核对所有统计表，不用总体均值重造 H 或绝对差。

准确率与 H 显示百分比，差值显示百分点。理想目标保持报告方向，不作为评分、发布、选择或重跑门槛。重复冻结 benchmark 数据透明声明，不能称为新的独立确认。

## 显式 baseline

没有 baseline 时不自动寻找历史文件。同三阶段完整 score 可作为显式 baseline，要求 source 身份、完整矩阵、物理 query IDs 和类注册逐 parent 匹配。

旧 `score_d92_confirmation.py` 的 `SCORED/results` 格式必须同时显式给 `baseline_metadata={method, spec, startup, complete}`；多 cohort 使用等长列表。CLI 对应重复的 `--baseline-score`、`--baseline-spec`、`--baseline-startup`、`--baseline-complete` 和明确的 `--baseline-method`。

先核对原方法 spec/runtime 的真实完成字段及完整原矩阵，再按 `(model_seed, receiver, scenario, K, new_count, support_seed)`、capsule/checkpoint、split、classes 和 query count 配对当前 parent。旧记录无 physical IDs 时使用同一 VALIDATED_ONCE capsule+split_id 的已有身份绑定，实际 baseline completion 和预测根由 root 独立核实，不新增数据重验。原 source `reuse_row_root` 与当前 row_root 明确一致。

Legacy 只比较 C 旧类准确率、新类准确率和 H。缺失 A、B 和遗忘全部 N/A，并解释缺失原因；不能补造阶段，也不能用旧 m4 整体均值直接减当前均值。Startup 实际 commit 与 complete 实际 commit 必须一致，但不把旧 preparation parent 当成 runtime commit。

## 资源与验证边界

本次 predictor marker 中实际 SUM 计数、两个 QP 峰值 MAX、推理/总墙钟和磁盘字节原样保留，报告明确来源与测量口径，不把计数称为 FLOP 或把总 prediction 墙钟称为纯训练时间。Scorer 自己的 truth join 时间单列。

`aggregate_training_resources` 对逐 row 的 `actual_training_counters` 做 SUM，只有 `margin_qp_peak_factor_buffer_bytes` 和 `margin_qp_peak_explicit_solve_temporary_bytes` 做 MAX。某字段在任一行未知或缺失时 aggregate 为 null，不能缺省填零。Reporter 按逐行记录重算并核对该 aggregate。

缺失的可训练参数、纯训练耗时、设备型号、最小部署常驻量、GPU 峰值、星地新增传输与能耗记 N/A。包字节不等于传输字节，也不证明部署完成。

合成 tests 包括全矩阵与正常 query K1/new0、所有流完成后 truth join、三流类表/argmax/缺失/重复/顺序/alias 拒绝、actual B/C 继承与 namespace、source/runtime/预算/partial 拒绝、晚失败行不能评分成功前缀、truth 角色绑定、parent-first H、资源未知、完整报告和 legacy baseline 身份/缺阶段边界。测试 fixtures 使用人工 opaque ID、有限手写 scores 和明确 synthetic 资源，没有模型或真实数据。

本 Agent 只进行 AST/UTF-8 静态检查。数值测试与真正 production callback/entry 集成由 root 统一串行验证；源码完成不代表 query 性能、完整发布或实验目标完成。
