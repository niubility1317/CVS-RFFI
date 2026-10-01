# Ground A 支持集配对补充评分

日期：2026-10-01。状态：`IMPLEMENTED_SOURCE_FROZEN / ROOT_SYNTHETIC_TESTS_VERIFIED`。本 worker 只编写 scorer、合成测试和本文，未运行测试、Conda、Git、SSH、发布、真实评分或实验。

## 范围与输出

[scorer](../tools/score_d92_ground_a_support.py)为已冻结的 GroundClassifierA 小包提供独立支持集评分入口。它不拟合、校准、训练、修改模型或选择 adapted B，不接入 query，也不更新原方法 summary、report、trace、spec、core 或运行中的任务。结果只保存在新建的独占补充目录。

Ground A 输入是 owner 已绑定的 source-only packet 和同 checkpoint 的 raw float32 `z_id=feat_joint[N,160]`。模型始终逐记录面对全部六个原 ground 类，严格保留原 classifier 列序，精确并列选择原头第一列。模型接口没有 truth、role、配额、适应状态、trace 或 query 参数。

**当前 row 的选择范围是必填项。** producer cache 可包含完整 900 split 的 support 并集，不能把它全部作为本 row 的评分输入。`predict_support(selection=...)`依据该 row 显式 split identities，与 support manifest 的 identity/`support_ids` 元数据匹配，仅把选中 split 的物理 support ID 并集送入模型。选择阶段不访问 `support_labels` 或 `support_indices`，不按真值或角色筛选。NPZ 容器可能包含额外 feature 行；这些行不进入模型、数值统计或训练。正常 executable wrapper 从同 run/row 的 lane startup.selection 取得原选中 40 split，不改 spec。

## 固定预测与 truth join

两阶段公共 API 明确分离：

| API | 输入与行为 |
|---|---|
| `predict_support(packet, support_features, output, run_id, row_id, selection, expected_checkpoint_sha256, expected_capsule_id, expected_model_seed)` | Keyword-only。独立加载小包，校验来源/cache/feature 身份，只推断显式选中 support ID。它不读取 fit_trace，不访问 support truth 值。 |
| `load_fixed_predictions(predictions)` | 读回完成标记和完整预测 NPZ，验证 finite/float32/shape、原列序、物理 ID、原列序 argmax、绑定与固定状态。 |
| `score_fixed_support(predictions, fit_trace, support_splits)` | Keyword-only。只消费已经持久化的 A；不调用 packet loader、模型、fit 或 calibration。此时才读取选中 support truth，核对同 run 实际 B/C 并配对统计。返回补充 JSON 对象。 |
| `score_support(packet, support_features, fit_trace, output, run_id, row_id, expected_checkpoint_sha256, expected_capsule_id, expected_model_seed, selection=None)` | Keyword-only 的组合入口。预测前只读取该 lane startup 的 row selection 元数据；显式 selection 如提供必须相同。先写并读回 A 完整预测，随后执行独立 truth join，再写新的补充结果。 |

固定步骤写 `ground_a_predictions.npz`，包含完整 float32 `[selected_N,6]` scores、`ordered_classes` 原头列序、`physical_ids`、`prediction_columns` 和 `predictions`。`prediction_complete.json`的 schema 为 `d92_ground_a_support_predictions_v1`，状态为 `GROUND_A_SUPPORT_PREDICTIONS_FIXED`，记录 selection、选中 ID 集、checkpoint/capsule/run/row/model-seed、实际 head/feature 契约及 `prediction_fixed_before_support_truth_join=true`。保存后独立读回，才允许 truth join。

预测仅使用小包的 `load_packet(packet) -> GroundClassifierA`，不加载原 checkpoint 或 encoder。Torch/NumPy 共享数组 ABI 不作为前提；逐记录以显式 float32 scalar-list 构造 CPU tensor，返回 native float32 score 后保存。桥接与 native score 时间分别测量，不改为 float64 矩阵 oracle、标签 margin、softmax、温度或 prototype。

## 实际 B/C 配对

支持的原 trace schema 为 `d92_affine_joint_local_ridge_v1`和 `d92_conditional_joint_local_ridge_v1`，仅使用实际 `R_AFFINE_seq`或 `R_CONDITIONAL_seq`路径。lane `startup.json`和末 `probe_complete.json`必须完整绑定同 run、row、checkpoint、capsule和 model seed。parent `inheritance_binding`、candidate stage及 state namespace 按两个实际 executable ENTRY 的既有 ABI，只核对 run、row、split和当前路径的 fold/trial、parent K、train K；不要求生产档案从未写出的 checkpoint/capsule/seed 字段。selected split identity和当前物理训练点也必须相同。缺失 actual adapted 路径或旧 held ID、类列映射不符、其他 run 的 B、缺 stage、partial trace 均拒绝。

Conditional 保留真实 `prediction_status=FIXED_BEFORE_SUPPORT_TRUTH_JOIN`或 `NO_HELD_PREDICTIONS`检查。Affine v1没有该字段，不能给历史档案编造状态。其固定性按冻结 ENTRY 的实际顺序验证：先归档对应 outer-held feature，使用该路径 final states评分，再构造 `assess_paths`结果。scorer严格检查真实 outer feature archive的全分支 shape/dtype和 `OUTER_SUPPORT_HELD` namespace、同路径 final state refs、完整分数/列序及完成 lane。Affine trace若额外声明 prediction_status也必须合法；Conditional 缺少状态不会借用 Affine 路线绕过。补充输出中的 `fixed_score_contract`明确标识采用的真实 schema 契约。

评分层按 manifest 合法 support truth重建 OOF 和 proxy 的真实物理划分，逐项核对 trace 的 B/C train/held IDs。A 的六类分数与 B 的旧类分数按物理类 ID对应；C 必须按全部已注册类竞争，再计算相同旧 held 点上的准确率。A 不能代表新类预测，R0/B0 不能代替 A 或实际 B。新增类为 0 时，C 按 trace 的精确 B 复用验证。

OOF 在每个 parent 内，每个 held 物理 ID 只计一次。proxy 平均全部 anchor 的指标后形成一个 parent；proxy 与真实 K1 独立 holdout分开。真实 K1 没有 held，`A_old_accuracy`与 `adaptation_gain_B_minus_A`及其他 held 指标均为 null/N/A，即使已对授权 support ID 保存 A 预测也不能把训练点准确率填入 held 表。H、旧类下降和新旧差先在 parent 内求值，再平均 parent，不能从总体平均准确率重新求 H。

## 补充结果 ABI 与报告接入

完成补充的机器可读 schema 为 `d92_ground_a_support_pairing_v1`，状态为 `GROUND_A_SUPPORT_PAIRING_COMPLETE`，scope 为 `SUPPORT_ONLY_GROUND_A_PAIRED_OLD_HELD_NOT_QUERY_EVALUATION`。

`pairing.json`保留 `binding`、run/row、原 method/schema、实际 adapted path、原 trace和 support manifest引用、A 预测目录、原 ground 列序、parent 数、K1 数、每 parent 的 OOF/proxy 指标和逐路径配对物理例项。`actual_A_scope`明确为 `MATCHED_OLD_SUPPORT_HELD_ONLY_NOT_QUERY_OR_DEPLOYMENT_ACCURACY`。每 parent 的指标包括：

- `A_old_accuracy`、`B_old_accuracy`、`C_old_accuracy`、`C_new_accuracy`、`C_h`
- `adaptation_gain_B_minus_A`、`total_old_accuracy_drop`、`C_abs_new_old_gap`

统计表为 `overall`、`by_k_new_count`、`by_receiver_scene`和 `by_model_row`，保留 `parent_count`、`measured_parent_count`、`null_parent_count`、`mean`、`minimum`、`maximum`。最后一张表使用真实 model seed 和 row ID，不猜 cohort。保留完整已声明 K×new 表，准确率和 H 的内部单位是比例，报告显示百分数；变化和差距显示百分点。

输出还包括 `paired_parents.jsonl`、四张统计 CSV、新的 `report.md`和补充 `complete.json`。未来报告只能按 `binding.run_id/row_id`、split ID、diagnostic 和实际 adapted path精确接入这些补充字段。原 summary/report 中的 A/N/A 档案原样保留；不能覆盖旧 summary 的 `actual_A`、用不同 run 的补充填列，或把 support 信息诊断当 query 泛化结果。本任务未修改原 reporter，接入字段与独立补充已经备好。

## 来源、资源与失败

packet loader检查实际小包完整性、六类原行映射、实际 scale/eps、明确的 `logit_corrections=none`及既有 source-only scratch verdict。scorer核对 packet、raw cache和实际 B 的同 checkpoint身份与 model seed。它消费已存在的来源结论，不新增 receipt、签名、哈希链、checkpoint挑选或数据重验。source registry只约束类集合，不能替代原 classifier 行序。

scorer记录实际 packet加载、cache读取、float32桥接、逐记录 native评分和本地总耗时，以及实际文件 stat/numeric nbytes。packet 总本地文件字节、weight文件字节、cache和预测文件字节各有明确口径。新增网络传输、部署峰值内存/GPU、能耗均为 null/N/A，除非后续另有实际测量，不能用 head 形状或本地包大小冒充传输测量。

已有输出在 packet读取前拒绝；输出也不得位于原 packet、cache或 trace目录之内，防止新增文件改变原清单。预测失败保留部分文件并写 `prediction_failed.json`，禁止其 truth join；评分失败写 `scoring_failed.json`并保留已固定的预测。失败后不删除、覆盖、自动重试或重新调用模型；原文件保持不变。

CLI：`python tools/score_d92_ground_a_support.py --packet <packet> --support-features <cache> --fit-trace <actual-lane/fit_trace.jsonl> --output <new-supplement-directory> --run-id <run> --row-id <row> --expected-checkpoint-sha256 <sha> --expected-capsule-id <capsule> --expected-model-seed <seed>`。当前 row 的 selection来自该 lane startup，不能通过 CLI 换成全部 producer矩阵。

发布的直接依赖为 `tools/export_d92_ground_classifier_a_packet.py`、`code/cvsrffi/d92_ground_classifier_a.py`和 `tools/export_d92_branch_support_features.py`；后者传递 import `tools/cvs_native_artifacts.py`。模型/scorer依赖实际 runtime 的 Torch/NumPy，不新增 encoder执行依赖。此任务未发布或调用 CLI。

## 合成验证与真实接入状态

[测试](../tests/test_score_d92_ground_a_support.py)创建纯合成 Torch checkpoint、Ground A小包、float32 support cache和合法 manifest。随后用实际 `probe_affine_joint`及 `probe_conditional_joint`实现生成**纯合成输入的完整训练 trace**。这里的“实际实现”不表示读取真实实验 trace；worker未读取真实权重、cache、实验 trace、support_summary、query、全局索引或交接。

测试覆盖原头非字典序列和精确并列、预测持久化前不访问 truth、truth join禁用 packet/model、同 run/row/state/物理 ID及全类列映射、缺实际 adapted path或固定 feature archive拒绝、错误显式状态/NaN/partial拒绝、真实 K1 N/A、来源/dtype/contract拒绝、独占输出、失败保留、parent-first H/gap/B−A。额外 cache记录和未选 split使用独立 sentinel，guard证明它们不进入 predictor或统计。missing physical ID在预测前独立拒绝；另一项负测在 A 已固定后改变合成 support_labels，证明 truth不一致拒绝并保留预测。两个方法的 fixture均使用实际 production ENTRY 的 minimal run/row/split context，不向 stage伪造额外身份字段。测试由 root串行运行，本 worker未自行执行。

root已说明实际 source-only来源、原 classifier class rows和 factory scale由其只读核实；worker没有接触这些真实包或实验。真实接入仍需要 owner提供该 row完成导出的 packet路径、同 checkpoint raw cache路径、同 run/row完成 lane trace与独占补充输出路径，并由 root执行真实评分及后续报告接入。这些实际执行字段和产物目前未由本 worker读取或产生。当前没有新增实际 A、B−A或性能结果。

root首轮串行合成测试为 30 passed、11 failed，输出 prefix为 `E:/type10-7/.codex_tmp/pytest_utf8_1790824322953885500`。失败来自 scorer错误要求 Affine拥有 Conditional的 prediction_status，以及负测改变 selected physical ID后错误期待模型已执行。root另指出 fixture过度扩展生产 stage身份字段。已按上述实际 ABI修复三项假设；只改本任务 scorer/tests/doc，未改方法或现有档案。

root报告修复后的相关批量检查 **71 passed**，包含本 scorer与 runner/publisher测试的合计；不将 71解释成 scorer单文件的测试数，也不猜单项耗时。本 worker未运行这些测试。修复后验证状态：`ROOT_SYNTHETIC_TESTS_VERIFIED`。上述验证仍为合成输入，不代表真实 Ground A support评分或报告接入已经执行。
