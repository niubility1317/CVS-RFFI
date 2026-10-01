# MarginJoint 冻结 query predictor

本文件说明 `tools/evaluate_d92_margin_joint_benchmark.py` 的源码接口。方法保持 `D92-MarginJointLocalRidge-v1`，不改 core、算法、QP 资源或训练预算。开发仅使用源码、literal synthetic metadata 和 mocks；未读取真实 cache/packet/checkpoint/query/成绩，未运行数值测试、发布或启动实验。正式结果与完成状态只能由后续 root-owned 运行及独立评分证据确定。

## 接口和输入权限

```text
predict(*, run_id, row_id, release_commit, row_root, capsule, output, config,
        expected_capsule_id, expected_checkpoint_sha256, branch_features, ground_packet)
preflight(*, 同上参数)
```

`run_id`、`row_id` 来自明确登记，`release_commit` 必须是实际已发布代码的 40 位 SHA，由 supervisor 的 `--commit` 传入。不能从目录名、Phase1 row 或 preparation parent commit 推断。`row_root` 是原 full-cache loader 核对 Phase1 来源所需的原始来源 row 目录，不是新 benchmark run。

config 必须精确为：

```text
{"algorithm": <冻结 core.FROZEN_CONFIG>,
 "qp_resources": {"max_transitions": <显式正整数>,
                  "max_factor_buffer_bytes": <显式正整数>}}
```

没有默认资源值、参数扫描或选择。bool 不能充当整数。`preflight` 返回 `MARGIN_QUERY_PREFLIGHT_COMPLETE`，复用已有 immutable cache/capsule/source-only/packet 绑定读取，不创建输出目录、不 fit、不 score，不新增 IQ 数据重验。

CLI 的字段名为上述参数的 kebab-case；`--config` 指向 JSON 文件，`--preflight-only` 选择只读检查。输出必须是新独占目录，并位于原 source row/capsule/cache/packet 之外；不覆盖原证据或自动重试。

## 每 split 的实际 B→C

原 received 五块 float32 cache 只读复用：`z_id`、`fft`、`t_emb`、`f_emb`、`pa_local`。已有 loader 核对 checkpoint/capsule/model seed、原生特征契约、source-only scratch/final200 来源及原物理 IDs；不加载 encoder/checkpoint，不执行特征导出，不读取原 IQ 或 query truth。

每个 split 只用自己的合法 support labels/IDs：先在六旧类 support 上调用冻结 Margin core 的 prepare/fit 得到实际 B，再在全部 old+new support 上准备 C 并直接继承当次 B。C 的最终 prior ref 必须等于当次同 split B 的最终 state ref。B/C 都从原 received support 重新建立，不读取 support probe 的已拟合状态，不跨 split/row 继承。new0 不创建新 C head，C 是 B 原对象，query 的 C 分数也复用 B 已算分数。K1 的实际退化/更新行为完全由 core 决定，入口不虚构 optimizer steps。

旧类筛选仅用于合法 support 训练；旧/新类注册来自 support registry，不从 query role、truth、真实类别数量或配额推断。B/C 完成后才开始当前 split query 推理。技术 chunk size 固定为 1 并记录，所有 normalization、bandwidth、adapter basis 都来自 support。每次 score 只接收一个 observation 的五块 features；不在 query batch 上拟合统计量或重排预测。

A 使用显式 Ground A packet 的原生 float32 六旧类 CosFace 头，每个相同 query 记录独立评分，无适应。原 native class order 和 first-native-column tie 保留；B 为 core 实际 sorted 六旧类列，C 为 core 实际 sorted 全注册类列。Ground packet 与 cache 的 checkpoint、类集合、原 raw z_id 契约和已声明 model seed 必须一致。packet 只走已冻结 `load_packet` 的 raw bin 路由，不调用 checkpoint 反序列化或原 native encoder 模块。

## 固定预测和元数据 ABI

三个流为 `predictions_A.jsonl`、`predictions_B.jsonl`、`predictions_C.jsonl`。每行精确包含：

```text
{split_id, query_id, classes, scores, prediction}
```

query_id 是原 opaque 物理 ID；三流每 split 的 ID 顺序完全相同，沿原 split 的 query_indices 顺序写出。没有 truth/role/accuracy 字段。`predictions.jsonl` 是 C 的逐行兼容副本；不是原旧 predictor 的 split-array 行格式，独立新 scorer 必须按本 schema 读取。

`startup.json` 和 `predictions_complete.json` 共同绑定：

- `schema=d92_margin_joint_query_predictions_v1`、固定 method、run_id/row_id/release_commit、capsule/checkpoint/model seed、algorithm/qp_resources。
- `run_binding`：明确的 prediction_output_root、row_root、capsule、branch_features、ground_packet 路径。
- `source_identity`：checkpoint_sha256、model_seed、source_only_verdict、source_role_comparison、checkpoint_epoch、checkpoint_inheritance、target_access_before_freeze、cache_schema、feature_contract。
- `ground_packet_identity`：path、checkpoint_sha256、model_seed、packet_declared_model_seed、ordered_classes、实际 scale/norm_eps、feature_contract、source_only_verdict、packet_total_file_bytes、head_weight_file_bytes、head_weight_numeric_bytes 和 byte_scope。
- ordered_ground_classes 为 A 的原顺序，old_classes 为 canonical 六旧类；query_fit_access/source_fit_access/truth_read 均为 false。

每个 `splits` 项含 split_id/receiver/scenario/k/new_count/support_seed、support_ids/old_support_ids/new_support_ids、query_ids/query_count、old_classes、registered_classes（实际 C 列）和 declared_registered_classes（原 split 列）。完成元数据为 `status=COMPLETE`，其 split 项另含 `b_state_ref`、`c_state_ref`、`c_inherited_from_b_state_ref`、`c_reuses_b`、B_classes/C_classes 和 support-only 权限字段。startup 保存预检物理身份；complete 才追加实际拟合 refs。

complete 的 `streams.A/B/C` 各为 `{path, record_count, file_bytes}`；`compatibility_predictions` 为 C alias 路径/实际字节。记录 split_count、completed_split_count、query_record_count 和实际 stage_count。唯一独立 scorer 在完整三流及这些身份核对之后连接 truth；predictor 从不计算任何 query 准确率，也不依据结果选模或重跑。

## 数值状态、日志与资源

原 `StateArchive` 归档 primitives 保存所有真实 prepare/initial/gradient/trial/step/final 回调数组到 output 根的 `state_arrays/*.npz`，manifest 为 `state_manifest.json`。namespace 精确为 run_id、row_id、split_id、`scope=query_benchmark_support_training`、fold=null、trial=null，以及 state 为 B_prepare/C_prepare/B_MARGIN/C_MARGIN_seq。core 的引用保留实际数值 shape/dtype/nbytes；manifest 另含 finite 信息及文件字节。独立检查须按实际 ref/manifest 元数据核对，不把维度或阶段猜成常量。

完整事件在 `training_events.jsonl`，原测量文本在 `training.log`；冻结 core 的 compact scalar interface 生成 `training_events_compact.jsonl/.csv`。保留实际 RMSCE/损失分量/权重参数、梯度、step/trial/Armijo/QP/耗时与 source_validation=null 原因，不添加 epoch。准备及完成 audit 分别写入 preparations.jsonl 和 fit_stages.jsonl；fit_trace.jsonl 记录 split 的实际 B/C refs 与物理身份。

`resources.actual_training_counters` 来自实际准备/阶段 audit，一般工作 SUM、两个 QP numeric buffer 峰值 MAX；缺失计数为 null，不沿用固定 factor/谱公式。每次 B/C 的实际 singleton score audit 写 `query_score_work.jsonl`；new0 明示 C 无额外 score call。A 调用计时包含 float32 scalar-list bridge/native head score；B/C 各保存实际调用数与时间，不将 C 输出记录数误当新 C score 次数。

packet 物理文件、native weight file、float32 weight numeric buffer、cache 和状态 archive 字节分别记录，不标成星地 wire 字节。进程峰值 RSS 是运行 CPU 进程 lifetime high-water 实测（仅支持平台返回），包括本进程库/缓存开销，不是星载部署或 GPU 实测；GPU peak、wire transfer 和 energy 未测，记 null。没有省算力或实际卫星效能结论。

core 或推理阶段失败时保留 completed stage/split refs、完整已写流、失败 audit 和失败数值 NPZ，manifest 标 technical failure，不生成 predictions_complete。已完成工作与失败中的未知工作分开；当前 query 的已返回调用账保留，不把失败调用费用补成 0。不自动重试。只读 input permission/preflight 失败发生在创建输出之前，由 supervisor 的失败证据记录；不会冒充已开始的训练阶段。

## 源码验证范围

直接项目依赖：冻结 Margin core、Ground A module、Ground packet exporter 的 load_packet、full received cache exporter 的只读 loader、orbit cache 的 validate_split，以及 Margin probe 的 StateArchive/json_native。依赖旧 helper 不等于执行它们的 exporter、模型或 query scorer；publisher 必须包含实际传递导入闭包并独立检查。

root 串行测试：

```text
python -s -m pytest tests/test_evaluate_d92_margin_joint_benchmark.py
```

新增一项 production-core 合成集成测试：固定 seed 341、六旧类加一新类、K=3，五块 float32 人工随机特征；只 mock 输入/cache/packet I/O，恢复真实 prepare、fit 及 State.score_with_audit。测试检查真实更新事件、B→C 当次引用、最终 NPZ 数组与 manifest、紧凑标量日志和全部逐样本三流输出。显式 128 transitions/1,000,000 factor-buffer bytes 仅为该合成用例的技术容量，不是任何真实 run 的资源选择；不搜索 seed、不重试、不改变冻结算法。该新增用例待 root 串行数值验证。

测试用 literal split/cache metadata、mock support solver、真实 StateArchive 和合成 native A head，核对 preflight 零 fit/score、只传 support 到 fit、B→C 同次对象/引用、new0/K1、逐样本推理、native ties、query 换序/修改其他 query 的独立性、三流固定 schema/完整顺序、资源计数、配置/来源拒绝、C 失败保留 actual B 和非有限失败数组。开发者只执行 AST/UTF-8 检查，数值验证由 root 串行完成；没有真实 benchmark 完成声明。
