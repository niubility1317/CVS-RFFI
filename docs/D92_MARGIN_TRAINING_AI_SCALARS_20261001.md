# Margin 完整训练诊断的 AI 标量导出

`tools/export_d92_margin_training_ai_scalars.py` 是独立、纯 stdlib 的转换工具。它不修改训练/core/collector，不运行模型，不读取 NPZ、原 run、cache、source/query、历史结果或索引，不选择状态或重跑实验。原完整诊断保持原处；输出仅便于完整阅读已有训练测量。

```text
python tools/export_d92_margin_training_ai_scalars.py --diagnostics-root <completed-diagnostic-dir> --output <new-exclusive-dir>
```

Python API：`export_scalars(*, diagnostics_root, output) -> summary`。输入仅为指定目录的 `summary.json`、`stages.jsonl`、`curves.jsonl`、`preparations.jsonl`。引用字段只作为元数据，不沿引用打开文件。输出目录必须是原诊断目录之外的新独占目录；已有输出不覆盖、不自动重试。

## 输入绑定和完整性

summary 必须为 `COMPLETE_MARGIN_JOINT_TRAINING_DIAGNOSTICS_DERIVED`，schema/method 为冻结 Margin 身份，scope 为完整训练诊断。检查显式 run、实际 runtime commit、source summary 绑定、CE-only/RMSCE 身份与两个正整数 QP 资源界限，bool 不能充当整数。实际 counters 保留非负整数或 null；QP forward/adjoint 工作字段和两个 buffer 峰值必须存在，SUM/MAX 口径必须明确。技术资源上限不会改写，也不解释为实际消耗或进程峰值。

第一遍流式读取全部三个 JSONL，核对 summary 的总记录数，拒绝重复 stage/preparation 身份或跨 run 的记录。每个 B/C stage 必须有对应 preparation、一次 PREPARED、INITIAL 和 FINAL，以及 stage 记录明确给出的实际 GRADIENT/TRIAL/STEP 数。实际计数为 0 的阶段允许没有该类事件；缺少测量字段不会被补成 0。真实 K1、退化停止等已有无更新阶段仍保留 INITIAL/FINAL 和实际零事件数，不发明 epoch 或模型更新。

失败或不完整诊断不能被转为完整标量产物。输入 size/mtime 在两遍转换间必须保持一致，防止转换途中原文件被改写。这只是本次输入一致性检查，不新增 receipt、签名或授权链。

## 保留和排除的字段

每个 compact 记录只含 Python 原生 `str/bool/int/float/null`。除原记录顶层 scalar 外，还按下列明确容器展开，summary 的 `field_mapping` 记录每列的原 JSON 路径：

| 流 | 可展开容器 | 示例 |
|---|---|---|
| stages | audit、initial_objective、final_objective | `audit__fit_seconds`、`audit__trainable_parameter_count`、`initial_objective__RMSCE` |
| curves | measured_scalars、objective、metrics | `measured_scalars__step_size`、`objective__loss_total`、`metrics__CE_gradient_norm` |
| preparations | audit | `audit__preparation_seconds`、`audit__prior_factorization_count` |

已有的损失、权重、学习率、梯度测量、方法状态、step/trial、耗时、可训练参数和工作 counters 因此不会因嵌套而整体丢失。未知数值保留 null；第一遍 union header 中某记录没有的列在该记录输出 null，不推断 0。

数组、scores/labels、class vectors、fold lists、audit/ref/record 大载荷和 text/log 不递归展开。字符串只保留明确类别字段（身份、event、mode、status、rule/reason 等），长度上限为 160 字符，拒绝换行和以 JSON object/array 开头的内容。不会把巨型 audit/ref/text 通过 `json.dumps` 塞进某个“标量”列。所有输出使用 `allow_nan=False`，非有限 JSON 常量或选中的非有限数值直接失败，不用 `default=str` 掩盖类型。

每条记录都显式保存 `source_validation=null` 和 `source_validation_reason=PHASE2_SOURCE_ACCESS_FORBIDDEN`。已有非空 source validation 会被拒绝；这里说明 Phase2 权限，不伪造源验证测量。

## 接受目标与平均 CE

保留 collector 的原始派生 objective/mean-CE 字段。TRIAL 另外输出两个互不替代的布尔量：

- `accepted_RMSCE_increase`：仅在 accepted 已记录，且 objective before/after 和“objective−RMSCE=0”证据存在时，精确比较记录值。
- `accepted_mean_CE_increase`：仅在 accepted 和 arithmetic mean CE before/after 已记录时，精确比较记录值。

被拒绝的 trial 两项为 false；非 trial 为 null。已接受但缺少相应证据时为 null，并输出明确原因。不会把 RMSCE 当算术平均 CE，不从接受标记推断平均 CE 一定下降，也不重新读取 logits/labels 计算缺失量。这些计数描述训练观测，不是性能门槛或泛化证据。

## 输出与核对

输出六个紧凑文件：每个 stages/curves/preparations 各一份 `_compact.jsonl` 和 `_compact.csv`。第一遍取字段 union 并排序，第二遍逐记录输出；不把全量曲线装进内存。随后逐行读回 CSV 与 JSONL，要求记录数、header、字段集合及每个序列化标量一致。CSV 用 `null` 表示未知，bool 为 `true/false`，字符串保留原文本。

最后独占写入并读回 `summary.json`，包含输入绑定、run/runtime、QP resources、各流记录数、字段映射、accepted increase 的 true/false/unknown 数、原 counter scope、转换 wall time，以及实际输入文件和六个 compact 文件字节数。输出 summary 自身不计入该输出字节口径，避免自引用估计。文件字节不是传输、内存或 FLOPs。

转换期间 `conversion_training_operations=0`、`conversion_model_updates=0`，不等于原训练阶段没有模型更新；原 optimizer/参数计数仍以前缀字段保留。第二遍或读回失败时保留已生成文件和 `failed.json`，不写成功 summary、不自动重试。

## 合成验证

root 串行测试入口：

```text
python -s -m pytest tests/test_export_d92_margin_training_ai_scalars.py
```

测试仅用 literal synthetic JSONL：完整阶段/事件、嵌套测量和字段映射、原日志不变、缺失/null、Unicode、native bool、非有限值拒绝、超过 1 MiB 的嵌套和 JSON 字符串载荷过滤、独占输出、source/资源绑定、缺失或重复 phase、CSV 读回篡改与失败保留。开发阶段只执行 AST/UTF-8 静态检查；未运行数值测试、Conda、Git、SSH 或任何真实输入读取。

root已在verified ssr-gpu环境串行验证：26 passed in 0.92s，证据pytest_utf8_1790842765480567600。只验证人工合成输入，尚未导出真实Margin训练诊断；正在运行的runtime没有修改。
