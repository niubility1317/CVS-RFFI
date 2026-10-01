# GroupBarrierJoint 完整 query scorer 限定源码审查

原资源 P1 已关闭：冻结 scorer 已补齐推理耗时元数据校验及同域 SUM 读回。在本次限定审查范围内没有剩余 P0/P1。该结论只来自源码、合成测试代码及根 Agent 报告的合成验证，不代表真实产物已经验证，也不包含性能结论。

## 范围与身份

审查者同时是本次 query prediction 入口作者。本次独立范围仅为其他作者实现的 [scorer](../tools/score_d92_group_barrier_joint_benchmark.py) 和 [scoring 合成测试](../tests/test_d92_group_barrier_joint_benchmark_scoring.py)，没有把自己编写的入口作为独立复审对象。入口的独立审查由另一位作者承担。

本次不读取任何真实 run、预测、truth、cache、checkpoint、结果、索引或报告，不执行模型、数值测试、Conda、Git、SSH、发布或评分。根 Agent 已报告该 scorer 原 30 项合成测试通过，证据前缀为 `.codex_tmp/pytest_native_activation_1790868383981773500`；计时局部修复后，该单文件实际 39 项合成测试全部通过，证据为 `E:/type10-7/.codex_tmp/pytest_native_activation_1790870586543935400`。审查者没有运行测试，也没有读取这些日志。39 项来自根 Agent 的实际验证消息，不是从源码推算的数量。

## 已关闭的问题

| 级别 | 位置 | 可定位原因与影响 | 局部修复范围 |
|---|---|---|---|
| P1，已关闭 | scorer `_seconds`、`_seconds_sum_readback`、`_work_seconds`，第 239 至 258 行；`_decision_stream` 第 324 至 325 行；`_fixed_resources` 第 350 至 354 行；测试第 375 至 404 行 | 修复前未校验 marker 的 A/B/C 外层耗时及 public predict 总耗时，逐记录内部 `score_seconds` 也可缺失或为负。修复后，marker 要求精确 A/B/C 秒数字典，所有值必须为有限非负数，bool 不作数值；B 和非 new0 的 C 内部工作必须包含合法 `score_seconds`，其他已保存内部秒数也逐项校验；public predict 总耗时与完整逐记录同域秒数合计核对。 | 只读核对局部修复后关闭。根 Agent 实际运行单文件 39 项合成测试全部通过；新增负测覆盖缺失、负值、非有限值、bool 及错误 SUM，另有机器累计误差读回测试。未增加模型调用、字段、数据验证或审批链。 |

本次收尾只核对上述计时局部修复及对应合成测试源码，没有重审其余主链，也没有修改 scorer 或测试。query work 的 12-key ABI 保持不变。

marker 的 `query_score_seconds` 是 A/B/C 外层调用计时；B/C 的 `work.score_seconds` 是内部计时，C 外层还包含证书构造。两者范围不同，修复没有强行要求相等，也没有伪造缺失的逐记录 A 外层读回。只有已同域保存的 `C_public_predict_seconds` 做 SUM 比较：以 `math.fsum` 为读回参考，误差界为 `(gamma_n + epsilon) * abs(expected)`，其中 `gamma_n = n * epsilon / (1 - n * epsilon)`，并要求 `n * epsilon < 1`。这允许 binary64 顺序累加误差，不是可调耗时容差。

## 已核对的主链

| 检查项 | 源码位置 | 源码判断 |
|---|---|---|
| 固定全部矩阵后才读取 truth | `validate_declared_matrix` 起始行 546；`_physical_matrix` 起始行 570；`score_benchmark` 起始行 577 | 固定两 model seed × 两 cohort 共四行；每个 cohort 的 receiver、scenario、support seed、K、新增类数笛卡尔积与物理 split 一致，总计 2400 个 parent。全部行验证完后，逐行重新读取完整预测与 marker，再重新读取 supervisor startup/complete。首次 truth 读取在这些步骤之后；没有部分前缀评分路径。 |
| 实际 runtime 与同一行来源 | `_source_identity` 起始行 154；`load_fixed_predictions` 起始行 358；`score_benchmark` 起始行 577 | 首次 startup/complete 都绑定全六项身份；实际 runtime commit 与全局 marker 一致，preparation commit 单独保留。当前 row 的 checkpoint、model seed、capsule、source paths、ground packet、support/query 物理 ID、类列、B/C state namespace 和实际继承 ref 均参与核对。没有拿其他 run 的 B 或 R0 补 A/B。 |
| 三路固定预测与 native A 列序 | `_stream` 起始行 197；`load_fixed_predictions` 起始行 358 | 三路逐 split 使用相同 query ID 顺序；每条记录仅有五个预测字段。A 保留 ground 的原六列次序，并按原首列处理并列；B 使用旧类完整列。C 不被要求等于舍入 log score 的 argmax。C alias 与 C 流完全一致。 |
| public C 结构证书 | `verify_structured_decision` 起始行 261；`_decision_stream` 起始行 300 | 独立稳定 logsoftmax 与所有列组合分数读回；组内 winner 来自 raw score argmax，group gap 按保存的运算顺序核对，零 gap 取物理类 ID 字典序。`old_raw_scores` 必须与同一物理 query 的 B 流分数完全一致。证书证明保存记录的结构决策一致性，不通过再调用模型伪造独立证据。 |
| new0 与 K1 | `verify_structured_decision` 起始行 261；`_truth_parent` 起始行 476；结果 `metric_definitions` | new0 的 C 逐记录精确复用实际 B，新增类准确率、H、gap、新增类 macro F1 为 null。K1 在完整 query 基准中仍有效，不被误写为 support-held 的 N/A。 |
| 配对统计与 seed SD | `_truth_parent` 起始行 476；`score_benchmark` 起始行 577；`_statistics` 起始行 426 | A/B/C 在同一物理旧 query 子集配对；跨新增数量核对旧 support 与旧 query 集合。H、绝对 gap、下降在每个 parent 内先计算，再平均。先计算各 model seed 的 parent 均值，再计算两个 seed 的等权均值及描述性样本 SD，不称置信区间。各类 macro F1 的物理子集范围明确。 |
| 训练计费与未知项 | `_fixed_resources` 起始行 335；`aggregate_training_resources` 起始行 530 | factor 总数含 new Ridge、gate forward 和 gate adjoint 的实际分项；五项峰值取 MAX，其余非负工作计数及秒数取 SUM。跨行缺失字段记 unknown/null，不用零替代。scorer 不测训练或部署，未测 RSS/GPU/部署/传输/能量项为 null。原推理耗时 P1 已按上面的同域范围关闭。 |

## 权限与证据边界

scorer 的 imports 只有标准库与 NumPy。唯一 NPZ 访问是 `received.npz['ids']` 的不透明物理 ID 注册表，没有打开 IQ 或特征成员。其余输入为协议 split 元数据和已固定预测、状态 manifest、资源与来源 marker；没有导入 core、Torch、packet loader、fit、predict 或 evaluator。support labels 只用于核对协议中的 support 物理集合，不参与评分器拟合。

truth 连接之后只计算已固定预测的配对统计并独占写出补充评分文件。输出明确禁止选择反馈、自动晋级和重新训练；代码没有子集选择、参数搜索或重跑入口。两遍读回是当前冻结产物一致性检查，不宣称密码学不可变性、独立重新推理、native A 权重重验或 C 训练数学证明。

本次审查不新增 receipt、签名、固定审批或复审要求。已关闭的 P1 仅涉及本次入口已实际保存的成本字段，不能据此重调冻结方法、选择 split 或利用任何评分结果。
