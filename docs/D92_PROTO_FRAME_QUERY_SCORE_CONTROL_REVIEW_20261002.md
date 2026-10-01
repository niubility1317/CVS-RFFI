# ProtoFrame query 评分与控制面有限源码审查

审查日期：2026-10-02。结论：本次只读源码检查发现 2 项直接 P1，均位于独立 scorer 的最终状态闭合检查。修复归 scorer 作者和 root；本审查不修改 producer、core、方法、资源上限或健康任务。未发现所检查 run/publisher 路径中的直接 P0。

## 范围与证据

审查对象为 [独立 scorer](../tools/score_d92_proto_frame_joint_benchmark.py)、[root supervisor](../tools/run_d92_proto_frame_joint_benchmark.py)、[publisher](../tools/publish_d92_proto_frame_joint_benchmark.py)、[scorer 的 literal tests](../tests/test_d92_proto_frame_joint_benchmark_score.py) 和 [控制面 literal tests](../tests/test_d92_proto_frame_query_supervisor.py)。只核对 producer 公开的 schema、流、状态引用和账本字段；没有自审 producer 内部，没有复审数学 core。

本次未读取真实数据、缓存、权重、packet、运行产物、日志、结果、评分、索引或真实 matrix/spec；未运行数值、Conda、Git、SSH、发布或实验。root 告知 scorer 33 cases、query producer 22 cases、control 13 cases、analysis 25 cases 通过，累计 204 个不同合成 cases。本审查没有见证这些原始测试证据，因此不把该信息写为本审查独立验证结果。下列结论依据所读源码和合成测试定义。

## 直接问题

### P1：最终状态描述符未闭合到实际 NPZ

`_state_ref` 核对引用与 manifest 的 metadata、有限性声明、`shape/dtype/nbytes` 的算术关系、相对路径格式和同 split 的 namespace。`load_fixed_predictions` 随后核对所有阶段引用，但没有核对这些路径对应的 NPZ 文件是否存在，也没有读回文件字节数、成员集合或真实数组的 shape/dtype/nbytes/有限性。因而缺失或损坏的真实最终状态不能被这一层独立检查拒绝。

直接源码证据：审查时 scorer 的 `_state_ref` 在第 215 行开始，`load_fixed_predictions` 在第 499 行开始。literal `row_bundle` 的 `ref` 在测试第 144 行附近只追加 `state_arrays/*.npz` 描述符，没有生成相应文件；该 bundle 作为 `test_literal_five_streams_native_A_distinct_R0_lineage_new0_and_parent_metrics` 的合法输入。因此当前正测本身允许只有描述符、没有实际最终状态的完成记录。

限定修正：对当前 canonical state manifest 的实际文件完成路径、字节数、NPZ 成员和真实数组的严格读回闭合，保持现有 metadata 检查。只读本次方法状态档案；不读取 IQ、feature cache、checkpoint 或地面 packet，不重放模型。缺文件、少成员、不同 dtype/shape/nbytes、非有限完成数组均应拒绝。合成 fixture 应生成真实 NPZ，并增加缺失或篡改最终状态的负测。该项是已经要求的完整状态归档正确性，不是新的科学权限或额外发布门槛。

### P1：实际 C prior 审计未与实际 B final ref 交叉绑定

scorer 核对 marker 的 `c_inherited_from_b_state_ref == b_state_ref`，也核对返回 C stage 的 `final_state_ref` 与 marker 的 C ref 一致。但 `_training_ledger` 没有核对 C stage audit 的 `final_prior_ref`，也没有核对 C preparation audit 的 `full_prior_ref`。这两个公开字段来自实际准备/拟合返回的审计，必须指向同 split 的实际 B final ref；只核 marker 的继承声明不能替代该连接。

直接源码证据：审查时 `_training_ledger` 在第 413 行开始；candidate audit 循环核对物理 support 和 1 次 GGN/12 次 trial 预算，缺少上述 prior 交叉绑定。literal 合法 fixture 的 candidate audit 与 preparation audit 也不含这两个字段，故目前正测未覆盖此实际继承接口。

限定修正：从同 split 的 `B_PROTO_FRAME` 返回 audit 获得实际 `final_state_ref`，同时要求 `C_PROTO_FRAME_seq.audit.final_prior_ref` 和 `C_prepare.audit.full_prior_ref` 与其完全相等；保持 marker 的继承检查。`new0` 保持没有额外 C prepare/fit、候选 C/B 引用完全相同。增加缺失、跨 split 和篡改 prior 的负测，不新增 teacher、refit、方法选择或数值容差。

两项均已在本次审查期间发给 root 和 scorer 作者。本文件记录发现时的源状态；不声称修复或修复测试已经完成。

## 已核对的边界

| 项目 | 有限源码结论 |
| --- | --- |
| 全矩阵与 truth-last | `score_benchmark` 先要求 4 个 model/cohort rows 完成，逐 row 验证矩阵和同 cohort 物理 metadata，要求总计 2400 parents，独立重读全部固定流和 supervisor metadata，再进入 truth 读取。失败、子集或重读变化不能从已验证前缀开始评分。 |
| SourceRaw 契约 | scorer 使用独立 literal：原生四个 160 维键为 `z_id/t_emb/f_emb/pa_local`；FFT 独立声明 96 维和历史频谱 sketch。该字面量与公开 source identity 完整比较，不把 FFT 塞入四个 native branches，也不接受近似契约。scorer 不 import fitting/model/cache/packet reader。 |
| 原生 A 与五流 | A 保留原地面六列顺序和 first-column tie；B、C、R0_B、R0_C 使用 canonical 类顺序。五流对同 split/query 物理 ID、顺序和完整列集合；R0 独立 stage/ref/audit，不能冒充 A 或候选 B。 |
| C 结构判断 | certificate 绑定同 query 的实际 B raw scores，稳定重算组内 logsoftmax、gate gap 和全部 log-score；跨组精确 tie 使用 lexical 类 ID。允许公共 score 的共同偏移舍入导致浮点 argmax tie 合并，仍以记录的结构判断核验公共 C prediction。此检查没有读取 query role/count 或硬路由真值。实际 prior 引用检查仍有上面的 P1。 |
| `new0` 与 K1 | `new0` 必须候选 C=B、R0 C=B，引用、分数、预测相同；新类指标、H、gap 为 N/A。query benchmark 的 K1 有独立 query，不能套用 support held 诊断的 K1 N/A 规则。 |
| 工作与秒数 | core preparation/stage/score 和总账覆盖独立声明的全部 WORK keys。计数 SUM、峰值 MAX；秒数用 `math.fsum` 与非负顺序累加的 `gamma_n` 舍入界比较，不把各 phase 已舍入的和当精确总值。R0 返回账和 EDF extra solves 独立计费；公开 predict 与 R0 score 的未知内部工作继续 N/A，嵌套计时不再相加为额外 wall time。 |
| 无反馈 | scorer 仅输出冻结预测的独立指标与描述性统计，记录 `selection_feedback_forbidden=True`、`automatic_promotion=False`；没有 fit、参数搜索、fallback、结果驱动重跑或停止健康任务路径。 |
| root lane/资源 | supervisor 固定 root owner、CPU lanes=2、BLAS threads=2，先给全部 rows 完成 preflight 决策，再用 2 线程运行已通过 rows。逐 row output 和 run root exclusive 创建；原生输入与输出分离。失败 row 保留目录，其他已授权 rows 继续；没有自动 retry。 |
| 发布与读回 | publisher 只打包显式源码白名单和当前 spec，独立比较当前 pushed OID，先检查远端碰撞；安全解包、隔离 import 后单次派发。读回 `launch` 和 supervisor `startup` 仅证明派发及启动，不宣称预测完成。传输、派发或读回不确定时标 UNKNOWN，保留产物并要求只读 reconcile，不自动重发。 |

## 有限结论

本次审查冻结为一次源码结论：除上述 2 项 scorer 状态闭合 P1，所检查的 truth-last、完整物理矩阵、独立 SourceRaw 契约、五流、结构 C 判断、SUM/MAX 账本和 root 发布失败保护路径未见新的直接 P0/P1。该结论不证明真实性能、实际硬件收益、真实运行完成或真实数据正确性，也不授权新实验、重新发布、重训练或修改健康任务。修复后由作者和 root 执行受影响的有界合成验证即可，不要求再次完整数学审查。
