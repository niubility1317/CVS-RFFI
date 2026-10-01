# GroupBarrier 重复 query benchmark 入口源码审查

日期：2026-10-01。结论：**NO_UNRESOLVED_P0_P1**，仅针对本次冻结的 query 入口、发布控制和下列两处局部修复。结论不表示真实预测、评分或发布已经完成，不表示性能或星载资源收益得到验证。

## 范围与独立性

审查范围是 `tools/evaluate_d92_group_barrier_joint_benchmark.py`、`tools/run_d92_group_barrier_joint_benchmark.py`、`tools/publish_d92_group_barrier_joint_benchmark.py`、本次授权的新配置源码 `configs/d92_group_barrier_joint_repeat_20261001.json` 及 `tests/test_d92_group_barrier_joint_benchmark_pipeline.py`。仅按调用边界读取既有 cache、packet 和冻结核心源码，未打开配置引用的真实输入。

审查者没有编写上述入口，但编写了低层 gate 和本次 GroupBarrier scorer。**本文件不包含对 gate 或本人 scorer 的独立复审。** 根任务要求的实际 B 分数继承核对是一个有界源码事实确认，不作为 scorer 全面独立验证。整个过程保持 query-blind：未读取真实 run、日志、数据、cache、checkpoint、truth、prediction、评分、报告、实验索引或交接，未运行数值测试、Conda、Git、SSH 或实验。

## 发现并关闭的 P1

| 问题 | 修复与验证范围 |
| --- | --- |
| 外层失败归档可能覆盖原始异常。核心已经保存原始数值异常后，evaluator 再次调用失败归档；如果 callback 或存储设备失败，外层归档异常可能取代原始 cause。 | evaluator 第 362–397 行将失败 audit、NPZ、manifest、CSV、JSON 和 stderr 记录逐步隔离，单列 `reporting_failures`，最终裸 `raise` 保留原异常及 cause。能写入的证据按实际状态保留，无法写入时不声称归档成功。新增合成回归覆盖真实 StateArchive callback 设备错误及原 cause 保留。 |
| public C predict 失败时，已返回的 C 几何工作记录可能丢失。 | evaluator 第 297–315 行先保存 `_structured_score` 返回的 `C_work` 和结构证书，再执行独立 public predict。尝试数、成功返回数和实际 elapsed 分开；失败调用未暴露的内部工作仍记未知。新增合成回归覆盖几何已成功、public predict 随后失败。 |

入口作者确认上述局部修复及测试源码已冻结，并报告 AST/UTF-8 检查通过。审查者只复核相应局部变化，没有重新审查或修改固定数学方法、预算、core、scorer 或已有健康运行。

## 关键控制与数值接口

**训练和继承。** evaluator 的 `_inputs` 复用原 received 五块 cache loader 的来源、capsule、checkpoint、model seed、原始 float32 特征契约核对，并用同来源 Ground packet。只读取 received 文件的 opaque ID 成员，不执行 encoder 或 checkpoint 载入。每个 split 的 B 仅接收旧 support，C 接收该 split 全部合法 support；query 数组不进入 prepare/fit。C 明确检查 `state.prior is actual_B` 及同阶段 prior ref。new0 直接复用当次 B 对象、状态引用和分数，不生成新头或额外 C 推理调用。

**A 与 C 的真实推理语义。** A 通过 Python list → Torch float32 tensor 调用原生冻结头，保留原始六列和原列序 ties。query 每次只处理一个物理记录，不估计 query 全局统计或使用 truth、role、真实类数、配额。C 第一次调用与原 score wrapper 相同的几何路径，保存 raw old/new scores、log probabilities 和 gate gap；第二次调用冻结 public `predict`，要求结果与结构证书一致。没有第三次模型推理，也没有用舍入后的 log-score argmax 替代结构化决策。

**完整身份与矩阵。** runner 固定两模型 × rx3/rx1 的四 row。每模型 rx3 为 900 个 split、rx1 为 300 个 split，合计 2400 parent。核对全部 receiver × scenario × K × 新增类数 × support seed 坐标，数量相同但重复或缺失坐标不能通过。各 row 输出独占；同 cohort 两模型的物理 split metadata 相同。startup/complete 绑定实际 runtime commit、PID、源路径、checkpoint/capsule/model seed、完整算法与资源参数；preparation commit 不冒充发布后的实际 runtime OID。

**实际工作与失败。** 训练工作保留原 B 与新 Group 的联合计数字段；前向、反向 gate 和新类 Ridge 的实际 factor 尝试均进入总数。工作和 float 秒数按 SUM，五类数值峰值按 MAX。C 的两次几何计算分别显示为 score 和 public predict；第二次 public predict 无内部审计接口，因此内部细项为 null，不能将第一遍的细分账冒充两遍总账。失败记录区分已完成准备/阶段、失败 fit、已返回 query 调用和未测失败调用。完整状态、事件、详细文本及紧凑 JSONL/CSV 均保留。

**发布与 truth-last。** publisher 在远端变更前核对本地已提交白名单、当前 HEAD 与远端已推送 OID；显式最小传递依赖经隔离 import 检查。新 release/run 独占创建，唯一 root supervisor，固定 CPU 两 lane/每进程 BLAS 两线程；禁止自动重试或干预已有任务。入口只生成固定预测，完整四 row 才能形成全局 `all_predictions_fixed`；publisher 和 runner 不自动调用 scorer。独立评分是后续 root 操作，不将 launch readback 当作预测或评分完成。

## 实际 B 继承的窄核对

现有 scorer 的 `verify_structured_decision` 将 C 证书旧类列序与 canonical 旧类表绑定，并要求 `old_raw_scores` 与同 query 的 B 流分数精确一致（当前第 279 行）。同函数由这些原始 B 分数取得组内 winner，再按 gate gap 和 lexical tie 选最终类；`_decision_stream` 按 split/query 顺序连接证书和 B/C 流。`_stream` 对 A/B 流核对各自列序的完整 argmax。

因此，当 C 对某个 query 输出旧类时，该类必须等于同记录的 B winner。对同一物理旧类 query 集合，可以推出 C 正确集合是 B 正确集合的子集，即配对 `B_old_accuracy - C_old_accuracy` 非负。**该关系不能推出遗忘不超过 1 个百分点，也不能证明适应提升或新旧类差距达到理想目标。** 这里只确认既有直接检查存在，没有改动 scorer 或重复其测试。

## 证据及限制

根任务已串行执行 query 入口两处局部修复的相关测试，6 项全部通过（`pytest_native_activation_1790869729050579200`）。scorer 完整受影响测试文件在资源秒数校验修复后，39 项全部通过（`pytest_native_activation_1790870586543935400`），包含完整 2400-parent fixture。以上均为根任务报告的实际执行结果，不是从测试源码或启动命令推断通过；scorer 结果仅作为接口相关合成证据，不构成审查者对本人 scorer 的独立审查。本次只更新验证记录，没有重复源码审查或测试。

本次没有真实数据、真实 query 指标或设备实测证据。重复 benchmark 仍是既有数据复用，不是新的独立泛化确认。gate factor buffer 限额只表示其内部因子缓冲；CPU RSS、完整常驻状态、传输 bytes、GPU 与能耗口径不可互换。未测量项保持 N/A，没有新增性能门槛、数据重验、receipt/签名链、额外审批或发布授权。
