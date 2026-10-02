# SupportMetric 固定输出补救评分入口审查

审查结论：`NO_UNRESOLVED_P0_P1`。在本次限定源码范围内，未发现会接受未完成预测、扩大失败接受范围、提前读取 truth、改写原 run 或重新拟合的直接 P0/P1 问题。本结论是源码审查，不是实际输出已闭合、评分已执行或方法性能已验证的声明。

## 范围与证据边界

本次只读检查新 `tools/score_d92_support_metric_fixed_outputs_after_scalar_failure.py`、scalar-r02 scorer 的公开验证接口，以及 `tools/run_d92_support_metric_joint_benchmark.py` 的 producer 终态控制结构。为核对边界覆盖，读取了对应的字面合成测试 `tests/test_d92_support_metric_fixed_outputs_after_scalar_failure.py`。没有复审数学 core、原 scorer 的全部数学实现或旧实验结果。

未读取真实 cache、packet、checkpoint、预测、状态 NPZ、truth、日志、成绩、索引或 handoff。未执行数值、测试、Git、SSH 或拟合。合成测试文件声明的 12 个用例由 root 负责执行；本审查没有见证它们的数值运行或使用测试产物。

本次唯一交付是该文档，WS/WT 镜像。被审代码和原 supervisor 文件均未修改。

## 原终态与失败接受范围

新入口 `close_fixed_outputs`（第 23 行）先调用冻结 config 和完整矩阵验证，随后读取原 `startup.json`、`complete.json` 与 `state.json`。第 32 至 46 行要求原 spec、run/group、实际 runtime commit 和准备 commit 一致；原 terminal 必须仍为 `SUPPORT_METRIC_QUERY_BENCHMARK_PREDICTIONS_FAILED`，原 `all_predictions_fixed` 必须为 false，且原记录没有 truth、scorer 或自动重试访问。

第 42 至 51 行要求 startup、terminal、state 的 row 集合完全相同，state 等于 terminal 的 rows，全部 4 行只能是 COMPLETE 或 FAILED，不能含未完成状态；原已完成 row/episode 数必须与这些状态一致。

第 55 至 60 行要求每个 producer 有实际记录的正整数 PID 和精确整数 exit 0。FAILED 行只接受 `phase='PREDICTION'`、`error_type='ValueError'`、`error='Trial float tolerance differs'`，并且不能自行携带 completed marker。其他失败、非零 exit、PREFLIGHT 失败、未终态或 partial 均不进入补救评分。

这一接受条件与 supervisor 第 270 至 277 行的控制顺序匹配：先保存 producer 的 PID/returncode，再调用固定输出验证；验证异常进入 PREDICTION 的 FAILED 分支，保留已记录的 PID/returncode 和 prediction 目录。补救入口仍会完整执行 r02 验证，错误字符串本身不能代替实际输出闭合。

## 固定输出、来源与实际状态

新入口第 61 至 78 行逐 row 调用 scalar-r02 的 `_validated_row_fixed`，并核对 marker PID 与原观察 PID。原 COMPLETE 行的 `lane['marker']` 必须逐元素等于当前重新读出的 marker（第 67 行）；FAILED 行的 preserved prediction 目录必须等于该 row 的既定目录（第 68 行）。第 70 至 76 行还将实际 release、model seed、checkpoint、capsule、source paths、output 与原 spec 绑定，没有只检查一个错误文本后跳过来源检查。

scalar-r02 的 `_validated_row_fixed`（第 1232 行）使用明确的 run/row/capsule/checkpoint/model/runtime binding 调用 `load_fixed_predictions`，核对完整物理矩阵，并将 preflight 的公开身份与 producer startup 比较。只有实测秒数从跨进程身份比较中剔除，来源路径、字节、冻结标志和其他身份字段继续核对（第 1244 至 1259 行）。complete 的 source/deployment/output 还直接与 spec 比较。

`load_fixed_predictions`（第 976 行）继续要求 producer startup 与 marker 是同一新 schema/method/config/resource，marker 为 COMPLETE，没有 prediction failure 文件，query/source fit、truth、checkpoint/encoder、跨 split 适应状态访问为 false。它使用既有合法 capsule metadata 的注册类和 opaque ID，检查一次 row basis 的 owner、物理 NPZ archive、每个 split 的实际 B/C 与 R0 状态引用，以及 new0 的精确 B 引用复用。

该公开接口读取全部 `A/B/C/R0_B/R0_C` 五流，要求 opaque query 顺序一致、C 公共 alias 完全一致，并调用既有 decision、training ledger 和 resource 验证（第 1044 至 1063 行）。新补救入口没有新增任何拟合或模型执行调用，也没有用 query role、配额或实际类别数量生成预测。

## 全行闭合与 truth-last

第 77 至 86 行要求同一 cohort 的两个 model row 使用相同物理 split 元数据，并要求全部 4 行合计 2400 个完整 parent。验证不能从一个已完成前缀进入 truth。

第 87 至 93 行对每一行再次调用同一个 `load_fixed_predictions`，比较完整返回的固定证据；随后再次读取原 startup、terminal、state，要求与首次读取完全一致。固定流、actual state、once-row-basis、ledger 与原终态均走同一公开验证路径。

`close_fixed_outputs` 自身没有 truth 读取。`score_fixed_outputs` 在上述全行验证及第二次读回结束后，先独占写入并独立读回 `closure.json`（第 112 至 115 行），第一次 truth 读取才出现在第 121 行。此后只从已固定五流连接显式 truth 并计算 parent 内的 A/B/C/R0 指标；没有补阶段、选择 row、重跑、预测反馈或自动晋级。

closure 是普通派生结果记录，明确标记原 supervisor 仍 FAILED、原状态不变、无训练/模型调用/自动重试。它不作为新许可、authority 或 approval artifact（第 94 至 101 行）。

## 写入隔离与失败保留

`score_fixed_outputs` 第 108 至 111 行要求派生输出目录不存在，并在解析实际路径后拒绝位于原 run 内的输出。原远端 run 与显式 `run_root` 本地映射均受该限制。新入口只在独占派生目录写 closure、summary 或 failure，不将原 FAILED 改写为 COMPLETE。

summary 写入后独立读回（第 154 至 155 行）；异常写入新目录的 `failure.json` 并重新抛出（第 157 至 164 行）。源状态、原 marker、原 state arrays 和健康 producer 不在写入或启动路径内。

## 合成覆盖与未验证项

本次 12 个字面测试覆盖混合 COMPLETE/FAILED 与全 FAILED 的合法终态，全部 4 行首次验证及第二次读回完成后才能读取两个显式 truth；不同错误、非零 exit、缺行、不足 2400、第二次变化、PID 不符、原 COMPLETE marker 变化、非终态、state/terminal 不一致均在 truth 前拒绝；既有目录和原 run 内的新目录拒绝写入。测试还逐字节检查原 startup/state/complete 保持不变。

这些测试复用独立 scorer 已有的字面完整矩阵 fixture，限定检查新 orchestration；实际五流、NPZ、basis、actual B 数值验证沿 scalar-r02 的公开接口，不由新入口另写弱化版检查。实际输入与终态是否满足这些条件、root 的数值验证结果及正式 truth-last 评分均不在本次源码审查的实证范围内。

状态：`FREEZE`，`NO_UNRESOLVED_P0_P1`。不增加新的 gate、receipt、审批或数据重验。
