# SupportMetric 联合方法预登记配置审查

结论：`NO_UNRESOLVED_P0_P1`。在本次授权的配置、CONFIG_ONLY 预登记及 metadata 证据范围内，未发现会使矩阵、来源权限、继承、资源限制或唯一启动执行错误的直接 P0/P1。此结论是启动前声明与 metadata 的一致性检查，不是运行成功、准确率、泛化或星载成本结论。

## 范围和证据口径

本次只读以下五份材料，唯一交付为本文：

- [方法配置](../configs/d92_support_metric_joint_support_20261002.json)。
- [CONFIG_ONLY 实验登记](../automation_reports/CV-SincNet/20261002-phase2-d92-support-metric-joint-support-m2-r01/experiment.json)。
- [CONFIG_ONLY 报告](../automation_reports/CV-SincNet/20261002-phase2-d92-support-metric-joint-support-m2-r01/report.md)。
- 工作区 `E:/type10-7/automation_reports/CV-SincNet/20261002-phase2-d92-support-metric-joint-support-m2-r01/evidence/preflight_20261002.json`。
- [隔离源码 bundle 证据](D92_SUPPORT_METRIC_SOURCE_BUNDLE_IMPORT_20261002.json)。

未读取上述材料引用的历史报告、全局索引、真实 NPZ、checkpoint、packet、特征、预测、标签或评分。未执行测试、Conda、Git、SSH 或实验；未重新审查已冻结的数学 core 或本 agent 编写的 producer。metadata 的有限计数和一致性比较不调用模型。

检查时登记为 `CONFIG_ONLY`，`results=null`、`actual_runtime_commit=null`、`publication_status=NOT_LAUNCHED`。root 告知本轮相关合成验证共 191 cases 通过；这不是审查者执行或见证的数值验证，也不是任何真实指标。

## 实际声明矩阵

配置与 experiment 的 `code/execution/permissions/probe/rows` 内容一致。两个 cohort 内的 evaluator 配置均与顶层算法、六资源和本 cohort 的 selection、producer matrix 一致。

| 项目 | 每行 | 四行合计 |
| --- | ---: | ---: |
| parent | 40 | 160 |
| K1 parent | 10 | 40 |
| 有 OOF 的 parent | 30 | 120 |
| proxy anchor | 350 | 1400 |
| sequence path | 450 | 1800 |
| R0 head / candidate preparation / candidate stage / candidate final head | 各 810 | 各 3240 |
| row basis construction | 1 | 4 |
| preparation basis binding / 实际 Gram 使用 | 各 810 | 各 3240 |

行由两个模型 seed `2026092701/2026092702` 与 `rx3/rx1` 交叉组成。rx3 选择 receiver `19-1`，rx1 选择 `20-19`；每行两场景为 `practical_high/practical_low_urban`。K 固定为 `1/5/10/20`，新增类数固定为 `0/2/5/10/20`，support seed 固定为 `2026092711`。每个 selection 的 40 个 split ID 和 `(receiver,scenario,K,new_count,support_seed)` 均唯一，完整覆盖这个乘积；全部注册类数为 `6/8/11/16/26`，共享同一六旧类集合，无重复类。

每个 K1 parent 有一条完整支持路径；其余 parent 有 3 条 OOF 加 K 条 proxy 路径。因此每行路径数为 `2×5×[1+(3+5)+(3+10)+(3+20)]=450`。新增 0 只保留 B，其余新增值各有 B/C 两阶段，因此每行阶段数为 `450×9/5=810`。这些是完成矩阵的结构计数，不是已经完成的 fit 数。

原 capsule producer matrix 分别含 rx3 的 900 个 split 和 rx1 的 300 个 split。两个 selected matrix 均只是各自预声明的 40 个 split，且完全属于原 producer matrix；没有把缓存的其他 receiver、场景或 support seed 纳入本次实验。物理 support 继续使用既有 `p2_min_v1/VALIDATED_ONCE` 绑定，不增加数据重验。

## 来源、权限和继承

experiment 的 checkpoint 段（检查时第 93 至 105 行）声明两个固定 source-only、scratch、final200 来源及既有精确 source contract/SHA/native architecture 核验；本次不重新加载 checkpoint。配置中相同 model seed 跨 cohort 的 checkpoint SHA、Ground A packet 和 ground summary 路径一致。两种模型的 SHA 分别为 `f7ea5064d56711c3173b16636a52af27018ba11a459f0205de950c134362e53b` 和 `72f26413e495dcf82317898f8729caa3d35499bf0b41e9671ffdceab2c90924b`。

六种 seed 角色明确：model 如上，split/data 为 `2026092705`，augmentation 为 `2026092707`，support 为 `2026092711`，evaluation 为 `null`。不从 model seed 猜测其他随机性，也不把 evaluation 缺失补成固定随机数。

权限明确禁止 source 样本、source 逐样本特征和 replay；禁止 query IQ、view、标签、truth、角色、配额和评分进入本次诊断。历史目标评分不参与适应或选模。冻结 center 只构造 Q/U 几何，不作为 teacher target，也不增加 support。

新 ABI 为 `d92_support_metric_joint_local_ridge_v1`，方法为 `D92-ProtoFrameSupportMetric-GGN1-LocalRidge`，坐标为 `EXACT_STORED_Q_RANK_PHYSICAL_U`，实际秩 `r≤5`、名义坐标 5。禁止从旧方法 lift adapter。每行只由冻结 Q 构建一次 basis；后续 preparation 绑定同一 row basis，并计实际 `U.T@U` 工作。

B 从本次零坐标开始；C 只继承同一路径当次实际 B 的 U/theta 和冻结旧条件函数，不继承其他 parent、fold、模型或历史适应状态。new0 复用 B。旧条件函数保持不等于保证旧类胜率不下降；C 仍面对全部注册类竞争。inner-held 只作合法 support 内训练监督；outer-held 先固定预测、后作独立诊断。K1 无 outer-held 指标时记 N/A。R0 的独立 B/C 拟合与候选顺序继承明确区分。

以上来源 provenance 是登记中的核验声明。本次读取的 preflight 则记录四行 `binding=VERIFIED`、各 40 selected split、相同 Ground packet/summary/部署状态，并明确没有读取 feature 或 ground 数值。本审查未重新检查原始源样本、权重或 packet，不将声明替换成新的实数据证明。

## 算法、资源和账目

顶层算法及两个 evaluator 固定 `initial_step=1`、最多 1 次更新、最多 12 次 trial、折半系数 0.5、半径 0.5、Armijo 系数 `1e-4`。trial 为 `1×2^-j`，以真实 RMSCE 回读；不是旧 ProtoFrame 的 0.125 起点，也不是参数网格。阻尼为实际 U Gram 加类均衡 prediction Fisher，GGN 使用真实曲率；完整头导数范围包含核两端、Ridge/free-intercept、gate 和阈值。

| 必需资源键 | 固定值 | 限制口径 |
| --- | ---: | --- |
| `max_newton_iterations` | 100 | head 迭代 guard |
| `max_line_search_trials` | 64 | head 回溯 guard |
| `max_factor_buffer_bytes` | 167772160 | 160 MiB，显式 factor buffer；不是 process RSS |
| `max_integer_bits` | 65536 | 精确 basis 整数中间量 guard |
| `max_fraction_operations` | 65536 | 已 instrument 的 Fraction 操作 guard |
| `max_secular_iterations` | 128 | 度量子问题求解 guard |

六键均为显式正整数，preflight 记录与配置完全一致。guard 耗尽保留技术失败和已耗工作，不自动增限、重试、使用 jitter/pinv 或回退旧方法。整数位宽和操作 guard 不是病态输入必定成功的证明；完整 head 的严格 JVP 误差界未建立，证据等级仍是浮点子问题 diagnostic。

预登记最大 informative stage 为每行 162、全行 648；最大 trial 为每行 1944、全行 7776。有效方向按实际 rank 计，名义上限每行 810、全行 3240；rank0 不更新 adapter。`ggn_step_count/support_metric_step_calls` 是实际 step 调用口径，不是接受更新数。Fisher/secular/factor 的配置上界是执行上界，不可机械填成已发生计数。

row construction 只加一次，preparation binding/Gram、stage 和 score 分开记录，再按完整 WORK 字段 SUM/MAX 汇总。逻辑 integer payload、certificate UTF-8、numeric buffer 与 RSS 分开；头的 5 个填零方向 RHS 仍按实际求解计费，不用有效 rank 隐藏已执行的 RHS。未测设备、峰值内存、native wire/model 初始传输等记 N/A。`already_deployed=false` 的声明保守计完整 ground component 文件，不据小参数量声称训练或推理更省。

## 执行闭包和当前未完成项

唯一 launch owner 为 root；配置固定 CPU 两 lane、每 lane BLAS 两线程、CUDA 不可见。四个输出目录均在本次新 run 下且互不相同。metadata preflight 记录新 run/release/tar 路径当时均不存在；该记录不是后续时刻的实时文件系统保证。

隔离 bundle 证据记录 `VERIFIED_ISOLATED_SOURCE_IMPORT`、37 个已导入模块、46 个声明文件、`missing=[]`，包括新 basis/step/core/evaluator/preflight/run/publisher 和当前配置；同时明确 `data_access=false`、`native_encoder_loaded=false`、`checkpoint_loaded=false`、`launched=false`。这是本地隔离源码闭包证据，不是远端运行或数值验证。新独立分析在所有四行关闭后执行；不把 support diagnostic 写成 query performance。

技术失败只结束所属 lane，保留已完成阶段、partial state 和工作；健康 lane 继续，不因弱诊断停止或重试。当前健康任务不得被此新 run 停止、热改或重启。准备 parent commit `aa951242f0a14dfb7cb3d45742f90730e20111c6` 与实际发布 runtime OID 分开，后者检查时尚未产生。

检查时发现两项非阻断文字差异，已告知 root，root 将在 READY 交付时修正：

1. `experiment.artifact_plan.row_predictions`（检查时第 4553 行）写为 `<row>/probe/predictions.jsonl`；当前 support producer 的固定预测产物名为 `fixed_predictions.jsonl`。这是登记中的产物指针差异，本次未发现它驱动错误执行的证据。
2. `execution.implementation_validation` 和报告仍保留 `CORE_75_SYNTHETIC_PASS_ENTRY_VALIDATION_PENDING`/75 cases 的旧文字。root 已告知当前合成验证为 191 cases 通过，将更新为 READY、尚未 launch。本审查保留检查时的 CONFIG_ONLY 状态，不要求重新测试或重新审查这些文字更新。

实绩仍未验证：实际 runtime commit/PID、所有声明路径完成、真实 support 数值指标、独立分析闭合及任何 query 性能。这里不推断准确率提升、零遗忘、泛化成功或成本优势，也不新增 receipt、签名、审批或泛化门槛。
