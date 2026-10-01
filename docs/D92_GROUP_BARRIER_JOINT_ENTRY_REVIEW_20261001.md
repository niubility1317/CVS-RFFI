# GroupBarrier 联合核心与入口 P0/P1 源码审查

审查日期：2026-10-01。结论：**NO_UNRESOLVED_P0_P1**，限定为本文件列出的冻结源码及局部修复。此结论不表示实验已发布、运行完成、性能改善或真实数据验证通过。

## 范围与独立性

本次只审查 `code/cvsrffi/d92_group_barrier_joint_local_ridge.py`、四个 `tools/{evaluate,run,preflight,publish}_d92_group_barrier_joint_probe.py`、新入口的固定配置源码和对应合成测试；数学依据为 `D92_GROUP_FACTORIZED_JOINT_DERIVATION_20261001.md` 及现行项目 Phase2 协议。入口作者明确冻结后才完成结论，随后只核对已发现问题的局部修复。

审查者是低层 `d92_group_barrier_gate.py` 的作者。本次对**联合核心和入口**独立审查，不将 gate 公式或本人既有实现重复审查称作独立验证。未读取任何真实 run、cache、checkpoint、prediction、truth、评分、报告、实验索引或交接；未运行数值测试、Conda、Git、SSH 或实验。所读配置是此次授权的新源码配置，未打开其中引用的输入文件。

## 已发现并关闭的问题

| 问题 | 局部修复及源码位置 |
| --- | --- |
| P1：新类 Ridge 的后续三角求解失败时，已执行工作未完整计费。 | 联合核心 `_ridge_triangular`、`_new_ridge_fit`、`_new_ridge_adjoint`（第 144 行起）逐次记录尝试、完成、实际 RHS 与时间；失败前完成的因子及中间 RHS 留存，不把失败尝试当零成本。 |
| P1：复用 forward cache 后再次反向失败，旧伴随数组可能覆盖本次失败状态。 | 联合核心 `_backward`（第 365 行起）在 gate 反向之前清除同批六个伴随字段及 `adjoint_arrays`，保留合法 forward 头状态。合成回归分别覆盖同 cache 成功后 gate 先失败及第二次 Ridge solve 失败。 |
| P1：原生 Ground A 的 Torch/NumPy 共享 ABI 桥可能阻断固定头推理。 | evaluator 第 494 行改为显式 float32 Python list → Torch tensor，输出经 `.tolist()`；保持真实原头、原列序和 native float32 数值路径，未改为替代头。 |
| P1：失败快照含 NaN/Inf 时，有限性检查会遮蔽原始技术失败。 | evaluator 第 399–407 行仅将真实失败 key 路由至失败归档。联合核心第 95–119 行新增本模块的失败专用只读封装和 Recorder；第 680–707 行保留原始 cause、audit、arrays，归档或日志失败单列，成功状态继续严格检查有限性。未修改旧 Affine 模块。 |
| P1：复合 factor 总数遗漏实际执行的 gate adjoint factorization。 | evaluator 第 330–338 行和 runner 第 204–205 行均纳入 forward 与 adjoint 的实际尝试数，仍分别保留完成数和细分 ledger。 |
| P1：仅验证动态预算可接受删减后的三 row 配置。 | runner 第 107、148–149 行明确固定两模型 × 两 cohort 的四 row 交叉及 160 parent，逐 row 选择仍核对完整 K × 新增类数矩阵。 |

最后一处非有限失败修复还覆盖归档设备异常：该异常不替换原始数值错误；`CALLBACK_RETURNED` 仅陈述 callback 已返回，不冒充独立文件读回成功。测试源码通过真实 NPZ callback 后以 `allow_pickle=False` 读回合成 NaN/±Inf，另检查成功归档仍拒绝非有限数组。

## 关键行为核对

- **实际 B→C 与支持集权限。** B 使用原 Margin 的合法旧 support 适应；C 继承当次同路径实际 B 和 `U_B`。inner teacher 仅在对应 old inner-train 重新拟合；held 标签不进入 teacher 或头拟合。new0 精确返回继承 B 对象，不虚构 C 头或更新。所有旧、新物理记录保留，不按重复核行去重。
- **完整微分链。** 新类 affine 头保留自由截距及全部 RHS；gate 和新类头的 K/L 端点伴随汇入几何，再传至 U/Z。外层保持既定 CE 目标、投影球和实际位移 Armijo；未追加 keep loss、曲率 floor 或方法参数。退化情形不冒充有效 held 更新。
- **预测语义。** C 使用 public 结构化预测，对全部注册类统一竞争。旧类组内使用实际 B 的顺序，组间 gate 仍可能改变旧类是否胜出，因此组内结构保持不等于全局无遗忘，也不等于能修复 B 的旧类组内错误。入口记录额外 public predict 的实际调用和耗时；未暴露的内部工作记未知，未混入另一轮 score 的账。
- **输入及三阶段配对。** 入口复用原 support-only 五块 cache 与来源契约，核对 capsule/checkpoint/model seed；Ground A 独立绑定真实原六类 packet。固定预测写入并 flush 后才连接合法 held truth。K1 无 held 证据，A 缺失记 N/A；未读取 query 特征、标签、role 或配额参与训练。
- **完整矩阵与归档。** 四 row、160 parent 以及全 K × 新增类数选择严格核对。实际训练事件、全状态、紧凑 JSONL/CSV 与各阶段物理 ID 和继承引用保留。SUM 工作量与 MAX 峰值分开；完成 parent 与失败 fit 的统计范围分开，不将部分成本伪装为整 run 完整账。
- **有限 barrier 的证据边界。** 理论中心路径 gap、实测 complementarity 与驻点/可行性残差区分。有限精度近似解没有被直接标为全局精确 dual gap 证书。gate factor buffer 上限只覆盖该内部因子缓冲，不能表示总训练内存、既存 Ridge/prior cache、进程 RSS 或星载内存上限。
- **发布控制。** publisher 在远端变更前核对当前提交与已推送远端 OID；显式依赖白名单经隔离 import 检查。release/run 独占创建、唯一 supervisor、无自动重试或健康任务干预。运行时提交与 preparation parent 分开，launch 证据不冒充实验完成。未要求额外 receipt 链、重复数据核验或旧性能门槛。

## 验证与限制

以下为根任务报告的合成验证，不是审查者自行运行或读取真实产物：最终 Recorder 修复后，完整受影响联合核心测试文件 16 项通过（`pytest_native_activation_1790866340064246400`），包含部分求解计费、两条同 cache 失败路径和非有限失败保存回归；入口局部修复后 21 项通过（`pytest_native_activation_1790866022687514400`）。此前 gate 及 scale 的 21 项通过且本轮未修改（`pytest_native_activation_1790864090119972200`；该次合计 32 项另含当时联合核心 11 项）。本次没有为已有通过的未变行为追加重复测试。

本文件只提供有界源码正确性结论。真实数据权限链的实际输入核验、真实资源测量、完整独立结果复算及 query 泛化结论均不在本次证据内。没有新增性能硬门槛或批准步骤，也不授权发布、启动、停止、重跑或改变固定方法。
