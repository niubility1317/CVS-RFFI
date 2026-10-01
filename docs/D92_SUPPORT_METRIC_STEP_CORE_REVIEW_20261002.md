# SupportMetric step 与联合核心限定源码审查

日期：2026-10-02。结论：**NO_UNRESOLVED_P0_P1**。本次只审查已经由作者明确 FREEZE 的新 step、新联合核心及二者接口。没有读取真实数据、模型、缓存、运行产物、配置、索引或成绩，没有执行数值测试、Conda、Git、SSH 或实验。

审查者是 `d92_support_metric_basis.py` 的作者，因此本次不对 basis 自称独立审查；也不重复审查旧 ProtoFrame 核心。对已有 geometry/head helper 的读取仅用于确认新接口传参、完整 JVP 和实际计费的连接关系。只新增本文，未修改其他作者的源码或测试。

## 已关闭的具体问题

1. **P1：secular 终止条件与最终互补残差口径不一致。** 原来的长度差条件只把半径平方差控制在约两倍尺度，解析一维边界例可能在循环结束后被最终互补检查拒绝。作者增加 `_metric_secular_ready`，使用当前原坐标方向的 `abs(lambda * (.25 - d.T @ M @ d)) / max(1, lambda * .25)` 与既有容差比较，和最终回读共享 `_feasible_direction`。没有增大最终容差、半径或预算，也没有改变目标。见 [step 源码](../code/cvsrffi/d92_support_metric_step.py#L355)，尤其第 376 至 393 行及求解循环。
2. **P1：后续 secular 尝试失败可能混入前次快照。** `_evaluate_secular` 现在在尝试开始时删除全部旧 `secular_*` 字段，先保存当前乘子/RHS，再逐阶段保存当前因子、forward 解和最终方向；`_chol` 在新尝试前清除同名旧因子。后续 factor 或 backward 失败不会把旧方向/因子与新 RHS 配在一起。见 [step 源码](../code/cvsrffi/d92_support_metric_step.py#L286)，第 330 至 352 行。作者加入第三/第四 factor 失败及第三次 secular backward 失败回归；失败后的时间与工作账仍由原 ledger 更新。

这两项修改只进行了局部源码回看，没有重新审查不受影响部分。

## 新联合核心与 step 的检查结果

| 检查范围 | 源码对应与结论 |
|---|---|
| 物理坐标 ABI | [joint 第 92 至 158 行](../code/cvsrffi/d92_support_metric_joint_local_ridge.py#L92)：状态 `theta` 属于当前 U 的 r 维坐标。仅为复用已有完整 head API 补零至五维；实际五方向组合 RHS 照实求解、计费，最后截回 r 维，不从旧 Q 坐标提升状态。 |
| 完整导数与曲率 | [joint 第 292 至 328 行](../code/cvsrffi/d92_support_metric_joint_local_ridge.py#L292)：所有物理 OOF 行先拼合，再计算 RMS 类均值 CE。传给 step 的是 `rms.curvature`，没有把旧 `I5` 阻尼叠加进去。调用保留两端 kernel、Ridge 自由截距、新类条件 log-probability 阈值和 gate 自由截距的 JVP。 |
| prediction Fisher 与实际 Gram | [step 第 190 至 248 行](../code/cvsrffi/d92_support_metric_step.py#L190)按 `1 / (C * n_y)` 对概率中心化 score JVP 构造 Fisher，未用观测标签梯度外积替代。joint 第 230 至 236、384 至 389 行使用实际 `U.T @ U`；step 分别形成 `M = physical_gram + Fisher` 与 `H = GGN + M`。 |
| 求解与真实试步 | [step 第 436 至 557 行](../code/cvsrffi/d92_support_metric_step.py#L436)通过 M 的 Cholesky 白化和有限 secular 求解处理度量球，回读原坐标约束/KKT/互补残差；不加 jitter、伪逆或裁掉小秩方向。joint 第 397 至 418 行从 eta=1 开始，至多 12 次减半；每次重新计算完整实际 heads 和 RMSCE，使用实际增量的 Armijo 项，最多接受一次更新。浮点比较容差、严格不等式结果和观测到的目标增加分别记录。 |
| 同次实际 B 继承 | [joint 第 189 至 204 行](../code/cvsrffi/d92_support_metric_joint_local_ridge.py#L189)核对新方法 B 类型、context、资源、旧物理 IDs、标签、全部原始分支和冻结字典。C 使用 B 的同一 basis 对象和 theta；full prior 指向实际 B，fold prior 仅在对应 old inner-train 重拟合，见第 247 至 276 行。 |
| 退化职责 | joint 第 241 至 244、365 至 420 行：new0 直接返回原 B 对象；K1 不制造 OOF；rank0 不求方向。后两种情况仍完成 final head，未用不更新 adapter 代替 head 拟合。单新类条件 softmax 恒为 1，因此其新头和阈值 Jacobian 为零是合法退化，不应强制非零。 |
| 输入与推理边界 | API 只接收显式合法 support 和冻结几何对象，没有文件、checkpoint、源样本或 truth 读取。每类物理 support、注册列及 ID 校验在 joint 第 172 至 201 行；第 356 至 362 行限制推理为单物理样本，并沿用全部已注册类的结构化决策。没有把 query 批统计传给 fit。 |
| 实际成本与失败 | joint 第 35 至 77 行新增 basis/绑定/step 的 SUM 与 MAX；预构造 basis 不重复收取构造成本，绑定和实际 Gram 计算另记。头五方向 RHS 不被记为五次 Cholesky。step 的 attempt/completed、三角 RHS、谱检查、secular 与诊断回读分别计费；失败保留已耗账和当前 partial。joint 第 280 至 289、323 至 328、443 至 454 行保留 cause、失败数组和已完成状态引用，归档失败不替换原异常。 |

## 验证证据与适用边界

审查者仅做源码、测试差异和本文 UTF-8 静态检查，未执行数值。以下执行事实由 root 提供；本文没有读取相应日志或产物：

- step 首次合成执行为 30 PASS / 1 FAIL，解析边界失败证据前缀 `pytest_native_activation_1790890404223986800` 保留；局部修复后 34 cases 全部 PASS，前缀 `pytest_native_activation_1790891020451775200`。
- joint 首次合成执行为 16 PASS / 1 FAIL，前缀 `pytest_native_activation_1790891040686837600`。失败是一个新类时强制阈值 Jacobian 非零的测试断言；作者未改变 production。冻结测试改为两个不同新类的完整差分，并新增单新类精确零例。root 报告重测 18 cases 全部 PASS，前缀 `pytest_native_activation_1790891239194824000`。

root 说明这些测试经官方 `CmdExeActivator` stack 激活 `ssr-gpu`，使用 Python 3.10、NumPy 2.2.6、SciPy 1.15.3 和 Torch 2.1 CPU，未通过 Conda CLI/hooks。basis 的 23 PASS（前缀 `pytest_native_activation_1790890081239776100`）属于 root 的单独验证，本文不据此把自己的 basis 实现称为已独立审查。

证据级明确为 `FLOAT64_SUBPROBLEM_DIAGNOSTIC_NOT_COMPLETE_HEAD_CERTIFICATE`：实际 Gram 与浮点 KKT 回读不等于完整 head/JVP 区间认证，`direction_error_bound` 或完整 JVP 误差界仍为 N/A。本文不证明对所有有限输入都成功，也不证明严格浮点单调、query 泛化或性能提升。

因子 buffer 峰值只覆盖声明的显式单因子输入/输出；常驻 ndarray 负载、累计返回字节、序列化证书字节、Python 有理数对象、工作区及进程 RSS 是不同口径。未测设备成本、整机峰值、能耗和传输仍为 N/A，不能从 r≤5 推导星载省算力。当前结论不新增准入门槛，不授权启动、重跑或修改健康实验。

交付状态：**FREEZE**。本次限定审查结束；没有新问题时不追加重复审查。
