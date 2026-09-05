# CVS最小实验流程

本文件落实[AGENTS.md](../AGENTS.md)，不定义方法目标、科学协议或额外gate。当前用户要求与项目协议优先于旧optimizer规则。只执行当前阶段需要的步骤。

## 八项最小流程

1. **输入权限**：按当前`项目.md`检查本次stage/query边界；Stage2主方法核对`p2_min_v1`、`VALIDATED_ONCE`、`capsule_id`、`split_id`。其他阶段和外部对比按协议适用条款执行。
2. **本地版本**：用一个Git提交固定实际代码与配置，按AGENTS完成push读回；远端修改必须来自该本地版本。
3. **聚焦验证**：完成与改动直接相关的协议负测；launcher第一步执行一次真实checkpoint无query smoke，PASS后立即继续，不创建smoke许可artifact。
4. **独立审查**：每候选一次P0/P1正确性审查，仅限会让下次真实实验跑错、越权、覆盖输出、误杀进程、无法启动或不能产生合法prediction的问题。修复后最多一次原问题定点复审；P2和文档完善不阻断。
5. **最小预登记**：`E:/type10-7/automation_reports/CV-SincNet/<run-id>/report.md`记录候选/矩阵、commit、命令、环境/CWD、输入输出/log路径、GPU、技术停止规则和预期artifact。方法简述与科学晋级条件直接写在该报告，不另设设计审批。
6. **落地**：分配不可覆盖run ID/output root和唯一launch owner；完成一次资源/路径preflight。一个release归档只做一次本地/远端SHA比较、一次远端编译；单文件同步仅校验该文件一次。按[N607操作](../docs/workflows/n607.md)执行。
7. **启动与运行**：启动后一次核对PID/CWD/cmdline/run-root、GPU和log增长；之后多row只看计数、worker、GPU、进度和确定性异常。仅执行预登记的系统技术失败停止规则。
8. **评分与结论**：prediction完整固定后，独立scorer按opaque ID连接truth。报告同row指标和预登记的科学判定，不把评分回流到该确认集的调参、选择或重跑。

这八项按生命周期发生；启动后核验与最终评分不是启动前要求。G0/G1/G2只加载当前实际消费的能力。
白名单外事项为NONBLOCKING，遇到新增门槛记录REJECTED_EXTRA_GATE并继续；不为额外门槛编写代码、证明或复审。Git已固定代码，不做成员SHA、报告SHA、signature/authority/receipt链、TOCTOU审计、重复数据验证或smoke授权。
管理员权限、破坏性操作、query隔离、所属进程核实和产物保护仍须遵守AGENTS。

## 矩阵与运行选择
默认从单seed关键Target5/Target25或更小同row可证伪矩阵开始；达到预登记科学门槛才进入多seed/完整125。已有冻结矩阵按用户要求保持；本次文档优化不更改运行中的矩阵。
健康运行继续只读监控。低性能是科学结果，不能作为系统故障停止；下一候选需要合法研发证据和原有授权，不能从只读监控或空闲GPU推导启动授权。
`VALIDATED_ONCE`重验范围由AGENTS/项目协议唯一规定，只修复失败的数据项，其余有效切片继续。

## 状态与完成
使用`LOCAL_VERIFIED -> LANDED -> RUNNING -> ARTIFACTS_COMPLETE -> ANALYZED`；技术停止记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
外部、远端、提权结果按独立读回分类VERIFIED/FAILED/UNKNOWN；exit code、超时或终端文本不能单独证明完成。
Phase1训练结束还须保存选定checkpoint身份、评估配置、clean test及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`各自指标/log。缺少任一项不得标为ARTIFACTS_COMPLETE/ANALYZED。
Phase2先确认prediction覆盖本次注册类和query，再评分。报告每row的receiver/TX、K、seed、old/seen-new/unknown（适用时）、floor、forgetting、资源和verdict，不能拼接不同run的单项最高分。
训练分析使用完整可用结构化日志，区分已实现、实际启用和缺证据机制；缺loss曲线就限定loss结论，不能补造证据或把遥测字段当新启动门槛。
完成后在原报告追加状态、同row结果、异常、解释和下一步，并按AGENTS镜像、提交、push与远端读回。

## DA与注册
联合研究使用`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`。
DA效应分别为DA1_REG0−DA0_REG0、DA1_REG1−DA0_REG1；注册效应分别为DA0_REG1−DA0_REG0、DA1_REG1−DA1_REG0。四态均定义的指标报告差分中的差分。
REG0中新类accuracy和old/new harmonic为N/A，不是0；旧类accuracy/floor、资源和延迟可在四态报告。复用冻结预测，不因命名补跑或扩展矩阵。

## 简短交接
记录目标、run ID/commit、候选/矩阵、精确命令、环境/CWD、路径、GPU、当前证据、缺项、停止规则及是否允许fresh run。后续Agent先读交接与相关证据，再执行下一步。
