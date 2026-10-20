# R51：指定 35.05 万恢复器的源域 C4 相位一致性检验

状态：LOCAL_VERIFIED，尚未发布。18 项聚焦检查及 CLIhelp 通过，一次限 P0/P1 审查发现 code_commit 接口错误，已修复并完成原问题一次定点复审 PASS。

本检验服务性能优先的压缩优化，网络仍为用户指定 R46 prune_refine350521，不选择新的父/宽度。复用原 source 7/63/30 与 R07 缓存，从 6300 Ls/56700 Us 各均匀无放回抽取 64 条（固定 seed2026102051）；clean、received01、received20 各测试 0/90/180/270 度精确 I/Q 交换与符号变换。对同一个冻结完整接收链 G，先逆转输出坐标，再比较 G(x) 和 R^-1 G(Rx)，以及两者相对 paired clean 的 IQ NMSE，全部三 stage 均报告。

模型固定为 R46 方法 bbb8daf817de050b958d217b70d4c33f4aefe341、source-final E20/4940、350521 参数；继承原 R44/R38/R17/R07/R04 源域来源，original7/63/30 与已核实角色/物理 ID 完全一致。源缓存 READY/EXACT_MATCH 直接复用，不增加数据重验或哈希链。来源已在父及 R48/R49 报告核实，launcher 再消费同一 parent/source 句柄并在真实两条 Ls 上执行完整 G 无 query smoke，PASS 立即继续。

128 唯一源样本、1536 主检验前向样本、额外 2 Ls smoke，3 视图×4 旋转×3 stage×overall/Ls/Us=108 聚合单元。每例误差除以 paired clean 的 I/Q 平均能量、下限1e-4；报告均值、中位、P95、最大值。零旋转等变误差必须精确为0。预先定义两 received 视图、三个非零角度、最后 stage 的等变 NMSE 均值>0.001 为“在此源子集存在非微小相位依赖”的描述条件；不是 TX 保真、跨 RX 泛化、正式准确率或训练晋级证明。无统计显著性声明，不以该阈值改动现有结果。

CPU2/0GPU、FP32/TF32关闭、eval/inference_mode，所有参数 requires_grad=False，最终逐张量核对原状态不变；0训练/新网络版本/新头/F/TX标签/V/target/truth。Us 仅使用 IQ 和 L/U 角色，隐藏 TX 真值。微小源域诊断不会替代已完成的全部八版正式七场景测试。父是用户在历史结果揭示后指定的工程基准，不称盲测。该坐标变化检验仍不能证明每个恢复细节来自输入 TX，原 lowurban70%、其他五景距 rawclean≤5pp 及更强 TX 细节目标保持 ACTIVE。

唯一 launch owner codex/root/R46-source-phase；新 run/release/output 不可覆盖。为不叠加本任务 CPU 负载，在原 R50 评分及其独立收集结束后启动；不修改、停止或重启其他 owner 或现有健康任务。一轮资源/源句柄预检、单归档两端 SHA、远端 compile、实际 PID/CWD/argv/空 CUDA/日志读回按既有最小流程执行。技术失败保留本 run 输出，记录失败，不自动重试/热改/回退或误杀。

REJECTED_EXTRA_GATE：brainstorming 的额外设计审批要求与用户指定八项最小流程冲突，不实施或等待该额外 gate。固定问题、矩阵、18 项聚焦检查与一次 P0/P1 审查均保留。完整命令、环境/CWD、路径、seed、停止规则及预期产物见 experiment.json。
