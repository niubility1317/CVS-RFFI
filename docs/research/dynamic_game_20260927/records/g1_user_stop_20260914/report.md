# G1用户停止记录

用户明确要求停止G1_FISHER_GATE。结果：**VERIFIED / STOPPED_BY_USER**。

2026-09-14 11:54:18（UTC+8）独立远端读回确认：主进程612456及16个所属子进程均不再存活，GPU中无所属PID；原批次dispatcher亦已退出，不会从该调度器重新启动G1。

操作前核对主进程实际argv中的output_dir、CWD、UID和PID启动时间，沿PPID收集所属子树；仅向这17个已核实进程发送SIGTERM，没有使用SIGKILL，没有终止无关任务。批次原调度器自行处理子进程退出并结束，未额外发信号停止它。

停止时最后完整训练记录为E160/200。E80–E160共9个测试点、对应9份epoch checkpoint、完整日志、预测和评分全部保留，没有删除或覆盖训练产物。未完成E200，不宣称有最终E200模型。最后E160测试Clean=76.7399%，三LEO均值=66.9883%。

用户停止原因保存在该row下的user_stop_20260914.json。通用dispatcher可能将进程被终止映射为TRAIN_FAILED；科学/操作报告以明确用户停止记录为准，不将其解释为新数值故障或自动修复授权。

证据：[停止记录](stop_result.json)、[独立读回](verification.json)。
