# P75 两匹配尾段实际运行

代码 ecdefc9bb562535bcc8087b5d59e38af25ab817a。源checkpoint/ownEMA严格加载与两条真实sixL合成图PASS：1540整网样本，192个native快照，未访问source IQ/query或做优化更新。C同IQ MSE0，N新噪声MSE>0，默认head和Sinc均有有限非零梯度，smoke模型状态未变。

两个新row已在GPU0/1启动，独立/proc核实PID/CWD/argv，nvidia-smi实际GPU映射一致，各GPU当前所有owner共1个进程。尚未读取任何新正式分数。启动原始证据和私人机器细节保留local_artifacts。源条件与矩阵不变，自动七视图与独立truth-last照既定流程。

后续独立读回确认：两行均完成尾段第1轮；两条日志均从约31KB增长至36KB，进程持续存活，且无run-owned failure。PID/CWD/argv/GPU/log增长的发布后核验完整通过。具体曲线与损失在固定预算结束后全量分析，当前不作性能判断。
