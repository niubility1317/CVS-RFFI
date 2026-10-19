# 双判据一致修复的源验证

状态：ANALYZED / VERIFIED；科学结论 SOURCE_NOT_QUALIFIED。单候选、单既有模型 seed，无新训练和目标访问。

只有冻结质量门 logit≥0 且冻结 R03 相对波形改变量 q>0.02 时才输出 R03，其他情况整条保留原 IQ。质量门直接复用上一候选固定末轮 E20/4940 权重；q 阈值沿用此前 8192 条 L/U 校准并封顶得到的 0.02。本候选不调整阈值、不拟合任何参数，不用 F 置信度、真实 clean、场景或 TX/RX 标签作运行时决策。质量门 E20、R03 E40 和原 F50 的历史成本分别报告，新成本为 0 次训练更新。

这是根据此前源 V 机制诊断设计的新保守组合，复用同一源 V 面板，不冒充新的独立盲测。所有参数在本轮 V 访问前固定，不能据结果再次改变。先加载真实 R03 和质量权重，用 2 条 L_s 执行无 query smoke，PASS 后立即继续。原 F50 与默认头直接冻结读出，不另训或替换头。

单行原 1800 条源 V，分别为 clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。三种读出为 identity、R03、双判据 R03；全部 7 视图预测固定后另起 CPU scorer 接 TX 标签与配对 clean，报告准确率、Macro-F1、修复/损坏/净修复、逐 TX/RX/day 和 NMSE。

预登记源规则：全部 clean IQ 原样保留且逐条身份决策等于原 IQ；每种 practical accuracy/Macro-F1 不低于 R03、NMSE 不高于 R03。失败保留负结果，不扫阈值/扩种子/选择性重跑。本 run source-only；本对话目标测试仍只允许 clean，本 run 不访问任何目标数据。

使用 N607 GPU1，最多一个本 run GPU worker，初始化前持共享容量锁，核实空闲显存至少 12 GB 且总 owner 进程少于 2；CPU 2 线程。不可变 release/output，保留其他健康任务，尤其 R07/GPU5。非法输入、父权重、代码、输出冲突、非有限或预测缺失时保留产物并停止，不自动重试。13 项聚焦检查通过，包含上一个质量门 9 项和新组合 4 项；本候选只做一次独立 P0/P1 审查。

命令、环境、矩阵和路径见 [experiment.json](experiment.json)。


## 源域结果

全部 7 视图预测和独立 CPU truth-last 评分完成，无技术异常。clean 原样保留 1798/1800，仍改动 2 条 IQ，NMSE=0.0000466223，因此未通过波形完全保持规则。全部 clean 身份决策等于原 IQ，准确率 98.5000%，Macro-F1=98.4953%；修复/损坏均为 0。六种 practical 均不保留原 IQ，所有输出、accuracy/Macro-F1 和 NMSE 与 R03 相同，没有新增源域识别收益。完整 630 项 TX/RX/day 聚合记录及混淆矩阵见 [summary.json](results/summary.json)。

新增训练更新为 0。已解析完整 stdout，未发现异常、NaN/Inf、OOM 或警告。13 项聚焦检查和一次独立审查通过；真实 R03＋质量门两条 L_s smoke、单次归档比较、远端 compile/import 及 PID/CWD/命令/GPU/日志增长均核实。最初发布辅助脚本误用旧 run ID，被输出保护检查拒绝，未发生远端写入；纠正后发布到独立新 root。原质量门登记已恢复，旧产物保持。

方法 commit：`8730bf2f6a5173f2103f1ee412463a8a7565919e`。该 run 明确 source-only，target_reads=0。用户随后明确性能以目标测试集为准；后续另行登记冻结的完整目标 clean 配对测试，不把本源域结果当成性能改善，也不因该要求追溯改写本 run。目标比较不会调整已冻结的参数或重做源测试。

用户最新要求：后续不向另一个对话发送指令或消息。本路线独立推进输入保护与残差恢复，性能主判据为目标 clean 识别结果，源 V 为开发诊断。本轮不新增 practical 目标测试。
