# 有界接收响应校正三种子源诊断

状态：RUNNING_CALIBRATION_COMPLETE_F_PENDING。只研究固定F即用式前处理。完整PAMF／单SIG_TIME／联合默认头和对应三个真实E20复用；不新增F训练、头拟合、DURA或目标测试。当前协方差矩阵保持健康完成，新诊断GPU部分只等待其完成状态，不读取目标分数。全部U/RX统计为一次确定性CPU拟合，随后三份模型都完成V27000 clean/三H的raw／校正成对读出，共648000个默认头决策。

实际固定算子、源规则和解释边界见[源计划](../../../docs/goals/PAMF_BOUNDED_RESPONSE_SOURCE_PLAN_20261027.md)；完整矩阵/代码/命令/环境/CWD/数据角色/父源/预算/资源/停止规则见experiment.json。校准只用全部U56700和已知RX，不用U真TX、F输出、V或target。每seed先真实L2 raw/corrected checkpoint smoke，再V；负源结果不取消其他seed。源V不更新任何持久状态。

固定校正强度0.1，增益±0.5dB、相位±0.025rad，单包质量门控；保存12项源统计，神经网络新增参数0。源规则只证伪固定F直接前处理，不把通过或失败扩大为新训练收益／不可能性。三seed、完整预算和原全面强于PL的最终目标不因该诊断缩小。local检查PASS和计划不是已运行／已获性能结果；发布后独立读回实际状态。

本地8项针对性检查与模块编译PASS。唯一独立P0/P1发现读出器参数计数错误：整native的863730不能用于encoder子模型；已改查SIG_TIME18976、原头[6,160]和两份有限[2,6]输出，总reader参数只记录，定点复查PASS。无其他P0/P1。实际三份checkpoint的L2 smoke仍在发布后首步执行，不把本地构建当成真实权重证据。


实际执行commit `f461a04dc254e5581c4847530ad1b37632f50fa4`，archive传输比较VERIFIED、远端compilePASS，controller实际PID/CWD/argv已独立核实。CPU校准完整222批U56700（末批124），五RX各11340，实际U256响应smokePASS，完整stdout无致命标记；耗时2.8375s，0F/G权重、0U真TX、0V/target、0optimizer。首次读取时CPU worker已自然结束，未为获取存活快照重跑。

两个RX主方向保留99.9912%的五RX均值差异能量。该比例只是用于拟合的4D响应统计，不是原IQ能量或独立泛化证据，更不是身份保护或识别收益。12项固定统计保存在远端私有npz，未提交其数值或样本描述。

当前1个校准job完成、3个冻结F评估排队，controller健康；三份实际checkpoint的L2 smoke尚未执行。GPU部分继续等待原协方差矩阵完整结束，只读取其完成状态，不消费分数。当前没有新F训练或性能结论。[CPU校准及运行核实](calibration_summary.json)。
