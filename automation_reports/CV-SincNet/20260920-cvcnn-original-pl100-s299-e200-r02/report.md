# 指定9月17日CVCNN版本重跑

来源：../20260917-cvcnn-pl-clean-sg-rx4-s299-e200-r01/report.md。用户要求按该版本重跑并说明与本任务STAR E配置CVCNN区别。

新run从零训练，无旧checkpoint/EMA继承；seed299，RX0/1/2/3/5源域，RX4仅评估，TX0–9，200轮。完全复用原版模型、训练主体、数据loader及遥测；固定原数据目录的clean/Loo配对，不在线生成或修复Loo，不叠加CFO/IQ。CE对拼接2B视图整体mean，相当于0.5CE_clean+0.5CE_Loo。普通硬伪标签阈值0.9、权重0.65，自第100轮（含）启用，对通过视图mean。

|项目|用户指定9月17日版（本次重跑）|本任务此前STAR E配置CVCNN|
|---|---|---|
|网络|STAR复数编码器、后128维、确定性余弦头|相同|
|SNN/RD/Orth|全部关闭，sigma冻结|相同|
|seed/阈值/轮数|299/0.9/200|相同|
|监督CE/PL系数|clean/SG各0.5；PL0.65|相同|
|PL启用|E100|E30|
|Loo增强|固定clean/SG样本对|50%固定+50%在线动态Loo，sigma课程1→3，极端衰落训练修复|
|学习率|Adam恒定0.001|Adam基准0.001，warmup+余弦衰减|
|有标签采样|普通shuffle|RX×TX分层循环采样|
|无标签采样|原全局RNG shuffle|独立92001+epoch RNG|
|梯度裁剪|无|clip10|
|推理EMA|无|E151起decay0.99|
|原选模方式|source best-val|固定E200 EMA主/student辅助|

本次保留原版best-val辅助评估，并额外保存第200轮student为正式结果，符合当前最后轮约定；未修改训练、增加EMA或用目标选模。该评估追加是与历史版唯一有意代码行为差异。目标预测保存后再连接truth计算accuracy/Macro-F1。完整产物在各自final_best_val与final_last_epoch下；最后轮评估完成才写COMPLETE.json。

验证范围：静态编译、原训练函数AST一致性、模型/data/telemetry文本一致性、配置和输出路径检查、一次独立P0/P1审查，以及真实启动PID/GPU/log读回；沿用原报告不额外训练测试要求。原目录及已完成的其他CVCNN保留，不覆盖。唯一launch owner，独占输出；沿用当前用户突破两进程限制授权，本run最多每卡3计算进程且空闲显存至少8500MiB，不停止其他任务。

状态：准备发布，实际状态以remote_inspect.json及登记事件为准。单seed，不做显著性声明。

启动VERIFIED：GPU5，PID184795；完成epoch=17，batch日志数=1013。PID/CWD/argv/GPU、配置文件及增长日志已核实，尚未E200完成。证据remote_inspect.json。
