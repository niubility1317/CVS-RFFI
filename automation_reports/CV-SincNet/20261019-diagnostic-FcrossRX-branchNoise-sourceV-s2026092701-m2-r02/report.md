# P74：冻结默认头的源域噪声贡献归因

状态 **ANALYZED_SOURCE_ONLY / VERIFIED**。已完成两个原P43冻结E30模型的全部V1800×2draw×mid/low读出、分层分析、三PNG/三PDF和独立168组整体混淆矩阵/配对效应复算；无新训练或目标数据访问。

[完整结果与目标判定](final_analysis.md)、[预登记](preregistration.md)、[实现/验证](validation.md)、[原始矩阵](experiment.json)、[生命周期](events.jsonl)。运行代码 b49877aba4095b15c2e9f01a3bf3b4d3d33ab626。

单个分支的预定噪声解释和屏蔽规则均未通过；不追加分支屏蔽/对齐或E50。源去噪反事实不作为模型性能提升。既有三seed完整As的low正式增益+17.2849pp，但clean−.3210pp，跨seed仍不满足≤.1pp阶段目标，没有重复双提升证据。

新训练0epoch/0更新；32404整网样本与219652额外head-only样本；新模拟0。源读出29.8733s，分析6.8874s，r02进程自然退出且无failure。r01启动缺Path导入在0NN/CP/IQ前退出，原失败目录保留；一次有界技术修复使用独立r02根，数据/矩阵/阈值不变。

完整目标仍ACTIVE；low>60、跨seed clean保持和可重复双提升未完成。逐条件/逐状态表及图片的实际本地路径见完整报告。
