# LocalRidge与监督谱adapter联合适应及新类注册

- run_id：`20260930-phase2-d92-joint-spectral-support-m2-r01`
- group_id：`d92-joint-spectral-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

LocalRidge为最终分类器；支持集内折监督训练两个有界谱参数，B后C顺序继承，与原方法、固定度量和C重置路径完整配对。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

联合谱adapter已完成55项不同合成检查；原始失败证据保留，theta0布局已修复。四路径完整配对，固定160个support任务；practical residual绑定预检VERIFIED。当前PLANNED，尚未发布或启动；无真实联合性能结论。详细算法及API：docs/D92_LOCAL_RIDGE_JOINT_DESIGN_20260930.md；独立审查：docs/D92_LOCAL_RIDGE_JOINT_REVIEW_20260930.md。

VERIFIED：独立读回确认supervisor645034及两个CPU worker645046/645047存活，argv/CWD匹配；runtime commit38d407699816f125ac8ccf7fb0e4f8c33039ba31。4row中rx3两行训练中、rx1两行待运行。无query/source读取，完整结果未完成，不能以partial指标评价或改配置。

VERIFIED：四row/160任务完成并独立全量分析，936训练阶段/7488更新/29875头拟合；R_joint与R0报告准确率及H一致，无晋级或query评分。OOF新类存在任务B71.041667%、C旧63.767361%、C新54.846354%、H58.416796%。全日志核对795阶段参数改变、6340非零更新；平方距离收缩上界最大0.1748045%，训练有效执行但没有外层准确率收益。下一轮通道adapter联合设计仅DESIGN_ONLY_NOT_FROZEN_NOT_RUN；目标ACTIVE。
