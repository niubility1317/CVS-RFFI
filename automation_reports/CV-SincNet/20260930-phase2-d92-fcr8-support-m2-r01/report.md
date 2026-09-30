# LocalRidge与合法support函数坐标Residual8监督adapter联合适应及新类注册

- run_id：`20260930-phase2-d92-fcr8-support-m2-r01`
- group_id：`d92-fcr8-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ARTIFACTS_COMPLETE（实际状态按events.jsonl及独立证据更新）

## 目的与对照

LocalRidge为唯一最终分类器；固定DCT/GELU字典，仅学习函数坐标Z并物化原U，最多5888参数；完整123616维interaction保留半份；函数位移近端与旧教师间隔约束，B后C精确继承原U。

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

VERIFIED：单一数学驱动FCR8–LocalRidge实现完成，47个不同相关数值/入口/汇总/调度检查通过；独立P0/P1审查完成，完整phi、独占run身份及混合零带宽N/A汇总问题已修。固定字典函数坐标Z、平均函数位移近端、B原U精确继承；无参数组合搜索。已登记完整160parent，新输出与source-only缓存身份preflight VERIFIED，root sole launch owner；尚未发布，尚无FCR真实性能。

VERIFIED：已单次发布并启动FCR8，实际runtime commit33673fe02a857d790aa69a1ace17fdb9638099d0，supervisor PID247876、workers247888/247889独立读回argv/cwd匹配。首次读回14/160parent完成，rx3两row训练中、rx1两rowPENDING，没有技术错误。CPU两lane，query/source样本不读，无encoder训练；尚未完成，不报告真实性能。

VERIFIED：FCR8既有run推进到114/160parent。rx3两row各40parent已有完整probe_complete产物，原绑定/实际计数核对通过；rx1两row各17parent完成，worker263424/263930实际argv/cwd匹配，主进程247876存活。完整run尚未结束，不读取部分成绩进行分析或选模，不重复启动。

VERIFIED：FCR完整训练诊断collector实现完成，8/8合成检查通过（1.13s）。真实数据尚未收集，仅在完整独立support summary后单次只读调用；当前run继续既有训练，不热修改或重启。派生Z的task=total-keep-Z，保留动态秩与实测机制/N/A、完整事件/准备/教师、实际费用与B绑定去重，不消费外层成绩/历史/query。

ARTIFACTS_COMPLETE/VERIFIED：FCR8完整四row/160parent结束，原supervisor及worker均退出；完整marker/state/实际计数独立读回一致。实际更新2673次、头拟合30708次、latent SVD648次。未读query/源样本；完成不代表性能改善。下一步单次独立support分析与完整训练机制诊断，root sole owner，目标ACTIVE。
