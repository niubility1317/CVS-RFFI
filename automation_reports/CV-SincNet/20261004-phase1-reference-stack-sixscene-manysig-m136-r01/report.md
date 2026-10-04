# reference_response剩余136行：clean及完整六residual环境自动测试

- run_id：`20261004-phase1-reference-stack-sixscene-manysig-m136-r01`
- group_id：`cvs-reference-phase1-stack-sixscene-evaluation`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

等待父136行全部训练、源域冻结及既有测试收尾后，自动评估全部固定E200模型的clean与六个完整residual星地视图；每视图168000物理query。

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

## 自动测试设计

父实验136行完成训练、五阶段源域选择与原测试收尾后，独立等待进程自动启动本评估。等待阶段不占GPU，不读取query或truth，不改变健康训练。136个固定E200模型全部评估，不按旧测试分数筛选。每行clean及六个环境各完整168000条，共159936000次预测。

六环境为郊区高/中/低仰角、城区高/中/低仰角，统一residual/post_sync/noeq。直接复用20261002已验证的received_views，物理ID与原clean capsule一致，不重新生成信道或重复builder验证。模型加载前重新核对本行scratch来源、源L/U/V、E200预算及完整冻结记录。

最多16个预测任务，每卡最多2个。全部136×7组预测固定后，独立评分进程连接truth，计算总体/RX/TX的Accuracy、Macro-F1和混淆矩阵；第二实现bincount逐项复算。最终输出四seed均值/标准差、最差RX和固定配对差值。测试不反馈模型、超参数、候选排序或重跑。

等待父实验失败时，本评估保留失败记录并停止，不自动重试。完整六环境与原3环境分层不同；Phase2适应/K/新增类/H均N/A。当前尚无本评估预测或评分结果。4项定向测试和独立P0/P1审查已通过。
