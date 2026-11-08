# 有界投影恢复器：GPU 6 上的第 26 轮技术续训

- run_id：`20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r04`
- group_id：`source-input-bounded-projected-prior-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原 r03 队列未提交训练；使用授权 GPU 6 恢复原 E26 模型、AdamW 和随机状态，续训 E27–E30 并完成既定测试。

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

## 资源切换与实验边界

复用原 r02 第 26 轮的模型、AdamW、CPU/GPU 随机状态和源顺序；继续第 27 至 30 轮，累计 7410 步。r03 未提交任何训练或预测，发布前仅精确停止该队列控制器。新 run 仅变更调度，数据、损失、随机种子、冻结 F0、全量七视图和独立 truth-last 不变。

GPU 6 已获用户授权总计算进程最多 5 个；本任务至多一个训练或预测进程。入场最多 3 个其他计算进程、空闲显存至少 12288 MiB；动态监控总数超过 5 或无法核实则只停止本 run 的子进程并保留输出。实际源样本、冻结 F0 及 E26 恢复检查为训练第一步。

## 停止准备

原 r03 队列在本方案准备期间自动获得 GPU 2 并已提交训练，PID 1311999；真实源样本与 E26 模型、优化器、随机状态恢复检查均通过。因此本 r04 未发布、未启动，保留原 r03 健康运行，避免重复训练。为避免保留未使用的执行路径，r04 的调度代码和临时验证已从当前版本移除；历史准备提交仍可追溯。
