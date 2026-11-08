# 可靠度加权响应恢复器：资源中断后 E07 技术续训

- run_id：`20261108-phase1-Restore-ResponseReliability-E30-s2026092701-r02`
- group_id：`source-reliability-weighted-response-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

沿用原从零训练的 E07 模型、AdamW 与随机状态，按原源日程继续 E08–E30，累计 30 轮/7410 步；独立输出并自动完成原 Main 七景。

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

## 续训队列已发布

VERIFIED / QUEUED。N607 发布包一次校验与远端编译通过；原 E07 的配置、代码身份、轮次和 1729 次更新已读回核对。独立读回确认恢复控制器存活、CUDA 为空、等待日志持续增长，尚无训练子进程或目标预测。GPU 0/1 准入需至多 1 个既有计算进程、至少 12288 MiB 空闲，并持续监测总数不超过 3；只可停止本任务子进程。实际 E07 与原冻结 F0 的源样本检查在取得资源后执行，尚不能称为已续训。

## 已恢复 E07 并开始续训

VERIFIED / RUNNING：控制器 PID 828766 在 GPU 1 原登记容量内启动本任务 worker PID 1030799，入场时已有 1 条其他计算进程、空闲显存 21302 MiB；随后核实本任务与其他 2 个进程共 3 条，未触发资源停止。原 F0、源 2 条数据、物理／细节梯度隔离和严格 E0 重载检查 PASS；E07 父模型、优化器和随机状态恢复后的实际源检查 PASS。

恢复点来自原任务第 7 轮／1729 次更新，只用于技术恢复，并非根据源验证或目标测试选取。新输出日志前 7 行为原先完成的历史轮次；当前从第 8 轮继续，累计目标仍是 30 轮／7410 次更新。两个总体七视图预测与独立评分尚未开始。
