# Phase2 第二组：其他方法优先、RadioNet 暂缓

- run_id：`20261104-phase2-csource-crosscil-priority-m4-r06`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Prioritize all non-RadioNet methods, retain the complete matrix, and run RadioNet after their completion.

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

## 2026-11-04 其他方法优先接管

原调度器 2336987 等待 23 个自有任务自然结束后退出；新调度器 411365 已接管，真实源模型无 query smoke 21/21 通过。独立核实复用 23283 个闭合任务、145083 个预测文件；其中已复用的 13 条先导结果保留评分引用，其他先导结果将在对应任务复用。新队列先运行其他 B 方法，后运行 RadioNet。实际 POSTER 组合已派发，23 个工作进程运行，预测产物增长；观察时 23444/40572 个任务闭合，失败列表为空。全部预测冻结后才独立评分，资源共享及后续吞吐以实测为准。

## 2026-11-04 非有限故障

Hu→MoPC、seed 2026092704 的 P01/MIX04/2-1 和 P01/MIX06/8-8 在 K10/N10 注册时报非有限训练项。任务停止派发并排空，30572/40572 个任务已闭合；两个未闭合单元各保留10条预测。尚未固定全部预测，独立评分未启动。故障原因待只使用合法 source/support 的诊断；不按目标得分改变方法或重算已闭合结果。
