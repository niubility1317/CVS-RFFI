# Phase2 第二组：C 主干交叉对比提高 GPU 并发

- run_id：`20261103-phase2-csource-crosscil-moregpu-m4-r04`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Continue the fixed full matrix with increased GPU concurrency.

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

## 2026-11-03 GPU 并发接管

原 controller 2199356 等待 16 个自有 worker 自然完成后退出；新 controller 2336987 完成接管。源 checkpoint 无 query smoke 21/21 PASS；独立核实复用 1626 个闭合任务、10266 个预测文件，先导评分引用保留。实际 GPU 0 至 6 各 3 个本任务进程、GPU 7 两个；启动前剩余显存至少 6000 MiB。新进程的预测产物增长；全部预测冻结后才独立评分。实际吞吐待统计，增加进程数不等于相同比例加速。

## 2026-11-04 其他方法优先接管

原调度器 2336987 等待 23 个自有任务自然结束后退出；新调度器 411365 已接管，真实源模型无 query smoke 21/21 通过。独立核实复用 23283 个闭合任务、145083 个预测文件；其中已复用的 13 条先导结果保留评分引用，其他先导结果将在对应任务复用。新队列先运行其他 B 方法，后运行 RadioNet。实际 POSTER 组合已派发，23 个工作进程运行，预测产物增长；观察时 23444/40572 个任务闭合，失败列表为空。全部预测冻结后才独立评分，资源共享及后续吞吐以实测为准。
