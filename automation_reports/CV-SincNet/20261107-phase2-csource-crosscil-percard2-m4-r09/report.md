# Phase2 第二组：每卡最多两个进程的安全接管

- run_id：`20261107-phase2-csource-crosscil-percard2-m4-r09`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Drain the quarantined C dispatcher and resume unchanged work with a maximum of two task workers per GPU.

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

## 2026-11-07 接管启动与自然排空

用户要求 Phase 2 每张 GPU 最多两个本任务进程，B、C 合计。新控制器 PID 4078896，start_ticks 51885803；原 r08 PID 1364637、start_ticks 31022133 已核实暂停新增派发。23 个旧 worker 自然完成，当前 DRAINING_PARENT；GPU 0–6 上仍有各 3 个旧 worker，GPU 7 有 2 个，尚不能声称每卡两进程已生效。全排空后执行 source-only smoke、复用完整结果并继承两个祖先隔离项，再以每卡 2 个上限派发。旧控制器不得 SIGCONT，异常保留产物和唯一后继声明。B 不得并行启动，除非共享限额覆盖两组。源代码提交 f50c42e3c33b6fbf7cbc6cff7e24d01d16223742；定向测试 27 项通过，独立 P0/P1 与针对性修正复核通过；发布包传输 SHA 与远端 compile 已核实。当前预测屏障和评分关闭。

## 2026-11-07 安全降额接管已完成

原控制器已退出，23 个旧 worker 自然排空。真实源 checkpoint 的 21 组 source-only/no-query smoke 通过；独立核实复用 35,687 个完整任务、222,137 份硬链接预测及 19 条 pilot 评分引用。两项祖先隔离各保留 10 份 partial。新控制器 PID 4078896，16 个真实 worker，GPU 0–7 各 2 个；B 组控制器和 watcher 已退出。本轮未启用完整预测屏障或评分。新 worker 已派发，首次读取尚无新增预测，继续观察实际产物增长。预登记的科学配置、训练预算和两组限制保持；不与未计入上限的 B 队列并行。

## 接管后预测增长

独立读回已闭合 35723 项，比接管时复用的 35,687 项增加 36 项；每卡至多两个本任务 worker，failed=[]，两项祖先隔离未解除。完整预测屏障和评分未启动。
