# Phase2 第二组：每卡最多两个进程的安全接管

- run_id：`20261107-phase2-csource-crosscil-percard2-m4-r09`
- group_id：`phase2-group2-csource-bc`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
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


## 2026-11-07 安全降额接管已完成


## 接管后预测增长

独立读回已闭合 35723 项，比接管时复用的 35,687 项增加 36 项；每卡至多两个本任务 worker，failed=[]，两项祖先隔离未解除。完整预测屏障和评分未启动。

## 2026-11-08 按用户要求取消未完成 RadioNet

用户取消两组全部尚未完成的 RadioNet 工作，并要求保留已有结果。C r09 调度器与经身份核验的在途 RadioNet worker 已停止；原 root、已有完整预测、已产生的部分预测和日志均未删除，父后继声明保留。停止前独立清点 C 组闭合 35,983 项，其中 RadioNet→MoPC 1,209 项；其他 RadioNet 未完成项取消，不作为完整矩阵结果。Hu→MoPC 原两处隔离继续保留，各 10 份 partial；全预测屏障及评分关闭。worker_state 中 RUNNING 与 active 是停止前快照，实时状态以进程读回为准。
