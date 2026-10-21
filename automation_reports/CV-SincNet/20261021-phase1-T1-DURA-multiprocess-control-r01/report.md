# 最终 T1＋DURA 多进程 GPU 调度接管

- run_id：`20261021-phase1-T1-DURA-multiprocess-control-r01`
- group_id：`phase1-cric-dura-complete-system`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Resource-only controller replacement; all 769 original jobs/configs, scientific worker release 3e880690, healthy PIDs, completed outputs and scientific budgets retained. No new model experiment.

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

## 接管与资源约定

本次仅更换调度器。保留原 769 个任务、三种子、T1 13L＋37U、全部 G 谱系和七视图 truth-last。科学 worker 的 CWD/代码仍固定 3e880690；不重启健康训练，不重复完成任务。

每卡最多 2 条本 PAMF worker、3 条总计算进程，启动前空闲至少 12000 MiB；每卡最多 1 条 G/G_student。外部 360 行未全部 ANALYZED 前，GPU2/3 保留原对比队列容量。正式时延保持独占；本次检查已见独立时延 r02 的 500 个 case-round 全部完成，释放后的 GPU7 可正常参与多进程训练。

只在 PID、完整命令和 CWD 全部匹配后撤下原健康调度器，扫描并接管其原 worker（包括 CPU scorer）。独立时延恢复、外部源数量队列和其他 owner 均保持。科学 smoke 复用原实际权重 no-query PASS；新增资源规则用聚焦测试与一次 P0/P1 审查验证。
