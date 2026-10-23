# 剩余CPU评分四路并行与快速调度

- run_id：`20261023-phase1-PAMF-cpu-balance-control-r01`
- group_id：`phase1-cric-dura-complete-system`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Resource-only handover of the unchanged frozen-prediction score queue; no new scientific experiment, training, inference or target prediction.

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

## 资源调整完成

**ANALYZED / VERIFIED**。执行 commit `4ce21e7955003375d804dbecc12b745561b2c1b0`，只修改资源调度：独立 CPU 评分最多 4 路、每路 2 线程、轮询 1 秒；完整 2296 项原任务清单、科学配置、依赖及输出路径保持。16 项聚焦检查和一次独立 P0/P1 审查通过，发布归档校验与远端编译通过，实际 4 路 CPU 进程已独立核实。

接管时还剩 33 个评分／汇总阶段，在首个完成回读时已于提交后 154 秒内全部结束。当前有效计数 2078 COMPLETE、1 个已替代历史 FAILED，无 QUEUED／RUNNING；516 个外部评分、51 个公平性评分、结构恢复评分及两个汇总均闭合。没有新增故障，7 条用户停止的旧错配 F 仍停止。控制器终态 PARTIAL_TECHNICAL_FAILURE 只保留旧 RIEI 时延导入失败，合法 latency-r02 已完成，不删除历史或重跑。

证据：`local_artifacts/cpu_score_balance_20261023/{verification_first.json,verification_final.json,completion_artifact.json}`。CPU 约 30%、可用内存约 397 GiB；未新增 GPU 任务。此处完成的是服务器计算／评分，不代表后续论文、图表和最终导出已验收。
