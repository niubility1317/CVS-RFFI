# Phase2：优先运行 B 主干尚未开始组合

- run_id：`20261105-phase2-bsource-unstarted-priority-m4-r04`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Pause only verified C dispatcher admission, drain C workers naturally, retain 74 complete B jobs and two quarantined partial jobs across r02/r03, and execute 13448 previously unstarted B-backbone jobs. Resume the same C dispatcher after B completion/failure. No algorithm/seed/budget changes.

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

## 预登记恢复范围

精确复用 r02 的15项和 r03 的59项完整任务；两处各33条partial仍在祖先root隔离，不重跑。其余13448项按原7B×3C×4seed×483 capsule矩阵与预算运行。C组 r08 仅暂停派发，旧worker自然完成。B组结束或技术失败排空后恢复同一C控制器，独立watcher处理协调器意外退出。GPU0–6各3、GPU7各2，两组合计；启动前显存≥6000MiB。C排空超时4小时自动恢复，不占额外GPU验证。全部隔离解决及预测固定后才评分。

23项定向测试通过。独立P0/P1指出旧C回收窗口和过早后继claim风险，已修复：暂停后核对所有launch闭合，B的完整输入检查和真实新root创建后才写后继claim；补充故障恢复测试。尚未启动。

## 实际启动核实

VERIFIED：优先协调器 PID1793346/start_ticks34568895，真实release CWD/argv已核实；独立恢复watcher PID1793355/start_ticks34569083。C r08调度器1364637/start_ticks31022133已处于T，仅暂停新派发，23个旧worker仍按原任务自然运行。当前DRAINING_C_WORKERS，B组source-only smoke、完整结果复用和新worker尚未开始，不能称已运行其他组合或提速。全部旧worker闭合后自动继续；4小时超时/准备失败自动恢复同一C调度器。唯一归档SHA和远端compile VERIFIED。证据保存在私有drain_verified.json。
