# DURA 大范围修正训练

- run_id：`20261108-phase1-dura-context-wide-residual-source-s2026092701-r01`
- group_id：`phase1-dura-context-wide-residual`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

扩大冻结 DURA 上下文修正分支的逐时刻幅度上限，检验源域可达性限制。

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

## 启动核验（2026-11-08）

- 代码 4edf5c67ec48de647c5ad837f8bc2d169c961179；源 pilot 可达性依据见[源域分析](analysis.md)。新分支 14793858 参数；17 项针对性测试通过，独立 P0/P1 审查未见阻断。真实源数据与固定 DURA 上，原版 96 条输出逐位一致；新版完成 6 步 GPU 检查。
- N607 归档校验与远端编译通过。GPU 5 启动时空闲 21157 MiB、已有 2 个计算进程；队列 PID 899850、控制器 PID 899866、正式训练 PID 900066。源域 smoke 通过，正式源训练已启动。
- 固定 20 轮和最终检查点；之后自动完成 Phase1 地面七景预测，全部预测固定后独立 truth-last。扩大修正范围同时改变初始梯度尺度，当前没有正式性能结果。

## 七景正式评分完成

冻结模型的七景预测及独立评分完成：六个受损场景均低于 DURA 和原范围稳定版。低城区分别为 61.5631%、68.8012% 和 63.1601%；完整配对、TX/RX/day 见[七景分析](../20261108-phase1-dura-context-wide-residual-ground-test-s2026092701-r01/analysis.md)。源几何理想下界不能代替目标识别。
