# DURA 累积量源信道轮换七景测试

- run_id：`20261109-phase1-dura-cumulant-diverse-ground-test-s2026092701-r01`
- group_id：`phase1-dura-cumulant-diverse`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保持累积量网络和总优化预算，20 轮轮换 30 组合法源模拟信道，每条源记录每轮 3 视图。

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

## 固定方案

在与固定三信道累积量版相同的 32,768 个合法源记录、网络初始化、AdamW、batch 32 和 20 轮预算下，每轮使用 3 个源模拟信道；30 个预先缓存的信道视图按 10 组三视图轮换两次。保持累计 1,966,080 次记录视图曝光和 61,440 次优化更新；额外生成冻结 DURA 的源上下文，有额外计算和 CPU 内存成本。模型训练不读取 TX 标签、F 权重或输出、V 和目标。

源元数据覆盖分析：固定 21/22/23 三个视图下 20,834/32,768 条源记录从未出现 low urban；30 视图下为 350/32,768，30,528/32,768 条见过全部七种源场景。该统计不证明训练效果提升。训练后固定 final20，依次完成七景预测和独立 truth-last；不得根据源诊断取消测试。

代码 c7aa9e4fdf65d527cdd3664cae97af1d2477a788；轮换覆盖、非法日程及 checkpoint 负例三项检查通过，独立 P0/P1 静态审查未发现阻断问题。GPU5 前驱为本任务现有七景预测 PID 2461493，其后仅 CPU 评分；新队列等待该 PID 退出并检查显存与总进程数。首批真实 checkpoint 无查询 smoke 为新运行首步。
