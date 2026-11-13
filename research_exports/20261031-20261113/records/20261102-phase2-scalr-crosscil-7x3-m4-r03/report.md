# Phase2 第二组：保留故障并复用已完成任务，续跑未启动组合

- run_id：`20261102-phase2-scalr-crosscil-7x3-m4-r03`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Continue only unstarted tasks, reuse 15 closed r02 jobs, defer failed job and its 33 saved partial rows; global scoring remains blocked until failure is resolved.

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

## 启动与限制

2026-11-02：源代码 commit 26f04394f4a4a4c3d07d4c68ebdc25755471a17d，独立关键缺陷审查及定向复核通过，25 项本地测试通过；发布包传输校验及远端编译通过。七个真实源检查点无 query smoke PASS。controller 1623650 已核实运行，15 个 r02 完整预测任务已绑定复用；故障任务及其 33 个 partial 预测文件保留并延后，不重试。13508 个未启动任务派发中。全局预测和独立评分屏障保持关闭，不能将本运行称为第二组实验完成；目标数值故障尚未复现。
