# Phase2 B 主干隔离 DADDA 注册故障后继续其他方法

- run_id：`20261105-phase2-bsource-dadda-quarantine-m4-r06`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

沿 r02 至 r05 复用 3,281 个完整任务及 62,988 份预测。原四处隔离保留 129 份部分预测，新增 DADDA 故障目录保留 fit 日志和零预测。10,238 个未开始任务先排 Two-stage、Hu、POSTER，再排剩余 DADDA 和 ProtoNet-CDA，RadioNet 最后。保留原科学配置和停止规则。

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

## 恢复核对与并发

已用 r05 同一版本入口清点 r02 至 r05 四代产物。3,281 个完整任务按引用复用，五处隔离原样保留。10,238 项未启动任务先安排 Two-stage、Hu、POSTER 共 5,796 项，再安排 DADDA 923 项、ProtoNet-CDA 1,587 项，最后 RadioNet 1,932 项。C r08 仅暂停派发，旧 worker 自然结束后开始 B 的仅源检查。B 排空后自动恢复同一 C 调度器，异常退出由独立 watcher 恢复。未解决隔离前保持两组预测屏障及评分关闭。

代码沿用 r05 的既有提交。20 项调度和恢复测试通过，配置独立 P0/P1 审查通过。此前故障的非有限损失及梯度已有证据，首个非有限中间量待仅源与 support 诊断。

## 启动核实


## 派发与复用核实

真实源 checkpoint 的仅源检查 7/7 通过，target/query 均为 0。独立核对四代祖先的 3,281 个完整任务及 62,988 份预测按引用复用。五处隔离仍在原目录，保留 129 份部分预测，DADDA 故障任务仍为零预测。逐项比对 10,238 个派发描述及两代后继声明。队列先 Two-stage、Hu、POSTER，随后 DADDA、ProtoNet-CDA、RadioNet。观测时 Two-stage 新闭合 322 项及 6,006 份预测，活跃 GPU worker 23 个。GPU0–6 各 3、GPU7 共 2。C 原调度器仍按计划暂停派发，C worker 已全部排空。两组预测屏障和独立评分仍关闭。

## 第二处 DADDA→MoPC 数值故障

任务 5944（DADDA-SDA，seed 2026092703，P02/MIX05/2-1）在 GPU2 的 K1/N20 MoPC 注册第 2 阶段、第 9 轮、第 9 步停止。总损失、交叉熵、梯度范数和多个梯度张量元素出现非有限值。100 条 B 阶段日志记录的浮点标量均有限。两处 DADDA 故障的最初非有限中间量及共同根因仍待定位。保留 K1 的 N2、N5、N10 三个 C 方法共 9 份部分预测。

B r06 停止派发后自然排空，新增闭合 5,959 项，其中 Two-stage、Hu、POSTER 各 1,932 项，DADDA 163 项。五代 B 运行共闭合 9,240 项、174,195 份预测。六处隔离共 138 份部分预测，另有 4,278 项未启动，包括 DADDA 759 项、ProtoNet-CDA 1,587 项和 RadioNet 1,932 项。后继声明保留。C r08 原调度器已恢复，已闭合 35,110 项，真实 worker 23 个，未发现新失败。两组预测屏障和独立评分仍关闭。

保留失败任务与原参数，继续以 source/support-only 诊断。下一次 B 恢复须沿 r06 及祖先复用完整产物，精确保留隔离及未启动任务。
