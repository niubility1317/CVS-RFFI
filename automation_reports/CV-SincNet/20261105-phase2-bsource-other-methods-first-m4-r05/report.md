# Phase2 B 主干先运行其他未开始的方法

- run_id：`20261105-phase2-bsource-other-methods-first-m4-r05`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

沿 r02、r03、r04 复用 341 个完整任务，四处隔离共保留 129 份部分预测。13,179 个未启动任务先排 MRIOR、DADDA、Two-stage、Hu 和 POSTER，再排 ProtoNet-CDA，RadioNet 最后。记录后续注册数值异常的字段和步数，保留原训练、预测和停止规则。

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

## 恢复范围和启动规则

祖先 r02、r03、r04 共 341 项完整任务和 6,918 份预测。四处隔离的 129 份部分预测仍在原目录。其余 13,179 项保持原矩阵，前 9,660 项属于 MRIOR、DADDA、Two-stage、Hu 和 POSTER，随后是未开始的 ProtoNet-CDA 1,587 项和 RadioNet 1,932 项。新协调器只暂停已核实的 C r08 调度器派发，等待已启动 worker 自然闭合。B 队列结束或技术故障排空后恢复同一 C 调度器，独立 watcher 覆盖协调器意外退出。所有预测固定且隔离解决后才独立评分。

33 项定向测试通过，独立 P0/P1 通过。故障日志新增注册步骤和非有限字段，合成梯度验证可区分范数统计溢出和梯度元素非有限。现有四处 B 故障及两处 C 故障仍待定位。

## 启动核实

已独立读回协调器 PID 1878069/start_ticks 35415026 和 watcher PID 1878078/start_ticks 35415211 的 CWD 与命令行。C r08 原调度器 PID 1364637/start_ticks 31022133 只暂停派发，23 个旧 worker 自然排空。状态为 DRAINING_C_WORKERS，B 组仅源 smoke 和新预测尚未开始。单份发布归档 SHA 与远端 compile 已核实。

## 实际派发与复用核实

源 checkpoint 的仅源 smoke 7/7 通过，target/query 均为 0。独立核对三代祖先 341 个闭合任务和 6,918 份预测按引用复用，四处隔离的 129 份部分预测保留在原目录。核对 13,179 个调度文件，MRIOR、DADDA、Two-stage、Hu、POSTER、ProtoNet-CDA、RadioNet 依次进入队列。观测时已新闭合 MRIOR 506 项及 10,113 份预测，共 23 个活跃 GPU worker，GPU0–6 各 3、GPU7 共 2。C 调度器仍处于预期暂停状态，C 在途 worker 为 0。两个组的预测屏障及评分保持关闭，六处隔离故障待定位。

## 技术故障与保留产物

DADDA-SDA→MoPC-HR 的 seed 2026092703、P01/MIX05/7-7、K1/N2 在 GPU2 注册第 1 阶段、第 19 轮、第 19 步停止。日志明确记录总损失、交叉熵和梯度范数非有限，且多个梯度张量含非有限元素。该次故障涉及实际梯度张量，根因待定位。任务未生成预测，已有 fit 日志及任务目录保留。

B r05 停止派发并自然排空，新增闭合 2,940 项，其中 MRIOR 1,932 项、DADDA 1,008 项。r02 至 r05 四代运行 共 3,281 项闭合、62,988 份预测，五处隔离仍保留此前 129 份部分预测，10,238 项未启动。两代后继声明保留。B 无残留进程，C r08 原调度器已恢复，当时闭合 35,087 项且有 23 个真实 worker。两组预测屏障和独立评分仍关闭。

后续以 source/support-only 输入定位首个非有限中间量，保留原 seed、预算和方法。下一 B 运行须从 r05 沿完整链复用闭合结果，保留全部隔离，避免重发失败单元。
