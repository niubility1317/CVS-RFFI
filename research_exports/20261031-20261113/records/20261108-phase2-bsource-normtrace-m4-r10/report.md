# Phase2 B 主干：保留失败记录并诊断梯度范数

- run_id：`20261108-phase2-bsource-normtrace-m4-r10`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Reuse 9394 complete ancestor tasks, retain eight quarantines and 171 partials; run 1556 ProtoNet-CDA then 634 DADDA unstarted tasks. Cancel 1932 unfinished RadioNet.

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

## 未开始任务与失败诊断

逐项清点祖先 9,394 个完整任务、8 个隔离任务共 171 份部分预测，取消 1,932 个未开始 RadioNet；待运行 1,556 个 ProtoNet-CDA，随后 634 个 DADDA。失败时记录有限梯度的 FP64 范数和最大值，不更改训练、种子或原非有限停止条件；既有失败任务不重发，任何新失败停止派发并排空。该记录不声称故障根因已解决。每 GPU 本任务 B/C 合计最多 2 个进程，启动前空余至少 6,000 MiB；全预测固定及缺项处理前评分关闭。18 项定向测试和独立 P0/P1 审核通过；真实源 no-query smoke 为启动第一步。

## 实际派发


## 新故障：实际非有限梯度

新增闭合 283 项后，任务 277（ProtoNet-CDA→MoPC，P02/MIX02/2-1，seed 2026092702，K10/N10）在 C 阶段 2 第 11 轮第 44 步出现非有限 loss、cross_entropy、gradient_norm，以及 18 个包含非有限元素的梯度张量。此故障存在真实非有限梯度，不能解释为仅范数聚合溢出；首次非有限中间量仍未知。该任务 30 份部分预测和日志保留，控制器退出、worker 排空；原预测与隔离、后继 claim 和 RadioNet 取消范围保留，完整预测屏障及评分关闭。
