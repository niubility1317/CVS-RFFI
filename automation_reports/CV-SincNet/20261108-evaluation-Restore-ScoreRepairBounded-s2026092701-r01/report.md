# 有界投影恢复器：固定预测独立评分修复

- run_id：`20261108-evaluation-Restore-ScoreRepairBounded-s2026092701-r01`
- group_id：`source-input-bounded-projected-prior-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原版完成训练并固定两总体七视图，评分因缺少原始评分模块在连接真值前失败；仅对既有预测独立评分。

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

## 固定预测评分恢复

父任务两个总体全部七视图预测均已固定；原评分在导入 pamf_repl_view_shuffle 时失败，未进入原评分器、未读取 truth，也未产生 score 文件。本 run 只使用固定预测和原 F0 source release 的原评分器，CPU、共享评分锁、独占新输出目录；先检查完整矩阵、识别器身份和原失败原因，再由原 scorer 在读取 truth 前核对物理 ID 和视图。零重训、零模型前向、零预测及目标拟合。

## 独立评分和复核完成

VERIFIED：固定父预测的独立 truth-last 评分完成；完整 42 单元、1302 分组、868 raw 配对及匹配对照复核通过。详见 analysis.md 与聚合 CSV；不重训、不重预测、不修改默认 DURA。

评分进程 PID 1736602 已退出，独立输出目录的 complete.json、score/complete.json 和 results.json 均已读回；实际日志位于 releases/20261108-evaluation-Restore-ScoreRepairPair-s2026092701-r01-release/bounded_score.log。一次发布包传输校验、远端编译、固定预测和原评分器无 truth 检查通过。
