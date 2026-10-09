# 完整冻结预测独立评分：R1混杂诊断

- run_id：`20261009-phase1-receiver-residual-score-manysig-m16-r05`
- group_id：`cvs-cross-tx-displacement-branch-contribution`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定并复用已经完成的16×7预测，修复接收机编号与物理名称的评分口径；CPU独立truth-last评分，输出完整RX/TX/day分层。无训练、无重新预测、无目标反馈。

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

## 评分恢复

原32×7预测全部固定后，评分器把truth中的物理RX名称当作编号而报错；没有写出评分结果。原预测与失败目录保留，本轮只在独占目录完成CPU评分。映射来自原数据集rx_list，不从性能推断。正式结论使用R2，R1为随机混杂诊断；所有负结果保留。

## 最终独立测试结果

ANALYZED / VERIFIED：16模型×7视图，各168000样本。1568条总体/RX/TX主结果+448条完整日期结果。全部混淆矩阵指标、分区与四seed统计经本地独立重算。

[完整本run结果](results/full_results.md)、[逐seed/RX/TX原始CSV](results/scores.csv)、[逐seed/day CSV](results/day_scores.csv)、[混淆矩阵JSON](results/scores.json)。

原r03/r04预测保持固定；无重新训练/预测、适应或目标反馈。R1仅为随机混杂诊断；R2为正式四seed消融，四seed不声明统计显著性。Phase2/K/新增类/H均N/A。
