# 冻结E200独立测试：R1随机混杂诊断

- run_id：`20261009-phase1-receiver-residual-test-manysig-m16-r03`
- group_id：`cvs-cross-tx-displacement-branch-contribution`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

修复旧source role契约无classes字段的冻结检查错误；只使用对应原run全部16个已训练E200模型，不训练、不适应、不改变算法/预算；clean+六完整practical视图全部预测后独立truth-last评分。

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

## 独立测试恢复

源训练已全部完成，原pipeline在访问目标前因`expected[classes]`报错。旧角色契约没有该字段，训练导出才追加实际物理类别顺序。修复在新评估release/独占输出中执行，原模型、权重和失败产物保留。根因已本地复现，新增核验逐项比较全部原契约字段，并单独检查原固定六类顺序、receivedIQ与scratch来源。

原训练run：`20261008-phase1-receiver-residual-manysig-m16-r01`，source commit：`3712e8e4f004f21b8ffdc74ce0459ad8a2020f3d`。本轮无训练、无适应，测试全部16个ownE200。R1只作随机混杂诊断；正式收益以R2为准。所有32×7数组完成并检查后才开truth；当前尚未评分。


完整训练记录核验VERIFIED：两轮32个模型共320000步、6400个epoch，逐条解析step/epoch/CSV/161轮源L统计及完整stdout；全部E200×50=10000步。未发现训练异常或PL/EMA/域骨干意外启用，卫星CE从E80开始，辅助机制从E41开始。源类映射/原角色契约逐字段一致，核验未读取truth。证据见evidence/completed_training_audit.json。

## 固定预测与独立评分交接

PREDICTIONS_COMPLETE：本run的16×7固定数组保留。原评分在物理RX名称/编号检查处失败，未输出指标，旧failure保留。最终评分在独立run `20261009-phase1-receiver-residual-score-manysig-m16-r05` 完成，VERIFIED；无重预测。
