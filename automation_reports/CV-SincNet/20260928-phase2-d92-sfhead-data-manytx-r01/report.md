# D92无源辅助确认的独立received构建

- run_id：`20260928-phase2-d92-sfhead-data-manytx-r01`
- group_id：`d92-fixed-phase1-sourcefree-upgrade`；类别：`cvs`；阶段：`Phase2-data-preparation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：LOCAL_VERIFIED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

RX20-19、26TX各180独立物理记录；3场景support30/query30，300划分。仅数据可用性选取，无模型评分。

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

Preregistered one-time fresh received builder; support pool30 and query30 per scene, three physically disjoint scenes. No checkpoint/source data accessed by builder.
