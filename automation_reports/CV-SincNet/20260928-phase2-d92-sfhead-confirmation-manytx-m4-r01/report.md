# D92无源数据辅助分类头完整配对确认

- run_id：`20260928-phase2-d92-sfhead-confirmation-manytx-m4-r01`
- group_id：`d92-fixed-phase1-sourcefree-upgrade`；类别：`cvs`；阶段：`Phase2-joint-confirmation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：LOCAL_VERIFIED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定4个Phase1模型；仅合法support训练统一分类头，冻结teacher约束旧类；全K/新类规模/5supportseed，不读取source样本或历史评分。

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

Current authorized design and fixed loss: docs/D92_SOURCEFREE_AUX_20260928.md. 37 focused tests plus 42 lifecycle/scorer tests passed. Independent P0/P1 review passed. Source-TX auxiliary training was superseded and never launched. No new ground-statistics transfer required; all2412 predictions must finish before scoring.
