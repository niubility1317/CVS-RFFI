# D92新旧类竞争的源域分类头遮蔽代理校准

- run_id：`20260928-diagnostic-d92-registration-proxy-s2026092701-r01`
- group_id：`d92-fixed-phase1-support-cv-upgrade`；类别：`cvs`；阶段：`source-registration-proxy`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源域六个已见TX的全部20种3旧/3注册角色组合；只隐藏分类头入口，不声称真正新TX。L支持拟合、单一V评分；K1权重source-only选择。

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

Source-only proxy review: no P0/P1; synthetic all-role mapping test passed. All six TX seen by Phase1, not novel-TX evidence. Actual release commit is recorded in startup.json.
