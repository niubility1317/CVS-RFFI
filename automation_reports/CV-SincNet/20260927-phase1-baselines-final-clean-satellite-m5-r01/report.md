# 最后一轮权重：clean与三种residual_noeq目标域评估

- run_id：`20260927-phase1-baselines-final-clean-satellite-m5-r01`
- group_id：`phase1-final-clean-practical-target`；类别：`comparison`；阶段：`Phase1-final-evaluation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结源模型的独立最终测试；全矩阵预测固定后评分，测试结果不用于选择或调参。

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

## 启动读回

VERIFIED: dispatcher PID913497, CWD/child process/log progress independently read back; state=BUILDING_DATA. CPU only. Source40rows retain last.pt and200epoch selection.

16项聚焦测试及一次P0/P1审查通过。发布commit：`614d79e5da86925e600444a2629ad89b7010fbd3`。旧训练日志不追写，新日志格式从新launcher启动生效。
