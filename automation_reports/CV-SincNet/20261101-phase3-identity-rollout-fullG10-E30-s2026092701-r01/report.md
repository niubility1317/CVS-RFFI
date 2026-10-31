# 全量 G10 源配对的完整轨迹恢复网络

- run_id：`20261101-phase3-identity-rollout-fullG10-E30-s2026092701-r01`
- group_id：`phase3-identity-rollout-fullg10`；类别：`cvs`；阶段：`Phase3-restoration-source`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

独立从零训练两条四步生成轨迹，使用原 63,000 条 L_s/U_s 的固定 G10/clean 配对；完整七景与 Phase3 truth-last 对照。

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

## 排队控制器停止：VERIFIED

独立审查发现本控制器未识别另一对话的部分训练和预测进程。已核实并仅停止本任务等待中的 PID 2156846；当时训练目录不存在、未做 smoke、未启动 GPU 或训练。原日志保留；修正资源识别后使用独立 r02，不覆盖或重复启动 r01。
