# 恢复域软监督退步后的冻结源域表现与接收机几何诊断

- run_id：`20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01`
- group_id：`phase1-pamf-restored-IQ-source-shift-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定PLControl/RawSoft/DURASoft九份F50及同G-LU，完整sourceV比较raw/H/G分类、恢复误差、类内跨RX几何和原头方向变化；只诊断不拟合。

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

## 发布前范围

最小预登记见 [preregistration.md](preregistration.md)。仅冻结源域诊断；九份最终 F50 和同 G-LU，不训练、不拟合新头、不访问目标数据。两条聚焦测试和一次独立 P0/P1 审查通过；真实 checkpoint smoke 由启动器先执行。登记已完成，尚未发布。
