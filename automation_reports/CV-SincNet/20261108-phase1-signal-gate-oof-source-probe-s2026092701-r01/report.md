# 未见记录门控的源域输出诊断

- run_id：`20261108-phase1-signal-gate-oof-source-probe-s2026092701-r01`
- group_id：`phase1-signal-gate-oof-source-probe`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定父模型未见源记录训练的门控，在不重叠源 pilot 检查权重与波形误差，不读取 TX 或目标。

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

## 运行方案

冻结 DURA、父修正器和使用父未见源记录训练的 final20 门控，在 672 条先前已探索 source pilot 上统计波形 NMSE 和非零修正的门控权重。代码显式确认 pilot 与父模型及门控的拟合记录均不重叠；DURA 历史重叠未知。不读 F 权重/输出、TX 标签、V 或目标。

代码 `677b4ccfbb9baa8c825da8e63a2af15ae1888b57`；两项针对性测试、独立 P0/P1 审查、一次远端归档校验及编译通过。N607 CPU 2 线程、PID 1012067，8 条真实源记录 smoke 通过；此诊断不改变已启动的七景正式测试。

- 完成 672 条源记录检查；受损 576 条中 α ≥ 0.999 的有 317 条，前版为 576 条。源波形分组有升有降，详见 [analysis.md](analysis.md)；不是目标识别准确率。
