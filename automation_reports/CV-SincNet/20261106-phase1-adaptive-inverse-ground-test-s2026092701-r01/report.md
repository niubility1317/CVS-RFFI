# 独立自适应复数滤波网络 Phase1 七景测试

**2026-11-08 状态更新：七景预测、独立评分和 DURA 配对均已完成。完整结果见[七场景分析](analysis.md)。下方“尚无结果”文字保留为历史记录。**


- run_id：`20261106-phase1-adaptive-inverse-ground-test-s2026092701-r01`
- group_id：`phase1-adaptive-inverse`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

逐窗口估计 33 抽头复数和共轭滤波器，结合独立局部波形残差；从随机初始化训练，无 DURA、F、TX 标签、V 或目标。

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
