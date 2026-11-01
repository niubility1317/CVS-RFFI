# Phase1 线性相位漂移反事实源域诊断

- run_id：`20261101-diagnostic-phase1-phase-ramp-oracle-s2026092701-r01`
- group_id：`phase1-privileged-phase-ramp-source`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

利用配对 clean IQ，诊断复数增益加线性相位漂移的拟合与冻结基座一致性。

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

## 源训练反事实结果

真实冻结 Tri-SIG-TIME SSL，对固定 512 条源训练记录（L_s 128、U_s 384）分别在两个源视图上计算。每条使用配对 clean IQ 搜索覆盖整个离散频率范围的 4096 点相位斜率及复数增益，仅为波形均方误差最优的网格反事实，不是物理受限估计，也不是识别准确率最优搜索。

| 源视图 | BLOCKED 数量 | DURA 与 clean 教师一致 | 复数标量一致 | 相位斜率反事实一致 |
|---|---:|---:|---:|---:|
| 01 | 83 | 65/83 | 22/83 | 18/83 |
| 20 | 62 | 48/62 | 15/62 | 13/62 |

全样本均值 IQ NMSE 从复数标量的 0.8320、0.8378 降至相位斜率的 0.7689、0.7721，但 clean 教师一致率从 43.75%、45.12% 降至 35.16%、33.40%。结果提示波形均方误差与基座判断不等价，不能推广为所有相位估计或复杂均衡无效。该诊断使用训练源数据，clean 教师一致性不是 TX 准确率，不是测试或可部署结果；未读取 U_s 标签。
