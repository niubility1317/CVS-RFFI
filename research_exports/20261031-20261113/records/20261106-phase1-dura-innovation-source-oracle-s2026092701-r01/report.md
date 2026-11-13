# DURA 物理创新方向的无标签源域理想步长诊断

- run_id：`20261106-phase1-dura-innovation-source-oracle-s2026092701-r01`
- group_id：`phase1-dura-physical-innovation-oracle`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：ANALYZED（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

只使用源 L_s/U_s 成对 IQ 和固定 DURA 测量反向投影方向与源 clean 波形误差；不训练、拟合部署参数或访问 F、TX、V、目标。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

真实 DURA 两条源样本 smoke 通过；672 条源 IQ 和 14 个接收机组／场景单元的物理方向理想线搜索完成。RX8 低城区符号允许正负时的组均 NMSE 降低约 0.594%，28/48 条最优系数为负。详细定义、全部分组和局限见 analysis.md。未训练或发布恢复模型，未访问 F 或目标。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
