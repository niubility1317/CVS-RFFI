# 以实际B函数为先验的残差LocalRidge与support监督adapter联合适应注册

- run_id：`20261001-phase2-d92-anchor-joint-support-m2-r01`
- group_id：`d92-anchor-joint-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：LOCAL_VERIFIED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定旧类参考测度与核尺度，B分类函数作为C先验；全类残差闭式LocalRidge经平滑CE联合学习函数坐标adapter；保留完整interaction，不做参数网格。

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

LOCAL_VERIFIED/VERIFIED：AJLR单一数学结构已实现，54个不同相关检查通过；core22、ops23、entry4、summary5。整数dtype、summary唯一参考pair、实际1800路径和原生bool验收边界均已修复；完整正常复算与10种篡改拒绝通过，独立审查NO_UNRESOLVED_P0_P1。实际B函数prior、固定旧物理参考测度与tau/gamma、残差闭式头、全类CE伴随联合微调；只有R0/R_AJLR_seq。完整四row/160parent预登记与既有source-only缓存身份preflight已核实，新run/release/archive无冲突。未发布、未启动，尚无真实性能；A与B−A=N/A，目标ACTIVE。
