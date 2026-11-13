# PAMF 去参考响应：完整论文实验补齐

- run_id：`20261109-phase1-pamf-no-reference-completion-m190-r01`
- group_id：`phase1-pamf-no-reference-paper`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：RUNNING（2026-11-09 启动快照；实际进度以运行记录为准）

## 目的与对照

11数据条件×监督/普通PL增强四格及完整半监督，完整方法扩至8种子；补齐四个分类损失结构控制与8种子E200架构对照。复用已完成三分支面板和四外部方法；从零补齐缺失模型。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

本次 190 个逻辑 PAMF 模型中，6 个同契约模型及其冻结预测与评分复用，184 个模型需要新训练。已完成的 36 个第一阶段分支消融模型、528 个四方法 Full 对照及历史 14 网络八种子 clean 结果进入后续汇总；不重复训练或重算这些结果。新增模型训练、逐行预测和独立评分尚在进行。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

实际执行提交为 `33e46b2ab2b726900a2f0d9e775b0a2778820f39`。完整矩阵、训练配方和资源边界见 experiment.json；发布后核验见 launch_verification.json。恢复时先核对原运行和已有输出，不重复启动。
