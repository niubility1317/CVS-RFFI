# DURA 可逆耦合 GPU 训练

- run_id：`20261107-phase1-dura-reversible-source-s2026092701-r01`
- group_id：`phase1-dura-reversible`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：TRAINING_COMPLETE；[源训练分析](analysis.md)，[七景正式结果](../20261107-phase1-dura-reversible-ground-test-s2026092701-r01/analysis.md)。

## 目的与对照

在冻结 DURA 输出上训练四层时间奇偶加性可逆耦合，目标源 IQ 加 DURA 波形保护。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

固定五轮源训练完成。源 pilot 的受损分组波形误差略低于 DURA，但该指标不是身份准确率；正式七景结果均低于 DURA。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
