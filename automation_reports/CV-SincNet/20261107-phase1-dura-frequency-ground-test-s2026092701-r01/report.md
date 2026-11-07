# DURA 源频域回归七景测试

- run_id：`20261107-phase1-dura-frequency-ground-test-s2026092701-r01`
- group_id：`phase1-dura-source-frequency-regression`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；[七景正式结果](analysis.md)。

## 目的与对照

冻结 DURA，以源 IQ 拟合四组噪声条件化广义线性复频域残差。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

七景完整预测固定后已完成独立 truth-last 和 DURA 同记录配对。各景均低于 DURA，低城区 68.4845%；完整场景、净纠错、Macro-F1、TX/RX/day 边际分组及目标口径见[结果分析](analysis.md)。

## 交接

固定结果和源拟合产物保留，不按目标结果选模、调参或重训。
