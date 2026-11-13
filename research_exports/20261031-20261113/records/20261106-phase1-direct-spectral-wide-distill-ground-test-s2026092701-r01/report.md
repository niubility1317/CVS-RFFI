# 扩大源数据独立时频学生七景测试

- run_id：`20261106-phase1-direct-spectral-wide-distill-ground-test-s2026092701-r01`
- group_id：`phase1-direct-spectral-wide-distill`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：ANALYZED；七景正式结果。

## 目的与对照

独立初始化时频学生，利用 32768 条合法源 IQ 和冻结 DURA 源波形教师训练；部署只需单条接收 IQ。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

七景完整预测后已完成独立 truth-last 和 DURA 同记录配对；详见分析。

## 交接

正式评分已完成；固定结果和历史训练产物保留，不按目标结果选模或重训。
