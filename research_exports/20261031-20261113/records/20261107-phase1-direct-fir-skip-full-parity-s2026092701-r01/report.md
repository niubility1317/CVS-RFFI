# 独立 FIR 学生省略 DURA 的完整七景决策核验

- run_id：`20261107-phase1-direct-fir-skip-full-parity-s2026092701-r01`
- group_id：`phase1-direct-fir-skip-full-parity`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl

## 目的与对照

等待既有七景预测全部固定后，省略 DURA 上下文逐条核对原始和恢复决策，不读取真值。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

112 条源波形和 logits 的逐值检查已通过；完整七景决策尚未重放，无完整一致性结论。

## 交接

待原七景 manifest 固定，核验物理 ID 和候选版本后逐条对照 raw／恢复决策，不连接真值，不修改原预测。
