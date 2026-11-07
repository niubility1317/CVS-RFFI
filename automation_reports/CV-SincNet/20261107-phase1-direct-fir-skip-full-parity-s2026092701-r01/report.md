# 独立 FIR 学生省略 DURA 的完整七景决策核验

- run_id：`20261107-phase1-direct-fir-skip-full-parity-s2026092701-r01`
- group_id：`phase1-direct-fir-skip-full-parity`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；七景类别重放已完成，[完整一致性结果](analysis.md)。

## 目的与对照

等待既有七景预测全部固定后，省略 DURA 上下文逐条核对原始和恢复决策，不读取真值。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

112 条源波形和 logits 的逐值检查已通过。七景各 168000 条原始输入和恢复类别均与原预测一致，完整范围与耗时限制见[分析](analysis.md)；全量波形和 logits 未在原预测中保存，无法完成逐值比较。

## 交接

完整无真值重放完成；原预测和模型保留，不因技术加速将该学生识别表现宣称为达标。
