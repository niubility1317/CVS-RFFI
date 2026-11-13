# 独立 FIR 学生跳过 DURA 源域验证

- run_id：`20261107-phase1-direct-fir-skip-source-s2026092701-r01`
- group_id：`phase1-direct-fir-skip-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：ANALYZED；源域检查结果。

## 目的与对照

比较固定独立学生原路径和省略 DURA 上下文及其分类的源域输出与计时。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

112 条源 IQ 上波形及 raw／恢复 logits 逐值一致；原路径 1.7113 秒，快速路径 0.5427 秒。完整七景决策一致性尚在独立核验。

## 交接

完整重放不读取真值，不中断或改变原预测；仅在原七景预测全部固定后启动目标输入重放。
