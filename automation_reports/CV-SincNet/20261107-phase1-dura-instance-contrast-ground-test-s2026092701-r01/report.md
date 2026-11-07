# DURA 源记录对比约束七景测试

- run_id：`20261107-phase1-dura-instance-contrast-ground-test-s2026092701-r01`
- group_id：`phase1-dura-source-instance-contrast`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；七景预测和独立评分均已完成。

## 目的与对照

源域辅助编码器以同一物理记录的模拟观测与 clean 为正对，约束 DURA 修正；部署不使用编码器。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

六个受损场景均低于 DURA，低城区为 68.1292%，相对 DURA 低 0.6720 个百分点；修复 3640 条、损伤 4769 条。两种 3% 口径均 0/6 达标。clean 得分与 DURA 相同，但有 2 条错变对和 2 条对变错，不是预测逐条一致。详见 [完整分析](analysis.md)，不据此修改其他已登记训练和测试。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
