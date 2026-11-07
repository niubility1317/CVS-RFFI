# DURA 条件 IQ 流七景

- run_id：`20261107-phase1-dura-iqflow-ground-test-s2026092701-r01`
- group_id：`phase1-dura-conditional-iqflow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；七景预测与独立评分完成。

## 目的与对照

固定 DURA 上下文，源 IQ 路径速度匹配训练，四步确定性积分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

七景均低于 DURA；低城区 68.4554%，相对 DURA 低 0.3458 个百分点，修复 3104 条而损伤 3685 条。clean 相对 DURA 低 0.2458 个百分点。六景按绝对降幅不超过 3 个百分点计 1/6 达标，按相对下降不超过 3% 计 0/6。详见 [完整分析](analysis.md)，不据此改变其他已登记训练或测试。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
