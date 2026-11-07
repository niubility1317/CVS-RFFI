# DURA 留一视图教师七景测试

- run_id：`20261107-phase1-dura-leaveout-multiview-ground-test-s2026092701-r01`
- group_id：`phase1-dura-leaveout-multiview`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；完整七景预测和独立评分完成。

## 目的与对照

冻结 DURA，多尺度修正训练时使用同一源记录另外两视图的 DURA 输出均值作为教师。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

六个受损场景均低于 DURA，低城区 67.2149%，下降 1.5863 个百分点，修复 3146 条、损伤 5811 条。两种 3% 口径均 0/6 达标。clean 得分相同，但一条错变对、一条对变错，预测并非完全相同。六景 × 四个日期的 24 个日期边际分组均低于 DURA。完整配对与逐组数据见 [正式分析](analysis.md)。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
