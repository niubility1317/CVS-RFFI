# DURA 合成响应约束七景测试

- run_id：`20261107-phase1-dura-counterfactual-response-ground-test-s2026092701-r01`
- group_id：`phase1-dura-counterfactual-response`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；完整七景预测和独立评分完成。

## 目的与对照

固定 DURA 物理作用，对源 IQ 合成微小扰动并重算 DURA，训练多尺度修正保留输出响应。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

低郊区、中城区和低城区相对 DURA 分别提高 0.0530、0.1065、0.1583 个百分点，净多对 89、179、266 条；其他三个受损场景分别下降 0.2786、0.1274、0.2399 个百分点。clean 净多对 1 条。两种 3% 口径均仅 1/6 达标，是局部正收益，不是整体替代或目标达成。详见 [完整七景及分组分析](analysis.md)。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
