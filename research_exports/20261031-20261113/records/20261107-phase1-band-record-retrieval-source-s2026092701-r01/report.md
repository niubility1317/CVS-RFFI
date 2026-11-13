# 源跨视图频段平衡同记录检索诊断

- run_id：`20261107-phase1-band-record-retrieval-source-s2026092701-r01`
- group_id：`phase1-band-record-retrieval-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：ANALYZED；完整源域结果。

## 目的与对照

比较普通 IQ 与四频段归一化在已冻结源记录同记录检索中的表现，不使用身份标签。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

4096 条源记录的三视图检索中，DURA 普通表示 top-1 为 9770/12288，频段归一为 9667/12288；这不是 TX 身份成绩或部署方案。

## 交接

固定频段候选仍需按预登记进行七景测试；不能据此改变或取消已登记实验。
