# 独立多视图教师学生七景测试

- run_id：`20261107-phase1-direct-multiview-distill-ground-test-s2026092701-r01`
- group_id：`phase1-direct-multiview-distill`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；源训练和源预测 smoke 完成，N607 预测 PID 3992504 已核实。

## 目的与对照

从头训练单输入独立时频学生，使用同一源记录三个模拟视图的 DURA 平均输出作为训练教师。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

学生已按固定最终轮冻结，七景预测正在运行；全部预测固定后独立 truth-last 并与同记录 DURA、raw clean 对照。当前暂无正式身份识别结果。

## 交接

保持唯一 owner；全部预测固定后自动评分，不依据源 pilot 波形指标取消测试。
