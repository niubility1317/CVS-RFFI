# 独立多视图教师学生 GPU 训练

- run_id：`20261107-phase1-direct-multiview-distill-source-s2026092701-r01`
- group_id：`phase1-direct-multiview-distill`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：TRAINING_COMPLETE；[源训练分析](analysis.md)。冻结后已进入七景正式预测。

## 目的与对照

从头训练单输入独立时频学生，使用同一源记录三个模拟视图的 DURA 平均输出作为训练教师。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

真实源 checkpoint smoke 和固定 20 轮源训练已完成。学生在源 pilot 的 14 个分组平均波形 NMSE 均高于 DURA；波形误差不是身份准确率，详见[源分析](analysis.md)。七景正式预测正在运行，暂无正式识别结果。

## 交接

七景全部预测固定后自动独立评分；恢复时核实唯一 owner，不重复启动。
