# 遮挡教师批量预计算源域对照

- run_id：`20261107-phase1-masked-teacher-batch-parity-source-s2026092701-r01`
- group_id：`phase1-masked-teacher-batch-parity-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：ANALYZED，源教师批次对照已完成；完整结果见分析。

## 目的与对照

比较冻结教师小批与大批的逐值输出及本机耗时，不更新模型。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

已完成两次交替计时，教师输出逐值一致。该对照未训练模型，未访问目标或生成正式预测；不能用教师计时代表整轮加速。

## 交接

下一步须检验完整训练轨迹和 [计算环境] 实际加速，再考虑接入未来训练；运行中的既有训练不热修改。
