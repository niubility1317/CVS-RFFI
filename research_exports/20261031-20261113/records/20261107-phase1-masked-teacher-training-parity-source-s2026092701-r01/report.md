# 遮挡教师预计算训练轨迹对照

- run_id：`20261107-phase1-masked-teacher-training-parity-source-s2026092701-r01`
- group_id：`phase1-masked-teacher-training-parity-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：ANALYZED；分析结果。

## 目的与对照

源域小样本配对训练验证预计算与原训练的逐步损失及最终参数。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

两次配对训练的每步损失及最终参数一致，未发现可重复的整段训练加速。仅为本机源训练对照，未生成正式预测或保存权重。

## 交接

不将此预计算接入正式训练；原有 [计算环境] 正式任务继续，完整七景结果须等待预测和独立评分。
