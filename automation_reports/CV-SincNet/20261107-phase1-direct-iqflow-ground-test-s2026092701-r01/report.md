# 独立 IQ 流七景

- run_id：`20261107-phase1-direct-iqflow-ground-test-s2026092701-r01`
- group_id：`phase1-direct-iqflow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；固定第 20 轮模型已开始七景预测。

## 目的与对照

以单条 received IQ 为锚点，用源 clean IQ 训练路径速度，四步确定性积分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

源训练和源预测检查完成，七景预测 PID 524635 已核实。全部七景预测固定后独立 truth-last 评分；目前没有正式身份识别结果。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
