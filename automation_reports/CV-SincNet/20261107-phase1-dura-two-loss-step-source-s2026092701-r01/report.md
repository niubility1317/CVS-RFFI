# 源域 AdamW 双损失方向约束有限步诊断

- run_id：`20261107-phase1-dura-two-loss-step-source-s2026092701-r01`
- group_id：`phase1-dura-two-loss-step-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；[源域诊断结果](analysis.md)。

## 目的与对照

比较单步 AdamW 和投影后的实际源 IQ 与合成响应损失，不生成恢复模型。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

预先固定的源记录和两种学习率已完成。实际 AdamW 单步对两项损失均为一阶下降方向，额外投影未改变本次结果；见[分析](analysis.md)。未保存网络权重，也未测目标识别。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
