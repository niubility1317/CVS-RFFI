# 投影细节分支第11轮异常：固定检查点的源端压力测试

- run_id：`20261108-diagnostic-Restore-ProjectedE11-sourceLU-s2026092701-r01`
- group_id：`source-projected-detail-instability-E11`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

比较原投影细节版源训练 E10/E11 技术检查点，在源 draw11 固定压力样本上的提案、投影修正与输出误差。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 固定源压力测试完成

VERIFIED / ANALYZED。原版E10/E11技术状态各在149条固定源压力样本上完成冻结前向，最大投影修正NMSE分别0.1508/0.1176，未复现第11轮轮均异常；模型状态不变。源样本极值分组非总体代表，不能排除轮内瞬态或未选样本。零TX/V/OE/目标读取与优化。 [完整分析](analysis.md)
