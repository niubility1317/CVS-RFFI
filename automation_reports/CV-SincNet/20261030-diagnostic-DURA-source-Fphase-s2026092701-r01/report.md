# DURA 同冻结F源域全局相位敏感性诊断

- run_id：`20261030-diagnostic-DURA-source-Fphase-s2026092701-r01`
- group_id：`phase1-dura-source-Fphase-sensitivity`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

CPU2/零训练；固定源L128/U384，8角度检查原F50完整160D特征和原头后验/选择；不读G、逐样本TX真值、V或target。

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


## 最小研究说明

只读原F50/defaulthead与原source L/U，固定512条×8角度，不加载G、不训练或读逐样本TX/V/target。比较完整160D特征和原头soft后验、教师选择，不能将教师选择一致写成TX准确率。source class-map元数据仅由既有F合同检查读取。参考响应分支不变性不能替代整个模型的实测性质。

全部7个非零角度必须同时满足：总体教师选择一致率≥99.5%，每个源RX≥99%，总体平均特征余弦误差≤0.001且KL(clean||rotated)≤0.001。任何失败即不据本诊断启动全局相位商空间损失训练，不放宽阈值或改选小角度。通过也只是源clean研究资格，不推出连续相位、restored或target不变性，不直接部署或更新默认。该设计明确使用F信息，不能称F-free。原3pp/clean/效率目标保持。
