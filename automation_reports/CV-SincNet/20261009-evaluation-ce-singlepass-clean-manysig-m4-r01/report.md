# 四种batch已完成E200模型：只测clean独立评分

- run_id：`20261009-evaluation-ce-singlepass-clean-manysig-m4-r01`
- group_id：`cvs-ce-singlepass-batch-clean-evaluation`；类别：`cvs`；阶段：`evaluation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用户明确只测clean；修复评估器旧44400预算绑定，复用四个既有E200权重，不重训、不读取LEO测试视图。

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

## 固定clean-only测试

四个batch256/512/1024/2048，seed2026092701，使用原r03的E200最终权重，仅168000个clean物理query。修复source_provenance为逐行预算核对，消除原e.design重设44400检查器的问题；保留scratch来源、物理角色、完整配置、实际checkpoint args和无query加载smoke。预测循环VIEWS严格设为clean，四组预测固定后独立truth-last评分。源训练不重跑，原失败产物不覆盖。沿用既有不派生侧对话Agent边界，不声称新独立审查；复用已审查native实现并做本地预算与冻结负测。
