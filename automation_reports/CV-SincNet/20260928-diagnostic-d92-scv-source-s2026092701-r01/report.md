# 固定Phase1的D92-SCV源域适应诊断

- run_id：`20260928-diagnostic-d92-scv-source-s2026092701-r01`
- group_id：`d92-fixed-phase1-support-cv-upgrade`；类别：`cvs`；阶段：`source-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定第一个新模型种子，不按目标成绩选择。源域L拟合、单一V验证；60个RX/scene/K切片对照D92与冻结DG。目标是为后续新旧类H优化验证实现和旧类行为。

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

## 实施与审查

当前状态LOCAL_VERIFIED。完整目标与机制见docs/D92_SUPPORT_UPGRADE_20260928.md；新候选8项聚焦测试通过，唯一P0/P1审查完成。源域仅旧6类，H不适用；不得声称目标性能提升。原D92和所有历史输出保持不变。精确运行命令、commit、PID与状态在启动后读回。

## 第一轮源域结果（VERIFIED）

远端state及complete均显示180条完整；dispatcher/worker均已退出。33300个源域物理记录导出完整，原L6300、单一V27000；checkpoint严格加载，缺失/多余key均为0，未更新encoder、未访问target。

|K|冻结DG|原D92|D92-SCV-v1|相对D92（百分点）|相对DG（百分点）|
|---|---:|---:|---:|---:|---:|
|1|84.60%|68.71%|84.60%|+15.89|+0.00|
|5|84.60%|72.20%|84.20%|+12.00|-0.40|
|10|84.60%|80.38%|84.24%|+3.87|-0.36|
|20|84.60%|81.04%|84.71%|+3.67|+0.11|

每档K为15个RX×scene切片的均值，单一model seed、单一support draw；V每类50条。候选相对原D92改善明显，但主要通过避免破坏已有Phase1判决。尚不能证明比零适应更强，更不能宣称新类H改善或目标域泛化改善。逐切片与完整选择证据见results/和本地full_trace_path。

后续：联合注册predictor已实现并通过11项输入边界负测，但尚未发布/运行。确认集范围待用户选择：优先独立未暴露样本，或在本批数据上做冻结后的一次描述性对比。元数据核实19-1、8-14、8-7三个接收机未进入最近一批source/target集合，均覆盖现有全部26类、每类至少200条；历史所有实验是否暴露尚未证明，不能直接称严格未暴露确认集。
