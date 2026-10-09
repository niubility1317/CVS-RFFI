# 纯CE正常遍历：batch256/512/1024/2048速度与泛化对照

- run_id：`20261009-phase1-ce-singlepass-batch-manysig-m4-r02`
- group_id：`cvs-ce-singlepass-batch-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用户明确选择每轮正常遍历6300源L、200轮，并要求多个更大batch。固定单seed、同学习率和源契约；非同更新次数对照。

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

## 固定设计与边界

用户选择正常遍历6300源L、batch256，并追加多个更大batch。固定256/512/1024/2048、seed2026092701、200轮、每轮25/13/7/4次更新，保留最后不完整batch。源L每轮完整且仅遍历一次，源U只保留角色元数据、不读取U批次计算损失。监督CE，LEO/EMA/pseudo/entropy及所有附加机制关闭，reference_response网络从零训练，固定E200。原source循环、provenance、native args检查和评分器复用已审查实现；本次对迭代/预算/数据覆盖做聚焦检查。用户禁止侧对话子Agent，未伪称新独立审查完成。所有组训练完、固定源模型后自动完成七视图预测与独立评分。与旧CE比较时披露batch、更新次数和尾批策略均变化，不能单独归因batch。显存不足记录真实技术失败，不调整batch重跑。

## FAILED：源数据加载器接口

2026-10-09T10:56:39.893688+08:00，四行均在训练前因labeled_loader键不存在退出，优化器更新0次，未读target。真实接口为train_loader；r03增加actual source loader检查并保留原目录。
