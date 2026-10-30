# 完整粗恢复宽度＋紧凑展开的单阶段30轮容量对照

- run_id：`20261031-phase1-DURA-wideCoarse-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-coarse-capacity`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只恢复coarse原宽度，action/two-prior/4CG及早期IQhint配方保持；G从零1466771参数E30，复用原630k教师缓存，不重新制备/加载教师，不混strongphase。

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


## 最小容量候选说明

从旧source hint拟合差距与参数盘点固定本矩阵：coarse106138→1305958，action79477/uniqueprior81336保持，总1466771；只改coarse宽度、不混strongphase损失。学生从零同一model/data/probe RNG，旧模块初值逐张量一致，教师权重复制0。仍联合E30/7410，前10hint1→.1、后20为0，noVar四基础项不变。原64b8的10×63000 R17coarseIQ缓存只读复用、当前教师加载/前向0，历史coarse20/bundle60及132.756秒制备/1.29024GB另列，不隐藏知识成本。较原R17少68.558%参数，但比当前DURA大449.453%；合成单包图Conv/Linear MAC116098088、默认25068072，实际时延/全训练成本待测。11检查/唯一P0P1通过，真实2Ls新G梯度/严格重载/原F noquery为首步；GPU6按原授权容量，评价先精确核实strongPhase旧worker/scorer已退出，全七景完整truth-last后验收。当前待发布，默认与逐景3pp/clean目标保持。
