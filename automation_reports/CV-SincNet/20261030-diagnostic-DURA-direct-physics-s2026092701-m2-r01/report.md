# DURA两份单阶段模型的源物理误差归因

- run_id：`20261030-diagnostic-DURA-direct-physics-s2026092701-m2-r01`
- group_id：`phase1-dura-source-physics`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

FixedfastLR/noVarAux sourcefinals;512source128L384U xepochs01/20;native/CG16/oracleaction/noise/cleanprior;0F/TX/V/target/updates

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

## 本次要区分的问题

固定fastLR和noVarAux两侧完整源final，不选目标分数最好的模型。比较native CG2和固定同一final prior的learned CG16；再逐个替换source模拟器算子/方差，保持CG16来隔离算子与噪声权重。原始clean prior只作源端偏差参照，任何oracle输出都不进入部署或目标预测。两份模型的配置加载互相隔离，复用既有源数据与原诊断算术；不重做已完成的四阶统计训练。

后续增强算子监督的源证据门槛预定为：同一model两缓存epoch中，oracleAction CG16相对learned CG16的IQ均值误差均降低至少10%，且probe0算子误差均>1%。这只是支持方向的条件，不证明正式收益；若条件不满足，不以本诊断为理由盲目增加算子损失或CG。每模型首步实际两Ls checkpoint smoke后直接继续，0更新/标签/F/V/目标。

文献说明PA非线性、频谱外延可以承载RFF线索：[Xie等的Wi-Fi研究](https://arxiv.org/html/2412.07269)和[Yang等的LoRa研究](https://arxiv.org/abs/2302.13724)。它们的数据、提取方式及分类训练权限与本任务不同，不能挪用其准确率或将其当成本任务的性能保证。项目R29/R30/R31已有四阶统计方向，新的源诊断先定位当前单阶段物理误差，不直接重复添加该类损失。
