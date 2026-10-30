# DURA两份单阶段模型的源物理误差归因

- run_id：`20261030-diagnostic-DURA-direct-physics-s2026092701-m2-r01`
- group_id：`phase1-dura-source-physics`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

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

## 完整源结果

VERIFIED/ANALYZED_SOURCE_ONLY。两份checkpoint均先两Ls smoke PASS，12个模型/缓存epoch/角色单元全部闭合；CPU2/0GPU约13.119秒，全部权重/缓冲区逐位不变。0优化器/TX/F/V/target。执行commit c20aa8f0ef2c9f176cb2a3cd44a7553d6856ffb5，一次N607 preflight/归档SHA/compile通过，唯一P0/P1审查PASS，既有checkpoint协议负测1 PASS。首轮回查前PID3744647已结束；独立实际stdout两smoke和四个完成event、complete/per-example/raw聚合均回收，日志674bytes。未声称直接观测到该短CPU进程存活期间状态。

原始per_example.npz、opaque source positions与源role索引独立复算全部12单元/4176个标量统计一致；该复算不读IQ/F/TX或target。每epoch同512位置128Ls+384Us；两个epoch重复物理位置，不当成1024个独立样本。

|模型|缓存epoch|native IQ|learned CG16 IQ|oracle A CG16 IQ|oracle A相对CG16降低|probe0误差|
|---|---:|---:|---:|---:|---:|---:|
|fastLR|1|0.02446608|0.02577503|0.02243007|12.9775%|0.08334918|
|fastLR|20|0.02283946|0.02427188|0.02104777|13.2833%|0.05971603|
|noVarAux|1|0.02235236|0.02232527|0.02018408|9.5909%|0.02849227|
|noVarAux|20|0.02112201|0.02110850|0.01882089|10.8374%|0.02434837|

fastLR满足原定两epoch都至少10% oracle A改善且probe0>1%的门槛。noVarAux约9.5909%/10.8374%，只有一组超过10%，因此不降低门槛将它判定通过。本诊断支持在原fastLR上检验更强算子监督，不能据此支持在noVarAux上同样操作。它也不证明真实TX恢复收益：oracle A仍与固定learned prior可能失配，且不是可部署输入。

CG16相对nativeCG2在fastLR上误差增加约5.35%/6.27%，在noVarAux上仅降低约0.12%/0.06%；不加CG预算。真实方差替换未改善这两版源IQ，因此不把恢复器正值一致性参数当成必须校准到物理噪声的量。既有R29/R30/R31统计保持路线保留，不当成未经检验的新方案。

下一独立run只将原fastLR中source action probe权重0.05改0.2，其他五项目标/普通随机初始化/原生插值/E20/4940/数据/seeds/学习率全匹配。4倍系数使既有冻结源梯度面板上的action/IQ范数比中位约0.076→0.305，是源端尺度选择，不是目标调参；不宣称新训练轨迹也保持该比值。noVar/identity/slice均不混入；新权重仍从零训练，source-only依据在fastinterp正式结果读取前固定。冻结后自动全部七景，所有负结果保留。
