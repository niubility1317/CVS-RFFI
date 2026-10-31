# 时序签名中的接收机均值方向：留出源样本诊断

- run_id：`20261031-diagnostic-DURA-source-SIG-RX-subspace-s2026092701-r01`
- group_id：`phase1-dura-source-SIG-receiver-mean-subspace`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

排除固定512源样本，用其余62488源clean SIG与公共RX拟合rank4均值span；比较正交补与完整SIG修正，原F原头/source512×01/20，0G/TX-V-target。

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


## 固定源假设与执行矩阵

源clean时序签名可能包含接收机相关成分；这只是待检验假设，不能把接收机均值差异直接归因为纯接收机噪声。固定512条源观察（128 L_s/384 U_s）全部排除后，仅用其余62488条源clean实际加权SIG及公共RX计算五个均值，等权中心化、FP64 SVD取固定rank4，不搜索秩或参考。未读取TX、V或target，不是分类头训练。

仅固定默认DURA/source512×缓存01/20，复用原实际F分量，比较native、完整clean SIG差修正、其RX均值span正交补修正；RX均值span内修正仅作描述，主候选失败时不能替代。各混合特征均接原F50默认头，没有G加载/前向/优化。源clean参考不能用于部署，特征混合不保证对应可实现IQ；一致数是clean F判断代理，不是真实TX准确率或恢复上限。

两缓存均要求正交补版较完整SIG至少多2条一致；每源RX及clean不损失、每非clean景最多损失1条，整体完整特征误差不超过native的1.01倍。全部通过仅支持另登记源IQ可达性验证，不自动训练G；失败保持原阈值和负结果。原PA、IQ步进和PA+SIG失败均保留。

3项聚焦检查与实际入口通过，唯一P0/P1 PASS。只读复用此前验证的63000×160/40.32MB源签名缓存，不产生新缓存或神经前向；历史63000前缀/0.847077秒及F50源监督知识/11100更新成本披露。首步真实2Ls实际weightedSIG、原分量、原头与缓存匹配PASS后继续；主完整F0/显式原头4096，smoke完整F2/原头2。CPU2/空CUDA，0G/TX/V/target/模型更新，F全state和grad检查。一次preflight、归档SHA比较、远端编译及实际PID/CWD/argv/空CUDA/日志核实，随后全数组独立复算；默认DURA与完整3pp/clean/同质量效率目标保持。
