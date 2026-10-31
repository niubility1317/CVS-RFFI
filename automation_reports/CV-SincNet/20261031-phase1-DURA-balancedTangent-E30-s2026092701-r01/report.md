# 保留特征方向的梯度平衡：单阶段30轮

- run_id：`20261031-phase1-DURA-balancedTangent-E30-s2026092701-r01`
- group_id：`phase1-dura-source-balanced-tangent-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原266951G/noVar四物理项＋同alpha源clean切线项，每步逐样本detach缩放过强附加IQ梯度、保持方向；复用原raw J缓存，训练F加载/前向/VJP0；scratch联合30/7410及全七景。

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


## 固定更新规则与验收

新G从零联合30epoch/7410、266951参数/两prior四CG部署保持，原noVar四项＋原alpha.0033130025035573727切线项。唯一优化规则变更为逐样本s=min(1,||gIQ||/||gfeature||)，步内detach，原方向不改，弱记录不改；附加loss和IQ输出梯度范数≤原IQ项。不保证G参数梯度界，日志值是步内加权surrogate，不冒称全局固定能量的完整导数。

训练只读原F50 sourceclean raw J缓存645.12MB，不使用失败静态谱因子，不加载缓存producer G或target。历史F50/11100与缓存63000F/315000VJP/30.009秒披露，新训练F加载/前向/VJP0、新缓存存储0，G祖先0、直接TX/V/target0。数据/初值/CUDAprobe/npshuffle/LR/原物理loss与olduncapped30匹配；缓存01..20后01..10。源压力和方向数学检查来自已暴露训练池，不能代替正式身份收益。

7聚焦检查和唯一P0P1后发布；首步真实2Ls三模块G梯度、实际guide界、F冻结/默认头/strictreload后立即继续。每batch记录权重和限幅比例/压力，超界1+2e-5或非有限按技术停止，绝不按性能停。满30冻结后自动source1800×7/target168000×7/all7固定后独立truth-last、TX/RX/day/净纠错及全成本。每星地≥137216正确且clean保持，R17每景1pp和旧70%另报。默认DURA保持，完整目标ACTIVE。
