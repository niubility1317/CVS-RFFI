# 精确非线性F特征指导：从零单阶段30轮

- run_id：`20261031-phase1-DURA-exactReference-E30-s2026092701-r01`
- group_id：`phase1-dura-source-exact-reference-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同266951G/物理四项/源视图/E30预算，固定source10软跨RX目标，在线冻结F精确输入梯度/每步cap1/一次F reverse注入；新目标缓存2.52MB、0J读取。全七景。

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


## 固定精确F指导30轮

原266951参数G从零单阶段联合30轮/7410步，物理四项、seed/CUDAprobe、随机批次、学习率、源01..20后01..10保持。将head5缓存近似换为同源10维精确F约束，同时改变约束维数和导数方式，不能把结果归因为单项因素。beta0.25、alpha0.0033130025035573727、IQ梯度cap1保持；guide已含alpha，物理loss直接加guide。F50/source4df7/原默认头冻结、optimizer只G、无直接TX/V/target。

新特征目标63000×10 FP32/2.52MB由已有source F160/软RX均值/合法基底生成，生产F0/VJP0，不读J或GN修改IQ。训练在线F1890000输入+1890000输入反向，一次reverse后注入detach限幅梯度，F50历史知识/成本与新增缓存/真实时耗全部披露，不称F-free。F加载及额外F路径隔离Torch/CUDA随机流，避免扰动G/probe；固定最后30轮。

4项新集成测试、实际入口、唯一P0P1通过；真实2Ls三个G模块/精确IQcap/F冻结原头/严格epoch0重载后立即继续。满30自动profile、source1800×7和target168000×7，每role全预测固定后独立truth-last，即使源成绩差仍完成新G正式矩阵。每六星地≥137216正确且clean≥142256，R17每景1pp及旧低城区70%另列。默认DURA与完整目标保持。
