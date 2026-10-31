# 精确非线性源域特征约束：IQ可达性与梯度注入

- run_id：`20261031-diagnostic-DURA-source-nonlinearReference-s2026092701-r01`
- group_id：`phase1-dura-source-exact-nonlinear-reference`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同source10目标，固定8次半步GaussNewton重线性化检验实际IQ可达性；2Ls随机G检查一次F reverse的精确梯度注入。0优化/新G训练/TX-V-target。

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


## 固定非线性源域检查

同一source10参考/原F50冻结，固定512源128L384U，8次GaussNewton重线性化、每次0.5半步、Gram截断1e-10，最终第8次实际F检验；无早停、线搜索或参数搜索。最终相对偏移误差中位≤0.5/P95≤1、IQ位移NMSE均值≤0.001/P95≤0.005、512原clean判决保持。旧10维clean-J近似失败保持原结论，新候选不把它改写为通过。

精确source F约束使用一次输入反向，步内detach限幅后用零值线性项向G注入梯度，后续G反向不再执行F。3项聚焦测试与实际入口/唯一P0P1通过；首步真实2Ls F/G梯度、cap和状态检查后立即继续。主F4608/VJP40960，smokeF4/VJP2/G2，0优化/G训练/TX/V/target/新完整J。源clean路径只是诊断oracle，不是部署算法，也不构成多阶段G训练；GPU成本尚未测量。通过仅支持另登记GPU开销测量，不自动训练；默认与完整目标保持。
