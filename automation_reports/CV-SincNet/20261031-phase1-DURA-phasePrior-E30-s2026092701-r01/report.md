# 先验内部相位坐标统一：从零单阶段30轮

- run_id：`20261031-phase1-DURA-phasePrior-E30-s2026092701-r01`
- group_id：`phase1-dura-phase-canonical-prior-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原266951参数/两prior四CG，仅在prior前按coarse最强点转入统一相位坐标、prior后转回；不改IQ幅度/顺序/标量，不增参数或Core调用。scratchjointE30/完整七景。

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


## 固定结构与预算

从零单阶段30轮/7410步，原266951参数G及两prior/四CG/27FFT保持。每个prior内部以coarse base最强点为相位锚，全部4组IQ同时转入该坐标，Core计算后输出转回；标量logvar保持。只有base近零才回退到四组IQ最强点，全IQ零输出0。该约束只作用于prior，不声称整个G等变或F相位不变；不旋转训练目标、不四路平均，不增加参数/Core调用。新增argmax/gather/旋转成本不包含在ConvLinear MAC内，正式实测时延与训练成本。

原纯G8cf对应的初始参数逐项一致，权重共享和CPU/CUDA探针随机流保持；只改prior输入输出坐标。无F/教师/TX/V/target训练，物理四项finalIQ1/prior0.1/STFT0.05/action0.05、AdamW与源01..20后01..10保持，固定最终30轮。研发使用过源F诊断，不能将整条研发路线称完全F-free。3个聚焦检查、实际factory/预算/入口检查与唯一P0P1通过。首步真实2Ls三模块梯度、严格重载、native/optimized一致、prior等变及原F默认头smoke后立即继续。

自动profile、源1800×7、正式168000×7；每role全七景预测固定后独立CPU truth-last。即使源分数较差也完成新G正式矩阵。与同预算纯G配对，默认DURA保持；逐六景≥137216正确、clean≥142256及实际同质量效率仍为验收，R17每景1pp和旧低城区70%单列。无额外设计审批gate，方法与矩阵只在本最小登记维护。
