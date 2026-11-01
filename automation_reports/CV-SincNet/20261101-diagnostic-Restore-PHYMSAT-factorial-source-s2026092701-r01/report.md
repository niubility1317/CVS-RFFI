# PHY-MSAT源端信道与噪声四路冻结诊断

- run_id：`20261101-diagnostic-Restore-PHYMSAT-factorial-source-s2026092701-r01`
- group_id：`restoration-source-joint-physical-factorial`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原LsUs512包×20draw，冻结原F0与PHY-MSAT fixedfinal30，分别替换真实模拟H和noise；0更新/TX/V/OE/target，区分共适应系统的误差来源。

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

## 四路源端冻结诊断

原Ls128/Us384固定512包×20份R07训练draw=10240。冻结完整PHY-MSAT final30与原地面Tri-SIG-TIME SSL/defaulthead，比较blind、真实模拟H、真实noise、二者同时替换。四路均使用原两次不共享先验、每次8CG和real prior variance .01；真实noise<1e-7按1e-7处理，包含clean的1e-8哨兵，缓存不改。第一source2真实smoke需blind逐点等于原G，PASS立即继续。

零训练、TX/V/OE/novel/目标读取或模型更新，原f57源F0尺度仅作指标归一化。不使用父模型的目标误分类选择面板或超参；父模型固定final30已独立完成Main全七景。真实H/noise也改变学习先验上下文，结果只定位共适应系统敏感性，不能分解独立因果贡献或作为理论上限、Main准确率或部署方法。

环境为原N607 CVS-RFFI Python/Torch2.1.0+cu121、FP32/TF32off，CWD/commit/命令/路径/产物与停止规则见登记表。GPU4/6用UUID准入，free>=16000MiB、原compute<=2，只新增一个本任务NN进程、单卡总<=3；不改其他owner。失败保留root，不自动重试。完整源记录、所有20draw与各场景结果后独立NumPy复算，默认DURA及性能目标保持。
