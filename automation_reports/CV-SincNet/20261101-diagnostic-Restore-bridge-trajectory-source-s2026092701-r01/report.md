# 冻结四步恢复轨迹：原源面板逐步保真分析

- run_id：`20261101-diagnostic-Restore-bridge-trajectory-source-s2026092701-r01`
- group_id：`restoration-frozen-source-four-step-trajectory`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

两版固定final30在原源512×20记录逐步分析，与首步配对；零训练和目标IQ/真值读取，等父Main技术完成后运行。

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

## 冻结源端轨迹与主性能的关系

该诊断不训练新G，不读取目标IQ/真值/预测或评分。CUDA空控制器只等两版父run完整Main技术完成（不看达标与分数），再按资源准入做源推理，因此不占第三条本任务NN、不阻塞或修改父计划。父模型各自strict final30/7410/config/commit/参数和0TX/V/OE/目标训练来源核实；工厂先于F0命名空间加载，G/F0全冻结。3重点检查及一次独立P0P1通过。

原R07固定128Ls+384Us共512条、20份原缓存draw。记录两模型4次clean预测的IQ误差、F0连续特征及latent保真、原defaulthead对F(clean)判决一致率、raw复功率/单位投影误差和相对首步波形变化；同条case配对修复/损伤/净变化。首个实际源2检查manual轨迹末步逐值等于真正G.forward，F0/defaulthead有限，PASS立即全部轨迹；最后G/F0逐tensor不变。无TX标签、V/OE/novel/目标IQ、更新、拟合头或checkpoint选择。

81920=2×4×512×20条Gcore/输出F0特征；另512cleanF0参考、81920latent原默认头读出（仅诊断，非新头）。源2额外32 Gcore/4 F0特征。逐记录数组私有，公开聚合和同条配对，不能把Fclean一致当TX准确率或把重复512当10240独立包。四步有效性和解析捷径是否实际被学到仍有区别；本诊断不构成恢复上限、跨RX泛化或未知设备保真证明。

主验收仍是原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000七景，各practical距同一raw clean≤3pp且clean保持。源端更低误差或更多步数不替代该结果，默认DURA保持。原LsUs数据权限、OE仅冻结诊断、0目标优化和所有新G完整测试均不变。
