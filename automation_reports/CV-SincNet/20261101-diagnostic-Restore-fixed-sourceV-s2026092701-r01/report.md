# 恢复器固定源V验证：四个冻结G与默认DURA

- run_id：`20261101-diagnostic-Restore-fixed-sourceV-s2026092701-r01`
- group_id：`restoration-fixed-sourceV-frozen-validation`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅原固定V1800×七景，四冻结G/defaultDURA/raw/PHY真实H路径共七读出。0训练/TX/OE/novel/target；等待FullFGrad完整Main结束才运行，不参与checkpoint或候选选择。

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

## 固定源 V 只验证

用户已限定源V/novel只验证、G训练仅原LsUs；本项仅原V1800冻结验证，0训练/TX/OE/novel/目标/参数或统计更新、0checkpoint或候选选择。原R07缓存既有V1800角色/唯一ID、与训练63000互斥、5源RX×3源日期每格120和七景trace1800元数据已核实，不新建数据或重验哈希链。1800物理包×七景=12600实例相关，不能当12600独立包；原V历史曝光、F0/defaultDURA已有V知识与训练/验证receiverseed不同1902/1911均披露。

固定raw、defaultDURA、原MSAT、VarFr03、PHY、未限幅FullFGrad及PHY真实H七路，共88200条F0特征行；只报IQ NMSE、F0特征误差和F(raw clean)判决一致率，不读真实TX，不是Main准确率或理论恢复上限。四G严格原source-only final30/7410配置与commit，F0原Tri-SIG-TIME SSL/defaulthead冻结，f57源特征尺度不在V重拟合。真实H会同时改变学习先验context，不能拆成独立因果份额。

控制器CUDA为空，等FullFGrad整个Main ARTIFACTS_COMPLETE（不论科学成绩）后才申请GPU和加载模型；父技术失败即保留并结束，不自动重试。第一神经工作为原Ls前2条真实checkpoint无query smoke，PASS立即继续全部V七景。随后仅GPU4/6 UUID，剩余>=12GB/已有compute<=2/只新增1个本任务GPUworker/单卡总<=3，其他owner和当前FullFGrad不改。环境原N607 CVS-RFFI Python/Torch2.1+cu121/FP32/TF32off，CWD/commit/命令/路径/停止规则/产物见登记表。4针对性检查和一次P0/P1已通过，当前仅排队准备，真实源smoke尚待父完整测试结束。
