# MSAT 未限幅源特征指导：从零单阶段30轮

- run_id：`20261101-phase1-Restore-MSAT-FullFGrad-E30-s2026092701-r01`
- group_id：`restoration-full-feature-gradient-single-factor`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅移除每记录F特征输入梯度相对IQ的.5限幅；保留VarF网络/同初值/源数据/顺序/损失系数/全局clip1/30轮。源证据驱动的可证伪对照，不保证Main收益。

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

## 唯一方法变化与主验收

固定已完成VarF r03为匹配对照，只移除每记录F特征输入梯度相对IQ的0.5限幅。原MSAT4249922参数、同scratch初值/源R07二十份加前十份复用/顺序/种子/AdamW.001-warm247-cos2e-5-WD1e-4/B256/30轮7410与原IQ+.05STFT+.1(cos+源方差标准化特征MSE)均保留，全局参数梯度clip1保留；没有物理分支或额外H监督。F0冻结为原地面Tri-SIG-TIME SSL/3ff8/E50-11100/defaulthead，源特征指导，不读TX/logit/CE/原型、不另训头；OE仅冻结诊断，V/OE/novel/目标不训练、选模或早停。不存在旧G、DURA或神经IQ教师初始化。

依据为源端特征限制仍活跃和固定源反事实的剩余特征误差；平均权重约.23并不证明限幅有害，故只登记这一条可证伪对照，不作权重扫描。源指标不能代替正式收益。本版源2条真实梯度smoke及严格E0重载PASS立即从零训练，final30固定后自动原Main cross_rx126000与cross_both42000的clean+六景，所有42单元预测完成才独立CPU truth-last。raw/DURA复用同IQ-ID冻结预测，新增G逐条真实推理。每景至少140345/168000正确、83.538690%，距同F raw clean86.538690%≤3pp，并保持clean；完整分总体/TX/RX/day/修复损伤报告，性能是主线、效率仅成本项。

原N607 CVS-RFFI Python/Torch2.1+cu121、FP32/TF32off，实际commit、CWD、命令、独占路径与产物见登记表。仅GPU4/6 UUID，free>=12000MiB、原compute<=2，新增一个本任务GPU worker后单卡总<=3/自身每卡1/整体<=2，CPU truth共享原锁；不改其他owner。每轮模型/优化器/RNG仅技术恢复，全部科学结果冻结于final30。非有限/来源错配/输出冲突/技术失败保留root并停止自动重试，不以源质量取消正式测试。默认DURA与完整目标ACTIVE保持。

## 启动后核实

VERIFIED / RUNNING。实际源2条F0/G梯度及严格E0重载PASS；首轮247更新已完成，所有特征指导权重为1，E01模型/优化器/RNG独立CPU读回PASS。首轮73.579210秒，参数4249922；完整质量与成本待30轮及七景结束。GPU4准入余量22583MiB、原有2个计算进程，本任务新增1个。PID/CWD/命令/CUDA/GPU和日志进度已独立核实，其他owner保持。

## 完整执行与独立复算完成

VERIFIED / ANALYZED。完整从零30/7410及原Main两总体168000×七景预测固定后独立truth-last，4阶段全部exit0、所有本run进程退出。84单元/2604评分组/1736raw配对/1302额外配对独立复算PASS，raw/DURA判决差0；60轮日志、同初始状态/源顺序/配置仅限幅变化、30恢复文件和final30一致性通过。六景0/6、clean未保持；完整目标未达到，保持ACTIVE和默认DURA。完整分组和成本见analysis.md。
