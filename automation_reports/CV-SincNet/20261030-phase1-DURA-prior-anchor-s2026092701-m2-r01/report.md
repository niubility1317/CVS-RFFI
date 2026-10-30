# DURA内部先验配对监督与同预算续训

- run_id：`20261030-phase1-DURA-prior-anchor-s2026092701-m2-r01`
- group_id：`phase1-dura-prior-anchor-performance`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Source evidence: CG16 hurts and internal priors rely on biased learned consistency. FixedDURA v1 E20continuation vs +0.1 mean2prior pairedclean NMSE; same266951params/two priors/fourCG and automaticfrozenF50sevenvieweval

本run依据r02源诊断固定，不使用新目标分数制定方案。两行都从用户固定DURA v1.0的R69stage2 E20/4940严格加载，冻结同一DURA的两个stage输出为IQ教师。新增source pairedclean监督不是TX标签；合法Ls/Us的clean本来已为配对重建目标，仍不读取任何TX/F/V/target来更新恢复器。

对照`continuation`为相同架构常规E20续训；`prior_anchor`只增加0.1倍两个内部prior的clean归一化误差均值。原最终IQNMSE＋0.05多尺度STFT、teacher_weight线性1→0.1、全组件联合更新、globalbatch256、63000×20、AdamW重置2e-4→2e-5/warmup247/weightdecay1e-4/clip1全部匹配。每行4940步，L曝光126000、U曝光1134000；R69的已有120个组件epoch祖先成本另计，本次追加20，不能称从零20轮。

所有内部prior在训练时可见，推理仍使用原StageDistilled结构、266951参数、两prior/四CG，FP32/TF32off和既有graph/reuse路径。两行独立保存customschema权重，保留sharedalias严格校验；不覆盖默认权重、父模型、缓存或历史预测。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

本地3项检查通过：新增监督与对照差恰为0.1prior目标、梯度不进入教师、固定LR预算边界/非法arm拒绝、truth-last对最后一景NaN及错误ID拒绝。Python编译通过；本候选一次独立P0/P1审查PASS。launcher首步将实际两条Ls做原生forward精确重建、非零梯度、customschema严格存取和冻结F50原默认头无query smoke；PASS后直接排队继续，不新增smoke许可artifact。

每行固定finalE20后自动：同卡原DURA/本版graph/reuse的G与G+F、B1/B128三轮交替测速，source1800×7预测及独立评分，target168000×7预测及独立评分。每组全部七景新预测和同F50旧基线均完整有效后才能连接truth；target侧复用原rawF50和默认DURA graph七景预测，不重算旧版。评分给逐TX/RX/day、MacroF1、净纠错、相对DURA及rawclean差距。原R17每景1pp容忍将在同row结果汇总时单列。源成绩不会取消任何新版正式测试或改变另一健康行。

主要验收为用户2026-10-30已确认的六景各自距同一rawclean≤3pp，并保持clean性能；clean损伤≤1pp为原有容忍上限，并单列旧70%/5pp目标。单seed工程基准，目标已暴露；不得扩展为新盲测、当前Tri-SIG-TIME SSL或Phase2/3性能。完整源日志、训练时间、参数存储、MAC/FFT/CG、同卡时延和显存同时保留；共享负载测速不与旧独占时延直接算加速比。

当前无新训练或正式性能结果。健康其他owner不干预。启动资源按GPU可见总进程数与实际显存核对，最多两个本run GPU worker、每卡最多一个；全卡已有两条可见GPU进程则等待，CPU独立scorer最多一个。只对技术故障保留并报告，不以低准确率停机、调参或重试。总目标仍ACTIVE。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 已发布与实际启动

VERIFIED：执行代码`44bb16ed08b29e7e4ee61c21e1e1287d18791c75`已commit/push并独立核对远端OID；N607新不可变release的单archiveSHA一致、远端编译通过。CPUcontroller PID3491668的CWD/cmdline/runroot已独立读回，首步实际2Ls的两arm原生forward重建、非零梯度、严格checkpoint存取及原F50默认头smoke均PASS。

首次实际GPU训练为`continuation`，PID3492603/PPID3491668、物理GPU6、CWD/命令和日志增长已核实；E1/247次更新已完成，loss0.02422984、F_inputs0、TX_labels_read0，实际L6300/U56700曝光。`prior_anchor`当时仍排队，控制器继续按同一固定资源规则调度，不构成失败。此为阶段性读回，实时进度以原run日志/进程为准；未产生正式性能结论，不能将一行启动写成两行训练完成。

恢复工作先读本run原状态，保留健康进程。每行E20后自动评分；读取目标结果只作固定矩阵分析，不修改另一行。最终还需收集完整40轮日志、两份独立权重、两行七景及同卡成本，与DURA/原R17/同预算对照逐景比较。总目标active。
