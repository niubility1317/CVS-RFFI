# 冻结四步恢复轨迹：每一步实际改变了什么

**本报告只检验固定模型在原训练源面板上的逐步变化。它不训练、不选模，不以源端判决保持代替 Main 真实 TX 性能，也不给恢复极限。** 两版各自原 Main 七景技术流程完成后，才执行本诊断；控制器只等待完成状态，没有读取父预测或分数。

原 L_s/U_s 固定512条（128 L_s、384 U_s），20份缓存draw，共10240条相关观测。两版各4步共81920条轨迹记录，完整组数96。F(clean)判决是参考，真TX标签未读取，512条重复观测不能当作10240条独立记录。

![源端轨迹](E:/type10-7/automation_reports/CV-SincNet/20261101-diagnostic-Restore-bridge-trajectory-source-s2026092701-r01/source_trajectory.png)

| 模型 | 场景 | 步数 | 数量 | IQ NMSE | 输出Fclean一致率 | latent Fclean一致率 | 相对首步修复 | 损伤 | 净变化 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DirectBridge | ALL | 1 | 10240 | 0.02909790 | 95.869141% | 94.951172% | 0 | 0 | +0 |
| DirectBridge | ALL | 2 | 10240 | 0.02942937 | 95.859375% | 95.224609% | 28 | 29 | -1 |
| DirectBridge | ALL | 3 | 10240 | 0.02953710 | 95.849609% | 95.380859% | 29 | 31 | -2 |
| DirectBridge | ALL | 4 | 10240 | 0.02955640 | 95.820312% | 95.292969% | 26 | 31 | -5 |
| StateBridge | ALL | 1 | 10240 | 0.02687188 | 96.162109% | 95.625000% | 0 | 0 | +0 |
| StateBridge | ALL | 2 | 10240 | 0.03236510 | 95.419922% | 95.302734% | 69 | 145 | -76 |
| StateBridge | ALL | 3 | 10240 | 0.04855991 | 94.931641% | 94.833984% | 95 | 221 | -126 |
| StateBridge | ALL | 4 | 10240 | 0.05329727 | 94.902344% | 94.892578% | 100 | 229 | -129 |
| DirectBridge | practical_low_urban | 1 | 1436 | 0.08396486 | 84.749304% | 82.172702% | 0 | 0 | +0 |
| DirectBridge | practical_low_urban | 2 | 1436 | 0.08442150 | 84.888579% | 82.799443% | 13 | 11 | +2 |
| DirectBridge | practical_low_urban | 3 | 1436 | 0.08456443 | 84.958217% | 83.495822% | 14 | 11 | +3 |
| DirectBridge | practical_low_urban | 4 | 1436 | 0.08453908 | 84.749304% | 82.869081% | 10 | 10 | +0 |
| StateBridge | practical_low_urban | 1 | 1436 | 0.07612306 | 86.350975% | 84.192201% | 0 | 0 | +0 |
| StateBridge | practical_low_urban | 2 | 1436 | 0.08547184 | 84.331476% | 83.286908% | 36 | 65 | -29 |
| StateBridge | practical_low_urban | 3 | 1436 | 0.10861452 | 82.311978% | 81.476323% | 44 | 102 | -58 |
| StateBridge | practical_low_urban | 4 | 1436 | 0.11645158 | 81.615599% | 81.615599% | 45 | 113 | -68 |
| DirectBridge | unlocked | 1 | 1063 | 0.14225256 | 73.283161% | 68.203198% | 0 | 0 | +0 |
| DirectBridge | unlocked | 2 | 1063 | 0.14276695 | 73.565381% | 69.332079% | 17 | 14 | +3 |
| DirectBridge | unlocked | 3 | 1063 | 0.14290767 | 73.471308% | 70.460960% | 17 | 15 | +2 |
| DirectBridge | unlocked | 4 | 1063 | 0.14286334 | 73.377234% | 70.084666% | 17 | 16 | +1 |
| StateBridge | unlocked | 1 | 1063 | 0.13205424 | 74.976482% | 71.777987% | 0 | 0 | +0 |
| StateBridge | unlocked | 2 | 1063 | 0.14378740 | 70.460960% | 68.673565% | 47 | 95 | -48 |
| StateBridge | unlocked | 3 | 1063 | 0.17060619 | 66.603951% | 65.663217% | 55 | 144 | -89 |
| StateBridge | unlocked | 4 | 1063 | 0.18122739 | 65.569144% | 65.475071% | 54 | 154 | -100 |

## 能说明与不能说明的内容

DirectBridge在该冻结源低档城区面板中，第4步相对第1步的Fclean一致率变化+0.000000个百分点，净变化+0条；输出相对首步的IQ变化NMSE均值为0.00085176。这些量测可以区分‘四步实际有变化’与‘变化是否保持参考判决’，但不能直接转译为真实TX纠错、未知设备保真或目标泛化收益。
StateBridge在该冻结源低档城区面板中，第4步相对第1步的Fclean一致率变化-4.735376个百分点，净变化-68条；输出相对首步的IQ变化NMSE均值为0.05252746。这些量测可以区分‘四步实际有变化’与‘变化是否保持参考判决’，但不能直接转译为真实TX纠错、未知设备保真或目标泛化收益。

DirectBridge同时输入中间state、原received和时间；StateBridge的估计器只输入state/time/坐标，received仅初始化state。训练配对的解析捷径及理想固定点已由合成IQ验证，但实际神经网络并非该解析式；这里的轨迹差异也不能独立证明某个Main差值由捷径造成。[解析限制](../20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01/paired_state_shortcut.md)。

Main性能另见[DirectBridge完整七景](../20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01/analysis.md)与[StateBridge完整七景](../20261101-phase1-Restore-StateBridge-E30-s2026092701-r01/analysis.md)。原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景，各景距同一raw clean≤3pp并保持clean仍是主验收，默认DURA不更换。源端更好不自动成为通过证据，恢复IQ更像clean也不等于发射机信息恢复了某个比例。

## 输入、执行与独立核对

仅加载Direct84a063c9/Statec50a9159的固定final30/7410，严格核对配置、参数、实际来源和0TX/V/OE/目标训练；两版训练祖先从零，不由目标成绩选择。本诊断的源clean只作IQ/F0参考，F0原3ff8/defaulthead固定。没有模型更新、优化器、拟合头或阈值，OE/novel/目标IQ/真值均未读取，方法仍明确F-guided来源。

真实源2首先核对手工记录的第4步逐值等于G.forward，实际F0/defaulthead输出有限，随后全部20draw；G和F0结束状态逐tensor不变。冻结4步每模型40960 core，共81920 Gcore和81920输出F0特征；512源clean F0参考、81920次latent原默认头读出，仅为诊断。源2额外32 Gcore和4 F0特征。调用计数是请求行，不把F0内部三视图当作额外独立观测。

独立检查对齐原源面板、覆盖全部81920条输出/latent判决、96组均值和相对首步配对。最大汇总误差0，输出单位功率最大误差3.58e-07；全部20份draw及完整日志核实。旧源诊断Fclean参考差异为{'DirectBridge': 0, 'StateBridge': 0}，该跨执行参考比较不作为新增门槛；当前两模型各步使用相同本次Fclean参考。复算新增神经前向、训练、TX/V/OE/目标读取均为0。

本次完整worker耗时54.038849秒，包含加载、源I/O、G/F0和统计，不是隔离部署时延。共享GPU负载与模型调用成本不能用来替代识别收益。source_records.npz保持私有，公开CSV/JSON只含聚合，无逐记录IQ或embedding。
