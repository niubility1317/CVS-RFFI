# HQ 教学冻结归因：实现追踪

对应主实验 `20261015-phase1-sigtime-bridge-teaching-e50-manysig-s2026092701-m2-r01` 的 B15；本项不改变训练、不接触目标数据、不另训头。先完成本地实现，实际三个 final50 冻结模型齐备后才登记发布一次源域零更新分析。三个实际final50已冻结，本分析已执行并完成；原主训练完整step审计另待取回。

|ID|需求|目标位置|状态|验证及缺项|
|---|---|---|---|---|
|M01|只读同折 final50 原头，L_dev720，无 U/V/target 前向或拟合|cvs_urban_bridge_mechanism/design.py、probe.py|verified|严格资格、原头及物理fold守卫已实现，源资格/真实冻结模型检查PASS；真实三个final50/default head/ownEMA已严格加载，冻结state及grad检查通过|
|M02|同传播/接收状态的 full 与独立 HQ，源随机流单列|probe.py|verified|复用已验证coupled并检查实际证明字段，固定新增噪声−6dB/独立源流；实际720×6 full逐值及同H/receiver/kernel/AGC核对通过|
|M03|固定 A0 clean 正确集合及 CE 对照的修复/破坏、默认头间隔、分层|probe.py|verified|固定集合/修复破坏/缺失CE比较null负测PASS，只用原头；真实fixed703：SIG对CE修复11/新错7，head间隔未增；分层产物完整|
|M04|实际 raw112 三组、共同源尺度及各自尺度，身份几何/方差/融合贡献|probe.py|verified|组误差与实际bounded objective一致、dt排除/非法标签检查PASS；实际SIG捕获及冻结state/grad检查PASS|
|M05|三模型/两视图/六源场景、真实前向数/状态/墙钟；原训练完整日志核算|design.py、probe.py、full_log_audit.py|pending|源实际56882/50.256831s/0更新完成；原主训练两个11100步及旧控制的完整日志合并审计仍待执行|
|M06|源报告/图/登记/唯一发布及独立完成读回|figures.py、publish.py、registration.json|verified|archive/compile/preflight/真实CPsmoke通过；首次PID已自然结束，终态resolved/progress/results/complete/log独立核实；三图已生成并检查|

计数：verified5/pending1/deferred0/rejected0/blocked0。最高风险是训练局部相似改善却未增加默认头间隔；共同尺度及匹配 CE 控制可区分一部分原因，单 seed 与源代理仍限制结论。物理与网络路径保持严格一致，HQ 只是减少模拟新增噪声的可操作近似。

本地原4项检查3PASS/1fixture路径失败后，定向资格PASS.010s；新增发布接线PASS.006s。compileall通过。真实checkpoint smoke在发布runner首步PASS；实际源结果及三图已完成，不构成正式或跨seed结果。
