# HQ 教学冻结归因：实现追踪

对应主实验 `20261015-phase1-sigtime-bridge-teaching-e50-manysig-s2026092701-m2-r01` 的 B15；本项不改变训练、不接触目标数据、不另训头。先完成本地实现，实际三个 final50 冻结模型齐备后才登记发布一次源域零更新分析。当前主实验仍在训练，本文件不是运行证明。

|ID|需求|目标位置|状态|验证及缺项|
|---|---|---|---|---|
|M01|只读同折 final50 原头，L_dev720，无 U/V/target 前向或拟合|cvs_urban_bridge_mechanism/design.py、probe.py|verified|严格资格、原头及物理fold守卫已实现，源资格/真实冻结模型检查PASS；实际加载新final50待运行|
|M02|同传播/接收状态的 full 与独立 HQ，源随机流单列|probe.py|verified|复用已验证coupled并检查实际证明字段，固定新增噪声−6dB/独立源流；本次实际720×6算子核对待运行|
|M03|固定 A0 clean 正确集合及 CE 对照的修复/破坏、默认头间隔、分层|probe.py|verified|固定集合/修复破坏/缺失CE比较null负测PASS，只用原头；真实源指标待运行|
|M04|实际 raw112 三组、共同源尺度及各自尺度，身份几何/方差/融合贡献|probe.py|verified|组误差与实际bounded objective一致、dt排除/非法标签检查PASS；实际SIG捕获及冻结state/grad检查PASS|
|M05|三模型/两视图/六源场景、真实前向数/状态/墙钟；原训练完整日志核算|design.py、probe.py、full_log_audit.py|pending|预计56882源前向、0 optimizer/EMA/head-fit；训练审计复用旧控制完整证据，只新读两个11100步日志，实际运行待完成|
|M06|源报告/图/登记/唯一发布及独立完成读回|figures.py、publish.py、registration.json|pending|冻结前不发布，不以主实验训练启动冒充本分析已运行|

计数：verified4/pending2/deferred0/rejected0/blocked0。最高风险是训练局部相似改善却未增加默认头间隔；共同尺度及匹配 CE 控制可区分一部分原因，单 seed 与源代理仍限制结论。物理与网络路径保持严格一致，HQ 只是减少模拟新增噪声的可操作近似。

本地原4项检查3PASS/1fixture路径失败后，定向资格PASS.010s；新增发布接线PASS.006s。compileall通过。真实checkpoint smoke留在发布runner首步；当前无分析结果或图。
