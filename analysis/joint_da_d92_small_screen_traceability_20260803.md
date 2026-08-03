# 轻型DA×精简D92联合小筛选追踪

设计源：`docs/STAGE2_RD_GOAL_20260731.md`§1、§4、§5、§8、§10

当前状态：`GOAL_REVISED / DESIGN_NOT_FROZEN / IMPLEMENTATION_NOT_STARTED / NO_NEW_PERFORMANCE_RESULT`

本表只追踪会直接影响方法正确性、因果可识别性或下一次真实小筛选运行的项目。D119-CFO真值门、D102/D124全量回放、重复数据验证、通用authority、额外receipt和报告美化均为非阻塞P2，不进入本表。

|ID|目标要求|目标文件或工件|状态|完成证据|是否阻塞小筛选|
|---|---|---|---|---|---|
|JD92-01|最多3条原理不同的“DA+精简D92”联合候选；浅/中/晚层不是连续层扫描|联合设计卡|pending|候选公式、唯一干预点、共享项、删除项一次冻结|是|
|JD92-02|梯度型DA采用Phase1物理隔离`S_src→a¹→Q_src`元目标；Phase2只更新低维状态，基础模型与封存方向不更新|DA核心模块|pending|任务梯度非零、参数更新集合断言、S/Q物理ID互斥|是|
|JD92-03|无反传晚层候选与梯度候选保持独立机制；不叠加A+B或临时组合|候选feature provider|pending|每个候选唯一hook及identity parity|是|
|JD92-04|精简D92删除old/new角色分裂、双160×160协方差及部署态无用矩阵；全注册类标签置换等价|D92-Lite核心模块|pending|置换等价、紧凑状态、历史D92对照|是|
|JD92-05|K5形成真实D92-Lite交互；K1若统计不可辨识则显式qKNN边界，不伪造类内方差|D92-Lite状态与评分|pending|K1边界测试、K5非别名预测|是|
|JD92-06|冻结最小6臂`M0/M_DA/M_D92/M_DA_D92/M_L92/M_JOINT`，同一行不得跨run补臂|联合runner/scorer|pending|6臂键一致、状态复用和反事实映射测试|是|
|JD92-07|base/adapted各只做必要forward；qKNN、历史D92和精简D92共享规范化特征、support索引和query缓存|特征提供器与runner|pending|forward计数、cache binding receipt|是|
|JD92-08|同一真实checkpoint/received-IQ、receiver-held×class/TX-LOCO、K1/K5；先完整prediction后打开truth|矩阵卡与run报告|pending|冻结行清单、ID不交叉、truth-side独立scorer|是|
|JD92-09|小筛选只用方向性双收益门：DA和L92在各自必要对照上提升`H`与总正确数，单项退化不低于−0.5pp，receiver `H`不低于−2pp|分析规格|pending|同row paired表和词典序排序|否；只决定结果后晋级|
|JD92-10|精简D92目标：部署态≥90%缩减、K5拟合时间≥50%缩减、无额外backbone forward、K5 query MAC不增加|资源receipt|pending|历史D92与L92同机同输入实测|否；不阻塞运行，阻塞最终胜出|
|JD92-11|本地只做协议负测、真实checkpoint无query smoke、聚焦单测和独立`P0=0/P1=0`；不新增控制面|测试与审查记录|pending|`ssr-gpu`命令、review结论|是|
|JD92-12|本地Git提交和不可覆盖run报告后，由唯一runner把3候选分配到不同GPU；runner不改方法、不按性能停止或重跑|Git commit、run报告、runner handoff|deferred|commit/hash/命令/GPU/路径/health-stop冻结|是，但仅在实现后|

## 结果后停止语义

- 某候选完整小筛选不满足方向性双收益：`COMPLETED_DIAGNOSTIC_NEGATIVE / CLOSE_CANDIDATE`；
- 三候选均负：保存完整证据，转向新的方法原理，不在层、rank、步数、view、shrinkage或阈值上调参复活；
- 仅胜者进入一次fresh63行G1；小筛选不得冒充Target性能；
- 协议或确定性执行错误才允许技术停止；中间性能弱不得停止正在运行的完整冻结矩阵。

## 实现前必须补齐

1.三条联合候选的公式与非重叠文件所有权；
2.D92-Lite的解析收缩、logit组合及K1边界；
3.真实IQ feature provider与6臂共享缓存接口；
4.冻结source-held行集合、GPU分配和不可覆盖run ID。
