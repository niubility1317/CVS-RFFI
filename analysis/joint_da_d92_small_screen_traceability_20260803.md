# 轻型DA×精简D92联合小筛选追踪

设计源：`docs/STAGE2_RD_GOAL_20260731.md`§1、§4、§5、§8、§10

当前状态：`GOAL_REVISED / DESIGN_FROZEN / IMPLEMENTATION_NOT_STARTED / NO_NEW_PERFORMANCE_RESULT`

本表只追踪会直接影响方法正确性、因果可识别性或下一次真实小筛选运行的项目。D119-CFO真值门、D102/D124全量回放、重复数据验证、通用authority、额外receipt和报告美化均为非阻塞P2，不进入本表。

|ID|目标要求|目标文件或工件|状态|完成证据|是否阻塞小筛选|
|---|---|---|---|---|---|
|JD92-01|最多3条原理不同的DA候选共享一个精简D92；浅/中/晚层不是连续层扫描，不开发3个head|联合设计卡|design-frozen|`joint_da_d92_lite_codesign_frozen_20260803.md`§3–§4|是；只缺实现|
|JD92-02|梯度型DA采用Phase1物理隔离`S_src→a¹→Q_src`元目标；Phase2只更新低维状态，基础模型与封存方向不更新|DA核心模块|design-frozen|联合冻结设计§3.1|是；只缺实现|
|JD92-03|无反传晚层候选与梯度候选保持独立机制；不叠加A+B或临时组合|候选feature provider|design-frozen|联合冻结设计§3.2|是；只缺实现|
|JD92-04|精简D92删除formal288维管线的old/new角色分裂、重复稠密拟合、D62行拼接及无独立贡献的FFT96/RF32块；全注册类标签置换等价；与formal D92的差值只称全管线替换差|D92-Lite核心模块|design-frozen|联合冻结设计§4–§5|是；只缺实现|
|JD92-05|K5形成真实D92-Lite交互；K1若统计不可辨识则显式qKNN边界，不伪造类内方差|D92-Lite状态与评分|design-frozen|联合冻结设计§4|是；只缺实现|
|JD92-06|同一160维空间冻结核心`2×2`：`M0/M_DA/M_L92/M_JOINT`；`R_D92_FORMAL`只作公共同row全管线参照；`M_DA_D92`仅S1胜者可选诊断|联合runner/scorer|design-frozen|联合冻结设计§5|是；只缺实现|
|JD92-07|base/adapted各只做必要forward；qKNN和D92-Lite共享规范化z160、support索引和query缓存；formal参照独立复用z160+FFT96+RF32，不冒充同head|特征提供器与runner|pending|forward计数、cache binding receipt|是|
|JD92-08|Phase1 source-held只做资产审计；S0固定3receiver×K1/K5×3scene=18行，S1固定剩余2receiver×K1/K5/K10×3scene=18行；先完整prediction后评分|矩阵卡与run报告|design-frozen|联合冻结设计§6；精确receiver清单在run报告一次冻结|是；仅发布前一次冻结|
|JD92-09|S0仅保留3个方向条件：DA的`ΔH>0`、K5 Lite-after-DA的`ΔH>0`、联合`ΔH>0`且总正确数增加；其余指标报告不设0.5pp硬门|分析规格|design-frozen|联合冻结设计§6|否；只决定结果后晋级|
|JD92-10|相对formal Target D92：单平面INT8`B_lite=164C`、C26状态减少74.1%，联合状态≥50%缩减、K5拟合MAC≥90%缩减、拟合时间≥50%缩减、query head MAC减少44.44%；另报DA端到端资源|资源receipt|analytic-frozen/empirical-pending|联合冻结设计§7；同机实测待实现|否；不阻塞S0，阻塞最终胜出|
|JD92-11|本地只做协议负测、真实checkpoint无query smoke、聚焦单测和独立`P0=0/P1=0`；不新增控制面|测试与审查记录|pending|`ssr-gpu`命令、review结论|是|
|JD92-12|本地Git提交和不可覆盖run报告后，由唯一Terra Max runner把3候选分配到不同GPU；Luna不执行SSH或实验；runner不改方法、不按性能停止或重跑|Git commit、run报告、runner handoff|deferred|commit/hash/命令/GPU/路径/health-stop冻结|是，但仅在实现后|

## 结果后停止语义

- 某候选完整小筛选不满足方向性双收益：`COMPLETED_DIAGNOSTIC_NEGATIVE / CLOSE_CANDIDATE`；
- 三候选均负：保存完整证据，转向新的方法原理，不在层、rank、步数、view、shrinkage或阈值上调参复活；
- 仅胜者进入剩余2receiver的S1；S1失败不递补runner-up；S0/S1均不得冒充Target性能；
- 协议或确定性执行错误才允许技术停止；中间性能弱不得停止正在运行的完整冻结矩阵。

## 实施与发布待补

1.按非重叠文件所有权实现DA核心、D92-Lite核心和核心4臂整合；
2.完成聚焦协议负测、真实checkpoint无query smoke及独立代码级`P0=0/P1=0`；
3.在run报告一次冻结S0/S1精确receiver清单、GPU分配和不可覆盖run ID；
4.本地Git提交后交给唯一Terra Max runner。
