# Stage2轻型DA×精简D92完整目标追踪

状态：`RETROSPECTIVE_COMPLETE / JOINT6_DESIGN_FROZEN / IMPLEMENTATION_PENDING / NO_NEW_PERFORMANCE_RESULT`

|ID|来源|要求|目标文件/证据|状态|验证|备注|
|---|---|---|---|---|---|---|
|J6-01|持久目标|最多3条原理不同的联合候选，不得缩成单候选终点|`docs/STAGE2_RD_GOAL_20260731.md`|design-frozen|C1/C2交叉审查|冻结2条：CSPAR-2与SRDH-2；RDCE同族关闭|
|J6-02|持久目标|同一checkpoint、received-IQ、receiver-held×class-LOCO、K1/K5|method lock、matrix、runner report|design-frozen|同row key与42-fold receipt|不重验`VALIDATED_ONCE`数据|
|J6-03|持久目标|共享缓存的2种表示×3种头最小6臂|联合entry/scorer|design-frozen|6臂closure和缓存receipt|Q/Full160/Lite160；formal288仅外部参考|
|J6-04|持久目标|分别证明DA、精简D92、联合方法相对对应对照提升同row H和总正确数且不损害A_old/N/floor|独立score manifest|design-frozen|三项K5主比较|禁止边际极值拼接|
|J6-05|持久目标|实测减少D92状态、拟合计算或时延|两份resource receipt|design-frozen|字节、MAC、同机时延|head因果与全管线替换分开|
|J6-06|协议|query零fit/update/selection，每query全类独立竞争|协议负测、prediction receipt|implementation-pending|focused negative tests|禁止truth/role/quota/global reassignment|
|J6-07|效率|完整负收益候选立即关闭，不调参|goal、score decision|design-frozen|冻结判据|任一K5主比较失败即关闭|
|J6-08|后续门|胜者才进入588 G0、fresh63 G1、Target25及既有K10/K5/K1门|goal、后续reports|design-frozen|顺序artifact|当前不得提前运行|
|J6-09|发布|只保留协议负测、真实checkpoint smoke、P0/P1、Git、N607 preflight|AGENTS、release report|design-frozen|release checklist|不得扩建控制面|
|J6-10|分工|Sol负责集成/分析；Terra负责科学设计、实现、复核与按实时AGENTS规定的runner；Luna只做机械任务|goal/AGENTS|design-frozen|handoff记录|若持久目标与AGENTS冲突，runner遵循AGENTS|
|J6-11|回顾|吸收D127/D128四次prediction前技术停止|本次回顾报告|implemented|4份run报告|不得再复用其Phase1 autograd/outer-audit链|

最高风险已从目标层移除：旧D127链和D129单候选四臂已退出活动目标。下一风险是科学核心尚未实现，不能把设计冻结写成方法成效。
