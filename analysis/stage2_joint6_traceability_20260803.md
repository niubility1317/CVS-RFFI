# Stage2轻型DA×精简D92完整目标追踪

阶段：`RETROSPECTIVE_COMPLETE / JOINT6_DESIGN_FROZEN / IMPLEMENTATION_REVIEW / NO_NEW_PERFORMANCE_RESULT`

状态：`implemented`

|ID|来源|要求|目标文件/证据|状态|验证|备注|
|---|---|---|---|---|---|---|
|J6-01|持久目标|最多3条原理不同的联合候选，不得缩成单候选终点|`docs/STAGE2_RD_GOAL_20260731.md`、method lock|verified|C1/C2交叉审查|冻结2条：CSPAR-2与SRDH-2；RDCE同族关闭|
|J6-02|持久目标|同一checkpoint、received-IQ、receiver-held×seen-class-LOCO、K1/K5|method lock、matrix、runner report|implemented|42fold×2K=84row/候选|仅方向性proxy；不产生正式new-registration结论|
|J6-03|持久目标|共享缓存的2种表示×3种头最小6臂|联合entry/scorer|verified|35项聚焦测试中的六臂与公共R0测试|每原子row的R0三头只fit一次，候选refit=0|
|J6-04|持久目标|方向性proxy分别比较DA、Lite和联合；正式H/A_old/N只在Target25|独立score manifest|implemented|三项K5 proxy主比较|source-held只输出`H_retained_held_proxy`等字段|
|J6-05|持久目标|实测减少D92状态、拟合计算或时延|两份resource receipt|implemented|解析字节/MAC已实现；真实同机测量待实验|正式90%/50%/40%只由Target25同机receipt判定|
|J6-06|协议|query零fit/update/selection，每query全类独立竞争|协议负测、prediction receipt|verified|35项聚焦测试|禁止truth/role/quota/global reassignment；fold binding重算校验|
|J6-07|效率|完整负收益候选立即关闭，不调参|goal、score decision|verified|冻结proxy判据|任一K5代理主比较失败即关闭|
|J6-08|后续门|方向性胜者才进入588 G0、fresh63 proxy、Target25及既有K10/K5/K1门|goal、后续reports|verified|顺序artifact|正式new-registration只在Target25|
|J6-09|发布|只保留协议负测、真实checkpoint-derived archive smoke、P0/P1、Git、N607 preflight|AGENTS、release report|pending|release checklist|尚缺真实archive执行、复审、Git提交与runner交接|
|J6-10|分工|Sol负责集成/分析；Terra负责科学设计、实现、复核与按实时AGENTS规定的runner；Luna只做机械任务|goal/AGENTS|verified|handoff记录|runner按AGENTS使用Terra Max，Luna不接触SSH|
|J6-11|回顾|吸收D127/D128四次prediction前技术停止|本次回顾报告|verified|4份run报告|不得再复用其Phase1 autograd/outer-audit链|

最高风险已从目标层移除：旧D127链和D129单候选四臂已退出活动目标。当前实现和合成测试不等于方法成效；下一硬证据是固定真实archive的no-truth smoke及完整proxy prediction。
