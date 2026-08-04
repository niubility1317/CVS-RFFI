# NEXT-R6轻型DA×精简D92联合研发追溯与三轮回顾

状态：`RETROSPECTIVE_COMPLETE / RPPF160_DESIGN_FROZEN / IMPLEMENTATION_PENDING / NO_RUN_AUTHORITY`

## 1.目标逐项追溯

|ID|来源|要求|目标文件或证据|状态|验证|说明|
|---|---|---|---|---|---|---|
|R1|持续目标、`项目.md`|遵守`p2_min_v1`、单LEO观测、K个独立物理support、query零fit/update/selection、全注册类逐样本竞争|新method lock、runtime负测、真实checkpoint smoke|implemented|冻结设计已闭合输入与禁止项；代码证据待补|不得借用外部source/clean状态|
|R2|持续目标|同时研发轻型DA与精简D92头，不能把Q-only当最终联合方案|`analysis/next_r6_rppf160_design_20260805.md`、六臂矩阵|verified|FA-RDCE3×RPPF160设计级P0=0/P1=0|FA不改公式；RPPF为唯一新头|
|R3|持续目标|候选不超过3条、完整负收益立即关闭、不盲调|本文件、实验报告|verified|D130两候选、CER与LOO头均按完整或独立反证关闭|NEXT-R6本轮只允许1条原理不同候选|
|R4|持续目标|同一真实checkpoint/received-IQ上执行2种表示×3种头共享缓存六臂|NEXT-R6 plan/runtime/scorer|implemented|24条件/288 surface已冻结；代码待实现|臂为DA0/DA1×Q/Full160/RPPF160|
|R5|持续目标|分别证明DA、Lite头和联合替换提升同row H与总正确数且不牺牲old、seen-new和floor|四状态score与主效应表|pending|尚无NEXT-R6性能|REG0的seen-new/H必须为`N/A`|
|R6|持续目标|实测降低D92状态、拟合计算或时延|resource receipt|implemented|解析状态162C B、query160C MAC已冻结；实测待补|必须与formal D92资源口径分开|
|R7|持续目标|胜者才进入G0、fresh63、单seed Target25与后续门|阶段报告|deferred|当前尚无联合胜者|不得用Q-only Target5替代该顺序的联合证据|
|R8|`AGENTS.md`|发布门仅保留实际入口、聚焦负测、真实checkpoint smoke、独立P0/P1、Git提交和N607预检|报告与runner handoff|pending|未到发布阶段|重复数据验证、额外签名与通用平台不阻塞|
|R9|用户与`AGENTS.md`|Sol负责整合/分析；Terra/max负责科学设计/实现/审查；Luna/max负责冻结机械任务和唯一runner|agent handoff与报告|verified|当前研究作者均为Terra/max，主agent为Sol|方法作者不得自证|
|R10|用户四状态命名|统一使用DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1|plan、artifact、表格与对话|verified|`AGENTS.md`与现有Target5 plan均已落地|REG0新类指标严格`N/A`|

## 2.三轮探索回顾

### 2.1D130联合六臂Proxy84

D130在7receiver×6 held-class×K1/K5上完成两候选共168条candidate-row prediction和独立score。CSPAR-2的K5 DA主效应为`ΔH=-0.556pp、总正确数-9`；SRDH-2的K5 DA主效应为0。Lite160相对Full160虽有`ΔH=+0.164pp、总正确数+14`，但held-proxy准确率下降0.529pp、最差旧类下降1.270pp。结论：共享表示变换会负迁移或决策恒等；共享对角Lite头的平均收益不足以保护尾部。

### 2.2D138 D92-Lite-FULL288 Target125

D138 r4完整闭合125 outer、375 scene、750 surface和独立truth-side score。总体域适应前/新类注册前old为78.59%，域适应前/新类注册后old为60.35%，seen-new为41.96%，H为51.21%，forgetting为18.24个百分点，after-old floor为32.49%。K5/new20的old、seen-new、H分别为59.83%、43.24%、49.87%。该run只执行`M_JOINT`，没有六臂或四状态因果覆盖，不能识别头部、表示和注册效应。结论：真实执行链可复用，但FULL288头不能作为联合成功证据。

### 2.3NEXT-R4 FA-RDCE3×CER Proxy24

NEXT-R4 R2完整闭合24逻辑行、144唯一prediction和192 arm artifact。K5直接qKNN中，域适应后/新类注册前相对域适应前/新类注册前old BA`+1.667pp`、old-floor`+10.000pp`；域适应后/新类注册后相对域适应前/新类注册后old、seen-new与H均`+1.852pp`、all-floor`+11.111pp`、总正确数`+12`，9行正、3行平、0行负。K1适配H下降1.392pp；CER头相对Q的K5 H下降27.060pp且12/12行负。结论：只保留K5 FA-RDCE3；任何新头必须与Q形成独立、可辨识、尺度一致的主效应，不能再靠无证书残差混合。

## 3.NEXT-R6设计约束

1.本轮只允许1条原理不同的Lite头候选；不复活CSPAR、SRDH、CER、D91、FULL288单臂或已拒绝的LOO线性混合。
2.FA-RDCE3的公式、rank、量化、`rho`和Wiener系数保持冻结；K1不得伪造类内方差或域位移可辨识性。
3.六臂必须共享同一support/query、基础feature缓存和FA feature缓存；Full/Lite只改变head，Q不重复计算。
4.头部必须类置换等变、全注册类同式、support-only；禁止old/new role、query truth、quota、跨query分配和性能选参。
5.发布前先用source-held矩阵证明DA、Lite和联合三个主效应；任何一项完整负收益立即关闭，不进入Target。
6.`code/cvsrffi/stage2_next_r5_target5_plan.py`当前仅为四状态Q-only计划骨架，不具有NEXT-R6联合实验发布权限；待联合候选胜出后再决定复用或替换。

## 4.当前最高风险

设计级可辨识性已通过RPPF的独立分类分数闭合，不再混合Q/L温标。当前最高风险转为实现级因果错配：R0/R1若复用错误head state，或K1重复拟合而不是alias，会制造伪DA效应。聚焦测试和真实checkpoint smoke必须直接覆盖这两点。
