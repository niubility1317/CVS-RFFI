# D92严格Pareto后TCRA/CSOAS复盘（2026-08-13）

## 1.复盘范围

本次复盘承接`d92_three_round_retrospective_20260812.md`，只使用已经闭合的TCRA safe-v2和CSOAS Hard9+K1同排结果。协议继续为`p2_min_v1/VALIDATED_ONCE`；query不参与fit、update、selection、truth、role、quota或global reassignment。

## 2.新增反证

|方法|性能证据|资源证据|裁决|
|---|---|---|---|
|`E0_FULL_D42_TAIL_CLASS_ROW_ASCENT`|Hard9只通过3/8方向；old BA、old floor、forgetting等5项全部tie；8/9 outer标签完全不变|wall P90=`336.968ms`、paired median=`2.184×`，peak增量通过|support-tail逐原子保护没有迁移到held-query，且逐prefix真实复算越过硬门；拒绝TPCE/TCRA轴|
|`E0_FULL_CSOAS`|old BA和c_old各`+4.7222pp`、floor`+10.3704pp`、forgetting`-4.7222pp`；但H`-0.4233pp`、seen-new`-4.6667pp`、new→old`+3.3241pp`|wall P90=`30.895ms`、paired median=`0.2324×`、peak增量=`520,192B`，资源全过|鲁棒散度重估再次把边界推向旧类；拒绝CSOAS，不调rho、不扫参|

## 3.保留结论

D92 FULL仍是唯一同时具有强方向证据与可部署单头结构的统计基座。相对E0，它在H、old BA、old acc、floor、seen-new、forgetting和old→new七个方向改善，唯一小幅反向为new→old。下一候选应保留D92的类均值、group auto-shrinkage尺度和old/new固定等权，只削弱无法由多个类共同支持的跨块耦合。

注册后离散补丁、旧类bias、连续head微扰、第二次BLOCK fit、逐类Cauchy/OAS重估和rank-one Fisher更新都已有直接反证或排重结论。下一轮不得把这些机制换名重启。

## 4.唯一下一候选

冻结`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`。它从每类raw residual scatter中只提取off-block单位方向，以组内pairwise Frobenius cosine的闭式均值产生`rho_old/rho_new`，再在现有D92 FULL协方差与其`160/96/32`block-diagonal端点间作凸组合。raw scatter不参与最终求逆；没有新中心、人工权重、第二fit、迭代或搜索。

先运行单一K10三场景truth-free G0，只检验active、非E0 D42、真实量子、对称性、fit/query/state和资源。G0全过后才运行不重叠Hard9+K1；Hard9任一八项tie或反向即拒绝。用户已授权上述冻结门之间自动推进，不再重复请求流程性批准。
