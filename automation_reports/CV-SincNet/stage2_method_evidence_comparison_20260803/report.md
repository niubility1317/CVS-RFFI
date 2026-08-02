# Phase2自研方法证据分层与性能对比（2026-08-03）

## 1.结论先行

当前确实存在正收益版本，但必须按方法贡献拆开：

1.`D106 RDCE`是目前已重复复现的小幅正收益轻型域适应组件。
2.`D112 static ground head`在自己的完整source-held G1矩阵上是明确正收益head；但其SEAM motion没有改变任何prediction，收益不能归给SEAM。
3.`D122`中的identity ground head平均为正，但新增的`RDCE×ground head`组合降低all-class floor且factorial interaction为负，因此D122组合路线关闭。
4.`D110 SCPM`、`D121 LBR`、历史`D62`、`D91`和`SVRN-qKNN-BCRR`均不应继续调参或扩矩阵。

本报告严格区分两套不可混排证据：当前source-held G1因果矩阵与历史Target125/development诊断。跨套数值只能作为历史背景，不能直接排名。

## 2.当前项目边界

- Phase2协议为`p2_min_v1`；每个物理IQ只产生一个冻结的allowed `leo_*_weak`接收观测。
- K-shot表示K个独立物理support样本；support/query物理ID不相交。
- query只推理，禁止fit、update、selection、truth、role、quota和global reassignment。
- Phase2不得访问clean/source样本；只可使用与checkpoint共同封存的合法int8 Phase1聚合知识。
- 当前报告中的source-held G1是方法因果验证，不是Target部署结论；历史Target125也不得自动升级为当前`p2_min_v1`正式确认。

## 3.证据层级

|层级|可做的结论|方法|
|---|---|---|
|A：完整source-held G1同row因果证据|可判断本矩阵内DA/head/interaction方向|D106、D110、D112、D121、D122|
|B：完整历史Target125诊断|可判断该冻结历史矩阵内稳定性，不可与A层混排|D92、D62、SVRN-qKNN-BCRR|
|C：development/partial摘要|只能用于拒绝或设计参考，不可正式排名|D91|
|D：技术落地无performance|不得产生任何性能结论|D122-r1/r2等技术失败run|

## 4.Source-held G1正式结果

### 4.1 D104 source-held同一基线族：D106、D121、D122

三者共享同一个M0结果，可做严格横向机制比较。old BA、old floor、all floor按63行平均；seen-new、H按42个held-class行平均。

|run/arm|机制|old BA|seen-new|H|old floor|all floor|correct/row|
|---|---|---:|---:|---:|---:|---:|---:|
|D106 M0|identity Student-t qKNN|83.6560|83.7772|82.2378|57.9803|56.4199|288.9683|
|D106 M_DA|rank-3 RDCE＋qKNN|83.9163|84.1404|82.6826|58.2627|56.8637|289.8889|
|D106 M_HEAD|RCMR-2V head|83.8692|83.6562|82.1625|58.7346|57.2280|289.5556|
|D106 M_JOINT|RDCE＋RCMR-2V|84.1209|84.0194|82.4976|57.7096|55.8802|290.4762|
|D121 M_HEAD|identity＋LBR-qKNN|83.5597|83.7369|82.1950|57.7745|56.2410|288.6508|
|D121 M_JOINT|RDCE＋LBR-qKNN|83.8169|84.0194|82.5219|57.7496|56.2699|289.5397|
|D122 M_HEAD|identity＋static ground head|84.7348|85.0282|83.7209|58.8255|56.5657|292.7460|
|D122 M_JOINT|RDCE＋transported ground head|84.7233|85.0282|83.7389|58.6049|56.4257|292.7143|

### 4.2 同row简单效应

|方法贡献|Δold BA|Δseen-new|ΔH|Δold floor|Δall floor|结论|
|---|---:|---:|---:|---:|---:|---|
|D106 RDCE：M_DA−M0|+0.2604|+0.3632|+0.4447|+0.2824|+0.4438|五项同向，小幅正收益|
|D106 RCMR head：M_HEAD−M0|+0.2132|-0.1210|-0.0753|+0.7543|+0.8081|floor好，但old/new/H不一致|
|D106 joint−M0|+0.4649|+0.2422|+0.2598|-0.2707|-0.5397|均值accuracy上升但floor退化|
|D121 LBR：M_HEAD−M0|-0.0963|-0.0403|-0.0428|-0.2058|-0.1789|负收益，关闭|
|D121 LBR@RDCE：M_JOINT−M_DA|-0.0994|-0.1210|-0.1607|-0.5131|-0.5938|负收益更强，关闭|
|D122 ground head：M_HEAD−M0|+1.0788|+1.2510|+1.4831|+0.8452|+0.1457|平均正收益但receiver异质|
|D122 ground head@RDCE：M_JOINT−M_DA|+0.8069|+0.8878|+1.0563|+0.3422|-0.4380|all-floor转负，不晋级|
|D122 factorial interaction|-0.2719|-0.3632|-0.4268|-0.5030|-0.5838|组合没有互补增益|

D122的`HEAD_AT_DA`在BA/correct上为25胜/15平/23负与26胜/14平/23负；receiver 1-1贡献H `+10.723pp`和`+25 correct/row`，而18-2、19-2分别损失H `-2.870/-0.860pp`。因此平均正数不能替代跨receiver稳定性。

### 4.3 另一完整source-held族：D110与D112

该族使用自己的source-held archive和package，不能与4.1逐行配对，但D110与D112之间可以在同族内解释。以下为42个K1登记行。

|run/arm|机制|old BA|seen-new|H|old floor|all floor|结论|
|---|---|---:|---:|---:|---:|---:|---|
|M0|identity baseline|84.0388|84.0388|82.3063|59.4356|57.6720|基线|
|D110 M_DA/M_JOINT|SCPM metric|82.4515|82.4515|79.5106|49.3827|44.4444|H -2.7957pp，明显负收益|
|D112 M_HEAD_GROUND|static ground head|85.3616|85.3616|84.2799|64.0212|62.4339|old/new +1.3228pp，H +1.9736pp，正收益|
|D112 M_JOINT_SEAM|ground head＋SEAM motion|85.3616|85.3616|84.2799|64.0212|62.4339|与head逐prediction相同，SEAM贡献为0|

D112 ground head持久数值态4308B、query依赖态0B、每query额外上界960MAC；无需训练、反向传播或optimizer。D122组合态约4188B、每query额外上界960MAC，但科学上因floor和receiver异质性不晋级。

## 5.历史Target125/development矩阵

该表是历史诊断，不与第4节数值排名。B/C分别为新类注册前/后旧类指标。

|方法|row|B old|C old|B floor|C floor|seen-new|H|forgetting|证据边界|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D92|125|未在当前小型摘要中统一回收|65.56|未统一回收|未统一回收|58.93|61.57|未统一回收|历史完整125；仅历史参考|
|D62|125|81.5067|64.3933|59.7733|35.1467|59.1067|61.0887|17.1133pp|development-only；negative not promotable|
|SVRN-qKNN-BCRR/r4.2|125|73.1022|43.0333|45.1733|11.2133|23.4633|29.2506|30.0689pp|development-only；complete diagnostic negative|
|D91|7 candidates×15 outer rows|—|—|—|—|—|—|—|development diagnostic；无独立raw score；matched prediction与D62相同|

SVRN相对D62的125个matched row：C old `-21.36pp`、seen-new `-35.64pp`、H `-31.84pp`、C floor `-23.93pp`，且seen-new/H为0胜125负。因此SVRN已充分确定为弱方法，无需再跑。

D62虽然H与D92接近，但注册后旧类从81.51%降至64.39%、floor从59.77%降至35.15%，且缺少当前正式四臂authority，不能作为正收益方法。D91没有完成可替代D62的正式独立证据。

## 6.方法级裁决

|方法|裁决|是否继续|
|---|---|---|
|D106 RDCE|保留为当前轻型DA基线|继续作为DA对照，不单独调参|
|D112 static ground head|保留为当前正收益head|可作为分类头对照|
|D106 RCMR-2V|accuracy/floor冲突|不继续调|
|D110 SCPM|显著负收益|永久关闭|
|D121 LBR|identity与RDCE下均负|永久关闭|
|D122 RDCE×ground head|组件正，交互负，receiver/floor不稳|关闭组合；保留两个组件|
|D92|历史参考|不与当前G1混排|
|D62|历史完整诊断负|关闭|
|D91|development且无独立raw score|不晋级|
|SVRN-qKNN-BCRR|125/125 matched全面弱于D62|永久关闭|

## 7.下一轮研发原则

下一方法不再把两个已有正组件直接叠加，也不做超参数扫描。必须先满足：

1.在K=1时仍可由support识别，不能估计任意160维target变换。
2.对所有类保持置换对称，old/new使用同一规则。
3.显式抑制单receiver大收益主导，设计目标同时覆盖均值与receiver/floor风险。
4.复杂度保持低秩或对角：query额外复杂度优先`O(rd)`，无训练、无query状态。
5.先给数学不变性、失败条件和最小2×2因果臂；只有理论未被拒绝才实现。
6.首个真实性能验证只用完整63行G1，不跑125；弱则关闭并进入下一候选。

