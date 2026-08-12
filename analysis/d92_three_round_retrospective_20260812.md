# D92严格Pareto研发三轮回顾（2026-08-12）

## 1.目标与协议复核

- 主基线仍为`E0_FULL_ONLY`；目标仍是八项同排均值全部严格改善，同时query MAC、永久state不增加，注册计算显著低于完整D92。
- 协议仍为`p2_min_v1`，复用`VALIDATED_ONCE`数据；query不得参与fit、update、selection、truth、role、quota或global reassignment。
- 困难验证仍只使用冻结Hard10的10个performance outer加1个K1 liveness outer，不能替代完整Target125。
- 三轮均已完成独立实现、真实N607证据和不可覆盖artifact；本回顾完成前不发布第四条路线。

## 2.三轮结果

|轮次|方法|合法性能或部署结果|资源结果|结论|
|---|---|---|---|---|
|1|`E0_FULL_MAXMIN_FLOORBOOST`|相对E0：old floor`+10.3333pp`、old BA`+2.9444pp`、forgetting`-2.9444pp`；但seen-new`-12.1583pp`、H`-6.0634pp`、new→old`+17.4667pp`|wall中位数`142.708ms`、P90`199.324ms`、配对中位倍率`1.907×`|旧类整体bias把old-vs-new边界推向旧类；拒绝该机制及其系数扫描|
|2|`E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN`|10/10 performance outer均active且部署头非E0字节，但八项与E0全部精确tie；安全扰动没有改变最终类别决策|终态30个same-outer/same-scene资源配对：wall P90`179.172ms`、配对中位比`1.74784×`、peak最大增量`5,951,488`字节；query/state精确相等|连续扰动加多轮D42回缩既慢又无决策效应；拒绝该路线|
|3|`E0_FULL_BLOCK_PARETO_DISTILL`|真实K10 smoke三场景中，连续唯一候选经D42后均与E0完整头byte-exact并回退；正式score为0|smoke诊断wall中位数`280.772ms`、P90`283.718ms`、peak最大`15,040,512B`；实际fit=2|双fit连续蒸馏无法跨越D42格点且明显超资源；拒绝该路线|

## 3.保留与否决的科学结论

- 保留：E0的288维联合表示、task-balanced covariance、单FULL注册头、F0 query头、K≤2精确D92 FULL alias。
- 保留：旧类弱尾、pooled新类margin和双向混淆是必要support代理，但它们只能作为同一确定性公式的约束，不能从多个候选中择优。
- 否决：旧类统一加分、按历史困难类/receiver分支、FloorBoost强度扫描、NewGuard多级回缩、FULL+BLOCK第二次统计拟合、连续候选量化后再判定是否可用。
- 关键新事实：仅改变部署字节仍可能不改变决策；下一候选必须在构造时直接跨越至少一个真实D42分类margin量子，并在实际解码头上闭合support保护。
- 关键资源事实：额外BLOCK fit或多轮codec会把wall推过`150ms/1.50×E0`硬门；下一候选只能复用单FULL充分统计量并做一次小型码空间更新。

## 4.第四轮方法边界

- 方法家族切换为“单次D42离散码空间Pareto步”，从部署态E0码字出发直接选择有限整数码差，不再先构造连续头再量化。
- 只允许一次确定性support-only求解；不得扫描步长、lambda、候选arm或Hard10结果。
- 码差必须类标签置换等变，并同时约束六旧类固定弱尾、pooled新类margin、old→new和new→old代理；任一方向不闭合则精确回退E0。
- K>2实际FULL fit保持1；K≤2保持D92 FULL精确alias；query头、MAC和永久state与E0精确相同。
- 真实D42解码头必须非E0 byte-exact且产生至少一个跨量子support margin变化，否则不得发布性能矩阵。
- 先用真实checkpoint K>2 truth-free smoke验证“非零码差、保护闭合、资源门”；通过后才运行同一冻结Hard10+K1。

## 5.证据路径

- E0完整Target125：`E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1\analysis`。
- FloorBoost：`E:\type10-7\local_artifacts\d92_e0_full_maxmin_floorboost_hard11_20260812_v1\analysis`。
- NewGuard v3：`E:\type10-7\local_artifacts\d92_e0_full_bidirectional_newguard_hard11_20260812_v3\analysis`。
- Pareto Distill：`E:\type10-7\local_artifacts\d92_e0_full_block_pareto_distill_hard11_20260812_v1`。
