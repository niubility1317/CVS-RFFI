# D102-RB-MetaBias4-qKNN追踪

|ID|要求|设计位置|当前状态|验证门|
|---|---|---|---|---|
|D102-01|唯一表示级DA delta|设计冻结§1、§6|design-frozen|无D62/D92/BCRR训练项|
|D102-02|Phase1 class-free MetaBias资产|§3、§4|analytic-initializer-only|receiver-held、class-LOCO、TX泄漏、置换；episodic trainer仍阻断|
|D102-03|每个聚合≥2物理样本|§3.2|local-verified|aggregation receipt和负测|
|D102-04|禁止raw z_dom直接匹配|§3.1|local-verified|固定U、量化、TX probe|
|D102-05|唯一4维解析求解|§5|local-verified|Λ0正定、确定性box+ellipsoid映射|
|D102-06|S_C old/new逐类等权|§5|local-verified|class permutation和new-count负测|
|D102-07|K1可辨识审计|§5|local-verified-pending-real-held|information rank、prior fraction、净纠错|
|D102-08|非共同变换|§6、§7|local-verified-pending-real-held|mask、pairwise、neighbor、margin、argmax|
|D102-09|query只读和全类竞争|§2.2、§6|local-verified|重复query、无truth/role/quota输入|
|D102-10|INT8和资源闭合|§7|partial-local-verified|合成top1/state/MAC通过；真实held teacher/INT8仍阻断|
|D102-11|Target25 DA-only|§8|release-blocked|25job/75scene/100prediction/150score|
|D102-12|当前声明边界|§8|verified-design|DA_COMPONENT_FALSIFIER_NON_PROMOTABLE|

## 独立设计评审

并行域适应设计、分类头/runner审计和独立监督已完成。最终方法级裁决为`MERGE`，`P0=0、P1=0`；仅批准进入实现和held证伪，不授权Target25。阻断项为Phase1 MetaBias4资产、真实checkpoint无queryheld证据、INT8/资源闭合及独立release复审。

## 实现复审与当前边界

- `ssr-gpu`专项、协议负测、真实checkpoint无query reachability smoke及typed qKNN回归合计`45 passed`。
- Stage2核心已实现4维同式解析、`box→ellipsoid`、S_C全部注册类逐类等权、统一support重编码、只读全类query和完整state/MAC审计。
- Phase1当前是确定性SVD解析初始化器，并非设计§4冻结的episodic gradient trainer。方法锁`docs/D102_RB_METABIAS4_PHASE1_ANALYTIC_HELD_LOCK.json`明确`target25_authorized=false`。
- held-source的类别对称门使用匿名A/B分组，仅检查任意类别子群不得单边受损，不声称Phase2真实old/new生命周期。
- TX泄漏按最差receiver-held fold执行`max balanced accuracy≤25%`，不允许均值掩盖坏fold。
- class-LOCO覆盖每个excluded class×每个held receiver的K1 fold；每行adapted balanced accuracy均不得低于raw qKNN，并进入总门控。
- 独立实现复审结论：limited diagnostic范围`P0=0、P1=0、GO`；Target25仍为`NO-GO`。episodic trainer和真实held teacher/INT8预测等价均为Target25的P1阻断项。
- 余下P2为两个显式负例测试：构造`mean≤25%但max>25%`的TX fold和单个LOCO退化并断言总状态REJECT；现有测试已验证公式、覆盖数和门控布尔值。
