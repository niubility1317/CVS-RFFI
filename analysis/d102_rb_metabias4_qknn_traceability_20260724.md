# D102-RB-MetaBias4-qKNN追踪

|ID|要求|设计位置|当前状态|验证门|
|---|---|---|---|---|
|D102-01|唯一表示级DA delta|设计冻结§1、§6|design-frozen|无D62/D92/BCRR训练项|
|D102-02|Phase1 class-free MetaBias资产|§3、§4|analytic-initializer-rejected|真实source-held完成；episodic trainer仍未实现|
|D102-03|每个聚合≥2物理样本|§3.2|local-verified|aggregation receipt和负测|
|D102-04|禁止raw z_dom直接匹配|§3.1|local-verified|固定U、量化、TX probe|
|D102-05|唯一4维解析求解|§5|local-verified|Λ0正定、确定性box+ellipsoid映射|
|D102-06|S_C old/new逐类等权|§5|local-verified|class permutation和new-count负测|
|D102-07|K1可辨识审计|§5|real-held-rejected|平均净纠正+5，但1/7 receiver退化|
|D102-08|非共同变换|§6、§7|real-held-verified|真实mask和argmax变化存在，但不是晋级证据|
|D102-09|query只读和全类竞争|§2.2、§6|local-verified|重复query、无truth/role/quota输入|
|D102-10|INT8和资源闭合|§7|partial-local-verified|合成top1/state/MAC通过；真实held teacher/INT8仍阻断|
|D102-11|Target25 DA-only|§8|no-go|TX泄漏和LOCO门拒绝，未启动|
|D102-12|当前声明边界|§8|verified-reject|PHASE1_SOURCE_ONLY_NOT_TARGET_PERFORMANCE|

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

## N607 r6真实证据

- run：`d102_rb_metabias4_phase1_analytic_held_20260724_r6`，commit`b963bc32`，pipeline exit=0。
- tap：8,400行、33次forward、eager/reference z_id maxabs=0、strict hook bytes=true。
- K1：BA86.2383%→86.2974%，floor+0.0714pp，net correct+5，但1/7 receiver退化。
- K5：BA85.2488%→85.2846%，floor+0.2150pp，net correct+3，0/7 receiver退化。
- K10：BA85.1000%→85.1540%，floor+0.4662pp，net correct+4，但1/7 receiver退化。
- TX泄漏：mean BA35.1190%，max BA50.3199%>25%，拒绝。
- class-LOCO：42个fold完整，9个BA退化且9/9 net correct<0，拒绝。
- bundle：numeric state7,248B、28个bank cell、class-cell物理样本32–66；`formal_phase2_eligible=false`。
- 终态：`ARTIFACTS_COMPLETE / PHASE1_HELD_FALSIFIER_REJECT / TARGET25_BLOCKED`。
