# DURA单阶段noVar与原头特征指导的源分阶段对照

- run_id：`20261030-diagnostic-DURA-source-stage-compare-s2026092701-m2-r01`
- group_id：`phase1-dura-source-stage-compare`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

FrozennoVar/headFeature sourcefinals;source512x2views/10paths;reuse fixeddefault sourcepairedstagefacts;CPU2/noF-Gupdates/TX/V/target

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 完整源阶段对照与预定判定

VERIFIED/ANALYZED_SOURCE_ONLY。CPU2/0GPU共29.196秒，固定noVar与headFeature source-final各512源位置（L128/U384）×cached01/20，64个B32批次/10路径完成。默认R69同F、同位置/角色/scene-state-lock源结果精确复用，不重算已验证默认。两模型各真实source2Ls G/F smoke PASS、每批原生CG2精确复现，G/F状态不变且gradNone，0更新/直接TX/V/target。1项聚焦晋级规则与场景保护检查、唯一独立P0/P1审查PASS；一次N607 preflight/单归档两端SHA/compile通过，执行commit 645511b54c21aeae29c1c529b6fd4aa22a930e76。

PID59291存活期间独立核实CWD/argv/空CUDA绑定，日志7154bytes及headFeature最后视图320/512进度已读回；后续自然结束、日志7884bytes/64批，四份源npz完整回收。独立复算全部新source role/scene/state/lock统计及修复/损伤共15120标量，默认汇总与此前已独立复算7560标量的记录一致。原始位置、IQ衍生数组仅存私有artifact，Git只公开汇总。

|模型|缓存epoch|stage1 teacher一致率|最后prior teacher一致率|native teacher一致率|native Fcos误差|native head投影误差|
|---|---:|---:|---:|---:|---:|---:|
|DURA_default|1|90.039062%|63.085938%|94.921875%|0.01282546|0.14058048|
|DURA_default|20|92.578125%|64.453125%|97.265625%|0.01097861|0.10964842|
|noVarAux|1|86.523438%|91.406250%|93.359375%|0.01594542|0.16780807|
|noVarAux|20|92.382812%|95.898438%|96.875000%|0.01140164|0.10513840|
|headFeature|1|87.695312%|92.773438%|94.921875%|0.01271112|0.12157251|
|headFeature|20|93.554688%|97.265625%|97.851562%|0.00915896|0.08292045|

teacher一致率指与源clean F选择一致，不是真实TX准确率。headFeature较noVar的native teacher一致数分别净增8/512、5/512：源修复13/损伤5和修复8/损伤3；源Fcos误差和head投影误差均下降。最后prior仍比native少11/512、3/512一致，Fcos误差反而更大，因此两视图prior-only资格均False，不发布去掉最后求解的运行版，不放宽规则。noVar/default仅为对照，oracle A/噪声只能作源反事实，不能选父或部署。

## 已冻结源验证与共同日期对照

另在独立CPU truth-side分析中，先检查noVar/cos0.1/headFeature三模型完整source1800×7数组/物理ID/commit/无状态更新，再打开合法V标签，重新核对3×105=315个总体/TX/RX/day统计，与原source_summary一致；没有读取U_s或新target真值，也未新增训练、预测或选择checkpoint。该部分使用的是冻结后的V标签，区别于上节训练池阶段诊断V_reads=0。source_V_audit.json保留配对修复/损伤和分RX结果。

源V的day_i为原capture_date_list索引1/2/3，对应2021_03_08、2021_03_15、2021_03_23；目标既有day汇总使用日期字符串，按已核实映射对齐。目标共同三日期共126000条/景，下面目标列仅由已发布target_summary的day计数重新汇总，没有新目标输入、标签或阈值使用。

|场景|noVar源V准确率%|headFeature源V准确率%|源V净纠错|目标共同日期差pp|
|---|---:|---:|---:|---:|
|clean|98.388889|98.444444|+1|+0.200794|
|practical_high|97.944444|98.055556|+2|-0.573016|
|practical_mid|96.166667|97.166667|+18|-0.455556|
|practical_low_suburban|95.666667|96.444444|+14|-0.155556|
|practical_high_urban|96.500000|97.777778|+23|-0.446825|
|practical_mid_urban|90.000000|91.777778|+32|-0.345238|
|practical_low_urban|83.000000|85.722222|+49|-0.041270|

headFeature在未训练的源V记录上六星地均高于noVar；lowurban净多49条正确（114纠错/65损伤），五个source RX分别均有正净增益。因此收益不只局限于用于阶段诊断的训练池512位置。与此同时目标共同三日期的六景仍全部下降，不能只归因于目标多出的2021_03_01。它与跨采集条件/接收机迁移不足的解释相符，但未隔离RX、日期、记录分布等因素，不能声称已证明某一因果机制，也不把共同日期子集替代完整168000×7验收。

这支持下一步优先检验源端采集风格扰动的鲁棒性，而不是继续把同一源判别误差压低当成性能保证。新候选须先检查历史pair_aug_control/observable_response的实际机制与数据边界，明确新差异；参数和矩阵从源证据固定，不能按目标TX/场景/日期调权。当前无新权重或部署版本，默认DURA及六景3pp/clean/训练效率目标保持。
