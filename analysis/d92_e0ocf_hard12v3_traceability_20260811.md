# D92-E0OCF Hard12-v3追溯表

设计源：`docs/superpowers/plans/2026-08-11-d92-e0ocf-hard12v3.md`。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|OCF-01|冻结方法|五臂身份固定，OCF25唯一primary、OCF50仅diagnostic|slim、hard12、config|verified_local|arm/method-lock测试|禁止结果后换主候选|
|OCF-02|冻结方法|OCF只使用同一个DA0_REG1 support上的full/block head|D92 probe|verified_local|same-after支持闭包测试|拒绝跨state OSCA|
|OCF-03|冻结方法|old weight和bias同时去旧类行均值|D92 probe|verified_local|手算FP32 fixture|缺一不可|
|OCF-04|冻结方法|block old contrast用同一旧support RMS对齐到full|D92 probe|verified_local|RMS/ratio测试|ratio不得clip|
|OCF-05|冻结方法|保留FULL_ONLY old组均值与全部new行，不二次centering|D92 probe|verified_local|byte/group residual断言|new行byte-exact|
|OCF-06|冻结方法|lambda只允许0.25/0.50，禁止class/support/query选择|D92 probe、slim|verified_local|非法mode与lambda收据漂移拒绝|OCF25唯一primary|
|OCF-07|协议|K1/K2严格D92_FULL别名|probe、query evaluator、analysis|verified_local|state/prediction exact测试|双K1 liveness均保留|
|OCF-08|协议|query fit/update/selection/truth/role/quota/global reassignment全部false|slim、query evaluator、runner|verified_local|协议负测与smoke收据测试|逐样本全类argmax|
|OCF-09|资源|OCF after实际2 fits、two-state计数4，support新增MAC/瞬态内存写实|probe、slim、analysis|verified_local|inventory、保守内存上界、fit-audit桥接|不复用旧estimated LDA字段自证|
|OCF-10|资源|query MAC和永久state bytes等于FULL_ONLY|slim、analysis|verified_local|精确相等门测试|单一FP32 affine head|
|OCF-11|矩阵|Hard12-v3 12行、覆盖、v1/v2零交集、60job/180scene-arm|hard12、config|verified_local|manifest/coverage/双K1测试|不按新候选结果选行|
|OCF-12|运行|prediction与truth-side scorer隔离、immutable输出、系统异常停派|runner|verified_local|smoke先行、exact shard集合、distinct-outer负测|复用E0D成熟路径|
|OCF-13|分析|只允许OCF25晋级并执行两组性能/资源门|analysis、analyzer|verified_local|golden与OCF50反例|OCF50不参与promotion|
|OCF-14|分析|报告old→old、old→new、new→old和同排指标|analysis、report|verified_local|混淆率与paired rows测试|禁止孤立floor结论|
|OCF-15|发布|真实checkpoint truth-free smoke、P0=0/P1=0、Git提交、唯一N607 runner|report、launch|release_ready|三项独立复审P0=0/P1=0；153项主回归通过；v2 launch已冻结，N607 smoke待runner|不增加重复数据验证|
|OCF-16|结果|完整取回、冻结分析、报告更新、晋级或否决|report、artifacts|pending|artifact counts+analyzer|Hard12-v3非正式确认|
