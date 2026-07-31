# Stage2基线完整证据对比

状态：`EVIDENCE_BOUND / SAME_ROW_ONLY`

本报告只比较已经存在的协议合法artifact。`F_old,row`是先在每个job/scenario内取最低旧类准确率再宏平均；`F_old,pool`是先汇总同slice逐旧类整数计数再取最低类。两者不得替代。`H`先逐row计算再宏平均。

## 1.覆盖与证据级别

|方法|实际覆盖|证据级别|结论|
|---|---|---|---|
|D62 cross-fitted Fisher row splice|125/125 jobs；375场景；5receiver×5seed×5slice|完整125稳定性screen|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|D92 registration-balanced|125/125 jobs；375场景；8/8 shards|完整125稳定性screen|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|SVRN-qKNN-BCRR/r4.2|125/125 jobs；375场景；250 prediction+COMMIT；125 score/receipt|完整125稳定性screen|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|D91|3场景×5 outer folds×7候选；选中方法15 outer rows|单development cell|与同15行D62逐值相同；非125证据|
|D104 ANGQ|计划source-held；实际运行0|实现和机制诊断|`NO_PERFORMANCE_RESULT`|

完整125的slice均为`K10/new5、K10/new10、K10/new20、K5/new20、K1/new20`。

## 2.完整125性能

### D62

|slice|B_old|A_old|F_old,row|F_old,pool|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|---:|
|K10/new5|86.022|76.333|50.467|54.933|73.573|74.601|9.689|
|K10/new10|86.022|71.533|42.267|46.533|66.747|68.845|14.489|
|K10/new20|86.022|68.678|37.933|43.200|68.780|68.563|17.344|
|K5/new20|81.322|61.389|30.867|37.333|59.283|60.025|19.933|
|K1/new20|68.144|44.033|14.200|22.333|27.150|33.410|24.111|

### D92

|slice|B_old|A_old|F_old,row|F_old,pool|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|---:|
|K10/new5|86.111|76.189|49.800|54.467|74.133|74.803|9.922|
|K10/new10|86.111|72.533|44.200|48.267|66.353|69.106|13.578|
|K10/new20|86.111|71.333|42.667|47.333|68.150|69.555|14.778|
|K5/new20|81.267|63.711|33.200|40.933|58.883|60.955|17.556|
|K1/new20|68.144|44.033|14.200|22.333|27.150|33.410|24.111|

### SVRN-qKNN-BCRR/r4.2

|slice|B_old|A_old|F_old,row|F_old,pool|N|H|Forget|
|---|---:|---:|---:|---|---:|---:|---:|
|K10/new5|75.356|52.256|15.533|`UNKNOWN`|41.773|45.277|23.100|
|K10/new10|75.356|46.133|11.667|`UNKNOWN`|28.153|34.364|29.222|
|K10/new20|75.356|42.778|9.133|`UNKNOWN`|17.297|24.396|32.578|
|K5/new20|73.367|41.589|10.800|`UNKNOWN`|15.420|22.150|31.778|
|K1/new20|66.078|32.411|8.933|`UNKNOWN`|14.673|20.066|33.667|

SVRN现有汇总没有足够的同slice逐旧类pool整数计数，不能用`F_old,row`补写`F_old,pool`。

## 3.非125结果

|方法/范围|B_old|A_old|F_old,row|F_old,pool|N|H|Forget|
|---|---:|---:|---:|---:|---:|---:|---:|
|D91选中INT8，单开发cell的15 outer rows|92.778|82.222|50.000|53.333|84.667|82.624|10.556|
|同15行D62|92.778|82.222|50.000|53.333|84.667|82.624|10.556|
|D104|`UNKNOWN`|`UNKNOWN`|`UNKNOWN`|`UNKNOWN`|`UNKNOWN`|`UNKNOWN`|`UNKNOWN`|

D91的高绝对值来自单receiver/seed开发cell，预测hash变化为0/15，不能与完整125均值排名。D104没有运行性能实验。

## 4.严格matched差值

### D92−D62

|slice|ΔB_old|ΔA_old|ΔF_old,row|ΔN|ΔH|ΔForget|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|+0.089|-0.144|-0.667|+0.560|+0.202|+0.233|
|K10/new10|+0.089|+1.000|+1.933|-0.393|+0.261|-0.911|
|K10/new20|+0.089|+2.656|+4.733|-0.630|+0.992|-2.567|
|K5/new20|-0.056|+2.322|+2.333|-0.400|+0.929|-2.378|
|K1/new20|0|0|0|0|0|0|

D92的正信号集中在K5/K10旧类保存与floor；K10/new10、K10/new20和K5/new20的新类准确率均下降，K1完全无作用。

### SVRN−D62，全125 paired

|指标|平均差|95%CI|SVRN胜/负/平|
|---|---:|---|---:|
|B_old|-8.404|[-9.881,-6.928]|18/105/2|
|A_old|-21.360|[-23.185,-19.535]|2/123/0|
|F_old,row|-23.933|[-27.137,-20.730]|9/114/2|
|N|-35.643|[-38.411,-32.876]|0/125/0|
|H|-31.838|[-33.937,-29.739]|0/125/0|
|Forget|+12.956|[+11.849,+14.062]|2/123/0|

SVRN在125/125个matched row的`N`和`H`均低于D62。本结论只否定冻结的r4.2实例，不能外推到整个SVRN、qKNN或BCRR方法族。

## 5.功能激活

|方法|参数/状态激活|决策激活|解释|
|---|---|---|---|
|D62 before|171/375=45.6%；接纳387类行|未单独汇总argmax变化|K1为0/75|
|D62 after|24/375=6.4%；接纳40类行|未单独汇总argmax变化|K10/new5为17/75，K10/new10为5/75，其余K≥5各1/75，K1为0/75|
|D91|INT8为12/15非零共识因子|0/15预测hash变化|参数非零不等于分类功能|
|D92|K1为0/75|K5/K10精确激活率`UNKNOWN`|不得从性能差值倒推激活率|
|SVRN/r4.2|`UNKNOWN`|`UNKNOWN`|sentinel命中不是方法激活率|
|D104|8400个support几何中7575改善、825相同|无性能决策|只属于支持侧机制诊断|

## 6.方法边界

|方法|合法结论|禁止结论|
|---|---|---|
|D62|合法125完成；冻结Fisher row splice几乎无稳定收益|不能判断Phase1聚合ground知识是否有效；不能宣称K1或晋升|
|D91|共识参数、资源及量化probe闭合|不能称125、正式Phase2性能或优于D62|
|D92|K5/K10大注册规模旧类遗忘与row floor改善|不能称全面改善；K1无效且新类存在代价|
|SVRN/r4.2|冻结r4.2完整125显著失败|不能否定整个方法族|
|D104|代码和支持侧几何机制存在正信号|不能宣称source-held、Target、DA或分类性能|

## 7.证据索引与告警

- D62报告：`E:\type10-7\automation_reports\CV-SincNet\d62_comprehensive_125_20260720\report.md`，SHA256=`58481283e94aaec76ca5b55941039a998d50713f4206b751d1aaa4f7525dc87e`；summary SHA256=`655fc46261cd8756b83b1765d50c13f79822f91e15d067a19d63ac0bd727e479`。
- D91报告：`E:\type10-7\automation_reports\CV-SincNet\d91_crossfit_consensus_sigma_margin_20260720\report.md`，SHA256=`800207459cf2bb5b191369821f2541b95f8dc6c60ccc2eafcb7e11a96a1adb6b`。
- D92报告：`E:\type10-7\code\snapshots\d92_125wt\automation_reports\CV-SincNet\d92_registration_balanced_125_20260720\report.md`，SHA256=`037166041f8342611d876ace86beb77b3f1836994af654df8afdf2db604f8275`。
- SVRN报告：`E:\type10-7\automation_reports\CV-SincNet\svrn_qknn_bcrr_125_r4_retry2_20260724\report.md`，SHA256=`4f73413c3e5f112fd490ae64e22d1cf927853dfbc49993d3628fc987f1b4e215`；paired summary SHA256=`6c813b43a3e0b9fa2044a320b32a7f38679a1d458c39a6b9dde27eda3d6f1a64`。
- D104 source-held报告：`E:\type10-7\automation_reports\CV-SincNet\d104_r1_angq_sourceheld_20260731_r1\report.md`，SHA256=`9366ff62d1e0c02a16c6bb65aca3d0596000603f71f61f536c58f922ad04b7a6`。

D92报告登记的summary SHA256为`71bba2c9c8ae8fb3731c508438ce6db01d95d2b3fd5a00208ce8ca8ec54f5de9`，当前本地summary实时SHA256为`71bba2c9772b14df7d786253ab4c6ad3dd93b6435c9237c01dff1fcc1b1116e6`。原因目前`UNKNOWN`。本报告数值已经从当前row/per-TX artifact独立复算，但在正式发布D92证据前必须闭合这一hash差异。

## 8.D105结果闭包

D105的最小原子表为`job×scenario×arm`。G2必须闭合`25 jobs×3 scenes×4 arms=300`条prediction、COMMIT、score和receipt。必须保留版本、protocol/capsule/split/method lock、receiver/seed/K/new_count/scenario、support/query physical-ID root、DA/HEAD state hash、旧/新逐类整数计数、`B_old/A_old/F_old,row/F_old,pool/N/H/Forget`、参数/logit/argmax激活、wrong-to-correct/correct-to-wrong、资源与artifact hash。

四臂效应按同一row计算：

```text
Delta_DA_base=Y(M_DA)-Y(M0)
Delta_DA_head=Y(M_JOINT)-Y(M_HEAD)
E_DA=0.5*(Delta_DA_base+Delta_DA_head)
Delta_HEAD_base=Y(M_HEAD)-Y(M0)
Delta_HEAD_DA=Y(M_JOINT)-Y(M_DA)
E_HEAD=0.5*(Delta_HEAD_base+Delta_HEAD_DA)
I_syn=Y(M_JOINT)-Y(M_DA)-Y(M_HEAD)+Y(M0)
```

缺失、重复、hash不闭合或系统性零prediction均先判技术失败，不读取性能。G2单seed通过只记`TARGET25_SCREEN_PASS`；没有同method lock的fresh confirm seed25，不能宣称`PROMOTABLE`。
