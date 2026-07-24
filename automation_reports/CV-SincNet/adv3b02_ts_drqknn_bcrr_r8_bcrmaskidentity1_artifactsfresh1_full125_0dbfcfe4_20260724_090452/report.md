# ADV3B02/r8-bcrmaskidentity1 artifactsfresh1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r8_bcrmaskidentity1_artifactsfresh1_full125_0dbfcfe4_20260724_090452`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r8-bcrmaskidentity1`
- 科学Git commit：`0dbfcfe43c178068d91d6bbd07d22bd14bd959cf`
- 发布控制基线：`3ee597cfdab3ac0cf43ede0ed251122a3d53ac12`
- 状态：`ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`

本run只修复上一run由唯一runner提前创建`<run>/artifacts`的启动布局错误；科学commit、输入、方法、参数和完整125矩阵逐字节不变。正式child前必须最后证明`<run>/artifacts=ABSENT`，由matrix launcher首次创建。完整合同见根报告：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r8_bcrmaskidentity1_artifactsfresh1_full125_0dbfcfe4_20260724_090452\report.md`。

|字段|当前值|
|---|---|
|Git commit|`0dbfcfe43c178068d91d6bbd07d22bd14bd959cf`|
|run ID|`adv3b02_ts_drqknn_bcrr_r8_bcrmaskidentity1_artifactsfresh1_full125_0dbfcfe4_20260724_090452`|
|remote PID/exit|`1697137(wrapper),1697151(matrix child) / 0`|
|prediction/score|`1000/1000 / 1500/1500`|
|archive/coverage/parity|`MINIMAL_PERFORMANCE_ARCHIVE_GENERATED / 375/375 SCENE ARTIFACTS（无独立命名coverage） / NO_SEPARATE_ARTIFACT`|
|性能裁决|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|

## 完整125性能终态

独立只读解析闭合125/125份row receipt、1000/1000份prediction、1500/1500条logical score、375/375个scene slice，125个return code均为0，指标重算0处漂移，`query_rows_used_for_fit=0`。数值均为百分比，差值为百分点。

|arm|old-before|old-after|注册内old变化|seen-new|H|BA|floor|min-old|min-new|forgetting|old→new|new→old|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|M0|72.5956|43.0222|−29.5733|23.4860|28.9453|29.1681|2.2933|11.1733|2.6133|29.5733|44.1022|29.2280|
|M_DA|72.6556|43.0600|−29.5956|23.4400|28.9181|29.1673|2.2667|11.2000|2.5733|29.5956|44.1267|29.2407|
|M_OTHER|73.1244|43.0222|−30.1022|23.4860|28.9453|29.1681|2.2933|11.1733|2.6133|30.1022|44.1022|29.2280|
|M_JOINT|73.1044|43.0600|−30.0444|23.4400|28.9181|29.1673|2.2667|11.2000|2.5733|30.0444|44.1267|29.2407|

DA old-after gain=`+0.0378pp`，但seen-new=`−0.0460pp`、H=`−0.0272pp`、floor=`−0.0267pp`、min-new=`−0.0400pp`、forgetting=`+0.0222pp`。M_DA注册后改变2308/157500个argmax，产生old`+17`、new`−4`、合计`+13`个净正确决策；K5为old`−6`/new`+14`，K10为old`+23`/new`−18`，不满足old/new各自非负。

注册后`M_OTHER=M0`和`M_JOINT=M_DA`都覆盖375/375个slice、157500/157500个query。OTHER独立收益为0，JOINT相对DA增益为0。`I_syn`在375/375个slice逐值精确为0，正协同=`0/375`，正scene均值=`0/3`；125-row配对bootstrap 50000次的95%CI为`[0,0]pp`。

|比较|old-after|seen-new|H|BA|floor|min-old|min-new|forgetting|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|M_DA−M0|+0.0378|−0.0460|−0.0272|−0.0008|−0.0267|+0.0267|−0.0400|+0.0222|
|M_OTHER−M0|0|0|0|0|0|0|0|+0.5289|
|M_JOINT−M_DA|0|0|0|0|0|0|0|+0.4489|

|分层|level|slice数|H(M0)|H(M_DA)|DA差值|I_syn/正协同|
|---|---|---:|---:|---:|---:|---:|
|scene|clear|125|30.4452|30.3569|−0.0883|0/0|
|scene|low-elev|125|27.8189|27.8641|+0.0453|0/0|
|scene|rain|125|28.5718|28.5334|−0.0385|0/0|
|receiver|20-1|75|28.6880|28.7555|+0.0675|0/0|
|receiver|3-19|75|20.5962|20.3555|−0.2407|0/0|
|receiver|7-14|75|30.4543|30.6180|+0.1637|0/0|
|receiver|7-7|75|32.1445|32.1056|−0.0389|0/0|
|receiver|8-8|75|32.8435|32.7560|−0.0875|0/0|
|K|1|75|19.7025|19.7025|0|0/0|
|K|5|75|21.8982|21.9412|+0.0430|0/0|
|K|10|225|34.3753|34.3156|−0.0596|0/0|
|seed|713102|75|29.8912|29.8706|−0.0207|0/0|
|seed|713103|75|28.7122|28.6566|−0.0556|0/0|
|seed|713104|75|28.6871|28.7770|+0.0899|0/0|
|seed|713105|75|28.6163|28.6579|+0.0416|0/0|
|seed|713106|75|28.8197|28.6286|−0.1911|0/0|

逐类注册后accuracy的最差old为TX`6-15`：M0/M_OTHER=`16.6933`、M_DA/M_JOINT=`16.9867`；最差new为TX`3-8`：`5.3778/5.3556`。完整26类同row表保存在根报告及500份score JSON中。

INT8 top1 agreement mean/min=`100%/100%`，large-margin flip=`0`，最坏state wire=`206394B≈201.56KiB`，峰值CUDA显存=`250062336B`。单scene-state total latency mean/P95/max=`8544.145/39959.961/51469.003ms`。r8相对r7的state、MAC、参数和optimizer step增量均为0；minimal receipt未公开绝对MAC，故不虚构数值。

最终性能裁决：`REJECT / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。下一DA候选固定为`GRB-JP4-ADV-DRQKNN-BCRR/r1-sealed`，保留`z_id/z_dom`双qKNN和BCRR，只更换ground q4低秩模型DA机制并发布新的完整125。

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
