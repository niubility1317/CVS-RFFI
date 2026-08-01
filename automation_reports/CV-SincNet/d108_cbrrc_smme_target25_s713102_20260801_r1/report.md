# D108-CB-RRC-SMME单seed Target25实验报告

状态：`ANALYZED / TARGET25_COMPLETE / D108_WEAK_REJECTED`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d108_cbrrc_smme_target25_s713102_20260801_r1`|
|日期/operator|2026-08-01；主agent负责目标、协议和结果决策；Terra Max唯一runner负责N607落地、启动、监控、评分和回收|
|目标|用最小关键矩阵判断D108是否达到当前单seed Target25目标，避免五seed完整125的非必要计算|
|候选|`D108-CB-RRC-SMME/r1`，方法与已冻结D108完全相同|
|比较|`M0=D92`、`M_DA=CB-RRC＋D92`、`M_HEAD=D92＋SMME`、`M_JOINT=CB-RRC＋D92＋SMME`|
|证据边界|完整25-job单seed真实性能；达到目标后才考虑多seed确认|
|历史parent|`d108_cbrrc_smme_target125_20260801_r3`按用户范围缩减停止于968/3000，未开truth，永久`NO_PERFORMANCE_RESULT`|

## 假设与完整关键矩阵

D108假设CB-RRC能利用合法ground压缩知识和当前row旧类support改善接收机域偏移，SMME能在全注册类统一竞争中缓解注册后旧类／新类margin失衡；联合臂应同时提升旧类、new和floor，而不是只迁移错误。

固定矩阵：

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed=713102
slice={K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}
scene={leo_clear_weak,leo_low_elev_weak,leo_rain_weak}
arm={M0,M_DA,M_HEAD,M_JOINT}
phase={before,after}
```

闭包为25 outer、75 scene、300 arm pair、600 prediction surface。seed713102早于D108结果预登记，不是依据性能选择。不得删除任何receiver、slice或scene，不得按中间准确率停止。

## 性能目标

|切片|必须满足|
|---|---|
|K10/new5|`old_acc_after_increment≥92%`、`min_old_class_acc≥85%`、`seen_new_acc≥92%`|
|K10/new10|`old_acc_after_increment≥92%`、`min_old_class_acc≥85%`、`seen_new_acc≥90%`|
|K10/new20|`old_acc_after_increment≥92%`、`min_old_class_acc≥85%`、`seen_new_acc≥86%`|
|K5/new20|相对matched K10/new20的`A_old/F_old/N/H`下降均≤5pp|
|K1/new20|相对同row D92：`ΔH≥2pp、ΔF_old≥2pp、ΔA_old≥0、ΔN≥0`且old＋new总正确数严格增加|

结果必须按同一row同时报告receiver、slice、scene、arm、before/after old、old floor、seen-new、H、forgetting和correct count。D62、D92、SVRN只复用历史同row基线；D91仅列development证据。

## 本地版本与独立复审

|项目|值|
|---|---|
|Git分支|`codex/stage2-da25-r1`|
|实现commit|`437cb5a2fdc63db621743593fb0d0202e5cfeafe`|
|source archive|`E:\type10-7\code\snapshots\d108_cbrrc_smme_target25_s713102_20260801_r1_source_437cb5a2.tar`|
|archive SHA256|`d3f640108b796e3bc85856f6cca887f46dabe60a6bced99c6b86b74630a0e076`|
|Target25 module SHA256|`e7e107b5035f634132efc1102a6cd605629a6ac61fee9f0cfadf5f1f9b03c902`|
|CLI SHA256|`e858ee862eca26009bbfb51176ca87824ed6c29733e14ccc8242f36f597fd783`|
|独立review|`P0=0、P1=0 / RELEASE`；确认25-job笛卡尔积、D92／四臂逐值路径不变、query零fit/update/selection、8-shard不可覆盖、truth时序和异常恢复|
|本地验证|`ssr-gpu`中Target25＋D108 core／runner／truth聚焦回归24项通过；三个新文件`py_compile`通过|

根目录`E:\type10-7`不是Git仓库；本报告同时镜像到Git工作树并单独提交，不push、不上传GitHub。

## 冻结N607输入

|输入|路径／SHA256|
|---|---|
|D92 matrix manifest|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`；`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|D92 output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|ground component|`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`；manifest`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|
|method lock|`<run-root>/source/configs/stage2_d108_cbrrc_smme_r1.json`；`7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845`|

不重验`VALIDATED_ONCE`数据，不修改received-IQ、physical ID、receiver/TX、scenario、K或support/query split。

## N607路径、环境与命令

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d108_cbrrc_smme_target25_s713102_20260801_r1`
- CWD：`<run-root>/source`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 日志：`<run-root>/logs/`
- PID／exit：`<run-root>/remote_control/`
- expected：`prepared/target125_plan.json`、`prepared/target125_context.json`、`smoke/*`、8个shard manifest、`predictions/prediction_manifest.json`、`truth_catalog.json`、`score/score_manifest.json`

prepare命令：

```text
python code/scripts/run_d108_target25.py prepare --d92-matrix-manifest <D92-manifest> --d92-matrix-manifest-sha256 b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c --d92-output-root <D92-root> --checkpoint <checkpoint> --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --d108-method-lock configs/stage2_d108_cbrrc_smme_r1.json --d108-method-lock-sha256 7e8b310eeffc5e56aa39d60ef3b66c652207c3d9c1004e04d4499e6073862845 --ground-component-dir <ground-dir> --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c --output-dir <run-root>/prepared
```

smoke固定row0／scene0／GPU0，真实checkpoint、无truth；通过后立即启动8shard。每个物理GPU i使用：

```text
CUDA_VISIBLE_DEVICES=i python code/scripts/run_d108_target25.py predict-shard --plan-manifest <prepared-plan> --plan-manifest-sha256 <sha> --context-manifest <prepared-context> --context-manifest-sha256 <sha> --output-dir <run-root>/shards/shard_i --shard-index i --device cuda:0 --feature-batch-size 64
```

8shard全部退出0后严格merge和validate；prediction完整封存后才允许build-truth和score。唯一runner在实际执行时把所有占位符替换为本报告冻结路径并把完整命令、PID、GPU映射和SHA回填。

## 停止条件与成功条件

只在P0协议／安全问题、错误source／SHA／不可覆盖风险，或至少两个不同outer在prediction前产生同一确定性异常指纹时停止精确run-owned进程树。不得因accuracy、H、floor或任何中间性能值停止、选row、调参或重启。

技术完成要求：25/25 outer、75/75 scene、300/300 arm pair、600/600 prediction surface、8/8 shard、prediction manifest、150 truth surface、300 scene-arm score row、100 outer-arm聚合行全部闭合；truth在prediction封存后开启；异常为0。

性能完成要求：完整读取score和artifact，按本报告目标逐项判定。若未达目标，D108直接判弱并切换已release-ready的D109 Target25，不追加seed、不修D108阈值、不恢复完整125。

## 风险与完成后检查

- 主要性能风险：CB-RRC可能只产生小幅旧类改善；SMME可能伤害新类或floor；K1可能改变分数但没有联合收益。
- 主要执行风险：多GPU必须使用`CUDA_VISIBLE_DEVICES=i`配合统一`--device cuda:0`；严禁再次传`cuda:i`。
- 完成后回收完整prediction、score、truth-open event、shard manifests、日志、PID／exit、资源和SHA清单。
- D108 Target25现已有完整单seed真实性能；其技术闭包有效，但性能未达到本报告预登记目标，不得扩展为多seed正收益或promotion声明。

## N607执行记录

- fresh direct preflight：2026-08-01 23:50 CST通过；旧r3已知16个PID存活0；8张RTX3090均`0%/1 MiB`；`/home`余量7.4T；新run root确认`ABSENT`后首次创建。
- archive远端SHA256=`d3f640108b796e3bc85856f6cca887f46dabe60a6bced99c6b86b74630a0e076`；method lock、Target25 module、CLI SHA依次为`7e8b310e…62845`、`e7e107b5…3c902`、`e858ee86…d783`；JSON解析与Target25／truth／D108 core `py_compile`通过。
- prepare闭合`target25_seed=713102`、25 outer、600 surface，query fit／truth／update／role／selection均为false；plan SHA=`bfa492d5519eb71290ff789a92355b0a50e629c67c06905f5cbfc4c6edd0c355`，context SHA=`2f1702b7c950f0561ae1057ed2678e81683d8cf89e930205a2d7c72fab56794a`，receipt文件SHA=`0e2f6bcbc08cc1cb1b1bd3bd1d76f9bb80b4fe8f46f2eaa1df1bdce952cac0d3`。
- 真实row0／scene0 smoke通过：`D108_TARGET25_REAL_CHECKPOINT_NO_QUERY_FIT_SMOKE_PASS`；receipt／prediction文件SHA=`5a62c4cd…2e31`／`43dbcef6…c3c`。
- 8shard于`2026-08-01T15:55:00Z`一次性启动。wrapper PID=`3550838,3550841,3550845,3550849,3550853,3550857,3550860,3550864`；Python PID=`3550844,3550848,3550852,3550855,3550862,3550867,3550868,3550869`。
- 启动后逐项核验：Python环境`CUDA_VISIBLE_DEVICES=i`，CWD固定为`<run-root>/source`，cmdline固定`--shard-index i --device cuda:0 --feature-batch-size 64`，输出为`shards/shard_i`；物理GPU0—7各且仅有对应Python PID，初始显存556—558MiB，异常指纹0。

实际shard子命令为：

```text
env CUDA_VISIBLE_DEVICES=i PYTHONUNBUFFERED=1 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/run_d108_target25.py predict-shard --plan-manifest <run-root>/prepared/target125_plan.json --plan-manifest-sha256 bfa492d5519eb71290ff789a92355b0a50e629c67c06905f5cbfc4c6edd0c355 --context-manifest <run-root>/prepared/target125_context.json --context-manifest-sha256 2f1702b7c950f0561ae1057ed2678e81683d8cf89e930205a2d7c72fab56794a --output-dir <run-root>/shards/shard_i --shard-index i --device cuda:0 --feature-batch-size 64
```

## 完成、封存与回收

- prediction于`2026-08-01T17:05:03Z`闭合：S0=96，S1—S7各72，共600/600；8/8 shard manifest、8/8 exit=0、确定性异常0。
- merge与严格validate闭合25 outer、75 scene、300 arm pair、600 prediction surface；prediction manifest文件SHA256=`5dd8008a9ce33a6cac4d0584ff43431d85df407a9942961e7d6c645d9695c3f7`，内容SHA256=`38412e7a27dd065b670966e0ef30c3ff852d8c21fdc0bdaa50ce42fa92adc63a`。
- prediction manifest封存后才开启truth；truth catalog闭合150 surface，文件SHA256=`e9d2c8e6f19b66ddc51d9724d1b3f77bbb65d91bb20ef61d3049d2317e180b9e`，内容SHA256=`a5765cb1e701f33b75b718a956ff0a318ed69820ac3cb4daf3705d7fcf8dda51`。
- score闭合300 scene-arm metric row和100 outer-arm aggregate row；score manifest文件SHA256=`ec4470bc6255451802e27e749ec536e2956c20ca560a54f3bd9029c76e1ced62`，内容SHA256=`03baa9faa6ea566b0ca7e98000453eec35ce5d7763bcba9ab584f91dca6cd93b`；truth-open event文件SHA256=`89fecb5d065195173dc5d23afb6c771a3e1d1320373f6418cdd544139a488b45`。
- validate权限账本中clean/source runtime、query fit、truth、update、role、selection访问均为false；未做性能早停、row选择或重启。
- 600个prediction artifact共24137000 bytes，prediction query count=162000，registered support slot count=54120；完整回收包SHA256=`0bcc2090c71b153ebb3b7bf3991889f2ff507e306536af18ae7c566bd65ee9b7`，大小34731324 bytes。
- 本地回收目录：`E:\type10-7\automation_reports\CV-SincNet\d108_cbrrc_smme_target25_s713102_20260801_r1\artifacts\remote_r1`；包含prepared=3、smoke=2、logs=12、remote_control=19、shards=608、merged=601、truth=1、score=2个文件。
- 终审时16个已知wrapper／Python PID存活0，GPU compute process为空；每次短连接后本地`ssh.exe=0`、`ESTABLISHED TCP22=0`。
- 本地artifact QA复核SHA与600/150/300/100计数通过，并确认M_JOINT在25/25 outer上H、A-old、seen-new和注册后总正确数均低于M0；根报告与Git镜像一致。按runner边界未提交最终报告更新，Git工作树仅该报告为modified，`git diff --check`通过。

scorer的`target_thresholds_declared=false`表示通用score manifest没有内嵌本报告阈值；以下判定由runner严格读取本报告预登记目标后完成，不改变score artifact。

## 四臂总体结果

下表为25个outer row的等权均值；每个outer row内部对三个LEO weak scene做micro-average。`F-old`是注册后最差旧类准确率，`forgetting=B-old-A-old`。

|arm|B-old|A-old|F-old|seen-new|H|forgetting|相对M0的H/A/N/correct均值变化|结论|
|---|---:|---:|---:|---:|---:|---:|---|---|
|M0|82.81|66.46|37.60|59.73|62.43|16.36|基线|`D92_BASELINE`|
|M_DA|82.86|66.47|37.67|59.74|62.44|16.39|+0.01/+0.01/+0.01/+0.04|近似零效应，不promotion|
|M_HEAD|70.21|49.86|30.40|44.43|46.31|20.36|-16.12/-16.60/-15.31/-189.32|系统性负效应|
|M_JOINT|70.09|49.83|30.27|44.45|46.32|20.26|-16.11/-16.62/-15.28/-189.12|`D108_WEAK_REJECTED`|

M_JOINT相对M0在25/25 outer row上H、A-old、seen-new和注册后总正确数全部更低；H下降范围为4.23—24.40pp，总正确数每row减少61—339。M_DA只有量化噪声级变化，无法支持CB-RRC独立正收益；M_HEAD与M_JOINT的同向大幅退化把主要失败归因指向SMME路径，联合并未补偿。

## M_JOINT逐outer同row结果

`M0 A/F/N/H`与`JOINT B/A/F/N/H/forget`均来自同receiver、seed、slice、三个scene联合聚合的同一outer row；`delta H/A/N/correct`以M0为零点。完整300条scene-arm同row及100条outer-arm明细保存在`artifacts/remote_r1/score/score_manifest.json`。

|receiver|slice|M0 A/F/N/H|JOINT B/A/F/N/H/forget|post correct/query|delta H/A/N/correct|verdict|
|---|---|---|---|---:|---|---|
|20-1|K10/new5|79.44/51.67/79.33/79.39|75.56/60.28/45.00/66.00/63.01/15.28|415/660|-16.38/-19.17/-13.33/-109|`FAIL_TARGET`|
|20-1|K10/new10|77.50/50.00/72.83/75.09|75.56/56.39/38.33/58.50/57.43/19.17|554/960|-17.67/-21.11/-14.33/-162|`FAIL_TARGET`|
|20-1|K10/new20|76.11/51.67/67.00/71.27|75.56/51.67/28.33/54.17/52.89/23.89|836/1560|-18.38/-24.44/-12.83/-242|`FAIL_TARGET`|
|20-1|K5/new20|69.44/38.33/63.08/66.11|62.22/37.50/20.00/47.00/41.72/24.72|699/1560|-24.40/-31.94/-16.08/-308|`FAIL_TARGET`|
|20-1|K1/new20|42.50/23.33/27.75/33.58|62.50/27.22/16.67/21.25/23.87/35.28|353/1560|-9.71/-15.28/-6.50/-133|`FAIL_TARGET`|
|3-19|K10/new5|58.06/28.33/52.67/55.23|63.06/46.67/31.67/36.00/40.65/16.39|276/660|-14.58/-11.39/-16.67/-91|`FAIL_TARGET`|
|3-19|K10/new10|56.67/26.67/44.33/49.75|63.06/41.94/28.33/31.17/35.76/21.11|338/960|-13.99/-14.72/-13.17/-132|`FAIL_TARGET`|
|3-19|K10/new20|56.94/26.67/50.58/53.58|63.06/39.44/20.00/36.58/37.96/23.61|581/1560|-15.62/-17.50/-14.00/-231|`FAIL_TARGET`|
|3-19|K5/new20|50.28/25.00/41.42/45.42|50.83/30.56/5.00/24.42/27.14/20.28|403/1560|-18.28/-19.72/-17.00/-275|`FAIL_TARGET`|
|3-19|K1/new20|26.94/10.00/13.17/17.69|45.83/17.78/10.00/10.83/13.46/28.06|194/1560|-4.23/-9.17/-2.33/-61|`FAIL_TARGET`|
|7-14|K10/new5|72.22/48.33/84.00/77.67|70.28/53.33/41.67/60.00/56.47/16.94|372/660|-21.20/-18.89/-24.00/-140|`FAIL_TARGET`|
|7-14|K10/new10|72.50/43.33/70.67/71.57|70.28/50.28/35.00/49.67/49.97/20.00|479/960|-21.60/-22.22/-21.00/-206|`FAIL_TARGET`|
|7-14|K10/new20|70.56/40.00/71.33/70.94|70.28/50.83/18.33/55.58/53.10/19.44|850/1560|-17.84/-19.72/-15.75/-260|`FAIL_TARGET`|
|7-14|K5/new20|63.89/23.33/62.83/63.36|66.11/48.89/18.33/43.17/45.85/17.22|694/1560|-17.51/-15.00/-19.67/-290|`FAIL_TARGET`|
|7-14|K1/new20|55.00/5.00/31.67/40.19|84.17/53.06/18.33/23.00/32.09/31.11|467/1560|-8.10/-1.94/-8.67/-111|`FAIL_TARGET`|
|7-7|K10/new5|88.89/75.00/82.67/85.66|90.00/81.11/65.00/71.67/76.10/8.89|507/660|-9.57/-7.78/-11.00/-61|`FAIL_TARGET`|
|7-7|K10/new10|85.56/65.00/70.17/77.10|90.00/72.78/63.33/54.50/62.33/17.22|589/960|-14.77/-12.78/-15.67/-140|`FAIL_TARGET`|
|7-7|K10/new20|83.89/60.00/74.75/79.06|90.00/75.00/66.67/59.42/66.31/15.00|983/1560|-12.75/-8.89/-15.33/-216|`FAIL_TARGET`|
|7-7|K5/new20|78.33/55.00/67.25/72.37|87.78/58.61/41.67/52.92/55.62/29.17|846/1560|-16.75/-19.72/-14.33/-243|`FAIL_TARGET`|
|7-7|K1/new20|50.83/18.33/30.08/37.80|65.28/45.56/3.33/21.17/28.90/19.72|418/1560|-8.89/-5.28/-8.92/-126|`FAIL_TARGET`|
|8-8|K10/new5|80.28/55.00/82.67/81.45|68.89/58.33/43.33/57.00/57.66/10.56|381/660|-23.80/-21.94/-25.67/-156|`FAIL_TARGET`|
|8-8|K10/new10|74.17/41.67/74.17/74.17|68.89/51.39/38.33/49.00/50.17/17.50|479/960|-24.00/-22.78/-25.17/-233|`FAIL_TARGET`|
|8-8|K10/new20|73.33/38.33/79.50/76.29|68.89/48.61/26.67/58.67/53.17/20.28|879/1560|-23.12/-24.72/-20.83/-339|`FAIL_TARGET`|
|8-8|K5/new20|68.89/36.67/70.75/69.81|69.17/49.72/31.67/48.92/49.32/19.44|766/1560|-20.49/-19.17/-21.83/-331|`FAIL_TARGET`|
|8-8|K1/new20|49.17/3.33/28.67/36.22|55.00/38.89/1.67/20.75/27.06/16.11|389/1560|-9.16/-10.28/-7.92/-132|`FAIL_TARGET`|

## 预登记目标判定与下一步

|receiver|K10/new5|K10/new10|K10/new20|K5/new20稳健性|K1/new20相对M0|总判定|
|---|---|---|---|---|---|---|
|20-1|FAIL|FAIL|FAIL|FAIL|FAIL|`REJECT`|
|3-19|FAIL|FAIL|FAIL|FAIL|FAIL|`REJECT`|
|7-14|FAIL|FAIL|FAIL|FAIL|FAIL|`REJECT`|
|7-7|FAIL|FAIL|FAIL|FAIL|FAIL|`REJECT`|
|8-8|FAIL|FAIL|FAIL|FAIL|FAIL|`REJECT`|

五个receiver的全部五类目标均失败。特别是K1/new20相对同row M0没有任何一个receiver满足`delta H>=2pp、delta forgetting>=2pp、delta A-old>=0、delta seen-new>=0`且总正确数增加；实际五个receiver的H均下降、总正确数均减少。因此D108按预登记规则直接判弱，不追加seed、不修改D108阈值、不恢复完整125。下一实验应回到主agent执行已release-ready的D109 Target25；本runner不自行调参或发起新run。
