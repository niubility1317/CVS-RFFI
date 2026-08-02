# D114-HBPD-qKNN source-held G1报告

状态：`ARTIFACTS_COMPLETE / ANALYZED / HBPD_DA_NEGATIVE / HEAD_POSITIVE / D114_CLOSED`

## 1.身份与目标

|字段|值|
|---|---|
|run ID|`d114_g1_sourceheld_hbpd_20260802_r1`|
|日期|2026-08-02|
|operator|主agent负责实现整合、预测发布、数据与结果分析；独立Terra Max负责P0/P1复核|
|目标|在已封存且独立于D114开发的source-held split上，判断HBPD在base与ground head存在时是否都有独立正收益|
|证据级别|source-held G1，不是Target Phase2，不与D62/D92 Target125数值混排|

假设：M0经验带宽把K-shot采样离散度当作全部预测不确定性；D114用sealed旧类条件方差与support离散度形成HBPD带宽，可能降低高噪类过尖密度和低噪类过宽密度，同时保留D112 ground单位质量head已证实的独立正收益。

## 2.冻结四臂与裁决

|臂|support密度|ground head|因果含义|
|---|---|---|---|
|`M0`|原经验带宽Student-t qKNN|无|共同基线|
|`M_DA`|D114 HBPD带宽|无|`DA_AT_BASE`|
|`M_HEAD`|原经验带宽|D112固定ground单位质量head|`HEAD_AT_BASE`|
|`M_JOINT`|D114 HBPD带宽|同一个D112 head与同一个rho，anchor使用同一HBPD带宽|`DA_AT_HEAD`|

同时报告`JOINT_VS_M0`与`FACTORIAL_INTERACTION=(M_JOINT-M_HEAD)-(M_DA-M0)`。禁止按结果修改方差、pseudo-degree、rho、核参数或臂定义。

晋级要求：`DA_AT_BASE`与`DA_AT_HEAD`必须都有独立正收益，并共同保护old BA、seen-new、H、old floor和negative tail；K1若总正确数、H或old floor系统下降则关闭。任一DA简单效应没有正收益即关闭HBPD，不以`M_HEAD`既有收益保留DA，不补seed、不跑125。

## 3.输入、实现与验证

- package root：`E:\type10-7\automation_reports\CV-SincNet\d112_g1_sourceheld_seam_20260802_r3\artifacts\packages`；复用既有21个package、63行scoring roles和独立truth seal，不重建split、不重验数据。
- package manifest SHA256=`e15de8783dd70da1d37f826bf844dbf8651f33c6229d67fdb046bc208f3ef955`；上游source-held archive SHA256=`f2ceae1b47f84027f21c561bd58f50cc9df5c511e4b8d110e04e8062db6bee41`。
- scorer-only truth seal SHA256=`279214e49d0dc95e9b8971002f667442c352db858daed9a9a25dab2cce4ba9f8`；truth SHA256=`8280bd4a57c094a8101360ae4183678ba1084999ea6248f5ca1179f2194da82d`。
- D106 tap SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 新增：`code/cvsrffi/stage2_d114_hbpd_g1.py`、`code/scripts/run_d114_hbpd_g1_sourceheld.py`、`tests/test_stage2_d114_hbpd_g1.py`。
- `ssr-gpu`下三个新增文件`py_compile`通过；D114 G1、D114 core和D112 head相关31项通过。
- 独立复审首轮`P0=0/P1=1`：缺factorial interaction；最小修复后增量复审`P0=0/P1=0/GO`，新增测试6项通过。

## 4.运行合同

|字段|值|
|---|---|
|CWD|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`|
|环境|`ssr-gpu`；本地NumPy闭式推理，无GPU训练|
|prediction output|`E:\type10-7\automation_reports\CV-SincNet\d114_g1_sourceheld_hbpd_20260802_r1\artifacts\predictions`，启动前ABSENT|
|truth open event|`...\artifacts\scoring\truth_open_event.json`，score前ABSENT|
|score outputs|`held_scores.pairwise.json`与最终`held_scores.json`，score前均ABSENT|

预测命令只读package、tap、receipt与support/query feature，没有truth参数。完整63行×4臂=252个prediction单元和63个唯一receipt封存后，才运行独立score命令打开truth。任何prediction覆盖、receipt、schema或truth-seal失败均停止且不评分。

## 5.实际结果

完整63行、252个prediction单元和63个唯一receipt在truth前封存；predict阶段`query_truth_access=false`、`target_access=false`、`query_state_updates=0`。独立score随后打开既有truth seal。

|artifact|SHA256／覆盖|
|---|---|
|prediction manifest|`d18c8662aaaea764d00209297c616a44ab52ca62e895a0c3cb94b39724b733b2`；63行／252单元|
|truth-open event|`7d581d8a9422d1f008d0554e001202ecd7eb66611ab71810ada6b2018ec686da`|
|pairwise base score|`18383509106b8eba322f95e47947e13b02851a6c6e7e4c46ebb887525a29cb9d`|
|final score＋factorial interaction|`d661e2f6afe37cd92e3db52f6503e6e8618dd4e5ab1fa68cd9e4947bb0116bd4`；score receipt=`97c714d...d73236`|

### 5.1全类一般行

|K|arm|balanced accuracy|mean row floor|correct/query|对照效应|
|---:|---|---:|---:|---:|---|
|1|`M0`|84.0388%|57.6720%|953/1134|基线|
|1|`M_DA`|80.3351%|34.9206%|911/1134|DA_AT_BASE：`-3.7037/-22.7513pp`，少42个正确|
|1|`M_HEAD`|85.3616%|62.4339%|968/1134|HEAD_AT_BASE：`+1.3228/+4.7619pp`|
|1|`M_JOINT`|82.8042%|43.3862%|939/1134|DA_AT_HEAD：`-2.5573/-19.0476pp`，少29个正确|
|5|`M0`|84.9896%|60.8696%|821/966|基线|
|5|`M_DA`|74.1201%|16.1491%|716/966|DA_AT_BASE：`-10.8696/-44.7205pp`，少105个正确|
|5|`M_HEAD`|85.6108%|60.8696%|827/966|HEAD_AT_BASE：`+0.6211/0.0000pp`|
|5|`M_JOINT`|74.1201%|16.1491%|716/966|DA_AT_HEAD：`-11.4907/-44.7205pp`，少111个正确|
|10|`M0`|84.3915%|54.7619%|638/756|基线|
|10|`M_DA`|76.5873%|36.5079%|579/756|DA_AT_BASE：`-7.8042/-18.2540pp`，少59个正确|
|10|`M_HEAD`|84.3915%|54.7619%|638/756|HEAD_AT_BASE持平|
|10|`M_JOINT`|76.5873%|35.7143%|579/756|DA_AT_HEAD：`-7.8042/-19.0476pp`，少59个正确|

21个一般行合并：M0的BA／floor为84.4733%／57.7678%，`M_DA`为77.0142%／29.1925%，即`DA_AT_BASE=-7.4592/-28.5753pp`，正确数从2412降到2206；`M_HEAD`为85.1213%／59.3551%，`M_JOINT`为77.8372%／31.7499%，即`DA_AT_HEAD=-7.2841/-27.6052pp`，正确数从2433降到2234。DA_AT_BASE在BA上正／零／负=`2/1/18`，floor=`1/3/17`；DA_AT_HEAD在BA上=`4/0/17`，floor=`0/3/18`。

### 5.2K1登记行

|arm|old BA|seen-new|H old/new|old floor|correct/query|相对基线因素|
|---|---:|---:|---:|---:|---:|---|
|`M0`|84.0388%|84.0388%|82.3063%|59.4356%|5718/6804|基线|
|`M_DA`|80.3351%|80.3351%|76.1119%|40.2998%|5466/6804|DA_AT_BASE：`-3.7037/-3.7037/-6.1944/-19.1358pp`，少252个正确|
|`M_HEAD`|85.3616%|85.3616%|84.2799%|64.0212%|5808/6804|HEAD_AT_BASE：`+1.3228/+1.3228/+1.9736/+4.5855pp`|
|`M_JOINT`|82.8042%|82.8042%|79.8424%|48.3245%|5634/6804|DA_AT_HEAD：`-2.5573/-2.5573/-4.4375/-15.6966pp`，比head少174个正确|

42个登记行中，DA_AT_BASE的old BA／seen-new／H／old floor正／零／负分别为`9/2/31`、`7/25/10`、`6/0/36`、`6/5/31`；DA_AT_HEAD分别为`7/6/29`、`8/25/9`、`7/4/31`、`2/9/31`。factorial interaction均值为old BA`+1.1464pp`、seen-new`+1.1464pp`、H`+1.7569pp`、old floor`+3.4392pp`，只说明ground head部分缓冲HBPD负迁移；不能把缓冲写成HBPD正收益，因为两个DA简单效应仍大幅为负。

### 5.3功能、资源与最终裁决

- 一般行prediction变化：DA_AT_BASE=307、HEAD_AT_BASE=52、DA_AT_HEAD=308、JOINT_VS_M0=308；HBPD不是零功能或回退，而是明确改变大量错误边界。
- D114 HBPD每row持久数值态192B；D112 ground head为4308B；query依赖态0B。HBPD不增加support-query点积数量，joint只保留ground head每query最多960MAC。
- `M_HEAD`逐值复现D112同split正收益，证明四臂和scorer对照面没有漂移。HBPD对base与head两种背景均损害BA、H、floor和总正确数，且K5最严重，符合“预测带宽峰值惩罚使弱类失去密度质量”的证伪情形。

最终裁决：`HBPD_DA_NEGATIVE / CLOSE_D114`。不调`sigma_prior`、pseudo-degree、带宽系数或rho，不补seed，不运行Target25或125。存在正收益的版本仍是`M_HEAD_GROUND`，它是分类head，不是D114域适应收益。下一方法必须更换生成机制，不得把HBPD改成浓度／温度扫描继续实验。
