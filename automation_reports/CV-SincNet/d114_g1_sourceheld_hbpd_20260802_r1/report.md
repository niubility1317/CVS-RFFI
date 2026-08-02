# D114-HBPD-qKNN source-held G1报告

状态：`PREREGISTERED / LOCAL_VERIFIED / PREDICTIONS_NOT_STARTED / NO_PERFORMANCE_RESULT`

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

|范围|arm|old BA|seen-new|H|old floor|correct/query|裁决|
|---|---|---:|---:|---:|---:|---:|---|
|待完成|`M0`|待完成|待完成|待完成|待完成|待完成|待完成|
|待完成|`M_DA`|待完成|待完成|待完成|待完成|待完成|待完成|
|待完成|`M_HEAD`|待完成|待完成|待完成|待完成|待完成|待完成|
|待完成|`M_JOINT`|待完成|待完成|待完成|待完成|待完成|待完成|

结果完成后补充全部同行效应、negative tail、资源、异常、与D112同split比较及最终关闭／晋级裁决。
