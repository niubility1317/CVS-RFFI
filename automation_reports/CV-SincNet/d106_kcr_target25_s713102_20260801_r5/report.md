# D106-KCR/r5完整Target25实验报告

状态：`LOCAL_VERIFIED / RELEASE_PENDING`

## 登记与目标

|字段|值|
|---|---|
|experiment ID|`d106_kcr_target25_s713102_20260801_r5`|
|日期/operator|2026-08-01；主agent负责方法、数据和结果分析，唯一Terra Max runner负责N607运行|
|目标|完成D106-KCR的完整Target25矩阵，并与D62、D91、D92、SVRN-qKNN-BCRR全面比较|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|版本管理|仅本地Git提交；不push、不上传GitHub|

## 方法、矩阵和输入

路由固定为`K1→M_DA`、`K5→M0`、`K10→M_HEAD`。每个state先计算`M0、M_DA、M_HEAD、M_JOINT`四臂，再派生`ROUTED`；路由不读取query truth、role、预测分数或receiver身份。

|维度|冻结值|
|---|---|
|receiver/seed|`20-1,3-19,7-14,7-7,8-8`；`713102`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|完整覆盖|25个outer jobs、75个scene rows、300个arm rows、600个prediction surfaces、125个评分行|
|matched K5|复用同receiver/seed/new20的D92 K10封包，只物化`rank<5`，因此K5⊂K10且query逐值相同|

## 精简发布gate与修复

只保留三个发布条件：真实封包可运行；query truth/role不进入预测；新run不可覆盖。无truth smoke通过即detach完整prediction，不再等待额外review。性能不得触发提前停止。

|项目|证据|
|---|---|
|runtime身份修复|`8b1166e1ba4648204571612543319cf7a64dd1e3`；D92 package runtime与D106/RDCE runtime分别验证、分别留痕，不再错误要求不同文件字节SHA相等|
|matched K5修复|`ba78d723da5ca87d32d9715bcd2dbcc28280512a`|
|payload修复|`0d3ebcfd56cb484cabadc026678cc50b73ff67f7`|
|本地验证|相关4文件`py_compile`通过；183项D106聚焦回归通过；`git diff --check`通过；工作树clean|
|保留的fail-closed边界|support/query package runtime必须一致且为合法SHA；D106 runtime必须为合法SHA；checkpoint、RDCE wire、method lock、seal、query隔离均不变|

## N607预登记

|字段|值|
|---|---|
|Python/GPU|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`cuda:0`，launch前记录占用|
|checkpoint及SHA|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D92 source/matrix SHA|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`；`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|RDCE wire及SHA|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`；`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r5`|
|CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r5/source`|
|日志/PID/输出|`logs/{prepare,smoke,predict}.log`；`control/predict.{pid,exit}`；`prepared/`、`smoke/`、`predictions/`|

## 性能目标

|slice|目标|
|---|---|
|K10/new5、new10、new20|`A_old≥92%`、`F_old≥85%`；`N≥92/90/86%`|
|K5/new20|相对matched K10/new20的`A/F/N/H`下降均≤5pp|
|K1/new20|相对同row D92：`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确严格增加|

## 执行、停止和分析

prepare后执行真实checkpoint、真实D92封包、`cuda:0`、无truth单state smoke；通过即发布完整prediction。只有P0协议/安全错误，或至少两个不同row在prediction前出现同一确定性异常指纹时停止；不得按accuracy、H、floor或遗忘停止。停止时只处理已绑定本run的PID树并保留artifact。

完成后按同一row报告receiver、slice、scene、before/after old、old floor、seen-new、H、forgetting、correct count和verdict。D62、D92、SVRN使用完整125证据；D91单列为15行development证据。
