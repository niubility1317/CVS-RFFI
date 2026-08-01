# D106-KCR/r2完整Target25实验报告

状态：`LANDED_PREPARE_COMPLETE_SMOKE_PAYLOAD_ALLOWLIST_FAILURE / NO_PERFORMANCE_RESULT`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d106_kcr_target25_s713102_20260801_r2`|
|日期|2026-08-01|
|operator|主agent负责协议整合、数据与结果分析；唯一Terra Max runner负责N607落地与运行证据|
|目标|运行D106-KCR/r1冻结方法的完整Target25矩阵，并与D62、D91、D92和SVRN-qKNN-BCRR作同证据边界比较|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|GitHub|不push、不上传；仅本地Git版本化|

## 冻结方法与矩阵

D106-KCR/r1仅按预先冻结的K值选择已完整计算的臂：`K1→M_DA`、`K5→M0`、`K10→M_HEAD`。路由不读取query truth、role、预测分数或receiver身份，也不裁剪四臂执行。

|维度|取值|
|---|---|
|receiver|`20-1,3-19,7-14,7-7,8-8`|
|seed|`713102`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|arm|`M0,M_DA,M_HEAD,M_JOINT`，完成后派生`ROUTED`|
|coverage|25个outer jobs、75个scene rows、300个arm rows、600个before/after prediction surfaces、125个同row评分行|

技术成功要求为25/75/300/600全闭合、四臂全部先完成、truth-open在完整prediction后、query fit/update/selection为0、输出不可覆盖。性能不得触发提前停止。

## 性能目标

|slice|目标|
|---|---|
|K10/new5|`A_old≥92%`、`F_old≥85%`、`N≥92%`|
|K10/new10|`A_old≥92%`、`F_old≥85%`、`N≥90%`|
|K10/new20|`A_old≥92%`、`F_old≥85%`、`N≥86%`|
|K5/new20|相对matched K10/new20的`A/F/N/H`下降均不超过5pp|
|K1/new20|相对同row D92满足`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确数严格增加|

## 发布修复与本地证据

|项目|状态|
|---|---|
|代码commit|`954c1df0ce5dddbe5a9641c4aa01b09e655f2ed6`；跨scene物理ID隔离修复为`69b5679bb215afd464bf9cf3d97d56820af38327`|
|Git archive命令|`git -c core.autocrlf=false archive --format=tar --output=<literal file> 954c1df0ce5dddbe5a9641c4aa01b09e655f2ed6`|
|archive SHA256|`9b7deecea051def8024358df8d3e0a7af5b0d59d47b96958a408fb2395afb08b`|
|解包lock SHA256|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
|修复边界|仅禁止Git归档EOL转换；不改loader、方法数学、矩阵或测试结论|
|本地验证|归档commit反查为`954c1df0`；解包三lock SHA精确匹配；沿用已完成的175项窄回归、`py_compile`和独立`GO / P0=0 / P1=0`|

## N607预登记

|字段|值|
|---|---|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D92 source|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`|
|D92 matrix SHA256|`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|RDCE wire|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`|
|RDCE wire SHA256|`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r2`|
|source CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r2/source`|
|GPU|`cuda:0`；launch前重新记录占用|
|日志/PID/exit|`logs/prepare.log`、`logs/smoke.log`、`logs/predict.log`；`control/predict.pid`、`control/predict.exit`|
|预期输出|`prepared/`、`smoke/`、`predictions/`；完整prediction manifest及600个surface|
|远端archive校验|SHA=`9b7deecea051def8024358df8d3e0a7af5b0d59d47b96958a408fb2395afb08b`；commit=`954c1df0ce5dddbe5a9641c4aa01b09e655f2ed6`；三lock SHA精确匹配；`py_compile=PASS`|
|prepare结果|`TARGET25_D92_PACKAGES_LOCATED`；25/75/300/600闭合|
|plan/context SHA|plan=`00f26d17d371472a1271078ed643e15fb2f30534d1524911191123990ce08e84`；context=`9dff1a120a0b3843da60e1718d00855acdf27e7b4eae84737b64658d2a4f0f0b`|
|smoke结果|失败关闭；`D92 physical-ID materialization drift`，底层指纹`support payload exact allowlist drift`|
|launch结果|未detach；`smoke/`、`predictions/`、predict PID和exit均不存在；无性能结果|

## 实际执行与失败证据

`prepare`使用归档内RDCE/RCMR/KCR lock SHA`e7a198…4cc1`、`be452c…6b0c`、`a3d530…ab6a`及已登记的D92 manifest、checkpoint、RDCE wire SHA，输出`prepared/target25_plan.json`、`prepared/target25_context.json`和`prepared/prepare_receipt.json`。随后对`row_index=0,scenario_index=0,state_index=0`以`device=cuda:0,feature_batch_size=64`调用`smoke_d106_target25_prepared_state`，无truth参数。

smoke在`_prepared_inputs→_expand_raw_rows→_derived_state→_support_rows`失败。D92原生helper`load_verified_somph_predictor_bundle`通过`_materialize_iq`解析并验证`manifest_json`后将其从array映射弹出；实际support payload为7项，但`_support_rows`仍按含`manifest_json`的8项NPZ成员表做exact-set比较，因此确定性失败。query路径存在同型5项对6项不一致，但尚未执行到该处。

本地取回证据：`prepare.log`SHA=`fe98d7293c78e366ab316db773b88754f4ed1aae14ab286c8b0c5ad59324121d`；`smoke.log`SHA=`99c15b4779315af59974999900f66483ce8c80e64788d0d0c324ff0b88f7f84d`；`prepare_receipt.json`SHA=`a8e7015b79b7f258d67ec691425874a13860a92d81393b33c37f5b8ccf2cef3a`。远端日志实际文件名带尾部CR，这是PowerShell stdin换行造成的日志名问题，不影响异常内容；远端证据未修改，本地副本规范命名。检查时无目标进程，GPU0–7均为0%利用率/1MiB，`ssh.exe=0`且目标TCP22连接为0。

r2不得覆盖、恢复、重试或重标为性能实验。修复必须在本地完成且不得改变方法；通过聚焦回归和独立P0/P1复核后使用新不可覆盖run ID。

## 健康停止规则

仅在P0协议/安全错误，或至少两个不同row在产生prediction前出现同一确定性异常指纹时停止本run的后续dispatch；不得因accuracy、H、floor或遗忘表现差而停止。停止前绑定run-owned PID/CWD/cmdline，只终止本run进程树并保留全部artifact。

## 完成后分析要求

结果按同一candidate/run row同时报告receiver、slice、scene、before old、after old、old floor、new、H、forgetting、correct count和verdict。主表与D62、D92、SVRN完整125同协议证据配对；D91仅作为15行development边界，不能冒充完整Target25。
