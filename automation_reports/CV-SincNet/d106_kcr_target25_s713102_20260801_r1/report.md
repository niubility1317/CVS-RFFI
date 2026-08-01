# D106-KCR/r1完整Target25实验报告

状态：`LOCAL_RELEASE_REPAIR_IN_PROGRESS / TARGET25_NOT_STARTED / NO_D106_TARGET_PERFORMANCE_RESULT`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d106_kcr_target25_s713102_20260801_r1`|
|日期|2026-08-01|
|operator|主agent负责协议整合、数据与结果分析；Terra Max负责功能实现；独立Sol High负责P0/P1审查；唯一Terra Max runner负责N607落地与运行证据|
|目标|在D62、D91、D92和SVRN-qKNN-BCRR基线上验证D106-KCR/r1的完整Target25真实性能|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|GitHub|不push、不上传；仅本地Git版本化|

## 假设与冻结方法

G1 source-held开发证据表明单一全局联合臂不能稳定保持旧类floor。D106-KCR/r1仅按预先冻结的K值选择已经完整计算的臂：`K1→M_DA`、`K5→M0`、`K10→M_HEAD`。路由不读取query truth、role、预测分数或接收机身份，也不裁剪四臂执行。

## 冻结矩阵

|维度|取值|
|---|---|
|receiver|`20-1,3-19,7-14,7-7,8-8`|
|seed|`713102`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|arm|`M0,M_DA,M_HEAD,M_JOINT`，完成后派生`ROUTED`|
|coverage|25个outer jobs、75个scene rows、300个arm rows、600个before/after prediction surfaces、125个同row评分行|

技术成功要求为25/75/300/600全闭合、四臂全部先完成、truth-open在完整prediction后、query fit/update为0、输出不可覆盖。性能不得触发提前停止。

## 性能目标

|slice|目标|
|---|---|
|K10/new5|`A_old≥92%`、`F_old≥85%`、`N≥92%`|
|K10/new10|`A_old≥92%`、`F_old≥85%`、`N≥90%`|
|K10/new20|`A_old≥92%`、`F_old≥85%`、`N≥86%`|
|K5/new20|相对matched K10/new20的`A/F/N/H`下降均不超过5pp|
|K1/new20|相对同row D92满足`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确数严格增加|

## 当前实现与发布门

|项目|状态|
|---|---|
|K条件路由|提交`d4b72a6b`，route lock SHA256=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
|production runner|提交`30d0eead`；主agent统一回归182项通过|
|独立代码审查|`GO / P0=0 / P1=0`；独立窄测83项通过|
|真实checkpoint无truth smoke|待D92现有sealed package直接入口完成后在N607预启动阶段执行并保存receipt|
|当前唯一修复|删除不存在的D105 formal-policy和外部split-locator依赖，直接从D92 package manifest与detached seal机械投影；不改方法数学|

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
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r1`|
|GPU|预检显示GPU0–7均空闲；正式launch前由唯一runner重新记录|

精确Git提交、源码包SHA、sync映射、服务器命令、PID、日志和输出SHA将在一次发布修复完成后、detach前补入本报告。运行只允许一个owner和一次不可覆盖detach。

## 健康停止规则

仅在P0协议/安全错误，或至少两个不同row在产生prediction前出现同一确定性异常指纹时停止本run的后续dispatch；不得因accuracy、H、floor或遗忘表现差而停止。停止时仅终止已证明属于本run的进程树，保留全部artifact，并标记`NO_PERFORMANCE_RESULT`。

## 完成后分析要求

结果必须按同一candidate/run row同时报告receiver、slice、scene、before old、after old、old floor、new、H、forgetting、correct count和verdict。主表与D62、D92、SVRN完整125同协议证据配对；D91仅作为15行development边界，不能冒充完整Target25。
