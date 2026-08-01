# D106-KCR/r1完整Target25实验报告

状态：`LANDED_PREPARE_CRLF_ARCHIVE_FAILURE / NO_PERFORMANCE_RESULT`

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
|production runner|初始提交`30d0eead`；D92 direct-seal发布修复`7531be61`；跨scene物理ID隔离修复`69b5679b`|
|本地验证|inputs/runner/evaluator/router/matrix统一窄回归175项通过；`py_compile`与`git diff --check`通过|
|独立代码审查|最终`GO / P0=0 / P1=0`；raw-path跨scene复用反例正确失败关闭|
|真实checkpoint无truth smoke|未执行；`prepare`先因CRLF Git archive触发canonical route-lock失败关闭|
|发布修复结果|已删除不存在的D105 formal-policy和外部split-locator依赖；直接从D92 package manifest、detached seal和真实payload派生物理ID与split身份；不改方法数学|

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
|Git source commit|`954c1df0ce5dddbe5a9641c4aa01b09e655f2ed6`|
|失败archive SHA256|`3a8191dac1001caecdf057239fedc1d742b72bf00fd2c5a0f461ea6f01d86ca2`|
|归档内lock SHA256|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`0a1745219cc1bd998928d3eb1375c401c7d3f0870a0cd2fd71dd891e4889f83e`；KCR=`ed4b76894d861b385de26fba3fc2a967ccd8eb85cfc0a5ba25561b7e1c253b6c`|
|发布结果|`prepare`失败关闭；未smoke、未detach、未产生prediction|

## 预启动技术失败证据

Windows当前Git导出配置把Git blob末尾LF转换为CRLF。归档commit仍为`954c1df0`，但RCMR与KCR字节SHA发生变化；KCR原始字节以CRLF结尾，触发`D106Target25InputError: KCR route-lock canonical schema/route drift`。失败发生在`prepare`，日志为`logs/prepare.log`；无smoke、无预测进程、无prediction或性能结果。连接结束后本地`ssh.exe=0`且目标TCP22连接为0。r1不得覆盖、恢复或重标为性能实验。

单次发布修复不改loader和方法，仅以禁用EOL转换的原生`git archive`重新导出同一commit，并使用新run ID`d106_kcr_target25_s713102_20260801_r2`。

## 健康停止规则

仅在P0协议/安全错误，或至少两个不同row在产生prediction前出现同一确定性异常指纹时停止本run的后续dispatch；不得因accuracy、H、floor或遗忘表现差而停止。停止时仅终止已证明属于本run的进程树，保留全部artifact，并标记`NO_PERFORMANCE_RESULT`。

## 完成后分析要求

结果必须按同一candidate/run row同时报告receiver、slice、scene、before old、after old、old floor、new、H、forgetting、correct count和verdict。主表与D62、D92、SVRN完整125同协议证据配对；D91仅作为15行development边界，不能冒充完整Target25。
