# D106-KCR/r3完整Target25实验报告

状态：`LOCAL_VERIFIED / RELEASE_PENDING`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d106_kcr_target25_s713102_20260801_r3`|
|日期|2026-08-01|
|operator|主agent负责方法整合、数据与结果分析；唯一Terra Max runner负责N607落地与运行证据|
|目标|运行D106-KCR冻结方法的完整Target25矩阵，并与D62、D91、D92和SVRN-qKNN-BCRR作证据边界一致的全面比较|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|版本管理|仅本地Git提交；不push、不上传GitHub|

## 冻结方法与矩阵

D106-KCR仅按K值选择已完整计算的臂：`K1→M_DA`、`K5→M0`、`K10→M_HEAD`。路由不读取query truth、role、预测分数或receiver身份；每个state仍先完整计算`M0、M_DA、M_HEAD、M_JOINT`四臂，再派生`ROUTED`。

|维度|取值|
|---|---|
|receiver|`20-1,3-19,7-14,7-7,8-8`|
|seed|`713102`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|coverage|25个outer jobs、75个scene rows、300个arm rows、600个before/after prediction surfaces、125个同row评分行|

## 精简发布gate

发布前只保留三个硬条件：代码在真实D92封包上可运行；query truth/role不能进入预测；新run路径不可覆盖。矩阵、方法和资产SHA已冻结，不再重复设计评审、数据重验或材料审查。无truth smoke通过后立即detach完整prediction；性能值不得触发提前停止。

|项目|证据|
|---|---|
|payload修复commit|`0d3ebcfd6fbc70facf4d9630f1114fcc368456fa`|
|修复内容|按D92 loader完成`manifest_json`验证并移除后的真实payload契约读取support/query；额外truth/role字段fail closed|
|本地验证|`py_compile`通过；176项聚焦测试全部通过；`git diff --check`通过|
|方法/矩阵变化|无；仅修复真实payload适配|

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
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r3`|
|source CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r3/source`|
|GPU|`cuda:0`；launch前记录实时占用|
|日志/PID/exit|`logs/prepare.log`、`logs/smoke.log`、`logs/predict.log`；`control/predict.pid`、`control/predict.exit`|
|预期输出|`prepared/`、`smoke/`、`predictions/`；完整prediction manifest及600个surface|

## 性能目标

|slice|目标|
|---|---|
|K10/new5|`A_old≥92%`、`F_old≥85%`、`N≥92%`|
|K10/new10|`A_old≥92%`、`F_old≥85%`、`N≥90%`|
|K10/new20|`A_old≥92%`、`F_old≥85%`、`N≥86%`|
|K5/new20|相对matched K10/new20的`A/F/N/H`下降均不超过5pp|
|K1/new20|相对同row D92满足`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确数严格增加|

## 执行与停止规则

prepare后对一个固定state执行真实checkpoint、真实D92封包、`cuda:0`、无truth smoke。smoke通过即发布完整prediction。仅在P0协议/安全错误，或至少两个不同row在产生prediction前出现同一确定性异常指纹时停止本run后续dispatch；不得因accuracy、H、floor或遗忘表现差而停止。若停止，只处理已绑定到本run的PID树并保留全部artifact。

完成后按同一candidate/run row报告receiver、slice、scene、before old、after old、old floor、new、H、forgetting、correct count和verdict。D62、D92、SVRN使用完整125证据；D91明确标注仅15行development证据。
