# D106-KCR/r6完整Target25实验报告

状态：`LOCAL_VERIFIED / RELEASE_PENDING`

## 登记

|字段|值|
|---|---|
|experiment ID|`d106_kcr_target25_s713102_20260801_r6`|
|日期/operator|2026-08-01；主agent负责方法、数据和结果分析，唯一Terra Max runner负责N607运行|
|目标|完整运行D106-KCR Target25，并与D62、D91、D92、SVRN-qKNN-BCRR全面比较|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|版本|本地Git；不push、不上传GitHub|

## 冻结方法、输入和矩阵

K路由为`K1→M_DA`、`K5→M0`、`K10→M_HEAD`。每个state先计算`M0、M_DA、M_HEAD、M_JOINT`四臂，再派生`ROUTED`；路由不读取query truth、role、预测分数或receiver。

|维度|冻结值|
|---|---|
|receiver/seed|`20-1,3-19,7-14,7-7,8-8`；`713102`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|coverage|25个outer jobs、75个scene rows、300个arm rows、600个prediction surfaces、125个评分行|
|matched K5|同receiver/seed/new20的D92 K10封包，仅物化`rank<5`，保证K5⊂K10及query相同|

## 精简gate和本地证据

只保留真实封包可运行、query truth/role隔离、run不可覆盖三个条件。no-truth smoke通过即detach完整prediction，不再等待额外review；不得按性能停止。

|项目|证据|
|---|---|
|RCMR wire修复commit|`7ff4b4e7c882aed6e7fb62123b906eb388462a1c`|
|新RCMR lock SHA256|`cc5795be25622da0f060056bb931e20c7bdee9a6f722621fc42768368983130b`|
|精确边界|class token JSON wire≤70B；冻结row ID JSON wire≤68B；canonical总wire仍≤90000B|
|runtime修复|`8b1166e1ba4648204571612543319cf7a64dd1e3`|
|matched K5/payload修复|`ba78d723da5ca87d32d9715bcd2dbcc28280512a`；`0d3ebcfd56cb484cabadc026678cc50b73ff67f7`|
|本地验证|`py_compile`通过；195项D106聚焦测试通过；`git diff --check`通过；工作树clean|

## N607预登记

|字段|值|
|---|---|
|Python/GPU|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`cuda:0`|
|checkpoint及SHA|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D92 source/matrix SHA|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`；`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|RDCE wire及SHA|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`；`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|
|run root/CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r6`；其`source/`|
|日志/PID/输出|`logs/{prepare,smoke,predict}.log`；`control/predict.{pid,exit}`；`prepared/`、`smoke/`、`predictions/`|

## 性能目标与停止

K10/new5、new10、new20要求`A_old≥92%`、`F_old≥85%`、`N≥92/90/86%`；K5/new20相对matched K10/new20的`A/F/N/H`下降均≤5pp；K1/new20相对同row D92要求`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确严格增加。

仅P0协议/安全错误，或至少两个不同row在prediction前出现同一确定性异常指纹时停止；不得看accuracy、H、floor或遗忘。完成后按同一row报告receiver、slice、scene、before/after old、old floor、seen-new、H、forgetting、correct count和verdict。D91只作15行development证据。
