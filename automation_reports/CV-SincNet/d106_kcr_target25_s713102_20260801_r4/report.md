# D106-KCR/r4完整Target25实验报告

状态：`LOCAL_VERIFIED / RELEASE_PENDING`

## 实验登记

|字段|值|
|---|---|
|experiment ID|`d106_kcr_target25_s713102_20260801_r4`|
|日期|2026-08-01|
|operator|主agent负责方法、数据与结果分析；唯一Terra Max runner负责N607落地和运行证据|
|目标|完成D106-KCR的Target25完整矩阵，并与D62、D91、D92、SVRN-qKNN-BCRR全面比较|
|claim scope|`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|版本管理|仅本地Git提交；不push、不上传GitHub|

## 冻结方法和矩阵

K条件路由固定为`K1→M_DA`、`K5→M0`、`K10→M_HEAD`。每个state先完成`M0、M_DA、M_HEAD、M_JOINT`四臂，再派生`ROUTED`；路由不读取query truth、role、预测分数或receiver身份。

|维度|取值|
|---|---|
|receiver|`20-1,3-19,7-14,7-7,8-8`|
|seed|`713102`|
|slice|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|coverage|25个outer jobs、75个scene rows、300个arm rows、600个prediction surfaces、125个同row评分行|

K5/new20使用同receiver、同seed、同new20的D92 K10封包，运行时只物化`rank<5`；因此K5是K10同一物理池的前5shot，query逐值相同。其他row的source pool K等于实际K。该绑定只修复matched输入，不改变方法、路由、矩阵或目标。

## 精简发布gate和本地证据

发布前只保留三个条件：真实封包可运行；query truth/role不进入预测；新run不可覆盖。资产、矩阵和方法SHA冻结后不重复评审或重验。无truth smoke通过即detach完整prediction，性能不得触发提前停止。

|项目|证据|
|---|---|
|matched K5修复commit|`ba78d723da5ca87d32d9715bcd2dbcc28280512a`|
|payload契约修复commit|`0d3ebcfd56cb484cabadc026678cc50b73ff67f7`|
|release source commit|`86c4255e8f1f62a459b8471ef72317de384a07f9`|
|LF-preserving archive SHA256|`25a27835eaf2c8718b144e7c49a7329c853e06f0f6a5bb4783c038208227a46d`|
|解包lock SHA256|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
|本地验证|4文件`py_compile`通过；178项D106聚焦回归通过；`git diff --check`通过；工作树clean|
|负例|`source_pool_k`篡改、K5错误绑定K5 manifest、query truth/role附加字段均fail closed|

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
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r4`|
|source CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r4/source`|
|GPU|`cuda:0`；launch前记录实时占用|
|日志/PID/exit|`logs/prepare.log`、`logs/smoke.log`、`logs/predict.log`；`control/predict.pid`、`control/predict.exit`|
|预期输出|`prepared/`、`smoke/`、`predictions/`、完整manifest及600个surface|

## 性能目标

|slice|目标|
|---|---|
|K10/new5|`A_old≥92%`、`F_old≥85%`、`N≥92%`|
|K10/new10|`A_old≥92%`、`F_old≥85%`、`N≥90%`|
|K10/new20|`A_old≥92%`、`F_old≥85%`、`N≥86%`|
|K5/new20|相对matched K10/new20的`A/F/N/H`下降均不超过5pp|
|K1/new20|相对同row D92满足`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确数严格增加|

## 执行和停止规则

prepare后执行真实checkpoint、真实D92封包、`cuda:0`、无truth单state smoke；通过即发布完整prediction。只有P0协议/安全错误，或至少两个不同row在产生prediction前出现同一确定性异常指纹时停止。不得按accuracy、H、floor或遗忘停止。停止时只处理已绑定到本run的PID树并保留全部artifact。

完成后按同一row同时报告receiver、slice、scene、before old、after old、old floor、seen-new、H、forgetting、correct count和verdict。D62、D92、SVRN使用完整125证据；D91明确标注只有15行development证据。
