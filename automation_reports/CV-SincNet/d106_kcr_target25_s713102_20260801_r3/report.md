# D106-KCR/r3完整Target25实验报告

状态：`LANDED_PREPARE_COMPLETE_SMOKE_K5_K10_PAIRING_FAILURE / NO_PERFORMANCE_RESULT`

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
|payload修复commit|`0d3ebcfd56cb484cabadc026678cc50b73ff67f7`|
|release source commit|`d0e17621a8a9b50f1aa604d436d15b7d350822b4`|
|LF-preserving archive SHA256|`08c5414cddb165d66e6e6965342fdad6e871815439166ab75470ebf3e0bdb6e3`|
|解包lock SHA256|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
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

## r3实际执行证据

|项目|结果|
|---|---|
|direct preflight|通过；r3 run root原先不存在；无D106进程；GPU0–7均0%利用率、1MiB占用|
|远端archive|SHA=`08c5414cddb165d66e6e6965342fdad6e871815439166ab75470ebf3e0bdb6e3`；commit=`d0e17621a8a9b50f1aa604d436d15b7d350822b4`|
|远端lock/compile|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`；`py_compile=PASS`|
|prepare|`TARGET25_D92_PACKAGES_LOCATED`；25/75/300/600闭合|
|plan/context|plan=`3ba353276daf1877ea23de97c34236534c4ebdd732dceff9ad9ffb6fdd37b6c7`；context=`3e0d64e5892ec6162270e135d4013ec750d02c57da4ff2e4a1d919b2fe2a9da6`|
|smoke|失败关闭：`D106Target25RunnerError: D92 K5 support/query pairing differs from matched K10`|
|prediction launch|未detach；PID=`NOT_CREATED`；`smoke/`、`predictions/`、`control/predict.pid`、`control/predict.exit`均不存在|
|清理状态|无run-owned预测进程；GPU0–7恢复0%利用率、1MiB占用；`ssh.exe=0`，目标TCP22连接为0|

实际`prepare`入口为`code/scripts/run_d106_target25.py prepare`，CWD为上表source CWD，输入为本报告登记的D92 matrix、checkpoint、RDCE wire和三个source内lock，输出为`<run root>/prepared`。实际smoke调用`smoke_d106_target25_prepared_state`，固定`row_index=0,scenario_index=0,state_index=0,device=cuda:0,feature_batch_size=64`，plan/context及其SHA如上，无truth参数。

本地取回证据位于`E:\type10-7\automation_reports\CV-SincNet\d106_kcr_target25_s713102_20260801_r3\artifacts\remote_r3`：`prepare.log`SHA=`2cca54104cef9214c172110d92cf848d5daa575a5536d8d25cb58cb2bfaa182b`；`smoke.log`SHA=`5efede20fe637c8c3de5af28f45cf25ff4466109f081c0b9fda3cb8e6d833a31`；`prepare_receipt.json`SHA=`7357b3bc629ccf928763e827296b02419810484f1d5534373cab110aa9a3ee48`。

### K5与matched K10只读差异摘要

固定`receiver=20-1`、`scene=leo_clear_weak`、`state=before`，仅通过已验证D92 support/query封包读取opaque token，不读取truth：

|比较项|K5|K10|结论|
|---|---:|---:|---|
|support数量|30|60|K5不是K10子集|
|query数量|120|120|数量相同，但token集合不同|

首个K5独有support为`sid_d7a85059b4bc1d687bf20631649043d2c285bcbdc288279f390082e3d3c68ce6`；首个K10独有support为`sid_84ca046a816598b493aebd109a03951aa2102b8a3203858821a154cdc2b48d5d`。query第0项即不同：K5为`qid_9be2a7b190c4d0328cfe0355c5baa46e845d3175dd95b86ae92717e6ec3f76d1`，K10为`qid_ac3c92bde0446ea92c7e460e58ad2ba0c37477e0279c93768a5595169ac9f0e4`。这说明现有D92 K5与K10封包是独立物理split；本次失败项是runner额外要求K5 support嵌套于K10且query token集合相同。

r3不得覆盖、恢复、重试或重标为性能实验。不存在可用于性能分析的prediction。

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
