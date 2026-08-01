# D106-KCR/r5完整Target25实验报告

状态：`LANDED_PREPARE_COMPLETE_SMOKE_RCMR_TOKEN_LIMIT_FAILURE / NO_PERFORMANCE_RESULT`

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
|release source commit|`9a27a171f8c21a435c671daf278204af1fe753a9`|
|LF-preserving archive SHA256|`8a54012f78fa4df7ed249a0e977051a317e4f11a7d8f068d167fd3b80da66e87`|
|解包lock SHA256|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
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

## N607实际执行结果

2026-08-01首次direct preflight通过，r5远端根目录确认不存在、无D106进程，GPU0—7均为`0% utilization / 1 MiB used`。随后创建唯一不可覆盖r5目录并同步冻结归档。归档同步后的direct SSH与verified lab bridge曾各出现一次连接超时；本地确认无残留SSH后，第二次direct preflight恢复并继续同一r5，没有重复创建run或覆盖输出。

|步骤|结果|证据|
|---|---|---|
|归档/commit|通过|远端归档SHA256=`8a54012f78fa4df7ed249a0e977051a317e4f11a7d8f068d167fd3b80da66e87`；`git get-tar-commit-id=9a27a171f8c21a435c671daf278204af1fe753a9`|
|三份method lock|通过|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
|远端编译|通过|`stage2_d106_*.py`及`code/scripts/run_d106_target25.py`通过`py_compile`|
|prepare|通过|`TARGET25_D92_PACKAGES_LOCATED`；25个outer、75个scene、300个arm pair、600个state surface|
|真实无truth smoke|失败|首state在M_HEAD/RCMR registry构建阶段抛出`D106RCMR2VError: registered class exceeds the sealed wire-token limit`|
|完整prediction|未启动|零prediction；无detach、`predict.pid`、`predict.exit`、prediction或score artifact|
|最终状态|无性能结果|r5永久只读，不重试、不覆盖、不进入性能分析|

prepare输出SHA256：`target25_plan.json=2e840fe4abbfc18593a29691d377b6068fa4e8a0292d54a75ef1c9b2d99377ce`；`target25_context.json=64fca1d8d07a6da4fd4418ad32ac1491b055b53110f2181160bfdb26fbb84419`；`prepare_receipt.json=efdcfaf8f168eedcae685ec4d6332ae537068bcd78c51f729fa4a00b767d8d5e`。

## Smoke失败根因

诊断只读取已物化的`plan_state.json`和实现常量，未读取truth，也未修改远端。首state为`d106-rx-20_1__seed-713102__k-10__new-5::leo_clear_weak::before`，其合法注册类标识采用`cls_`加64位十六进制哈希的opaque token，例如`cls_5e21163d07ec881f84e2c239db7083ad6408611cf89cdc5430f37decbe07089f`，共68个UTF-8字节；RCMR实现却冻结`MAX_REGISTRY_TOKEN_WIRE_BYTES=64`。因此数据、feature runtime和RDCE成功物化后，在`_registry_tokens()`中被错误拒绝。

这是一项明确的实现契约尺寸缺陷，而不是数据、checkpoint、runtime lineage或GPU故障。最小修复应使RCMR wire token上限覆盖当前合法opaque类标识，并保留非空、唯一性、总注册类数、query隔离和其它fail-closed检查；修复后必须使用新run ID，不能修改或重跑r5。

## 取回证据

|本地文件|字节|SHA256|
|---|---:|---|
|`artifacts/remote_r5/prepare.log`|1456|`1028422af666c07de2a579d196d9a8327f45be1b96cd1d17aaf94cff160401ca`|
|`artifacts/remote_r5/smoke.log`|1753|`e93dc7866acae80ab02a7a8918c25059fa1145403decce504b776056b82ccf90`|
|`artifacts/remote_r5/prepare_receipt.json`|917|`efdcfaf8f168eedcae685ec4d6332ae537068bcd78c51f729fa4a00b767d8d5e`|
|`artifacts/remote_r5/target25_plan.json`|575589|`2e840fe4abbfc18593a29691d377b6068fa4e8a0292d54a75ef1c9b2d99377ce`|
|`artifacts/remote_r5/target25_context.json`|56341|`64fca1d8d07a6da4fd4418ad32ac1491b055b53110f2181160bfdb26fbb84419`|
|`artifacts/remote_r5/smoke_state000/plan_state.json`|1190|`565ae43050bc9cdd371c5cd2e531af8b7d4242239a4ce41c25240ab6a9f20685`|
|`artifacts/remote_r5/smoke_state000/paired_features.npz`|133867|`9cbd85329d0a980d6f9d0df7a146d585fb2ce468bdaa6c8b20a7e8e32e1dcbc4`|
|`artifacts/remote_r5/smoke_state000/paired_features.receipt.json`|1493|`005acd99b2cd3de87a8fd3978aa40c5077e6010f7f4d63987ed8c8ab5e6f6b03`|
|`artifacts/remote_r5/smoke_state000/rdce_row_authority.json`|1741|`af10f4b7ef719d28b37305cfc54742c676effbe35794bf2b15fcfdd1e68ff5a7`|

收尾检查：`pgrep -af '[r]un_d106_target25.py'`无结果；GPU0—7均`0% utilization / 1 MiB used`；本地`ssh.exe=0`，到N607及lab bridge的`ESTABLISHED TCP/22=0`。本实验没有best epoch、checkpoint或逐行性能表，因为没有产生prediction；不得据此形成任何性能比较。

## 下一步最小动作

只修复RCMR合法opaque token的wire长度契约并运行对应聚焦回归；以新的不可覆盖run ID复用相同冻结D92/checkpoint/RDCE资产，真实无truth smoke通过后立即detach完整Target25。重复数据验证、额外签名、报告平台或其它P2工作均不是发布前硬门。
