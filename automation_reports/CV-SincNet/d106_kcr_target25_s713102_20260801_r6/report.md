# D106-KCR/r6完整Target25实验报告

状态：`RUNNING / SMOKE_PASS / FULL_PREDICTION_DETACHED`

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
|release source/archive|commit=`2f7c0581148abf8c989a0b4941726bc1e2fe868e`；SHA256=`0dbe5ca0b4549975c6a8af5a8213fada4d2b0fe7a94ff3bbc8ea77a57ec2033d`|
|解包lock SHA256|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`cc5795be25622da0f060056bb931e20c7bdee9a6f722621fc42768368983130b`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|

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

## N607落地与真实smoke

2026-08-01 direct preflight通过；r6远端根目录此前不存在，无D106进程，GPU0—7均为`0% utilization / 1 MiB used`。随后只创建该不可覆盖run目录并同步冻结归档。

|检查|结果|
|---|---|
|远端archive SHA256|`0dbe5ca0b4549975c6a8af5a8213fada4d2b0fe7a94ff3bbc8ea77a57ec2033d`，匹配|
|归档内嵌commit|`2f7c0581148abf8c989a0b4941726bc1e2fe868e`，匹配|
|解包locks|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`cc5795be25622da0f060056bb931e20c7bdee9a6f722621fc42768368983130b`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
|远端编译|`stage2_d106_*.py`及`code/scripts/run_d106_target25.py`通过|
|prepare|`TARGET25_D92_PACKAGES_LOCATED`；25/75/300/600完整闭合|
|真实smoke|真实D92、checkpoint、RDCE、`cuda:0`通过；`query_truth_access=false`，query fit/update/selection均为0|

prepare文件SHA256：plan=`c9b6e042a54cbee68cd87797fb29a2482098aebece4be20f32b6f1da74dedc21`；context=`b8894f24ee16644216502e1631a809977c94962d67ee3d864961ab1e14d08bd8`；receipt=`74e3a16ab7590c2de0e322d85b87c81994faad926092576b4bfbcf0b4516169b`。smoke receipt file SHA256=`bc069621b1d409806fd996c50ab98d5b389ba0fab96e0436b57c549d66bc2e7c`，首state prediction receipt=`fffed7fc27f9945d2f2cc7c08b6ff8a8983cf62bcf46a301066890d099c277d3`，route receipt=`f66eb80757451b863730eb612073b80466e7a1a9e4fac291f8dbb63cce62be54`。

## 完整prediction发布证据

smoke通过后未等待额外批准，于N607时间2026-08-01 17:34:04 CST detach完整prediction。

|字段|实际值|
|---|---|
|wrapper PID|`3352634`|
|Python child PID|`3352637`|
|CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r6/source`|
|输出/log/exit|`predictions/`；`logs/predict.log`；`control/predict.exit`|
|设备|`cuda:0`，`feature_batch_size=64`|
|精确输入绑定|r6 plan/context、冻结checkpoint、RDCE wire及新RCMR lock均出现在child cmdline中|
|启动验证|wrapper/child存活；PID、CWD、cmdline和run-root一致；predict log创建|
|首wave验证|运行4分39秒时已物化34/600个state输入、136个文件，越过首个24-state outer row；异常指纹为0，exit仍pending|
|资源|GPU0分配约624MiB；其它GPU约4MiB；未发现其它D106进程|

该入口在完整prediction闭合后一次性写最终JSON到`predict.log`，运行中以每state四份不可变输入文件增长作为进度证据。首wave只证明技术健康，不读取accuracy、H、floor或其它性能指标。

## 已取回的静态证据

|文件|SHA256|
|---|---|
|`artifacts/remote_r6/prepare.log`|`e6951fb6916e99fc99c55477d3248761d9c81c341b54a705917bde93edcd1836`|
|`artifacts/remote_r6/smoke.log`|`585c3a701b871b2655bf58026cde4442985ccfe20bc35fc5ada85fe3c84193bc`|
|`artifacts/remote_r6/prepare_receipt.json`|`74e3a16ab7590c2de0e322d85b87c81994faad926092576b4bfbcf0b4516169b`|
|`artifacts/remote_r6/target25_plan.json`|`c9b6e042a54cbee68cd87797fb29a2482098aebece4be20f32b6f1da74dedc21`|
|`artifacts/remote_r6/target25_context.json`|`b8894f24ee16644216502e1631a809977c94962d67ee3d864961ab1e14d08bd8`|
|`artifacts/remote_r6/smoke_receipt.json`|`bc069621b1d409806fd996c50ab98d5b389ba0fab96e0436b57c549d66bc2e7c`|

## 下一次检查

以短SSH继续核对state输入计数、PID/child、CWD、exit、GPU和异常指纹。只在P0或两个不同row出现同一确定性零prediction异常时停止；性能不得触发停止。完整artifact形成后取回prediction manifest并交由主agent独立打开truth和评分。
