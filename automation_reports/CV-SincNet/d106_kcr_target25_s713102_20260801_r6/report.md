# D106-KCR/r6完整Target25实验报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

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

## 完整prediction技术退出

launcher在首个未捕获异常处自行退出，wrapper PID`3352634`与Python child PID`3352637`均已结束，`control/predict.exit=1`。这不是按性能人工停止；没有读取truth或任何accuracy、H、floor指标。退出前物化46/600个state目录、共183个文件，未形成完整prediction manifest，因此所有部分结果均不可评分、不可比较、不可晋级。

精确失败state：

|字段|值|
|---|---|
|state|`state-045`|
|outer row|`d106-rx-3_19__seed-713102__k-10__new-20`|
|receiver/scene|`3-19`；`leo_low_elev_weak`|
|lifecycle|`after`|
|arm|`PRE_ARM_STATE_MATERIALIZATION`；尚未进入`M0/M_DA/M_HEAD/M_JOINT`任一arm预测|
|异常|`ZIDStudentTQKNNError: z_id rows contain a zero-norm vector`|

完整调用路径为：`run_d106_target25.py:main`→`predict_d106_target25`→`smoke_d106_target25_state`→`_D106RealStateMaterializer.__call__`→`build_typed_zid_support_bank(features.support_plus,...)`→`normalize_zid_rows`。异常发生在为该state建立typed qKNN support bank时，而非某个arm输出之后。

## 零范数只读诊断

诊断仅读取`state-045/paired_features.npz`中的无truth表征及其physical token；判定沿用实现中的`EPSILON=1e-12`。

|表征|shape|零范数行数|最小L2范数|零行physical token|
|---|---:|---:|---:|---|
|`support_plus`|`260×160`|1|0|`sid_4e807dc6e469c9c116c7c858ec5934103d1e7e24d6334409ba7697db32bef53b`，index=235|
|`support_signed`|`260×160`|0|3.7000724247374177|无|
|`query_plus`|`520×160`|0|0.013304952730689676|无|
|`query_signed`|`520×160`|0|3.418791828126275|无|

同一support physical token在`signed`表征中的L2范数为`14.562560077286715`。因此根因是ReLU后的`support_plus`单行完全零化，不是原始`signed`表征也为零。最小后续修复应只处理这种plus-view退化，同时保留signed信息、physical-token绑定、query隔离和fail-closed数值检查；不得远端修改或重跑r6。

## 最终取回与清理证据

|文件|SHA256|
|---|---|
|`artifacts/remote_r6/predict.log`|`b1719e3a075b8a3769819c46667efc0db5119cd15018f7c809a1434e64a01c07`|
|`artifacts/remote_r6/predict.pid`|`a4e0e1bf53d7187f86161f73b59c681e2153ae3705f15e9a185f066527052dcc`|
|`artifacts/remote_r6/predict.exit`|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|
|`artifacts/remote_r6/failure_state045/plan_state.json`|`ce8508995a0152bf92c00f98da3cf30dab1641d6c0dabe47d2eab668c23d90f2`|
|`artifacts/remote_r6/failure_state045/paired_features.npz`|`37313e8cd8d01fcb8e40b9b79ebdac205e20326fa015ff71e5ec34ca851a9c2c`|
|`artifacts/remote_r6/failure_state045/paired_features.receipt.json`|`f1803e772ea39a9b67237f6d162eff6adab5be2e7f30dc0a1d10cd4663a3589f`|

收尾时两PID均不存在，GPU0—7均`0% utilization / 1 MiB used`；本地`ssh.exe=0`，到N607及lab bridge的`ESTABLISHED TCP/22=0`。r6及其46个partial state目录永久保留为只读技术失败证据。

## 下一步最小动作

本地修复plus-view零向量处理，运行覆盖“plus为零但signed非零”的聚焦负例/正例及真实no-truth smoke，再以新run ID直接发布完整Target25。重复数据验证、额外签名、通用重试框架和报告平台均不是发布前硬门。
