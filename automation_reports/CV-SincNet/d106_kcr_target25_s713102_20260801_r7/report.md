# D106-KCR/r7完整Target25实验报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT / D106_END_TO_END_CLOSED`

## 登记与冻结实验

|字段|值|
|---|---|
|experiment ID|`d106_kcr_target25_s713102_20260801_r7`|
|日期/operator|2026-08-01；主agent负责方法、数据和结果，唯一Terra Max runner负责N607|
|目标/边界|完整D106-KCR Target25；`DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE`|
|Git|仅本地提交；不push、不上传GitHub|
|receiver/seed|`20-1,3-19,7-14,7-7,8-8`；`713102`|
|slice/scene|`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`；3个`leo_*_weak`|
|coverage|25 outer、75 scene、300 arm、600 prediction surfaces、125评分行|

路由固定`K1→M_DA`、`K5→M0`、`K10→M_HEAD`；每state先完成四臂再派生`ROUTED`。K5复用matched K10封包并只取`rank<5`。query truth、role、fit、update、selection均禁止。

## 精简gate和本地证据

发布只保留真实封包可运行、query隔离、run不可覆盖。no-truth smoke通过即detach完整矩阵，不再等待额外review；不得按性能停止。

|项目|证据|
|---|---|
|ReLU零行修复|`4830eca9cad50a69e7b1de2ffbb4b59c6300e6d4`；仅plus norm≤1e-12时以same-IQ signed行替换，signed零/非finite fail closed|
|审计|support/query分别落盘canonical receipt，绑定替换数、物理ID根、输入/输出数组receipt；truth access=false、state updated=false|
|RCMR wire修复/lock|`7ff4b4e7c882aed6e7fb62123b906eb388462a1c`；`cc5795be25622da0f060056bb931e20c7bdee9a6f722621fc42768368983130b`|
|其他运行修复|runtime=`8b1166e1ba4648204571612543319cf7a64dd1e3`；matched K5=`ba78d723da5ca87d32d9715bcd2dbcc28280512a`；payload=`0d3ebcfd56cb484cabadc026678cc50b73ff67f7`|
|本地验证|`py_compile`通过；204项D106聚焦测试通过；`git diff --check`通过；工作树clean|
|release source/archive|commit=`21b0cfae36083c1809e345d18117239e7179eb30`；SHA256=`37e388552b3e2598a15316a0881416669356e91d9ef30b5368d0437fa21ec252`|
|解包lock SHA256|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`cc5795be25622da0f060056bb931e20c7bdee9a6f722621fc42768368983130b`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
|终止重复发布规则|r7为D106最后一次端到端尝试；若再次出现新的技术失败，不创建r8；若完整性能未达标，立即转入下一方法研发|

## N607预登记

|字段|值|
|---|---|
|Python/GPU|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`cuda:0`|
|checkpoint SHA/path|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|D92 source/matrix SHA|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`；`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|RDCE wire/SHA|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`；`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|
|run root/CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_kcr_target25_s713102_20260801_r7`；其`source/`|
|日志/控制/输出|`logs/{prepare,smoke,predict}.log`；`control/predict.{pid,exit}`；`prepared/`、`smoke/`、`predictions/`|

## 性能目标、停止和分析

K10三slice要求`A_old≥92%`、`F_old≥85%`、`N≥92/90/86%`；K5/new20相对matched K10/new20的`A/F/N/H`下降均≤5pp；K1/new20相对同row D92要求`ΔH≥2pp、ΔF≥2pp、ΔA≥0、ΔN≥0`且总正确严格增加。

仅P0或至少两个不同row在prediction前出现同一确定性异常指纹时停止；不得看accuracy/H/floor/forgetting。完整后按同row报告全部指标，并与D62、D92、SVRN完整125比较；D91只作15行development证据。

## N607落地与smoke

2026-08-01 direct preflight通过；r7根目录此前不存在、无D106进程，GPU0—7均为`0% utilization / 1 MiB used`。随后仅创建该不可覆盖run目录并同步冻结归档。

|检查|结果|
|---|---|
|远端archive SHA256|`37e388552b3e2598a15316a0881416669356e91d9ef30b5368d0437fa21ec252`，匹配|
|归档内嵌commit|`21b0cfae36083c1809e345d18117239e7179eb30`，匹配|
|解包locks|RDCE=`e7a1982b4bdeaf5b8179993ce78f4a2af26965d8f4a3239440dbe636ebf14cc1`；RCMR=`cc5795be25622da0f060056bb931e20c7bdee9a6f722621fc42768368983130b`；KCR=`a3d530734b90454724166f620d7017f80e6de838fd4ca469c04abb155534ab6a`|
|远端编译|`stage2_d106_*.py`及`code/scripts/run_d106_target25.py`通过|
|prepare|`TARGET25_D92_PACKAGES_LOCATED`；25/75/300/600完整闭合|
|真实smoke|真实D92、checkpoint、RDCE、`cuda:0`通过；`query_truth_access=false`，query fit/update/selection均为0|

prepare文件SHA256：plan=`9aef55a97ad31db8e60b4a193d0df2d10fc2157b49eea01b2100d4142ac41368`；context=`e3cf5b15e29d5d907889874b1517da1ad77e5fa81085ed074d4c196af71830ba`；receipt=`2910debb302e04554b60167e9a8655d97f918e924ca6d5f9cc9671643a09441c`。smoke receipt file SHA256=`240c7eec67033cd043fc10e9b6a169f8aa9d1466dc7b45d462575655f42a615d`；prediction receipt=`9bc557ce9b02a098ba24e58b21e8db4b5829d5a66ba43de8a46cf3b2318fca96`；route receipt=`388bcb65e49f3df643c101d60353785e4157ec9090cd5e20938eea7b6098aab7`。

## 完整prediction发布与首wave

smoke通过后未等待额外批准，于N607时间2026-08-01 17:57:40 CST detach完整prediction。wrapper PID=`3366017`，Python child PID=`3366019`；CWD为r7`source/`，cmdline完整绑定r7 plan/context/predictions、冻结checkpoint、RDCE wire和RCMR lock。运行4分52秒时已物化36/600个state目录、216个文件，越过首个24-state outer row，异常指纹为0；GPU0约624MiB。该检查只用于技术健康，未读取性能。

## 最终技术失败

进程在第46个state目录处自行exit=1。两PID随后均退出，GPU释放。最终仅有46/600个state目录、272个文件；没有生成完整prediction manifest，因此没有可评分或可比较的性能结果。

|字段|值|
|---|---|
|失败目录|`state-045`|
|对应冻结state|outer=`d106-rx-3_19__seed-713102__k-10__new-20`；receiver=`3-19`；scene=`leo_low_elev_weak`；lifecycle=`after`|
|阶段/arm|`PRE_ARM_STATE_MATERIALIZATION`；尚未进入四臂预测|
|异常|`D106Target25EvaluatorError: publisher plus views must equal ReLU(signed)`|
|调用路径|`predict_d106_target25`→`smoke_d106_target25_state`→`_D106RealStateMaterializer.__call__`→`publish_d106_paired_features`|

ReLU totalization本身已执行并写出审计receipt：support输入plus SHA=`11b0c1ba335777f3c7a06850ce83da1a6388c6383df7093f665ff57e71073106`，输出plus SHA=`4dbef169591aa0fba3af604dd81c522292e3dec0f167f5d0d82a5af903e02dd8`，`replaced_count=1`；query输入/输出plus SHA均为`4863fda938048dfb2104a0d045824187c413b938deafaf55ad807c4a139236f3`，`replaced_count=0`。两者均声明`query_truth_access=false`、`state_updated=false`。随后旧publisher不接受totalized plus与原始`ReLU(signed)`不同，故在feature archive发布前失败。这是新技术失败，不是性能门或人工停止。

## 取回证据

|文件|SHA256|
|---|---|
|`artifacts/remote_r7/prepare.log`|`c02655bf3248bed377a16252a589fbe337297859081eb5c2b56664947de5e1d2`|
|`artifacts/remote_r7/smoke.log`|`51d3e7d09cda24de7dfb1b9cf9ee827539108bf63397dc59960b7d86784dd939`|
|`artifacts/remote_r7/predict.log`|`44ab8bf208c10f3b7f745d4ca049bc9045dee40d5f9f2f54114444217b4103de`|
|`artifacts/remote_r7/predict.pid`|`faa7707de7991ad62dc0cbaec919f4cedd525bc390ed18211229165bef753efa`|
|`artifacts/remote_r7/predict.exit`|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|
|`artifacts/remote_r7/prepare_receipt.json`|`2910debb302e04554b60167e9a8655d97f918e924ca6d5f9cc9671643a09441c`|
|`artifacts/remote_r7/target25_plan.json`|`9aef55a97ad31db8e60b4a193d0df2d10fc2157b49eea01b2100d4142ac41368`|
|`artifacts/remote_r7/target25_context.json`|`e3cf5b15e29d5d907889874b1517da1ad77e5fa81085ed074d4c196af71830ba`|
|`artifacts/remote_r7/smoke_receipt.json`|`240c7eec67033cd043fc10e9b6a169f8aa9d1466dc7b45d462575655f42a615d`|
|`artifacts/remote_r7/failure_state045/support_plus_totalization.receipt.json`|`ad904751d648821b6524139a1a16ab62e16998c12e18fe4cc6cd783da46f134f`|
|`artifacts/remote_r7/failure_state045/query_plus_totalization.receipt.json`|`33fdec65e483da2a20260101d9c14334d689292715068665dbc6afff8560e6b9`|

收尾检查为：wrapper/child均不存在；GPU0—7均`0% utilization / 1 MiB used`；本地`ssh.exe=0`，到N607及lab bridge的`ESTABLISHED TCP/22=0`。r7所有partial artifact永久保留且不可覆盖。

## 最终决策

按预登记终止规则，不创建D106 r8，不再扩张D106发布工程。D106没有获得完整Target25性能证据，不能与D62、D91、D92或SVRN作数值晋级比较。下一轮资源转入其它方法研发；若未来仅作代码债处理，应与新的性能实验目标隔离。
