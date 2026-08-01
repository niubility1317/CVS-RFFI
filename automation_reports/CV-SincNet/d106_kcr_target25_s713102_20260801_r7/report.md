# D106-KCR/r7完整Target25实验报告

状态：`LOCAL_VERIFIED / RELEASE_PENDING`

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
