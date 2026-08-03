# D128-A-ONE18 r1发布与结果报告

## 1.基本信息

|字段|值|
|---|---|
|实验ID|`d128_a_one18_s713102_20260803_r1`|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`|
|目标|用最小单A闭环验证轻型FSRG域适应、D92-Lite和二者联合是否产生真实同row正收益|
|Primary|Sol High：协议、集成、结果分析、关闭/晋级决策|
|唯一runner|Terra Max：N607落地、启动、健康检查、artifact回收；不得调参或作性能决策|
|前序|D127 r1/r2/r3均为prediction前技术停止；没有性能结论；不再原样发布三候选r4|

## 2.冻结方法、矩阵与判据

|字段|冻结值|
|---|---|
|protocol|`p2_min_v1`|
|candidate|A=`DA-A-FSRG-time_fuse`；B/C暂停|
|seed|`713102`|
|receiver|`20-1,3-19,7-14`|
|K/new_count|`K1/new20,K5/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|规模|18个row pair；before/after共36个state row|
|臂|`M0,M_DA,M_L92,M_JOINT`；K1按冻结等价alias，不重复计算head|
|G1|池化`H(M_DA)>H(M0)`|
|G2|K5池化`H(M_JOINT)>H(M_DA)`|
|G3|池化`H(M_JOINT)>H(M0)`且old＋new总正确数增加|

完整one-shot任一方向门不成立即关闭A并转入下一个原理，不调层、rank、步数、view、seed或阈值；三项均成立才恢复S0/S1。该运行不是正式S0、Target25或promotable证据。

## 3.本地版本与验证

|项|值|
|---|---|
|目标冻结|commit`58ee10f5`|
|冻结审计修复|commit`46284a3b`|
|单A闭环|commit`03af3d6`|
|method lock|`configs/d127_joint_s0_method_lock_20260803.json`，SHA256`7b8df3c029d8096033b9a39734d563452f1f9b4bcb6737ade63821fb4786a650`|
|联合回归|`29 passed`；仅既有AMP弃用warning|
|独立复核|`P0=0,P1=0,RELEASE_READY`|
|审计语义|训练forward仍严格可微；冻结asset评估走同一checkpoint downstream但不伪造caller graph|
|one-shot边界|只接受`single_candidate`且candidate列表精确为A；拒绝merged A/B/C；prediction truth-free、独占写|

|关键文件|SHA256|
|---|---|
|`stage2_d127_checkpoint_hooks.py`|`8814e78dfe8eeaaac24106e1d96c234ff9542abff94d5bc6c11c2bec331078b5`|
|`stage2_d127_phase1_release.py`|`42348f8f8b3fd1967d5cb3a6bf177cda7334a3b6e9d37fe41ca09fe727cc7887`|
|`stage2_d128_a_one18.py`|`7e388fed27bc7eebe93d552b724b89cc4cf68672587fc7b09418ae4e7b5737de`|
|`stage2_d128_a_one18_scorer.py`|`f1dc1bcdb16ef2ed2d79e4a5bb71c50f799bf42326d015909c1f0fd5e8a04f2f`|
|`run_d128_a_one18.py`|`bb3775926266b0cf9f453b301806a3a49d7dc4c8e696b02e2b894189e89f5226`|
|`score_d128_a_one18.py`|`c3562e4660f006a0cf9ed485cda8fd83eae9f04ac5b4fca42f4700b93f1084c3`|
|`build_d128_a_one18_truth_assets.py`|`d3a93954c192def257178150df3e3fdf5812088b45a716a83c372cd884528ac6`|

## 4.固定资产

|资产|路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|selected IQ receipt|同目录`d106_ls_received_iq.receipt.json`|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|L_s join|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|D92 root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720`|manifest`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|Target25 context|runner同步到r1 input|`e3cf5b15e29d5d907889874b1517da1ad77e5fa81085ed074d4c196af71830ba`|

## 5.N607交接

|字段|冻结值/待回填|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d128_a_one18_s713102_20260803_r1`，首次创建且不可覆盖|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|CWD|`$RUN/source`|
|GPU|A Phase1与prediction均GPU0，进程内`cuda:0`|
|执行顺序|直连预检→sync/hash/compile/help→fresh D127-compatible prepare→A单候选Phase1 bundle→用该冻结A bundle做唯一真实checkpoint微episode smoke→D128 A-only prediction→durable truth-open→D128 truth/formal assets→score|
|预期artifact|prepared plan/K5 receipt、A single bundle、A-only prediction、truth-open、truth/formal/build receipt、score、完整日志/资源/清理receipt|

唯一必要smoke：A单候选Phase1 bundle完成后，使用该冻结A资产和真实checkpoint在一个source-only微episode上验证FSRG支持态梯度非零、训练路径可微、冻结外层无caller graph，并确认target query/truth/role/quota访问均为0。该smoke不得读取性能；通过后直接进入Target prediction。

停止规则仅为P0协议/安全违规，或至少两个不同任务/row在prediction前产生相同确定性异常指纹；单进程launcher-wide确定性故障或输出覆盖风险也属于系统性故障。不得因accuracy、H、floor或forgetting停止。不授权自动retry/restart；D128若再次因同一bridge/release体系在prediction前停止，则关闭该实现路线，不创建r2式重复修复。

## 6.结果表

|receiver|scene|K|arm|old_before|old_after|seen_new|H|old_floor|forgetting|total_correct|verdict|
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|`N/A`|`N/A`|`N/A`|`N/A`|`N/A`|`N/A`|`N/A`|`N/A`|`N/A`|`N/A`|`N/A`|A bundle在prediction前技术退出；无性能结果|

本run未生成prediction，因此没有72条同row指标、pool或G1/G2/G3；这些字段不得以局部值或历史值补写。当前没有新性能结果。

## 7.唯一runner预落地记录

|字段|实测值|
|---|---|
|N607直连预检|`2026-08-03 17:25 CST`通过；普通账号、项目根和Python3.10.19可用|
|r1唯一性|`$RUN`不存在；未触碰D127或其他历史run|
|固定资产|checkpoint、selected IQ、receipt、`L_s` join及D92 manifest均逐项匹配第4节SHA256|
|资源|GPU0-7均`0%/1MiB`且无compute process；`/home`可用约7.4T|
|本地冻结输入|D127/D106最小source closure、5个D128文件、method lock及Target25 context共30项均存在并逐项计算SHA256；第3节7个新/更新关键SHA全部匹配|
|本地SSH清理|预检及资产核验后均为`ssh.exe=0`、至N607/lab bridge的TCP22连接数`0`|
|当前状态|`PRE_LANDING_VERIFIED/NO_NEW_PERFORMANCE_RESULT`；下一步首次创建r1专用不可覆盖根并精确同步|

## 7.1落地、哈希与运行时入口闭合

|字段|实测值|
|---|---|
|r1专用根|`$RUN`首次创建；仅含本run的`source/input/assets/predictions/truth/score/logs/receipts/smoke`目录|
|精确同步|21个模块、7个CLI脚本、method lock和Target25 context，共30个冻结输入；未复用D127-r3 source文件|
|远端SHA|30/30逐项匹配本地`b4810e80`冻结输入；checkpoint hooks=`8814e78d...078b5`、Phase1 release=`42348f8f...c7887`、D128 core=`7e388fed...5737de`|
|编译与CLI|28个Python文件`py_compile`通过；D127 Phase1/prepare及3个D128 CLI的`--help`均通过|
|日志|`$RUN/logs/preflight_compile_help.log`，完成标记`COMPILE_HELP_PASS`|
|SSH|每次SCP/SSH后均为`ssh.exe=0`、至N607/lab bridge的TCP22连接数`0`|

当前状态为`LANDED/PREFLIGHT_PASSED/PREPARE_PENDING/NO_NEW_PERFORMANCE_RESULT`。下一步直接fresh prepare，不复用D127-r2/r3产物。

## 7.2fresh prepare完成

|字段|实测值|
|---|---|
|状态|`D127_S0_PREPARED`，`truth_loaded=false`|
|计划规模|18个row pair、36个state row|
|prepared plan|`input/prepared/prepared_plan.json`；内容SHA256`1e6d931c6b0f833133e3a7589c6f7afa2cfbf4170e792a94f0bac1431d682108`；文件SHA256`c147120edb73481f2243535bcdc86da56ec0193ce052145e4515c8773ea76803`|
|K5 prefix receipt|`input/prepared/k5_prefix_receipt.json`；内容SHA256`13b5b827469a74f5581e969a762f250e874708936ce03bacc5d3d1324e124ba2`；文件SHA256`7688f4a4870377900185598637cc742ce5b2d31e0926f415bf6145a166a251a5`|
|日志/SSH|`logs/prepare.log`；连接结束后`ssh.exe=0`、至N607/lab bridge的TCP22连接数`0`|

当前状态为`PREPARED/PHASE1_A_PENDING/NO_NEW_PERFORMANCE_RESULT`。下一步重新核验GPU0占用并仅启动候选A；B/C保持禁止。

## 7.3候选A Phase1已启动

|字段|实测值|
|---|---|
|candidate|仅`DA-A-FSRG-time_fuse`；B/C未启动|
|PID/GPU|PID`450398`；外层GPU0，进程内`cuda:0`|
|CWD/cmdline|`$RUN/source`；完整cmdline绑定本run的`build_d127_phase1_assets.py`、A资产目录、冻结method lock与5个固定资产|
|log/output|`logs/phase1_DA-A-FSRG-time_fuse.log`；`assets/DA-A-FSRG-time_fuse`|
|启动后健康点|40秒时PID存活且PPID已归1；GPU0`15%/972MiB`；error marker`0`；日志暂未flush；资产尚未封存|
|runner包装说明|首次8秒检查末尾因`nvidia-smi | head`在`pipefail`下返回SIGPIPE/141；实验PID当时及后续均存活，非实验异常，未重启|
|SSH|两次启动健康连接后均为`ssh.exe=0`、至N607/lab bridge的TCP22连接数`0`|

当前状态为`RUNNING_PHASE1_A/NO_NEW_PERFORMANCE_RESULT`。下一步仅用短连接检查PID、日志增长、GPU和A bundle闭合，不读取accuracy/H等性能指标。

## 7.4系统性技术停止与路线封存

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`。

候选A在Phase1 outer-fold审计、bundle封存和任何prediction之前退出。准确异常位于`stage2_d127_phase1_release.py:1820`：`D127Phase1ReleaseError: D127 outer audit failed isolation/equivariance/nonzero/query-change closure`。该异常发生在`_outer_fold_audit_and_final_rebuild`，属于冻结handoff定义的同一bridge/release体系再次prediction前失败；因此本run停止后续唯一smoke、prediction、truth-open、truth builder和score，不重启、不创建r2、不在线调参。

|项|实测结果|
|---|---|
|Phase1 PID|`450398`已退出；停止后没有带`$RUN`路径的存活进程|
|Phase1日志|`logs/phase1_DA-A-FSRG-time_fuse.log`；SHA256`ec44126ebc8f9ca21e22a5a8a4a20a84396144bf922985591cf469da225f8814`|
|输出计数|assets=`0`、smoke=`0`、predictions=`0`、truth=`0`、score=`0`|
|GPU|GPU0-7均回到`0%/1MiB`|
|其余日志|preflight SHA256`3a196cb52c435e296244b10d0bc724cd2eb9cd60b7242e218ec3cac8a5716bad`；prepare SHA256`315c50e7bfa5124d29790d57dc41a8e07495584a845302779e91ba4532a91c15`|
|prepared证据|plan文件SHA256`c147120edb73481f2243535bcdc86da56ec0193ce052145e4515c8773ea76803`；K5 receipt文件SHA256`7688f4a4870377900185598637cc742ce5b2d31e0926f415bf6145a166a251a5`|
|artifact回收|上述3份日志、2份prepared JSON和PID receipt共6项已镜像到双报告的`artifacts/remote_r1/`，6/6逐项同SHA|
|SSH|所有连接结束后均为`ssh.exe=0`、至N607/lab bridge的TCP22连接数`0`|

本停止不证明候选A性能为负，也不产生D92联合效果结论；它只关闭当前bridge/release实现路线的重复修复发布。后续方法选择和最终晋级/关闭决定由Primary结合既有研发证据作出。
