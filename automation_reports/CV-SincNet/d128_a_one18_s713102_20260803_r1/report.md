# D128-A-ONE18 r1发布与结果报告

## 1.基本信息

|字段|值|
|---|---|
|实验ID|`d128_a_one18_s713102_20260803_r1`|
|状态|`LOCAL_VERIFIED/NO_NEW_PERFORMANCE_RESULT`|
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
|执行顺序|直连预检→sync/hash/compile/help→唯一真实checkpoint A微episode smoke→fresh D127-compatible prepare→A单候选Phase1 bundle→D128 A-only prediction→durable truth-open→D128 truth/formal assets→score|
|预期artifact|prepared plan/K5 receipt、A single bundle、A-only prediction、truth-open、truth/formal/build receipt、score、完整日志/资源/清理receipt|

唯一必要smoke：使用真实checkpoint和冻结A资产/微episode，验证FSRG支持态梯度非零、训练路径可微、冻结外层无caller graph，并确认query/truth/role/quota访问均为0。该smoke不得读取性能。

停止规则仅为P0协议/安全违规，或至少两个不同任务/row在prediction前产生相同确定性异常指纹；单进程launcher-wide确定性故障或输出覆盖风险也属于系统性故障。不得因accuracy、H、floor或forgetting停止。不授权自动retry/restart；D128若再次因同一bridge/release体系在prediction前停止，则关闭该实现路线，不创建r2式重复修复。

## 6.结果表

|receiver|scene|K|arm|old_before|old_after|seen_new|H|old_floor|forgetting|total_correct|verdict|
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|

完成后必须回填72条同row指标、scope/receiver/scene/K pool、G1/G2/G3、资源、artifact SHA和最终关闭/晋级建议。当前没有新性能结果。
