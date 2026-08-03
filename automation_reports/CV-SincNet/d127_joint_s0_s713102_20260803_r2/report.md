# D127 S0 r2发布与结果报告

## 1.基本信息

|字段|值|
|---|---|
|实验ID|`d127_joint_s0_s713102_20260803_r2`|
|当前状态|`LOCAL_VERIFIED/NO_NEW_PERFORMANCE_RESULT`|
|目标|联合轻量域适应与D92-lite分类头，在固定18行S0 before/after矩阵上形成完整同row性能证据|
|Primary|Sol High：协议、集成、数据/结果分析和最终晋级/关闭决策|
|唯一runner|Terra Max：N607落地、启动、健康检查、artifact回收；不得调参或作性能决策|
|前序技术运行|r1：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`；未生成prepared、Phase1或prediction|

## 2.冻结矩阵与判据

|字段|冻结值|
|---|---|
|protocol|`p2_min_v1`|
|seed|`713102`|
|receiver|`20-1,3-19,7-14`|
|K/new_count|`K1/new20,K5/new20`|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|规模|18个row pair；before/after共36个状态|
|候选|A=`DA-A-FSRG-time_fuse`；B=`DA-B-FSRG-t2norm`；C=`DA-C-RDHA-joint_proj`|
|臂|`M0,M_DA,M_L92,M_JOINT`|
|S0-G1|`M_DA-M0`池化`H>0`|
|S0-G2|K5的`M_JOINT-M_DA`池化`H>0`|
|S0-G3|`M_JOINT-M0`池化`H>0`且old＋new总正确数增加|

不设0.5pp门，不运行588/fresh63/125矩阵，不按局部性能提前停止。弱候选在完整S0后关闭，不调参复活。

## 3.版本与r1修复

|项|值|
|---|---|
|核心S0实现|commit`3458ecba`|
|r1报告封存|commit`68b390b9`|
|namespace bootstrap|commit`0bf96729`|
|NumPy2兼容修复|commit`3f025ffd`|
|兼容原因|N607`CVS-RFFI`为Python3.10/torch2.1.0+cu121/numpy2.2.5；r1在`torch.from_numpy`启动前失败|
|修复边界|显式Python值copy，保持shape/dtype/device/detach；D127中`from_numpy/as_tensor/Tensor.numpy`双向ABI桥清零；不改变数学|

|文件|SHA256|
|---|---|
|`stage2_d127_torch_compat.py`|`f80cee172a9e832ce179a4276d1d904f949266e63f3f4354e7bb75b841f7ddad`|
|`stage2_d127_checkpoint_hooks.py`|`5f006cbe84187266f009e4cd0e3314122158dacea1c461d4c0923b62d5b2aa2f`|
|`stage2_d127_phase1_assets.py`|`3c1ab7f6f689452bb30c4b9da6f700214e138dde4bbafc9bd922ccc142fcb7df`|
|`stage2_d127_phase1_release.py`|`d1f0b00dacfc9fc6c46c3b58c4a2f414809058d804daac1d63e1ff6b7abd7015`|
|`stage2_d127_s0_package_adapter.py`|`007b38f8cfc20c4ae439ba332015a15a3428f91e64e982cf5d1ea350ce42c379`|
|namespace bootstrap|`90f7447ed5ebc121aa1d4d6f47be389a9a54a8bd5b1ccd9d35591c3508eb508f`|

## 4.本地验证

|检查|结果|
|---|---|
|全D127聚焦回归|`91 passed`；仅既有AMP弃用警告|
|兼容关键测试|`21 passed`；禁用`from_numpy/as_tensor/Tensor.numpy`后LBFGS、qKNN、量化闭环仍通过|
|静态桥检查|D127源码中三类NumPy/Torch ABI桥为0|
|独立复核|`P0=0,P1=0,RELEASE_READY`|
|query边界|zero-fit、zero-update、zero-selection；truth仅在paired prediction封存后打开|

## 5.固定资产

|资产|路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected IQ|`.../d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|selected IQ receipt|同目录`d106_ls_received_iq.receipt.json`|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|L_s join|`.../d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|D92 manifest|`.../d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`|`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|method lock|`configs/d127_joint_s0_method_lock_20260803.json`|`7b8df3c029d8096033b9a39734d563452f1f9b4bcb6737ade63821fb4786a650`|
|Target25 context|本地r7回收artifact，runner同步到r2 input|`e3cf5b15e29d5d907889874b1517da1ad77e5fa81085ed074d4c196af71830ba`|

## 6.运行交接

|字段|冻结值/待回填|
|---|---|
|N607 run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d127_joint_s0_s713102_20260803_r2`（不可覆盖）|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|CWD|`$RUN/source`|
|GPU|Phase1和target worker均A→GPU0、B→GPU1、C→GPU2；每个进程内部`cuda:0`|
|执行顺序|sync/hash/compile/help→prepare→三候选Phase1→merge asset→三candidate-worker→merge paired→open→truth-assets→score|
|日志/output/PID|`PENDING`，由唯一runner回填|
|预期artifact|3个单候选bundle、merged bundle、prepared plan/prefix receipt、3个worker prediction、paired prediction、truth-open、truth/formal receipt、score|

停止规则仅为P0协议/安全违规，或至少两个不同任务/row在预测前产生相同确定性异常指纹。不得因准确率、H、floor或forgetting停止；不授权自动retry/restart。技术失败必须保留partial artifact并使用新run ID。

## 7.同row结果表

|candidate|receiver|scene|K|arm|old_before|old_after|seen_new|H|old_floor|forgetting|total_correct|verdict|
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|

**预注册结论：**r2只修复r1已证实的运行时ABI兼容问题，方法、矩阵、阈值和数据资产保持不变。当前为`LOCAL_VERIFIED/NO_NEW_PERFORMANCE_RESULT`；下一步直接交由唯一Terra Max runner发布。
