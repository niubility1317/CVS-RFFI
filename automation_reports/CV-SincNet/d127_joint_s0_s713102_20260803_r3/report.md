# D127 S0 r3发布与结果报告

## 1.基本信息

|字段|值|
|---|---|
|实验ID|`d127_joint_s0_s713102_20260803_r3`|
|时间|`2026-08-03`|
|当前状态|`LOCAL_VERIFIED/NO_NEW_PERFORMANCE_RESULT`|
|目标|联合轻量域适应与D92-lite分类头，在固定18行S0 before/after矩阵上取得完整同row性能证据|
|Primary|Sol High：协议、方法集成、数据/结果分析和最终晋级/关闭|
|唯一runner|Terra Max：N607落地、健康检查、artifact回收；不得改方法、调参或作性能决策|
|技术前序|r1：NumPy/Torch ABI停止；r2：历史D106 receipt当前checkout闭合漂移停止；均为`NO_PERFORMANCE_RESULT`|

## 2.假设、冻结矩阵与判据

假设：训练期source-only学习的轻型FSRG/RDHA域变换能提升目标域表征；D92-lite仅保留对K5新类注册有直接作用的局部—全局头，使联合臂在不牺牲旧类总正确数的前提下提高新旧调和性能。

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

不设0.5pp门，不运行588/fresh63/125矩阵，不按局部性能提前停止。候选在完整S0后若方向门不成立即关闭，不调参复活。

## 3.本地版本、改动与验证

|项|值|
|---|---|
|核心实现|`3458ecba`|
|NumPy2兼容|`3f025ffd`|
|历史receipt兼容修复|`675beef1`|
|method lock|`configs/d127_joint_s0_method_lock_20260803.json`，SHA256`7b8df3c029d8096033b9a39734d563452f1f9b4bcb6737ade63821fb4786a650`|
|修复文件|`stage2_d127_phase1_release.py`：读取固定历史seal，不再与当前运行时代码hash比较；测试与追踪文档同步更新|
|修复边界|仍校验completion marker、文件/数组/内容/physical roots、协议、场景、禁止访问字段及历史execution内部闭合|
|聚焦验证|历史loader 15项通过；修复前全D127 100项通过；`py_compile`与`git diff --check`通过|
|独立复核|真实r7 receipt的35项callable集合双向差集为空；`P0=0,P1=0,RELEASE_READY`|

关键本地文件SHA256：

|文件|SHA256|
|---|---|
|`code/cvsrffi/stage2_d127_phase1_release.py`|`930d468ba673c21167a375da9032252ca7d980ce8af403828ba85b8d1f8cb45e`|
|`tests/test_stage2_d127_phase1_release.py`|`14983e809c72b3f4e423aeb60e90c19e86d2db8346cc08dcfbef6d46d768fac7`|
|`analysis/d127_historical_receipt_loader_traceability_20260803.md`|`ec7ca4a8e0549014e242546c0ab365b10f8613ea020f004f3734944828fc596c`|

## 4.固定资产

|资产|路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|selected IQ receipt|同目录`d106_ls_received_iq.receipt.json`|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|L_s join|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|
|D92 manifest|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json`|`b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c`|
|Target25 context|runner同步到r3 input|`e3cf5b15e29d5d907889874b1517da1ad77e5fa81085ed074d4c196af71830ba`|

## 5.N607交接

|字段|冻结值/待回填|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d127_joint_s0_s713102_20260803_r3`，必须首次创建且不可覆盖|
|commit|`675beef1`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|CWD|`$RUN/source`|
|GPU|Phase1与target worker：A→GPU0、B→GPU1、C→GPU2；每进程内部`cuda:0`|
|执行顺序|直连预检→sync/hash/compile/help→真实r7 receipt只读source-loader smoke→fresh prepare→三候选Phase1→merge asset→三candidate-worker→merge paired→open→truth-assets→score|
|日志/output/PID|`PENDING`，唯一runner回填|
|预期artifact|3个单候选bundle、merged bundle、prepared plan/prefix receipt、3个worker prediction、paired prediction、truth-open、truth/formal receipt、score|

Phase1冻结命令模式：

```text
CUDA_VISIBLE_DEVICES=<0|1|2> $PY $RUN/source/code/scripts/build_d127_phase1_assets.py --candidate-id <A|B|C完整ID> --output-dir $RUN/assets/<candidate> --method-lock $RUN/input/d127_joint_s0_method_lock_20260803.json --method-lock-sha256 7b8df3c029d8096033b9a39734d563452f1f9b4bcb6737ade63821fb4786a650 --selected-iq-archive <第4节selected IQ> --selected-iq-archive-sha256 e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede --selected-iq-receipt <第4节receipt> --selected-iq-receipt-sha256 a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59 --ls-label-join-archive <第4节L_s join> --ls-label-join-archive-sha256 dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d --checkpoint <第4节checkpoint> --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --device cuda:0
```

停止规则仅为P0协议/安全违规，或至少两个不同任务/row在预测前产生相同确定性异常指纹。不得因accuracy、H、floor或forgetting停止。不授权自动retry/restart；技术失败必须保留partial artifact并使用新run ID。r2的prepared artifact不得复用。

## 6.结果表与完成判定

|candidate|receiver|scene|K|arm|old_before|old_after|seen_new|H|old_floor|forgetting|total_correct|verdict|
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|`PENDING`|

完成后必须记录完整同row表、池化G1/G2/G3、候选关闭/晋级决定、异常、artifact SHA、PID/GPU/SSH清理。当前没有新性能结果。
