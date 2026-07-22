# SVRN-qKNN-BCRR/r3 K5 held发布报告

- run_id:`svrn_qknn_bcrr_k5_held_r2_165ca031_20260723`
- candidate/revision:`SVRN-qKNN-BCRR/r3`
- lifecycle:`LOCAL_VERIFIED / RELEASE_PREREGISTERED`
- operator:主agent`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- method Git commit:`165ca03133a8fc724ecccd37e4a55e09a0596dff`
- parent method commit:`922293b1cc2e15a2f595fc124074bae217ae427e`
- protocol:`p2_min_v1 / VALIDATED_ONCE`；复用GEOFF/r8，不重验数据

## 1.目标、矩阵与假设

在receiver=`1-1`、K=5、6个pseudo-new×3个`leo_*_weak`场景的18个slice上产生72个同row分数。`M0`为基础qKNN；`M_DA`为SVRN＋qKNN；`M_OTHER`为qKNN＋连续BCRR；`M_JOINT`为SVRN＋qKNN＋BCRR。

`F=(1-ω)N(Q)+ωN(B)`，`0≤ω≤0.5`，`ω_q=floor(254ω*)/254`。raw/SVRN独立使用同物理ID LOO双向cross-view逐类CE安全集；query不更新η、ω、bank、温度、回退或任何状态，每个query独立面对全部注册类。

本run唯一技术delta：当`ω_q=0`时，部署态持久化全零qint8 BCR codes与合法正fp16 scale，以零teacher/student通过原INT8审计；当`ω_q>0`时，量化、`0.995`一致性门、large-margin flip门、receipt、融合和资源公式均不变。该delta不创造性能收益，只消除inactive residual导致的技术崩溃。

## 2.r1失败与修订证据

父run`svrn_qknn_bcrr_k5_held_r1_922293b1_20260723`在direct N607/GPU0/PID`520750`自然退出，exit=`1`，prediction=`0`、score=`0`，裁决为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。完整日志首错为`BCR INT8 teacher gate failed`。

只读复算72个C5/C6×raw/SVRN branch后，仅2个分支失败，均为`14-10/leo_low_elev_weak/C5`，且`eta=0、omega_q=0`、agreement=`0.96`、1个小margin flip、large-margin flip=`0`；所有`omega_q>0`分支通过原硬门。r3真实r8 support-only复测为72/72闭合、44 active、28 inactive、failure=`0`；原两支变为zero codes、positive scale、agreement=`1.0`、flip=`0`、error=`0`，query fit=`0`。

## 3.本地门

|项目|结果|
|---|---|
|`ssr-gpu` py_compile|PASS|
|聚焦协议/状态/四臂测试|`9 passed`|
|真实r8 72-branch无query smoke|PASS；44 active/28 inactive；失败0|
|独立设计监督|`MERGE / P0=0 / P1=0`|
|独立代码review|`MERGE / P0=0 / P1=0`|
|Git状态|方法提交后clean；发布合同另行提交|

本地smoke发现18行的`selected_eta`均为0，这只是一项support-only可证伪信号，不是prediction或性能结论；本run仍必须生成完整同row结果。

## 4.发布源与输入绑定

|artifact|SHA256/说明|
|---|---|
|源码ZIP|`E:/type10-7/code/snapshots/svrn_qknn_bcrr_k5_held_r2_165ca031_20260723/source_165ca031.zip`|
|ZIP SHA256/大小/entries|`f929e1824cdfdab3887de29a562a9eda92a1d21d7bac5e9aa163020d82f35730`；33,318,164B；4,478|
|ZIP唯一前缀|`source_165ca031/`；无绝对路径、`..`或反斜杠；integrity PASS|
|ZIP内core|`aa5401306cab361cdb06a41b7c11af3dc8b1aea0a00fe9e75b475c5d283deaf4`|
|ZIP内held|`eb6231732df8952651dca5903c5d2b8dc27b9a10d4bc6d96e6a15d3d12cad236`|
|ZIP内test|`c1f990dcf8a7e9c62c0debe21d34603e3f284ea9598a68f77123b90d4e741e81`|
|wrapper SHA256|`6cc02e6a109127c370e17a162356e3adc39a4a543fb5af3181227b3ca3c55c3e`；`bash -n` PASS|
|r8 parity|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|
|r8 archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|r8 manifest|`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|r8 coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|

## 5.N607冻结执行合同

- remote root:`/home/szu2070436088/2510044040/CV-SincNet/runs/svrn_qknn_bcrr_k5_held_r2_165ca031_20260723`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- source root:`<remote-root>/source_165ca031`
- r8 root:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8`
- 唯一启动:`CUDA_VISIBLE_DEVICES=<runner-selected> nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &`
- retry:`NO`；不得远端编辑、调参、重建数据、复用旧run、kill/restart或启动125。
- 预期artifact:`packet.json`、`truth.json`、`query.npz`、`prediction.json`、`score.json`、`sha256sums.txt`、`complete.marker`、PID、exit和完整log。
- 预期数量:prediction slice=`18`；score row=`72`；arm=`M0/M_DA/M_OTHER/M_JOINT`；candidate artifact=`SVRN-qKNN-BCRR/r3-held`。

唯一runner必须完成direct preflight、remote root不存在/GPU/进程/磁盘检查、ZIP/wrapper/source/r8 SHA、单根布局、远端`py_compile`/import、`bash -n`，然后只启动一次并用短连接监控、完整回收artifact。无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

## 6.性能停止门

必须同row报告old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、wrong→correct/correct→wrong、逐类、scene、η/ω、邻居变化、量化margin、state、MAC、mean/P95和VRAM。

- `M_DA`相对`M0`净正确>0，old/new净变化均非负，且不损害保护指标；
- `M_OTHER`具有独立正收益且不损害保护指标；
- `M_JOINT.H`严格高于两个单组件；
- mean`I_syn(H)>0`、严格正协同≥9/18 slice、正scene均值≥2/3；
- JOINT不损害old-after、seen-new、floor、min-old、min-new或forgetting；
- 量化、state、MAC、时延和显存门通过。

任一失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不得运行125。support fit、archive、coverage、进程完成和代码测试均不是性能成功。

## 7.完成后更新

待runner回填route、GPU/PID/exit、远端parity、artifact数量与SHA；artifact完整后立即独立复算72个score row并形成同row性能表与最终裁决。
