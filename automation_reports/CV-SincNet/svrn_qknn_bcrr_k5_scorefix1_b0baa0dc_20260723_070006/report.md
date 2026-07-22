# SVRN-qKNN-BCRR/r3评分闭合发布报告

- run_id:`svrn_qknn_bcrr_k5_scorefix1_b0baa0dc_20260723_070006`
- candidate/revision:`SVRN-qKNN-BCRR/r3-scorefix1`
- lifecycle:`LOCAL_VERIFIED / RELEASE_PREREGISTERED`
- operator:主agent`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- method Git commit:`b0baa0dc328ec7fe7a8d5870f35bdee256c9b686`
- runner wrapper Git commit:`7ac9805d58860da0b98512f695c14e357c0182cb`
- frozen parent prediction run:`svrn_qknn_bcrr_k5_held_r2_165ca031_20260723`
- protocol:`p2_min_v1 / VALIDATED_ONCE`；复用GEOFF/r8及父run封存artifact，不重验数据

## 1.目标与唯一delta

父run已生成18个完整prediction，但旧评分器把canonical JSON按键排序后的mapping迭代顺序误当成四臂语义顺序，因此score阶段技术失败。完整只读检查证明36个before/after mapping均精确包含`M0/M_DA/M_OTHER/M_JOINT`，prediction`COMMIT`、row/query绑定、logit shape/finite、argmax和neighbor receipt均闭合。

本run唯一delta是评分器将四臂检查从tuple顺序等值改为精确键集合等值，并继续按冻结`ARMS`顺序评分。缺臂或多臂即使重签`COMMIT`仍fail-closed。方法、prediction、truth、query、qKNN、SVRN、BCRR、状态、参数、资源公式和停止门均不变。

本run只读父run的四个固定artifact，生成正式`score.json`。不重新build、predict、选择方法、调参或更新任何状态。

## 2.本地最低门

|项目|结果|
|---|---|
|`ssr-gpu` py_compile|PASS|
|聚焦协议/状态/四臂测试|`9 passed`|
|canonical往返评分|PASS；72行|
|删臂/多臂重签负例|PASS；均拒绝|
|父prediction直接评分|PASS；18个slice、72个score row|
|独立review|`MERGE / P0=0 / P1=0`|
|方法Git状态|提交后clean|

一般性的内部query SHA增强被独立review记为P2，不阻塞本run：score不读取query；wrapper在评分前对父packet/truth/query/prediction四个绝对路径逐一校验固定SHA，报告同时封存四SHA。

## 3.源码与wrapper

|artifact|SHA256/说明|
|---|---|
|源码ZIP|`E:/type10-7/code/snapshots/svrn_qknn_bcrr_k5_scorefix1_b0baa0dc_20260723_070006/source_b0baa0dc.zip`|
|ZIP SHA256/大小/entries|`21538751f8e1cdc53d0cb127588f0a239ed9250890eba89ea1b49b93d96ed3ef`；33,326,077B；4,481|
|ZIP根与安全性|唯一`source_b0baa0dc/`；绝对路径、`..`和反斜杠危险项均为0|
|ZIP内core|`aa5401306cab361cdb06a41b7c11af3dc8b1aea0a00fe9e75b475c5d283deaf4`|
|ZIP内held|`1e71fd2934360a3d3f1082e4a3841bc307334807bc3496455b7c9b29d2366366`|
|ZIP内test|`ef0fee40e393b3917e83d0dc955053f599989067b500c55253a7a67cdad2445a`|
|wrapper SHA256|`f1fd6a0381b89b9f2c38c84d4db4637db846e72dbf804e8e091bec39f9268892`|
|wrapper语法|`bash -n`PASS|

## 4.父artifact冻结绑定

父remote root:
`/home/szu2070436088/2510044040/CV-SincNet/runs/svrn_qknn_bcrr_k5_held_r2_165ca031_20260723`

|输入|外部SHA256|内部绑定|
|---|---|---|
|`output/packet.json`|`ef15a8488d40ac70d129db9ac15c796418b4afe5fa64624883eab0f66fd4e95b`|packet SHA:`b503fc1d60e50785d9c2eec941e1f12b495aaf45a193b7f99857e80b46ee2b31`|
|`output/truth.json`|`9745068bc5961ebe90f6305c672cacc8ce338d745579e1c97d4ea503cbd06d8`|truth SHA:`637e845fec201627118181a5eb256861b86e76880c101d1b6a5452563cce64b4`|
|`output/query.npz`|`be089f42be790a73cd7a95d68cb13956a64735019b10f6cd4ba32199c33c56c9`|仅做父集合外部SHA绑定；score不读取|
|`output/prediction.json`|`0f9313e632884e9987caaa262e2e7d261338bfe9b7f84beae85753571b72e06e`|COMMIT:`2524a1aa291cb05ed055625c496f8abc12fc692b57736070334b65ce1c68211a`|

## 5.N607冻结执行合同

- remote root:`/home/szu2070436088/2510044040/CV-SincNet/runs/svrn_qknn_bcrr_k5_scorefix1_b0baa0dc_20260723_070006`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- source root:`<remote-root>/source_b0baa0dc`
- 唯一启动:`CUDA_VISIBLE_DEVICES=<runner-selected> nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &`
- retry:`NO`；不得远端编辑、重建prediction、调参、kill/restart或运行125
- 预期artifact:`score.json`、`parent_sha256sums.txt`、`sha256sums.txt`、`complete.marker`、PID、exit和完整log
- 预期数量:父prediction slice=18；新score row=72；arm=`M0/M_DA/M_OTHER/M_JOINT`
- 预期marker:`SVRN_QKNN_BCRR_K5_SCOREFIX1_ARTIFACTS_COMPLETE`

唯一runner必须完成direct preflight、remote root不存在/GPU/进程/磁盘检查、ZIP/wrapper/source/父四artifact SHA、单根布局、远端`py_compile`/import和`bash -n`，随后只启动一次并以短连接监控、完整回收artifact。

## 6.正式性能门

正式score完成后必须同row报告old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、wrong→correct/correct→wrong、逐类、scene、η/ω、邻居变化、量化margin、state、MAC、mean/P95和VRAM。

- `M_DA`相对`M0`净正确>0，old/new净变化均非负，且保护指标不退化；
- `M_OTHER`独立正收益且保护指标不退化；
- `M_JOINT.H`严格高于两个单组件；
- mean`I_syn(H)>0`、严格正协同≥9/18 slice、正scene均值≥2/3；
- JOINT不损害old-after、seen-new、floor、min-old、min-new或增加forgetting；
- 量化、state、MAC、时延和显存门通过。

任一失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不得运行125。父prediction的本地只读评分只能作为发布前诊断；本run的正式score、完整log和匹配SHA闭合前不作性能裁决。

## 7.完成后更新

runner回填route、GPU/PID/exit、远端parity、artifact数量与SHA。artifact完整后由独立分析agent复算全部72行、同row表、18个slice、3个scene、逐类、transition、量化和资源证据，并给出正式裁决。
