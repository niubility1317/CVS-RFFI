# SVRN-qKNN-BCRR/r3评分闭合第二次发布报告

- run_id:`svrn_qknn_bcrr_k5_scorefix2_b0baa0dc_20260723_071226`
- candidate/revision:`SVRN-qKNN-BCRR/r3-scorefix2-release`
- lifecycle:`LOCAL_VERIFIED / RELEASE_PREREGISTERED`
- operator:主agent`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- method Git commit:`b0baa0dc328ec7fe7a8d5870f35bdee256c9b686`
- runner wrapper Git commit:`56e746ea7f7ed406336c4c3f2264e3c132d80ea6`
- frozen parent prediction run:`svrn_qknn_bcrr_k5_held_r2_165ca031_20260723`
- protocol:`p2_min_v1 / VALIDATED_ONCE`；复用GEOFF/r8及父run封存artifact，不重验数据

## 1.父发布阻断与唯一修复

父score-only发布`svrn_qknn_bcrr_k5_scorefix1_b0baa0dc_20260723_070006`在direct preflight后、任何远端写入前被安全阻断：父`truth.json`实测外部SHA为`9745068bc5961ebe90f6305c672cacc8ce338d745579e1c97d4ea503cb3d06d8`，而旧wrapper误录为少一个`3`的值。其余packet/query/prediction SHA均匹配。

父发布未创建remote root、未SCP、未解包、未启动，无GPU/PID/exit/log/score/marker，状态为`BLOCKED_PRE_LAUNCH / NO_PERFORMANCE_RESULT`。本run使用全新不可覆盖run ID，唯一delta是把wrapper/report中的truth外部SHA改为本地回收文件与N607实测一致的64位值；方法、源码ZIP、父四artifact、内部truth SHA、prediction COMMIT、评分规则和停止门不变。

## 2.本地最低门

|项目|结果|
|---|---|
|父truth本地SHA复核|`9745068bc5961ebe90f6305c672cacc8ce338d745579e1c97d4ea503cb3d06d8`；64位|
|N607只读父truth SHA|与本地一致|
|wrapper根/Git镜像|字节一致|
|wrapper语法|`bash -n`PASS|
|方法专项|`ssr-gpu`下py_compile PASS、`9 passed`；方法commit未变|
|独立review|`MERGE / P0=0 / P1=0`|

本run仍只读父run四个固定artifact并生成正式`score.json`；不重新build、predict、选择方法、调参或更新任何状态。

## 3.源码与wrapper

|artifact|SHA256/说明|
|---|---|
|源码ZIP|`E:/type10-7/code/snapshots/svrn_qknn_bcrr_k5_scorefix1_b0baa0dc_20260723_070006/source_b0baa0dc.zip`|
|ZIP SHA256/大小/entries|`21538751f8e1cdc53d0cb127588f0a239ed9250890eba89ea1b49b93d96ed3ef`；33,326,077B；4,481|
|ZIP根与安全性|唯一`source_b0baa0dc/`；绝对路径、`..`和反斜杠危险项均为0|
|ZIP内core|`aa5401306cab361cdb06a41b7c11af3dc8b1aea0a00fe9e75b475c5d283deaf4`|
|ZIP内held|`1e71fd2934360a3d3f1082e4a3841bc307334807bc3496455b7c9b29d2366366`|
|ZIP内test|`ef0fee40e393b3917e83d0dc955053f599989067b500c55253a7a67cdad2445a`|
|wrapper SHA256|`c72510be802254a969494dc4fb7c99a750f748b94eb49a33a81b8999ad0c097b`|

## 4.父artifact冻结绑定

父remote root:
`/home/szu2070436088/2510044040/CV-SincNet/runs/svrn_qknn_bcrr_k5_held_r2_165ca031_20260723`

|输入|外部SHA256|内部绑定|
|---|---|---|
|`output/packet.json`|`ef15a8488d40ac70d129db9ac15c796418b4afe5fa64624883eab0f66fd4e95b`|packet SHA:`b503fc1d60e50785d9c2eec941e1f12b495aaf45a193b7f99857e80b46ee2b31`|
|`output/truth.json`|`9745068bc5961ebe90f6305c672cacc8ce338d745579e1c97d4ea503cb3d06d8`|truth SHA:`637e845fec201627118181a5eb256861b86e76880c101d1b6a5452563cce64b4`|
|`output/query.npz`|`be089f42be790a73cd7a95d68cb13956a64735019b10f6cd4ba32199c33c56c9`|仅做父集合外部SHA绑定；score不读取|
|`output/prediction.json`|`0f9313e632884e9987caaa262e2e7d261338bfe9b7f84beae85753571b72e06e`|COMMIT:`2524a1aa291cb05ed055625c496f8abc12fc692b57736070334b65ce1c68211a`|

## 5.N607冻结执行合同

- remote root:`/home/szu2070436088/2510044040/CV-SincNet/runs/svrn_qknn_bcrr_k5_scorefix2_b0baa0dc_20260723_071226`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- source root:`<remote-root>/source_b0baa0dc`
- 唯一启动:`CUDA_VISIBLE_DEVICES=<runner-selected> nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &`
- retry:`NO`；不得远端编辑、重建prediction、调参、kill/restart或运行125
- 预期artifact:`score.json`、`parent_sha256sums.txt`、`sha256sums.txt`、`complete.marker`、PID、exit和完整log
- 预期数量:父prediction slice=18；新score row=72；arm=`M0/M_DA/M_OTHER/M_JOINT`
- 预期marker:`SVRN_QKNN_BCRR_K5_SCOREFIX2_ARTIFACTS_COMPLETE`

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
