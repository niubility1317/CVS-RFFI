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

## 8.runner执行记录（2026-07-23）

- route/preflight:`direct N607`；`tools/n607_ssh_preflight.ps1`通过，项目根、远端Python与8张GPU可见；每个短SSH/SCP任务后均确认本地无`ssh.exe`和到N607/Lab bridge的ESTABLISHED:22连接。
- pre-launch:目标root先验不存在；磁盘可用`7.5TB`；无N607 GPU计算进程。父`packet/truth/query/prediction`外部SHA均与第4节一致；truth内部SHA=`637e845fec201627118181a5eb256861b86e76880c101d1b6a5452563cce64b4`，prediction COMMIT=`2524a1aa291cb05ed055625c496f8abc12fc692b57736070334b65ce1c68211a`。
- remote parity:创建唯一root后只SCP冻结ZIP与wrapper。远端ZIP SHA=`21538751f8e1cdc53d0cb127588f0a239ed9250890eba89ea1b49b93d96ed3ef`、wrapper SHA=`c72510be802254a969494dc4fb7c99a750f748b94eb49a33a81b8999ad0c097b`；布局唯一`source_b0baa0dc/`；core/held/test SHA与第3节一致；`bash -n`、远端`py_compile`及import通过。
- 唯一启动:GPU=`0`；PID=`559507`；CWD=`/home/szu2070436088/2510044040/CV-SincNet/runs/svrn_qknn_bcrr_k5_scorefix2_b0baa0dc_20260723_071226`；Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；命令=`CUDA_VISIBLE_DEVICES=0 nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &`。无retry、restart、kill、参数/数据/方法变更。
- natural exit:`pipeline.exit=0`；PID已自然消失；完整log含`parent_prediction_slices=18`、`score_metric_rows=72`与完成行。marker=`SVRN_QKNN_BCRR_K5_SCOREFIX2_ARTIFACTS_COMPLETE`。
- retrieved:`retrieved/score.json`、`parent_sha256sums.txt`、`sha256sums.txt`、`complete.marker`、`pipeline.log`、`pipeline.pid`、`pipeline.exit`。`score.json` SHA=`c3ac8b462009675e316929e82df58d6c53dd47ec4bf51ef426c2f96da8b738fe`，与远端`sha256sums.txt`一致；本地JSON复核为18个唯一row ID、72个metrics、`M0/M_DA/M_OTHER/M_JOINT`四arm，内部truth/COMMIT一致。
- lifecycle:`LOCAL_VERIFIED -> LANDED -> RUNNING -> ARTIFACTS_COMPLETE`。评分artifact的`decision.verdict=COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；runner不作性能推广，交由独立分析agent完成同row正式解释。

## 9.独立性能复算与正式裁决

### 9.1 Artifact完整性

独立分析从父prediction＋truth逐样本重算，而非照抄score decision。父四artifact、新score外部SHA、packet/truth/prediction内部seal、truth绑定、prediction COMMIT、exit0、marker、4行完整log和schema链全部PASS。query只含1105个唯一opaque ID与`float32[1105,160]`的`z_id`，全部finite并与packet query并集一致。

18个唯一pseudo-new×scene row、每arm 18行、共72行闭合。独立解码144个logit块、291,720个float32元素及144份neighbor receipt；shape、finite、argmax、receipt hash、support物理ID和四臂隔离全部PASS。独立复算全部标量、逐类和transition，与score最大绝对差=`2.220446049250313e-16`，无大于`1e-12`的差异。

### 9.2 四臂mean

|arm|old-before|old-after|old gain|seen-new|H|BA|floor|min-old|min-new|forgetting|old→new|new→old|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|M0|0.817038|0.785392|-0.031647|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|
|M_DA|0.817038|0.785392|-0.031647|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|
|M_OTHER|0.820042|0.797489|-0.022553|0.793192|0.764833|0.793192|0.476318|0.493128|0.793192|0.022553|0.040127|0.206808|
|M_JOINT|0.820042|0.797489|-0.022553|0.793192|0.764833|0.793192|0.476318|0.493128|0.793192|0.022553|0.040127|0.206808|

关键matched delta：

|比较|Δold-after|Δseen-new|ΔH|ΔBA|Δfloor|Δmin-old|Δmin-new|Δforgetting|Δold→new|Δnew→old|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|M_DA−M0|0|0|0|0|0|0|0|0|0|0|
|M_OTHER−M0|+0.012098|+0.011408|+0.017067|+0.011408|+0.056410|+0.046141|+0.011408|-0.009093|-0.002469|-0.011408|
|M_JOINT−M_OTHER|0|0|0|0|0|0|0|0|0|0|

OTHER是真实独立正收益来源；DA没有产生任何收益，JOINT没有超出OTHER。

### 9.3 18个同row结果

|pseudo-new|scene|H0|HDA|HOTHER|HJOINT|I_syn|J old-after|J seen-new|J floor|J min-old|J min-new|J forgetting|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|14-10|clear|0.9115|0.9115|0.9184|0.9184|0|0.8875|0.9516|0.7794|0.7794|0.9516|-0.0032|
|14-10|low-elev|0.7853|0.7853|0.7853|0.7853|0|0.6615|0.9661|0.2034|0.2034|0.9661|0.0062|
|14-10|rain|0.8369|0.8369|0.8555|0.8555|0|0.7621|0.9750|0.4462|0.4462|0.9750|0|
|14-7|clear|0.8522|0.8522|0.8655|0.8655|0|0.9132|0.8226|0.7794|0.7794|0.8226|0.0096|
|14-7|low-elev|0.3244|0.3244|0.3244|0.3244|0|0.8012|0.2034|0.2034|0.2264|0.2034|0.0994|
|14-7|rain|0.7415|0.7415|0.7404|0.7404|0|0.8090|0.6825|0.4462|0.4462|0.6825|-0.0069|
|20-15|clear|0.9169|0.9169|0.9239|0.9239|0|0.8857|0.9655|0.7794|0.7794|0.9655|0.0095|
|20-15|low-elev|0.7802|0.7802|0.7802|0.7802|0|0.6396|1.0000|0.2034|0.2034|1.0000|0.0552|
|20-15|rain|0.8243|0.8243|0.8453|0.8453|0|0.7404|0.9848|0.4462|0.4462|0.9848|0.0211|
|20-19|clear|0.8911|0.8911|0.9176|0.9176|0|0.8889|0.9483|0.7794|0.7794|0.9483|0.0381|
|20-19|low-elev|0.3516|0.3516|0.3516|0.3516|0|0.7866|0.2264|0.2034|0.2034|0.2264|0.1280|
|20-19|rain|0.4202|0.4202|0.5884|0.5884|0|0.8636|0.4462|0.4462|0.6825|0.4462|0.0035|
|6-15|clear|0.9065|0.9065|0.9134|0.9134|0|0.8896|0.9385|0.7794|0.7794|0.9385|0.0032|
|6-15|low-elev|0.7732|0.7732|0.7732|0.7732|0|0.6548|0.9437|0.2034|0.2034|0.9437|0|
|6-15|rain|0.8075|0.8075|0.8263|0.8263|0|0.7604|0.9048|0.4462|0.4462|0.9048|0.0660|
|8-20|clear|0.8403|0.8403|0.8458|0.8458|0|0.9246|0.7794|0.7794|0.8226|0.7794|0.0066|
|8-20|low-elev|0.7214|0.7214|0.7214|0.7214|0|0.7016|0.7424|0.2034|0.2034|0.7424|0|
|8-20|rain|0.7747|0.7747|0.7904|0.7904|0|0.7845|0.7963|0.4462|0.4462|0.7963|-0.0303|

全部18个slice的`I_syn=0`，严格正slice=`0/18`。

### 9.4 Scene、逐类与transition

|scene|H0|HDA|HOTHER|HJOINT|mean I_syn|J old-after|J seen-new|J floor|J min-old|J forgetting|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|0.886422|0.886422|0.897442|0.897442|0|0.898241|0.900977|0.779412|0.786607|0.010642|
|leo_low_elev_weak|0.622687|0.622687|0.622687|0.622687|0|0.707559|0.680332|0.203390|0.207227|0.048139|
|leo_rain_weak|0.734188|0.734188|0.774371|0.774371|0|0.786667|0.798267|0.446154|0.485551|0.008879|

正scene均值=`0/3`。held receiver固定为`1-1`、K=`5`；packet无独立可变seed字段，复现身份由packet SHA封存。

|真实类|M0|M_DA|M_OTHER|M_JOINT|ΔJOINT−M0|
|---|---:|---:|---:|---:|---:|
|14-10|0.964238|0.964238|0.964238|0.964238|0|
|14-7|0.574709|0.574709|0.569503|0.569503|-0.005206|
|20-15|0.983455|0.983455|0.983455|0.983455|0|
|20-19|0.466630|0.466630|0.540282|0.540282|+0.073652|
|6-15|0.928962|0.928962|0.928962|0.928962|0|
|8-20|0.772711|0.772711|0.772711|0.772711|0|

OTHER/JOINT的主要收益集中于`20-19`，同时`14-7`小幅退化；六个pseudo-new分组的mean I_syn均为0。

|arm|changed|wrong→correct|correct→wrong|net|wrong→wrong|old rescue/harm/net|new rescue/harm/net|
|---|---:|---:|---:|---:|---:|---|---|
|M0|0|0|0|0|0|0/0/0|0/0/0|
|M_DA|0|0|0|0|0|0/0/0|0/0/0|
|M_OTHER|144|96|18|+78|30|80/15/+65|16/3/+13|
|M_JOINT|144|96|18|+78|30|80/15/+65|16/3/+13|

### 9.5 Mechanism、量化、资源与coverage

C5/C6的η均只取0，非零row均为`0/18`。C5 fallback为`class_direction_below_eta0`2行、`direction_disagreement`11行、`selected_identity`5行；C6为`direction_disagreement`12行、`selected_identity`6行。DA neighbor order变化before/after均为0。raw与SVRN的ω逐row相同；C5、C6分别有10/18、12/18个非零ω，均值0.275591/0.330709，范围`[0,0.496063]`。BCRR按设计只改logit、不改qKNN neighbor order。

|项目|结果|门|
|---|---:|---|
|qKNN/BCR top1 agreement最小值|1.0/1.0|PASS|
|qKNN margin flip/BCR large-margin flip|0/0|PASS|
|qKNN/BCR最大logit误差|0.270576/0.00329032|PASS|
|query-fit rows/state updates|0/0|PASS|
|state数量/最大字节|36/33,598B|PASS，预算262,144B|
|persistent FP32 sidecar/optimizer steps|0/0|PASS|
|build mean/P95/max|243.456/279.114/280.991ms|PASS，预算30s|
|build MAC最大值|799,200|PASS，预算100M|
|四臂query MAC/样本最大值|11,520|PASS，预算20M|
|prediction mean/P95/max|395.666/446.713/447.306ms|PASS，预算5s/row|
|aggregate four-arm MAC|76,377,600|账本匹配|
|backend/峰值显存|numpy_cpu/0B|PASS|

GEOFF/r8 coverage SHA仍为`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`：8400行，6类、7receiver、4day、3scene，168个receiver×day×class cell，zero=0、min=32、K5最小余量27；`feature_arrays_read=[]`、`held_fold_selected=false`。本run每row C5/C6 support为25/30条，support-query overlap为0，query-fit为0。

### 9.6 Promotion gate与裁决

|冻结门|结果|证据|
|---|---|---|
|artifact/72行完整性|PASS|全部SHA、seal、schema、logit、neighbor、exit/marker闭合|
|M_DA独立正收益|**FAIL**|decision change=0，net correct=0|
|M_DA old/new净变化及保护|PASS|全部为0/不退化|
|M_OTHER独立正收益及保护|PASS|96 rescue、18 harm、net=+78，old/new均正|
|JOINT.H严格高于两单臂|**FAIL**|0.764833257，与M_OTHER相等|
|mean I_syn>0|**FAIL**|0|
|正slice≥9/18|**FAIL**|0/18|
|正scene≥2/3|**FAIL**|0/3|
|JOINT保护|PASS|与M_OTHER相同并优于或等于M_DA|
|量化/资源/coverage|PASS|全部硬门通过|

最终状态：`ARTIFACTS_COMPLETE -> ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。不得进入125。

失败是确定的机制结果，不是artifact或scorer问题：DA在18/18行全部选择identity，没有任何邻居或预测变化；JOINT完全退化为OTHER，因此`I_syn=0`。OTHER虽取得真实独立净收益`+78`，但不能替代失败的DA与协同门。
