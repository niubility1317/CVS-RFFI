# CID-BPP K5 held技术修订性能run

## 1.身份与状态

- run ID：`cid_bpp_k5_held_r2_e3aa2da5_20260723`
- 时间：`2026-07-23 +08:00`
- operator：主agent；唯一N607 runner=`gpt-5.6-terra high`
- candidate：`JOINT-CID-BPP/r0-spike-tech1`
- scope：`PHASE1_HELD_PROXY_NON_PROMOTABLE`
- 当前状态：`ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- 方法Git commit：`e3aa2da5af520e493d40ec343b913ce24e7629dd`
- 原失败run：`cid_bpp_k5_held_r1_30c11b75_20260723`，direct/GPU0/PID458576/exit1/prediction0

## 2.目标与唯一delta

本run只验证fallback-aware verifier技术修订，然后立即完成原冻结的18个K5 held slice、72个同row四臂prediction/score。唯一delta是：非fallback C-id继续强制support provenance；`jackknife_no_direction/jackknife_overlap`强制canonical identity的null provenance、rank0及identity metric receipt。fit、family、metric、BPP、四臂公式、support/query、量化和资源门均不变。

为避免重复计算或重新选择，直接复用原run在失败前已封存的`packet.json/truth.json/query.npz`；旧run失败发生在predict前，因此这些artifact尚未产生任何性能反馈。本run不重建packet，不重选family，也不读取outer truth进行选择。

## 3.不可变输入

`p2_min_v1`及`VALIDATED_ONCE`继续复用；received IQ、物理ID、receiver/TX、场景、K、support/query和schema均未改变，不重复数据验证。

|输入|SHA256/状态|
|---|---|
|原run packet file|`e7132ea454927782acf976c5baa0ed960c37f722791ca45f6c1c7e40d6f1bcc8`|
|packet internal SHA|`b7e56a7b307d91bfb5062519d921e59c35c79ce5f372462218c1a8db5d8a0144`|
|原run truth|`87c37bc8af7d83400a641e87b81602ac7eeb3e8e12b2dc8e26e1f8b597a81a4e`|
|原run query|`be089f42be790a73cd7a95d68cb13956a64735019b10f6cd4ba32199c33c56c9`|
|r8 parity|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|
|r8 archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`，8400 rows|
|r8 manifest|`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|r8 coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|coverage摘要|6类/7receiver/4day/3scene；168 cells；zero=0；min=32；K5余量27；`feature_arrays_read=[]`|

## 4.本地门与独立review

`ssr-gpu`下三文件`py_compile`、专项＋相邻回归`18 passed`、`git diff --check`通过。真实r8 archive的失败同型25条C4 support无query smoke复现overlap=`0.4833083158142645`、`jackknife_overlap`、rank0、null provenance和identity receipt一致。

独立review对代码delta裁决`P0=0，P1=1→REVISE`；唯一P1是N607生成的其他低秩metric receipt无法在本机数值环境复算，必须在N607同环境、launch前用当前Git提交完整执行原packet的`_verify_packet`。这不是方法或协议修改，不允许为此改代码、放宽receipt或重建数据。

## 5.P1远端封存门

runner同步并验证当前Git source后，必须在启动wrapper前对原packet执行support-only完整`_verify_packet`，以noclobber写出：

`preflight_packet_verify.json`：

```json
{"method_commit":"e3aa2da5af520e493d40ec343b913ce24e7629dd","packet_file_sha256":"e7132ea454927782acf976c5baa0ed960c37f722791ca45f6c1c7e40d6f1bcc8","packet_sha256":"b7e56a7b307d91bfb5062519d921e59c35c79ce5f372462218c1a8db5d8a0144","status":"PASS","verifier":"cvsrffi.cid_bpp_fixed_held_spike._verify_packet"}
```

预期文件SHA256=`b153049167629c4ccd1932934d5149d1868183dd4dcfb530a6d0df98f70113b3`。只有该SHA精确通过，P1才清零并允许同一runner启动一次；失败则停止为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重试。

## 6.发布源

|artifact|值|
|---|---|
|release contract commit|`77a41240a26851726092f6c5789e394cbe341055`|
|Git archive|`E:/type10-7/code/snapshots/cid_bpp_k5_held_r2_e3aa2da5_20260723/source_e3aa2da5.zip`|
|ZIP SHA256|`0638a944abc939ebdafcf2c57cd5e06e40676113d4a55eefe0c2b771f75166d8`|
|ZIP size/files|`33,268,292 bytes / 4,469 entries`|
|wrapper SHA256|`b132a1c2a929b10b07c0e3dee4d3ae9988dfa84886f20880e410265868da5605`|
|selector raw SHA256|`05958df26be904884b19b2fbbcdcff5c61a78612a5e064f997dfd611901c9a59`|
|held module raw SHA256|`81e397a58c4d63bdd29defd35a98c4843ac4774d16892df279824f14b896d7f1`|
|test raw SHA256|`728c07923fd851735aa5e15dd80539d304feaffc2a6516ab138edee62e336095`|

## 7.N607冻结执行合同

- 新run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cid_bpp_k5_held_r2_e3aa2da5_20260723`
- 原输入root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cid_bpp_k5_held_r1_30c11b75_20260723/output`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：新run root；source：`source_e3aa2da5`
- 唯一启动命令：`CUDA_VISIBLE_DEVICES=<runner-selected> nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 &`
- retry：`NO`；不得运行build、target或125。
- 预期输出：P1 receipt、packet、truth、query、prediction、score、sha256sums、marker、PID、exit和完整log。

runner必须direct preflight、新run root不存在/GPU/进程/磁盘检查、ZIP/wrapper/source/原run inputs/r8 inputs SHA、单根布局、`py_compile`/import、`bash -n`；执行第5节远端无query门并核验receipt SHA；P1清零后仅启动一次，短连接监控并完整回收。禁止远端编辑、调参、重建、retry、kill/restart或125。

## 8.性能与停止条件

完成后必须独立复核18个prediction row、72个score row、logits→argmax、同row四臂、逐类/scene/K/receiver、wrong→correct/correct→wrong、coverage、量化、MAC、CPU mean/P95、VRAM和state bytes。

- 无完整prediction：`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- prediction=18且score=72，但DA零决策变化、JOINT=HEAD、mean`I_syn<=0`或联合伤害old/new/floor/min并增加forgetting：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 只有DA、HEAD各自正收益、JOINT优于两者、`I_syn>0`且全部保护门通过，才允许125稳定性screen。

## 9.N607执行与artifact完整性

- route=`direct`，GPU=`0`，PID=`475079`，exit=`0`，lifecycle=`ARTIFACTS_COMPLETE`。
- P1同环境packet verifier回执=`PASS`，SHA256=`b153049167629c4ccd1932934d5149d1868183dd4dcfb530a6d0df98f70113b3`；唯一P1在launch前清零。
- 完整输出：18个prediction slice、72个score row；marker=`CID_BPP_HELD_R2_ARTIFACTS_COMPLETE`。
- prediction SHA256=`c0b0c29cdc63d8a2fb6c67bedf33d0b9b31172570b961836307fbb4b8dc76441`；score SHA256=`de4f7f860f82dd230f7f74d453360bcfa66dd9a03890bb5b0b2ae0d71e968f5f`。
- 独立复算packet/truth/prediction/score的canonical seal、交叉绑定、144个logit矩阵argmax和72行全部数值；最大绝对差=`0.0`，shape/非有限值/argmax不一致均为0。
- query仅含`query_ids,z_id`，1105个ID唯一且精确覆盖packet query并集；`z_id.shape=(1105,160)`、`float32`、全有限。
- r8 archive/coverage继续复用，没有数据协议输入改变，没有重验数据。archive=`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；coverage=`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`。

## 10.同row四臂主结果

以下均为相同18个K5 held slice的算术均值，不拼接不同run或不同row极值。

|arm|old-before|old-after|old adaptation gain|seen-new|H|BA|floor|min-old|min-new|forgetting|old→new|new→old|I_syn|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|M0|0.817038|0.785392|-0.031647|0.781784|0.747766|0.781784|0.419908|0.446988|0.781784|0.031647|0.042596|0.218216|0.003168|
|M_DA|0.817570|0.785570|-0.031999|0.781784|0.747863|0.781962|0.418967|0.446046|0.781784|0.031999|0.042774|0.218216|0.003168|
|M_HEAD|0.824655|0.804355|-0.020299|0.799331|0.776810|0.799034|0.531354|0.543206|0.799331|0.020299|0.038612|0.200669|0.003168|
|M_JOINT|0.826082|0.806121|-0.019961|0.801428|0.780075|0.801013|0.533770|0.545614|0.801428|0.019961|0.038095|0.198572|0.003168|

关键matched delta：

|比较|Δold-after|Δseen-new|ΔH|ΔBA|Δfloor|Δmin-old|Δforgetting|
|---|---:|---:|---:|---:|---:|---:|---:|
|M_DA−M0|+0.000179|0|+0.000097|+0.000177|-0.000942|-0.000942|+0.000353|
|M_HEAD−M0|+0.018964|+0.017547|+0.029044|+0.017250|+0.111446|+0.096218|-0.011347|
|M_JOINT−M_HEAD|+0.001766|+0.002096|+0.003265|+0.001979|+0.002416|+0.002408|-0.000339|
|M_JOINT−M_DA|+0.020551|+0.019643|+0.032212|+0.019052|+0.114804|+0.099568|-0.012039|

## 11.18个slice完整结果

`low-elev`表示`leo_low_elev_weak`。表内所有J列均来自同一M_JOINT row。

|pseudo-new|scene|H M0|H DA|H HEAD|H JOINT|I_syn|J old-b|J old-a|J new|J BA|J floor|J min-old|J min-new|J forgetting|J old→new|J new→old|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|14-10|clear|0.911482|0.913225|0.895352|0.895352|-0.001743|0.861736|0.858521|0.935484|0.875220|0.705882|0.705882|0.935484|0.003215|0|0.064516|
|14-10|low-elev|0.785292|0.785292|0.783099|0.783099|0|0.658385|0.658385|0.966102|0.679068|0.226415|0.226415|0.966102|0|0.034161|0.033898|
|14-10|rain|0.836934|0.836934|0.890770|0.890770|0|0.855305|0.819936|0.975000|0.843817|0.666667|0.666667|0.975000|0.035370|0.051447|0.025000|
|14-7|clear|0.852238|0.852238|0.843558|0.843558|0|0.900322|0.884244|0.806452|0.875034|0.705882|0.705882|0.806452|0.016077|0.025723|0.193548|
|14-7|low-elev|0.324426|0.324426|0.365161|0.366143|0.000982|0.860248|0.801242|0.237288|0.687883|0.237288|0.264151|0.237288|0.059006|0.111801|0.762712|
|14-7|rain|0.741483|0.741483|0.756757|0.756757|0|0.892361|0.875000|0.666667|0.843817|0.666667|0.753846|0.666667|0.017361|0.027778|0.333333|
|20-15|clear|0.916940|0.916940|0.906323|0.906323|0|0.853968|0.853968|0.965517|0.875034|0.705882|0.705882|0.965517|0|0.031746|0.034483|
|20-15|low-elev|0.780198|0.780198|0.774076|0.781095|0.007019|0.704545|0.652597|0.972603|0.687883|0.237288|0.237288|0.972603|0.051948|0.051948|0.027397|
|20-15|rain|0.824309|0.824309|0.884985|0.884985|0|0.803509|0.803509|0.984848|0.843817|0.666667|0.666667|0.984848|0|0.021053|0.015152|
|20-19|clear|0.891100|0.891100|0.887931|0.887931|0|0.904762|0.863492|0.913793|0.875034|0.705882|0.705882|0.913793|0.041270|0.092063|0.086207|
|20-19|low-elev|0.351618|0.351618|0.351313|0.395489|0.044176|0.914634|0.786585|0.264151|0.687883|0.237288|0.237288|0.264151|0.128049|0.137195|0.735849|
|20-19|rain|0.420194|0.420194|0.801964|0.801964|0|0.863636|0.856643|0.753846|0.843817|0.666667|0.666667|0.753846|0.006993|0.083916|0.246154|
|6-15|clear|0.906489|0.906489|0.888889|0.888889|0|0.857143|0.857143|0.923077|0.872583|0.691176|0.691176|0.923077|0|0|0.076923|
|6-15|low-elev|0.773157|0.773157|0.770604|0.777199|0.006595|0.667742|0.667742|0.929577|0.690408|0.237288|0.237288|0.929577|0|0|0.070423|
|6-15|rain|0.807529|0.807529|0.859992|0.859992|0|0.819444|0.819444|0.904762|0.841292|0.666667|0.666667|0.904762|0|0.010417|0.095238|
|8-20|clear|0.840286|0.840286|0.784963|0.784963|0|0.908197|0.908197|0.691176|0.872769|0.691176|0.790323|0.691176|0|0.003279|0.308824|
|8-20|low-elev|0.721428|0.721428|0.726792|0.726792|0|0.698413|0.698413|0.757576|0.681593|0.226415|0.226415|0.757576|0|0.003175|0.242424|
|8-20|rain|0.774681|0.774681|0.810051|0.810051|0|0.845118|0.845118|0.777778|0.841292|0.666667|0.666667|0.777778|0|0|0.222222|

## 12.分层、决策变化与相对reference

|scene|H M0|H M_DA|H M_HEAD|H M_JOINT|mean I_syn|
|---|---:|---:|---:|---:|---:|
|clear|0.886422|0.886713|0.867836|0.867836|-0.000291|
|low-elev|0.622687|0.622687|0.628508|0.638303|+0.009795|
|rain|0.734188|0.734188|0.834086|0.834086|0|

|class|after M0|after M_DA|after M_HEAD|after M_JOINT|
|---|---:|---:|---:|---:|
|14-10|0.964238|0.964238|0.958862|0.958862|
|14-7|0.574709|0.573767|0.570135|0.568343|
|20-15|0.983455|0.983455|0.972639|0.972639|
|20-19|0.466630|0.468636|0.631351|0.641653|
|6-15|0.928962|0.928962|0.919139|0.919139|
|8-20|0.772711|0.772711|0.742078|0.745445|

JOINT的均值收益主要来自`20-19`救援；相对M0，另5类after accuracy均下降。`I_syn`分布为正4/18、零13/18、负1/18，正收益主要集中于`20-19/low-elev`。

|比较|变化/6630|old错→对|old对→错|old错→错|new错→对|new对→错|new错→错|净正确变化|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|M_DA−M0|10|5|4|1|0|0|0|+1|
|M_HEAD−M0|554|206|108|146|42|22|30|+118|
|M_JOINT−M_HEAD|18|12|2|2|2|0|0|+12|
|M_JOINT−M_DA|553|212|105|142|43|21|30|+129|
|M_JOINT−M0|549|211|103|141|43|21|30|+130|

相对matched R2A：M0逐项相同；CID的M_DA H`+0.000097`但floor`-0.000942`、forgetting`+0.000353`；M_HEAD H`+0.024398`、old-after`+0.024937`、seen-new`+0.023077`、floor`+0.068550`；M_JOINT H`+0.027663`、old-after`+0.026704`、seen-new`+0.025173`、floor`+0.070966`、forgetting`-0.000070`。这证明CID-BPP优于R2A具体实例，但不覆盖自身四臂晋级门。

## 13.coverage、量化与资源

- coverage：8400 rows；physical/observation各8400唯一；6类/7receiver/4day/3scene；168 cells；zero=0；min/max=`32/66`；K5最小余量27；`feature_arrays_read=[]`；`held_fold_selected=false`。
- 64个inner C4/C5 audit state的aggregate及各臂top1 agreement最小值均为1.0，large-margin flip总数0。qKNN最大logit误差0.525818，BPP最大1.440208；这些只证明outer-excluded量化一致性，不代替held性能。
- C5 state min/mean/max=`17,882/18,123/18,601B`；C6=`19,990/20,191/20,703B`。
- 最大build MAC：C5=`4,748,640`，C6=`4,879,040`；四臂MAC/query均值：C5=`9,794`，C6=`11,678`。
- 全36 state build mean/P95=`26.660/27.474ms`；prediction每row batch mean/P95=`17.004/18.203ms`，不是单query时延。
- aggregate four-arm MAC=`77,451,296`；backend=`numpy_cpu`；optimizer steps=`0`；CUDA tensor=`0`；peak VRAM=`0`。
- C6 fallback：`jackknife_overlap`12/18，无fallback 6/18；rank0/1/2=`12/4/2`。

## 14.最终裁决

|门|结果|依据|
|---|---|---|
|技术闭环|PASS|exit0、18 prediction、72 score、marker、SHA和独立复算闭合|
|M_DA产生决策变化|PASS|10/6630，净增1个正确old决策|
|M_DA独立正收益且安全|FAIL|H仅+0.000097；floor/min-old下降0.000942，forgetting增加0.000353，seen-new不变|
|M_HEAD独立正收益|PASS|H+0.029044，old/new/floor改善，forgetting下降|
|M_JOINT aggregate优于两个单臂|PASS|相对HEAD H+0.003265；相对DA H+0.032212|
|mean I_syn>0|PASS|+0.003168|
|协同跨slice/scene成立|FAIL|4正/13零/1负；clear为负，rain为0，仅low-elev为正|
|允许进入125|NO|预注册要求DA、HEAD各自正收益且全部保护门通过，不得看结果后放宽|

最终状态：`ANALYZED / COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。GEOFF/r2技术闭合；CID-BPP相对R2A有实质进展，但DA主效应过弱并损伤floor/forgetting，协同集中于少数low-elev row，不得启动125、不得声明联合成功。下一工作项转向下一冻结候选的最小可证伪窄实验；本run不用于调参。
