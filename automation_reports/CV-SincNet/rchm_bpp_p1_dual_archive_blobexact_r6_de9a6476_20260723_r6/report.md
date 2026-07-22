# GEOFF/r2.2 Phase1双表征归档r6技术run报告

## 1.身份与状态

- run_id:`rchm_bpp_p1_dual_archive_blobexact_r6_de9a6476_20260723_r6`
- candidate:`P1-DUAL-ARCHIVE-GEOFF/r2.2-BUFFER`
- 主agent:`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- protocol:`p2_min_v1`；数据状态:`VALIDATED_ONCE_REUSED`
- 当前状态:`LOCAL_VERIFIED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- retry:`NO`

本run不改变方法、received IQ、physical ID、receiver/TX、scene、K、support/query split或protocol schema，不重复数据验证。相对r5的唯一release delta是以`git -c core.autocrlf=false archive --prefix=source_de9a6476/`生成ZIP，同时闭合正确顶层根和原始Git blob字节；GEOFF、buffer transport、阈值、输入、命令、archive和coverage门全部不变。

## 2.matched failure与即时falsifier

- r4已落地冻结输入，但根层ZIP与wrapper的`source_de9a6476/`合同不一致，package-layout verifier exit=`1`，pipeline未启动。
- r5未交runner。独立本地review发现Windows全局`core.autocrlf=true`使3163/3918个ZIP blob从LF变成CRLF，裁决`P0=1，P1=1→REVISE`。

r6假设：唯一安全顶层根且3918/3918文件逐Git blob一致的ZIP可通过远端布局与源码闭包门，并执行完全相同的GEOFF/r2.2 pipeline。任一prefix、blob、SHA、asset、compile、parity、archive或coverage门失败立即停止，不重试、不修改冻结输入。

## 3.Git、源码包与本地证据

|项目|冻结证据|
|---|---|
|方法commit|`de9a6476e60428ba89d243af63f1eca6229d52c6`|
|r4失败记录commit|`61d2c0cb`|
|分支|`codex/ground-prototype-da-rd`|
|源码ZIP|`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_blobexact_r6_de9a6476_20260723_r6/source_de9a6476.zip`|
|ZIP SHA/大小/成员|`282c4f085266f6e95345ce66e82178ee08ab3950ce9e7efb2c2c72572edb9127 / 33,093,737B / 4,443`|
|路径闭包|唯一顶层`source_de9a6476/`；3918个文件；prefix外、路径穿越、缺失、额外和重名均为0|
|内容闭包|对每个ZIP文件重算Git blob SHA-1并与`git ls-tree -rz de9a6476`比较，`3918/3918`一致，mismatch=`0`|
|wrapper SHA|`065d1f5f6e9d33f384b194329bd3e996e2f98771b17769f6152fe8421df40618`|
|wrapper静态门|`bash -n`通过；未分配`CUDA_VISIBLE_DEVICES`时exit=`70`|
|本地方法测试|沿用`de9a6476`的`ssr-gpu py_compile + 8 passed`及方法独立MERGE；r6仅修复发布包字节与布局|

6项关键源码SHA不变：dual archive exporter=`31a6a464f470ae9bdb6cbc8814581ff6c73403d5c99b497a224b3f783831fe64`；single archive exporter=`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`；dual exporter=`e2cbc0ce19402e8c665489fb2b13bb63f988c14e138cbc39dd20f7c9e2b12090`；parity verifier=`0a3e80b226997b353c577de94aa8c8e92fb25f5bf13d8b81c9bc27f448ef284b`；dual forward=`1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56`；cache loader=`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`。

## 4.N607发布合同

- remote run root:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_blobexact_r6_de9a6476_20260723_r6`
- source root:`<run-root>/source_de9a6476`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 先执行direct preflight并核验GPU/进程/磁盘及run root不存在；只同步冻结ZIP和wrapper。
- 解包前核验ZIP SHA、唯一安全prefix、3918个blob与source commit逐项一致；解包后核验4项外部资产、6项源码SHA、`py_compile`与`bash -n`。
- 只允许一次detached launch；不得retry/restart、远端编辑、数据重验、target/query/held/125、调参、kill或干预无关作业。

冻结外部输入SHA：checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；adapter=`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`；cache_set=`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`；selection salt=`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`。

## 5.技术完成门与预期artifact

预期runtime/export/parity、archive NPZ/manifest、coverage receipt、sha256sums、PID/exit/log及completion marker。parity保持schema v2、batch=`[1,8,256]`、每batch3次调用、三输出maxabs≤`1e-5`。archive manifest必须为v2且内部verify通过。

coverage硬门保持：row=8400、unique physical=8400、unique observation=8400、class/receiver/day/scenario=`6/7/4/3`、receiver×day×class cells=168、zero=0、min>10，并报告K1/K5/K10余量；`feature_arrays_read=[]`、`held_fold_selected=false`。coverage只描述归档，不是性能。

全部通过也只能标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`；失败标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。本run预期prediction=`0`。

## 6.完成后回填

|字段|结果|
|---|---|
|release-control Git commit|`PENDING`|
|route/GPU/PID|`PENDING`|
|remote SHA/blob/layout/compile|`PENDING`|
|exit/marker|`PENDING`|
|parity receipt|`NOT_GENERATED`|
|archive/manifest|`NOT_GENERATED`|
|coverage|`NOT_GENERATED`|
|prediction count|`0`|
|回收路径/SHA|`PENDING`|
|最终状态|`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`|

coverage真实通过后，主agent立即以其真实SHA冻结并发布最小held四臂性能矩阵。
