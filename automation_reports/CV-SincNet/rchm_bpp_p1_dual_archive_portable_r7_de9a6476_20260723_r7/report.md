# GEOFF/r2.2 Phase1双表征归档r7技术run报告

## 1.身份与状态

- run_id:`rchm_bpp_p1_dual_archive_portable_r7_de9a6476_20260723_r7`
- candidate:`P1-DUAL-ARCHIVE-GEOFF/r2.2-BUFFER`
- 主agent:`/root`；唯一N607 runner:`/root/r7_n607_runner`
- protocol:`p2_min_v1`；数据状态:`VALIDATED_ONCE_REUSED`
- 当前状态:`TECHNICAL_FAILURE / PRELAUNCH_P1 / NOT_RUNNING / NO_PERFORMANCE_RESULT`
- retry:`NO`

本run不改变方法、源码ZIP字节、received IQ、physical ID、receiver/TX、scene、K、support/query split或protocol schema，不重复数据验证。相对r6的唯一release delta是取消对N607本地Git对象库的冗余依赖：本地已独立证明3918/3918个ZIP文件与`de9a6476`Git blob一致；远端只需核验同一ZIP整体SHA、安全prefix/路径和解包后6项运行关键源码SHA，即可传递该闭包。GEOFF、buffer transport、阈值、输入、命令、archive和coverage门均不变。

## 2.matched failure与即时falsifier

r6 direct预检通过且远端run根不存在，但N607项目根不是Git工作树，`git cat-file de9a6476`以exit=`128`失败；runner按P1即停，未创建run根、未SCP、无PID，parity/archive/coverage/prediction均未生成。r6永久为`TECHNICAL_FAILURE / NOT_LANDED / NO_PERFORMANCE_RESULT`。

r7假设：远端ZIP整体SHA与本地独立逐blob审查形成可移植证据链，无需在N607复制Git对象库。任一ZIP SHA、prefix、安全路径、解包后关键源码SHA、asset、compile、parity、archive或coverage门失败立即停止，不重试、不修改冻结输入。

## 3.Git、源码包与本地证据

|项目|冻结证据|
|---|---|
|方法commit|`de9a6476e60428ba89d243af63f1eca6229d52c6`|
|r6失败记录commit|`4e9adcfeed08feed6c5ea08a706b86c2d6b4cd3b`|
|分支|`codex/ground-prototype-da-rd`|
|源码ZIP|`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_portable_r7_de9a6476_20260723_r7/source_de9a6476.zip`|
|ZIP SHA/大小/成员|`282c4f085266f6e95345ce66e82178ee08ab3950ce9e7efb2c2c72572edb9127 / 33,093,737B / 4,443`|
|本地路径闭包|唯一顶层`source_de9a6476/`；3918个文件；prefix外、路径穿越、缺失、额外和重名均为0|
|本地内容闭包|每个ZIP文件重算Git blob SHA-1并与`git ls-tree -rz de9a6476`比较，`3918/3918`一致，mismatch=`0`；独立r6 review为`P0=0，P1=0→MERGE`|
|wrapper SHA|`03460b472fd77230d648464494ccced4671ffa95847df22a222387ca0cb73a2e`|
|wrapper静态门|`bash -n`通过；未分配`CUDA_VISIBLE_DEVICES`时exit=`70`|
|本地方法测试|沿用`de9a6476`的`ssr-gpu py_compile + 8 passed`；r7只修复发布证据的远端可移植性|

6项关键源码SHA不变：dual archive exporter=`31a6a464f470ae9bdb6cbc8814581ff6c73403d5c99b497a224b3f783831fe64`；single archive exporter=`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`；dual exporter=`e2cbc0ce19402e8c665489fb2b13bb63f988c14e138cbc39dd20f7c9e2b12090`；parity verifier=`0a3e80b226997b353c577de94aa8c8e92fb25f5bf13d8b81c9bc27f448ef284b`；dual forward=`1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56`；cache loader=`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`。

## 4.N607发布合同

- remote run root:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_portable_r7_de9a6476_20260723_r7`
- source root:`<run-root>/source_de9a6476`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- direct preflight后核验GPU/进程/磁盘及run root不存在；只同步冻结ZIP和wrapper。
- 远端必须核验ZIP整体SHA=`282c4f...b9127`、大小、唯一安全prefix、无路径穿越/重名；不要求远端Git仓库或`git cat-file`。解包后核验4项外部资产、6项关键源码SHA、`py_compile`和`bash -n`。
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
|remote ZIP/layout/compile|`PENDING`|
|exit/marker|`PENDING`|
|parity receipt|`NOT_GENERATED`|
|archive/manifest|`NOT_GENERATED`|
|coverage|`NOT_GENERATED`|
|prediction count|`0`|
|回收路径/SHA|`PENDING`|
|最终状态|`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`|

coverage真实通过后，主agent立即以其真实SHA冻结并发布最小held四臂性能矩阵。

## 7.r7实际runner终态回填

本节覆盖第6节中的`PENDING`字段。runner遵守`retry=NO`，在首个P1预启动门失败后立即停止；没有启动、修复、重试或触及held/target/125。

|字段|实测结果|
|---|---|
|release-control Git commit|`b333489439ee374d64c6d4a3b7cd696398b9088f`|
|PRECHECK|direct N607通过；初始run root不存在；8张RTX3090均为10MiB、无同名进程；`/home`可用约7.5T|
|SYNC/SHA/LAYOUT|ZIP与wrapper精确SCP完成；远端ZIP大小`33093737`、SHA=`282c4f085266f6e95345ce66e82178ee08ab3950ce9e7efb2c2c72572edb9127`；布局`4443`entries/`3918`files/唯一`source_de9a6476/`/路径穿越符号链接重名均为`0`|
|layout回执|远端`<run-root>/zip_layout_receipt.json`；SHA=`eb424a5457702bdf5285b131e4839a3078a0b94f9734c9a60cfd23593ebbe60e`；已回收为同目录`zip_layout_receipt.remote.json`|
|首个P1根因|安全解包后，`source_de9a6476/code/scripts/export_phase1_singleobs_dual_feature_archive.py`实际SHA=`b3748fe8468e1927c803c9b55d6d1c231ec15043e297d7676d36c921516e7dc0`，不等于冻结预期`31a6a464f470ae9bdb6cbc8814581ff6c73403d5c99b497a224b3f783831fe64`；预启动命令exit=`72`|
|源码/外部资产/compile|wrapper SHA门已通过；第1项源码SHA失败后`set -e`停止，剩余5项源码、4项外部资产、`py_compile`与`bash -n`均未执行，不得推定通过|
|PID/GPU/exit|未启动；`pipeline.pid`、`pipeline.exit`、`pipeline.log`均不存在；无GPU分配、无自然child exit|
|parity/archive/coverage|均未生成；`output/`不存在|
|prediction|`0`|
|remote路径|`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_portable_r7_de9a6476_20260723_r7`|
|最终裁决|`TECHNICAL_FAILURE / PRELAUNCH_P1 / NOT_RUNNING / NO_PERFORMANCE_RESULT`|

该失败说明：在远端已验证ZIP整体SHA与完整性、且布局检查通过的前提下，冻结ZIP内至少该运行关键源码字节与r7预期SHA不一致。该run永久停止，不能据此生成性能或发布结论。
