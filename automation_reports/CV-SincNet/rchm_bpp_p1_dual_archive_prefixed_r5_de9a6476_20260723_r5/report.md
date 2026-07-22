# GEOFF/r2.2 Phase1双表征归档r5技术run报告

## 1.身份与状态

- run_id:`rchm_bpp_p1_dual_archive_prefixed_r5_de9a6476_20260723_r5`
- candidate:`P1-DUAL-ARCHIVE-GEOFF/r2.2-BUFFER`
- 主agent:`/root`；唯一N607 runner:`NOT_ASSIGNED`
- protocol:`p2_min_v1`；数据状态:`VALIDATED_ONCE_REUSED`
- 当前状态:`LOCAL_REJECTED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- retry:`NO`

本run不改变方法、received IQ、physical ID、receiver/TX、scene、K、support/query split或protocol schema，不重复数据验证。相对r4的唯一release delta是以`git archive --prefix=source_de9a6476/`生成ZIP，使冻结源码根与wrapper的`SOURCE_ROOT`一致；GEOFF、buffer transport、阈值、输入、命令、archive和coverage门全部不变。

独立发布复审裁决为`P0=1，P1=1→REVISE`：路径集合虽与提交树一致，但Windows全局`core.autocrlf=true`使ZIP内3163/3918个blob由LF变成CRLF，不能作为逐blob commit-bound源码；r5也尚未进入Git。故本run未交runner、未连接N607，prediction=`0`，该ZIP保持原样且禁止发布。

## 2.matched failure与即时falsifier

matched run=`rchm_bpp_p1_dual_archive_buffer_r4_de9a6476_20260723_r4`，远端ZIP与wrapper SHA均匹配，但package-layout verifier以exit=`1`拒绝根层ZIP；pipeline未启动，无PID、parity、archive、coverage或prediction。r4永久为`TECHNICAL_FAILURE / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`，不得复用。

r5假设：带唯一`source_de9a6476/`根的commit-bound ZIP可通过不可覆盖布局门，并执行完全相同的GEOFF/r2.2 pipeline。任一prefix、SHA、asset、compile、parity、archive或coverage门失败立即停止，不重试、不修改冻结输入。

## 3.Git、源码包与本地证据

|项目|冻结证据|
|---|---|
|方法commit|`de9a6476e60428ba89d243af63f1eca6229d52c6`|
|r4失败记录commit|`61d2c0cb`|
|分支|`codex/ground-prototype-da-rd`|
|独立方法review|`P0=0,P1=0,P2=0→MERGE`|
|源码ZIP|`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_prefixed_r5_de9a6476_20260723_r5/source_de9a6476.zip`|
|ZIP SHA/大小/成员|`e6ac6dcfa6839ceae3b14c72734f08697648b30a8a3f81070b0d1c49b85a156a / 33,169,552B / 4,443`|
|ZIP布局|全部成员位于`source_de9a6476/`；prefix外成员=`0`；唯一顶层目录=`source_de9a6476/`|
|wrapper SHA|`259308231ba4153f22117f0d62fe7d731b2bca68efc8de65ebc634c43f435335`|
|本地方法测试|沿用`de9a6476`的`ssr-gpu py_compile + 8 passed`及r4独立MERGE；r5只增加包布局修正|

冻结源码成员SHA不变：dual archive exporter=`31a6a464f470ae9bdb6cbc8814581ff6c73403d5c99b497a224b3f783831fe64`；single archive exporter=`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`；dual exporter=`e2cbc0ce19402e8c665489fb2b13bb63f988c14e138cbc39dd20f7c9e2b12090`；parity verifier=`0a3e80b226997b353c577de94aa8c8e92fb25f5bf13d8b81c9bc27f448ef284b`；dual forward=`1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56`；cache loader=`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`。

## 4.N607发布合同

- remote run root:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_prefixed_r5_de9a6476_20260723_r5`
- source root:`<run-root>/source_de9a6476`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 先执行direct preflight并核验GPU/进程/磁盘及run root不存在；只同步冻结ZIP和wrapper。
- 在任何解包前核验ZIP只有一个安全顶层根且无路径穿越、重名或prefix外成员；随后核验远端ZIP/wrapper、4项外部资产、6项源码SHA、`py_compile`和`bash -n`。
- 只允许一次detached launch；不得retry/restart、远端编辑、数据重验、target/query/held/125、调参、kill或干预无关作业。

冻结外部输入SHA：checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；adapter=`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`；cache_set=`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`；selection salt=`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`。

## 5.技术完成门与预期artifact

预期runtime/export/parity、archive NPZ/manifest、coverage receipt、sha256sums、PID/exit/log及completion marker。parity保持schema v2、batch=`[1,8,256]`、每batch3次调用、三输出maxabs≤`1e-5`。archive manifest必须为v2且内部verify通过。

coverage硬门保持：row=8400、unique physical=8400、unique observation=8400、class/receiver/day/scenario=`6/7/4/3`、receiver×day×class cells=168、zero=0、min>10，并报告K1/K5/K10余量；`feature_arrays_read=[]`、`held_fold_selected=false`。coverage只描述归档，不是性能。

全部通过也只能标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`；失败标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。本run预期prediction=`0`。

## 6.完成后回填

|字段|结果|
|---|---|
|release-control Git commit|`PENDING_LOCAL_REJECT_RECORD`|
|route/GPU/PID|`NOT_LANDED / 无PID`|
|remote SHA/layout/compile|`NOT_RUN`|
|exit/marker|`NOT_RUN`|
|parity receipt|`NOT_GENERATED`|
|archive/manifest|`NOT_GENERATED`|
|coverage|`NOT_GENERATED`|
|prediction count|`0`|
|回收路径/SHA|`无远端artifact`|
|最终状态|`LOCAL_REJECTED / NOT_LANDED / NO_PERFORMANCE_RESULT`|

coverage真实通过后，主agent立即以其真实SHA冻结并发布最小held四臂性能矩阵。
