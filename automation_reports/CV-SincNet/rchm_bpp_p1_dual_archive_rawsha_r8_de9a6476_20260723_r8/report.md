# GEOFF/r2.2 Phase1双表征归档r8技术run报告

## 1.身份与状态

- run_id:`rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8`
- candidate:`P1-DUAL-ARCHIVE-GEOFF/r2.2-BUFFER`
- 主agent:`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- protocol:`p2_min_v1`；数据状态:`VALIDATED_ONCE_REUSED`
- 当前状态:`LOCAL_VERIFIED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- retry:`NO`

本run不改变方法、ZIP、数据或科学协议，不重复数据验证。相对r7的唯一delta是把wrapper中6项运行关键源码SHA从旧Windows CRLF归档值替换为当前raw Git blob ZIP内逐文件实测SHA；GEOFF、buffer transport、输入、命令、parity、archive和coverage门均不变。

## 2.matched failure与falsifier

r7的远端ZIP整体SHA、33,093,737B、4443/3918布局与安全prefix均通过，但解包后首项raw源码实际SHA=`b3748f…`，旧冻结预期为CRLF SHA=`31a6a4…`，预启动exit=`72`。pipeline未启动、无PID，parity/archive/coverage/prediction均为0。

r8假设：用同一raw ZIP内6项实测SHA替换旧CRLF值后，可通过预启动源码门并进入原冻结pipeline。任一SHA、asset、compile、parity、archive或coverage门失败立即停止，不重试、不修改冻结输入。

## 3.Git、冻结输入与本地验证

|项目|冻结证据|
|---|---|
|方法commit|`de9a6476e60428ba89d243af63f1eca6229d52c6`|
|r7失败记录commit|`9a9a41a3fa096dd871ad3478968029870f14876f`|
|源码ZIP|`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8/source_de9a6476.zip`|
|ZIP SHA/大小/布局|`282c4f085266f6e95345ce66e82178ee08ab3950ce9e7efb2c2c72572edb9127 / 33,093,737B / 4,443 entries / 3,918 files / 唯一source_de9a6476/根`|
|内容闭包|本地独立review已证明3918/3918文件Git blob一致；r7远端已证明同一ZIP整体SHA与安全布局一致|
|wrapper SHA|`ccccabbd786223dea6f89505b4b87f0bdbfaf06260180a13f28c2fa24c4ce614`|

raw ZIP关键源码SHA：

|相对路径|SHA256|
|---|---|
|`code/scripts/export_phase1_singleobs_dual_feature_archive.py`|`b3748fe8468e1927c803c9b55d6d1c231ec15043e297d7676d36c921516e7dc0`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`ab4d3c40251f2bd147e7948ced392d185d0ef7b3f45c18924e7ab1bd457dac6d`|
|`code/scripts/export_adv3b02_dual_feature_torchscript.py`|`180f41e844d4aa4cc033cb4c3366eece68a3ce14298a16417f80e058890a696e`|
|`code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py`|`f1f5226194015c3e8a1f632898236a2e3e46986f2259ba24a3a81e400dbd0ec8`|
|`code/cvsrffi/dual_feature_forward.py`|`eeaca06f84f5771c90dfb92e6bbbc4980f2772e9fcdf80d54e06fee387afd815`|
|`code/cvsrffi/leo_weak_cache.py`|`19c98daafcc6f3e6f2de038883b83ea10c4d59edca62ff0e73cb509175c57ef8`|

本地门：6项值从冻结ZIP成员流直接计算并与wrapper逐项一致；`bash -n`必须通过；未分配`CUDA_VISIBLE_DEVICES`必须exit=`70`。方法测试沿用`de9a6476`的`ssr-gpu py_compile + 8 passed`。

## 4.N607发布合同

- remote root:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；source root:`<run-root>/source_de9a6476`。
- direct preflight后核验run root不存在、GPU/进程/磁盘；只同步ZIP和wrapper。
- 远端核验ZIP整体SHA/安全布局，解包后核验上述6项raw源码SHA、4项外部资产、`py_compile`与`bash -n`；不要求远端Git。
- 只允许一次detached launch；不得retry/restart、远端编辑、数据重验、target/query/held/125、调参或干预其他作业。

外部输入SHA：checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；adapter=`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`；cache_set=`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`；selection salt=`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`。

## 5.完成门与回填

parity保持schema v2、batch=`[1,8,256]`、每batch3次调用、三输出maxabs≤`1e-5`。archive manifest必须为v2且内部verify通过。coverage必须为8400行、physical/observation均唯一、6/7/4/3覆盖、168个receiver×day×class cell、zero=0、min>10、K1/K5/K10余量正确、`feature_arrays_read=[]`、`held_fold_selected=false`。

|字段|结果|
|---|---|
|release-control Git commit|`PENDING`|
|route/GPU/PID/exit|`PENDING`|
|remote SHA/source/compile|`PENDING`|
|parity receipt|`NOT_GENERATED`|
|archive/manifest|`NOT_GENERATED`|
|coverage|`NOT_GENERATED`|
|prediction count|`0`|
|最终状态|`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`|

全部通过也只能标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`；coverage真实通过后立即冻结并发布最小held四臂性能矩阵。

## 6.N607一次性执行终态

- 执行时间：2026-07-23；唯一N607 runner：`/root/r8_n607_runner`；retry：`NO`。
- 直接预检通过：run root启动前不存在；8张RTX3090均空闲；`/home`可用空间7.5TB；未发现其他训练进程。
- 本地与远端封包核验通过：ZIP=`282c4f085266f6e95345ce66e82178ee08ab3950ce9e7efb2c2c72572edb9127`、33,093,737B、4443 entries/3918 files；wrapper=`ccccabbd786223dea6f89505b4b87f0bdbfaf06260180a13f28c2fa24c4ce614`。远端安全布局、6项raw源码SHA、4项外部资产SHA、`py_compile`、`bash -n`全部通过；未执行远端Git门。
- 唯一detached launch：`CUDA_VISIBLE_DEVICES=0 nohup bash ./run_pipeline.sh`；PID=`405693`；GPU=`0`；自然exit=`0`。未重试、未重启、未干预其他作业。
- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8`；日志：`logs/pipeline.log`；输出：`output/`。

|同一run结果行|技术证据|结果/摘要|SHA256|裁决|
|---|---|---|---|---|
|`rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8`|base parity receipt|`PASS`；schema v2；batch`[1,8,256]`；每batch3次调用；`max_abs_output_delta=0.0<=1e-5`|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|通过|
|同上|archive与manifest|8400行；archive=`DEVELOPMENT_ONLY_NOT_FORMAL`|archive:`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`；manifest:`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|技术归档完成|
|同上|coverage receipt|8400 physical/observation各自唯一；6类/7 receiver/4 day/3 scenario；168 cells；zero=0；min=32；K1/K5/K10余量=31/27/22；`feature_arrays_read=[]`；`held_fold_selected=false`|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|通过，但仅描述性覆盖|
|同上|prediction|未生成任何prediction artifact|`0`|无性能结果|

### 终态与回收

终态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。该run只完成Phase1离线dual runtime、archive和coverage技术证据；无held/target/query/125执行，不能作为性能、Stage2或可推广方法结论。

已回收的小型证据位于根报告同run目录的`retrieved/`：`pipeline.pid`、`pipeline.exit`、`pipeline.log`、`dual_export_receipt.json`、`base_parity_receipt.json`、`base_parity_vector.json`、archive manifest、`coverage_receipt.json`和`sha256sums.txt`。本地复核的receipt/manifest/coverage文件SHA均与远端清单一致；完整archive保留在远端且SHA已验封。
