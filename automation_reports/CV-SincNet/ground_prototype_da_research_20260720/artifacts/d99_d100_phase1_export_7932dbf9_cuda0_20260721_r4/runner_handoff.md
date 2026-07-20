# D99/D100 Phase1 exporter r4 runner handoff

## 最终状态

- run ID：`d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4`
- 状态链：`LOCAL_VERIFIED -> LANDED -> RUNNING -> ARTIFACTS_COMPLETE -> ANALYZED`
- child exit：`0`
- archive内部verify：`PASS`
- 产物性质：`DEVELOPMENT_PHASE1_TEMPORARY_ASSET`，不是formal archive或target性能结果
- LODO：未启动
- 代码/参数修改：无
- r1/r2/r3补写：无

## 时间、进程和资源

|字段|实际值|
|---|---|
|preflight时间|`2026-07-21 03:18:24 CST`|
|开始时间|`2026-07-21T03:25:52.952111399+08:00`|
|结束时间|`2026-07-21T03:26:04.823417358+08:00`|
|PID|`1436954`|
|CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9`|
|GPU|物理GPU0，`requested_device=resolved_device=cuda:0`|
|CPU threads|`OMP=2,MKL=2,OPENBLAS=2`|
|batch size|`256`|
|等价前向批数|`ceil(8400/256)=33`|
|启动前GPU0|`0%`,`10/24576 MiB`，无compute process|
|终态GPU0|`0%`,`10/24576 MiB`，PID已退出|

每次SSH/SCP后均核对短连接退出，最终为：

```text
ssh_processes=0
n607_established=0
```

## 冻结版本与落地验证

|项|值|结果|
|---|---|---|
|代码commit|`7932dbf9`|匹配预登记|
|报告/证据commit|`9f7f4f361eacad2f204066fe734d0729aad79e2f`|匹配预登记|
|源码ZIP|`E:\type10-7\code\snapshots\d99_d100_phase1_export_7932dbf9_20260721_r3\source_7932dbf9.zip`|复用r3冻结ZIP|
|源码ZIP SHA256|`40a2ecc9d01a12759d9a67693d7eaa974751bae02c05faab0dabfc580efdbd72`|本地/远端一致|
|ZIP大小/成员|`31168653 bytes`/`4342`|一致|
|安全解压|`4342 members`,`3837 files`|绝对路径、`..`、反斜杠、重复target、symlink均为0|
|selection salt SHA256|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|匹配|
|cache set SHA256|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|匹配|
|runtime SHA256|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|匹配|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|匹配|

r4 run/log/source创建前均确认不存在。隔离源码`py_compile`和`cvsrffi.leo_weak_cache` import均通过。

### ZIP成员执行身份

|文件|ZIP成员/远端SHA256|
|---|---|
|`code/cvsrffi/leo_weak_cache.py`|`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`|
|`tests/test_export_phase1_singleobs_feature_archive.py`|`493c6edd167f743aaf797202e530d65f01a8a47107353eb0b2c00df720031adb`|

三项ZIP成员与远端解压文件逐字节一致。

### r1 ground只读复核

|文件|SHA256|
|---|---|
|`build_result.json`|`b9c0213f9dbbd2f438ab31541ac4404d852388bd42fbe439b9a768fbf87714c2`|
|`d99_base_method_lock_dev.json`|`7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`|
|`d99_ground_aggregation_spec.json`|`f4db8091aeb7204bb4a641d02810c19acbb9fc002ec8e82c82fd9f4fe2820efe`|
|`d99_ground_bundle_dev.manifest.json`|`f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`|
|`d99_ground_bundle_dev.npz`|`e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`|

## 唯一child命令

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code/scripts/export_phase1_singleobs_feature_archive.py --mode development --cache-set /home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json --cache-set-sha256 125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74 --runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --expected-runtime-sha256 f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a --class-ids 14-10,14-7,20-15,20-19,6-15,8-20 --selection-salt-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json --selection-salt-receipt-sha256 38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/phase1_feature_archive --device cuda:0 --batch-size 256
```

wrapper使用不可覆盖的`exporter.log/.pid/.exit`。第一次尝试用LF-only本地管道发送launcher时，远端三个wrapper文件和output均保持不存在，明确证明child未启动；之后才执行本轮唯一一次实际child启动。实际启动完成后，launcher末尾CRLF导致外层bash多报告一次`$'\r': command not found`，但不可覆盖文件、PID和完整日志证明child只运行一次；该传输噪声不进入child日志或产物。

## 产物及内部verify

|产物|远端路径|字节|SHA256|
|---|---|---:|---|
|feature archive NPZ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/phase1_feature_archive/phase1_singleobs_feature_archive.npz`|5173473|`cdd8747d267336b48e8c555ce7e010206f042ff07c695af351541a97187fad03`|
|archive manifest|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/phase1_feature_archive/phase1_singleobs_feature_archive.manifest.json`|4740|`5f363bc09503f882c66aa92805657199ca57484f627c5d2805254cad07bffa15`|

同一冻结源码的`verify_phase1_singleobs_archive`重新执行为`PASS`，验证精确member allowlist、逐数组SHA注册表和NPZ文件SHA。

### 精确成员、shape与有限性

|member|shape|dtype|验收|
|---|---|---|---|
|`features`|`[8400,288]`|`float32`|全有限，min=`-0.2199309021`，max=`7.2967853546`|
|`labels`|`[8400]`|`<U5`|6类|
|`receiver_ids`|`[8400]`|`<U4`|7个receiver|
|`day_ids`|`[8400]`|`<U10`|4天|
|`physical_ids`|`[8400]`|`<U34`|8400唯一，每ID一行|
|`scenario_names`|`[8400]`|`<U17`|3个场景|
|`class_ids`|`[6]`|`<U5`|顺序冻结|
|`checkpoint_reference_logits`|`[8400,6]`|`float32`|全有限，min=`-9.8191347122`，max=`15.9391593933`|

8400行等价完成33个batch前向；GPU0设备冲突不再出现。

## lineage与协议字段

- archive manifest schema：`cvs.phase1.single_leo_feature_archive.v2`
- status：`DEVELOPMENT_PHASE1_TEMPORARY_ASSET`
- `formal_archive=false`，`development_archive=true`
- cache outer observed schema：`cvs_leo_weak_iq_cache_set_v1`
- 三个cache inner observed schema：均为`cvs_leo_weak_iq_cache_v1`
- `cache_legacy_schema_compatibility=true`
- original cache SHA=`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`
- authority cache SHA=`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`
- runtime SHA=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`
- checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- `selected_observations_per_physical_id=1`
- `unselected_observations_forwarded=0`
- `one_output_row_per_physical_id=true`
- access audit：clean/target/query/raw-IQ/received-IQ/unselected-IQ访问或持久化均为0/false
- lifecycle：temporary selection asset；当前`phase2_bundle_ingest_allowed=false`、`phase2_runtime_access_allowed=false`

注意：archive自身schema是v2；“outer/inner v1”指它绑定并记录的历史cache lineage，不能混写为archive schema v1。

## 最终分布

|维度|计数|
|---|---|
|class|`14-10:1400`,`14-7:1400`,`20-15:1400`,`20-19:1400`,`6-15:1400`,`8-20:1400`|
|receiver|`1-1:1195`,`1-19:1217`,`14-7:1198`,`18-2:1186`,`19-2:1215`,`2-1:1230`,`2-19:1159`|
|day|`2021_03_01:2147`,`2021_03_08:2115`,`2021_03_15:2082`,`2021_03_23:2056`|
|scenario|`leo_clear_weak:2852`,`leo_low_elev_weak:2820`,`leo_rain_weak:2728`|

class顺序：`[14-10,14-7,20-15,20-19,6-15,8-20]`。

## 本地最小证据

|文件|字节|SHA256|
|---|---:|---|
|`logs/exporter.log`|2318|`2d1b4b8e909c1aafe0a490413c83b14449393d7d71edeec219550ba5f29fdf21`|
|`logs/exporter.pid`|8|`64a9901157d7ae6054c5d9babc36ada009cddf736841ba043b99ff6ab0936f12`|
|`logs/exporter.exit`|2|`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`|
|`archive/phase1_singleobs_feature_archive.manifest.json`|4740|`5f363bc09503f882c66aa92805657199ca57484f627c5d2805254cad07bffa15`|
|`input/d99_d100_phase1_selection_salt.json`|440|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|
|`authority/cache_set.json`|4501|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|

本地未复制5.17MB NPZ；其远端不可覆盖路径、大小、SHA及内部verify证据已完整记录。

## LODO四个archive字段

```text
feature_archive_path=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/phase1_feature_archive/phase1_singleobs_feature_archive.npz
feature_archive_sha256=cdd8747d267336b48e8c555ce7e010206f042ff07c695af351541a97187fad03
feature_archive_manifest_path=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/phase1_feature_archive/phase1_singleobs_feature_archive.manifest.json
feature_archive_manifest_sha256=5f363bc09503f882c66aa92805657199ca57484f627c5d2805254cad07bffa15
```

这些字段仅供主线在另行完成配置预登记后使用；本runner未生成LODO config，也未启动LODO。
