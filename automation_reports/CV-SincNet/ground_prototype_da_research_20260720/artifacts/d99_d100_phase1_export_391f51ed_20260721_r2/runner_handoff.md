# D99/D100 Phase1 exporter r2 runner handoff

## 终态

- run ID：`d99_d100_phase1_export_391f51ed_20260721_r2`
- runner状态链：`LOCAL_VERIFIED -> LANDED -> RUNNING -> FAILED_DIAGNOSTIC`
- 未达到：`ARTIFACTS_COMPLETE`、`ANALYZED`
- 子进程退出码：`1`
- 启动：`2026-07-21T02:38:12.783180527+08:00`
- 结束：`2026-07-21T02:38:14.311612649+08:00`
- PID：`1414309`，终态已退出
- GPU：`cuda:4`；终态`0%`、`10/24576 MiB`
- 自动重试：无
- builder重跑：无
- LODO启动：无
- unrelated workload干预：无

## 冻结版本与输入

|项|值|
|---|---|
|代码commit|`391f51ed`|
|预登记报告commit|`576c4a8a46a2294326f113da93a04c059a321faa`|
|本地/远端Git archive ZIP SHA256|`faad85ee50b83353015dbba51653b33975985296d0930e6ee0b2635897ff236e`|
|ZIP大小/成员数|`31150916 bytes`/`4331`|
|cache set SHA256|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|runtime SHA256|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selection salt SHA256|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|

remote fixed-input SHA及r1 ground五项均在启动前逐项匹配。r1 ground核心SHA：

|项|SHA256|
|---|---|
|build result|`b9c0213f9dbbd2f438ab31541ac4404d852388bd42fbe439b9a768fbf87714c2`|
|base lock|`7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`|
|aggregation spec|`f4db8091aeb7204bb4a641d02810c19acbb9fc002ec8e82c82fd9f4fe2820efe`|
|ground release manifest|`f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`|
|ground NPZ|`e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`|

### 执行字节身份说明

本轮执行身份是预登记Git archive ZIP，而不是Windows工作树CRLF字节。ZIP内/远端解压成员SHA一致：

|文件|Git archive/远端SHA256|Windows工作树SHA256|
|---|---|---|
|`code/cvsrffi/leo_weak_cache.py`|`da0fe679a984876d1e620cd7560b9ee589c6946d0b04187ab575542662f00b6c`|`851ceeaacf8146e4c7e480d22278df9914db6051eb63c50032f9471f66d28b86`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`|`ab4d3c40251f2bd147e7948ced392d185d0ef7b3f45c18924e7ab1bd457dac6d`|
|`tests/test_export_phase1_singleobs_feature_archive.py`|`a98a42e49d54b20a972332870fa1e8b49eba15354ede690cc373803e123bd8dd`|`d988183fd7a53febbf9c89663d3228c5e413321d2b77b12e24f5d07513d06499`|

Windows SHA只用于说明CRLF差异，不能作为远端执行字节身份；Git archive与远端解压成员逐字节一致。

## 路径与命令

- source CWD：`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/source_391f51ed`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_export_391f51ed_20260721_r2`

冻结子命令：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/source_391f51ed/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/source_391f51ed/code/scripts/export_phase1_singleobs_feature_archive.py --mode development --cache-set /home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json --cache-set-sha256 125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74 --runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --expected-runtime-sha256 f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a --class-ids 14-10,14-7,20-15,20-19,6-15,8-20 --selection-salt-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/input/d99_d100_phase1_selection_salt.json --selection-salt-receipt-sha256 38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_391f51ed_20260721_r2/phase1_feature_archive --device cuda:4 --batch-size 256
```

## 失败事实与根因定位

失败发生在任何特征导出之前：

```text
ValueError: LEO cache is missing required members: ['source_dataset_sha256', 'source_record_indices']
```

调用链：

```text
export_development_sha_only_phase1_singleobs_feature_archive
  -> _export_impl
  -> _load_verified_v1_only_source_validation_cache_set
  -> load_verified_leo_weak_cache_set
  -> load_verified_leo_weak_cache
  -> shared _REQUIRED_ARRAY_KEYS check
```

结论：r2的外层`accepted_schemas=(v1,)`允许schema枚举，但内层loader在读取manifest/schema分支前仍统一要求v2新增的两个逐行成员；所以“v1-only兼容入口”未真正兼容历史v1 member contract。

这不是NPZ SHA漂移、缓存缺失、GPU异常、环境异常或协议拒绝。三个v1 NPZ的历史provenance可由现有字段完整自洽复算，详见`v1_cache_structure_audit.md`。

## 产物与本地回收

- `phase1_feature_archive`目录：远端不存在。
- feature archive NPZ：未生成。
- feature archive manifest：未生成。
- 因无完整产物，不能进入LODO配置或实验。

wrapper artifact：

|文件|字节|SHA256|
|---|---:|---|
|`logs/exporter.log`|3585|`c81b8644b48d14b3f7b050f9a5a97fe5145e34e743b36b64ba9e16dcad8e9e92`|
|`logs/exporter.pid`|8|`f299f6507a4dcbee8ba50f809a3b8f586ae5027e4ec246761a3f1bc7e18e9f6e`|
|`logs/exporter.exit`|2|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|
|`input/d99_d100_phase1_selection_salt.json`|440|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|

本地还保存`authority/cache_set.json`及`v1_cache_structure_audit.md`。未下载31MB源码ZIP或三个大NPZ。

## 后续修复约束

1. 不得伪造历史v1缺失的`source_record_indices`，也不得把v1宣称为v2数据集SHA+record-index绑定语义。
2. 可行修复是按observed schema在member contract及`sample_overlay_provenance_fields`检查处显式分支：
   - v1使用历史`role|tx|rx|day|eq|sig`公式；
   - v2继续使用当前严格字段及语义；
   - 两者均校验逐行sample ID、唯一性、root、IQ哈希、overlay ID、manifest SHA及外层cache audit。
3. 必须新增真实v1 fixture回归测试，确保不是只在合成v2-shaped fixture上把schema字符串改成v1。
4. 修复、测试、Git版本化和重新预登记后才能创建新run ID；本r2不可覆盖或重启。

## 下一LODO字段状态

以下四项因feature archive未生成而全部不可填：

- `feature_archive_path`
- `feature_archive_sha256`
- `feature_archive_manifest_path`
- `feature_archive_manifest_sha256`

r1 ground路径可继续沿用，但不得单独启动LODO：

- `ground_bundle_npz_path=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.npz`
- `ground_bundle_npz_sha256=e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`
- `ground_release_manifest_path=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.release_manifest.json`
- `ground_release_manifest_sha256=f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`
- `base_d99_lock_path=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/base_d99_lock.json`
- `base_d99_lock_sha256=7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`
- `checkpoint_sha256=2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`

## SSH连接终态

每次SSH/SCP后均执行本地连接核查；最后观测为：

```text
ssh_processes=0
n607_established=0
```
