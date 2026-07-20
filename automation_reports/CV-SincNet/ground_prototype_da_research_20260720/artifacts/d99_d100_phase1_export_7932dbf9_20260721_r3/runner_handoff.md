# D99/D100 Phase1 exporter r3 runner handoff

## 最终裁决

- run ID：`d99_d100_phase1_export_7932dbf9_20260721_r3`
- 状态链：`LOCAL_VERIFIED -> LANDED -> RUNNING -> FAILED_DIAGNOSTIC`
- 子进程退出码：`1`
- 失败阶段：真实v1 loader全部通过后的首个TorchScript forward
- 未达到：`ARTIFACTS_COMPLETE`、archive内部验证、LODO
- 自动重启/重试：无
- 参数修改：无
- r1/r2补写：无
- LODO启动：无

## 时间、进程与资源

|字段|实际值|
|---|---|
|N607 preflight时间|`2026-07-21 03:00:15 CST`|
|wrapper开始|`2026-07-21T03:07:05.283113353+08:00`|
|wrapper结束|`2026-07-21T03:07:08.954731261+08:00`|
|PID|`1427919`|
|CWD|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/source_7932dbf9`|
|device|`cuda:4`|
|CPU threads|`OMP=2,MKL=2,OPENBLAS=2`|
|batch size|`256`|
|启动前GPU4|`0%`,`10/24576 MiB`|
|终态GPU4|`0%`,`10/24576 MiB`|
|终态进程|PID已退出|

preflight确认8张RTX3090均为`0%`、`10/24576 MiB`，无compute process。所有SSH/SCP均为短连接；最后本地核查：

```text
ssh_processes=0
n607_established=0
```

## 冻结版本、源码落地与输入

|项|值|结果|
|---|---|---|
|代码commit|`7932dbf9`|匹配预登记|
|报告commit/HEAD|`322973426dec505d00dbd02b7f96fb3c1bd5b6dd`|匹配预登记|
|源码ZIP SHA256|`40a2ecc9d01a12759d9a67693d7eaa974751bae02c05faab0dabfc580efdbd72`|本地/远端一致|
|源码ZIP大小|`31168653 bytes`|一致|
|ZIP成员数|`4342`|一致|
|安全解压|`4342 members`,`3837 files`|路径穿越、绝对路径、反斜杠、重复target、symlink均为0|
|selection salt SHA256|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|本地/远端一致|
|cache set SHA256|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|匹配|
|runtime SHA256|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|匹配|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|匹配|

r3 run、log与source路径创建前均确认不存在。上传后ZIP/salt再次验SHA。隔离源码`py_compile`通过，`cvsrffi.leo_weak_cache` import通过。

### ZIP成员口径执行身份

|文件|ZIP成员/远端解压SHA256|Windows工作树SHA256|
|---|---|---|
|`code/cvsrffi/leo_weak_cache.py`|`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`|`3fc35aeea182560fc67cd468a7615ca110b528ca210327c0620370d1b68606fb`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`|`ab4d3c40251f2bd147e7948ced392d185d0ef7b3f45c18924e7ab1bd457dac6d`|
|`tests/test_export_phase1_singleobs_feature_archive.py`|`493c6edd167f743aaf797202e530d65f01a8a47107353eb0b2c00df720031adb`|`9b3dd452b195b409b922442d4ec7fcadd7ba1de2bbce104fc8c3453f7f2e2be8`|

远端执行字节以预登记Git archive ZIP为准；三项ZIP成员SHA与远端解压文件逐字节一致。Windows工作树SHA仅记录换行口径差异。

### r1 ground五文件只读复核

|文件|SHA256|
|---|---|
|`build_result.json`|`b9c0213f9dbbd2f438ab31541ac4404d852388bd42fbe439b9a768fbf87714c2`|
|`d99_base_method_lock_dev.json`|`7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`|
|`d99_ground_aggregation_spec.json`|`f4db8091aeb7204bb4a641d02810c19acbb9fc002ec8e82c82fd9f4fe2820efe`|
|`d99_ground_bundle_dev.manifest.json`|`f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`|
|`d99_ground_bundle_dev.npz`|`e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`|

## 唯一child command

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/source_7932dbf9/code/scripts/export_phase1_singleobs_feature_archive.py --mode development --cache-set /home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json --cache-set-sha256 125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74 --runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --expected-runtime-sha256 f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a --class-ids 14-10,14-7,20-15,20-19,6-15,8-20 --selection-salt-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/input/d99_d100_phase1_selection_salt.json --selection-salt-receipt-sha256 38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_20260721_r3/phase1_feature_archive --device cuda:4 --batch-size 256
```

wrapper使用不可覆盖的`exporter.log`、`exporter.pid`和`exporter.exit`，只启动一次。通过stdin发送的本地launcher末尾CRLF使启动命令在打印`LAUNCHED pid=1427919`后额外产生一次`$'\r': command not found`并令外层SSH返回1；PID、日志和唯一wrapper文件证明child已正常单次启动，因此没有重发launcher。该外层传输噪声与child失败根因无关。

## 真实v1 loader验证

r3 child已越过`_load_verified_v1_only_source_validation_cache_set`并进入`_forward_torchscript`。随后用同一r3源码、同一cache set和同一生产loader做只读审计，完整通过；不是绕过式metadata检查。

### 外层

- observed outer schema：`cvs_leo_weak_iq_cache_set_v1`
- outer cache SHA256：`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`
- physical sample count：`8400`
- physical observation count：`25200`
- 三个场景physical root均为`d2def2acf96a9338f94b4626f77ca9b7b106a65f41615dd5c703b1b76461e1a3`

### 每场景

三个NPZ均为精确17-member、8400行，ordered member list相同：

```text
leo_weak_iq,raw_labels,domain_labels,tx_ids,rx_ids,day_ids,eq_ids,
sig_ids,dataset_role,channel_views,sat_scenarios,satellite_seeds,
overlay_applied,sample_ids,post_channel_iq_sha256,overlay_ids,manifest_json
```

|场景|inner schema|legacy|NPZ SHA256|physical root|IQ root|overlay root|manifest SHA256|
|---|---|---|---|---|---|---|---|
|`leo_clear_weak`|`cvs_leo_weak_iq_cache_v1`|true|`18a4ed923d8438ef2d69ff4226f46281b56191409582d24e79485fd97688179f`|`d2def2acf96a9338f94b4626f77ca9b7b106a65f41615dd5c703b1b76461e1a3`|`eb658fc62342e6f6cec36339b541190238d2da8601143cb8de529e9a9154c4f6`|`539ca4f877adc69ea9617c66d0809a1f64703728a759c93c7ca66eed66ddf0b3`|`e098566428e59e4e1320ed42e78ff73864f15b8052460c05a5cd6ca8c0858211`|
|`leo_low_elev_weak`|`cvs_leo_weak_iq_cache_v1`|true|`a82f37034f27a23cb0f45ab849807b9cb13b4ce3e79d0582403ed0aa5e946712`|同上|`8e389ff05b265d5e356a1d315a44586eabf1bd58b314e0198b2c5423216703ef`|`180824fc7a791e23f913583968ea0bcb119fe636fcb06b6174da1b97d4584f92`|`811f7ca882505c836dd84fe3a7bc35a2dda0de58c4b44ad5597a1a71462d4c6b`|
|`leo_rain_weak`|`cvs_leo_weak_iq_cache_v1`|true|`2de300f81246f03c6a10a21301ec31bc3c15bf595e5aecaf2ba7667664210b4b`|同上|`47eb3dd877fe0e966b75292b58e05a7e0b60593f73bde02abfabfa63cc3ce319`|`826f462dabab01972576d07d33b4653f1696333382cb7c4781af0fb6475900e0`|`d0a64adcd3afd193ec64c92231d603d862044b605078265c71924151211374b6`|

每场景audit同时记录：`forbidden_members_checked_before_iq_read=true`、`clean_sample_access=false`、root policy=`role|tx|rx|day|eq|sig`。生产loader逐行重算并通过sample ID、IQ digest、overlay ID、唯一性、inner manifest和outer root检查。

## loader输入分布

由于首批模型前向失败，以下是三个v1 loader输入各自的分布，不是最终selection/archive分布。三个场景的分布完全相同，只有场景标签不同。

|维度|计数|
|---|---|
|class|`14-10:1400`,`14-7:1400`,`20-15:1400`,`20-19:1400`,`6-15:1400`,`8-20:1400`|
|receiver|`1-1:1195`,`1-19:1217`,`14-7:1198`,`18-2:1186`,`19-2:1215`,`2-1:1230`,`2-19:1159`|
|day|`2021_03_01:2147`,`2021_03_08:2115`,`2021_03_15:2082`,`2021_03_23:2056`|
|scenario|对应场景各`8400`|
|physical IDs|每场景`8400`个唯一ID；三场景顺序相同|

selection后的“一physical ID仅一个固定scenario”步骤未形成产物，不能宣称通过或给出selection后场景分布。

## 精确新根因

失败堆栈位置：

```text
_export_impl:864
  -> _forward_torchscript:696
  -> model(tensor)
  -> TorchScript id_backbone.forward
  -> torch._convolution
```

终止异常：

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices,
cuda:4 and cuda:0! (when checking argument for argument weight in method
wrapper_CUDA__cudnn_convolution)
```

可确认事实：冻结命令把输入送至`cuda:4`，TorchScript卷积weight仍位于`cuda:0`。这不是v1 member/provenance、cache/runtime SHA、GPU占用、Conda环境或数据协议错误。runner不擅自改为GPU0、不增加`CUDA_VISIBLE_DEVICES`映射、不修改runtime/exporter，也不重启。

## 产物与本地回收

- 远端`phase1_feature_archive`：不存在。
- archive NPZ：未生成。
- archive manifest：未生成。
- archive行数/receiver/day/class/scenario分布：不可用；上文仅报告loader输入分布。

|本地文件|字节|SHA256|
|---|---:|---|
|`logs/exporter.log`|6341|`35d0a76cd01e69f2dc8eb43b4ca3c156d3ee7941c3ef88d0dbc2ffa28ae68ff2`|
|`logs/exporter.pid`|8|`4ecae3d474652d46ff186d4d5ef70c4e4d85da3cd7e1b7f61037683de0be30a2`|
|`logs/exporter.exit`|2|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|
|`input/d99_d100_phase1_selection_salt.json`|440|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|
|`authority/cache_set.json`|4501|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|

未下载31MB源码ZIP或三个大NPZ。

## LODO四字段

本轮失败，四项均不可提供：

- `feature_archive_path=UNAVAILABLE`
- `feature_archive_sha256=UNAVAILABLE`
- `feature_archive_manifest_path=UNAVAILABLE`
- `feature_archive_manifest_sha256=UNAVAILABLE`

LODO不得启动。
