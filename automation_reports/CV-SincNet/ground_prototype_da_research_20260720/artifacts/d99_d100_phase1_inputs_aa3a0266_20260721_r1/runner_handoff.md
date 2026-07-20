# D99/D100 Phase1输入发布runner handoff

## 结论与状态链

- run ID：`d99_d100_phase1_inputs_aa3a0266_20260721_r1`
- 唯一launch owner：`/root/n607_d99_d100_release_prep`
- 代码基线：`aa3a02662b5609d633fdf2ce1bcfde4c3bab0efb`
- 发布报告提交：`9f4ec377e084be4f4290b124e90209bcb669d924`
- 状态链：`LOCAL_VERIFIED→LANDED→RUNNING→STOPPED_EXPORTER_INPUT_SCHEMA_MISMATCH`
- `ARTIFACTS_COMPLETE`：**未达到**。exporter exit1且没有feature archive；builder exit0并完成五文件内部验证。
- 行为边界：两个child各启动一次；没有改方法、参数、输入或run ID；没有重启失败任务；没有干预其他作业；没有生成LODO config或访问target/query。

## Preflight与落地证据

2026-07-21T01:38:18+08:00执行规定的`tools/n607_ssh_preflight.ps1`并通过。N607项目根为`/home/szu2070436088/2510044040/CV-SincNet`，Python为3.10.19，Torch为2.1.0。8张RTX3090均为0%利用率、10MiB显存占用，`nvidia-smi`没有compute app；没有D99/D100相关进程。以下远端目标在创建前均不存在：

- `/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1`
- `/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_inputs_aa3a0266_20260721_r1`
- `<run>/source_aa3a0266`

本地与远端输入均复算一致：

|输入|字节/成员|SHA256|
|---|---:|---|
|`source_aa3a0266.zip`|31,121,736B；4,314成员|`5185e2847b09191419ea58c010c214e6954faea8c6ebde31880dc39e1bc4640c`|
|`d99_d100_phase1_selection_salt.json`|440B|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|
|source-validation`cache_set.json`|4,501B|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|ADV3B02 runtime|4,613,201B|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|
|ADV3B02 checkpoint|—|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|D19 manifest|2,391B|`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|
|D19 INT8 NPZ|5,363B|`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`|

远端固定Python逐项拒绝绝对路径、`..`、反斜杠、重复目标和符号链接后安全解压4,314成员到隔离source；无路径逃逸。目标脚本`py_compile`和import smoke均通过。关键隔离源码SHA：

|文件|SHA256|
|---|---|
|`export_phase1_singleobs_feature_archive.py`|`5a45d9e47ef27b600a329207ddd60844029d33e6772cafeada99b15314dd9079`|
|`build_d99_receiver_ground_bundle.py`|`5751d51a4cf766d3d2e7c8091ce75dae36a294fb0ffb5bed488ee7448dfe5b26`|
|`stage2_d99_d100_phase1_lodo.py`|`aa99b3d726338481ed7f22f4acc5cdf2cfe4b2ef420e44da6f2ff2f674841e0e`|
|`stage2_d99_ra_cgtmk_d81.py`|`c166a5e375b0b8be5c95e678e63a6f04526474cd1a01544616829106af52f56f`|
|`stage2_d100_ra_cgspr_lgf.py`|`86c185ee13222bc0c97c4576984b9cd07f981201da4f0b62f8d4bc66970b4714`|

## Wrapper与child记录

两个job使用同一不可覆盖wrapper模板；`.log`由外层`set -C`重定向创建，`.pid`由外层`set -C`创建，`.exit`由child wrapper的`set -C`创建：

```bash
set +e
source_root=$1
exit_file=$2
job_name=$3
gpu_label=$4
shift 4
cd "$source_root"
cd_rc=$?
started_at=$(date -Ins)
printf 'wrapper_job=%s\nstarted_at=%s\nchild_cwd=%s\ngpu=%s\n' "$job_name" "$started_at" "$PWD" "$gpu_label"
printf 'child_command='; printf '%q ' "$@"; printf '\n'
if [ "$cd_rc" -ne 0 ]; then rc=$cd_rc; else "$@"; rc=$?; fi
ended_at=$(date -Ins)
printf 'ended_at=%s\nchild_exit=%s\n' "$ended_at" "$rc"
set -C
printf '%s\n' "$rc" > "$exit_file"
exit "$rc"
```

实际wrapper形态为：

```bash
nohup bash -c "$WRAPPER_BODY" <wrapper-name> <source-root> <exit-file> <job-name> <gpu-label> <exact-child-argv...> > <log-file> 2>&1 < /dev/null &
printf '%s\n' "$!" > <pid-file>
```

|child|PID|GPU|开始|结束|exit|CWD|
|---|---:|---|---|---|---:|---|
|exporter|1390812|`cuda:4`|2026-07-21T01:44:25.815938779+08:00|2026-07-21T01:44:27.345600407+08:00|1|`<run>/source_aa3a0266`|
|ground builder|1390814|CPU|2026-07-21T01:44:25.817752364+08:00|2026-07-21T01:44:27.939992997+08:00|0|`<run>/source_aa3a0266`|

完整exact child command保存在回收的`logs/exporter.log`和`logs/ground_builder.log`首部，逐参数与预登记报告一致。2026-07-21T01:50:33+08:00终态快照确认两个PID均不存在，GPU4为0%/10MiB，没有compute process。

## Exporter失败证据

exporter在cache loader入口失败，尚未进入runtime特征前向：

```text
ValueError: LEO cache-set manifest contract failed: ['schema']
```

冻结输入`cache_set.json`实际schema为`cvs_leo_weak_iq_cache_set_v1`；提交`aa3a0266`中的`leo_weak_cache.py`固定要求`cvs_leo_weak_iq_cache_set_v2`。因此`<run>/phase1_feature_archive`未创建，两个预期文件均不存在：

回收的原始manifest SHA仍为`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`。顶层字段按文件顺序为：`schema`,`artifact_stage`,`cache_set_id`,`cache_scope`,`phase2_sample_view_policy`,`clean_sample_access`,`clean_derived_signal_access`,`target_channel_view`,`target_channel_scenarios`,`output_roles`,`cache_npz_by_scenario`,`cache_sha256_by_scenario`,`cache_audits`,`physical_sample_ids_sha256`,`builder_sha256`,`build_spec_sha256`,`build_spec_path_exposed_to_phase2`；`cache_scope=source_validation`。三个scenario成员绑定为：

|scenario|相对NPZ路径|SHA256|
|---|---|---|
|`leo_clear_weak`|`leo_clear_weak.npz`|`18a4ed923d8438ef2d69ff4226f46281b56191409582d24e79485fd97688179f`|
|`leo_low_elev_weak`|`leo_low_elev_weak.npz`|`a82f37034f27a23cb0f45ab849807b9cb13b4ce3e79d0582403ed0aa5e946712`|
|`leo_rain_weak`|`leo_rain_weak.npz`|`2de300f81246f03c6a10a21301ec31bc3c15bf595e5aecaf2ba7667664210b4b`|

- `phase1_singleobs_feature_archive.npz`
- `phase1_singleobs_feature_archive.manifest.json`

这不是GPU OOM或进程中断。按预登记停止条件没有自动重试、schema转换、换输入、改参数或换run ID。

## Ground builder完整产物

远端固定Python按单测同构路径执行：NPZ精确成员加载→`ExternalGroundAggregationReceipt`→typed ground bundle→development ground authority→development prior lock→narrow loader，并复算`build_result.json`中的所有SHA。结果为`internal_verify=PASS`。

|文件|字节|SHA256|
|---|---:|---|
|`build_result.json`|814|`b9c0213f9dbbd2f438ab31541ac4404d852388bd42fbe439b9a768fbf87714c2`|
|`d99_base_method_lock_dev.json`|1,430|`7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`|
|`d99_ground_aggregation_spec.json`|1,657|`f4db8091aeb7204bb4a641d02810c19acbb9fc002ec8e82c82fd9f4fe2820efe`|
|`d99_ground_bundle_dev.manifest.json`|2,073|`f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`|
|`d99_ground_bundle_dev.npz`|8,660|`e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`|

内部固定点：

- typed bundle SHA：`78904ed3c569f79815021ef76cc9e46bb42fd0a06c7bb1e91483237e2606ce78`
- aggregation receipt SHA：`5ba0731a9ebb87fc330075ddc0c8913e045b053c3a8112022cd23d86f0c048ac`
- narrow loader lock digest：`9d12c638176e0bc7dfa2d27664f737e606ec2064eb8d280a9a059d91c2122063`
- release schema/status：`cvs.phase1.d99.ground_release_manifest.development.v1`/`PREREGISTERED_DEVELOPMENT_GROUND_AGGREGATE_NONFORMAL`
- authority status：`BLOCKED_DEVELOPMENT_GROUND_RELEASE`
- domain顺序：`1-1,1-19,14-7,18-2,19-2,2-1,2-19`
- class顺序：`14-10,14-7,20-15,20-19,6-15,8-20`
- shapes：codes`[7,6,160]`、scales/mask/count`[7,6]`
- mean/min requantization cosine：`0.9999880194664001`/`0.9999678134918213`

## Wrapper证据SHA

|文件|字节|SHA256|
|---|---:|---|
|`logs/exporter.log`|2,950|`b6762c46a0ebd1e8bb9313c6e5987a45032bc4e3997072ecbd1fbd80fc3e3ccb`|
|`logs/exporter.pid`|8|`dc30775cc91037015f21ab36619f6f0741d708ea3006df803f741432a1ec8286`|
|`logs/exporter.exit`|2|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|
|`logs/ground_builder.log`|1,928|`95920fa75b7e0a63b533be4c0f704367ad861ecb2c04b79a2215601f106fcf73`|
|`logs/ground_builder.pid`|8|`d79ab14928cc63fb02e62a1c6b9bd99dfe32e7ac5f46bb41a4cee8391132b44d`|
|`logs/ground_builder.exit`|2|`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`|

回收目录：`E:\type10-7\code\snapshots\ground_proto_da_rd_wt\automation_reports\CV-SincNet\ground_prototype_da_research_20260720\artifacts\d99_d100_phase1_inputs_aa3a0266_20260721_r1`。本地复算SHA与远端逐项一致。`input_evidence/cache_set.json`SHA为`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`。

## 下一步LODO config字段

本run禁止生成或启动LODO config，因为四个必要archive字段不存在：

|字段|值|
|---|---|
|`feature_archive_path`|`UNAVAILABLE_EXPORTER_EXIT_1`|
|`feature_archive_sha256`|`UNAVAILABLE_EXPORTER_EXIT_1`|
|`feature_archive_manifest_path`|`UNAVAILABLE_EXPORTER_EXIT_1`|
|`feature_archive_manifest_sha256`|`UNAVAILABLE_EXPORTER_EXIT_1`|
|`ground_bundle_npz_path`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.npz`|
|`ground_bundle_npz_sha256`|`e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`|
|`ground_release_manifest_path`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_ground_bundle_dev.manifest.json`|
|`ground_release_manifest_sha256`|`f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`|
|`base_d99_lock_path`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_inputs_aa3a0266_20260721_r1/d99_receiver_ground_bundle/d99_base_method_lock_dev.json`|
|`base_d99_lock_sha256`|`7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`|
|`phase1_checkpoint_sha256`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|

在新的本地代码修订、专项验证、Git提交、报告预登记及全新不可覆盖run ID出现前，不得对本run重试或补写archive。

## 异常与SSH卫生

- 一次启动后只读探针因本地PowerShell提前展开远端变量而未读取状态；该探针没有远端写操作。随后改为绝对路径短连接，得到可信终态。
- 每次SSH/SCP结束后均检查本地`ssh.exe`和到N607/bridge的TCP22连接。最终结果：`ssh_process_count=0`，`n607_bridge_established_count=0`。
- 本文件和回收证据未写入主`report.md`，未执行Git提交。
