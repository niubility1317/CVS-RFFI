# GEOFF/r2.1 Phase1双表征归档与coverage技术run报告

## 1.身份与当前状态

- run_id:`rchm_bpp_p1_dual_archive_cli_r3_9842ff42_20260723_r3`
- 预注册时间:`2026-07-23T00:30:00+08:00`
- 主agent:`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- candidate:`P1-DUAL-ARCHIVE-GEOFF/r2.1-CLI`
- protocol:`p2_min_v1`；数据状态:`VALIDATED_ONCE_REUSED`
- 当前状态:`ANALYZED / TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- retry:`NO`

本run不改变received IQ字节、physical ID、receiver/TX集合、scene、K、support/query split或protocol schema，因此不重复数据验证。唯一技术delta是将五个archive CLI路径参数显式映射到既有函数`*_path`契约；不改变archive内容、runtime、manifest、coverage门、权限或方法机制。

## 2.目标、假设与matched failure

目标是在N607同一Torch/CUDA/device合同下重新生成strict ADV3B02 base/candidate dual runtime、独立base parity receipt/vector、8400行Phase1 single-observation dual archive/manifest及只读coverage receipt。

matched failure为`rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2`：PID=`357994`、自然exit=`1`；export与base parity均PASS，其中base parity batch=`[1,8,256]`、每batch调用3次、maxabs=0，但archive入口因`unexpected keyword argument 'cache_set'`失败，archive/coverage/prediction均未生成。该run永久为`ANALYZED / TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，本run不复用其远端目录、PID、输出或run ID。

本run假设：修正CLI→函数签名映射后，冻结wrapper可进入既有archive核心；若任何后续技术门失败，仍立即标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不补跑、不放宽门。

## 3.Git、源码包与本地验证

|项目|冻结证据|
|---|---|
|方法修复commit|`91e6930ee8f7d56895997906f5ff5cab60af2cb6`|
|commit-bound源码|`9842ff42cb3d2cd0ee9c28aab7e828a7cc725a7e`|
|分支|`codex/ground-prototype-da-rd`|
|独立review|`P0=0,P1=0,P2=0→MERGE`|
|专项测试|`ssr-gpu`中`py_compile`与`tests/test_export_phase1_singleobs_dual_feature_archive.py`通过，`7 passed`|
|wrapper静态门|`bash -n`通过；未分配CUDA时exit=`70`|
|源码ZIP|`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_cli_r3_9842ff42_20260723_r3/source_9842ff42.zip`|
|ZIP SHA256|`6ae7adf4d947ac4b94e599f94cb393ad0ae5d572ef4118b8aacf127c0cd18f26`|
|ZIP大小/成员|`33,017,368B / 4,439`|
|wrapper SHA256|`b2075fc246c97179b85dfabb6b86556eebde3e3788a113d7d4da7f867fcd9b38`|

冻结源码成员SHA256：

|相对路径|SHA256|
|---|---|
|`code/scripts/export_phase1_singleobs_dual_feature_archive.py`|`c3e81a7b778d4981652a90752e7c37dcb477c005ff3f1230206698cb5aaba5a0`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`|
|`code/scripts/export_adv3b02_dual_feature_torchscript.py`|`e2cbc0ce19402e8c665489fb2b13bb63f988c14e138cbc39dd20f7c9e2b12090`|
|`code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py`|`0a3e80b226997b353c577de94aa8c8e92fb25f5bf13d8b81c9bc27f448ef284b`|
|`code/cvsrffi/dual_feature_forward.py`|`1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56`|
|`code/cvsrffi/leo_weak_cache.py`|`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`|

本地验证命令：

```text
conda activate ssr-gpu
python -m py_compile code/scripts/export_phase1_singleobs_dual_feature_archive.py
python -m pytest -q tests/test_export_phase1_singleobs_dual_feature_archive.py
bash -n automation_reports/CV-SincNet/rchm_bpp_p1_dual_archive_cli_r3_9842ff42_20260723_r3/run_pipeline.sh
```

## 4.N607发布合同

- direct preflight:`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
- remote run root:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_cli_r3_9842ff42_20260723_r3`
- remote source root:`.../rchm_bpp_p1_dual_archive_cli_r3_9842ff42_20260723_r3/source_9842ff42`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- sync仅包含上述ZIP和`run_pipeline.sh`，分别落到新run的`input/`。
- runner须先确认run root不存在、记录GPU占用、校验ZIP/wrapper/4项冻结输入/6项源码SHA、安全解包、远端`py_compile`和`bash -n`。
- 唯一启动命令形态：
  `nohup env CUDA_VISIBLE_DEVICES=<runner-selected> bash <run-root>/input/run_pipeline.sh > <run-root>/logs/pipeline.log 2>&1 < /dev/null &`
- runner拥有preflight、同步、验证、单次启动、短连接监控、自然exit、artifact回收和报告回填；主agent不得并发启动同一run。
- 未授权retry、restart、kill、远端编辑、调参、target/query/held/125访问或无关作业干预。

冻结输入：

|输入|远端路径|SHA256|
|---|---|---|
|checkpoint|`.../runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|adapter|`.../runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/effective8_adapter_fp16.pt`|`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`|
|source-validation cache_set|`.../runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|selection salt|`.../runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json`|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|

## 5.预期artifact与硬门

预期输出：

- `output/runtime/base_dual_runtime.pt`
- `output/runtime/candidate_dual_runtime.pt`
- `output/runtime/dual_export_receipt.json`
- `output/runtime/base_parity_receipt.json`
- `output/runtime/base_parity_vector.json`
- `output/archive/phase1_singleobs_dual_feature_archive.npz`
- `output/archive/phase1_singleobs_dual_feature_archive.manifest.json`
- `output/coverage_receipt.json`
- `output/sha256sums.txt`
- `logs/pipeline.pid`、`logs/pipeline.exit`、`logs/pipeline.log`

技术完成要求：child exit=0；日志含`PIPELINE_ARTIFACTS_COMPLETE`；export/parity/archive execution contract逐值与SHA闭合；parity batch=`[1,8,256]`、每batch三次runtime调用、三输出maxabs均`≤1e-5`；archive manifest v2且内部verify通过。

coverage冻结门：row=8400、unique physical=8400、unique observation=8400、class/receiver/day/scenario=`6/7/4/3`、receiver×day×class cells=168、zero cells=0、min cell>10，并只读元数据报告K1/K5/K10余量。`feature_arrays_read=[]`、`held_fold_selected=false`。coverage只描述Phase1归档，不是性能结果。

任一门失败：不生成完成marker，状态为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。全部通过也只能到`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`；不得把测试、receipt、archive或coverage当作性能成功。

## 6.完成后回填

|字段|结果|
|---|---|
|release-control Git commit|`6fa015c27bc8048603b59f87d0a46514348eece4`（主agent负责提交本报告更新）|
|runner/route|`/root/geoff_r21_n607_runner / direct N607`|
|preflight/GPU|`PASS / GPU0`；启动前GPU0-7均0%/10MiB且无compute app|
|remote ZIP/wrapper/member SHA|`PASS / 6ae7adf4...c0cd18f26 / b2075fc2...fcd9b38 / 4439`|
|remote compile/bash|`PASS / 6项源码SHA匹配、py_compile、bash -n`|
|PID/CWD/cmdline|`371059`；`runs/rchm_bpp_p1_dual_archive_cli_r3_9842ff42_20260723_r3`；`nohup env CUDA_VISIBLE_DEVICES=0 bash input/run_pipeline.sh`|
|child exit/marker|`1 / PIPELINE_ARTIFACTS_COMPLETE=ABSENT`|
|parity receipt/vector|`PASS / cvs.phase1.adv3b02_dual_runtime_checkpoint_parity_receipt.v2 / eea06edf...635acb / vector root=86a9a69e...5c44a4 / batch=[1,8,256] / calls=3 / maxabs=0`|
|archive/manifest|`NOT_GENERATED`|
|coverage receipt|`NOT_GENERATED`|
|prediction count|`0`|
|回收路径/SHA|`recovered/`；详见第7节|
|最终状态|`ANALYZED / TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

coverage通过后，主agent将以其真实SHA冻结最小held性能矩阵并使用另一个全新run ID发布；本技术run不包含性能预测。

## 7.自然退出、回收与结论

唯一启动于`2026-07-23`经direct N607完成，GPU0、wrapper PID=`371059`、child Python PID=`371110`。启动后自然exit=`1`，未retry、restart、远端编辑或干预其他作业。失败前export与base parity均输出`PASS`，随后archive入口在`export_phase1_singleobs_dual_feature_archive.py:386`的`model(torch.from_numpy(chunk).to(device))`自然抛出`TypeError: expected np.ndarray (got numpy.ndarray)`；调用链为`619→main`、`615→export_phase1_singleobs_dual_feature_archive`、`551→_forward_once_per_selected_iq_batch`、`386`。

|candidate/run|机制|GPU/seed|export与parity|archive/coverage|预测|同row结论|
|---|---|---|---|---|---|---|
|`P1-DUAL-ARCHIVE-GEOFF/r2.1-CLI`/`rchm_bpp_p1_dual_archive_cli_r3_9842ff42_20260723_r3`|Phase1双表征runtime导出、archive、只读coverage|GPU0/20260721|export schema=`cvs.phase1.adv3b02_dual_feature_torchscript_export.v2`，SHA=`302faad0068346d1b9f7f509bc631928ed701b519b4a114edc1728d9215b7b64`，maxabs=`2.384185791015625e-06`≤`1e-5`；base parity schema=`cvs.phase1.adv3b02_dual_runtime_checkpoint_parity_receipt.v2`，SHA=`eea06edf8631caad44c05687ffe16eafb23a75ccd5ddbcdb0b1d423578635acb`，batch=`[1,8,256]`、calls=`3`、maxabs=`0`|archive、manifest、coverage、sha256sums、completion marker均未生成；8400/6/7/4/3/168/min/K余量无可用receipt|`0`|`ANALYZED / TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

回收清单已落至`recovered/`，本地SHA与远端一致：`pipeline.log=79685b4b8cbbf399c1abbb4a84e0dcc56d1a8222b625facbfdfee25a0b027875`，`pipeline.exit=4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`，`pipeline.pid=9872e40fd04e96c73cf0fb2bc4eed7a31b23a2d688b5722aaa51c987a45ab0e8`，`base_dual_runtime.pt=8ae25df6e088e73ac276eee7d37767d4ec28b7da1f084c475943a5af75e1c29a`，`candidate_dual_runtime.pt=7ba7b4f3b72c5d32a3b0e924ab59187496d4ad19fb819a947c356b6c1a35c883`，`dual_export_receipt.json=302faad0068346d1b9f7f509bc631928ed701b519b4a114edc1728d9215b7b64`，`base_parity_receipt.json=eea06edf8631caad44c05687ffe16eafb23a75ccd5ddbcdb0b1d423578635acb`，`base_parity_vector.json=c667fff8d33a4ec447ac58bf179a08d515df71a65bd1116074b9a4fe7929bde7`。远端定向清单确认无prediction文件。没有archive/manifest/coverage/sha256sums/completion marker可回收，故不存在8400、6、7、4、3、168、min cell或K余量的coverage证据。
