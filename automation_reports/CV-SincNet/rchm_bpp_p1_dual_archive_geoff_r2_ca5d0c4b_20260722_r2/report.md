# GEOFF/r2 Phase1双表征归档与coverage技术run报告

## 1.身份与当前状态

- run_id:`rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2`
- 预注册时间:`2026-07-22T23:48:22+08:00`
- 主agent:`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- protocol:`p2_min_v1`；数据状态:`VALIDATED_ONCE_REUSED`；本run不改变received IQ、physical ID、receiver/TX、scene、K、support/query split或schema，因此不重复数据验证。
- objective:在N607同一Torch/CUDA/device合同下执行`P1-DUAL-ARCHIVE-GEOFF/r2`，生成strict ADV3B02 base/candidate dual runtime、独立base parity receipt/vector、8400行Phase1 single-observation dual archive/manifest及只读coverage receipt。
- hypothesis:在首次JIT边界前fail-closed设置并严格回读`graph_executor_optimize=false`，可消除r1的TorchScript CUDA冷/热执行计划漂移，同时保持batch1/8/256、三输出、三次runtime调用的`maxabs≤1e-5`冻结门。
- matched failure:`rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1`永久为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；其256行maxabs为`z_id=1.9640e-4`、`z_dom=5.0431e-4`、`tx_logits=3.2592e-3`，未生成parity receipt、archive、coverage或prediction。本run使用全新ID，绝不复用r1目录。
- 当前状态:`LOCAL_VERIFIED / NOT_LANDED / NO_PERFORMANCE_RESULT`

## 2.冻结范围与禁止项

唯一delta是GEOFF/r2 execution contract；export、parity和archive schema为各自v2，runtime tensor output schema保持v1。exporter、verifier和archive consumer必须在首个JIT边界前设置并回读`graph_executor_optimize=false`，绑定policy、setter/getter、readback、Torch/CUDA版本、device、`max_abs=1e-5`与canonical contract SHA。API缺失、异常、回读非False、版本/contract/SHA漂移或任一数值门失败均立即停止。

本run仅访问既有Phase1 source-validation received-IQ cache及冻结checkpoint/adapter/salt；不访问target/query、clean/source runtime sidecar、held prediction或125，不做方法选择、调参、数据重验、bundle生成或性能声明。coverage脚本只读archive元数据数组，`feature_arrays_read=[]`。

## 3.Git、源码包与本地验证

- 方法提交:`ca5d0c4bcf8fb295cdfb70e067f9009617bb3a5f`
- 分支:`codex/ground-prototype-da-rd`
- 提交前实现审查:`P0=0,P1=0,P2=0→MERGE`
- 本地验证:`ssr-gpu`中`py_compile`通过；GEOFF专项35项、含dual-forward/joint core相邻回归共`48 passed`；本地回收runtime无数据CUDA探针在batch1/8/256的第1↔2及第1↔3三输出最大差全0。
- Git archive:`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2/source_ca5d0c4b.zip`
- archive大小:`33,007,669B`；成员数:`4,436`；SHA256:`5adbef8a1ebf2f0846132226f702e95648c99334a0ba5296b7487e45095e4778`
- wrapper:`run_pipeline.sh`；SHA256:`e1f497a757d54cef95a9559ac3de910a26cf2d9a3d0407d3cc865b628847afcf`
- 旧r1解包树已标记`CONTAMINATED_LOCAL_EXTRACTION / DO_NOT_RELEASE`；本run源码只来自上述Git archive，且本地不解包。

|冻结成员|SHA256|
|---|---|
|`code/scripts/export_phase1_singleobs_dual_feature_archive.py`|`f98dbcb2665c57acd5007cc3f08a14588f626ba88126804fea57d2d6399864fc`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`|
|`code/scripts/export_adv3b02_dual_feature_torchscript.py`|`e2cbc0ce19402e8c665489fb2b13bb63f988c14e138cbc39dd20f7c9e2b12090`|
|`code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py`|`0a3e80b226997b353c577de94aa8c8e92fb25f5bf13d8b81c9bc27f448ef284b`|
|`code/cvsrffi/dual_feature_forward.py`|`1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56`|
|`code/cvsrffi/leo_weak_cache.py`|`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`|

## 4.冻结远端输入

|资产|远端路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|adapter|`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/effective8_adapter_fp16.pt`|`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`|
|cache set|`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json`|`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`|
|selection salt|`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json`|`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`|

## 5.同步映射、环境与远端命令

- run根:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2`
- source目录:`<run-root>/source_ca5d0c4b`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- child CWD:`<run-root>/source_ca5d0c4b`
- GPU:`PENDING_LIVE_ALLOCATION`；runner先记录全部GPU compute进程，任一GPU最多两个训练/计算进程。
- local ZIP→`<run-root>/input/source_ca5d0c4b.zip`
- root wrapper→`<run-root>/input/run_pipeline.sh`

runner必须先执行本地只读`tools/n607_ssh_preflight.ps1`，默认直连`N607`；确认新run根不存在、资产可见和GPU状态后，唯一地创建`<run-root>/input`与`<run-root>/logs`，同步两项输入，核验ZIP/wrapper SHA，安全解包到唯一source目录并核验6个成员SHA、`py_compile`和`bash -n`。任何已有run/output/PID/exit路径、SHA或compile漂移均停止，不覆盖、不重试。

唯一launch模板：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2/source_ca5d0c4b
nohup env CUDA_VISIBLE_DEVICES=<live-assigned-gpu> bash /home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2/input/run_pipeline.sh > /home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_geoff_r2_ca5d0c4b_20260722_r2/logs/pipeline.log 2>&1 < /dev/null &
```

wrapper用`BASHPID`原子写`logs/pipeline.pid`，trap原子写`logs/pipeline.exit`。runner在首个短连接中核验PID、GPU、完整cmdline和CWD；仅用短连接监控，完成后回收artifact并确认本机无残留SSH连接。retry固定为`NO`；不得kill/restart、调参或干预无关作业。

## 6.预期artifact与硬门

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

技术完成要求：child exit=0；日志含`PIPELINE_ARTIFACTS_COMPLETE`；export receipt为v2 PASS；base parity receipt/vector为v2 PASS，batch=`[1,8,256]`、每batch调用字段均为3、三输出最大绝对差均`≤1e-5`；archive manifest为v2且内部verify通过；execution contract在export/parity/archive三处逐值与SHA闭合。

coverage硬常量保持r1冻结口径：总行数=8400，unique physical=8400，unique observation=8400，class/receiver/day/scenario计数=6/7/4/3，class registry=`[14-10,14-7,20-15,20-19,6-15,8-20]`，scene registry为三个`leo_*_weak`，receiver×day×class cell=168、zero cell=0、min cell>10，并报告K1/K5/K10剩余行。coverage receipt schema保持`cvs.phase1.singleobs_dual_feature_coverage_receipt.v1`，只作计数描述，不选择held fold。

任一硬门失败时不生成完成marker，最终状态为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；即使全部通过也只能标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。archive/coverage不是方法性能、promotion或`1+1>2`证据。

## 7.完成后回填

|字段|结果|
|---|---|
|release-control Git commit|`d45f4cc22ac379c287ad09baed53fe07cdb791d2`|
|runner/route|`PENDING`|
|preflight/GPU|`PENDING`|
|remote ZIP/wrapper/member SHA|`PENDING`|
|remote compile/bash|`PENDING`|
|launch PID/CWD/cmdline|`PENDING`|
|child exit/marker|`PENDING`|
|parity receipt/vector|`NOT_GENERATED`|
|archive/manifest|`NOT_GENERATED`|
|coverage同row结果|`NOT_GENERATED`|
|回收路径/SHA|`PENDING`|
|最终状态|`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`|
