# GEOFF/r2.2 Phase1双表征归档与coverage技术run报告

## 1.身份与当前状态

- run_id:`rchm_bpp_p1_dual_archive_buffer_r4_de9a6476_20260723_r4`
- 预注册时间:`2026-07-23T01:05:00+08:00`
- candidate:`P1-DUAL-ARCHIVE-GEOFF/r2.2-BUFFER`
- 主agent:`/root`；唯一N607 runner:`PENDING_GPT_5_6_TERRA_HIGH`
- protocol:`p2_min_v1`；数据状态:`VALIDATED_ONCE_REUSED`
- 当前状态:`LOCAL_VERIFIED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- retry:`NO`

本run不改变received IQ、physical ID、receiver/TX、scene、K、support/query split或protocol schema，不重复数据验证。唯一delta是在archive一次性离线批处理内，用标准buffer把C连续native-float32数组送入Torch，并用Python binary64中转将已核验float32输出写回NumPy；选择、batch、runtime调用、数值、manifest、coverage门和权限均不变。

## 2.matched failure与假设

matched run=`rchm_bpp_p1_dual_archive_cli_r3_9842ff42_20260723_r3`，PID=`371059`、自然exit=`1`。其export与base parity均PASS，base parity batch=`[1,8,256]`、每batch3次调用、maxabs=0；archive在`torch.from_numpy`处抛`TypeError: expected np.ndarray (got numpy.ndarray)`，因此archive/coverage/prediction均未生成，永久为`ANALYZED / TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

本run假设：`memoryview→torch.frombuffer→reshape→clone`绕开N607的NumPy类型身份桥，同时保持输入float32逐bit一致；Tensor float32经Python binary64再到NumPy float32逐bit回转。任一技术门失败即停止，不retry、不放宽。

## 3.Git、源码包与本地证据

|项目|冻结证据|
|---|---|
|方法commit|`de9a6476e60428ba89d243af63f1eca6229d52c6`|
|commit-bound source|`de9a6476e60428ba89d243af63f1eca6229d52c6`|
|分支|`codex/ground-prototype-da-rd`|
|独立review|`P0=0,P1=0,P2=0→MERGE`|
|专项测试|`ssr-gpu`中`py_compile`与archive专项`8 passed`；模拟`torch.from_numpy`抛出r3同型TypeError仍完成archive|
|数值探针|正负零、subnormal、最大有限值及随机float32共4096元素，经Tensor→Python float→NumPy float32逐bit一致|
|源码ZIP|`E:/type10-7/code/snapshots/rchm_bpp_p1_dual_archive_buffer_r4_de9a6476_20260723_r4/source_de9a6476.zip`|
|ZIP SHA/大小/成员|`ec3c418cc7225504322fedf4a4bdb9c124d470e30d762d81519a9e98ff8a9f0d / 33,027,282B / 4,442`|
|wrapper SHA|`bcf7193141c98fa3787189a7242376db18c6d07de82847ae3c10feb30ebf3fec`|
|wrapper静态门|`bash -n`PASS；CUDA未分配时exit=70|

冻结源码成员：

|相对路径|SHA256|
|---|---|
|`code/scripts/export_phase1_singleobs_dual_feature_archive.py`|`31a6a464f470ae9bdb6cbc8814581ff6c73403d5c99b497a224b3f783831fe64`|
|`code/scripts/export_phase1_singleobs_feature_archive.py`|`81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687`|
|`code/scripts/export_adv3b02_dual_feature_torchscript.py`|`e2cbc0ce19402e8c665489fb2b13bb63f988c14e138cbc39dd20f7c9e2b12090`|
|`code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py`|`0a3e80b226997b353c577de94aa8c8e92fb25f5bf13d8b81c9bc27f448ef284b`|
|`code/cvsrffi/dual_feature_forward.py`|`1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56`|
|`code/cvsrffi/leo_weak_cache.py`|`656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86`|

## 4.N607发布合同

- 先执行本地只读direct preflight；direct不可用时仅按AGENTS允许的lab bridge。
- remote run root:`/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_buffer_r4_de9a6476_20260723_r4`
- source root:`<run-root>/source_de9a6476`
- Python:`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 只同步上述ZIP和wrapper；先证明run root不存在，再创建该root的`input/`和`logs/`。
- runner核验ZIP/wrapper、4项冻结资产、6项源码SHA、安全解包、远端`py_compile`与`bash -n`后，仅启动一次。
- 启动形态：`nohup env CUDA_VISIBLE_DEVICES=<selected> bash <run-root>/input/run_pipeline.sh > <run-root>/logs/pipeline.log 2>&1 < /dev/null &`
- 禁止retry/restart、远端编辑、数据重验、target/query/held/125、调参、kill或无关作业干预。

冻结外部输入SHA保持不变：checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；adapter=`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`；cache_set=`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`；selection salt=`38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0`。

## 5.技术完成门与预期artifact

预期runtime/export/parity、archive NPZ/manifest、coverage receipt、sha256sums、pipeline PID/exit/log与completion marker。parity保持schema v2、batch=`[1,8,256]`、每batch3次调用、三输出maxabs≤`1e-5`。archive manifest必须为v2且内部verify通过。

coverage硬门保持：row=8400、unique physical=8400、unique observation=8400、class/receiver/day/scenario=`6/7/4/3`、receiver×day×class cells=168、zero=0、min>10，并报告K1/K5/K10余量；`feature_arrays_read=[]`、`held_fold_selected=false`。coverage只描述归档，不是性能。

即使全部通过也只能标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`；失败则为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。本run不生成prediction。

## 6.完成后回填

|字段|结果|
|---|---|
|release-control Git commit|`PENDING`|
|route/GPU/PID|`PENDING`|
|remote SHA/compile|`PENDING`|
|exit/marker|`PENDING`|
|export/parity|`NOT_GENERATED`|
|archive/manifest|`NOT_GENERATED`|
|coverage|`NOT_GENERATED`|
|prediction count|`0`|
|回收路径/SHA|`PENDING`|
|最终状态|`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`|

coverage真实通过后，主agent立即以其SHA冻结并发布最小held四臂性能矩阵。

