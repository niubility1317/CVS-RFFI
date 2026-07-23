# DSSC-ZDOM-JG-qKNN-R4-BCRR/r1f-techfix3完整125实验报告

## 身份与状态

- run ID：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix3_3bc31826_20260723_165358`
- 创建时间：`2026-07-23T16:53:58+08:00`
- operator：主agent；sole launch owner：待分配的`gpt-5.6-terra high`runner
- candidate：`DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`；implementation tag=`techfix3`
- 当前状态：`LOCAL_VERIFIED -> PREREGISTERED / NO_PERFORMANCE_RESULT`
- 科学方法提交：`849fa342cd46cb8294b5d9b4f5358cea630d0643`
- techfix3代码与source package提交：`3bc318266006a040bb5957b297a4a6cd2345a0f2`
- parent技术失败run：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix2_b77cc6c4_20260723_155058`
- retry=`false`；本run使用全新本地与远端root，不得复用或覆盖parent路径。

## 目标、根因与唯一技术delta

目标是在不改变方法、数据、五臂、矩阵、参数、loss、adapter、qKNN、BCRR、INT8或fallback的前提下，闭合legacy SVRN内部字典序类轴与sealed registry的old-prefix/new-append类轴冲突，并直接取得完整125的真实prediction及同row性能结果。

parent在GPU0–7完成125/125个launcher job，但全部在query预测前因`registered class registry drift`退出，row receipt、prediction和score均为0，故状态为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。techfix3只在DSSC→legacy SVRN接口内部投影到字典序，qKNN、BCRR及五臂出口再逆置换回原sealed registry；类集合、support、距离和决策公式不变。`BCRRState`保留旧三参数位置兼容，无法安全恢复类轴时fail-closed。

## 本地闭合证据

- `ssr-gpu`下method、runner与专项test的`py_compile PASS`。
- 专项、协议负例、非字典序五臂与legacy BCRR兼容测试：`30/30 passed`。
- 真实checkpoint无query smoke：SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；`20-1/leo_clear_weak/K10`；11类110条support；`z_id`有限FP32`[110,160]`；sealed registry保持非字典序、内部bank为字典序、五臂出口恢复sealed顺序；`query_packages_loaded=false`、`query_rows_used=0`。
- 独立终审：`MERGE / P0=0 / P1=0`。
- `git diff --check PASS`；提交后仅保留用户既有未跟踪RBSC与pytest临时目录，不进入本run。

## 冻结完整125与裁决门

- receiver：`20-1,3-19,7-14,7-7,8-8`
- seed：`713102,713103,713104,713105,713106`
- slice：`(K10,new5),(K10,new10),(K10,new20),(K5,new20),(K1,new20)`
- scene：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- 总数：`125 jobs / 375 prediction slices / 1875 score rows`
- 五臂：`M0/M_DA_NG/M_DA/M_OTHER/M_JOINT`
- 调度：GPU0–7各最多1个本run worker，父进程用`CUDA_VISIBLE_DEVICES=<physical_gpu>`隔离，子进程统一逻辑`cuda:0`。
- `I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。

技术完成必须有125份launcher receipt与row receipt、375份prediction slice、1875行score、完整`matrix_exit.json/aggregate_index.json`及全部artifact/hash/row闭合；否则严格标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不kill、restart或retry。

性能完成后必须同row报告old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类、receiver、scene、K、seed、new-count、coverage、量化margin、MAC、时延、显存和state bytes。M_DA或M_OTHER无独立正收益、JOINT不胜两者、mean`I_syn<=0`、正协同少于188/375或少于2/3个scene、任一保护指标退化、协议/INT8/资源门失败，均裁为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；不得用125回调方法。

## 不可变发布包

|artifact|SHA256或闭合值|
|---|---|
|source ZIP|`512e1f74a27f6bed4c6d8820bf448f23c8eadf4832c9f3243e81deeff5ff689d`；33,008,269B|
|source tree|`3bc318266006a040bb5957b297a4a6cd2345a0f2`；3,959/3,959个regular safe entry与raw Git blob一致|
|ZIP runner entry|`ab29bb5c6ceb64e16f5cea3c5d91948d061ad123847f027f44abe7392316f55b`|
|ZIP method entry|`d087030f78f730f0eb930ab9f298ae902d9709f02d896e1a1b6e122c9127bc1f`|
|ZIP test entry|`b95224bed811eec738ea140e7b43928432d811a9f55634166a620164fec4d0bb`|
|DSSC method lock|`7663bbc4b7b199d98caa85b7736547a6927a2c7eb8e6a4de636967edca1e9c10`|
|ground bundle|`109724913cac4f82ff58359b927a7f1e7f7e7d233c0bfd0d05d323f94b1b12da`|
|SOMPH lock|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|
|coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`；`PRESENT_REUSED / NOT_REGENERATED`|
|checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|sealed runtime|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|
|Phase1 archive/manifest|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0` / `34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`；`PRESENT_REUSED / NOT_REGENERATED`|
|parity receipt|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`；`PRESENT_REUSED`|

本run只同步source ZIP与4个冻结小型input；checkpoint和sealed runtime使用远端既有冻结路径。方法变化不触发数据重验。

## N607发布合同

- preflight：`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`；direct N607优先，只有direct网络路径失败且身份无歧义时才使用既定lab bridge。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- remote run：`/home/szu2070436088/2510044040/CV-SincNet/runs/dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix3_3bc31826_20260723_165358`
- source ZIP：`<run>/input/source_3bc31826_rawblob_deflated.zip`；解包到`<run>/source`
- cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix`
- authority：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- runtime：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt`
- output：`<run>/artifacts`；日志=`<run>/logs/matrix.stdout.log`和`matrix.stderr.log`；PID=`<run>/launcher.pid`；exit=`<run>/launcher.exit`。

Runner必须先确认remote run root不存在，再同步5个文件，核验ZIP整体SHA、3,959个safe regular entry、runner/method/test条目、4项input、checkpoint/runtime SHA和`py_compile`。GPU1预启动smoke必须在独立单卡命名空间以逻辑`cuda:0`完成零IQ sealed runtime前向，并执行非字典序registry的无query五臂state构建/逆置换专项；不得打开query package或生成prediction。实时GPU安全门通过后只允许detach启动一次：

```text
env PYTHONPATH=<run>/source/code OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python <run>/source/code/scripts/run_dssc_zdom_jg_qknn_r4_bcrr_125.py matrix --phase1-checkpoint <checkpoint> --sealed-runtime <runtime> --package-method-lock <run>/input/somph_method_lock.json --dssc-method-lock <run>/input/dssc_method_lock.json --ground-bundle <run>/input/phase1_dssc_zdom_jg_ground_bundle.npz --coverage-receipt <run>/input/coverage_receipt.json --cache-root <cache> --authority-root <authority> --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

监控只用短SSH并主动断开；完成后回收完整日志、PID/exit、matrix/aggregate、125 launcher与row receipt、prediction、score及SHA清单。不得回收数据、checkpoint或runtime。

## Runner与性能回填

|字段|当前值|
|---|---|
|direct/bridge preflight|`DIRECT PASS`；bridge未使用|
|remote root immutable check|`PASS`；创建前`ABSENT`|
|source/input/checkpoint/runtime SHA|`PASS`；7项冻结SHA|
|GPU1 zeroIQ＋nonlexical noquery smoke|`PASS`；物理GPU1→逻辑`cuda:0`，无query|
|remote PID/exit|`826851 / 1`；自然退出|
|prediction/score|`0 / 0`|
|archive/coverage generation|`NO / NO`；只复用冻结artifact|
|最终状态与裁决|`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

## Runner回填：自然技术失败闭合

- direct preflight、remote root不存在性、5项landing、7项冻结SHA、3,959个safe regular ZIP entry、runner/method/test entry和`py_compile`全部通过。
- GPU1 smoke通过：物理GPU1隔离为逻辑`cuda:0`；zeroIQ真实checkpoint/sealed runtime前向输出有限FP32`[1,160]`；非字典序五臂state及逆置换通过；未打开query package，`query_rows_used_for_fit=0`。
- 唯一detach PID=`826851`，3秒后自然exit=`1`；未kill、restart或retry。GPU0–7退出后均`0%/10MiB`且无compute process；本地SSH残留为0。
- 根因：发布准备预先创建了空`<run>/artifacts`，而matrix的`--run-root`合同要求该路径不存在，入口立即报`matrix run root must be new and cannot be overwritten`。失败发生在row、query和GPU子任务之前，不是方法、registry、数据、checkpoint、runtime或性能失败。

|回收项|SHA256|
|---|---|
|`launcher.pid`|`424b2e6824a0e10cdf071dc6ce3e72fa5068e8ff6c72e215fe63d31b13459ec3`|
|`launcher.exit`|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|
|`matrix.stderr.log`|`9fac51e67f434b6e38d896986616ae4bbe50ebbd99be977092cec4a1d46ddb4c`|
|`matrix.stdout.log`|`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`|

artifact计数：launcher=`0/125`、row=`0/125`、prediction=`0/375`、score=`0/1875`、`matrix_exit/aggregate=0/0`、remote artifact files=`0`。parity/archive/coverage均为`PRESENT_REUSED / NOT_GENERATED`；remote run保留且本run禁止复用。最终状态严格为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
