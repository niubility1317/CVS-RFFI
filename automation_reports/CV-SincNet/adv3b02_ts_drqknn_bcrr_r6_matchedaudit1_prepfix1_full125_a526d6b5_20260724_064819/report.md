# ADV3B02/r6-matchedaudit1-prepfix1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_prepfix1_full125_a526d6b5_20260724_064819`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r6-matchedaudit1`
- scientific commit：`a526d6b53f10829e96c61aabc9489c9dbd1bfb44`
- release-prep commit：`04ebcf06ffa5a010c6c952d568033273d06d7954`
- 创建时间：`2026-07-24T06:48:19+08:00`
- operator：主agent；唯一N607 launch owner为单一`gpt-5.6-terra high`runner
- 状态：`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- parent technical run：`adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_full125_a526d6b5_20260724_062228`
- parent终态：`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；PID=`1592872`，exit=`1`，prediction/score=`0/1000 / 0/1500`

## 目标与冻结范围

本run在GPU0–7执行完整125，取得`M0/M_DA/M_OTHER/M_JOINT`同row真实性能。比较固定为：`M0=基础z_id qKNN`、`M_DA=z_id/z_dom双qKNN`、`M_OTHER=基础qKNN+BCRR`、`M_JOINT=双qKNN+BCRR`。矩阵固定为5receiver×5seed×`{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每row覆盖3个LEO弱场景；期望125份row receipt、375个scene slice、1000份prediction和1500个logical score row。

本run不包含科学delta。r6方法、matrix、checkpoint、sealed runtime、qKNN四臂、BCRR、prediction/scorer、数据和资源门均固定于scientific commit。相对parent的唯一发布准备修复是保持`<run>/artifacts`严格不存在，由冻结matrix首次创建。

## 本地闭合与独立review

- r6方法测试：`100 passed / 3 Windows POSIX skipped / 0 failed`；真实checkpoint无query support-only smoke通过。
- 方法终审：`MERGE / P0=0 / P1=0 / P2=0`。
- release-prep独立review：`MERGE / P0=0 / P1=0 / P2=0`；launcher无需修改。
- fresh-root合同：本地与远端run根创建前均为ABSENT；runner只预建`input/source/logs`，不得预建、touch或写入`artifacts`；PID/stdout/stderr/exit仅进入`logs`。

## 发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_prepfix1_full125_a526d6b5_20260724_064819\input`

|输入|SHA256|说明|
|---|---|---|
|`source_git_raw_blobs_a526d6b5.zip`|`b8cd7f4dd8c646c94df0fff1c3cef695799e501180c3edec9be82434dbf4f3f5`|34,598,165B；复用scientific commit的3,990个safe regular raw Git blob包，不重打包|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|2,202B；复用冻结method lock|

外部输入继续复用既有checkpoint、sealed runtime及GEOFF/r8 cache/authority路径；不重新验证数据。

## N607路径与正式命令

远端run根创建前必须为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_prepfix1_full125_a526d6b5_20260724_064819`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`，GPU固定为0–7。正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

日志固定为`<run>/logs/matrix.stdout.log`、`<run>/logs/matrix.stderr.log`和`<run>/logs/matrix.exit`。唯一runner负责direct preflight、精确同步、远端核验、唯一detach、PID/CWD/cmdline/GPU核验、启动健康检查、短连接监控、完整日志读取、artifact回收和结构化handoff。主agent不得并发启动本run。

## 健康止损、完成与性能裁决

- 启动后立即检查row提交、子进程、stderr和GPU；出现系统性技术故障时立即健康止损，不得等待自然退出或自动retry。
- 只有125/125 receipt、1000/1000 prediction、1500/1500 score及completion/archive/coverage/parity闭合后才进入性能分析。
- 完成后按同row报告old-before、old-after、DA old gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、量化margin、MAC、时延、显存、state bytes和`I_syn`；正协同门为至少188/375个slice。
- 无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完整prediction但未达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

|字段|当前值|
|---|---|
|scientific commit|`a526d6b53f10829e96c61aabc9489c9dbd1bfb44`|
|release-prep commit|`04ebcf06ffa5a010c6c952d568033273d06d7954`|
|run ID|`adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_prepfix1_full125_a526d6b5_20260724_064819`|
|remote PID/exit|`1602781 / 1`（completion+traceback交叉证明；`matrix.exit`缺失）|
|prediction/score|`992/1000 / 1488/1500`|
|archive/coverage/parity|`NOT_GENERATED / NOT_GENERATED / NOT_GENERATED`|
|性能裁决|`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`

## 唯一runner执行记录

- `2026-07-24T06:52:34+08:00`：direct-only只读预检通过；N607项目根、Python、GPU0–7和7.5T可用空间可见，GPU无计算进程。本地`source_git_raw_blobs_a526d6b5.zip`与`somph_method_lock.json`的SHA256分别匹配预登记值`b8cd7f4d...f4f3f5`与`0496594d...04523`。
- 新run远端根在创建前已确认`ABSENT`；checkpoint、sealed runtime、cache和authority路径均可见。runner未检查、创建或触碰`<run>/artifacts`。
- 本次runner交接声明release-prep/report commit为`49f5c376`；本报告既有头部仍记录`04ebcf06ffa5a010c6c952d568033273d06d7954`。该记录差异不改变冻结scientific源码包、method lock或child command，保留为可追溯发布元数据差异。
- 已创建且仅创建远端`input/`、`source/`、`logs/`；两份input远端SHA复核匹配。源码安全解包至`<run>/source/code`后，关键入口/模块存在、`py_compile`通过，POSIX哨兵证明仅终止run-owned进程组且不影响无关进程。
- `LANDED / RUNNING`：detached matrix主PID=`1602781`，PPID=`1`，PGID/SID=`1602781`，CWD为`<run>/source/code`，命令行与本报告冻结child command匹配。启动核验时8个row child均存在，并分别占用GPU0–7约678MiB；matrix runtime manifest由launcher首次创建，stderr无异常指纹，completion尚未生成，receipt/prediction/score为`0/0/0`（启动首波，非性能结论）。每次SSH/SCP后本地ssh.exe和至N607端口22的ESTABLISHED连接均为零。
- 首个worker波次健康完成：job目录已派发`16`，其中`8`份row receipt闭合、`64/1000`份prediction artifact已生成；冻结每row`12`个logical score row，因此已闭合`96/1500`个logical score row。另`8`个row child继续在GPU0–7运行；父级failure记录、stderr异常指纹均为`0`。以上仅为运行完整性进度，不读取或裁决任何partial性能。

## 终态与最小失败证据

- 终态：`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。matrix主PID=`1602781`已退出；completion记录`submitted=125`、`succeeded=124`、`failed=1`、`active=0`、`prediction=992/1000`、`logical score=1488/1500`、`scene_slices=372/375`。因完整125不可达，本run不得进入性能分析、不得重试或复用。
- 唯一失败行：`adv3b02_r5_q2f32_bcr3_qzero1_rx_20-1_s_713102_k_5_n_20`（receiver=`20-1`，seed=`713102`，K=`5`，new=`20`），child exit=`1`，在prediction=`0`、query_rows_used_for_fit=`0`时抛出`ADV3B02StateError: affine actual branch audit/state drift`。完整child traceback已回收；launcher随后以`ADV3B02LauncherError: one or more full125 rows failed`退出。
- `matrix_runtime_completion.json`声明`performance_status=NO_PERFORMANCE_RESULT`；没有生成archive、coverage或parity文件。远端`logs/matrix.exit`缺失，child exit=`1`由completion的returncode和完整traceback交叉证明。
- 失败launcher log不包含`quantization_audit`的具体数值；该量化审计值缺口明确保留，不能从本run推断或补写。
- GPU0–7终检均为`0% / 10MiB`，不存在run-owned进程；每次SSH/SCP后本地ssh.exe和N607:22 ESTABLISHED连接均为零。
- 本地最小证据已回收至`recovered_failure/`：`matrix_runtime_completion.json` SHA256=`b28aa3ebec8cab108f354e612425663b50cf43dd4e2fa2bc987c4374b358f69e`；`matrix_runtime_manifest.json`=`e5f5da64d8185b3c61701945b437a4fba22ef14f1bfa40c1e421b0f519e6d830`；失败child launcher log=`18228ef24f4c493a3a23070a12285c70a79656813f0263048481b06e51662b36`；matrix stderr=`c1f1dba9088afd16d7015d1752bfd1eb13f286de25d848c1ebb7b70e715c5967`；before/after enrollment receipt分别为`918e5a57ae5ad3647d7b0f48277386234693f0c5ac2b1129de8f14db099b6fdc`和`3dc76920143257a93cfcece9382fa5e2b749663a76b4b6d952cd88352cd26165`。

## diag1～diag3首源收敛

- diag1（PID=`1637313`）因诊断wrapper误把query转为float64而与正式失败指纹不一致，按预注册停止；prediction=`0`。stderr SHA256=`8d8b41230b6a17f20880c8b4667133c2585ab3549ec14bf45f9fea9b5d22ba8a`。
- diag2（PID=`1641223`）捕获了首个通过scene而未复现正式失败，故不用于归因；其JSON SHA256=`b1a3fd99eebc1c8d8a1748b6aa8bd636717179ce04e32f1dfff0d503938d3e57`。
- diag3（PID=`1645657`）在真实失败scene确定唯一false谓词为`qknn_top1_ge_0995=false`：validation=`130`、top1=`129/130`、any/large flip=`1/0`、MAE/max=`0.000134723/0.001352310`；BCR及全部dtype/shape/finite/scale/no--128守卫通过。JSON SHA256=`f8d17c0b9a51aa32a8e1a65d298d5ec78e58554059c072159bb97fa5309eb136`。

## diag4匹配失败审计的Q3/scale归因

- 诊断ID：`adv3b02_ts_drqknn_bcrr_r6_matchedaudit4_a526d6b5_20260724_074119`；wrapper SHA256=`fad32b20fe3052c2d2fc5a85333f919c96e89b57a8016d63d9f63bdde5344e34`，远端复核/编译通过。PID=`1649677`在JSON落盘后预注册`AffineAuditCaptured`退出，prediction=`0`。
- 原audit严格匹配diag3：`validation=130`、top1=`129/130=0.9923076923076923`、any flip=`1`、large flip=`0`、MAE/max=`0.00013472314417009522/0.0013523101806640625`。因此本次捕获有效。
- 反事实（均相对teacher-support+teacher-scale）：Q2+teacher-scale仍top1=`0.9923076923076923`、any/large=`1/0`、MAE/max=`0.00013331379410783215/0.001354217529296875`；teacher+deployed-scale为top1=`1.0`、any/large=`0/0`、MAE/max=`4.512177416558802e-06/3.0517578125e-05`；Q2+deployed-scale仍top1=`0.9923076923076923`、any/large=`1/0`、MAE/max=`0.00013472314417009522/0.0013523101806640625`；Q3+deployed-scale为top1=`1.0`、any/large=`0/0`、MAE/max=`4.8025119939499356e-06/6.103515625e-05`。
- 唯一归因：deployed-scale不是首源；Q2 support量化在teacher/deployed两种scale下均保留同一flip，Q3 residual层消除该flip。该结论仅描述构造器量化审计，不是正式性能结论或方法修改。
- 证据位于`recovered_failure/affineqdiag4/`：JSON SHA256=`86f158a8cc473c06bc6957f637f3135913008496cd014be0a19724fff899e995`，stderr SHA256=`96100647189db07b7d924d6904e57f889d535213a4addf0aab84b3ecd3d40e83`。GPU0–7终检为`0%/10MiB`，SSH清理完成。
