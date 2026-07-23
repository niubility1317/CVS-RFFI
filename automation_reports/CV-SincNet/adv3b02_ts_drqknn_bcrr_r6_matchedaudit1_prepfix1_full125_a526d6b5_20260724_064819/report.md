# ADV3B02/r6-matchedaudit1-prepfix1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_prepfix1_full125_a526d6b5_20260724_064819`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r6-matchedaudit1`
- scientific commit：`a526d6b53f10829e96c61aabc9489c9dbd1bfb44`
- release-prep commit：`04ebcf06ffa5a010c6c952d568033273d06d7954`
- 创建时间：`2026-07-24T06:48:19+08:00`
- operator：主agent；唯一N607 launch owner为单一`gpt-5.6-terra high`runner
- 状态：`PREREGISTERED / NOT_LANDED / NO_PERFORMANCE_RESULT`
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
|remote PID/exit|`NOT_LAUNCHED / NOT_AVAILABLE`|
|prediction/score|`0/1000 / 0/1500`|
|archive/coverage/parity|`NOT_GENERATED / NOT_GENERATED / NOT_GENERATED`|
|性能裁决|`NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
