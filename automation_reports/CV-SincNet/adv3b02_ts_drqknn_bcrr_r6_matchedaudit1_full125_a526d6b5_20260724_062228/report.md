# ADV3B02/r6-matchedaudit1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_full125_a526d6b5_20260724_062228`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r6-matchedaudit1`
- Git commit：`a526d6b53f10829e96c61aabc9489c9dbd1bfb44`
- 创建时间：`2026-07-24T06:22:28+08:00`
- operator：主agent；唯一N607 launch owner为单一`gpt-5.6-terra high`runner
- 状态：`PREREGISTERED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- parent run：`adv3b02_ts_drqknn_bcrr_r5_q2f32_bcr3_zidtotal1_qzero1_full125_1b2359b4_20260724_050526`
- parent终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 目标、冻结比较与唯一技术delta

本run在GPU0–7执行完整125，取得`M0/M_DA/M_OTHER/M_JOINT`同row真实性能。比较固定为：`M0=基础z_id qKNN`、`M_DA=z_id/z_dom双qKNN`、`M_OTHER=基础qKNN+BCRR`、`M_JOINT=双qKNN+BCRR`。矩阵为5receiver×5seed×`{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每row覆盖3个LEO弱场景；期望125份row receipt、375个scene slice、1000份prediction和1500个logical score row。

r6唯一delta是修正Stage2-C append的qKNN量化审计参照面：旧类使用冻结Stage2-B deployed bank解码与已部署bandwidth，新类使用current FP32 support，并对new suffix两平面5个数组及class bandwidth hi/lo共7个字段进行独立冻结公式逐字节闭合。部署bank、state/schema、q2 codec、BCRR、DA、qKNN四臂、prediction/scorer、runner、矩阵和资源门均不变。

## 本地闭合与独立review

|文件|SHA256|用途|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|`b334780cf896e9d8122aac1ad71eba2381a0430858747470562532f0958811ea`|matched append audit及new suffix独立闭合|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|`8257fe32f4248f8b968868fc4baf8045b2b9132d678bf63c9f165aa9b1d59c42`|K1/K5/K10、协议负例及7字段tamper回归|
|`docs/ADV3B02_TS_DRQKNN_BCRR_R6_MATCHEDAUDIT1_DESIGN_FROZEN.md`|`910cbd15aef09f2adc17b3f14cfba562b3875712462edab5b8e5ac0bae426352`|冻结合同|

- `ssr-gpu`完整方法测试：`100 passed / 3 Windows POSIX skipped / 0 failed`；`py_compile`与`git diff --check`通过。
- 独立终审：`MERGE / P0=0 / P1=0 / P2=0`；额外完成K1/K5/K10专项14项及180例production/reference逐字段bitwise对拍。
- 真实checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`的本地`cuda:0` support-only smoke复用原失败row`7-14/713102/K10/new10`；三scene均top1=1.0、any/large flip=0、old prefix全字段保留、wire=`102,153B<256KiB`，query/truth读取与fit query rows均为0。

## 发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_full125_a526d6b5_20260724_062228\input`

|输入|SHA256|说明|
|---|---|---|
|`source_git_raw_blobs_a526d6b5.zip`|`b8cd7f4dd8c646c94df0fff1c3cef695799e501180c3edec9be82434dbf4f3f5`|34,598,165B；3,990个safe regular raw Git blob；raw bytes=`231,313,953B`；path-set SHA=`a8dc44a0d837b05579d8f0485651036bb0c20b2f755ecb2b4adf0a0e7a14e109`；missing/extra/raw mismatch/unsafe=`0/0/0/0`|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|2,202B；复用冻结method lock|

外部输入继续复用既有checkpoint、sealed runtime及GEOFF/r8 cache/authority路径，不重新验证数据。

## N607路径与正式命令

远端run根创建前必须为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_full125_a526d6b5_20260724_062228`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`，GPU固定为0–7。正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

日志固定为`<run>/logs/matrix.stdout.log`、`<run>/logs/matrix.stderr.log`和`<run>/logs/matrix.exit`。唯一runner负责direct preflight、精确同步、远端raw blob/hash/compile核验、run根不可覆盖检查、唯一detach、PID/CWD/cmdline/GPU核验、启动健康检查、短连接监控、完整日志读取、artifact回收和结构化handoff；主agent不得并发启动本run。

## 健康止损、完成与性能裁决

- 首个系统性技术失败一旦使125/125不可达，立即停止派发并只终止本run；不得自然退出、自动retry、读取partial性能或扩大修复。
- 只有125/125 receipt、1000/1000 prediction、1500/1500 score及completion/archive/coverage/parity闭合后才进入性能分析。
- 完成后按同row报告old-before、old-after、DA old gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、量化margin、MAC、时延、显存、state bytes和`I_syn`；正协同门为至少188/375个slice。
- 无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完整prediction但未达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

|字段|当前值|
|---|---|
|Git commit|`a526d6b53f10829e96c61aabc9489c9dbd1bfb44`|
|run ID|`adv3b02_ts_drqknn_bcrr_r6_matchedaudit1_full125_a526d6b5_20260724_062228`|
|remote PID/exit|`NOT_LAUNCHED / NOT_AVAILABLE`|
|prediction/score|`0/1000 / 0/1500`|
|archive/coverage/parity|`NOT_GENERATED / NOT_GENERATED / NOT_GENERATED`|
|性能裁决|`NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`

## 唯一runner落地与启动级技术闭合

- `2026-07-24T06:33+08:00`：direct N607只读预检通过；项目根可见，GPU0–7均为空闲，磁盘剩余`7.5T`。远端run根在创建前为`ABSENT`，未复用旧run。
- 精确同步两份冻结input后，远端ZIP SHA=`b8cd7f4dd8c646c94df0fff1c3cef695799e501180c3edec9be82434dbf4f3f5`、method lock SHA=`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`；安全解包为`3990`个regular blob、`231313953B`，无unsafe path/symlink/目录项/重复路径；关键源码SHA、checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`、sealed runtime SHA=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`一致。`py_compile`和冻结POSIX进程树sentinel通过。
- 唯一启动尝试主PID=`1592872`，exit=`1`；启动后无本run子进程、无GPU占用。stderr唯一异常为`ADV3B02LauncherError: matrix root must be fresh`，位于`run_matrix`第1697行。
- 首源：runner在预建run目录时错误预创建了`<run>/artifacts`，而冻结matrix将该路径作为必须不存在的`--run-root`。这是启动准备错误，不是方法、数据、协议或性能故障；无row提交、无prediction、无score、无receipt、无completion/archive/coverage/parity。
- 本run不得复用或重启；已回收最小失败证据到本地报告目录：`matrix.stdout.log` SHA=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`、`matrix.stderr.log` SHA=`109d6c3342a04b69c9a4a874eb085a4bfe87d6c8615802fd4fae72dc6df5504e`、`matrix.exit` SHA=`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`、`matrix.pid` SHA=`ab74a9e87abe2844e212525de40ec8edfa8f3b80cab491ee94f9072d202358b2`。

|字段|终态|
|---|---|
|remote PID/exit|`1592872 / 1`|
|submitted/succeeded/failed/active|`0 / 0 / 0 / 0`|
|prediction/score|`0/1000 / 0/1500`|
|completion/archive/coverage/parity|`ABSENT / ABSENT / ABSENT / ABSENT`|
|状态|`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|性能可分析性|否；未产生任何性能artifact|

下一步仅允许修正共享runner的预建目录规则、专项回归、独立review、Git提交与全新不可覆盖run ID发布；不得改变r6方法、矩阵、数据或协议。

## release-prep独立review

独立裁决=`MERGE / P0=0 / P1=0 / P2=0`。冻结launcher的fresh-root契约正确，无需修改方法或runner代码。新run的唯一目录修正规则为：

1.本地报告根和远端`<run>`在创建前均必须ABSENT；
2.唯一runner只预建`<run>/input`、`<run>/source`和`<run>/logs`；
3.启动前`<run>/artifacts`必须ABSENT，禁止占位、touch、验证器输出或预写manifest；
4.child CWD仍为`<run>/source/code`，`--run-root=<run>/artifacts`，由matrix首次创建；
5.PID、stdout、stderr和exit只写入`<run>/logs`；
6.scientific commit=`a526d6b5`、source package=`b8cd7f4d`、command、matrix和数据均不变。

任一fresh-root、CWD/cmdline、package/源码或PID绑定不符即停止为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重试同一run ID。
