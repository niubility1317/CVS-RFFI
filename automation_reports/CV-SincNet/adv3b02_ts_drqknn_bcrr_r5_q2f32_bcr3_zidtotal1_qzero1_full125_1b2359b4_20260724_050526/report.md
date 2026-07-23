# ADV3B02/r5-qzero1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r5_q2f32_bcr3_zidtotal1_qzero1_full125_1b2359b4_20260724_050526`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r5-q2f32-bcr3-zidtotal1-qzero1`
- Git commit：`1b2359b455f0466019a98caa7e51cb165f5463be`
- 创建时间：`2026-07-24T05:05:26+08:00`
- operator：主agent；唯一N607 launch owner为单一`gpt-5.6-terra high`runner
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- parent run：`adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_procbindfix1_full125_467b8aa5_20260724_035813`
- parent终态：`STOPPED_EARLY_DETERMINISTIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 目标、假设与冻结比较

本run用GPU0–7执行完整125，取得`M0/M_DA/M_OTHER/M_JOINT`同row真实性能，验证`z_id/z_dom`双qKNN域适应、统一qKNN分类和BCRR是否产生正协同。r5唯一delta是对finite componentwise exact-zero query `z_id`采用冻结的Student-t解析分数，使此前零prediction技术row也能完成全注册类决策；zero row四臂逐字节相同且`I_syn=0`，不得把技术总化记为DA或OTHER收益。

冻结比较仍为：`M0=基础z_id qKNN`、`M_DA=z_id/z_dom双qKNN`、`M_OTHER=基础qKNN+BCRR`、`M_JOINT=双qKNN+BCRR`。正常query、DA、BCRR、state/codec/append、prediction/scorer格式、资源门、矩阵和健康策略均继承parent。矩阵为5receiver×5seed×`{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每row覆盖3个LEO弱场景；期望125份row receipt、375个scene slice、1000份prediction和1500个logical score row。

## 本地闭合与独立review

|文件|SHA256|用途|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|`8294e843990ebb9a17cdeb305f1e90895ee94c698e1a9fa04685a07493e66290`|r5 exact-zero query总化与四臂统一决策|
|`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`|`aa9ada403192f0d13fe4bca257473d1f5f666a563ef7b8b5db67f814bb99a3a0`|before/after统一prediction及qzero runtime receipt|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|`b857da58bdb9d35149d159d58c837dcc344069cf17ad1b91fbbad5ef02dcb6b7`|zero/normal、负例、重排及receipt回归|
|`docs/ADV3B02_TS_DRQKNN_BCRR_R5_Q2F32_BCR3_ZIDTOTAL1_QZERO1_DESIGN_FROZEN.md`|`9d5da20b598b90bdb4c7023edabf66165fd849811ecc28c3a6ec523d6e0fda45`|冻结合同|

- `ssr-gpu`目标测试完整通过：89 passed、3项仅Windows POSIX skipped；最终review专项7 passed。
- 三文件`py_compile`和`git diff --check`通过。
- 真实checkpoint三scene×before/after共6个support-only state smoke通过，未读取query或truth。
- 独立Terra终审：`MERGE / P0=0 / P1=0 / P2=0`。

## 发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r5_q2f32_bcr3_zidtotal1_qzero1_full125_1b2359b4_20260724_050526\input`

|输入|SHA256|说明|
|---|---|---|
|`source_git_blobs_1b2359b4.zip`|`bdbbed80f36f06145bc8da72a3493f7cb358716ab49bb16a304a5b01ed643054`|34,580,207B；3,983个safe regular raw Git blob；raw bytes=`231,188,363B`；path-set SHA=`5a451a362af3c456adcc93d79fd88bd59aa773c659772294b0a669ab3026b9e8`；missing/extra/raw mismatch=`0/0/0`|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|2,202B；直接复用冻结method lock|

外部输入继续复用checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`、sealed runtime SHA=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2`及GEOFF/r8既有cache/authority路径，不重新验证数据。

## N607路径与正式命令

远端run根创建前必须为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r5_q2f32_bcr3_zidtotal1_qzero1_full125_1b2359b4_20260724_050526`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`，GPU固定为0–7。正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

日志固定为`<run>/logs/matrix.stdout.log`、`<run>/logs/matrix.stderr.log`和`<run>/logs/matrix.exit`。唯一runner负责direct preflight、两个输入精确同步、远端SHA和安全解包、关键文件`py_compile`、冻结POSIX sentinel、run根不可覆盖核验、唯一detach、PID/CWD/cmdline/GPU核验、启动后健康检查、短连接监控、完整日志读取、artifact回收和结构化handoff；主agent不得并发启动本run。

## 健康止损、完成与性能裁决

- 首个失败row一旦使125/125不可达，立即停止继续派发、回收最小失败证据并只终止本run；不得自然退出浪费GPU，不得自动retry或合并第二方法修复。
- 不得按partial性能早停或读取partial性能；不得修改方法、参数、矩阵、数据或共享发布系统。
- 只有125/125 row receipt、1000/1000 prediction、1500/1500 logical score row及完整completion/archive/coverage/parity绑定闭合后，才进入性能分析。
- 完成后同row报告old-before、old-after、DA old gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、量化margin、MAC、时延、显存、state bytes和`I_syn`；正协同slice门为至少188/375。
- 无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完成prediction但不达科学门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

|字段|当前值|
|---|---|
|Git commit|`1b2359b455f0466019a98caa7e51cb165f5463be`|
|run ID|`adv3b02_ts_drqknn_bcrr_r5_q2f32_bcr3_zidtotal1_qzero1_full125_1b2359b4_20260724_050526`|
|remote PID/exit|`1551147 / runner-stopped after first failed row`|
|prediction/score|`392/1000 / 588/1500`|
|archive/coverage/parity|`NOT_GENERATED / NOT_GENERATED / NOT_GENERATED`|
|性能裁决|`NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`

## Runner落地与预启动证据

- 发布状态：`LANDED / PRELAUNCH_VERIFIED / NO_PERFORMANCE_RESULT`。
- direct预检：本地`N607`配置与身份文件有效，但直连在banner exchange阶段被拒绝；确认本地无遗留`ssh.exe`或N607 TCP22连接后，按固定`lab bridge`路径完成有界操作。
- 远端新run根：创建前为`ABSENT`，已创建为`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r5_q2f32_bcr3_zidtotal1_qzero1_full125_1b2359b4_20260724_050526`；Python、checkpoint、sealed runtime、cache和authority均存在；GPU0–7启动前均为`0%`、`10/24576MiB`。
- 精确同步与安全解包：ZIP SHA=`bdbbed80f36f06145bc8da72a3493f7cb358716ab49bb16a304a5b01ed643054`、method lock SHA=`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`；3983个safe regular blob、raw bytes=`231188363`、path-set SHA=`5a451a362af3c456adcc93d79fd88bd59aa773c659772294b0a669ab3026b9e8`，远端解包后复核一致。
- 远端预启动验证：指定三文件`py_compile`通过；冻结`posix-sentinel`通过，run-owned root process group清零且unrelated sentinel保持存活。

## Runner终止与最小失败证据

- 最终技术状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。该run不满足完整125，不作性能读取、分析或方法裁决。
- 首个失败：`adv3b02_r5_q2f32_bcr3_qzero1_rx_7-14_s_713102_k_10_n_10`，failure fingerprint=`ADV3B02StateError: affine actual branch audit/state drift`，failure code=`TECHNICAL_EXCEPTION`，prediction发布前计数=`0`，query fit rows=`0`，failure receipt SHA=`48f3afe42562a1be23ce512b6ef68011783cb115c2bd6121db62b1189a18d6e3`。
- 停止时技术计数：submitted launcher logs=`51`；succeeded row receipt=`49/125`；failed marker=`1`；active=`0`；prediction=`392/1000`；logical score=`588/1500`。首个失败已使125/125不可达，因此停止后续派发并结束已验证的本run进程组。
- 终止控制审计：首次失败检测时计数为submitted=`42`、receipt=`33`、prediction=`264`。进程绑定采集后，首个停止辅助命令因shell语法错误而未发送信号，第二个因瞬态已退出PID的严格校验提前返回；第三个命令才向matrix PGID发送`TERM`，随后对已验证row PGID执行终止。该控制延迟使最终submitted达到`51`，已如实保留；未重新启动、未重试、未修改方法或读取性能。
- 进程与GPU：matrix PID/PGID=`1551147/1551147`，启动CWD=`<run>/source/code`；对该PGID及当时每个已核验CWD/cmdline属于`<run>`的row PGID依次`TERM`后必要时`KILL`。最终run-owned process scan为空，run-owned GPU process scan为空；所有远端连接均完成本地SSH/TCP22清理。
- completion/archive/coverage/parity均未生成，这是技术早停的预期结果。`qzero count`仅允许从完整artifact读取，本run为`NOT_AVAILABLE_IN_INCOMPLETE_ARTIFACTS`。
- 已回收最小失败证据至`recovered_failure/`：失败row launcher log、matrix runtime manifest、stdout/stderr、runner exit receipt；共5文件、92,871B，inventory SHA=`45637f642cb34a4927a4827c91ec002ead68f210c57fe7c6ccd9a311b5f01ebe`；未回收或读取部分性能产物。
