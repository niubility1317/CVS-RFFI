# ADV3B02/r3-q2f32-bcr2-zidtotal1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r3_q2f32_bcr2_zidtotal1_full125_aa22820c_20260724_023120`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r3-q2f32-bcr2-zidtotal1`
- scientific Git commit：`aa22820cfbefe45b020c7e6190a53a7237b290b7`
- 创建时间：`2026-07-24T02:31:20+08:00`
- operator：主agent；唯一N607 launch owner为单一`gpt-5.6-terra high`runner
- 状态：`PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`
- parent run：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_bindfix1_full125_00b81000_20260724_005555`
- parent终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 目标与冻结矩阵

目标是用完整125验证`z_id/z_dom`双qKNN域适应、统一qKNN分类和BCRR是否形成同row正协同。相对parent的唯一delta是qKNN support部署codec：固定affine INT8主平面＋固定INT8 residual平面＋双FP16 class bandwidth；DA、BCRR、四臂、K、repair、scorer、矩阵与健康门不变。

四臂固定为`M0/M_DA/M_OTHER/M_JOINT`，`I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。矩阵固定为5receiver×5seed×`{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每job覆盖3个LEO弱场景；期望125个row receipt、375个scene slice、1000个prediction artifact和1500个score row。

## 本地闭合

|文件|SHA256|用途|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|`214272e0fbc388893d7fac2da897c65af93abd67af0ae80aa6cb714853af65fd`|Q2-FP32 support codec、wire、append和资源闭合|
|`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`|`6628111714905642d06ceee9cccb4e6e0efa456c14fd8dd05f78135e2dc27045`|仅revision schema和job ID|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|`3250dcb9cdbef4f2028e558286ce5124cbc0e83e54b490b71cce96e74e4fb3a3`|codec、wire、K、tamper、append、无sidecar、资源负例|
|`docs/ADV3B02_TS_DRQKNN_BCRR_R3_Q2F32_BCR2_ZIDTOTAL1_DESIGN_FROZEN.md`|`ee8f3d9f25f84e92c425343ea452cfac694608614b10c29583ab86852c255514`|冻结合同|

- `ssr-gpu`目标测试：`77 passed、3 Windows POSIX skipped、0 failed`；相邻DSSC：`36 passed、0 failed`；`py_compile`与`git diff --check`通过。
- 真实checkpoint support-only smoke：两触发row×三scene×before/after共12个state，qKNN/BCR top1均为1、翻转0，最大logit误差`0.001622`，最大wire`159,691B<256KiB`；query/truth/apply打开数与fit query rows均为0。
- 独立Terra终裁：`MERGE / P0=0 / P1=0`。完整bandwidth数值列表不进入audit/append/state receipt，仅保留不可逆SHA和类数。

## 发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r3_q2f32_bcr2_zidtotal1_full125_aa22820c_20260724_023120\input`

|输入|SHA256|说明|
|---|---|---|
|`source_git_blobs_aa22820c.zip`|`0d57496c87356676780fa8486e00c1cb670ed04e4231f905764fe9f2be16f174`|33,153,749B；3977个safe regular Git blob；path-set SHA=`91bba4b94ca45076fcc8e864eaa124de69ec6d001fecac1080420539496482ae`；raw mismatch=0|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|复用冻结method lock|

外部输入保持checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`、sealed runtime=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`及既有GEOFF/r8 cache/authority路径。

## N607路径与命令

远端run根创建前必须为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r3_q2f32_bcr2_zidtotal1_full125_aa22820c_20260724_023120`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`，GPU固定为0–7。正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

日志固定为`<run>/logs/matrix.stdout.log`、`<run>/logs/matrix.stderr.log`和`<run>/logs/matrix.exit`。runner先执行direct preflight、精确同步、远端源码/输入hash、`py_compile`、POSIX sentinel和run根不可覆盖检查，再唯一detach一次并立即健康检查。任一row出现无合法prediction且完整125已不可能闭合时立即停止继续派发，并只终止已核验属于本run的进程树；不得retry、调参、覆盖或按性能值早停。

## 性能裁决

完整后必须同row报告old-before、old-after、DA old gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、量化margin、MAC、时延、显存、state bytes和`I_syn`。

立即证伪：`M_DA`无净正确增益；`M_OTHER`无独立正收益；`M_JOINT.H<=max(M_DA.H,M_OTHER.H)`；mean`I_syn<=0`；联合臂损害old-after、seen-new、floor、min-old或min-new或增加forgetting；正协同不足188/375个scene slice或不足2/3个scene均值为正。完成完整prediction但不达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

## 执行状态

|字段|当前值|
|---|---|
|Git commit|`aa22820cfbefe45b020c7e6190a53a7237b290b7`|
|remote PID/PGID|`PENDING`|
|launcher/matrix exit|`PENDING`|
|prediction/score|`0/1000`；`0/1500`|
|最终裁决|`PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
