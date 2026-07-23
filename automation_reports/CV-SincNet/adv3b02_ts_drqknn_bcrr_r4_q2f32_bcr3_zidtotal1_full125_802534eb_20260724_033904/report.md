# ADV3B02/r4-q2f32-bcr3-zidtotal1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_full125_802534eb_20260724_033904`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r4-q2f32-bcr3-zidtotal1`
- scientific Git commit：`802534eb8036fb8a31f060fd55af5050d0fe7961`
- 创建时间：`2026-07-24T03:39:04+08:00`
- operator：主agent；唯一N607 launch owner为单一`gpt-5.6-terra high`runner
- 状态：`PREREGISTERED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- parent run：`adv3b02_ts_drqknn_bcrr_r3_q2f32_bcr2_zidtotal1_full125_aa22820c_20260724_023120`
- parent终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 目标、假设与冻结矩阵

目标是用完整125验证`z_id/z_dom`双qKNN域适应、统一qKNN分类和BCRR是否形成同row正协同。parent在Stage2-C 26类BCR两平面INT8审计中零prediction失败；本revision唯一delta是固定增加第三个逐类对称INT8残差平面，第三残差相对实际float32 `D2=(Q1+Q2)`计算，query按`(Q1+Q2)+Q3`同序重建。FP64 teacher、ridge/LOO、BCRR `omega`、DA、qKNN、四臂、K、repair、scorer、矩阵、调度与健康门不变。

四臂固定为`M0/M_DA/M_OTHER/M_JOINT`，`I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。矩阵固定为5receiver×5seed×`{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每job覆盖3个LEO弱场景；期望125个row receipt、375个scene slice、1000个prediction artifact和1500个logical score row。禁止partial性能解读。

## 本地闭合

|文件|SHA256|用途|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|`f61a33994e5dca24428f92c544732bd10e391b4774f4d757cb61e7e43d71f1d0`|固定三平面BCR state、receipt、query解码和资源闭合|
|`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`|`e7679daa9726eb00ae35e44767e7e89b283730323e8df27bfec3416ecea857d6`|仅r4 launcher schema和job ID|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|`1ab59e40caffcd994de55d51229669ec7e1c31c9d532da7a718456b80b82e84d`|codec、六数组tamper、K/类数、append/prefix、无sidecar和资源负例|
|`docs/ADV3B02_TS_DRQKNN_BCRR_R4_Q2F32_BCR3_ZIDTOTAL1_DESIGN_FROZEN.md`|`59a664e5a5ef1890505c99e4fe0312dda39f74404f4543173e530eeaac8edb67`|冻结合同|

- `ssr-gpu`目标测试：`81 passed、3 Windows POSIX skipped、0 failed`；相邻DSSC：`36 passed、0 failed`；`py_compile`与`git diff --check`通过。
- 真实checkpoint support-only CUDA smoke：首失败row三scene×before/after共6个state，BCR top1均为1.0，any/large flip均为0，26类最坏logit误差`6.58e-8`；query/truth读取和fit query rows均为0。
- K1/K5/K10 parent/r4对比：qKNN wire、bank codes、metric、BCR lambda、analytic/directional LOO、raw/dual `omega`和domain alpha逐项相等。
- 资源：C26 BCR wire=`12,636B`，最大总state=`163,903B<256KiB`；第三平面解码mean=`20.50µs`、P95=`31.8µs`，每次新增4,160乘法和4,160加法。
- 独立终裁：`MERGE / P0=0 / P1=0 / P2=0`。

## 发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_full125_802534eb_20260724_033904\input`

|输入|SHA256|说明|
|---|---|---|
|`source_git_blobs_802534eb.zip`|`e340bcc9c908223f9f0cdd003a88924dddc12b117b43cbc287fa4626b6d17c89`|34,560,096B；3980个safe regular raw Git blob；raw bytes=`231,138,377B`；path-set SHA=`e2b3c999af049d1f872c3b91c1044b8ceac71f135ec39d01140e8ac5e77a89e0`；missing/extra/raw mismatch/unsafe=`0/0/0/0`|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|2,202B；复用冻结method lock|

外部输入保持checkpoint=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`、sealed runtime=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`及既有GEOFF/r8 cache/authority路径。

## N607路径、命令与launch owner

远端run根创建前必须为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_full125_802534eb_20260724_033904`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`，GPU固定为0–7。正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

日志固定为`<run>/logs/matrix.stdout.log`、`<run>/logs/matrix.stderr.log`和`<run>/logs/matrix.exit`。唯一Terra runner负责direct preflight、资源/无关进程记录、精确同步、远端SHA与安全解包、关键文件`py_compile`、POSIX sentinel、run根不可覆盖检查、唯一detach、PID/CWD/cmdline/GPU核验、短连接健康监控、完整日志读取、artifact回收和结构化handoff。主agent不得并发启动同一run。

## 健康止损与完成条件

- 启动后立即检查首波row、异常指纹、prediction生成和GPU映射；系统性零prediction技术故障必须停止继续派发并只终止本run，禁止自然跑完、retry或复用run ID。
- 若再次出现`feature row has zero or non-finite L2 norm`，按新run的唯一首源回收最小support-only证据；不得在本revision顺带修改。
- 只有125/125 row receipt、1000/1000 prediction、1500/1500 logical score row、完整completion/archive/coverage绑定闭合后才进入性能分析。
- query不得进入fit/state；wrong checkout/hash、artifact绑定失败、资源/INT8生命周期越界为发布阻塞。P2格式、报告美化和重复数据/authority/hash审计不得扩展。

## 性能输出与裁决

完成后必须同row报告old-before、old-after、DA old gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、coverage、量化margin、MAC、时延、显存、state bytes和`I_syn`。

立即证伪：`M_DA`无净正确增益；`M_OTHER`无独立正收益；`M_JOINT.H<=max(M_DA.H,M_OTHER.H)`；mean`I_syn<=0`；联合臂损害old-after、seen-new、floor、min-old或min-new或增加forgetting；正协同不足188/375个scene slice或不足2/3个scene均值为正。完成完整prediction但不达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

## 执行状态

|字段|当前值|
|---|---|
|Git commit|`802534eb8036fb8a31f060fd55af5050d0fe7961`|
|run ID|`adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_full125_802534eb_20260724_033904`|
|remote PID/exit|`NOT_LANDED / NOT_LAUNCHED`|
|prediction/score|`0/1000`；`0/1500`|
|最终裁决|`PREREGISTERED / NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`

