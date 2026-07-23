# ADV3B02/r4-bcr3-procbindfix1完整125发布报告

- run ID：`adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_procbindfix1_full125_467b8aa5_20260724_035813`
- candidate：`ADV3B02-TS-DRQKNN-BCRR/r4-q2f32-bcr3-zidtotal1`
- scientific Git commit：`802534eb8036fb8a31f060fd55af5050d0fe7961`
- release-fix Git commit：`467b8aa561f41eada827c48588e8c6598b49eed0`
- 创建时间：`2026-07-24T03:58:13+08:00`
- operator：主agent；唯一N607 launch owner为单一`gpt-5.6-terra high`runner
- 状态：`PREREGISTERED / NOT_LANDED / NO_PERFORMANCE_RESULT`
- parent run：`adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_full125_802534eb_20260724_033904`
- parent终态：`RELEASE_BLOCKED_PRELAUNCH_SYSTEMIC_RUNNER_SAFETY_FAILURE / NO_PERFORMANCE_RESULT`

## 目标与冻结边界

本run用GPU0–7执行完整125，验证`z_id/z_dom`双qKNN域适应、统一qKNN分类和三平面BCRR是否形成同row正协同。科学方法、四臂`M0/M_DA/M_OTHER/M_JOINT`、`I_syn`、125矩阵、scorer、数据、资源门和健康止损均继承scientific commit，不作任何改变。

唯一release fix是共享runner在`Popen`后读取`/proc/<pid>/cmdline`时，仅对空结果执行最多50次、每次10ms的有界重读。首个非空cmdline立即进入原有严格CWD/cmdline/PGID/output-root验证；进程退出、读满仍空或任一非空身份不匹配均fail-closed。该修复不新增gate、receipt、validator或数据检查。

冻结矩阵为5receiver×5seed×`{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`；每job覆盖3个LEO弱场景。期望125份row receipt、375个scene slice、1000份arm-state prediction和1500个logical score row。禁止partial性能解读。

## 本地闭合与独立review

|文件|SHA256|用途|
|---|---|---|
|`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`|`f61a33994e5dca24428f92c544732bd10e391b4774f4d757cb61e7e43d71f1d0`|冻结r4科学方法|
|`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`|`ecf8a23f87c304206ca8cc31f848743138a3dff37ef35d2a8947dd168737c7dd`|共享POSIX进程身份捕获修复|
|`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`|`9bfce16e5033e1dfd1aca6aca33ba08e5f66886a735abf3f7d31a3a56062cbf3`|空读重试、退出和耗尽负例|

- `ssr-gpu`目标测试文件通过；3项仅在Windows跳过的POSIX专项须在N607启动前实跑。
- `py_compile`和`git diff --check`通过。
- 独立Terra终审：`MERGE / P0=0 / P1=0 / P2=0`。
- r4既有真实checkpoint无query smoke、INT8、state、MAC和时延证据保持不变；本fix不重复科学验证。

## 发布输入

本地输入根：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_procbindfix1_full125_467b8aa5_20260724_035813\input`

|输入|SHA256|说明|
|---|---|---|
|`source_git_blobs_467b8aa5.zip`|`0d4bb25384972d62250c9467778745bc3d80daac378cf5e0b29d48708ea86108`|34,566,690B；3,981个safe regular raw Git blob；raw bytes=`231,151,918B`；path-set SHA=`8ad180b9260df0e881fd48bfb6facf8968cb464cf59dbe8eb48f03a7deacae76`；raw mismatch=`0`|
|`somph_method_lock.json`|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|2,202B；复用冻结method lock|

外部输入继续使用checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`、sealed runtime SHA=`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`和既有GEOFF/r8 cache/authority路径。

## N607路径与正式命令

远端run根创建前必须为ABSENT：

`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_procbindfix1_full125_467b8aa5_20260724_035813`

Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为`<run>/source/code`，GPU固定为0–7。正式child command：

```text
PYTHONPATH=<run>/source/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python scripts/run_adv3b02_ts_drqknn_bcrr_125.py matrix --phase1-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --sealed-runtime /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt --package-method-lock <run>/input/somph_method_lock.json --cache-root /home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix --authority-root /home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1 --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

日志固定为`<run>/logs/matrix.stdout.log`、`<run>/logs/matrix.stderr.log`和`<run>/logs/matrix.exit`。唯一Terra runner负责direct preflight、精确同步、远端SHA和安全解包、关键文件`py_compile`、冻结POSIX sentinel、run根不可覆盖核验、唯一detach、PID/CWD/cmdline/GPU核验、启动后健康检查、短连接监控、完整日志读取、artifact回收和结构化handoff。主agent不得并发启动同一run。

## 健康止损与完成条件

- POSIX sentinel若仍失败，保持`RELEASE_BLOCKED_PRELAUNCH / NO_PERFORMANCE_RESULT`，不得启动矩阵。
- 启动后立即检查不同row异常指纹、prediction生成和GPU映射；至少2个不同row在零prediction处出现同一确定性异常时，立即停止继续派发并只终止本run。
- 不得按partial性能早停；不得修改方法、参数、矩阵或重验数据。
- 只有125/125 row receipt、1000/1000 prediction、1500/1500 logical score row和完整completion/archive/coverage绑定闭合后才进入性能分析。

## 待回填性能与裁决

完成后同row报告`M0/M_DA/M_OTHER/M_JOINT`的old-before、old-after、DA old gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new/new→old、逐类/receiver/scene/K/seed/new-count、coverage、量化margin、MAC、时延、显存、state bytes和`I_syn`。无完整prediction只能标记`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完整prediction不达门标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

|字段|当前值|
|---|---|
|Git commit|`467b8aa561f41eada827c48588e8c6598b49eed0`|
|run ID|`adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_procbindfix1_full125_467b8aa5_20260724_035813`|
|remote PID/exit|`NOT_LANDED / NOT_LAUNCHED`|
|prediction/score|`0/1000`；`0/1500`|
|最终裁决|`PREREGISTERED / NO_PERFORMANCE_RESULT`|

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
