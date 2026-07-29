# CVS-RFFI Phase2 states T1 fresh v14 CUDA初始化修复运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|仅修复row executor首次CUDA统计重置前未初始化device的问题，重新执行325逻辑/325物理states矩阵|
|Git实现|待v14修复commit与独立复审后回填；源plan及每行`git_commit`必须重新绑定该commit|
|状态|`PREREGISTERED / LOCAL_FIX_VERIFIED / REMOTE_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`|

## 复用与fresh路径

|字段|值|
|---|---|
|formal release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`；必须fresh clone到v14 commit且tracked/untracked clean|
|source plan|必须按v14 commit同时重新生成325行states与1425行Stage2-C两个源plan；旧d1f两个plan均作废|
|binding registry|复用v12已由官方builder逐项验证的325 logical binding registry；其125项唯一输入identity不变|
|Stage2-A scorer|复用v11的25/25 v3 sidecar|
|Stage2-B scorer|复用v12的25/25 v3 sidecar|
|feature/package|复用300/300 feature cache、v9 24份package与v5 1份reuse package|
|request root|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`|
|row log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`|
|driver log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init_driver`；必须与row log root分离|
|input/seal root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`|

## v13关闭证据

|字段|结果|
|---|---|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|派发/完成|15 physical launched；0 physical complete；0 logical score complete|
|prediction/score|0/0|
|主异常指纹|`95766ca70b2136846cc1032da821ebc436670073dc9cfd0f26e7cb21c0389d29`，14个row重复|
|未派发|308|
|P0|0|
|根因|首次CUDA初始化前调用`torch.cuda.reset_peak_memory_stats(cuda:0)`|
|清理|原v13不续跑、不覆盖；run-owned进程=0；本地SSH连接=0|

## v14启动门

启动前只执行必要检查：direct N607 preflight；fresh release精确commit与clean；新states source plan的顶层及325行`git_commit`均绑定v14 commit；复用registry经新release官方validator通过；sealed plan为325 logical/325 physical、16个worker槽、P0/P1=0；plan run/log root与独立driver root均不存在；记录当前GPU外部进程并确保每卡总训练进程不超过2。不得重做D18数据校验，不重建feature/package/sidecar，不读取性能值。

本轮只允许seal并launch 325行states。同步重生成的1425行Stage2-C source plan仅作为后续输入保存，在states达到`ARTIFACTS_COMPLETE`前不得seal或launch；届时必须另建fresh run ID、request/run/log/driver根并取得独立启动授权。

首个完成或失败row及首个worker wave检查main/child PID、CWD/cmdline、GPU映射、physical/logical/prediction/score计数和标准化异常指纹。两个不同row在prediction前产生同一确定性指纹或任何P0时，只停止本run精确进程树并保留全部制品。

## 独立复审

2026-07-30独立复审结论为`P0=0 / P1=0`。复审确认CUDA执行顺序为device选择、context初始化、峰值统计重置；初始化失败先于row输出根创建。启动模板在所有commit、clean、fresh检查前启用`set -euo pipefail`，并保留`noclobber`；325行states与1425行Stage2-C源plan均须绑定同一v14代码commit重新生成。
