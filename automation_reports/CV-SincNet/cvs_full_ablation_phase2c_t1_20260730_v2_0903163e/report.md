# CVS-RFFI Phase2-C T1完整消融v2运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2c_t1_20260730_v2_0903163e`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|完成设计报告Stage2-C全部1425行screening矩阵，不基于中间性能缩小范围|
|比较目标|19个arm×5个receiver×3个method seed×5个K/new_count身份|
|正式release代码|`0903163e12902114b2685722125b1d304b97e2fa`|
|修复目标|关闭v1最终summary中`PosixPath`不可JSON序列化的统一技术故障；package、feature、sidecar三个controller同时修复|
|状态|`LOCAL_VERIFIED / INDEPENDENT_REVIEW_P0_0_P1_0 / FRESH_REMOTE_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`|

## v1边界与v2非覆盖要求

v1=`cvs_full_ablation_phase2c_t1_20260730_v1_14552df1`固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。v1的input、package、driver、日志和部分产物全部保留；v2不得读取v1 package产物作为正式完成证据，不得补写v1 summary，不得覆盖或复用v1任何输出根。

## 固定矩阵与前置证据

|字段|值|
|---|---|
|source plan|`stage2c_screening_plan_0903163e.json`|
|source plan SHA256|`e652802a60960a207e915570f48e9cf502e9e80927939b640d6f3349e9bb4d60`|
|矩阵|1425个唯一logical row；19 arms×75 identities；预计1350个physical execution，`P2-F3`仅复用同身份`P2-FULL`prediction|
|states前置|`cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`已`ARTIFACTS_COMPLETE`：325/325 prediction与score闭合，失败0、缺件0|
|输入补齐|fresh构建45 package、75 feature identity和30 sidecar；不重新审计D18数据，不要求不同启动间数据相同|
|协议|`p2_min_v1`；query test-only；无clean/source访问、无query truth、无role oracle、无class quota|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU并发|8张GPU，每卡总compute进程最多2个；每波动态扣减服务器既有compute进程|

## fresh服务器路径

|字段|路径|
|---|---|
|formal release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2c_t1_20260730_v2_0903163e`|
|input/seal|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v2_0903163e`|
|package|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e`|
|feature|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v2_0903163e`|
|sidecar|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v2_0903163e`|
|request|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v2_0903163e`|
|run|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v2_0903163e`|
|row log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v2_0903163e`|
|driver|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v2_0903163e_driver`|

## 发布链与完成门

精确命令分别见`release_evidence/package_launch_template.txt`、`feature_launch_template.txt`、`sidecar_launch_template.txt`、`seal_launch_template.txt`和`launch_template.txt`。release必须精确checkout代码commit=`0903163e`且tracked/untracked clean；12个fresh目标根启动前全部不存在。

package完成门为summary存在且45/45 launched/completed/succeeded/validated、失败0，并以detached SHA、45个精确身份和正式package/scoring loader逐项重载。feature完成门为75/75、225个scope cache、GPU总进程上限2，并重载全部stage2a/b/c cache。sidecar完成门为30/30、60个文件，并以正式scorer loader重载。随后生成75-entry index、1425-entry registry及1425 logical/1350 physical sealed plan；source与sealed plan均须通过detached SHA和精确矩阵验证。

正式launch前再次运行states完整产物屏障、三类输入完整门和sealed plan门。detached launch后核对main PID、CWD/cmdline、run/log绑定、16个GPU槽和日志增长。首个row、首个worker wave及后续短连接报告prediction/score/失败/异常指纹计数。任何P0或两个不同row在prediction前产生同一确定性异常指纹时，只停止v2精确进程树并保留产物；不得因accuracy、H、BA或floor停止。

## 本地验证与独立复审

三个controller统一将进入summary的task `Path`转换为字符串；其余summary字段均为JSON原生类型。`compileall`通过；focused test新增package/feature/sidecar三控制器summary序列化覆盖；相邻完整测试共68项通过。独立复审确认代码修复和v1技术关闭均为`P0=0 / P1=0`，允许创建fresh v2。

## 完成后检查

完成后在本报告追加package、feature、sidecar及sealed plan的完整计数、SHA、PID/GPU/SSH清理证据，以及1425行同一实验结果表。主解释必须基于同一候选同一行的old/seen-new/unknown、coverage、rollback/defer、loss/adapter和最终判定，不使用跨行拼接的单指标极值。
