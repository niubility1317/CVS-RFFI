# CVS-RFFI Phase2 states T1 fresh v13运行报告

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2_states_t1_20260730_v13_d1f5e45c`|
|目标|执行325逻辑/325物理states正式矩阵；8 GPU×2槽，上限包含外部进程|
|Git执行release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2_states_t1_20260730_v12_d1f5e45c`，HEAD=`d1f5e45c72f20e6d81ea5d6fef5e05fcd5f56f0e`，clean|
|复用输入|125项cache/package/scorer index；325项官方binding registry；Stage2-A v3 sidecar 25/25；Stage2-B v3 sidecar 25/25；feature cache 300/300|
|Conda/Python|`CVS-RFFI`；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|request root|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2_states_t1_20260730_v13_d1f5e45c`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2_states_t1_20260730_v13_d1f5e45c`|
|row log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_states_t1_20260730_v13_d1f5e45c`；启动前不存在|
|driver log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2_states_t1_20260730_v13_d1f5e45c_driver`；与sealed row log root分离|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|

v12首次landing因driver提前创建sealed `log_root`而在0 dispatch处fail-closed，原日志保留且不续跑。v13只修正driver日志根位置，不改变plan、method、数据、binding、GPU队列或停止规则。启动后按首个完成/失败row和首个worker wave检查physical/logical/prediction/score计数、run-owned PID、GPU槽位和标准化异常指纹；不读取性能值。两个不同row在prediction前出现同一确定性异常指纹或任何P0时，由sole runner停止其自身精确进程树并保留制品。

## 技术停止闭合

|字段|结果|
|---|---|
|main PID|`1033160`；启动时CWD与cmdline精确绑定d1f release和v13 sealed plan，停止后不存在|
|runner summary|`launched_physical_count=15`、`completed_physical_count=0`、`completed_logical_score_count=0`、`failed_physical_count=17`、`not_launched_systemic_stop_count=308`、`systemic_stop=true`|
|prediction/score|0/0；`performance_values_visible_to_scheduler=false`|
|P0|0|
|主异常指纹|`95766ca70b2136846cc1032da821ebc436670073dc9cfd0f26e7cb21c0389d29`，14个不同row在prediction前重复|
|根因|`stage2_ablation_row_executor.py:416`在该子进程首次CUDA初始化前调用`torch.cuda.reset_peak_memory_stats(runtime_device)`，PyTorch抛出`RuntimeError: Invalid device argument 0: did you call init?`|
|停止与清理|runner按系统性零prediction故障自动停止；308项未派发；run-owned进程=0；GPU只剩外部PID`957815`；本地`ssh.exe=0`、N607 TCP22连接=0|
|处置|原v13永久保留，不续跑、不覆盖；本地修复CUDA初始化顺序并完成测试、独立复审、Git提交后，使用fresh v14 run ID重新发布|
