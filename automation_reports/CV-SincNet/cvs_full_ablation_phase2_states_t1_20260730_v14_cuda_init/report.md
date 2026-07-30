# CVS-RFFI Phase2 states T1 fresh v14 CUDA初始化修复运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|仅修复row executor首次CUDA统计重置前未初始化device的问题，重新执行325逻辑/325物理states矩阵|
|Git实现|代码commit=`14552df1ca50f8fe100621f5fd4f099942b08322`；plan/report封存commit=`68733dd6a77cd6390958b7be6e29adca2ac1a21a`；独立复审P0=0/P1=0|
|状态|`ARTIFACTS_COMPLETE / ANALYSIS_PENDING`|

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

## 发布与首波证据

|字段|结果|
|---|---|
|direct preflight|PASS；启动前8张GPU均无外部进程|
|release|HEAD精确为`14552df1ca50f8fe100621f5fd4f099942b08322`且tracked/untracked clean；远端CUDA init/reset smoke PASS|
|源plan|states SHA256=`9f31dd67766637229d55c351a0706924721b2d2df81a8149faa2ac5b2b923084`；Stage2-C SHA256=`9ff944572d8513b751a7c4971307b7a590ce4b5c5c7e35189a1a5bac0520dffb`，后者未seal|
|registry/seal|registry=325 logical bindings；sealed plan=325 logical/325 physical；formal authority=true；predict/score requests=325/325|
|registry/seal SHA256|registry=`c80b2f8734c2d04e3b55fca25669cc0804b91ce5ca2ed6c9b658c1061569454a`；sealed plan=`90f9e489ff4fe739969aa76e7fd85d36bb5f72dadc934b517cbd979b50d12ed8`|
|main|PID=`1046265`；CWD与cmdline精确绑定v14 release和sealed plan|
|首波计数|`completed=43`、`succeeded=43`、`failed=0`、`prediction_complete=43`、`scores_complete=43`、`P0=0`、`exception_fingerprints={}`|
|GPU槽|16个活跃predictor按`CUDA_VISIBLE_DEVICES`分布为GPU0–7各2个；外部Phase1进程=0；每卡总训练进程=2|
|可观测增长|status从28增至43；row log=59个、163954 bytes；16个predictor均为运行态并处于高CPU特征阶段|

首波仅证明技术健康，不读取或报告任何性能值。继续使用短连接监控直至runner summary与325份同row制品闭合。

## 运行中增量检查

|字段|结果|
|---|---|
|最新技术健康计数|`completed=116/325`、`succeeded=116`、`failed=0`、`prediction_complete=116`、`scores_complete=116`、`P0=0`、`exception_fingerprints={}`|
|进程状态|main PID=`1046265`仍存活；监控时仍观察到run-owned predictor持续换波运行；`runner_summary.json`尚未生成|
|后续输入只读检查|5个receiver各5份target-cache-set可读，共25/25；Phase1 `deployment_binding`、`method_lock`、checkpoint和component均可读|
|复用边界|只检查路径为普通可读文件；未打开或重验D18内容，未要求跨启动数据一致，未启动package、feature、sidecar或Stage2-C|

## 最终技术闭环

|字段|结果|
|---|---|
|runner summary|`logical=325`、`physical=325`、`launched=325`、`completed=325`、`logical score complete=325`、`failed=0`、`not launched=0`、`reused=0`|
|运行健康|`systemic_stop=false`、`failure_fingerprints={}`、`thread_errors=[]`、`performance_values_visible_to_scheduler=false`|
|sealed plan|formal authority=true；325 logical/325 physical；alias=0；绑定代码commit=`14552df1ca50f8fe100621f5fd4f099942b08322`|
|矩阵覆盖|`P2-S2A=25`；`P2-S2B-FULL=100`；`P2-S2B-PROTO=100`；`P2-S2B-DIAGOFF=100`|
|逐行闭环|325个唯一physical ID、325个唯一logical row key、325份status、325个physical目录、325份prediction、325份logical score、325份score completion、325份row log，缺失0|
|receipt边界|325/325为`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`；prediction artifact、payload、manifest和seal标识齐全；只读标记为true；`fit_query_rows_used=0`；`query_truth_opened=false`|
|请求闭环|predict request=325；score request=325|
|进程/GPU清理|main PID=`1046265`已退出；run-owned child=0；N607 GPU compute process=0；本地`ssh.exe=0`；N607/bridge TCP22连接=0|
|关键文件SHA256|sealed plan=`90f9e489ff4fe739969aa76e7fd85d36bb5f72dadc934b517cbd979b50d12ed8`；runner summary=`abfa9fb8da66ee4024f01d048a77079a2555572b7e6430ed4530f746308c7a9a`|
|Stage2-C边界|源plan仍未seal、未launch；等待独立复审与Git release commit|

本节只确认执行与制品闭环，不读取性能值，不形成候选排序或性能结论。v14由`RUNNING`推进到`ARTIFACTS_COMPLETE`，后续性能分析应读取完整同row制品后另行记录。

## 独立复审

2026-07-30独立复审结论为`P0=0 / P1=0`。复审确认CUDA执行顺序为device选择、context初始化、峰值统计重置；初始化失败先于row输出根创建。启动模板在所有commit、clean、fresh检查前启用`set -euo pipefail`，并保留`noclobber`；325行states与1425行Stage2-C源plan均须绑定同一v14代码commit重新生成。

## v14代码与源计划封存

|字段|结果|
|---|---|
|v14代码commit|`14552df1ca50f8fe100621f5fd4f099942b08322`|
|CUDA相关回归|row executor、executors、release、quantization共4个测试文件全部通过|
|states源plan|`stage2_states_plan_14552df1.json`；325/325个唯一row key；顶层及每行commit均为v14代码commit|
|Stage2-C源plan|`stage2c_screening_plan_14552df1.json`；1425/1425个唯一row key；顶层及每行commit均为v14代码commit|
|计划回归|plan builder、full spec、Stage2 release共3个测试文件25项通过|
|与旧计划关系|除`git_commit`外分别与已审d1f版325行及1425行计划完全一致；旧d1f commit在新计划中出现0次|
|计划独立复审|`P0=0 / P1=0`；官方行验证通过；states保持25行Stage2-A+300行Stage2-B，Stage2-C保持19个arm×75行|
