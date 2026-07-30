# CVS-RFFI Phase2-C T1完整消融运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2c_t1_20260730_v1_14552df1`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|完整执行设计报告Stage2-C screening矩阵，不基于中间性能缩小范围|
|比较目标|19个已注册arm在5个receiver、3个seed bundle、K∈{1,2,5,10}、new_count∈{5,20}上的同计划比较|
|Git实现|`14552df1ca50f8fe100621f5fd4f099942b08322`|
|状态|`PREREGISTERED / EXACT_INPUT_COMPLETION_LOCAL_VERIFIED / WAITING_FOR_STATES_ARTIFACTS_COMPLETE / REMOTE_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`|

## 固定矩阵与复用边界

|字段|值|
|---|---|
|source plan|`stage2c_screening_plan_14552df1.json`；1425个唯一logical row key；独立复审`P0=0 / P1=0`|
|预期封存规模|1425 logical；1350 physical；`P2-F3`仅按既有正式别名规则复用`P2-FULL`|
|前置屏障|`cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`必须达到`ARTIFACTS_COMPLETE`，325/325 prediction与score闭合且0失败|
|输入复用|仅复用已验证D18 IQ cache、formal Phase1 deployment和既有50行package命令矩阵中的不可变输入路径；旧50行摘要是历史失败运行的命令模板证据，不作为成功产物复用。其文件、50个receiver/seed/stage身份和除允许变化字段外的命令结构均被固定；screening seeds与旧confirmation cache不相同，因此fresh构建45个package、75个精确Stage2-C feature identity和30个Stage2-C v3 sidecar；不重审D18数据，不要求跨批次数据identity/hash一致|
|协议|`p2_min_v1`；query test-only；无clean/source访问、无query truth、无role oracle、无class quota|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|并发|8张GPU，每卡最多2个predictor；启动前记录外部GPU进程并计入上限|

## fresh服务器路径

|字段|值|
|---|---|
|formal release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2c_t1_20260730_v1_14552df1`|
|package root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v1_14552df1`|
|feature root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v1_14552df1`|
|sidecar root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v1_14552df1`|
|input/seal root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v1_14552df1`|
|request root|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v1_14552df1`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v1_14552df1`|
|row log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v1_14552df1`|
|driver log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v1_14552df1_driver`|

## 启动与健康门

启动前执行direct N607 preflight；确认states前置屏障；fresh release精确checkout固定commit且tracked/untracked clean。先用Git承载的controller从既有合法D18 cache构建15个before、15个new20、15个new5 package；再以8GPU×2槽构建60个new20和15个new5 Stage2-C exact feature identity，同时发布并正式重载30个v3 Stage2-C truth sidecar。随后生成75-entry cache-binding index，用当前1425行plan调用官方registry builder逐项验证，sealed plan必须为1425 logical/1350 physical、formal launch authority=true；所有fresh根必须不存在。不得重做数据验证，不得读取中间性能选择row或arm。

输入补齐命令见`release_evidence/package_launch_template.txt`、`feature_launch_template.txt`和`sidecar_launch_template.txt`，输入汇总验证、75-entry index、1425-entry registry及sealed plan生成见`release_evidence/seal_launch_template.txt`，正式矩阵命令见`release_evidence/launch_template.txt`。输入summary在下游使用前必须以完成后记录的detached SHA重新验证，同时与固定45/75/30精确身份集合逐项比较，并重新调用正式package、feature和sidecar loader检查内部产物；占位符未替换时失败关闭。source plan必须固定为1425行、19 arms×75 identities；sealed plan必须固定为1425 logical/1350 physical、formal authority=true并绑定同一source plan，两个阶段都使用完成后记录的detached SHA验证。feature调度每波读取N607全部现有compute进程，并按每卡`max(0,2-existing)`分配本任务槽位。正式模板必须先运行`verify_states_completion.py`，对固定states sealed plan和`runner_summary.json`执行detached SHA、schema、run ID、commit、325/325计数、0失败、0系统停派、全部prediction/score正式validator及逐physical status一致性校验；两个states SHA占位符在states完成并独立复审前不得替换。detached launch后立即记录main PID、CWD/cmdline、release/run/log绑定、GPU映射和日志增长。首个row、首个worker wave及后续短连接快照记录logical/physical、prediction/score、成功/失败和标准化异常指纹。

仅当任何P0出现，或至少两个不同row在prediction前产生同一确定性异常指纹时，停止本run精确进程树并保留全部制品；不得因accuracy、H、BA、floor等中间性能停止或缩小矩阵。

## 完成判据

`ARTIFACTS_COMPLETE`要求sealed matrix完整执行，所有预期physical prediction、logical score、row exit、coverage/archive/receipt制品通过正式validator，失败数为0，且无缺行、覆盖或覆写。完成后在本报告追加逐实验同一行结果表、异常、解释和下一步结论。

## 本地实现与验证

|文件|用途|
|---|---|
|`stage2c_package_completion_controller.py`|从已验证50行命令矩阵保留不可变输入路径，fresh生成45个screening-seed package|
|`stage2c_feature_completion_controller.py`|按8GPU×2槽构建75个精确Stage2-C identity，并用当前formal loader重载全部225个伴生scope cache|
|`stage2c_sidecar_completion_controller.py`|从30个Stage2-C package truth发布v3 sidecar并用当前formal loader逐项验收|
|`build_stage2c_cache_binding_index.py`|生成与1425行计划身份集合完全一致的75-entry cache-binding index|
|`verify_states_completion.py`|机器验证固定states run的完整产物屏障|
|`verify_input_completion_summary.py`|以detached summary SHA、完整身份集合、逐条成功状态和实际产物路径验证package/feature/sidecar闭环|
|`verify_stage2c_plan_identity.py`|固定1425行source plan和1425/1350 sealed plan的矩阵身份、Git提交、formal authority及detached SHA|

本地`compileall`和5个Bash模板语法检查全部通过；静态矩阵验证为package 45项（15 before+15 new20+15 new5）、feature 75项（60 new20+15 new5，动态计入外部GPU进程后每卡总数≤2）、sidecar 30项/60文件；cache index的75个identity与1425行source plan去重后的身份集合完全相等。predictor package、feature builder、scoring sidecar、binding registry和release相邻5个测试文件连同focused测试共65项全部通过；focused测试覆盖旧50行来源身份/命令结构、GPU动态空槽、1425行source plan、summary摘要绑定及精确身份集合失败关闭。
