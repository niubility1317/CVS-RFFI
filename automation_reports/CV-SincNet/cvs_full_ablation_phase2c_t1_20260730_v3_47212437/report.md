# CVS-RFFI Phase2-C T1完整消融v3运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|完成设计报告Stage2-C全部1425行screening矩阵，不基于中间性能缩小范围|
|正式release代码|`4721243770bee654bb7c41ad0bdd128d2dbfb863`|
|修复目标|package阶段使用source-sidecar loader读取合法v2 schema；最终发布的sidecar继续由formal scorer loader验收v3 schema|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / ORIGINAL_ROOTS_PRESERVED / NO_PERFORMANCE_RESULT`|

## v1/v2边界与允许复用

v1与v2均固定为技术关闭且`NO_PERFORMANCE_RESULT`，不得patch、补写、覆盖或从原run继续执行。v3仅只读复用v2已成功发布的45个package及其不可变summary：

|复用项|固定值|
|---|---|
|package root|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e`|
|package summary SHA256|`ed61d7c8403c203f4b1c73d4dd87251a67cf01b7da7f6f580143dfa852b026cb`|
|summary计数|expected/launched/completed/succeeded/validated均45；failed=0；systemic_stop=false|
|package制品|45个predictor manifest/seal、45个scoring manifest/source truth sidecar；package gate以predictor正式preflight及source-sidecar loader逐项重载|

该复用符合用户“前面的能复用就复用”和项目允许复用既有合法输入的要求。v3不重新审计D18数据，不要求不同启动的数据相同。

## 固定矩阵

|字段|值|
|---|---|
|source plan|`stage2c_screening_plan_47212437.json`|
|source plan SHA256|`6cc5fcba0194adaa65efb1c238479c7ac77d160228c2e1227db91d1852ffcbe4`|
|矩阵|1425 logical；19 arms×75 identities；预计1350 physical|
|states前置|325/325 `ARTIFACTS_COMPLETE`，失败0、缺件0|
|待fresh生成|75 feature identities、225 scope cache、30 formal v3 sidecar/60文件、75-entry index、1425-entry registry和sealed plan|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU并发|8张GPU，每卡总compute进程最多2个；每波动态扣减既有compute进程|

## v3 fresh服务器路径

|字段|路径|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|input/seal|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|feature|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437`|
|sidecar|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437`|
|request|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|run|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|row log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437`|
|driver|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v3_47212437_driver`|

## 发布链与门禁

`package_launch_template.txt`在v3中只做v2 package只读验证，不创建或修改package。验证45/45后，`feature_launch_template.txt`和`sidecar_launch_template.txt`使用fresh v3输出根启动；feature完成门为75/75、225个scope cache且正式loader重载全部成功，sidecar完成门为30/30、60个文件且formal v3 loader逐项成功。`seal_launch_template.txt`生成75-entry index、1425-entry registry和1425 logical/1350 physical sealed plan。`launch_template.txt`再次执行states、package、feature、sidecar、source/sealed plan门后启动全部1425行。

启动前release必须精确HEAD=`47212437`且clean；所有v3目标根必须fresh不存在。首个feature wave、sidecar完成、seal完成、首个正式row和首个worker wave均记录计数、PID/CWD/cmdline、GPU槽、日志增长和异常指纹。仅P0或两个不同row在prediction前出现同一确定性故障时停止v3精确进程树；不得因性能值停止或缩小矩阵。

## 本地验证与复审

package gate保留predictor正式preflight，将scoring侧改为`cvsrffi.stage2_scoring_sidecar`读取source v2；sidecar controller随后发布并以`cvsrffi.stage2_metric_scorer`正式重载v3。新增loader分层focused test，完整相邻测试69项通过，compileall通过。独立复审结论`P0=0 / P1=0`，允许只读复用v2 package并创建fresh v3。

## N607落地、输入闭合与正式启动

直接N607预检通过，服务器可见8张GPU，正式启动前GPU为空闲；Python固定为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`（3.10.19）。release以增量Git bundle落地，bundle SHA256=`e399d071ef762a5a5961393b743fccbfe6350522d181d427ad3eed891e0be5b3`；远端HEAD精确为`4721243770bee654bb7c41ad0bdd128d2dbfb863`且工作树clean。同步的7个Python文件、5个模板和source plan均逐项匹配本地SHA256，远端compile、shell语法与plan identity验证通过。

|启动前制品|闭合结果|SHA256|
|---|---:|---|
|复用v2 package|45/45验证通过；只读复用；失败0|`ed61d7c8403c203f4b1c73d4dd87251a67cf01b7da7f6f580143dfa852b026cb`|
|v3 feature|75/75完成；225/225 scope cache；失败0|`423ea50e5f74a1dcc055f605d093d699865f371b92dd0ee7960a31993c94a91d`|
|v3 formal sidecar|30/30完成；60/60文件；失败0|`7d7de4a6dbb22caa5ec4cde594cdf66dc678423c8639882af99cbfce88bb5c45`|
|cache index|75 entries|`85fe57e8b6234a660e687165ac900439a2a0089a276fda20ec8e78fe1a7b190b`|
|registry|1425 entries|`a413e160bcc9bfa0a6c40864b5f716c0dc5805192a3ff3ba73d708dbd2c430a4`|
|sealed plan|1425 logical；1350 physical；75 aliases|`d5a6e0b9154d2d9fdb938628bb489cb9d802079a97be71f3cc77785e374636d8`|
|前置states sealed plan|325/325完成；失败0|`90f9e489ff4fe739969aa76e7fd85d36bb5f72dadc934b517cbd979b50d12ed8`|
|前置states runner summary|`ARTIFACTS_COMPLETE`|`abfa9fb8da66ee4024f01d048a77079a2555572b7e6430ed4530f746308c7a9a`|

2026-07-30正式启动前，`launch_template.txt`再次复验release、states、source/sealed plan、package、feature和sidecar，所有门均通过后才创建fresh run/rowlog/driver根并启动冻结矩阵。主进程PID=`1390511`，CWD精确为上述release目录，cmdline绑定上述sealed plan、正式row predictor和truth-side scorer。首次只读短连接核验到16个直属worker，`CUDA_VISIBLE_DEVICES=0..7`各2个，单进程显存约338MiB，没有超出每卡2进程。

|健康快照|physical完成/总数|预测/评分闭合|失败|异常指纹|P0|判定|
|---|---:|---:|---:|---:|---:|---|
|首次完成row|3/1350|3个status均`prediction_complete=true`且`scores_complete=true`|0|0|0|继续完整矩阵，不中断|
|首个完整worker波次|18/1350|prediction=18；score=18；logical score=18|0|0|0|16个worker继续运行，GPU0–7各2个|

监控采用独立短SSH连接，快照命令结束后连接已退出；未开启交互shell、端口转发或持久复用连接。当前仅报告执行健康，不读取或解释任何性能指标。

## v3系统性技术停止与闭合

runner在首个确定性prediction前故障仅出现1个row时继续运行；当同一指纹出现在第2个不同physical row后，按冻结规则自动停止派发并终止其自有在途worker。监控方没有手工停止、重启、续跑或覆盖该run。

|类别|physical数|logical输出数|说明|
|---|---:|---:|---|
|`COMPLETE`|73|73|prediction与score均闭合；仅保留为部分技术制品，不形成性能结果|
|`FAILED`|17|17|2个触发故障；14个为系统停止时终止的在途predictor；1个为系统停止时中断的scorer|
|`NOT_LAUNCHED_SYSTEMIC_STOP`|1260|1335|含75个logical alias；均发布显式失败闭合记录|
|合计|1350|1425|与sealed plan完全一致|

runner summary记录`launched_physical_count=90`、`completed_physical_count=73`、`failed_physical_count=17`、`not_launched_systemic_stop_count=1260`、`completed_logical_score_count=73`、`systemic_stop=true`、`thread_errors=[]`和`performance_values_visible_to_scheduler=false`。summary大小825541 bytes，SHA256=`eb9175ce6a27bf4f48e2be41b876b8832802cd6b57c7aea567cb50add3e96fa8`。

|指纹|distinct failed rows|分类|证据|
|---|---:|---|---|
|`cda5cbec4b200048eee72ca5ee6a0002b870f38f830b5c91f5c023e6bb66b4a1`|2|系统停止触发根因|`phys_82699244e1b3ff0ccaab85a0`与`phys_3284187224cb454ddd32511f`均为`P2-BASE-ADAPTER-HEAD`、prediction前零预测；`torch.from_numpy(np.ascontiguousarray(stacked))`抛出`TypeError: expected np.ndarray (got numpy.ndarray)`|
|`f73d366fe0360a4b7aba4107b20a8cb266468962b1fc34ae0c91be1d41db274d`|14|停止传播|触发停止时被精确终止的在途predictor；prediction未发布，14个`.out`为空|
|`efe27cdf1686a08aa7cb09a7645ae30303a12994930d40bf2fa1c41013fa4de9`|1|停止传播|`phys_7e548657357f5ce6e9b9dfad`已发布只读prediction，`SCORE_START`后scorer随系统停止中断|

全量只读审计读取了全部1350个terminal status、1425个logical score output和90个已启动日志。1350个predict request、1425个score request和1425个score output均存在；73个成功score completion全部存在并与输出SHA256、logical key和physical ID匹配。74个已发布prediction及row execution receipt全部存在，制品权限均为0444，`query_truth_opened=false`且`fit_query_rows_used=0`。90个日志中76个非空、14个为空；74个包含prediction完成标记，2个包含上述同一根因traceback。全量审计未发现不可读状态、缺失映射或部分覆盖。

停止后主进程PID=`1390511`已退出，run绑定进程为0，NVIDIA compute进程为0；GPU已释放。远端release、input、run、row log和driver根均原样保留。小型关闭证据已拉取到本报告`release_evidence`目录，包括runner summary、2个触发row、1个scorer中断row和1个被终止在途row的status/log。

## 本run关闭与后续

v3固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。73个成功row属于部分技术制品，不得读取、排序或用于方法选择。原run不得resume、覆盖或重启；若继续，需要先在本地修复上述NumPy/Torch边界，完成独立复审与Git提交，再创建新的不可覆盖run ID。
