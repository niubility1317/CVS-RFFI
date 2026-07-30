# CVS-RFFI Phase2-C T1完整消融v5补全报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2c_t1_20260730_v5_77949764`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|只读复用v4已完整闭合的641个physical prediction，补跑剩余709个physical，最终闭合设计报告全部1425个Stage2-C logical row|
|正式代码commit|`779497647f1e616f1a143121635fdc183f3ec0bb`|
|当前状态|`LOCAL_VERIFIED / READY_FOR_N607_HANDOFF`|

## 假设与比较目标

v4的D43/D45失败来自FP32有限精度下类公共中心化改变support argmax；FP64代数等价性未破坏。v5在消融builder的每次fit调用中显式允许已记录的FP32中心化argmax漂移，非消融调用仍保持严格，D45的FP64硬门始终保留。方法、超参数、数据权限、19个Stage2-C arm、评分规则和8×2资源预算不变。比较目标仍为设计报告冻结的完整Stage2-C消融矩阵，不依据中间性能选行或缩小范围。

## v4终态与复用边界

v4固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT / TERMINAL_ARTIFACTS_COMPLETE`，不得resume、覆盖或补写。v4终态为641个`COMPLETE`、18个`FAILED`、691个`NOT_LAUNCHED_SYSTEMIC_STOP`；v5只复用641个`COMPLETE`对应的不可变`row_execution_receipt.json`及prediction，18个失败和691个未启动physical全部进入v5执行队列。

|项目|数量|v5处理|
|---|---:|---|
|v4完整physical|641|`reuse_prediction`，重新执行同一truth-side scorer|
|v4失败physical|18|fresh execute|
|v4未启动physical|691|fresh execute|
|v5实际GPU执行physical|709|8张GPU×每卡2进程|
|完整logical矩阵|1425|必须全部产生same-row评分和completion|

复用控制只读取v4 sealed plan、binding registry、runner summary和641个完成回执，不读取性能值，不重建或重验数据集，不要求跨批次数据一致。Package、feature cache、formal sidecar和Phase2 states继续只读使用既有完整产物。

## 固定矩阵与路径

|字段|值|
|---|---|
|source plan|`stage2c_screening_plan_77949764.json`|
|source plan文件SHA256|`10ec27b86328062152cb2ba4ff66bc82482f38c07b45c45e8d81836272220d41`|
|source plan canonical SHA256|`383b7be16002e00d60346352efd8548265e6baf703e26fd1419a78a3c4155bd0`|
|矩阵|19 arms×75 identities=1425 logical；1350 physical|
|复用/执行|641 reuse＋709 execute|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU并发|8张GPU，每卡2个worker，总计16个并发worker|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2c_t1_20260730_v5_77949764`|
|input/seal|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764`|
|request|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v5_77949764`|
|run|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764`|
|row log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764`|
|driver|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764_driver`|

## 本地修改与验证

|文件|目的|
|---|---|
|`code/cvsrffi/stage2_ablation_executors.py`|仅在Stage2消融builder中显式允许FP32中心化argmax审计漂移|
|`code/scripts/probe_d43_structured_covariance.py`等7个probe|将允许开关按每次fit显式传递，保留默认严格与FP64硬门|
|相关D43/D44/D45/D46/D62/D81/D92和executor测试|覆盖允许路径、默认严格路径、旧monkeypatch兼容及FP64失败门|
|`release_evidence/build_v5_reuse_binding_registry.py`|将v4中仅641个`COMPLETE` physical绑定为`reuse_prediction`，其余709个保持`execute`|

本地`ssr-gpu`验证结果：

- 最终独立复审：`P0=0 / P1=0`。
- 19个相邻发布链测试文件共180项通过；核心独立复跑109项通过；compileall和diff-check通过。
- D43和D45两个历史根因row真实无truth smoke均为`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`，`query_truth_opened=false`、`fit_query_rows_used=0`、stderr=0。
- v5 source plan为1425个logical、19个arm。
- v5复用构建器structure-only检查：641 reuse、709 execute、1350 physical、1425 logical，`performance_values_read=false`、`dataset_revalidated=false`；py_compile通过。

## 同步映射

|本地|N607|
|---|---|
|当前commit的增量Git bundle|`releases/cvs_full_ablation_phase2c_t1_20260730_v5_77949764`|
|`stage2c_screening_plan_77949764.json`|`stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764/source_plan.json`|
|`release_evidence/build_v5_reuse_binding_registry.py`|`stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764/preflight/build_v5_reuse_binding_registry.py`|

## N607启动步骤与精确命令

sole launch owner必须先执行direct read-only preflight，确认普通账号、项目根、8张GPU和现有compute进程。若发现无关任务已占用GPU，只能在不超过每卡总计2个训练进程的前提下排布，不干预无关任务。

在fresh v5 input根中运行复用构建器，读取v4控制面与完成回执并发布新的registry：

```bash
PYTHONPATH="$release/code" /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python "$input/preflight/build_v5_reuse_binding_registry.py" \
  --source-plan "$input/source_plan.json" \
  --base-binding-registry "/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/binding_registry.json" \
  --prior-sealed-plan "/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/sealed_plan.json" \
  --prior-runner-summary "/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/runner_summary.json" \
  --output "$input/binding_registry.json"
```

随后以`CVS-RFFI`环境封存fresh v5计划并启动：

```bash
PYTHONPATH="$release/code" /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python "$release/code/scripts/seal_full_ablation_stage2_plan.py" \
  --plan "$input/source_plan.json" --binding-registry "$input/binding_registry.json" \
  --run-id "cvs_full_ablation_phase2c_t1_20260730_v5_77949764" \
  --request-root "/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v5_77949764" \
  --run-root "/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764" \
  --log-root "/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764" \
  --python-environment-id CVS-RFFI --review-p0-count 0 --review-p1-count 0 \
  --device cuda:0 --shared-view-count 5 --output "$input/sealed_plan.json"

PYTHONPATH="$release/code" nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u \
  "$release/code/scripts/run_full_ablation_stage2.py" \
  --plan "$input/sealed_plan.json" --repo-root "$release" \
  --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  --predictor-script "$release/code/scripts/run_full_ablation_stage2_row.py" \
  --scorer-script "$release/code/scripts/score_full_ablation_stage2_row.py" \
  --execute > "$driver/runner_driver.out" 2>&1 < /dev/null &
```

## 启动健康门和停止规则

启动前必须确认v5的release、input、request、run、row log和driver均为fresh且不覆盖v4；release HEAD必须精确为`779497647f1e616f1a143121635fdc183f3ec0bb`且tracked clean。sealed plan必须报告1425 logical、1350 physical、641 reused、8张GPU、每卡2槽。

启动后立即记录main PID、CWD/cmdline、16个worker、GPU映射、日志增长、首个execute row和首个worker波次。641个reuse row不占用predictor GPU训练槽，但需经过同一truth-side scorer闭合；709个execute row使用8×2队列。只在P0协议/安全违规，或两个不同新execute row在prediction前出现相同确定性异常指纹时停止精确v5进程树；不得按accuracy、H、BA或其他性能值停止。

监控仅在10%、25%、50%、75%、100%里程碑或真实异常时进行短连接检查，不高频轮询。完成标准为1350个physical状态全部闭合、1425个logical same-row score及completion全部存在且可读、失败0、P0=0、runner summary完整，然后进入全量结果分析。

## 风险与完成后检查

主要风险为完成回执复用路径漂移、旧FP32中心化指纹再次出现、单row异常、GPU worker退出或artifact不完整。任何异常必须保留原根并按精确run ID处理，不得覆盖、删除或缩小矩阵。完成后需按candidate/run同一行汇总receiver、K-shot、seed、old/seen-new/unknown、H、coverage/rollback/defer、loss/adapter摘要和最终判定，并与Phase1 44/44及Phase2 States 325/325一起完成设计报告全量消融闭环。

## 2026-07-31正式发布与启动证据

当前状态：`RUNNING`。本节只记录发布、启动与技术健康证据，不读取或解释accuracy、H、BA等性能值。

### Direct preflight与发布面

- `2026-07-31 00:05:31 CST`执行`tools\n607_ssh_preflight.ps1`，普通账号`szu2070436088`、主机`dell-DSS8440`、项目根和8张GPU均可见；GPU0-7均为0%利用率、1MiB显存占用，无compute进程。
- v5的release、input、request、run、row log和driver六类根在创建前均不存在；创建后未覆盖v4或其他任务。
- 远端release精确HEAD为`779497647f1e616f1a143121635fdc183f3ec0bb`，tracked status为空。
- `source_plan.json` SHA256=`10ec27b86328062152cb2ba4ff66bc82482f38c07b45c45e8d81836272220d41`。
- `build_v5_reuse_binding_registry.py` SHA256=`f1f0adfa71919e34438186fa3a14c679d228207397c90c38287e26ca098fc2a3`。
- v4过渡bundle SHA256=`c1feff733d44f896c0c7599158a7043db30d385f5bd4f252111f4bf8b3cf35b6`；v5增量bundle SHA256=`6318f8d1761cc2ad0ab2072696c0cd789805e7a4c1b5683894c46e483d225338`。
- 本次修改涉及的执行器、D43/D44/D45/D46/D62/D81/D92探针和复用registry构建器均通过远端`CVS-RFFI`环境`py_compile`。

### 复用构建、无真值冒烟与seal

- structure-only与正式构建结果一致：1425 logical、1350 physical、641 reuse、709 execute；`performance_values_read=false`、`dataset_revalidated=false`。
- `binding_registry.json` SHA256=`1efc663de9cf23e4c6f0aa651c20c85df4aa31ac28e8845a5048ca2aad3e25d0`；复用物理ID集合SHA256=`343322ae49468740ec02ca028d67676550c2abe6fb49db13d438f3309cf54bcd`。
- 三个历史根因row均使用本地生成且只修改`output_root`的请求，在独立preflight输出根运行；未调用truth-side scorer。

|row|历史根因覆盖|请求SHA256|预测行数|状态|真值门|
|---|---|---|---:|---|---|
|`phys_1b9d0cee16897a454ddb3aa7`|D43相关路径|`9967c9c0d8360a72e11294632664c6fbf93d4186687b0096d111a0de0311f973`|1560|`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`|`query_truth_opened=false`，`fit_query_rows_used=0`|
|`phys_af88df635bf6b18beb105d08`|D43相关路径|`23f1b51ead2e37f7c04c0ab3c9c99eea82f96aa91f9578065d709465c502e7c2`|1560|`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`|`query_truth_opened=false`，`fit_query_rows_used=0`|
|`phys_37cc012d2b44700e361a5a9c`|D45相关路径|`26daf3ccc522109bee8c8567ea251584b58bdc3547a17991e58eca6f80116c6a`|1560|`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`|`query_truth_opened=false`，`fit_query_rows_used=0`|

- sealed plan：`formal_launch_authority=true`、P0=0、P1=0、8张GPU、每卡2槽、1425 logical、1350 physical、641 reuse、709 execute、75 alias。
- `sealed_plan.json` SHA256=`6029b64f2f9fbd2d6ac24ff71de8c90f95028b86491b55171a32b41bd1cad15b`。
- seal生成709个predict请求和1425个score请求；请求树SHA256=`e31c4c798d98f92bbdfd1047cece85f930ef15e83988edcb5357306e487ef7fc`。
- 709个execute物理行已完整分配到GPU0-7×slot0-1的16个槽位，每个槽均非空。

### Detached launch与即时健康

- 使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`启动；main PID=`1935906`。
- main cmdline精确绑定本v5的`sealed_plan.json`、release、predictor和scorer；CWD为`/home/szu2070436088`，所有正式输入、输出和日志路径均为绝对v5路径。
- 启动后main存活，直接子进程数为16。即时波次先并发闭合641个复用行的truth-side scorer；随后事件型短连接确认16个直接子进程已全部切换为`run_full_ablation_stage2_row.py` predictor，GPU0-7各有2个compute进程、每进程约338MiB，8×2首个execute GPU波次成立。
- driver日志路径为`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v5_77949764_driver/runner_driver.out`；PID文件为同目录`launcher.pid`。
- 每次SSH/SCP短连接结束后，本地`ssh.exe=0`且指向N607/bridge的`ESTABLISHED TCP22=0`。
- 首个execute GPU波次快照中状态面有640个`COMPLETE`，没有非`COMPLETE`状态，规范化异常指纹集合为空，未见P0；该计数仅为并发切换瞬时技术健康快照，不用于性能或最终覆盖结论。
- 下一次只在10%里程碑或真实技术异常时短连接检查；继续沿用P0或两个不同新execute row在prediction前出现相同确定性异常指纹的唯一技术停止规则。

## 2026-07-31 10%技术里程碑

`2026-07-31 02:18:55 CST`执行正式低频只读检查；本节只统计执行健康与artifact闭合，不读取性能值，不执行数据审计或哈希对齐。

|检查项|结果|
|---|---|
|main/worker|main PID=`1935906`存活；16个直接子进程，全部为predictor，scorer子进程0|
|physical状态|`COMPLETE=794`、`FAILED=0`、运行中16、排队540，合计1350|
|复用闭合|641/641个`reuse_prediction`均为`COMPLETE`|
|fresh execute进度|153/709=`21.58%`，已跨过10%门槛；execute失败0|
|prediction闭合|153个`predictions.cvspred`、153个`row_execution_receipt.json`|
|score闭合|794个score、794个completion，与794个完成physical对应|
|GPU资源|GPU0-7各2个compute进程；瞬时利用率均0%，每卡691MiB，每进程约338MiB|
|异常控制|P0=0；非`COMPLETE`状态0；规范化异常指纹集合为空|
|连接清理|direct preflight通过；短SSH结束后本地`ssh.exe=0`、N607/bridge`ESTABLISHED TCP22=0`|

判定：v5已健康跨过10%里程碑，继续原矩阵运行，不修改、不重启、不停止。下一次仅在25%里程碑或真实技术异常时检查。

## 2026-07-31 25%技术里程碑

`2026-07-31 02:42:58 CST`执行正式低频只读检查；本节只统计执行健康与artifact闭合，不读取性能值，不执行数据审计或任何任务修改。

|检查项|结果|
|---|---|
|main/worker|main PID=`1935906`存活；16个直接子进程，全部为predictor，scorer子进程0|
|physical状态|`COMPLETE=827`、`FAILED=0`、运行中16、排队507，合计1350|
|复用闭合|641/641个`reuse_prediction`均为`COMPLETE`|
|fresh execute进度|186/709=`26.23%`，已跨过25%门槛；execute失败0|
|prediction闭合|186个`predictions.cvspred`、186个`row_execution_receipt.json`|
|score闭合|827个score、827个completion，与827个完成physical对应|
|GPU资源|GPU0-7各2个compute进程；瞬时利用率均0%，每卡691MiB，每进程约338MiB|
|异常控制|P0=0；非`COMPLETE`状态0；规范化异常指纹集合为空|
|连接清理|direct preflight通过；短SSH结束后本地`ssh.exe=0`、N607/bridge`ESTABLISHED TCP22=0`|

判定：v5已健康跨过25%里程碑，继续原矩阵运行，不修改、不重启、不停止。下一次仅在50%里程碑或真实技术异常时检查。

## 2026-07-31 50%技术里程碑

`2026-07-31 04:45:55 CST`执行正式低频只读检查；本节只统计执行健康与artifact闭合，不读取性能值，不执行数据审计或任何任务修改。

|检查项|结果|
|---|---|
|main/worker|main PID=`1935906`存活；16个直接子进程，全部为predictor，scorer子进程0|
|physical状态|`COMPLETE=1000`、`FAILED=0`、运行中16、排队334，合计1350|
|复用闭合|641/641个`reuse_prediction`均为`COMPLETE`|
|fresh execute进度|359/709=`50.63%`，已跨过50%门槛；execute失败0|
|prediction闭合|359个`predictions.cvspred`、359个`row_execution_receipt.json`|
|score闭合|1000个score、1000个completion，与1000个完成physical对应|
|GPU资源|GPU0-7各2个compute进程；瞬时利用率均0%，每卡691MiB，每进程约338MiB|
|异常控制|P0=0；非`COMPLETE`状态0；规范化异常指纹集合为空|
|连接清理|direct preflight通过；短SSH结束后本地`ssh.exe=0`、N607/bridge`ESTABLISHED TCP22=0`|

判定：v5已健康跨过50%里程碑，继续原矩阵运行，不修改、不重启、不停止。下一次仅在75%里程碑或真实技术异常时检查。
