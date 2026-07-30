# CVS-RFFI Phase2-C T1完整消融v4运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|修复v3的NumPy 2.x/PyTorch 2.1转换故障后，完整执行Stage2-C全部1425个logical row，不按中间性能缩小范围|
|正式代码commit|`1ca64a586b85c97fbaa2a677a6ca5776ffd239b3`|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT / TERMINAL_ARTIFACTS_COMPLETE`|

## 假设与比较目标

v3的停止根因仅位于`P2-BASE-ADAPTER-HEAD`训练适配器的NumPy→Torch输入桥。v4以`torch.frombuffer(...).reshape(...).clone()`替代`torch.from_numpy`，并以`detach().cpu().tolist()`替代`Tensor.numpy()`输出桥；方法、超参数、数据权限、1425行矩阵、GPU预算和评分规则保持不变。比较目标仍为设计报告冻结的19个Stage2-C消融arm。

## v3关闭与复用边界

v3固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得resume、覆盖或补写。v4只读复用已经完整闭合的输入，不复用v3的run、request、row log、driver或部分预测/评分：

|复用输入|只读路径|闭合证据|
|---|---|---|
|Package|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e`|45/45；summary SHA256=`ed61d7c8403c203f4b1c73d4dd87251a67cf01b7da7f6f580143dfa852b026cb`|
|Feature|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437`|75/75 identities；225/225 scope cache；summary SHA256=`423ea50e5f74a1dcc055f605d093d699865f371b92dd0ee7960a31993c94a91d`|
|Formal sidecar|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437`|30/30；60/60文件；summary SHA256=`7d7de4a6dbb22caa5ec4cde594cdf66dc678423c8639882af99cbfce88bb5c45`|
|Phase2 states|`cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`|325/325；失败0|

v4仅重新生成与新commit和新run ID绑定的75-entry cache index、1425-entry registry和sealed plan。该操作不重建或重验数据集。

## 固定矩阵与服务器路径

|字段|值|
|---|---|
|source plan|`stage2c_screening_plan_1ca64a58.json`|
|source plan SHA256|`d8c420c22a0a775bc7d6c79af513d79adf78d29a25713e03591a4a79e9136fd7`|
|矩阵|19 arms×75 identities=1425 logical；预计1350 physical|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU并发|8张GPU，每卡总compute进程最多2个|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|input/seal|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|request|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|run|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|row log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|driver|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58_driver`|

## 本地修改与验证

|文件|目的|
|---|---|
|`code/cvsrffi/stage2_trainable_lowrank_support_adapter.py`|移除该模块全部`torch.from_numpy`与`Tensor.numpy`桥，保留float32/int64、shape、device和clone隔离|
|`tests/test_stage2_trainable_lowrank_support_adapter.py`|模拟两个旧桥均抛出N607兼容异常并执行真实`fit_locked`|

`ssr-gpu`环境验证：focused 11项通过；相邻Stage2-C执行、工厂、row、input-release、scorer和sidecar链路96项通过；compileall通过。独立复审`P0=0 / P1=0`，允许commit和fresh v4 release。

## 启动和健康门

`reuse_seal_template.txt`先验证三类复用输入的闭合summary，再在fresh v4 input根生成index、registry和sealed plan。`launch_template.txt`验证release HEAD/clean、Phase2 states、source/sealed plan和三类输入后，才创建fresh driver并启动完整矩阵。

启动后立即记录main PID、CWD/cmdline、16个worker、GPU0–7每卡2个槽、日志增长、首个row和首个worker波次。只在P0协议/安全违规，或两个不同row在prediction前出现同一确定性异常指纹时停止v4精确进程树；不得按性能停止、选行或缩小矩阵。完成后必须闭合1350个physical状态、1425个logical score output、全部prediction/score/log和runner summary，再进入结果分析。

## N607落地、根因smoke与fresh封存

direct preflight通过：普通N607账号、项目根和8张GPU均可见，启动前GPU compute进程为0。v3主进程和run绑定进程均为0，v3全部根保持原样；v4的release、input、request、run、rowlog和driver六个目标根在落地前均不存在。

v4 release通过增量bundle落地，bundle大小122596 bytes，SHA256=`c1feff733d44f896c0c7599158a7043db30d385f5bd4f252111f4bf8b3cf35b6`。远端release精确HEAD=`1ca64a586b85c97fbaa2a677a6ca5776ffd239b3`且clean；source plan、bundle及6个preflight文件逐项同步，远端py_compile、两个bash语法检查和source plan identity verifier均通过。

正式封存前，以v3触发根因row`phys_82699244e1b3ff0ccaab85a0`执行单个fresh predictor smoke。smoke仅复用既有合法request输入，将request、prediction、receipt和日志写入v4 input的`preflight/smoke`，没有启动scorer。PID=`1465459`自然退出；日志末行与保存receipt一致，而row入口在打印该receipt后固定返回0。receipt状态为`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`，`query_truth_opened=false`、`fit_query_rows_used=0`，prediction存在且只读；日志中没有`torch.from_numpy`、`Tensor.numpy`、traceback或`SCORE_START`。

|fresh封存制品|计数|SHA256|
|---|---:|---|
|cache binding index|75 entries|`85fe57e8b6234a660e687165ac900439a2a0089a276fda20ec8e78fe1a7b190b`|
|binding registry|1425 entries|`a413e160bcc9bfa0a6c40864b5f716c0dc5805192a3ff3ba73d708dbd2c430a4`|
|sealed plan|1425 logical；1350 physical；75 aliases|`cf1f98e1a17c8df52ee94c3f17b28df5e725be8cbee96cdfd5f97959dbc258cd`|

reuse封存再次只读验收package 45/45、feature 75/75/225 scope cache和formal sidecar 30/30/60文件，未写旧输入根。sealed plan固定8张GPU、每卡2槽、formal launch authority=true、P0=0、P1=0。正式launch又复验states 325/325、三类summary、source/sealed plan及fresh run/log/driver门后启动完整矩阵。

|正式启动证据|值|
|---|---|
|main PID|`1469792`|
|CWD|v4 release根|
|cmdline绑定|v4 sealed plan、正式row predictor和truth-side scorer|
|并发|16个直属worker；GPU0–7各2个|
|首次完成row|3/1350 physical；prediction=3；score=3；logical score=3|
|健康|failed=0；P0=0；异常指纹0；runner继续完整矩阵|

首个完整worker波次快照为21/1350 physical，prediction=21、score=21、logical score=21、failed=0、unreadable=0、P0=0、异常指纹0；main仍存活，16个worker继续保持GPU0–7各2个槽。

监控均使用带显式短超时的独立SSH连接，连接完成后立即退出。当前只报告执行健康，不读取或解释性能。

## 运行里程碑

|时间|physical状态|prediction/score|logical闭环|异常检查|进程与GPU|结论|
|---|---:|---:|---:|---|---|---|
|2026-07-30 16:40:23|137/1350 COMPLETE；FAILED=0；剩余1213|137/137|137；expected=137|unreadable=0；field mismatch=0；P0=0；fingerprints=0；旧`cda5...b4a1`=0|main PID存活；16 workers；GPU0–7各2个worker和2个compute进程|跨过135/1350首个10%里程碑；运行健康，继续完整矩阵|
|2026-07-30 18:45:23|340/1350 COMPLETE；FAILED=0；剩余1010|340/340|340；expected=340|unreadable=0；field mismatch=0；P0=0；fingerprints=0；旧`cda5...b4a1`=0|main PID存活；16 workers；GPU0–7各2个worker和2个compute进程|跨过338/1350的25%里程碑；运行健康，继续完整矩阵|

里程碑监控不读取性能值。`runner_summary.json`尚未生成，当前状态仍为`RUNNING`。

## 孤立技术失败

2026-07-30 21:18:17的低频检查发现首个prediction前失败。22:21:02发现第二个prediction前失败，但它产生不同异常指纹；截至22:21:43，两个指纹分别只出现在1个physical row，仍未达到“两个不同row在prediction前出现同一确定性异常指纹”的系统性停止门槛。runner继续分发其余冻结矩阵，不覆盖、不重启失败行。

|字段|值|
|---|---|
|physical execution|`phys_1b9d0cee16897a454ddb3aa7`|
|logical row|`P2-BASE-FULL-BLOCK-LDA__rx_7_14__k_10__new_20__support_7282201__query_7282301__draw_7282401`|
|GPU/slot|GPU0/slot0|
|launch PID|`1804941`；失败后已退出|
|launch绑定|CWD为v4 release；cmdline绑定正式row入口和v4 request；output root绑定v4 physical根|
|异常|`D43ProbeError: D43 FP32 centering changed support argmax`|
|指纹|`3c5905f17ef1f213abb92e9b0d49e355619d280abdaf690a510cf89d9bd9a759`；计数1|
|产物状态|`prediction_complete=false`；`scores_complete=false`；`zero_prediction=true`；`p0_protocol_violation=false`|
|证据|完整读取2386-byte row日志、terminal status和launch artifact；全部原样保留|
|处置|不干预健康运行的其余矩阵；完成后以新的不可覆盖补跑run补齐该孤立行，不覆盖v4、不调参、不改变方法|

第二个孤立失败：

|字段|值|
|---|---|
|physical execution|`phys_37cc012d2b44700e361a5a9c`|
|logical row|`P2-BASE-FULL-BLOCK-LDA__rx_8_8__k_10__new_20__support_7282203__query_7282303__draw_7282401`|
|GPU/slot|GPU0/slot0|
|launch PID|`1836233`；失败后已退出|
|launch绑定|CWD为v4 release；cmdline绑定正式row入口和v4 request；output root绑定v4 physical根|
|异常|`D45ProbeError: D45 locked D42 full-component centering drift`|
|指纹|`d51f60a1e166b42422a19157fca87909115ffeea24979982212d2694bde5a8dd`；计数1|
|产物状态|`prediction_complete=false`；`scores_complete=false`；`zero_prediction=true`；`p0_protocol_violation=false`|
|证据|完整读取2401-byte row日志、terminal status和launch artifact；全部原样保留|
|处置|不干预健康运行的其余矩阵；完成后以新的不可覆盖补跑run补齐该孤立行，不覆盖v4、不调参、不改变方法|

## 系统性停止与终态闭合

v4随后在第二个不同row复现指纹`3c5905f17ef1f213abb92e9b0d49e355619d280abdaf690a510cf89d9bd9a759`。runner按预注册规则停止继续分发并退出，未按性能值作出停止决定。终态固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得将641个已完成row用于v4性能结论，也不得resume或覆盖v4。

|physical终态|计数|说明|
|---|---:|---|
|COMPLETE|641|prediction、score、logical闭环字段一致|
|FAILED|18|3类指纹；全部prediction前失败、zero prediction、P0=false|
|NOT_LAUNCHED_SYSTEMIC_STOP|691|系统性停止后未分发；映射766个logical row|
|合计|1350|status JSON全量可读；unreadable=0|

|FAILED指纹|physical数|错误/角色|判定|
|---|---:|---|---|
|`3c5905f...b66b4a1`|2|`D43ProbeError: D43 FP32 centering changed support argmax`|两个不同row复现；系统性停止触发指纹|
|`d51f60a...de5a8dd`|1|`D45ProbeError: D45 locked D42 full-component centering drift`|独立孤立失败|
|`f73d366f...1db274d`|15|停止收尾时其余活跃worker返回非零；15个row日志均为0-byte|系统性停止后的清退后果；原样保留|

终态全量只读审计：

- 1350个status JSON全部解析；P0=0、COMPLETE字段不一致=0。
- 1425个logical输出全部存在并解析：641个`same_row_score/PASS`，784个`failed_row/FAILED`；后者由18个真实FAILED logical和766个NOT_LAUNCHED logical构成，全部`zero_prediction=true`。
- 641个完成physical的prediction文件均存在且非空，641个row receipt均可解析，641份完整row日志均可读且非空；641个logical completion receipt均可解析。
- 641个PASS logical输出均记录truth在prediction commit后才打开；scheduler记录`performance_values_visible_to_scheduler=false`。
- `runner_summary.json`与上述计数一致：`physical_execution_count=1350`、`logical_row_count=1425`、`completed_physical_count=641`、`completed_logical_score_count=641`、`failed_physical_count=18`、`not_launched_systemic_stop_count=691`、`systemic_stop=true`、thread error为空。
- main PID、16个worker及全部v4绑定进程均已退出；GPU0–7 compute进程均为0。本地`ssh.exe=0`，N607 TCP22 established连接=0。远端全部run、request、log、driver和partial artifact保持原样。

回收的小型终态证据位于`terminal_evidence/`：

|文件|字节|SHA256|
|---|---:|---|
|`runner_summary.json`|788508|`0f2dc74c7809bc5093b231442834f0f0c3920890957ae36be059760bb1b8308e`|
|`phys_1b9d0cee16897a454ddb3aa7.out`|2386|`cea8e5b9b4770d5b6c8857988e077529b1b1ac208027644be4a43e46346b15a5`|
|`phys_af88df635bf6b18beb105d08.out`|3100|`70874c0fb021512a57c9364121f27bed5d93e333451d2b9a1255d8f1af05f1bf`|
|`phys_37cc012d2b44700e361a5a9c.out`|2401|`eff98e221090939376545d9a9d8d146c2eede45bb9d9bb0a0f605bf2236a624c`|

下一步必须先在本地修复并覆盖三类终态技术失败，完成针对性测试、真实输入smoke、独立复审和新commit，再创建不可覆盖的v5补跑计划。v5至少补齐18个FAILED physical和691个NOT_LAUNCHED physical，并按原registry恢复全部784个未成功logical闭环；是否能够安全复用v4的641个已完成row，须由v5冻结计划显式绑定和验证，不能在v4内补写。

## 根因行no-truth输入回收

为支持本地修复后的prediction-only smoke，已从v4保留根只读回收3个真实根因row的完整request、对应feature manifest sidecar和feature NPZ，共9个文件，保存于`release_evidence/root_cause_no_truth_inputs/`。3份manifest均声明`query_truth_present=false`和`query_role_present=false`；NPZ不含query label/truth/role数组。未复制数据集、query truth、scoring sidecar或其他大文件，未重验数据、未修改远端。

逐文件原始N607路径、字节数、完整文件SHA256、manifest canonical绑定SHA256、本地derived request规则和no-truth smoke依赖清单见该目录的`README.md`。

本地`ssr-gpu`只读校验3/3通过：request row绑定、payload SHA、manifest canonical SHA均一致；每份NPZ的21个数组均不含query label/truth/role数组。本证据补充未执行smoke。

## v5复用控制文件回收

为支持新v5只补跑709个physical并显式复用641个v4 COMPLETE，已只读回收v4的`sealed_plan.json`、`binding_registry.json`和`cache_binding_index.json`至`release_evidence/v5_reuse_control/`。未回收641个prediction大文件，未重验数据、未修改N607。

本地`ssr-gpu`集合核验通过：sealed plan的1350个physical与runner summary的1350个status ID完全一致；其中641个COMPLETE、709个待v5补跑。sealed plan的1425个logical与registry的1425个binding完全一致，cache index为75项，三文件candidate lock一致。N607上641个COMPLETE对应的`row_execution_receipt.json`全部存在且可解析，missing=0、unreadable=0、binding mismatch=0。逐文件远端路径、大小、SHA和v5复用边界见该目录`README.md`。
