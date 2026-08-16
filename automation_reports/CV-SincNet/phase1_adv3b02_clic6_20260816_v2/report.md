# Phase1 ADV3B02 CLIC六折v2技术烟测与正式入口预注册报告

## 1.状态、范围与非性能边界

- 实验ID：`phase1_adv3b02_clic6_20260816_v2`。
- 当前状态（最终技术封存）：`ARTIFACTS_COMPLETE / NON_PROMOTABLE_P0_DISABLED / NO_PERFORMANCE_RESULT`。第10—13节保留此前release、smoke与reconcile历史；本节记录最终既有formal自然完成证据。
- v1状态：`phase1_adv3b02_clic6_20260813_v1`永久保持`SMOKE_STOPPED_TECHNICAL_GATE / FORMAL_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`；formal invocation=`0`，retry=`NO`，不得恢复、重试、覆盖或改写v1。
- 本v2只修正技术烟测对训练器既有可恢复梯度跳过的观测合同；不改ADV3B02方法、loss、fold、seed、epoch、数据角色、source-only边界或target权限。
- 本报告和任何v2 receipt均为技术证据，不记录、读取或解释accuracy、DG、proxy或其他性能值。

## 2.已确认根因与证据边界

远端只读日志为`/home/szu2070436088/2510044040/CV-SincNet/logs/.smoke_phase1_adv3b02_clic6_20260813_v1_F1/F1_ADV3B02_CLIC.out`，大小10614B。其终端计数为：`batches=3`、`forward_batches=3`、`backward_batches=3`、`optimizer_attempts=2`、`optimizer_effective_steps=2`、`optimizer_nonfinite_batches=1`。

本地与发布版本同路径控制流表明：loss有限时，训练器执行`scaler.scale(loss).backward()`、`scaler.unscale_(optimizer)`；若`_grads_are_finite(model)`为false，则执行`optimizer.zero_grad(set_to_none=True)`与`scaler.update()`，设置`skipped_nonfinite_grad=1`并继续下一raw batch。只有`grads_finite=true`才递增`optimizer_attempts`并执行`scaler.step(optimizer)`。因v1已有3次backward，故该记录可确定为unscale后梯度非有限，而不是nonfinite loss；现有v1日志没有raw batch index、参数/term来源或AMP scale，不能把它声称为已定位的具体数值项或纯AMP overflow。

根因是v1烟测把固定3个raw batch错误地当作3个有效optimizer step；这与普通formal训练的可恢复梯度跳过语义不一致。v2不放宽方法数值要求：只预注册一个最多4个连续raw batch的技术窗口，用以验证0或1次既有恢复skip后能取得恰3个有效step。

## 3.v2冻结技术合同

|字段|冻结值/规则|
|---|---|
|方法身份|`ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL`，与v1逐字段同方法profile。|
|run ID|`phase1_adv3b02_clic6_20260816_v2`。|
|formal路径|`runs/phase1_adv3b02_clic6_20260816_v2`与`logs/phase1_adv3b02_clic6_20260816_v2`；必须全新且不可覆盖。|
|smoke路径|`runs/.smoke_phase1_adv3b02_clic6_20260816_v2_F1/F1_ADV3B02_CLIC`与对应logs根；必须全新且不可覆盖。|
|目标|恰3个有限、有效的forward/backward/optimizer step。|
|raw上限|4个连续raw batch；达到3个有效step立即退出；不足3个有效step时cap耗尽即拒绝。|
|可恢复skip|只允许0或1次`skipped_nonfinite_grad=1`，且必须`skipped_nonfinite_loss=0`。第二次grad skip立即拒绝。任何loss非有限立即拒绝。|
|技术序列|每个raw batch记录连续index、loss/grad有限标志、optimizer attempted/effective标志和AMP scale before/after；不得记录性能字段。|
|访问边界|成功、cap耗尽、第二次grad skip或loss非有限均必须在source-V、target、query、test evaluation或selection前退出。|
|formal门|v2正式六fold入口只在独立v2技术receipt完整且匹配时允许调用；否则formal invocation保持0。|
|重试|v2 smoke invocation最多1次；v2 formal invocation最多1次；任何失败均为`NO_PERFORMANCE_RESULT`，不得覆盖。|

## 4.设计—实现可追溯表

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|V2-01|v1远端技术日志与训练器控制流|保留v1 flag/schema/行为；新增独立v2 control和schema。|`code/SSDG/train_ssdg.py`|passed|focused pytest与py_compile|v1 finalizer/schema/旧flag行为保持；v2为独立控制。|
|V2-02|根因合同|cap=4；目标3 effective；允许0/1 grad skip；loss非有限/第二skip/cap耗尽拒绝。|`code/SSDG/train_ssdg.py`、`code/tests/test_phase1_adv3b02_clic6_baseline.py`|passed|真实训练循环RED/GREEN|一次post-backward非有限grad后四raw batch取得三effective；双skip与loss非有限负测拒绝。|
|V2-03|访问与连续性合同|逐raw记录index、finite、optimizer、AMP scale；所有退出在source-V/target/query/test前。|`code/SSDG/train_ssdg.py`、测试|passed|真实循环receipt/拒绝断言|成功receipt断言四条连续record；所有三条控制流测试均断言source-V未打开。|
|V2-04|新run隔离|v2 smoke/formal路径新建、不可覆盖；formal须验证v2 receipt。|新增v2 formal/smoke launcher、测试|passed|`bash -n`、dry-run、root collision测试|正式六fold机械复用v1 profile，仅替换v2身份/路径；receipt或新root不满足即拒绝。|
|V2-05|完整profile/env守卫|v2从formal v2 dry-run恢复F1，并保持BASH_ENV/PATH最小环境防伪造。|`code/SSDG/train_ssdg.py`、测试|passed|profile/env负测|直接BASH_ENV伪造可见，但训练器以最小环境复原formal F1并拒绝漂移。|
|V2-06|正式路径不变|两个v2 control均为0时，普通formal parser/runtime路径不受限。|`code/SSDG/train_ssdg.py`、测试|passed|focused pytest|v1 flag=0与v2 flag=0均不触发v2合同。|
|V2-07|本地发布准备|记录diff、验证、哈希和N607路径；本轮不commit、不sync、不launch。|本报告|passed|最终diff/status待主控集成审阅|本子任务未commit、未sync、未启动N607。|

## 5.本地RED/GREEN与静态验证证据

- RED：新增v2测试先运行`python -m pytest code/tests/test_phase1_adv3b02_clic6_baseline.py -q`，得到8项预期失败，原因仅为v2 formal/smoke入口、parser flag和v2 receipt/schema尚未实现；旧v1测试未出现新失败。
- GREEN：在`ssr-gpu`环境串行运行`python -m py_compile code/SSDG/train_ssdg.py code/tests/test_phase1_adv3b02_clic6_baseline.py`，exit=`0`；再运行`python -m pytest code/tests/test_phase1_adv3b02_clic6_baseline.py -q`，exit=`0`，`37 passed`，无失败。仅有5项既有PyTorch AMP API FutureWarning，不改变训练语义。
- 静态：`git diff --check`、两个v2 launcher的`bash -n`、v2 formal六行dry-run和v2 F1 smoke dry-run均通过。dry-run核对显示F1--F6的fold/seed/epoch/method/loss参数与v1一致；v2 smoke仅追加`--phase1_adv3b02_technical_smoke_v2_max_batches 4`并使用独立v2 smoke root。
- 本地真实控制流：注入一次post-backward非有限梯度时，receipt记录raw index=`1..4`、loss均有限、首条grad非有限、其余三条有限、3次optimizer attempted/effective、AMP scale before/after有限；没有source-V、target、query或test evaluation。注入第二次grad skip或任意loss非有限时，均在source-V前拒绝且不写成功receipt。

本轮本地SHA256：

|文件|SHA256|
|---|---|
|`code/SSDG/train_ssdg.py`|`5D4DF42F9A9C2B6AA1D862D4D2C41F55E28B4C643A48C8F146068C0079930F9F`|
|`code/tests/test_phase1_adv3b02_clic6_baseline.py`|`571FA104216F1EE2C635CF02DB6A6A54A3FBEFC8C09D3BDAC856FF0BC60F652C`|
|`code/scripts/launch_phase1_adv3b02_clic6_v2_20260816.sh`|`40960CFBB436968AB1F9BB740EF88AC6844F538DFFC468C1FD8EA47F87AC6AD7`|
|`code/scripts/smoke_phase1_adv3b02_clic_f1_v2_20260816.sh`|`14DEAC6E1ADD2B5FA9F25A90C84FE2DD45CC15A7FADA4FD3A1D13727055BF7EF`|

## 6.独立P1复审闭环

- 复审发现：v2 formal receipt gate先前没有绑定`base_candidate`、`fold`及`source_val_rows_opened`、`query_rows_opened`、`target_rows_opened`、`test_rows_opened`、`selection_feedback_count`。隔离伪造receipt把方法改成`FORGED_METHOD`并将五个访问计数改成`1`后，旧gate会放行至后续dataset/root检查，故不能证明formal invocation保持在技术门之后。
- 最小修复：仅在`code/scripts/launch_phase1_adv3b02_clic6_v2_20260816.sh`的formal receipt validator中增加`base_candidate=ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL`、`fold=F1`，并要求五个访问计数均为严格`int`类型的`0`（`bool`不作为`int`接受）。不改变方法、loss、fold矩阵、seed、epoch、run身份或smoke控制。
- P1 RED：新增真实launcher负测后，旧实现报`missing WiSig dataset`而不是receipt-gate错误，证实伪造receipt越过formal gate。
- P1 GREEN：修复后，review攻击、`fold=F2`、`False`伪装访问计数、缺失访问字段均在任何run/log root或dataset检查前被`requires a complete v2 technical smoke receipt`拒绝。`ssr-gpu`下`py_compile` exit=`0`，ADV focused pytest exit=`0`，`38 passed`；只有5项既有PyTorch AMP API FutureWarning。两个v2 launcher的`bash -n`、formal/smoke dry-run与`git diff --check`均通过。

## 7.本轮禁止事项

- 不启动N607，不同步，不提交，不执行v1或v2 formal。
- 不改变source角色、fold、seed、epoch、loss、checkpoint选择或任何target接口。
- 不将一个处理过的nonfinite skip视为性能、候选选择或方法通过证据。

## 8.N607 runner执行记录（2026-08-16）

- runner：Luna/max；唯一run owner；本节只记录机械静态门证据，不读取或解释性能，不作方法、参数或晋级判断。
- 合同commit：`2e0b0b8990c013659c362b346f20d47d53e48ac7`。既有archive：268175360 bytes，SHA256=`C1A876278424B21DA45D47550F4E01F5BEF0849163FAC8B6316C2B61C6DEBFFE`；按brief不重建、不解包、不重复SCP。
- direct N607 preflight：`VERIFIED`。server=`dell-DSS8440`，user=`szu2070436088`，project可见；GPU0--7均24,576 MiB、utilization=0、memory.used=1 MiB；独立`/proc`扫描无v2目标进程。upload bytes/SHA与local archive闭合；staging存在，final不存在；formal run/log、smoke run/log/outer均不存在。
- archive member safety：5103 members，absolute/traversal=0，link members=0。staging相关入口、输入ManySig、fresh roots均通过；远端Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；trainer/test `py_compile=PASS`，两份v2 `bash -n=PASS`，formal dry-run=6，smoke dry-run=1且raw cap=4，冻结关键flags=PASS，禁止target/query/truth/package/scorer输入=NONE。
- 历史初始静态停止（按brief原始hash注册）：brief要求trainer canonical SHA=`5D4DF42F9A9C2B6AA1D862D4D2C41F55E28B4C643A48C8F146068C0079930F9F`；archive/staging trainer physical SHA=`1E9EA659E77466BDBB4F94944671FA691418F17188379EFFD6E6E187C59B2068`，LF-normalized SHA=`2E5A6C6AA72CBA049D5D03023F29542F2A21F86534979F1A6786C19466171279`。该历史停止证据保留，不表示release通过。
- 历史停止时终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。未将staging原子rename为final；SMOKE invocation=`0`、FORMAL invocation=`0`、retry=`NO`；无smoke receipt、checkpoint、run artifact或formal log；release final artifact=0、formal logs=0、smoke logs=0。无run-owned PID可停止；GPU保持空闲；本地SSH/SCP进程与TCP22均清理为`VERIFIED`。
- 首个技术fingerprint：`STATIC_HASH_MISMATCH trainer canonical_sha256 expected=5D4DF42F9A9C2B6AA1D862D4D2C41F55E28B4C643A48C8F146068C0079930F9F actual_archive_stage_raw=1E9EA659E77466BDBB4F94944671FA691418F17188379EFFD6E6E187C59B2068 actual_archive_stage_lf=2E5A6C6AA72CBA049D5D03023F29542F2A21F86534979F1A6786C19466171279`。
- worktree中其他agent的source-metrics staged/unstaged改动及未跟踪`conversation_index/`未触碰、未stage、未revert。

## 9.证据更正（2026-08-16）

- 主控独立复核确认Git/archive canonical冻结SHA为`2E5A6C6AA72CBA049D5D03023F29542F2A21F86534979F1A6786C19466171279`，与`git show 2e0b0b89:code/SSDG/train_ssdg.py`及archive/staging的LF-normalized内容一致。
- trainer physical SHA分别为：当前Windows worktree=`5D4DF42F9A9C2B6AA1D862D4D2C41F55E28B4C643A48C8F146068C0079930F9F`；local archive member raw=`1E9EA659E77466BDBB4F94944671FA691418F17188379EFFD6E6E187C59B2068`；remote staging raw=`1E9EA659E77466BDBB4F94944671FA691418F17188379EFFD6E6E187C59B2068`；archive/staging LF-normalized=`2E5A6C6AA72CBA049D5D03023F29542F2A21F86534979F1A6786C19466171279`。
- 更正结论：`5D4DF...`是当前Windows worktree physical hash，不是Git/archive canonical hash；原始静态停止保留为hash registration defect的历史记录。该更正不构成release通过、final落地、smoke通过或formal授权。
- 当前状态：`STATIC_HASH_REGISTRATION_DEFECT_CONFIRMED / REMOTE_RESUME_NOT_YET_AUTHORIZED / SMOKE0 / FORMAL0`；不得恢复远端流程，除非主控另行完成并授权新的registration闭环。

## 10.恢复静态门（2026-08-16）

- 独立hash review裁定`P0=0/P1=0/ALLOW_RESUME`后，按更新brief重新完成即时reconcile与STATIC_ONLY；不重复SCP、不重建archive。
- canonical trainer LF/Git SHA=`2E5A6C6AA72CBA049D5D03023F29542F2A21F86534979F1A6786C19466171279`；archive/staging trainer physical SHA=`1E9EA659E77466BDBB4F94944671FA691418F17188379EFFD6E6E187C59B2068`；Windows worktree physical SHA=`5D4DF42F9A9C2B6AA1D862D4D2C41F55E28B4C643A48C8F146068C0079930F9F`仅作物理记录。
- 即时reconcile=`VERIFIED`：staging存在、final/formal run+log/smoke run+log/outer均不存在、目标进程NONE、GPU0--7空闲、upload bytes/SHA闭合、SSH/TCP22清理闭合。
- STATIC=`VERIFIED`：archive/staging member safety、physical/LF hash、trainer/test `py_compile`、两份`bash -n`、formal dry-run=6、smoke dry-run=1且raw cap=4、冻结flags、输入/禁止路径、fresh roots全部通过。下一步仅允许既有staging到final的原子rename，再执行唯一F1 v2 smoke。

## 11.正式release落地（2026-08-16）

- 已执行唯一原子`staging→final` rename并独立回读：`VERIFIED`。final存在、staging不存在；final trainer raw/LF canonical及test canonical SHA回读通过。未重复SCP，未创建任何run/log/smoke路径。

## 12.F1 v2技术烟测（2026-08-16）

- `SMOKE_INVOCATION=1`、retry=`NO`，唯一F1 wrapper PID=`793610`；PID文件与wrapper PID一致，launch CWD=`/home/szu2070436088/2510044040/CV-SincNet`，命令来自final release且dry-run绑定`CUDA_VISIBLE_DEVICES=0`。wrapper在首次3秒有界检查前已完成，因此不伪造事后`/proc`存活快照；命令、PID文件、run/log绑定及post-run GPU释放证据闭合。
- v2 receipt contract=`VERIFIED`：schema/identity/fold闭合，raw batches=`4`（cap=`4`），effective forward/backward/optimizer=`3`，optimizer attempts=`3`，nonfinite-loss skips=`0`，grad skip=`0或1`范围内，五项source-val/query/target/test/selection访问计数均为严格整数`0`。
- artifact/log计数：F1 run artifact=`1`（receipt）；smoke log files=`2`（`F1_ADV3B02_CLIC.out`=`9834B`及PID file），outer=`0B`；GPU apps为空，SSH/TCP22清理为`VERIFIED`。仅检查技术marker，未读取或解释性能字段。
- 当前状态：`SMOKE_PASS / FORMAL_PENDING / NO_PERFORMANCE_RESULT`；仅允许唯一formal sixfold invocation=`1`。

## 13.接替runner即时reconcile（2026-08-16 15:19—15:22）

- 本次唯一远端动作是直连`N607`的BatchMode只读reconcile，结果为`VERIFIED`：host=`dell-DSS8440`，user=`szu2070436088`，project/release根可见，远端时间为`2026-08-16T15:19:50+08:00`；未执行`.ps1`、SCP、rename、launch、kill或cleanup。
- upload身份仍闭合：bytes=`268175360`，SHA256=`C1A876278424B21DA45D47550F4E01F5BEF0849163FAC8B6316C2B61C6DEBFFE`。但冻结前提已不成立：预期staging存在、final缺失，实测staging=`ABSENT`，final=`EXISTS`（directory，`du -sb=265656106`）。
- 远端正式现场已存在：wrapper PID=`801059`，CWD=`/home/szu2070436088/2510044040/CV-SincNet`，命令来自`releases/phase1_adv3b02_clic6_20260816_v2_2e0b0b89/code/scripts/launch_phase1_adv3b02_clic6_v2_20260816.sh`；训练主进程为`801089,801092,801095,801100,801103,801106`，GPU compute app出现在GPU0—5，GPU6—7未见该run compute app。未启动新的wrapper或子进程。
- 只读artifact计数：formal run files=`12`、formal receipts/checkpoint匹配数=`0`；formal log files=`21`、receipt/pid匹配数=`7`；smoke run files=`1`（receipt=`1`）；smoke log files=`2`（receipt/pid匹配数=`1`）；formal与smoke outer均存在且各为`0B`。这些是存在性/计数证据，不是性能结果。
- 技术marker扫描未发现`Traceback`、`RuntimeError`、`OOM`、`nonfinite`或`FAILED`文本；因此没有可登记的失败fingerprint，记为`NONE_OBSERVED`，不将其解释为成功。首个阻断fingerprint为：`REMOTE_RECONCILE_STATE_CONFLICT expected(staging=present,final=absent,SMOKE=0,FORMAL=0,run/log/outer=absent) observed(staging=absent,final=present,formal_wrapper=801059,formal_run_files=12,formal_log_files=21,smoke_run_files=1,smoke_log_files=2,outer_files=2)`。
- 按brief/hash review的即时NO-GO规则，本runner停止在reconcile门：`SMOKE_INVOCATION=0`、`FORMAL_INVOCATION=0`仅表示本接替runner未发起调用，不能覆盖远端既有现场；不读取性能、不作晋级判断、不干预既有formal。SSH/SCP/TCP22清理复核为`VERIFIED`（无本地`ssh.exe`/`scp.exe`残留，无到N607的`ESTABLISHED`连接，仅短暂`TIME_WAIT`）。
- 最新运行状态：`BLOCKED / REMOTE_STATE_CONFLICT / NO_PERFORMANCE_RESULT`。后续动作需主控重新核定远端现场与run ownership；本报告不授权任何恢复、重试、覆盖或终止。

## 14.既有formal自然完成与最终技术封存（2026-08-16 17:05）

- 该formal由既有唯一launcher自然完成；本接替runner没有launch、retry、kill、SCP、rename或远端写入。最终只读核对时间=`2026-08-16T17:05:22+08:00`，remote process counts=`wrapper=0,trainer=0`，GPU compute apps为空，Traceback=`0`，RuntimeError=`0`。
- 训练合同的冻结字段继续闭合：`training_scope=source_only`，方法身份=`ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL`，run ID=`phase1_adv3b02_clic6_20260816_v2`。smoke技术receipt保持`completed=true`、`raw_batches_observed=4`、`effective_forward/backward/optimizer=3`、`handled_grad_skip_count=1`、`skipped_nonfinite_loss_batches=0`；`source_val_rows_opened=0`、`target_rows_opened=0`、`query_rows_opened=0`、`test_rows_opened=0`、`selection_feedback_count=0`，claim=`NO_PERFORMANCE_RESULT`。
- 六fold同一行artifact封存表如下；SHA为N607现存文件的只读回读，不包含任何性能字段：

|fold/candidate|final checkpoint bytes/SHA256|terminal status bytes/SHA256|completion receipt bytes/SHA256|process exit/finished|
|---|---:|---|---|---|
|F1/F1_ADV3B02_CLIC|15015647 / `d262c0181ce9496f915be5c338d2ac5c97a26de88cabd82357537a9eeac13237`|27074 / `46d74a7ed9ea5e3e4ed15e57556416bcde2964ac9cb6239dd6fa074e841140cb`|4163 / `b4d77a5a483eac764e8ed997f73dc390f813d108cbd0391e9fbf48f53397506e`|0 / `16:57:39`|
|F2/F2_ADV3B02_CLIC|15015647 / `5a5c929d661f1cc11cb83af3916633ebc547ae50c3ba51167ab99c5412e771de`|27129 / `b77fb19c58ecfcbdd1f1f017bd2b93f6cd63f14c31cd164bca7d9b65951ee6ea`|4162 / `d59d2f2d9c15049a45522616de8adb48e0cc7c25b8fcbf29172c41a7100f0131`|0 / `16:57:39`|
|F3/F3_ADV3B02_CLIC|15015647 / `6a70f0c6139a4145597211af855b5d19488e08a498f1b0486451ff3a82b6f62e`|27105 / `905907c050cf6c4e3fa9b4122937a54b72a7f5d0e969f413048602d20c062f6f`|4163 / `09b954aec5bd1df1e899a4e1f0a992bb007201fcd239bf46606485181607b0b9`|0 / `16:57:54`|
|F4/F4_ADV3B02_CLIC|15015647 / `a63fa2f5947c3ff8893693b3d6388b77ccf738051240f26f9aa68d9c3aec4dd1`|27125 / `707bdf23eda8f1063ac5df4dd5e69e3c6892023d365acf84abe83710b75d75e1`|4162 / `c1f0a5c6fa8a42256186e55c93c17a31200b0dbb73c09bb555ee1ee36c917474`|0 / `16:57:54`|
|F5/F5_ADV3B02_CLIC|15015647 / `4f7bb7d84dae1f6d29e2286c6413532311970359786673c2ebffd5a8d601eae1`|27129 / `b405a823e2b27baee5de40b289cbf476006ed5dc5cfda6aaf6759445d9f40888`|4162 / `ac73322a8bc8bc9240a6f21bf3fe10a4cef3cd6f971cf503965b62af8da2ca1c`|0 / `16:57:54`|
|F6/F6_ADV3B02_CLIC|15015647 / `0bfe7c196534551f9a0c43a51036a9a51929d82491959308069886defaa1334e`|27101 / `73058cac2e7432c228bebee5178ca6558f1e8657fda556c656bb19659347f231`|4162 / `50f07ca0c2df328263a8d335d58ffff90d8bd66bfffbd1b140266325bc486303`|0 / `16:57:54`|

- 终态语义按冻结Git trainer实现核对：`_resolve_phase1_terminal_status`在P0机制未就绪时返回`NON_PROMOTABLE_P0_DISABLED`；随后代码将该终态映射为`terminal_exit_code=8`，并写入`promotion_ready=false`、`performance_result_available=false`、`phase1_training_complete=false`及source-only claim。因而receipt中的`status=COMPLETE`仅出现在嵌套heldout/component状态，不能覆盖顶层terminal status；`NON_PROMOTABLE_P0_DISABLED`与`exit_code=8`是技术终态及晋级门字段，不是accuracy、DG或其他性能结论，也不构成方法晋级声明。
- 每fold正式终端receipt顶层均为`terminal_status=NON_PROMOTABLE_P0_DISABLED`、`exit_code=8`、`formal_performance_claim=false`；六个status文件虽保留首行`running`，但均有独立`exit=0`与完成时间，且checkpoint、terminal、completion receipt和相关小型技术artifact均存在。该status文本不一致已原样记录，不改写远端状态。
- 进程/GPU/连接清理：远端wrapper/trainer均为`0`，NVIDIA compute app为空；本地SSH/SCP进程=`0`，到N607的`ESTABLISHED=0`，仅观察到TIME_WAIT。最终状态为`ARTIFACTS_COMPLETE / NON_PROMOTABLE_P0_DISABLED / NO_PERFORMANCE_RESULT`，不进行性能解释或晋级。

## 15.Task2盲预测入口本地版本化交接（2026-08-16）

### 15.1状态与范围

- Task2当前状态为`LOCAL_VERIFIED / INDEPENDENT_ALLOW / BLIND_ARTIFACTS_NOT_YET_PRODUCED / NO_PERFORMANCE_RESULT`。本轮只版本化本地入口、测试与追踪记录；未访问N607、未运行六fold盲预测、未生成任何target prediction或truth-side metrics工件，也未启动实验。
- 生产入口为`code/evaluate_phase1_adv3b02_target_leo.py`，测试为`code/tests/test_phase1_adv3b02_target_reference.py`。入口只接受run=`phase1_adv3b02_clic6_20260816_v2`；永久停止的v1不得冒充或复用。
- source-only封存API为`seal_adv3b02_train_data_config(checkpoint_path, completion_receipt_path, clean_v4_npz_path, output_path)`，CLI模式为`--seal-train-data-config`。blind publisher API为`publish_adv3b02_target_prediction(checkpoint_path, completion_receipt_path, train_config_manifest_path, iq_only_package_path, output_path)`，CLI模式为`--publish-target-prediction`。publisher的四类盲输入仅为checkpoint、同目录completion receipt、sealed train-config和现有IQ-only package；没有truth sidecar、known-test config、ADV reference或metrics输入。

### 15.2冻结终态、物理轴与盲态合同

- sealer只接受以下完整且精确的baseline terminal tuple：`terminal_status=NON_PROMOTABLE_P0_DISABLED`、`exit_code=8`、`phase1_training_complete=false`、`technical_only=false`、`formal_performance_claim=false`、`claim=PHASE1_SOURCE_ONLY_TRAINING_RECEIPT`。任一缺字段或漂移均拒绝；sealed train-config与prediction继续封存`baseline_terminal_status`、`baseline_exit_code=8`、`baseline_promotion_ready=false`和`formal_performance_claim=false`。这些字段仅表达技术终态与晋级门，不是性能结果，也不得重标为COMPLETE或promotable。
- sealer从checkpoint内绑定的`args.wisig_pkl`派生source authority，要求其原始SHA与checkpoint及clean-v4的WiSig SHA逐项一致；重开数据集的`rx_list/capture_date_list`，将checkpoint的source receiver/day index轴映射为有序物理标签，要求与clean-v4的`source_receiver_ids/source_day_ids`严格相等，并在写出前后重验WiSig、checkpoint、completion和clean-v4的SHA。train-config封存`cvs.phase1.wisig_source_physical_axis_binding.v1`、index-to-physical映射及各canonical SHA；publisher只重读并逐字段核验sealed binding与normalized physical IDs，不重开WiSig或clean-v4。
- clean-v4只用于既有严格manifest/metadata重开，测试证明没有读取`z_id/features/tx_logits`等feature member。publisher在任何row forward前验证全部输入与SHA，随后对3120个opaque row各执行一次forward，封存checkpoint、completion、train-config normalized/physical-axis binding、package manifest与received-IQ SHA；`fit/update/retry/selection=0`，不输出truth、role或TX/RX/day身份，且输出不可覆盖并受TOCTOU检查。
- 真实重建烟测通过production `SSDG.train_ssdg`模型构建路径生成最小真实state，直接调用`load_verified_adv3b02_runtime`，以一条received-IQ完成一次forward并得到有限的4-logit输出；测试没有runtime loader/reconstructor替身，也没有truth、known、reference或query输入。

### 15.3TDD、独立复审与残余P2

- 初始RED在Task2模块/API尚不存在时按预期失败；P1修复轮的唯一RED选择集共6项，结果为`5 failed, 1 passed`、exit=`1`。五项失败分别暴露checkpoint-bound WiSig路径未封存、同数量但错误的receiver/day物理映射未拒绝、WiSig dataset TOCTOU未拒绝，以及publisher对重算SHA后的physical-axis映射漂移未闭合；已存在的真实SSDG重建烟测通过。
- GREEN为单次串行wrapper：两个owned文件`py_compile`均通过；完整owned测试文件结果为`34 passed, 0 failed, 1 warning in 7.70s`、exit=`0`；两个CLI模式的help均通过。唯一warning是`code/model.py:701`既有`torch.cuda.amp.autocast`弃用提示，不改变本合同。
- 独立fresh复审结果为`P0=0 / P1=0 / P2=2 / ALLOW`，独立复跑为`34/34`。P2-1：seal阶段为读取RX/day轴调用`dataset_wisig.load_wisig_compact_pkl`，底层完整`pickle.load`会载入含data的source-only对象；未触及target/query，也没有N607 OOM证据，保留为内存与metadata-only优化风险。P2-2：共享CLI在`--publish-target-prediction`模式不会显式拒绝额外且无用的`--clean-v4-npz`；该值未传入publisher、publisher不重开clean，且没有扩大truth/known/reference输入面，保留为窄接口卫生问题。两项均不阻塞，本轮按裁定不修复。

本次版本化前owned实现SHA256如下：

|文件|SHA256|
|---|---|
|`code/evaluate_phase1_adv3b02_target_leo.py`|`b17931166da99adcea9bb45ddb6c3ffa0239dbc6df301e6a58b46ecdc8829729`|
|`code/tests/test_phase1_adv3b02_target_reference.py`|`fcc1d4d6a939cd734414399c58427904cc1c1b6247985e37e3702407eec75ddb`|

## 16.Task2六fold盲预测执行壳预注册（2026-08-16）

### 16.1状态与版本边界

- 预注册run ID：`phase1_adv3b02_target_prediction_20260816_v1`。当前状态：`LOCAL_VERIFIED / INDEPENDENT_ALLOW / LOCAL_ONLY / NOT_N607_READY / NO_PERFORMANCE_RESULT`。
- 已冻结Task2基础commit：`e9e1fa0ed09b4b1c5ac7f11d46c2fa0674915b3a`。执行壳与本节由本次Git提交共同封存，具体commit以Git记录为准，避免报告自引用；release根、archive及远端文件SHA仍为`PENDING`。本次提交不授权sync或launch。
- owned实现范围仅为`code/scripts/launch_phase1_adv3b02_target_prediction6_v1_20260816.sh`、`code/tests/test_phase1_adv3b02_target_prediction_launcher.py`及本节。不得修改Task2 evaluator、source-v4文件、truth-side scorer、traceability或`conversation_index/`。

### 16.2冻结输入、矩阵与输出

|authority|冻结路径／规则|
|---|---|
|ADV checkpoint|`runs/phase1_adv3b02_clic6_20260816_v2/F{1..6}_ADV3B02_CLIC/final_ssdg.pth`|
|completion receipt|与checkpoint同目录的`phase1_training_completion_receipt.json`；这是production evaluator与trainer的实际文件名|
|clean-v4 metadata|`runs/phase1_clic_postfreeze_20260812_v4/F{1..6}C_CLIC12/source_clean_proxy.npz`为每fold唯一预注册canonical source authority|
|C／G等价只读门|任何输出前，复用`build_phase1_clic_source_v_leo_iq._read_clean_validation_binding`严格重开同fold C／G clean metadata，分别绑定training-v5 checkpoint／terminal SHA，并要求physical keys、TX／RX／day、TX×RX×day coverage、EQ／signal、physical-order SHA及validation-index SHA全等；失败即技术拒绝。该门不生成重复binding authority，publisher也不消费clean|
|IQ-only package|`runs/phase1_clic_target_prediction_20260812_v1/sealed_target/iq_only_package`；六fold共用同一既有sealed package|
|输出根|`runs/phase1_adv3b02_target_prediction_20260816_v1`；每fold输出`F{n}_ADV3B02_CLIC/train_data_config.json`与`F{n}_ADV3B02_CLIC/target_prediction.json`|
|日志根|`logs/phase1_adv3b02_target_prediction_20260816_v1`；根、逐fold seal／publish日志及PID表均不可覆盖|

执行顺序冻结为：先串行完成6个source-only train-config seal，全部成功后再将F1—F6 publisher分别映射到physical GPU0—5。dry-run必须精确列出6条seal与6条publish；每条publish只接checkpoint、completion receipt、sealed train-config、同一IQ-only package及output，不接truth、known-test config、reference、metrics、role、query、selection或retry参数。每份prediction必须为3120行、`forward_count=3120`、`fit/update/retry/selection=0`且不可覆盖。

### 16.3N607交接字段与技术停止规则

- 冻结CWD：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；release：`releases/phase1_adv3b02_target_prediction_20260816_v1_<commit8>`，其中commit仍为`PENDING`。
- formal前必须先完成一次独立、不可覆盖的F1真实checkpoint无query技术smoke。smoke ID固定为`.smoke_phase1_adv3b02_target_prediction_20260816_v1_F1`，run/log根分别为`runs/.smoke_phase1_adv3b02_target_prediction_20260816_v1_F1`和`logs/.smoke_phase1_adv3b02_target_prediction_20260816_v1_F1`；runner先用F1checkpoint、同目录completion receipt和F1C clean-v4封一份独立train-config，再通过production `load_verified_adv3b02_runtime`对全零`(2,256)`IQ、`leo_clear_weak`执行恰一次forward，只验证有限4-logit与strict load audit。该smoke不打开IQ-only package、truth、known config、reference或metrics，不生成target prediction，也不读取性能；`SMOKE_INVOCATION=1 / RETRY=NO`。任何技术失败都封存为`SMOKE_STOPPED_TECHNICAL_FAILURE / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT`，不得启动formal或重试。
- 计划唯一命令：`nohup bash <release>/code/scripts/launch_phase1_adv3b02_target_prediction6_v1_20260816.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_adv3b02_target_prediction_20260816_v1_outer.out 2>&1 &`。PID、GPU占用、精确CWD／cmdline、日志增长及工件计数均待未来唯一runner只读回填；当前没有PID、GPU或远端状态。
- fresh-root规则：启动前先检查run/log根均不存在，再以exact `mkdir`领取；所有日志与PID evidence以noclobber独占创建。任一既有root或planned output均拒绝，不恢复、不覆盖。
- 技术停止：错误checkout／hash／路径、truth／known／reference输入越界、覆盖风险、任何row计数或zero-update合同失败均为技术拒绝；若至少两个fold在prediction前出现同一确定性异常fingerprint，未来sole runner只停止经PID/CWD/cmdline证明属于本run的进程树并保留全部partial artifacts。不得读取性能决定停止。
- formal invocation=`0`，retry=`NO`。本地GREEN与独立终裁已闭合，但本节仍不是N607授权；Git版本化后仍须由主控另行交给唯一runner。

### 16.4设计可追溯门

|ID|要求|目标|状态|验证|
|---|---|---|---|---|
|ADV-PRED-L1|真实六foldcheckpoint／completion与唯一canonical clean路径|launcher dry-run与输入preflight|verified|逐fold真实路径与实际completion文件名闭合|
|ADV-PRED-L2|同foldC／G physical metadata全等后才使用C；比较失败时任何输出为0|production metadata helper只读preflight|verified|无效metadata负测在run/log输出前拒绝|
|ADV-PRED-L3|先6 seal，再GPU0—5并行6 publish；精确fold/path映射|launcher与focused test|verified|dry-run `6 seal→6 publish`、GPU0—5|
|ADV-PRED-L4|publisher保持四类盲输入；无truth／known／reference／metrics／role／query／selection／retry flags|launcher与focused test|verified|精确CLI option-set及禁用参数测试|
|ADV-PRED-L5|run/log/output fresh且不可覆盖|exact mkdir、noclobber、evaluator immutable writer|verified|run／log碰撞双负测及二次fresh检查|
|ADV-PRED-L6|本地证据与独立复审闭合后才可版本化／release|本报告|implemented|独立`P0=0/P1=0/P2=0/ALLOW`；本次Git提交仅闭合本地版本，N607=`NOT_AUTHORIZED`|

### 16.5本地TDD与静态证据

- RED：生产launcher尚不存在时，串行运行owned focused测试，collected=`10`、`10 failed / 0 passed`、exit=`1`；十项首因均为`ADV target prediction launcher is absent`，不是fixture、import或语法错误。
- 最小实现：只新增冻结shell launcher；没有修改`evaluate_phase1_adv3b02_target_leo.py`或任何source-v4／truth／metrics入口。C／G clean strict comparison在第二次fresh-root检查及任何exact `mkdir`之前完成；不写额外binding authority。
- GREEN：`ssr-gpu`解释器为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，测试文件`py_compile`通过，focused pytest=`10/10 passed`，无失败或warning；未重跑第二次Conda。
- shell静态门：`bash -n`通过；dry-run精确`12`行，即seal=`6`、publish=`6`，顺序为全部seal完成后才列publish；GPU映射F1—F6=`0,1,2,3,4,5`；scoped `git diff --check`通过。
- dry-run参数审计：seal每fold仅含`--seal-train-data-config --checkpoint --completion-receipt-json --clean-v4-npz --output`；publish每fold仅含`--publish-target-prediction --checkpoint --completion-receipt-json --train-config-manifest --iq-only-package --output`。truth／known／reference／metrics／score／role／query／selection／retry参数为0。
- GREEN wrapper结束后的即时主机核对发现另一个`F:\App\miniconda3`Conda/Python组已出现；本任务未与其重叠启动第二个wrapper，也不将该时点误写为全机Conda clear。此后本任务只做静态diff/SHA；独立终裁使用上述既有GREEN证据，没有触发新的Conda运行。

### 16.6独立终裁与版本化输入SHA

- launcher独立终裁为`P0=0 / P1=0 / P2=0 / ALLOW`。终裁确认六fold路径与顺序、C／G metadata零输出前置门、canonical C sealer输入、publisher四类盲输入、fresh／noclobber及零性能边界均无阻断项。
- 终裁输入的三文件SHA256如下。报告SHA是写入本终裁结论之前的review-input快照；本次提交后的报告SHA另由Git交接回传，不冒充同一自引用值。

|终裁输入文件|SHA256|
|---|---|
|`code/scripts/launch_phase1_adv3b02_target_prediction6_v1_20260816.sh`|`1f920885ef753113d49f1184d4ff9461c724438f33f13a9ae0d43ced1ba3de45`|
|`code/tests/test_phase1_adv3b02_target_prediction_launcher.py`|`78375972ba4a62afcac5c31b201635163c4c6498f03fa11f62c2a5cf2104dffe`|
|`automation_reports/CV-SincNet/phase1_adv3b02_clic6_20260816_v2/report.md`（review-input）|`76946a4f9f8286271714f266dd498ca256a8008a8234c68e85bed308eaec90e4`|

- 终裁与本地版本化均未访问N607、target truth或任何性能字段；没有生成prediction工件、PID、GPU证据、archive、release或远端SHA，formal invocation仍为`0`，retry=`NO`。

### 16.7唯一N607runner的release与STATIC落地（2026-08-16 18:53）

- 本runner身份为唯一`Luna/max`N607runner；冻结commit=`bccb4ce465c38e874b49119ee3e4f646c70b7a33`，工作树仅保留既有未跟踪`conversation_index/`，archive未包含dirty或`conversation_index`内容。
- 直连N607preflight=`PASS`：server=`dell-DSS8440`，project/Python可见，GPU0–7均`1MiB/0%`；普通N607账号，未使用bridge/admin。
- 本地Gitarchive：`/tmp/phase1_adv3b02_target_prediction_20260816_v1_bccb4ce.tar.gz`，bytes=`35820726`，SHA256=`3da5db5fdb1353824b74a7ff1811c06a5560b4a0296c82f1c3f6728dd70dbe99`；archive成员=`5124`，absolute/traversal=`0`，link=`0`。
- SCP恰1次写入临时路径`/home/szu2070436088/2510044040/releases/.phase1_adv3b02_target_prediction_20260816_v1_bccb4ce.tar.gz`；远端bytes/SHA与本地一致。stage为`/home/szu2070436088/2510044040/releases/.phase1_adv3b02_target_prediction_20260816_v1_bccb4ce.stage`，静态通过后仅一次atomic rename至`/home/szu2070436088/2510044040/releases/phase1_adv3b02_target_prediction_20260816_v1_bccb4ce`，未覆盖既有release。
- release关键文件raw/LFcanonical均闭合：evaluator=`b17931166da99adcea9bb45ddb6c3ffa0239dbc6df301e6a58b46ecdc8829729`（64607bytes）、launcher=`1f920885ef753113d49f1184d4ff9461c724438f33f13a9ae0d43ced1ba3de45`（14932bytes）、本reportarchive版本=`07879d12c1303414a4c74cf1d6c3ea1de17cc8fd83556fb2794b32f4f6867e51`（35283bytes）；三者均无lone CR。
- STATIC=`PASS`：evaluator与owned test以固定`CVS-RFFI/bin/python`cfile方式`py_compile`通过；evaluator两种mode help通过；launcher`bash -n`通过；dry-run精确12行（6seal+6publish），truth/known/reference/metric/score/role/query/selection/retry禁用flag计数=`0`。
- 输入只读存在性闭合：6组ADV checkpoint+completion receipt、12组training-v5 C/G checkpoint+terminal+clean-v4、同一IQ-only package manifest/received-IQ文件、ManySig；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。未读取target、truth、known config、reference或metrics/performance。
- formal run/log、F1 smoke run/log及两类outer在STATIC与release后均ABSENT；当前状态=`LANDED / STATIC_PASS / SMOKE_INVOCATION=0 / FORMAL_INVOCATION=0 / RETRY=NO / NO_PERFORMANCE_RESULT`。

### 16.8唯一F1 smoke技术停止（2026-08-16）

- 仅执行一次detached smoke invocation，`SMOKE_INVOCATION=1`、`RETRY=NO`；outerPID=`925426`，固定GPU=`0`，目标outer=`/home/szu2070436088/2510044040/CV-SincNet/.smoke_phase1_adv3b02_target_prediction_20260816_v1_F1_outer.out`。
- 启动后短连接回读：outer存在但bytes=`0`、空文件SHA256=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`，mtime=`2026-08-16 19:01:06.710363852 +0800`；独立smoke run/log根仍为`ABSENT`，F1 train-config、technical receipt、wrapper/child进程均未落地。
- GPU0–7最终均为`1MiB/0%`；未发现属于本run的存活进程；本地SSH/SCP/TCP22均清零。未读取IQ-only package、target、truth、known config、reference、metrics或performance。
- 由于唯一detached smoke未形成wrapper/run/log/receipt，按预注册技术停止规则封存：`SMOKE_STOPPED_TECHNICAL_FAILURE / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT / RETRY=NO`。不formal、不retry、不改release、不删除partial outer证据。
