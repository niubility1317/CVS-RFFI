# Phase1 ADV3B02 CLIC六折v2技术烟测与正式入口预注册报告

## 1.状态、范围与非性能边界

- 实验ID：`phase1_adv3b02_clic6_20260816_v2`。
- 当前状态：`STATIC_HASH_REGISTRATION_DEFECT_CONFIRMED / REMOTE_RESUME_NOT_YET_AUTHORIZED / SMOKE0 / FORMAL0`。
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
