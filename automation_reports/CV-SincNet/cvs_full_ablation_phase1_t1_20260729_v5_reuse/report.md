# Phase1 T1全消融v5复用发布报告

## 基本信息

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|时间|2026-07-29|
|操作方|Codex主代理；N607唯一runner：`/root/phase1_t1_n607_runner`|
|目标|完成Phase1 T1六臂×五配对种子的30行矩阵，同时复用v3已完整产物，避免重复训练和重复数据审计|
|当前状态|`ARTIFACTS_COMPLETE / ANALYSIS_PENDING`|

## 假设与比较

比较`P1-FULL/P1-SUP/P1-A0/P1-B0/P1-C0/P1-D0`，每臂使用种子`7281101–7281105`，训练比例`0.07/0.63/0.30`，每个新训练行200epochs。旧行与新行可来自不同发布批次；报告中保留`direct_reuse/reexport_only/new_train`来源标签，不把复用行声明为本次重新训练。

## 复用与缺口

|类别|数量|范围|
|---|---:|---|
|直接复用|10|`P1-SUP`五种子、`P1-A0`五种子|
|只补导出|1|`P1-B0__train_seed_7281101`，复用E200 source-val checkpoint|
|新训练|19|`P1-FULL`五行、`P1-C0`五行、`P1-D0`五行、`P1-B0`种子7281102–7281105|
|安全续训|0|中断行不续训|

## 启动前输出闭环

- 每个新训练行必须同时具有非空且可加载的checkpoint和prototype PT、可解析且非空的prototype JSON、resource summary、独立`frozen_phase1_heldout_eval.json`、terminal和completion receipt。
- prototype路径必须严格位于本行输出目录，禁止跨行引用。
- terminal、receipt和真实子进程退出码必须同时为0。
- PID文件使用独占创建；run root、log root和`launch.out`均不可覆盖。
- `runner_summary.json`必须按30行汇总并分列`direct_reuse/reexport_only/new_train`。
- 按用户指示不重新读取整份WiSig文件做SHA256审计，只确认文件存在并沿用已登记的数据标识。

## 本地实现与验证

|文件|用途|
|---|---|
|`code/SSDG/train_ssdg.py`|将独立held-out结果路径和摘要绑定到terminal/completion|
|`code/scripts/run_full_ablation_phase1_t1.py`|复用调度、补导出调度、产物可加载/非空/本行路径验证、19个新训练并发|
|`code/scripts/reexport_phase1_prototypes.py`|从v3的B0 E200 checkpoint仅补prototype导出|
|`code/scripts/seal_full_ablation_phase1_plan.py`|将补导出脚本和复用配置纳入release|
|`configs/full_ablation_20260728/phase1_t1_reuse_v5.json`|冻结10+1+19复用矩阵|
|`tests/test_run_full_ablation_phase1_t1.py`|复用、补导出、held-out和损坏产物故障注入|

验证结果：最终release定向回归`72 passed,1 skipped`（含真实B0 checkpoint导出）；复用九项故障注入全部拒绝；`py_compile`和`git diff --check`通过。

## N607发布参数

|字段|值|
|---|---|
|Conda/Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|工作目录|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v5_reuse_4592bdd9/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|launch log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v5_reuse.launch.out`|
|GPU调度|8张GPU，每张最多2个进程，共16槽|
|预期执行|19个完整训练+1个checkpoint-only导出；10行直接复用|
|成功条件|30/30有效闭合，19个新训练成功、1个补导出成功、10个复用行验证通过、无P0、无重复异常指纹系统停机|
|系统停机|P0协议/输出覆盖风险，或至少两个不同新执行行在生成prototype前出现同一确定性异常指纹|

## N607落地与启动证据

|字段|证据|
|---|---|
|发布提交|`4592bdd9497feffe69298a20c436abd177801231`；远端detached HEAD一致且tracked-clean|
|远端release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v5_reuse_4592bdd9`|
|独立复审|`PRELAUNCH_OUTPUT_CLOSURE_PASS`；P0/P1/P2均为0|
|seal后dry-run|30逻辑行、20个dispatch、10个direct reuse、1个reexport-only、19个new train、16个首波静态槽|
|PID绑定|`launch.pid=711522`为外层后台bash；其唯一直接子进程`711523`为正式Python runner，CWD与release/code一致|
|第二波状态|2026-07-29 18:41:05+08:00；16个逐行PID文件；completed=1、succeeded=1、failed=0；15个训练GPU进程活跃；日志总量828,564字节，约E5–E6|
|异常扫描|未检出`Traceback`、`RuntimeError`、`CUDA out of memory`或`unrecognized argument`；无异常指纹|
|SSH清理|本地`ssh.exe=0`，N607与bridge的ESTABLISHED TCP22连接均为0|

### B0补导出闭环

|字段|值|
|---|---|
|行/模式|`P1-B0__train_seed_7281101`；`reexport_only`|
|执行|PID`711835`；GPU2、slot1；22.32秒；return_code=0|
|验证|completion_receipt_valid=true；P0=false；receipt status=`COMPLETE`|
|输出|prototype PT 407,306字节；JSON 2,108,153字节；receipt 1,128字节|

### 第二波GPU快照

|GPU|活跃训练进程|利用率|显存|
|---:|---:|---:|---:|
|0|2|88%|6079MiB|
|1|2|100%|6307MiB|
|2|1|40%|3094MiB|
|3|2|99%|6383MiB|
|4|2|95%|6075MiB|
|5|2|93%|6177MiB|
|6|2|92%|6361MiB|
|7|2|97%|6169MiB|

GPU2的第二个静态槽用于B0补导出，完成后该槽无后续行，因此第二波为15个训练进程；首波16个任务均有独占PID证据。当前只有技术健康结论，无性能结论。

### 18:44短连接健康快照

|字段|值|
|---|---|
|服务器时间|2026-07-29 18:44:48+08:00|
|runner|PID`711523`仍存活，CWD仍为冻结release/code|
|行计数|launched=16、completed=1、succeeded=1、failed=0|
|活跃训练|15个本run GPU进程；GPU0/1/3/4/5/6/7各2个，GPU2为1个|
|训练进度|15个活跃行最新epoch为E12–E19；仅用于非停滞健康判断|
|日志增长|逐行日志总量由828,564字节增至1,300,787字节|
|异常/协议|硬错误0、P0=0、无异常指纹|
|汇总产物|`runner_summary.json`尚不存在，符合全矩阵运行中预期|
|SSH清理|本地`ssh.exe=0`，N607/bridge的ESTABLISHED TCP22连接=0|

### 18:49短连接健康快照

|字段|值|
|---|---|
|服务器时间|2026-07-29 18:49:02+08:00|
|runner绑定|实际runner PID`711523`、PPID`711522`，状态`Ssl`，运行14分15秒；CWD为冻结release/code；cmdline中的plan、repo root、run root、log root、WiSig路径、Python、训练脚本、复用manifest和reexport脚本均与交接命令一致，并带`--execute`|
|行计数|launched=16、completed=1、succeeded=1、failed=0；非零返回状态计数0|
|活跃训练|15个本run GPU进程；GPU0/1/3/4/5/6/7各2个，GPU2为1个|
|GPU利用率|GPU0–7依次为94%、99%、22%、98%、99%、94%、97%、97%；对应显存6083、6309、3100、6387、6079、6183、6365、6173MiB|
|训练进度|15个活跃行最新`[EPOCH-BEGIN]`为E17–E25；仅作为持续推进证据|
|日志增长|逐行日志总量由18:44的1,300,787字节增至1,757,758字节，增加456,971字节|
|异常/协议|硬错误扫描计数0；P0状态计数0；非零状态计数0；无异常指纹，未触发技术停机规则|
|汇总产物|`runner_summary.json`仍不存在，符合完整矩阵尚未完成的预期|
|SSH清理|本次命令退出码0；本地`ssh.exe=0`，N607/bridge的ESTABLISHED TCP22连接=0|

### 18:51直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 18:51:18+08:00按`tools\n607_ssh_preflight.ps1`执行；直连配置、普通账号身份、服务器时间、项目根目录和8张RTX 3090均通过；preflight退出码0|
|服务器快照时间|2026-07-29 18:51:46+08:00|
|runner绑定|实际runner PID`711523`、PPID`711522`，状态`Ssl`，运行16分59秒；CWD及完整cmdline继续与冻结release和交接命令一致|
|行计数|launched=16、completed=1、succeeded=1、failed=0；非零返回状态计数0|
|活跃训练|15个本run GPU进程；GPU0/1/3/4/5/6/7各2个，GPU2为1个|
|GPU利用率|GPU0–7依次为97%、99%、46%、98%、97%、97%、94%、91%；对应显存6087、6315、3100、6391、6081、6185、6367、6173MiB|
|训练进度|15个活跃行最新`[EPOCH-BEGIN]`为E20–E28；仅作为持续推进证据|
|日志增长|逐行日志总量由18:49的1,757,758字节增至1,986,730字节，增加228,972字节|
|异常/协议|硬错误扫描计数0；P0状态计数0；非零状态计数0；无异常指纹，未触发技术停机规则|
|汇总产物|`runner_summary.json`仍不存在，确认完整矩阵尚未结束，因此本次不进行性能读取或same-row结果分析|
|SSH清理|preflight后及监控后均确认本地`ssh.exe=0`，N607/bridge的ESTABLISHED TCP22连接=0|

### 19:05直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 19:04:45+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 19:05:15+08:00|
|runner绑定|实际runner PID`711523`、PPID`711522`，状态`Ssl`，运行30分28秒；CWD继续绑定冻结release/code|
|行计数|launched=16、completed=1、succeeded=1、failed=0；唯一状态行仍为已闭合的`P1-B0__train_seed_7281101`补导出|
|第二波状态|尚无训练行完成，因此原runner没有第二波训练可自动启动；当前PID行仍为首波的1个补导出加15个训练行，未进行人工补进程|
|活跃训练|15个本run GPU进程；GPU0/1/3/4/5/6/7各2个，GPU2为1个|
|GPU利用率|GPU0–7依次为23%、93%、23%、97%、93%、99%、98%、96%；对应显存6091、6319、3100、6395、6087、6185、6367、6175MiB|
|训练进度|15个活跃行最新`[EPOCH-BEGIN]`为E31–E46；仅作为非停滞证据|
|日志增长|逐行日志总量由18:51的1,986,730字节增至3,069,689字节，增加1,082,959字节|
|异常/协议|硬错误扫描计数0；P0状态计数0；非零状态计数0；无异常指纹，未触发技术停机规则|
|汇总产物|`runner_summary.json`仍不存在，完整矩阵未完成|
|SSH清理|preflight后及监控后均确认本地`ssh.exe=0`，N607/bridge的ESTABLISHED TCP22连接=0|

### 19:20直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 19:19:51+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 19:20:25+08:00|
|runner绑定|实际runner PID`711523`、PPID`711522`，状态`Ssl`，运行45分38秒；CWD继续绑定冻结release/code|
|行计数|launched=16、completed=1、succeeded=1、failed=0；唯一状态行仍为`P1-B0__train_seed_7281101`补导出|
|完成行与第二波|尚无训练行完成，因此没有新增行需要输出闭合核验，原runner也没有第二波训练可自动启动；未人工补进程|
|活跃训练|15个本run GPU进程；GPU0/1/3/4/5/6/7各2个，GPU2为1个|
|GPU利用率|GPU0–7依次为99%、98%、22%、97%、90%、96%、99%、94%；对应显存6091、6319、3100、6395、6087、6185、6367、6175MiB|
|训练进度|15个活跃行最新`[EPOCH-BEGIN]`为E43–E65；仅作为非停滞证据|
|日志增长|逐行日志总量由19:05的3,069,689字节增至4,308,607字节，增加1,238,918字节|
|异常/协议|硬错误扫描计数0；P0状态计数0；非零状态计数0；无异常指纹，未触发技术停机规则|
|汇总产物|`runner_summary.json`仍不存在，完整矩阵未完成|
|SSH清理|preflight后及监控后均确认本地`ssh.exe=0`，N607/bridge的ESTABLISHED TCP22连接=0|

当前仅能下结论为`LANDED / RUNNING / FIRST-WAVE HEALTHY`，不能据此形成任何性能结论。

### 19:49直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 19:46:51+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 19:49:41+08:00|
|runner与release绑定|实际runner PID`711523`、PPID`711522`仍存活；CWD为冻结release/code；完整cmdline中的plan、repo root、run root、log root、Python、训练脚本、复用manifest及reexport脚本继续与交接命令一致并带`--execute`；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|行计数|dispatch launched=16、completed=1、succeeded=1、failed=0、nonzero=0；唯一完成状态仍是已闭合的`P1-B0__train_seed_7281101`补导出；20个dispatch中尚有4个D0训练行未启动|
|完成行闭环|本次没有新增训练行产生`phase1_terminal_status.json`或`phase1_training_completion_receipt.json`，因此无需新增checkpoint、prototype PT/JSON、resource summary、held-out eval、terminal、completion receipt和退出码核验；既有B0补导出状态仍为return code 0且receipt有效|
|第二波状态|15个首波训练行均仍存活，原runner尚无已完成训练槽可自动调度后续4个D0行；未人工补跑、重启或改变队列|
|本run训练占用|T1在GPU0–7的活跃训练实验数依次为`2/2/1/2/2/2/2/2`，合计15|
|整机训练占用|GPU2另有1个label实验，其余GPU无额外训练实验；按训练主进程去除DataLoader子进程后，GPU0–7总训练实验数均为2，没有超过每卡2实验上限|
|GPU利用率|GPU0–7依次为85%、91%、73%、99%、97%、86%、95%、99%；对应显存6091、6319、6047、6395、6087、6217、6397、6191MiB|
|训练进度|15个活跃T1行最新`[EPOCH-BEGIN]`为E67–E95；仅作为非停滞健康证据，不读取或比较中间性能|
|日志增长|T1逐行日志总量由19:20的4,308,607字节增至6,559,248字节，增加2,250,641字节|
|异常/协议|完整T1逐行日志硬错误扫描计数0；P0标记计数0；失败状态0、非零状态0，因此没有异常指纹，未触发预注册技术停机规则|
|汇总产物|log root中的`runner_summary.json`仍不存在，符合完整矩阵尚未完成的预期|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`；到`172.31.111.215:22`及bridge`172.31.105.18:22`的ESTABLISHED连接均为0|

### 19:54直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 19:53:32+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 19:54:25+08:00|
|runner与release绑定|实际runner PID`711523`、PPID`711522`仍存活；CWD为冻结release/code；完整cmdline继续绑定原plan、repo root、run root、log root、CVS-RFFI Python、训练脚本、复用manifest及reexport脚本并带`--execute`；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|行计数|dispatch launched=16、completed=1、succeeded=1、failed=0、nonzero=0；唯一完成状态仍是已闭合的`P1-B0__train_seed_7281101`补导出；20个dispatch中尚有4个D0训练行未启动|
|新增完成行闭环|本次新增完整训练行数为0；没有新增checkpoint、prototype PT/JSON、resource summary、held-out eval、terminal、completion receipt和真实退出码需要验收|
|第二波状态|15个首波训练行继续存活，原runner尚无训练完成后释放的T1槽位，因此后续4个D0行仍在原冻结队列等待；未人工补跑、重启或改变调度|
|本run训练占用|T1在GPU0–7的活跃训练实验数依次为`2/2/1/2/2/2/2/2`，合计15|
|整机训练占用|按训练主进程剔除DataLoader子进程后，GPU0–7总训练实验数均为2；其中GPU2为1个T1加1个label，其余各卡均为2个T1，没有超过每卡2实验上限|
|GPU利用率|GPU0–7依次为93%、92%、90%、90%、94%、96%、98%、93%；对应显存6091、6319、6219、6395、6087、6219、6401、6191MiB|
|训练进度|15个活跃T1行最新`[EPOCH-BEGIN]`为E70–E98；仅作为持续推进证据，不读取或比较中间性能|
|日志增长|T1逐行日志总量由19:49的6,559,248字节增至6,891,953字节，增加332,705字节|
|异常/协议|完整逐行日志硬错误扫描计数0、P0标记计数0；失败状态0、非零状态0，没有异常指纹，未触发预注册技术停机规则|
|汇总产物|log root中的`runner_summary.json`仍不存在，完整矩阵尚未结束|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`；到N607及bridge的ESTABLISHED TCP22连接均为0|

### 20:13直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 20:12:55+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 20:13:32+08:00|
|runner与release绑定|实际runner PID`711523`、PPID`711522`仍存活；CWD和完整cmdline继续绑定冻结release/code、原plan、repo root、run root、log root、CVS-RFFI Python、训练脚本、复用manifest及reexport脚本并带`--execute`；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|dispatch计数|launched=16、completed=1、succeeded=1、failed=0、nonzero=0；唯一完成状态仍为已闭合的`P1-B0__train_seed_7281101`补导出；20个dispatch中尚有4个D0训练行等待|
|新增完成行闭环|本次没有新增训练行完成，因此没有新增checkpoint、prototype PT/JSON、resource summary、held-out eval、terminal、completion receipt或真实退出码需要验收|
|第二波调度|15个首波T1训练行仍全部存活，原runner尚未获得训练完成后释放的槽位；后续4个D0行继续留在原冻结队列，未人工启动、补跑或改变调度|
|本run与整机训练数|T1在GPU0–7的活跃训练实验数为`2/2/1/2/2/2/2/2`；按训练主进程剔除DataLoader子进程后，整机GPU0–7均为2个训练实验，其中GPU2为1个T1加1个label，没有超过每卡2实验上限|
|GPU利用率|GPU0–7依次为99%、91%、98%、98%、98%、65%、94%、98%；对应显存6127、6355、6235、6431、6123、6221、6403、6193MiB|
|训练进度|15个活跃T1行最新`[EPOCH-BEGIN]`为E85–E113；仅作为日志持续推进的健康证据，不读取或比较中间性能|
|日志增长|T1逐行日志总量由19:54的6,891,953字节增至8,309,286字节，增加1,417,333字节|
|异常/协议|完整逐行日志硬错误扫描计数0、P0标记计数0；failed=0、nonzero=0，因此没有异常指纹或重复异常指纹，未触发预注册技术停机规则|
|汇总产物|log root中的`runner_summary.json`仍不存在，完整矩阵尚未结束|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`；到N607及bridge的ESTABLISHED TCP22连接均为0|

### 20:28直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 20:27:55+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 20:28:33+08:00|
|runner与release绑定|实际runner PID`711523`、PPID`711522`仍存活；CWD和完整cmdline继续绑定冻结release/code、原plan、repo root、run root、log root、CVS-RFFI Python、训练脚本、复用manifest及reexport脚本并带`--execute`；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|dispatch计数|launched=16、completed=1、succeeded=1、failed=0、nonzero=0；唯一完成状态仍为已闭合的`P1-B0__train_seed_7281101`补导出；20个dispatch中尚有4个D0训练行等待|
|新增完成行闭环|本次没有新增完整训练行，因此没有新增checkpoint、prototype PT/JSON、resource summary、held-out eval、terminal、completion receipt和真实退出码需要验收|
|活跃与等待|15个首波T1训练行仍全部存活；原runner尚未获得训练完成后释放的槽位，后续4个D0行继续留在原冻结队列；未人工启动、补跑或改变调度|
|本run与整机训练数|T1在GPU0–7的活跃训练实验数为`2/2/1/2/2/2/2/2`；按训练主进程剔除DataLoader子进程后，整机GPU0–7均为2个训练实验，其中GPU2为1个T1加1个label，没有超过每卡2实验上限|
|GPU利用率|GPU0–7依次为91%、99%、93%、91%、99%、99%、99%、95%；对应显存6129、6357、6237、6433、6123、6221、6403、6195MiB|
|训练进度|15个活跃T1行最新`[EPOCH-BEGIN]`为E97–E130；仅作为持续推进健康证据，不读取或比较中间性能|
|日志增长|T1逐行日志总量由20:13的8,309,286字节增至9,422,361字节，增加1,113,075字节|
|异常/协议|完整逐行日志硬错误扫描计数0、P0标记计数0；failed=0、nonzero=0，因此没有异常指纹或重复异常指纹，未触发预注册技术停机规则|
|汇总产物|log root中的`runner_summary.json`仍不存在，完整矩阵尚未结束|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`；到N607及bridge的ESTABLISHED TCP22连接均为0|

### 20:35直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 20:35:15+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 20:35:56+08:00|
|runner与release绑定|实际runner PID`711523`、PPID`711522`仍存活；CWD和完整cmdline继续绑定冻结release/code、原plan、repo root、run root、log root、CVS-RFFI Python、训练脚本、复用manifest及reexport脚本并带`--execute`；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|子进程绑定|15个活跃T1训练主进程全部满足PPID=`711523`、CWD=冻结release根目录、cmdline同时包含本run ID和冻结`train_ssdg.py`，未发现越界或失联子进程|
|dispatch计数|launched=16、completed=1、succeeded=1、failed=0、nonzero=0、active=15、waiting=4；唯一完成状态仍为已闭合的`P1-B0__train_seed_7281101`补导出|
|新增完成行闭环|本次没有新增完整训练行，因此没有新增checkpoint、prototype PT/JSON、resource summary、held-out eval、terminal、completion receipt和真实退出码需要验收；既有B0补导出闭环保持有效|
|16槽整体占用|T1在GPU0–7的活跃训练实验数为`2/2/1/2/2/2/2/2`；GPU2另有1个label训练实验，因此按训练主进程剔除DataLoader子进程后，整机GPU0–7均为2个训练实验，16个允许槽位全部占用且没有超限|
|GPU利用率|GPU0–7依次为96%、99%、72%、99%、97%、87%、98%、98%；对应显存6129、6357、6255、6433、6123、6221、6403、6213MiB|
|训练进度|15个活跃T1行最新`[EPOCH-BEGIN]`为E103–E138；仅作为持续推进健康证据，不读取或比较中间性能|
|prediction/score与artifact|Phase1 T1训练阶段不产生Phase2 prediction/score，当前run root中的prediction文件数=0、score文件数=0，属不适用而非缺失；本轮没有新增完整训练artifact集合，不能把运行中的checkpoint或局部文件计为完成|
|日志增长|T1逐行日志总量由20:28的9,422,361字节增至9,961,870字节，增加539,509字节|
|异常/协议|完整逐行日志硬错误扫描计数0、P0标记计数0；failed=0、nonzero=0，因此没有异常指纹或重复异常指纹，未触发预注册技术停机规则|
|汇总产物|log root中的`runner_summary.json`仍不存在，完整矩阵尚未结束|
|数据复用边界|沿用既有数据与复用结果；本轮没有重新审计数据集，也没有进行跨批次数据一致性或hash对齐|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`；到N607及bridge的ESTABLISHED TCP22连接均为0|

### 20:59直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 20:59:09+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 20:59:46+08:00|
|runner与子进程绑定|实际runner PID`711523`、PPID`711522`仍存活；CWD和完整cmdline继续绑定冻结release/code、原plan/run/log、CVS-RFFI Python及`--execute`；15个活跃T1训练主进程均由PID`711523`直接持有，CWD、run ID和冻结训练脚本绑定全部通过|
|release状态|远端release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|dispatch计数|launched=16、completed=1、succeeded=1、failed=0、nonzero=0、active=15、waiting=4；唯一完成状态仍为已闭合的`P1-B0__train_seed_7281101`补导出|
|首个新完整训练行|尚未出现；本次没有新增new-train status，因此不能将运行中的checkpoint或局部输出计为完成，也没有新增完整artifact集合需要验收|
|waiting自动派发|15个首波T1训练仍全部存活，尚无训练槽位释放；原runner因此尚未自动派发后续4个D0行，队列未被人工改动|
|GPU槽位|T1在GPU0–7的活跃训练实验数为`2/2/1/2/2/2/2/2`；GPU2另有1个label训练实验，整机GPU0–7均为2个训练实验，16个允许槽位仍全部占用且没有超限或释放|
|GPU利用率|GPU0–7依次为99%、91%、94%、99%、92%、99%、99%、91%；对应显存6129、6357、6725、6433、6123、6253、6435、6229MiB|
|训练进度|15个活跃T1行最新`[EPOCH-BEGIN]`为E121–E163；仅作非停滞健康证据，不读取或比较中间性能|
|日志增长|T1逐行日志总量由20:35的9,961,870字节增至11,668,162字节，增加1,706,292字节|
|异常/协议|完整逐行日志硬错误扫描计数0、P0标记计数0；failed=0、nonzero=0，因此没有异常指纹或重复异常指纹，未触发预注册技术停机规则|
|汇总产物|log root中的`runner_summary.json`仍不存在，完整矩阵尚未结束|
|数据复用边界|继续复用既有结果；本轮没有数据集重审，也没有进行跨批次数据一致性或hash对齐|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`；到N607及bridge的ESTABLISHED TCP22连接均为0|

### 22:11直连preflight、完成行闭环与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 22:10:53+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 22:11:31+08:00|
|runner与release绑定|runner PID`711523`、PPID`711522`仍存活；CWD和完整cmdline继续绑定冻结release/code、原plan/run/log、CVS-RFFI Python及`--execute`；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|dispatch计数|launched=16、completed=8、succeeded=8、failed=0、nonzero=0、active=8、waiting=4；相较20:59新增7个完整new-train状态|
|逻辑矩阵状态|10个direct reuse+8个已完成dispatch=18个已闭合逻辑行；8个训练行活跃，4个D0尚未派发；共30行，未把运行中局部产物计为完成|
|新增完成行|`P1-C0`五个种子、`P1-D0__train_seed_7281101`、`P1-FULL__train_seed_7281105`；runner记录的真实return code均为0、completion receipt均有效、P0均为false|
|严格独立验收|在冻结release的`PYTHONPATH`和CVS-RFFI Python下，对7行逐一加载checkpoint与prototype PT，解析prototype JSON、resource、heldout、terminal和completion receipt，并核对本行绝对路径、输出hash、terminal/receipt/heldout COMPLETE状态和退出码；结果7/7通过、0失败|
|验收命令环境说明|第一次独立加载未带冻结release的`PYTHONPATH`，仅checkpoint反序列化出现`ModuleNotFoundError`；其他绑定检查通过。使用与runner一致的只读模块路径重验后7/7全部加载通过，该问题属于验收命令环境缺失，不是训练或artifact失败|
|活跃行|`P1-FULL`种子7281101–7281104和`P1-B0`种子7281102–7281105，共8行；8个训练主进程全部由runner PID`711523`直接持有，CWD与cmdline绑定正确|
|waiting自动派发|后续4个D0行位于GPU0/1的冻结slot队列，需等待对应4个活跃FULL行完成后由原runner自动派发；本轮在GPU2/5/6/7完成的行不会跨GPU改写冻结队列，因此waiting仍为4，未人工补跑或改调度|
|GPU训练数与空槽|T1在GPU0–7为`2/2/0/2/2/0/0/0`；label v2在GPU0–7为`0/0/2/0/0/2/1/1`；整机合计为`2/2/2/2/2/2/1/1`，GPU6和GPU7各释放1个允许槽位，其余卡仍为2个训练实验|
|GPU利用率|GPU0–7依次为99%、99%、95%、90%、99%、97%、23%、38%；对应显存6163、6391、5923、6467、6159、6263、3194、3202MiB|
|训练进度|8个活跃T1行最新`[EPOCH-BEGIN]`为E174–E178；仅作为非停滞健康证据，不读取或比较中间性能|
|日志增长|T1逐行日志总量由20:59的11,668,162字节增至16,235,026字节，增加4,566,864字节|
|异常/协议|完整逐行日志硬错误扫描计数0、P0标记计数0；failed=0、nonzero=0，没有异常指纹或重复异常指纹，未触发预注册技术停机规则|
|汇总产物|log root中的`runner_summary.json`仍不存在，完整矩阵尚未结束|
|数据复用边界|继续复用既有结果；本轮没有数据集重审，也没有进行跨批次数据一致性或hash对齐|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`；到N607及bridge的ESTABLISHED TCP22连接均为0|

#### 22:11新增完整训练行artifact验收

|行|checkpoint/B|prototype PT/B|prototype JSON/B|resource/B|heldout/B|terminal/B|completion receipt/B|return code|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`P1-C0__train_seed_7281101`|15,229,188|414,337|2,087,034|524|8,148|26,704|4,908|0|完整，通过|
|`P1-C0__train_seed_7281102`|15,229,252|397,309|2,024,391|524|8,218|26,750|4,908|0|完整，通过|
|`P1-C0__train_seed_7281103`|15,229,124|405,823|2,081,272|523|8,132|26,698|4,907|0|完整，通过|
|`P1-C0__train_seed_7281104`|15,229,060|405,823|2,041,139|524|8,158|26,707|4,908|0|完整，通过|
|`P1-C0__train_seed_7281105`|15,229,124|398,728|2,061,870|524|8,079|26,626|4,908|0|完整，通过|
|`P1-D0__train_seed_7281101`|15,259,724|394,471|2,056,002|524|8,217|26,779|4,908|0|完整，通过|
|`P1-FULL__train_seed_7281105`|15,262,796|394,471|2,057,588|528|8,195|26,754|4,922|0|完整，通过|

### 22:25直连preflight与健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 22:24:58+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 22:25:39+08:00|
|runner与子进程绑定|runner PID`711523`、PPID`711522`仍存活，CWD/cmdline继续绑定冻结release及原plan/run/log；8个活跃T1训练主进程全部由runner直接持有，CWD、run ID和冻结训练脚本绑定通过|
|dispatch计数|launched=16、completed=8、succeeded=8、failed=0、nonzero=0、active=8、waiting=4；相较22:11没有新增完成行|
|产物完整性|22:11独立严格验收通过的7个new-train行仍全部具有非空checkpoint、prototype PT/JSON、resource、heldout、terminal和completion receipt，JSON状态继续为COMPLETE且真实return code为0；本轮没有新行需要重复验收|
|waiting自动派发|活跃行仍为`P1-FULL`种子7281101–7281104和`P1-B0`种子7281102–7281105；后续4个D0继续等待GPU0/1对应FULL槽位释放，原runner尚未自动派发，未人工改变队列|
|16槽上限|T1在GPU0–7为`2/2/0/2/2/0/0/0`；label v2为`0/0/2/0/0/2/1/1`；整机为`2/2/2/2/2/2/1/1`，GPU6/7各空1槽，其余卡为2个训练实验，没有超过每卡2实验上限|
|GPU利用率|GPU0–7依次为99%、98%、94%、95%、98%、95%、37%、19%；对应显存6163、6391、6301、6467、6159、6267、3196、3204MiB|
|训练进度|8个活跃T1行最新`[EPOCH-BEGIN]`为E184–E189；仅作非停滞健康证据，不读取或比较中间性能|
|日志增长|T1逐行日志总量由22:11的16,235,026字节增至16,731,757字节，增加496,731字节|
|异常/协议|完整逐行日志硬错误扫描计数0、P0标记计数0；failed=0、nonzero=0，没有异常指纹或重复异常指纹|
|汇总产物|log root中的`runner_summary.json`仍不存在，完整矩阵尚未结束|
|数据边界|本轮未读取性能、未重启或修改任务、未重审数据，也未进行跨批次一致性或hash对齐|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`，N607及bridge的ESTABLISHED TCP22连接均为0|

### 22:54直连preflight、第二波派发与完成行闭环

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 22:53:50+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 22:54:27+08:00|
|runner与release绑定|runner PID`711523`、PPID`711522`仍存活，CWD/cmdline继续绑定冻结release及原plan/run/log；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|dispatch计数|launched=19、completed=16、succeeded=16、failed=0、nonzero=0、active=3、waiting=1；相较22:25新增8个完整new-train行，并由原runner自动派发3个D0|
|逻辑矩阵状态|10个direct reuse+16个已完成dispatch=26个已闭合逻辑行；3个D0活跃，1个D0等待，共30行|
|新增完成行|`P1-B0`种子7281102–7281105和`P1-FULL`种子7281101–7281104；runner记录的真实return code均为0、completion receipt均有效、P0均为false|
|严格独立验收|在冻结release的`PYTHONPATH`和CVS-RFFI Python下，对新增8行逐一加载checkpoint与prototype PT，解析prototype JSON、resource、heldout、terminal和completion receipt，并核对本行路径、输出hash、COMPLETE状态及退出码；结果8/8通过、0失败|
|第二波自动派发|`P1-D0__train_seed_7281102`和`7281103`已由原runner派发到GPU0，`P1-D0__train_seed_7281104`已派发到GPU1；3个训练主进程均由runner PID`711523`直接持有且CWD/cmdline绑定正确|
|剩余waiting|`P1-D0__train_seed_7281105`仍在冻结队列等待；GPU1当前已有1个T1 D0和1个label v2，达到每卡2进程上限，因此runner未超限派发。GPU6/7的空槽不用于跨GPU改写冻结队列|
|16槽上限|T1在GPU0–7为`2/1/0/0/0/0/0/0`；全机训练实验为`2/2/2/2/2/2/1/1`，所有GPU均不超过2个训练实验|
|GPU利用率|GPU0–7依次为97%、92%、99%、100%、92%、93%、31%、38%；对应显存6073、6273、6337、6161、6293、6287、3232、3238MiB|
|训练进度|3个第二波D0最新`[EPOCH-BEGIN]`为E8–E12；仅作启动后持续推进证据，不读取或比较性能|
|日志增长|T1逐行日志总量由22:25的16,731,757字节增至17,510,030字节，增加778,273字节|
|异常/协议|完整逐行日志硬错误扫描计数0、P0标记计数0；failed=0、nonzero=0，没有异常指纹或重复异常指纹|
|汇总产物|log root中的`runner_summary.json`仍不存在，完整矩阵尚未结束|
|数据边界|本轮未读取性能、未重启或修改任务、未重审数据，也未进行跨批次一致性或hash对齐|
|SSH清理|所有短连接退出后确认本地`ssh.exe=0`，N607及bridge的ESTABLISHED TCP22连接均为0|

#### 22:54新增完整训练行artifact验收

|行|checkpoint/B|prototype PT/B|prototype JSON/B|resource/B|heldout/B|terminal/B|completion receipt/B|return code|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`P1-B0__train_seed_7281102`|15,262,860|404,404|2,052,886|524|8,172|26,721|4,908|0|完整，通过|
|`P1-B0__train_seed_7281103`|15,262,796|402,985|2,088,200|524|8,172|26,731|4,908|0|完整，通过|
|`P1-B0__train_seed_7281104`|15,262,732|407,242|2,056,207|524|8,135|26,677|4,908|0|完整，通过|
|`P1-B0__train_seed_7281105`|15,262,732|398,728|2,012,016|522|8,168|26,714|4,906|0|完整，通过|
|`P1-FULL__train_seed_7281101`|15,262,860|410,080|2,086,829|527|8,181|26,752|4,921|0|完整，通过|
|`P1-FULL__train_seed_7281102`|15,262,924|394,471|2,010,448|528|8,193|26,759|4,922|0|完整，通过|
|`P1-FULL__train_seed_7281103`|15,262,796|394,471|2,063,968|528|8,095|26,655|4,922|0|完整，通过|
|`P1-FULL__train_seed_7281104`|15,262,732|401,566|2,029,383|528|8,129|26,684|4,922|0|完整，通过|

### 23:11 T1与label v2合并直连健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 23:10:36+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 23:11:16+08:00|
|runner与release绑定|T1 runner PID`711523`、PPID`711522`仍存活，CWD/cmdline继续绑定冻结release及原plan/run/log；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|dispatch计数|launched=19、completed=16、succeeded=16、failed=0、nonzero=0、active=3、waiting=1；相较22:54无新增完成行或派发行|
|闭合artifact|此前两批共15个new-train行已分别通过7/7和8/8严格独立验收；本轮这些行的checkpoint、prototype PT/JSON、resource、heldout、terminal和completion receipt仍全部非空，return code均为0；没有新行需要验收|
|活跃/等待|`P1-D0`种子7281102、7281103在GPU0，种子7281104在GPU1，3个训练主进程全部由runner直接持有且CWD/cmdline绑定正确；种子7281105因GPU1已有1个T1和1个label v2而继续等待|
|GPU上限|T1在GPU0–7为`2/1/0/0/0/0/0/0`；合并label v2后全机为`2/2/2/2/2/2/1/1`，所有GPU均不超过2个训练实验|
|GPU利用率|GPU0–7依次为97%、95%、98%、92%、91%、99%、21%、29%；对应显存6075、6273、6355、6167、6297、6321、3232、3240MiB|
|训练进度|3个D0最新`[EPOCH-BEGIN]`为E28–E32；仅作非停滞健康证据，不读取或比较性能|
|日志增长|T1逐行日志总量由22:54的17,510,030字节增至17,859,188字节，增加349,158字节|
|异常/汇总|硬错误、P0、failed、nonzero、异常指纹和重复异常指纹均为0；`runner_summary.json`仍不存在，矩阵尚未结束|
|数据边界|本轮未读取性能、未干预或重启任务、未重审数据，也未进行跨批次一致性或hash对齐|
|SSH清理|全部短连接退出后本地`ssh.exe=0`，N607及bridge的ESTABLISHED TCP22连接均为0|

### 23:36 T1与label v2合并直连健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 23:35:48+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 23:36:27+08:00|
|runner与release绑定|T1 runner PID`711523`、PPID`711522`仍存活，CWD/cmdline继续绑定冻结release及原plan/run/log；release为tracked-clean，HEAD=`4592bdd9497feffe69298a20c436abd177801231`|
|dispatch计数|launched=19、completed=16、succeeded=16、failed=0、nonzero=0、active=3、waiting=1；相较23:11无新增完成行或派发行|
|闭合artifact|此前两批共15个new-train行已分别通过7/7和8/8严格独立验收；本轮没有新完成行，不重复读取已闭合artifact|
|活跃/等待|`P1-D0`种子7281102、7281103在GPU0，种子7281104在GPU1；3个训练主进程全部由runner直接持有且CWD/cmdline绑定正确。种子7281105因GPU1已有1个T1和1个label v2而继续等待|
|GPU上限|T1在GPU0–7为`2/1/0/0/0/0/0/0`；label v2为`0/1/1/2/2/2/0/0`；全机为`2/2/1/2/2/2/0/0`，所有GPU均不超过2个训练实验|
|GPU利用率|GPU0–7依次为96%、98%、0%、95%、97%、98%、0%、0%；对应显存6075、6291、3160、6185、6315、6325、4、4MiB|
|训练进度|3个D0最新`[EPOCH-BEGIN]`为E57–E59；仅作非停滞健康证据，不读取或比较性能|
|日志增长|T1逐行日志总量由23:11的17,859,188字节增至18,341,633字节，增加482,445字节|
|异常/汇总|硬错误、P0、failed、nonzero、异常指纹和重复异常指纹均为0；`runner_summary.json`仍不存在，矩阵尚未结束|
|数据边界|本轮未读取性能、未干预或重启任务、未重审数据，也未进行跨批次一致性或hash对齐|
|SSH清理|全部短连接和artifact验收连接退出后本地`ssh.exe=0`，N607及bridge的ESTABLISHED TCP22连接均为0|

### 23:53 T1与label v2合并直连健康快照

|字段|值|
|---|---|
|本地直连preflight|2026-07-29 23:50:59+08:00通过；直连配置、普通账号身份、服务器时间、项目根目录及8张RTX 3090均正常；退出码0|
|服务器快照时间|2026-07-29 23:53:55+08:00；23:55:13补充复核GPU瞬时进程|
|runner与进程绑定|T1 runner PID`711523`、PPID`711522`存活，CWD/cmdline继续绑定冻结release及原plan/run/log；3个训练主进程均由runner直接持有，进程cmdline、CWD、row和run ID绑定通过|
|dispatch计数|launched=19、completed=16、succeeded=16、failed=0、nonzero=0、active=3、waiting=1；相较23:36没有新增完成行或派发行|
|闭合artifact|此前两批共15个new-train行已分别通过7/7和8/8严格独立验收；本轮没有T1新完成行，不重复读取已闭合artifact|
|活跃/等待|`P1-D0`种子7281102、7281103、7281104继续活跃，最新`[EPOCH-BEGIN]`为E76–E78；种子7281105继续等待。epoch仅作非停滞健康证据，不读取或比较性能|
|Phase1实际GPU训练数|T1在GPU0–7为`2/1/0/0/0/0/0/0`；label v2为`0/1/1/2/2/1/0/0`；合计`2/2/1/2/2/1/0/0`，所有GPU均不超过2个Phase1训练进程|
|GPU利用率|GPU0–7依次为96%、96%、20%、82%、98%、20%、0%、0%；对应显存6077、6311、3162、6223、6351、3118、4、4MiB|
|Stage2短时进程边界|补充进程检查曾观察到唯一Stage2 v3 deployment bundle `prepare`辅助进程PID`892322`、参数`--device cuda:0`；它不属于Phase1训练。在23:53:55及23:55:13两次GPU compute-app快照中均未形成额外GPU进程，随后自行结束；未误计、未干预|
|日志增长|T1逐行日志总量由23:36的18,341,633字节增至18,674,970字节，增加333,337字节|
|异常/汇总|完整日志硬错误、P0、failed、nonzero、异常指纹和重复确定性异常指纹均为0；`runner_summary.json`仍不存在|
|数据边界|本轮未读取性能、未干预或重启任务、未重审数据，也未进行跨批次一致性或hash对齐|
|SSH清理|全部短连接和artifact验收连接退出后本地`ssh.exe=0`，N607及bridge的ESTABLISHED TCP22连接均为0|

当前仅能下结论为`LANDED / RUNNING / FIRST-WAVE HEALTHY`，不能据此形成任何性能结论。

## 风险与完成后检查

- 复用行和新训练行不是同一发布批次，最终统计必须保留来源列。
- B0补导出必须使用当前修复后的endpoint exporter，原v3失败terminal不能当作完整训练receipt；使用独立reexport receipt闭合。
- 完成后检查30行同一行指标、最佳epoch/checkpoint、held-out指标、prototype、资源、异常指纹和GPU释放，再推进设计报告下一批实验。

### 2026-07-30 00:42只读健康快照

|字段|值|
|---|---|
|preflight/服务器时间|直连preflight于00:41:46+08:00通过；快照时间00:42:32+08:00|
|dispatch计数|launched=19、completed=16、succeeded=16、failed=0、nonzero=0、active=3、waiting=1|
|活跃行|`P1-D0`种子7281102、7281103、7281104；最新E130–E132仅作非停滞证据|
|进程绑定|runner PID`711523`、PPID`711522`及3个直接子进程的CWD/cmdline/row/run绑定通过|
|日志/异常|log_bytes=19,594,138；summary尚不存在；硬错误、异常指纹和P0均为0|
|合并GPU训练数|T1=`2/1/0/0/0/0/0/0`，label v2=`0/1/0/1/1/0/0/0`，合计=`2/2/0/1/1/0/0/0`，均不超过2|
|GPU利用率/显存|利用率=`90/91/0/23/24/0/0/0%`；显存=`6111/6325/4/3140/3240/4/4/4MiB`|
|操作边界|未读取性能，未启动、终止、修改或重试任务，未重审数据或要求跨批次一致|
|SSH清理|本地`ssh.exe=0`，ESTABLISHED TCP22=0|

状态保持`RUNNING / HEALTHY / NO_PERFORMANCE_RESULT`。

## 2026-07-31终态补录

本节于N607服务器时间约2026-07-31 11:47+08:00执行一次只读终态取证。只读取既有`runner_summary.json`和非性能控制字段；未启动、修改、重试或终止任务，未重验数据，未进行跨批次hash对齐，也未读取准确率等性能值。

|终态检查项|结果|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|summary时间|2026-07-30 04:02:51+08:00|
|矩阵闭合|30/30完成，30成功，0失败|
|来源构成|10条`direct_reuse`、1条`reexport_only`、19条`new_train`|
|逐行控制|30/30均`return_code=0`、`completion_receipt_valid=true`、`p0_protocol_violation=false`，无`completion_error`|
|系统状态|`systemic_stop=false`；异常指纹0；线程错误0|
|自然退出|原外层PID`711522`和runner PID`711523`均不存在；当前含本run ID的远端进程0|
|SSH清理|本地`ssh.exe=0`；N607及bridge的ESTABLISHED TCP22连接均为0|

技术终态为`ARTIFACTS_COMPLETE / ANALYSIS_PENDING`。30行Phase1 T1执行面已经闭合；同row性能表、最佳epoch/checkpoint引用和最终因果判定将在全量结果分析阶段补充。
