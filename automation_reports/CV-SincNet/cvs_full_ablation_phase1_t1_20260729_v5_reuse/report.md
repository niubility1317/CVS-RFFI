# Phase1 T1全消融v5复用发布报告

## 基本信息

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_t1_20260729_v5_reuse`|
|时间|2026-07-29|
|操作方|Codex主代理；N607唯一runner：`/root/phase1_t1_n607_runner`|
|目标|完成Phase1 T1六臂×五配对种子的30行矩阵，同时复用v3已完整产物，避免重复训练和重复数据审计|
|当前状态|`LANDED / RUNNING / FIRST-WAVE HEALTHY`|

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

## 风险与完成后检查

- 复用行和新训练行不是同一发布批次，最终统计必须保留来源列。
- B0补导出必须使用当前修复后的endpoint exporter，原v3失败terminal不能当作完整训练receipt；使用独立reexport receipt闭合。
- 完成后检查30行同一行指标、最佳epoch/checkpoint、held-out指标、prototype、资源、异常指纹和GPU释放，再推进设计报告下一批实验。
