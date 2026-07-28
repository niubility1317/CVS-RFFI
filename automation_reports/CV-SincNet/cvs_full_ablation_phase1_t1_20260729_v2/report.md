# CVS-RFFI Phase1第一层全量消融CUDA修复重发

## 身份与状态

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase1_t1_20260729_v2`|
|日期|2026-07-29|
|operator|Codex主代理；N607发布由唯一实验runner子代理执行|
|状态|`RUNNING / FIRST_WORKER_WAVE_HEALTHY`|
|协议|Phase1 source-only；`0.07/0.63/0.30`|
|Git分支|`codex/full-ablation-20260728`|
|代码提交|`f8a46c1b3d889e0eeea2c3fb75b6d8c871c6881a`|
|独立审查|`P0=0、P1=0 / APPROVE_LOCAL_VERIFIED`|
|前序run|`cvs_full_ablation_phase1_t1_20260728_v1`，系统性技术失败、无性能结果、不可恢复|
|性能结论|无；本run执行中|

## 目标、假设与对照

目标是完成`P1-FULL/P1-SUP/P1-A0/P1-B0/P1-C0/P1-D0`六个第一层arm的5个paired seed重训，共30个完整row。科学假设、对照、数据划分和停止规则与前序run一致；本次只修复训练前CUDA显存审计的context初始化顺序，不改变方法、seed、split、epoch、checkpoint选择或评价。

前序run在任何checkpoint、prototype、prediction或score前失败，因此本次复用原预登记paired seeds不会利用性能反馈，也不构成择优重跑。

## 冻结矩阵与资源

- Phase1 seeds：`7281101–7281105`。
- 六个arm×5个paired seed=30 rows。
- 16个固定slot：GPU0–GPU7，每卡slot0/slot1。
- 每卡外部训练PID与本run PID合计最多2个；占用未知时失败关闭。
- epochs=`200`；checkpoint selection=`source_validation_only`。
- 任何run/log/output/status路径碰撞均失败关闭。

## 修复与本地验证

|项目|证据|
|---|---|
|失败指纹|`2c4936848af1568e890c52538321afa8444cad01fa3d806a62b86ab9779d65a6`|
|根因|`reset_peak_memory_stats`在CUDA context初始化前调用|
|修复|`set_device→torch.empty(0,device)→reset_peak_memory_stats`|
|方法变化|无|
|本地环境|`ssr-gpu`|
|主代理验证|61项聚焦测试、`py_compile`、`git diff --check`通过|
|独立验证|63项相关测试、真实CUDA初始化/reset、静态编译通过|
|资源副作用|零元素tensor不保留显存、不消耗随机数；reset后正式峰值从0计|

## N607发布预登记

|字段|预登记值|
|---|---|
|远端项目根|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|WiSig|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`|
|已知WiSig SHA256|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；启动前重验|
|release checkout|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v2_e6383147`|
|sealed plan|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v2_e6383147.sealed.json`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v2`|
|启动命令/PID/GPU|见“正式启动与首波健康证据”；main PID/PGID/SID=`280646`；GPU0–7各2个worker|

发布必须使用精确Git bundle、Git blob SHA、tracked-clean独立checkout、当前review receipt、30-row sealed plan和`CVS-RFFI`environment receipt。旧v1 release/run/log只读保留，不得覆盖、移动、删除或作为本run输入。

## 本地不可变发布证据

|artifact|SHA256或绑定值|
|---|---|
|Git bundle|`b4ed0ad27261644f6828f600e73c4a551d2849a91b3db7cd16a1f6c743fcee54`|
|sealed plan文件|`6f1aeb65cf5f43e0f0a153a71a3280a13bf1cc8e30af69402ff22545d9437bf2`|
|sealed content|`bbc6d989104708cf82fbfc8524a680bf36a588bceca788430300ad9205c1f545`|
|review receipt|`96f0af1e254be1d5789f62678a94a012d6d393a448382e45684861cee25e4b5a`|
|精确checkout|commit=`e63831476f17c11cbac9d5d49075847c574b729d`；tracked-clean；release文件哈希全部匹配|
|矩阵验证|30 rows；16 slots；GPU0–7各2 slots；`formal_launch_authority=true`|

## N607只读预检

预检时间为`2026-07-29T01:43:21+08:00`，结论为`LAUNCH_READINESS=PRECHECK_PASS_RESOURCE_AND_PATH_READY`。

|项目|实时证据|
|---|---|
|主机|`dell-DSS8440`；项目根可见|
|远端环境|`CVS-RFFI`；torch=`2.1.0+cu121`；CUDA=`12.1`；8个CUDA device|
|GPU占用|GPU0–7均为0%利用率、1/24576MiB、无compute PID、无训练进程|
|可用容量|每卡2个新增slot，共16个slot|
|WiSig|2,359,341,461字节；SHA256与预登记一致|
|v1状态|release/run/log保留；主PID和17个row PID存活数均为0|
|v2路径|release、sealed plan、run、log四个目标均不存在|
|磁盘|`/home`可用约7.5TB，使用率29%|
|连接闭合|本地`ssh.exe`残留0；N607和bridge TCP22残留0|

本次预检全程只读，未执行SCP、mkdir、launch、kill或任何远端写入。

## 正式启动与首波健康证据

用户已在当前任务明确授权“启动”。唯一runner于`2026-07-29 02:55:00 +08:00`启动v2；`launch.out`为`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v2.launch.out`。

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v2_e6383147/code && nohup setsid env PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v2_e6383147/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v2_e6383147/code/scripts/run_full_ablation_phase1_t1.py --plan /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v2_e6383147.sealed.json --repo-root /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v2_e6383147 --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase1_t1_20260729_v2 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v2 --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --train-script /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase1_t1_20260729_v2_e6383147/code/SSDG/train_ssdg.py --poll-seconds 30 --execute > /home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase1_t1_20260729_v2.launch.out 2>&1 < /dev/null &
```

|项目|启动后证据|
|---|---|
|main进程|PID/PGID/SID=`280646`；PPID=`1`；CWD、cmdline与v2 release/run/log精确绑定|
|receipts|dataset=`52bb5f...a8a9`；environment=`4c9efd...582b`；sealed plan文件=`6f1aeb...7bf2`|
|worker|16/16存活，均为main直接子进程；GPU0–7各2个|
|GPU|每卡恰好2个compute PID；显存约1.3–4.0GiB；利用率20%–99%|
|日志|16个row日志均增长；最新6行E002/200、10行E003/200|
|中间artifact|16个checkpoint；status=0；terminal=0|
|异常|errors=0；尚无异常指纹|

|GPU|slot0 row/PID|slot1 row/PID|
|---:|---|---|
|0|`P1-FULL/s7281101/281222`|`P1-FULL/s7281102/281451`|
|1|`P1-FULL/s7281103/281329`|`P1-FULL/s7281104/281224`|
|2|`P1-FULL/s7281105/281220`|`P1-SUP/s7281101/281260`|
|3|`P1-SUP/s7281102/281232`|`P1-SUP/s7281103/281517`|
|4|`P1-SUP/s7281104/281518`|`P1-SUP/s7281105/281240`|
|5|`P1-A0/s7281101/281515`|`P1-A0/s7281102/281229`|
|6|`P1-A0/s7281103/281235`|`P1-A0/s7281104/281516`|
|7|`P1-A0/s7281105/281238`|`P1-B0/s7281101/281519`|

## 健康门与停止规则

- 启动后立即核对main PID、PGID/SID、CWD、cmdline、run/log、16槽、GPU映射和receipt。
- 首个row及首个worker wave报告launched/completed/succeeded/failed、终局artifact计数、活跃PID、GPU利用率/显存和异常指纹。
- 任一P0立即停止新派发；两个不同row在终局artifact前出现同一确定性指纹同样停止。
- 只终止已证明属于本run的进程组；不干预无关作业。
- 不因accuracy、H、floor或其他性能值停止。
- 技术早停固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

## 预期输出与结果表

每个row必须生成terminal/completion/resource receipt、source-validation checkpoint、source-only prototype、完整日志和退出证据。当前尚无运行结果：

|arm|seed|split|参数量|source UDU|hard-TX|LEO clear|LEO low|LEO rain|状态|判定|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
|首波16行|`7281101–7281105`|`0.07/0.63/0.30`|待终局|—|—|—|—|—|`RUNNING`|仅技术健康，无性能结论|

## 风险与后续

1. 本次仅授权启动v2；若本run技术失败，不自动授权再次重试。
2. 远端CUDA/torch/driver或GPU占用变化必须在启动前重新检查。
3. 双进程/GPU只用于吞吐，不作为隔离时延或峰值资源主张。
4. 30个row完整闭合后才进行同row统计和Phase2 bundle选择。
