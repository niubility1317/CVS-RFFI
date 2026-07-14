# DRIFT/RIEI论文一致性修复后正式复现实验

## 基本信息

- 实验ID：`paper_repro_repaired_riei_drift_seed1337_20260714_103000`
- 时间：2026-07-14
- 操作者：Codex
- 目标：在修复DRIFT Eq.(25)与RIEI Eq.(10c)/(11)更新语义后，重跑DRIFT Table I及RIEI Table III全部12行。
- 对照：`paper_repro_original_riei_drift_seed1337_20260714_090706`；DRIFT last5=`49.37±3.04%`，RIEI 12行last10平均=`54.06%`、MAE=`19.24pp`。
- 声明边界：仅为原论文closed-set cross-receiver复现，不是CVSStage2、卫星/LEO或部署证据。

## 修复假设

|方法|论文要求|旧实现偏差|修复|
|---|---|---|---|
|DRIFT|Eq.(25)在每个mini-batch按receiver计算center，并对receiver domain loss求和|跨batch EMA center；domain mean使中心约束缩小`1/D`|`center_mode=batch`；domain sum|
|RIEI|Eq.(10c)与Eq.(11)对同一`theta_F`依次更新|FED同时属于两个独立Adam实例，两步使用不同动量/二阶矩状态|EC/RC专用optimizer；FED只属于一个optimizer，每iteration连续执行CE step与MI/IE step|

DRIFT的raw negative-MSE、RIEI的CE/MI/IE sum reduction以及两篇论文的模型、数据计数、超参数和last-N评估保持不变。

## 实验矩阵与判定

- DRIFT：Day1；train RX=`1-1,14-7,7-7`；7个held-out RX；200epoch；last5；论文目标`75.62%`，单seed差值绝对值≤3pp初判通过。
- RIEI：Table III全部12个“两source receiver→一held-out receiver”组合；200epoch；last10；论文12行均值逐行比较；要求至少10/12落入论文±2SD且MAE≤3pp。
- seed=`1337`；batch64；Adam；输入equalized WiSig I/Q `2x256`；无satellite augmentation、无target support、无无标签路线。

## 本地版本、验证与快照

- 根目录`E:\type10-7`不是Git仓库；Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`，关键提交：`73f694a`、`90f81e8`、`50b7ab1`。
- 测试：`pytest tests/test_riei_alternating_paper_parity.py tests/test_drift_eq25_paper_parity.py`为`3 passed`；相关Python文件`py_compile`通过；根目录DRIFT parity unittest为`13/13 OK`。
- launcher：`code/scripts/launch_paper_repro_repaired_matrix_20260714.sh`；`bash -n`通过；dry-run展开13个job。
- 快照：`E:\type10-7\code\snapshots\paper_repro_repaired_riei_drift_seed1337_20260714_103000\`。

## N607资源与队列计划

2026-07-14 10:31+08:00只读inventory显示另一个已存在的`phase1_dgleo_corepath8_20260714`任务在8张GPU各占1个compute process。按每GPU最多2个训练的规则，本矩阵只允许每GPU新增1个峰值训练。新launcher把13个job分配成8个per-GPU顺序队列：GPU0-4各排2个，GPU5-7各排1个；因此每GPU`current=1+planned_peak=1`，总峰值为2，不会并发启动同一GPU上的第二个本矩阵job。

- N607工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- run根：`paper_reproduction/runs/paper_repro_repaired_riei_drift_seed1337_20260714_103000`。
- log根：`paper_reproduction/logs/paper_repro_repaired_riei_drift_seed1337_20260714_103000`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 同步文件：`baselines/drift/losses.py`、`baselines/drift/train_cvs.py`、`baselines/riei_fd/train.py`、`baselines/riei_fd/train_cvs.py`、`run_wisig_paper_scope_queue.sh`、新launcher。
- 安全边界：不修改、终止或重启已有Phase1任务；只同步RIEI/DRIFT专用文件和独立launcher；启动前再次执行capacity gate。

## 完成后检查

完整读取13份200epoch训练日志与metrics；报告DRIFT同一run七receiver结果、RIEI Table III 12行结果、loss/feature norm曲线、硬错误、与旧run及论文逐行差值。best-val仅作诊断，不能替代论文last5/last10。

## 同步与启动状态

- 同步后N607 SHA256与本地快照一致：DRIFT losses=`40f709b0...`、DRIFT train=`fdead828...`、RIEI train=`e2f6797c...`、RIEI train_cvs=`5ad054a7...`、paper queue=`5f93bc7c...`、repaired launcher=`33fde40a...`。
- 远端两个launcher均通过`bash -n`；repaired launcher dry-run展开13个job与8个顺序队列。
- 正式启动时间：2026-07-14 10:35:18+08:00。
- 精确命令：`bash code/scripts/launch_paper_repro_repaired_matrix_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。
- capacity gate实测GPU0-7均为`current=1,planned_peak=1,total_peak=2,max=2`。
- queue PID：GPU0=`269549`、GPU1=`269551`、GPU2=`269553`、GPU3=`269557`、GPU4=`269560`、GPU5=`269567`、GPU6=`269573`、GPU7=`269580`。
- 约1分钟健康检查：8个本任务训练均已进入epoch，DRIFT到epoch9，7个首批RIEI到epoch4-5；DRIFT日志确认`center_mode=batch`，RIEI日志确认CE/MI/IE均为`sum`；8份训练日志均无硬错误。
- GPU证据：每张GPU恰有1个已有Phase1 compute process和1个本任务compute process，总计2/GPU；本矩阵未越过并发上限。
- 连接清理：本轮SSH/SCP结束后本地`ssh.exe=0`，到N607的`ESTABLISHED TCP22=0`。
- heartbeat`riei-drift`已更新为每30分钟监控本修复后run；运行中只读，不干预本run或已有Phase1。

当前状态为`RUNNING_STARTUP_HEALTHY`，不是artifact-complete或论文复现成功结论。

### 2026-07-14 10:39+08:00只读监控检查点

- 状态：`RUNNING_HEALTHY`；已完成job=`0/13`，8个per-GPU queue PID均存活，当前本任务训练进程=`8`。
- 进度：DRIFT到epoch26；首批7个RIEI均到epoch12；当前共8份训练日志，第二批5个RIEI尚在各自GPU顺序队列中等待，符合设计。
- 队列：GPU0-7各出现1个`QUEUE-JOB-START`且尚无`QUEUE-JOB-END`；无非零退出状态。
- 健康性：8份当前训练日志均完整扫描，未发现`Traceback`、`RuntimeError`、OOM、CUDA error、`NaN`或`Killed`。
- GPU occupancy：GPU0-7均为`compute=2`；每GPU由1个既有Phase1训练和1个本任务训练组成，本任务未超过新增1个/GPU的边界；显存约`4022-4102MiB/GPU`。
- Phase1保持运行，本检查点未启动、终止、重启、覆盖或修改任何远端任务与产物。
- SSH/SCP均为短连接；检查完成后未保留交互式shell或转发。

### 2026-07-14 11:12+08:00只读监控检查点

- 8个本任务训练仍在运行；每GPU同时有1个Phase1训练，合计2个compute process/GPU，未超限。
- DRIFT已完成epoch200且queue status=`0`；GPU0已顺序进入第二个RIEI job，未并发叠加。
- 其余当前RIEI进度为epoch117-120，GPU0第二批RIEI到epoch45；已完成queue job数=`1/13`。
- 完整扫描当前26份`.log`，精确硬错误模式下Traceback、RuntimeError、OOM、Killed、NaN和unrecognized arguments均为0；宽泛`Inf`命中均来自`split_info`字段，不是数值Inf。
- 按用户要求，后续fix_optimized联合矩阵已通过独立deferred launcher提交；它仍只读等待本run全部退出，不影响本run或Phase1。

### 2026-07-14 11:42+08:00只读监控检查点

- 状态保持`RUNNING_HEALTHY`：已完成queue job=`1/13`（DRIFT，status=0），当前本任务训练=`8`。
- 首批剩余7个RIEI到epoch188-193；GPU0第二批RIEI到epoch117。所有训练日志均持续增长。
- GPU0-7均为2个compute process：1个Phase1＋1个本任务；显存约4056-4154MiB/GPU，未超并发上限。
- 完整日志精确硬错误扫描仍为0：无Traceback、RuntimeError、OOM、Killed、NaN或参数错误。
- deferred fixopt PID=`289073`保持存活，日志显示`active_target_processes=16`，仍处于等待阶段，没有新增GPU训练。
