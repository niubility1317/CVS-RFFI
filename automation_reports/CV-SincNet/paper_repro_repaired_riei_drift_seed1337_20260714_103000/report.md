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

### 2026-07-14 12:12+08:00只读监控与wrapper异常诊断

- 训练artifact进度：DRIFT及首批8个RIEI均已完整到epoch200并写出`PAPER-EVAL-SUMMARY`和`FINAL-TEST`；剩余4个RIEI正在epoch72-74，当前训练进程=`4`。
- queue记录为已结束job=`9/13`，其中DRIFT status=0，8个已完成RIEI被记录为status=2。完整尾部诊断确认RIEI训练本身正常完成，例如`rx1_1_rx7_7_to_rx1_19` last10=`66.41±0.49%`且final=`66.00%`；status=2来自训练完成后wrapper继续读取脚本时出现`line 394: riei_fd: command not found`及`line 395: syntax error`。
- 根因：11:10为满足后续fixopt直接排队请求，同步了`run_wisig_paper_scope_queue.sh`；当时已有长时间运行的shell实例在Python训练返回后继续从同一路径读取脚本，远端文件原位替换导致这些既有shell在旧文件偏移处解析到新内容。同步前后的脚本分别通过`bash -n`，新启动的后续4个RIEI使用完整新文件并正常运行。
- 影响边界：8个RIEI的200epoch训练、last10、final和metrics均已完整落盘；queue脚本未启用`set -e`，因此第二批job继续顺序启动，没有训练artifact丢失。不得把status=2误报为模型训练失败，但最终报告必须记录wrapper异常。
- 处置：遵守“不自动重启”，不重跑已完整落盘的8个RIEI，不修改当前远端文件，不干预剩余4个RIEI或Phase1。后续不再在活动queue读取期间同步共享launcher。
- GPU occupancy：GPU0-4各2个compute process或正在切换；GPU5-7仅剩Phase1，总GPU compute=`12`，未超每卡2个上限。deferred fixopt仍等待所有旧run queue退出。

### 2026-07-14 12:42+08:00只读监控检查点

- 训练artifact已完成=`9/13`：DRIFT与8个RIEI均到epoch200；剩余4个RIEI分别到epoch161、159、160、159，训练日志持续增长。
- 当前本任务训练进程=`4`，全机GPU compute process=`12`；GPU0仅有Phase1，GPU1-4各为Phase1＋1个本任务，GPU5-7仅有Phase1，每卡均未超过2个训练进程。
- 完整日志精确硬错误扫描仍为0；既有8个RIEI的16条wrapper错误记录仅对应已确认的训练完成后脚本偏移异常，没有新增训练硬错误。
- deferred fixopt PID=`289073`及父包装PID=`289071`均存活，最新等待状态`active_target_processes=8`；fixopt run目录尚不存在，未新增GPU占用。
- 本检查点保持只读，未干预、重启、覆盖修复版或Phase1产物与进程。

### 2026-07-14 12:58+08:00修复版完成与完整结果分析

修复版13/13个训练artifact均自然完成200epoch；本地拉取并完整读取13份训练日志、13份`metrics.json`及8份queue日志。训练日志与逐epoch metrics未发现`Traceback`、`RuntimeError`、CUDA OOM、`Killed`、参数错误或loss/metric NaN/Inf。queue最终为DRIFT及后启动的4个RIEI status=0、同步前已启动的8个RIEI status=2；后者均在200epoch、`PAPER-EVAL-SUMMARY`、`FINAL-TEST`和metrics完整落盘后触发已诊断的脚本原位替换偏移异常，因此不改变模型结果，但属于必须保留的wrapper异常证据。本地小型证据位于`remote_artifacts/logs`和`remote_artifacts/metrics`。

#### DRIFT Table I

|candidate|协议|seed|last5|论文|差值|较修复前|final|best-val epoch/test|判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|`drift_table1_seed1337`|Day1；train RX=`1-1,14-7,7-7`；7个held-out RX|1337|39.99±3.13%|75.62%|-35.63pp|-9.38pp|37.29%|122/43.89%|`NOT_REPRODUCED`|

同一run最后5个epoch的七receiver结果：

|held-out receiver|last5均值±SD|
|---|---:|
|`1-19`|34.08±5.55%|
|`19-2`|44.40±4.67%|
|`2-1`|43.07±4.52%|
|`2-19`|37.88±5.15%|
|`20-1`|39.55±4.19%|
|`7-14`|46.62±9.27%|
|`8-8`|34.35±6.73%|

Eq.(25)语义修复没有恢复论文性能，反而使last5由修复前49.37%降至39.99%。完整200epoch轨迹显示raw negative-MSE继续主导并放大特征：`loss_mse`从-2729.35变为-247240.34，`loss_feature_norm`从10.78增至967.22，`z_tx_norm`从22.37增至319.28，`z_rx_norm`从28.44增至325.51；与此同时验证准确率达到99.17%，但held-out receiver仅37.29%。这支持fixopt对negative-MSE加cap的稳定性假设，不支持把paper-literal修复版标为复现成功。

#### RIEI Table III完整12行

|candidate|train RX→test RX|last10|论文|差值|较修复前|final|best-val epoch/test|epoch200 CE/MI/IE/FN|±2SD|
|---|---|---:|---:|---:|---:|---:|---:|---|---|
|`rx1_1_rx7_7_to_rx1_19`|`1-1,7-7`→`1-19`|66.41±0.49%|77.88±2.23%|-11.47pp|+1.14pp|66.00%|10/66.48%|0.004/0.0108/158.185/0.882|否|
|`rx1_1_rx8_8_to_rx1_19`|`1-1,8-8`→`1-19`|68.14±5.51%|79.43±1.66%|-11.29pp|+4.58pp|58.25%|61/66.56%|0.096/0.0091/157.682/0.695|否|
|`rx1_1_rx14_7_to_rx1_19`|`1-1,14-7`→`1-19`|51.63±5.51%|66.09±0.67%|-14.46pp|-5.28pp|60.40%|63/59.27%|0.017/0.0036/158.634/0.846|否|
|`rx7_7_rx8_8_to_rx1_19`|`7-7,8-8`→`1-19`|76.25±5.24%|70.51±3.53%|+5.74pp|+9.82pp|69.29%|39/70.44%|0.065/0.0069/158.213/0.593|是|
|`rx7_7_rx14_7_to_rx1_19`|`7-7,14-7`→`1-19`|59.30±3.06%|77.35±1.53%|-18.05pp|-14.82pp|55.79%|2/61.12%|0.051/0.0274/157.799/0.789|否|
|`rx8_8_rx14_7_to_rx1_19`|`8-8,14-7`→`1-19`|57.86±2.60%|75.48±1.21%|-17.62pp|-1.90pp|59.38%|3/60.25%|0.018/0.0007/159.005/0.463|否|
|`rx1_1_rx1_19_to_rx14_7`|`1-1,1-19`→`14-7`|40.22±6.76%|71.91±2.08%|-31.69pp|+0.78pp|37.83%|18/41.98%|0.016/0.0061/156.981/0.711|否|
|`rx1_1_rx7_7_to_rx14_7`|`1-1,7-7`→`14-7`|43.94±3.95%|68.33±2.37%|-24.39pp|-0.87pp|41.50%|23/45.46%|0.066/0.0891/157.940/1.107|否|
|`rx1_1_rx8_8_to_rx14_7`|`1-1,8-8`→`14-7`|44.65±3.06%|73.54±1.27%|-28.89pp|-6.76pp|47.42%|52/41.31%|0.010/0.0397/145.453/1.313|否|
|`rx1_19_rx7_7_to_rx14_7`|`1-19,7-7`→`14-7`|42.17±7.67%|73.52±3.15%|-31.35pp|+10.66pp|50.29%|56/37.98%|0.184/0.0103/156.761/0.663|否|
|`rx1_19_rx8_8_to_rx14_7`|`1-19,8-8`→`14-7`|53.41±4.01%|72.05±2.71%|-18.64pp|+5.54pp|55.33%|195/54.27%|0.144/0.0120/157.788/0.900|否|
|`rx7_7_rx8_8_to_rx14_7`|`7-7,8-8`→`14-7`|43.85±2.77%|73.46±2.00%|-29.61pp|-3.73pp|49.56%|7/46.65%|0.508/0.0630/150.684/0.893|否|

联合结果：12行last10平均53.99%，论文平均73.30%，MAE=20.26pp，仅1/12行进入论文±2SD；相对修复前平均54.06%反而下降0.07pp，MAE由19.24pp增至20.26pp。共享FED optimizer修复了论文更新状态语义，但没有修复跨receiver泛化，故RIEI Table III仍为`NOT_REPRODUCED`。该结果也再次确认历史62.74%仅属于DRIFT的`drift_day1`协议，不能替代此12行Table III证据。

#### 完成边界与后续

- 修复版结论：DRIFT和RIEI均未复现论文结果；paper-literal语义修复本身不足。
- deferred launcher在全部旧run进程退出并通过容量门后，于12:58自动启动fixopt 20-job矩阵；没有重启、覆盖或干预修复版及Phase1。
- fixopt当前先并行运行8个DRIFT候选，随后各GPU顺序进入RIEI Table III行；最终仍必须按同一run、同一receiver组合和论文last-N口径判定。
