# DRIFT/RIEI原论文正式复现实验报告

## 基本信息

- 实验ID：`paper_repro_original_riei_drift_seed1337_20260714_090706`
- 时间：2026-07-14
- 操作者：Codex
- 目标：在N607运行DRIFT原论文Table I协议与RIEI原论文Table III的12个receiver组合，检查修正后的paper-faithful实现是否达到论文结果。
- 声明边界：本实验只属于论文原始复现，不是CVSStage2、CVS部署证据或卫星/LEO评估；训练与评估均禁用satellite augmentation/evaluation。

## 假设与比较目标

此前早期paper-window结果明显偏低，但当时存在receiver标签空间、DRIFT raw negative-MSE、domain discriminator层数、center统计、RIEI损失reduction和last-N窗口等偏差。本轮使用已修正的paper-scope路径，检验偏差修复后能否接近原论文。

|方法|原论文协议|论文目标|本轮判定口径|
|---|---|---:|---|
|DRIFT|Day1；训练receiver`1-1,14-7,7-7`；测试7个未见receiver；每TX/RX训练800、测试200；最后5个epoch平均|七receiver平均`75.62%`|`drift_last5`与论文平均差绝对值不超过3pp视为单seed复现通过，同时保留7个receiver同一run的逐receiver结果|
|RIEI|WiSig Table III；两个source receiver训练、一个held-out receiver测试；12个组合；最后10个epoch窗口|12行论文均值分别为`77.88,79.43,66.09,70.51,77.35,75.48,71.91,68.33,73.54,73.52,72.05,73.46%`|逐行与论文同一receiver组合比较；单seed初判要求至少10/12行落入论文均值±2个论文标准差且12行MAE不超过3pp；论文报告均值±标准差，严格复现结论仍需后续多seed复核|

## 数据与配置

- 数据：`Dataset_WigSig/ManySig.pkl`
- N607数据SHA256：`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`
- seed：`1337`
- epoch：`200`
- batch size：`64`
- DRIFT：Adam，`lr=0.0001`，`lambda_grl=1.0`，`grl_coeff=1.0`，two-layer domain discriminator，EMA receiver center，`lambda_center=0.01`，raw negative-MSE，`lambda_mse=0.02`，last5。
- RIEI：FED/EC/RC交替训练，`lr_all=lr_fed=0.0001`，`lambda_mi=lambda_ie=1.2`，CE/MI/IE均使用`sum` reduction，last10。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/paper_repro_original_riei_drift_seed1337_20260714_090706`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/runs/paper_repro_original_riei_drift_seed1337_20260714_090706`
- 预期输出：每个job的`metrics.json`、训练日志、paper-window summary、scheduler manifest和PID表。

## 本地版本与验证

- 根目录`E:\type10-7`不是Git仓库。
- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`，当前分支`codex/cvs-rffi-release-20260626`；工作树已有与本任务无关的用户改动，本轮只暂存/提交本实验新增launcher和报告。
- 新增launcher：`code/scripts/launch_paper_repro_original_matrix_20260714.sh`。
- Git镜像：`github_publish/CVS-RFFI-repo/code/scripts/launch_paper_repro_original_matrix_20260714.sh`。
- 本地测试：`python -m py_compile ...`通过；`python -m unittest tests.test_drift_table1_paper_parity`为`12/12 OK`。
- 新launcher验证：根目录与Git镜像`bash -n`均通过；dry-run展开`13`个job，其中DRIFT`1`个、RIEI`12`个，GPU计划计数为`2,2,2,2,2,1,1,1`，全部满足每GPU最多2个训练。
- launcher SHA256：`6924da94fde2a7098e778211e0adb7c36252950d9c1fec444c4513fade209a97`。
- 非Git根目录快照：`E:\type10-7\code\snapshots\paper_repro_original_riei_drift_seed1337_20260714_090706\launch_paper_repro_original_matrix_20260714.sh`，SHA256与launcher一致。
- 同步映射：`E:\type10-7\code\scripts\launch_paper_repro_original_matrix_20260714.sh`→`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_paper_repro_original_matrix_20260714.sh`；本报告→N607同名`automation_reports/CV-SincNet/.../report.md`。
- 待完成：Git提交、SCP、远端hash/语法/dry-run和正式启动。

## GPU分配与安全边界

N607只读preflight于2026-07-14 09:03+08:00通过；8张RTX3090均空闲，训练inventory无活动训练进程，磁盘剩余约7.6TB。计划13个job，任一GPU不超过2个训练：GPU0运行DRIFT+1个RIEI，GPU1-4各2个RIEI，GPU5-7各1个RIEI。launcher在正式启动前再次按`nvidia-smi`执行`current+planned<=2`硬门禁，且拒绝覆盖已存在的run/log根。

## 风险与完成后检查

- RIEI论文Table III报告均值±标准差，单seed只能做初步复现判断；若单seed接近论文，将追加多seed复核。
- 论文未公开全部训练随机性与validation细节；即使均值有差距，也必须先检查完整训练日志、loss稳定性、数据计数与同一行receiver组合，不能只比较孤立最大值。
- 完成后逐job提取最后epoch、best epoch、paper-window均值/标准差、同一行receiver split与最终判定；DRIFT保留7个目标receiver同一run的完整行上下文，RIEI保留12行Table III逐行上下文。

## 启动状态

### 正式提交

- 启动时间：2026-07-14 09:12:09+08:00。
- 远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- Python环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch`2.1.0+cu121`，CUDA可用。
- 精确提交命令：`ssh -F tools\n607_ssh_config -o BatchMode=yes N607 'cd /home/szu2070436088/2510044040/CV-SincNet && bash code/scripts/launch_paper_repro_original_matrix_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2'`。
- scheduler PID与GPU：

|job|GPU|scheduler PID|
|---|---:|---:|
|`drift_table1_seed1337`|0|221630|
|`riei_rx1_1_rx7_7_to_rx1_19_seed1337`|1|221633|
|`riei_rx1_1_rx8_8_to_rx1_19_seed1337`|2|221639|
|`riei_rx1_1_rx14_7_to_rx1_19_seed1337`|3|221644|
|`riei_rx7_7_rx8_8_to_rx1_19_seed1337`|4|221652|
|`riei_rx7_7_rx14_7_to_rx1_19_seed1337`|5|221665|
|`riei_rx8_8_rx14_7_to_rx1_19_seed1337`|6|221674|
|`riei_rx1_1_rx1_19_to_rx14_7_seed1337`|7|221692|
|`riei_rx1_1_rx7_7_to_rx14_7_seed1337`|0|221707|
|`riei_rx1_1_rx8_8_to_rx14_7_seed1337`|1|221731|
|`riei_rx1_19_rx7_7_to_rx14_7_seed1337`|2|221755|
|`riei_rx1_19_rx8_8_to_rx14_7_seed1337`|3|221787|
|`riei_rx7_7_rx8_8_to_rx14_7_seed1337`|4|221818|

### 约4分钟启动健康检查

- 状态：`RUNNING_STARTUP_HEALTHY`，不是完成或论文结果复现结论。
- N607存在13个对应GPU Python训练进程；GPU0-4各2个，GPU5-7各1个，显存约`543-1074MiB/GPU`，未超过并发门禁。
- 13个训练日志均已进入epoch：DRIFT到epoch22；12个RIEI到epoch8-10。
- 未发现`Traceback`、`RuntimeError`、OOM、未知参数、`NaN`或`Killed`。
- DRIFT配置证据：`protocol=drift_day1`、train receiver`1-1,14-7,7-7`、7个held-out receiver、`domain_discriminator_layers=2`、`center_mode=ema`、raw negative-MSE、`paper_eval_last_n=5`。
- RIEI配置证据：12个不同Table III receiver组合、`ce_reduction=sum`、`mi_reduction=sum`、`ie_reduction=sum`、`paper_eval_last_n=10`。
- 所有job均记录`SAT_EVAL=0`，`[CONFIG-UNLABELED] route=none`；日志中出现的labeled/unlabeled默认比例字段未启用无标签训练路线。
- 本轮只确认提交落地与启动健康；是否复现论文结果必须等待200epoch及paper-window汇总后按同一receiver行比较。
- 已创建当前任务heartbeat`riei-drift`，每30分钟只读检查进度；运行期间不干预。全部完成后将拉取小型日志与metrics、分析完整训练日志、更新逐行结果表与复现结论，并停用heartbeat。

### 2026-07-14 09:48+08:00监控检查点

- 总体状态：`PARTIAL_RUNNING`。13个训练日志和13个增量`metrics.json`均存在；DRIFT已完成，12个RIEI仍在运行。
- 进程：N607当前有12个本矩阵GPU Python训练进程，对应12个RIEI job；DRIFT训练进程已随正常完成退出。
- RIEI进度：12个job完成epoch范围为98-102，最新已开始epoch范围为99-103，约完成一半。
- 健康性：13个训练日志均未发现`Traceback`、`RuntimeError`、OOM、未知参数、`NaN`或`Killed`；RIEI日志持续增长，无停滞证据。
- 本检查点未启动、终止、重启或修改任何远端训练。

DRIFT已得到正式paper-window结果，但整个13-job矩阵尚未完成：

|candidate|机制|receiver/TX split|seed|paper-window|论文目标|差值|final overall|逐receiver final accuracy|当前判定|
|---|---|---|---:|---:|---:|---:|---:|---|---|
|`drift_table1_seed1337`|DRIFT；ResNet18-1D；TX/RX拆分；GRL；EMA center；raw negative-MSE|Day1；train RX=`1-1,14-7,7-7`；test RX=`1-19,19-2,2-1,2-19,20-1,7-14,8-8`；6 TX；每TX/RX train=800、test=200|1337|`49.37±3.04%`，last5|`75.62%`|`-26.25pp`|`51.71%`|`1-19=52.67%`；`19-2=59.58%`；`2-1=59.17%`；`2-19=43.08%`；`20-1=36.42%`；`7-14=60.75%`；`8-8=50.33%`|`NOT_REPRODUCED`；单seed严格paper-window明显未达到论文Table I|

DRIFT的best-val触发测试最高曾到`62.88%`，仍低于论文`75.62%`，且不能替代论文规定的last5结果。最终总矩阵结论仍等待12个RIEI Table III行完成。

## 论文一致性修复追踪表

本表在代码修改前建立。来源页码按PDF页序记录；“验证”同时要求公式/算法映射、代码单测和后续N607重跑证据。

|ID|来源|论文要求|实现目标|状态|验证|备注|
|---|---|---|---|---|---|---|
|`DRIFT-CTR-01`|DRIFT Eq.(25)、Algorithm 1，PDF第7-8页|每个mini-batch按receiver domain即时计算中心|`baselines/drift/losses.py`、`train_cvs.py`、paper launcher|`implemented`|精确数值单测+dry-run已通过；待N607重跑|当前正式run误用了跨batch EMA center；本地默认与paper launcher已改为`batch`|
|`DRIFT-CTR-02`|DRIFT Eq.(25)，PDF第7页|各receiver domain的域内均值再对domain求和|`receiver_style_transfer_center_loss`|`verified`|两domain手算单测通过|已由domain mean改为domain sum；手算期望`1+4=5`|
|`DRIFT-MSE-01`|DRIFT Eq.(26)，PDF第7页|逐样本对特征维求平方和，再对batch求均值并取负|`negative_mse_separation(reduction="sum")`|`verified`|现有精确数值单测|raw negative-MSE本身与论文一致，不改为feature mean或归一化MSE|
|`DRIFT-GRL-01`|DRIFT Algorithm 1、实现细节，PDF第8页|two-layer domain discriminator；GRL符号与`lambda_1=1`|model/loss/launcher|`verified`|结构与梯度单测|当前实现一致|
|`DRIFT-PROTO-01`|DRIFT Table I与实现细节，PDF第8-9页|Day1、指定3个source receiver/7个target receiver、Adam`1e-4`、batch64、200epoch、last5|split/launcher/evaluator|`verified`|dry-run+当前run配置/计数|当前run协议与计数一致|
|`RIEI-ALT-01`|RIEI Eq.(10a-c)、Eq.(11)，PDF第3页|第一步用CE更新FED/EC/RC；第二步冻结EC/RC，仅用`lambda_MI L_MI-lambda_IE L_IE`更新FED|`alternating_training_step`|`verified`|训练步骤代码审计+参数差分单测|改变MI/IE只改变FED第二步结果，不改变已冻结EC/RC；同时修复`receiver_target`存在时仍急切求值fallback的批字段错误|
|`RIEI-LOSS-01`|RIEI Eq.(2-9)，PDF第3页|CE/MI/IE按论文求和，`lambda_MI=lambda_IE=1.2`|loss/launcher|`verified`|精确数值单测+当前run配置|当前paper launcher已显式使用`sum` reductions|
|`RIEI-ARCH-01`|RIEI模型与实现细节，PDF第2-4页|WiSig使用ResNet1D-18 FED、512维拆成两个256维特征、EC/RC为三层FC|architecture/model|`verified`|结构代码审计|当前实现一致|
|`RIEI-DATA-01`|RIEI实验设置，PDF第4页|WiSig去除无信号段并使用equalized数据；每receiver train/test计数与Table I一致|dataset/split/launcher|`verified`|`wisig_equalized=1`、split计数和当前run日志|`ManySig.pkl`按equalized键读取；train=14400、test=4800/receiver，额外val只用于监控，不改变paper last10|
|`LOG-DRIFT-01`|训练日志分析硬门禁|完整读取200epoch曲线及所有组件|本地日志/metrics分析|`verified`|200/200 epoch与异常扫描|完整日志801行；无硬错误；见下节发散证据|
|`LOG-RIEI-01`|训练日志分析硬门禁|完整读取12个200epoch日志及metrics|本地日志/metrics分析|`blocked`|12个job完成后分析|当前12个训练仍在运行，禁止干预|
|`REMOTE-REPAIR-01`|AGENTS.md/N607安全规则|本地修复、测试、快照、Git提交后才同步；活动job期间不覆盖|local/Git/N607|`blocked`|hash+SCP+远端dry-run|等待当前12个RIEI自然结束|

## DRIFT完整训练日志诊断与本地修复

已拉取并完整读取DRIFT训练日志801行及`metrics.json`的200/200个epoch。日志没有`Traceback`、`RuntimeError`、OOM、CUDA error、`NaN`或`Killed`。训练不是进程故障，而是paper-faithful目标实现偏差叠加raw negative-MSE的无界尺度增长：

|epoch|train loss|val TX|TX CE|RX CE|center loss|negative-MSE|TX feature norm|RX feature norm|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|-54.66|37.03%|1.2910|0.9811|1490.37|-3642.32|21.64|21.21|
|20|-212.79|99.36%|0.0646|0.0005|440.58|-10895.38|34.69|57.75|
|100|-1545.33|99.33%|0.0438|0.0007|3107.06|-78849.08|158.25|166.38|
|200|-4858.97|96.89%|0.1052|0.0245|10762.28|-248361.60|310.56|311.32|

当前run的`center_mode=ema`不符合DRIFT Algorithm 1按mini-batch计算`c_d`的要求；同时中心项对3个receiver domain取均值，使Eq.(25)的中心约束再缩小3倍。修复后使用当前mini-batch receiver center并对domain求和。Eq.(26)仍严格保留“特征维平方和、batch均值、取负”的raw negative-MSE；不把论文复现偷偷改成归一化、cap或feature-norm正则。由于原公式的negative-MSE存在无界尺度风险，是否仅靠修正后的Eq.(25)足以恢复论文结果必须通过新的隔离run验证，不能用本地单测宣称性能已修复。

本地验证：Git承载面`pytest tests/test_drift_eq25_paper_parity.py`为`2 passed`；根目录`python -m unittest tests.test_drift_table1_paper_parity`为`13/13 OK`；两个launcher均通过`bash -n`，paper-scope dry-run明确展开`--center_mode batch`且不再传`--center_momentum`。

RIEI参数差分单测进一步确认Eq.(10)-(11)的交替训练边界：CE阶段更新FED/EC/RC；disentanglement阶段冻结EC/RC，仅FED随MI/IE变化。审计时同时发现`batch.get("receiver_target", batch["receiver"])`会急切读取fallback字段，现已改为显式条件分支；当前N607数据批同时含两个字段，故该错误不是本轮低性能原因，但修复后paper训练函数可正确接受仅提供compact `receiver_target`的批。
