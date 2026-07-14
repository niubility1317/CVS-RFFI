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

尚未启动；报告先于远端状态变更创建。
