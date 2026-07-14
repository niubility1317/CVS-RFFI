# DRIFT Eq.(25)论文一致性修复验证报告

## 基本信息

- 实验ID：`paper_repro_drift_eq25_batch_seed1337_20260714_100800`
- 时间：2026-07-14
- 操作者：Codex
- 目标：隔离验证DRIFT Eq.(25)的mini-batch receiver center与domain-sum修复能否恢复论文Table I结果。
- 声明边界：仅为DRIFT原论文WiSig Day1复现，不是CVSStage2、卫星/LEO或部署证据。

## 假设与比较目标

上一正式run误用EMA center并对receiver domain取均值，last5仅`49.37±3.04%`，低于论文`75.62%`。本run只修复论文Eq.(25)：每个mini-batch按receiver即时计算center，并对domain loss求和；Eq.(26)继续使用raw negative-MSE，其他模型、数据、seed与超参数不变。单seed last5与论文平均差绝对值不超过3pp时初判复现通过。

## 配置与服务器计划

- 数据：`Dataset_WigSig/ManySig.pkl`，N607已验证SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- 协议：`drift_day1`；train RX=`1-1,14-7,7-7`；test RX=`1-19,19-2,2-1,2-19,20-1,7-14,8-8`；6 TX；train=800、val=200、test=200/每TX-RX。
- 训练：seed=1337、200epoch、batch64、Adam、lr=`1e-4`、`lambda_grl=1`、`grl_coeff=1`、two-layer domain discriminator、`lambda_center=0.01`、`center_mode=batch`、`lambda_mse=0.02`、raw negative-MSE、last5。
- 工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- 输出：`paper_reproduction/runs/paper_repro_drift_eq25_batch_seed1337_20260714_100800`。
- 日志：`paper_reproduction/logs/paper_repro_drift_eq25_batch_seed1337_20260714_100800`。
- GPU：待当前12个RIEI训练全部自然结束后，以实时process/CWD/cmdline/GPU证据选择空闲GPU；不与当前run重叠，不干预当前任务。

## 本地修改、验证与同步边界

|文件|修改|本地验证|N607目标|
|---|---|---|---|
|`baselines/drift/losses.py`|Eq.(25)由domain mean改为domain sum|两domain手算单测期望`5.0`|同路径|
|`baselines/drift/train_cvs.py`|paper默认center由EMA改为mini-batch|parser单测|同路径|
|`run_wisig_paper_scope_queue.sh`|DRIFT paper命令显式`--center_mode batch`并移除EMA momentum|`bash -n`+dry-run|同路径|
|`baselines/riei_fd/train.py`|修复`receiver_target`fallback急切求值|RIEI参数差分单测|同路径；不改变当前run语义|

Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，提交`73f694a`与`90f81e8`。根目录不是Git仓库；同步前必须创建`E:\type10-7\code\snapshots\paper_repro_drift_eq25_batch_seed1337_20260714_100800\`快照并记录hash。活动RIEI训练结束前禁止SCP覆盖或启动本run。

## 完成后检查

- 完整读取200epoch日志与metrics，不以tail或best-val代替last5。
- 检查CE、GRL、center、negative-MSE、feature norm曲线及硬错误。
- 报告同一run七个receiver的last5与final结果、论文差值和联合判定。
- 若raw negative-MSE仍产生无界feature norm，只能报告论文目标本身的数值稳定性风险；归一化、cap或正则版本必须另标`fix/diagnostic`，不得伪装成原论文结果。
