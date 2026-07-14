# RIEI/DRIFT修复语义与fix_optimized联合优化实验

## 实验定义

- 实验ID：`paper_repro_fixopt_riei_drift_seed1337_20260714_105000`
- 时间：2026-07-14 10:50+08:00
- 操作者：Codex
- 状态：`LOCAL_VERIFIED_DEFERRED_CAPACITY`
- 目标：在2026-07-14论文语义修复基础上恢复历史`fix_optimized`稳定性保护，使DRIFT Table I达到或接近论文`75.62%`，并完整还原RIEI论文Table III的12个receiver组合，而不是只复用DRIFT Table I中的RIEI单行结果。

## 假设与对照

|方法|假设|主要对照|
|---|---|---|
|DRIFT|batch receiver center、domain sum修复了Eq.(25)；raw negative MSE仍需有限幅度保护，`mse_cap`与`lambda_mse`联合搜索可避免特征范数失控并恢复跨receiver性能|论文Table I DRIFT=`75.62%`；历史`DRIFT_N02_raw_cap4000` final=`70.73%`、last5=`66.67±4.43%`；修复版无cap run|
|RIEI|共享FED optimizer修复Eq.(10c)/(11)更新状态；历史`lambda_feature_norm=1e-4`可稳定sum-reduction下的交替训练|RIEI论文Table III全部12行；历史`62.74%`仅是DRIFT Table I协议结果，不作为Table III复现证据|

## 数据与评价协议

- 数据：`Dataset_WigSig/ManySig.pkl`，WiSig equalized，输入`2×256`。
- seed：`1337`；训练`200epoch`；batch size=`64`。
- DRIFT：Day1，source RX=`1-1,14-7,7-7`，7个held-out receiver；每Tx/Rx训练800、验证200、测试200；论文口径last5。
- RIEI：论文Table III完整12行；每行两个source receiver、一个held-out receiver；四天；每Tx/Rx训练2400、验证800、测试800；论文口径last10。
- DRIFT初判成功：last5均值与`75.62%`绝对差≤3pp，并完整报告同一run的7receiver结果。
- RIEI初判成功：12行MAE≤3pp，且至少10/12位于论文对应行`±2SD`；必须逐行同receiver组合比较。
- best-val和target-oracle只能作为诊断，不能代替last5/last10论文口径。

## 实验矩阵

### DRIFT Table I八个联合修复候选

|候选|`mse_cap`|`lambda_mse`|`lambda_feature_norm`|固定语义|
|---|---:|---:|---:|---|
|D01|3000|0.020|0|batch center、domain sum、raw/sum MSE|
|D02|3500|0.020|0|同上|
|D03|4000|0.020|0|历史fix锚点＋新语义|
|D04|4500|0.020|0|同上|
|D05|4000|0.015|0|降低加权MSE强度|
|D06|4500|0.015|0|同上|
|D07|5000|0.015|0|历史高last5锚点＋新语义|
|D08|4000|0.020|`1e-5`|检查小型feature-norm guard是否降低末段波动|

### RIEI Table III

12行全部使用：CE/MI/IE sum reduction、共享FED optimizer、`lambda_mi=lambda_ie=1.2`、`lambda_feature_norm=1e-4`。矩阵覆盖目标RX=`1-19`的6种source组合和目标RX=`14-7`的6种source组合。

## 本地变更与版本面

- 根目录不是Git仓库；本任务改动必须镜像到`E:\type10-7\github_publish\CVS-RFFI-repo`后提交。
- 根目录变更：
  - `run_wisig_paper_scope_queue.sh`：增加向后兼容的RIEI/DRIFT稳定性参数环境入口，默认值保持原修复版不变。
  - `code/scripts/launch_paper_repro_fixopt_matrix_20260714.sh`：20-job、8个per-GPU顺序队列、硬容量门。
- Git承载面：
  - `code/scripts/launch_paper_repro_fixopt_matrix_20260714.sh`
  - `code/patches/run_wisig_paper_scope_queue_fixopt_env_20260714.patch`
  - `tests/test_paper_repro_fixopt_launcher.py`
  - 本报告镜像。

## N607容量与启动边界

- 2026-07-14 10:48只读inventory：每张GPU有1个`phase1_dgleo_corepath8_20260714`训练和1个`paper_repro_repaired_riei_drift_seed1337_20260714_103000`训练，共2个compute process/GPU，已达到默认上限。
- 本优化矩阵当前不得启动，也不得干预、终止或覆盖上述任务。
- 计划在修复版13个job全部完成且容量门允许后启动。8个DRIFT候选分别作为每张GPU的首个job，随后12个RIEI Table III行进入同一组per-GPU顺序队列；新增峰值始终为1/GPU。
- 计划远端根：
  - run：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/runs/paper_repro_fixopt_riei_drift_seed1337_20260714_105000`
  - log：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/paper_repro_fixopt_riei_drift_seed1337_20260714_105000`
- 计划命令：`bash code/scripts/launch_paper_repro_fixopt_matrix_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；cwd=`/home/szu2070436088/2510044040/CV-SincNet`。

## 本地验证结果

- `bash -n code/scripts/launch_paper_repro_fixopt_matrix_20260714.sh`：通过。
- `bash -n run_wisig_paper_scope_queue.sh`：通过。
- launcher dry-run：展开20个job；8个DRIFT候选完整；RIEI Table III目标RX=`1-19`六行、目标RX=`14-7`六行完整；GPU0-3各3个顺序job，GPU4-7各2个顺序job；新增峰值均为1。
- `conda run -n ssr-gpu python -m pytest tests/test_paper_repro_fixopt_launcher.py tests/test_riei_alternating_paper_parity.py tests/test_drift_eq25_paper_parity.py -q`：`5 passed`。
- 已建立根目录快照：`E:\type10-7\code\snapshots\paper_repro_fixopt_riei_drift_seed1337_20260714_105000\`。
- Git承载面提交：`109945b add RIEI Table III and DRIFT fixopt matrix`。
- 当前未同步或启动N607任务，原因是每GPU已有2个compute process；不会修改仍会被修复版后续队列读取的远端入口。

## 完成后必须检查

- 完整读取20份200epoch日志、metrics及scheduler/queue日志，不使用tail抽样代替完整分析。
- 检查OOM、NaN/Inf、Traceback、Killed、参数未生效、队列异常退出和GPU容量违规。
- DRIFT逐候选报告last5、final、best-val诊断、7receiver同run结果、loss MSE/center/GRL、特征范数及首个分歧epoch。
- RIEI按Table III论文顺序逐行报告论文均值/SD、复现last10均值/SD、差值、是否进入`±2SD`，以及12行MAE和命中率。
- 若DRIFT仍未达到≤3pp，下一轮只围绕最佳联合row局部搜索，不更改论文数据协议或用best-val替代论文口径。

## 风险

- `mse_cap`和feature-norm是工程稳定性保护，属于`fix_optimized`，不能标为严格paper-literal。
- 单seed与论文多次运行可能存在方差；达到阈值后仍需补多seed确认。
- 当前修复版尚在运行，本报告中的联合修复收益仍是假设，不是已完成结果。
