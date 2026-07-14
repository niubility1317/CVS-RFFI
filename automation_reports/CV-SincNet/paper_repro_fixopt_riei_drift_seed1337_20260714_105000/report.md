# RIEI/DRIFT修复语义与fix_optimized联合优化实验

## 实验定义

- 实验ID：`paper_repro_fixopt_riei_drift_seed1337_20260714_105000`
- 时间：2026-07-14 10:50+08:00
- 操作者：Codex
- 状态：`RUNNING_HEALTHY`
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

## 直接启动请求与安全排队

- 2026-07-14 11:07再次执行N607预检：GPU0-7仍各有2个compute process，分别来自Phase1和修复版复现；直接新增训练会达到3/GPU，违反默认硬上限。
- 根据用户“直接启动”要求，新增本地验证的`code/scripts/defer_launch_paper_repro_fixopt_20260714.sh`：它只等待指定修复版run的训练及queue进程全部退出，最长等待21600秒；随后调用原20-job launcher。真正启动前仍由launcher重新执行每GPU`current+planned_peak<=2`容量门，超限则失败关闭，不会强行启动。
- deferred launcher的`bash -n`和dry-run通过；Git提交：`20d5590 queue fixopt matrix behind repaired reproduction`。

## N607同步与安全启动提交

- 2026-07-14 11:10同步前核对：N607现有`run_wisig_paper_scope_queue.sh` SHA256=`5f93bc7c...`，与修复版启动时本地快照完全一致，未覆盖未知远端改动。
- 已同步并核对本地/远端SHA256：paper-scope queue=`26414994...`、20-job launcher=`73c41afe...`、deferred launcher=`9245f60d...`。
- 远端三个脚本均通过`bash -n`；20-job dry-run确认8个DRIFT候选、RIEI Table III完整12行及8张GPU顺序队列。
- 当前每GPU仍有Phase1和修复版复现各1个compute process，不能直接新增训练。已提交容量受控的deferred launcher：PID=`289073`，父包装PID=`289071`；日志=`paper_reproduction/logs/deferred_launchers/paper_repro_fixopt_riei_drift_seed1337_20260714_105000.out`。
- deferred launcher每30秒只读检查指定修复版进程；修复版完全退出后才调用20-job launcher，并再次执行`current+planned_peak<=2`硬容量门。11:12日志显示仍在等待，尚未占用额外GPU。
- 首次SSH提交因远端后台父shell保持channel导致本地命令超时；已立即终止残留本地`ssh.exe`并确认TCP22连接为0，随后通过只读远端证据确认deferred launcher已经landed，未重复提交。

### 2026-07-14 11:42+08:00等待检查点

- deferred PID=`289073`及父包装PID=`289071`均存活；等待日志持续更新，最新`active_target_processes=16`。
- 前序修复版尚有8个训练进程，所有GPU仍为2个compute process，因此fixopt尚未实际启动，符合容量门设计。
- 前序run无硬错误，预计首批剩余RIEI即将完成；deferred launcher继续等待全部13个job及queue退出。

### 2026-07-14 12:12+08:00等待与前序wrapper异常

- deferred PID=`289073`仍健康等待；前序run剩余4个RIEI训练到epoch72-74，fixopt尚未启动。
- 11:10同步共享paper-scope launcher时，前序长时间shell在训练返回后受到原位文件替换影响，8个已完整训练的RIEI出现wrapper status=2；训练日志、last10、final和metrics均完整，未重启或覆盖。
- 该异常不会改变deferred条件：只有前序所有训练与queue进程退出后才进入fixopt容量门。当前远端共享launcher已通过hash及`bash -n`验证，后续fixopt将从完整新文件启动。

### 2026-07-14 12:42+08:00等待检查点

- deferred PID=`289073`及父包装PID=`289071`均存活，最新等待状态`active_target_processes=8`。
- 前序修复版已完成9/13个训练artifact，剩余4个RIEI到epoch159-161；训练硬错误为0，既有wrapper异常没有扩展为训练失败。
- fixopt run目录尚不存在，说明20-job矩阵尚未实际启动；全机当前12个GPU compute process均属于Phase1或前序修复版，没有额外占用。
- 容量门继续等待前序所有目标进程退出，本检查点未干预任何远端进程或产物。

### 2026-07-14 12:58–13:12+08:00正式启动与健康检查

- 前序修复版13/13个训练artifact均完成200epoch且目标进程全部退出后，deferred launcher重新执行容量门并于12:58:11自动提交20-job矩阵；deferred PID随后正常退出。
- 8个per-GPU queue PID=`334756,334759,334761,334766,334772,334778,334787,334797`均存活；每张GPU当前恰有1个Phase1训练＋1个fixopt训练，共16个compute process，未超过每卡2个上限。
- 首批8个DRIFT候选全部启动，13:12完整读取当前日志后进度为epoch114–119；queue已开始job=8、完成job=0，12个RIEI Table III job仍在各GPU顺序队列等待。
- 8份DRIFT日志完整扫描未发现`Traceback`、`RuntimeError`、CUDA OOM、`Killed`、参数错误或loss/metric NaN/Inf；metrics持续写入，GPU显存约4084–4154MiB/GPU。
- 前序修复版最终结果已确认：DRIFT last5=39.99±3.13%，低于论文75.62%且较修复前下降9.38pp；RIEI Table III 12行平均53.99%、MAE=20.26pp、±2SD命中1/12。两者均未复现，fixopt继续运行具有明确诊断依据。
- 本检查点仅只读核验，没有干预、重启、覆盖fixopt或Phase1。

### 2026-07-14 13:42+08:00只读监控检查点

- 8个DRIFT候选均已自然完成200epoch并以queue status=0结束；8张GPU已各自顺序进入首批8个RIEI Table III job，当前进度epoch52–54。
- 8个queue PID仍全部存活；当前本任务训练进程=8。GPU0-7均为1个Phase1训练＋1个本任务训练，共16个compute process，未超过每卡2个上限。
- 完整扫描当前16份训练日志：DRIFT 8份均有`PAPER-EVAL-SUMMARY`，RIEI 8份持续增长；未发现`Traceback`、`RuntimeError`、CUDA OOM、`Killed`、参数错误或loss/metric NaN/Inf。
- DRIFT阶段结果如下；last5最佳为D02的66.06±1.52%，仍比论文75.62%低9.56pp，尚未达到≤3pp阈值。best-val测试最高为D07的70.96%@epoch109，但不能替代论文last5口径。

|candidate|cap|λ_mse|λ_FN|last5|与论文差值|final|best-val epoch/test|epoch200 MSE/FN/z_tx|阶段判定|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|`D01`|3000|0.020|0|58.65±5.35%|-16.97pp|56.52%|90/67.08%|-2999.63/27.82/65.74|未达到|
|`D02`|3500|0.020|0|66.06±1.52%|-9.56pp|65.52%|100/60.77%|-3499.64/33.91/71.93|当前last5最佳|
|`D03`|4000|0.020|0|62.37±7.05%|-13.25pp|59.50%|115/68.95%|-3999.74/39.96/74.75|未达到|
|`D04`|4500|0.020|0|58.63±2.84%|-16.99pp|60.01%|154/64.60%|-4499.83/43.33/77.93|未达到|
|`D05`|4000|0.015|0|65.86±1.58%|-9.76pp|63.61%|52/62.71%|-3999.85/38.14/74.10|次优|
|`D06`|4500|0.015|0|59.41±1.79%|-16.21pp|58.58%|83/57.54%|-4499.90/44.87/78.41|未达到|
|`D07`|5000|0.015|0|61.04±3.19%|-14.58pp|60.75%|109/70.96%|-4999.64/48.85/81.42|仅best-val较高|
|`D08`|4000|0.020|1e-5|64.39±3.52%|-11.23pp|62.04%|73/56.86%|-3999.98/38.81/75.64|未达到|

cap成功阻止修复版中`loss_mse=-247240`、feature norm=967.22的无界放大，但D02/D05仍与论文存在约9.6pp差距。按照既定边界，等待RIEI 12行全部完成后再做完整联合分析，并以D02同row为中心设计下一轮局部优化；当前不干预运行。

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
