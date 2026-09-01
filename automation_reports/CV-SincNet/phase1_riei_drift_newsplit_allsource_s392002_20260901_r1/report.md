# RIEI/DRIFT全source receiver Phase1实验报告

- run_id：`phase1_riei_drift_newsplit_allsource_s392002_20260901_r1`
- 当前状态：`ANALYZED`
- 候选矩阵：RIEI=`RIEI_C06_sum_featnorm1e4`、DRIFT=`DRIFT_N02_raw_cap4000`，各1行。
- 冻结代码提交：`2df2a33689fcd75587424e68afec44c0e13015d7`
- 本地环境/CWD：`ssr-gpu`；`E:\type10-7\github_publish\CVS-RFFI-repo`
- N607环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；不可变release目录`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1_2df2a336`
- 数据输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1`
- GPU：RIEI→GPU0；DRIFT→GPU1。
- 停止规则：固定200epoch，不按updates或性能提前终止；仅在数据/query越权、错误seed/day/receiver、输出碰撞、错误checkout、无prediction闭合、launcher级故障或两行出现相同确定性异常时，精确绑定并停止本run进程树，保留全部产物。
- 预期artifact：每行`best_by_val.pt`、`metrics.json`、clean与`leo_clear_weak`/`leo_low_elev_weak`/`leo_rain_weak`逐场景结果及日志；run根包含`matrix_manifest.json`和`launch_receipt.json`。

## 冻结矩阵与协议

|方法|seed|L_s/U_s/V|训练source receivers|GPU|
|---|---:|---|---|---:|
|RIEI|392002|0.07/0.63/0.30|[1,3,4,6,8]|0|
|DRIFT|392002|0.07/0.63/0.30|[1,3,4,6,8]|1|

两行均不使用fold或`--wisig_source_holdout_rxs`。训练使用day1–day3、单一source验证集`V=0.30`、200epoch和source V选模。星地增强固定为真实`clean+satellite`拼接CE-only，`lambda_sat_cls=0.68`、`lambda_sat_cons=0`，E80开始卫星辅助CE，使用三阶段`LEO_WEAK`课程。checkpoint冻结后才测试全部目标receiver、day1–day4的clean和三种LEO场景；目标结果不得反馈选种、调参或重训。

## 启动命令

```text
PYTHONPATH=<release>/code:<release> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m cvsrffi.phase1_baseline_fold_matrix --run-id phase1_riei_drift_newsplit_allsource_s392002_20260901_r1 --project-root <release> --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1 --log-root /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1 --python-bin /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --gpu-ids 0,1 --all-source --execute
```

## 本地验证与审查

- TDD：全source构造测试先因功能不存在失败，实施后通过；GPU默认P1测试先复现失败，修复后通过。
- 聚焦测试：`8 passed`；Python编译和`git diff --check`通过。
- 独立P0/P1审查发现并修复1个全source默认GPU数量P1；仅针对原问题定点复审后`READY`。
- 命令级dry-run：2行，完整source receivers均为`[1,3,4,6,8]`，无holdout参数。
- Git push与远端OID回读：`VERIFIED`，远端`work/cvs-active`=`2df2a33689fcd75587424e68afec44c0e13015d7`。

## Release、smoke与正式启动

- release映射：本地`E:\type10-7\release_archives\phase1_riei_drift_newsplit_allsource_s392002_20260901_r1_2df2a336.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_riei_drift_newsplit_allsource_s392002_20260901_r1_2df2a336.tar.gz`。
- 唯一归档SHA256本地/远端一致：`b2c5cfd161d4d716c88a8da23e4a5974bc5da6248e7d7be0cc40bb5daeff90c5`；远端解包和关键入口编译`PASS`。
- N607预检：普通账户直连`PASS`；GPU0/1空闲；新release、run、log路径启动前均不存在。
- 真实checkpoint无query smoke：`PASS`。RIEI_C06的epoch186 checkpoint载入132个有限tensor；DRIFT_N02的epoch76 checkpoint载入136个有限tensor；未读取数据集或目标query。
- 正式启动时间：2026-09-01约16:45 CST。
  - RIEI：PID`3637386`，GPU0，row=`RIEI_ALLSRC_S392002`。
  - DRIFT：PID`3637387`，GPU1，row=`DRIFT_ALLSRC_S392002`。
- 启动后回读：两个PID的PPID/PGID/SID、CWD、完整cmdline、run-root和`CUDA_VISIBLE_DEVICES`绑定正确；两行均已完成E1并进入E2，日志增长且已写出E1的`best_by_val.pt`；未见`Traceback`、`RuntimeError`、CUDA OOM、`Killed`或`AssertionError`。

## 完成状态与checkpoint

- 两行均完成固定200epoch，未按updates或中间性能提前终止。
- DRIFT结构化结果完成时间为2026-09-01 17:15:42 CST；RIEI为2026-09-01 17:21:27 CST。相对约16:45的启动时间，单行墙钟时间约31–36分钟。
- 完成核对时原PID`3637386`和`3637387`均已退出，GPU0–GPU3空闲；这与正常完成相符，不是训练中断。
- RIEI由source端`V=0.30`选中E180，`98.5296%`（26603/27000）；DRIFT选中E191，`98.2481%`（26527/27000）。两行最终测试均明确记录`checkpoint_source=best_by_val`和`last_epoch_tested=false`。
- checkpoint冻结后才执行目标测试；目标接收机结果未用于checkpoint选择、调参或重训。

## 目标域总体结果

准确率单位为%，括号中为正确数/样本数。每个clean或LEO场景均为7个目标接收机×4天×6个发射机×1000个物理样本，共168000个样本。

|方法|source V|clean|LEO clear weak|LEO low-elev weak|LEO rain weak|三LEO均值|三LEO最差|
|---|---:|---:|---:|---:|---:|---:|---:|
|RIEI|98.5296（26603/27000）|54.1780（91019/168000）|54.4893（91542/168000）|53.5810（90016/168000）|53.2208（89411/168000）|53.7637|53.2208|
|DRIFT|98.2481（26527/27000）|69.1470（116167/168000）|60.2167（101164/168000）|58.7643（98724/168000）|58.9762（99080/168000）|59.3190|58.7643|
|DRIFT−RIEI|-0.2815|+14.9690|+5.7274|+5.1833|+5.7554|+5.5554|+5.5435|

本次同seed、同物理划分的首轮比较中，DRIFT在clean、三种LEO逐场景、LEO均值和LEO最差值上均高于RIEI。DRIFT的clean优势最大，为`+14.9690pp`；LEO条件下优势收窄至`+5.18～+5.76pp`。相对各自clean，RIEI三LEO均值变化为`-0.4143pp`，DRIFT为`-9.8280pp`：DRIFT的绝对准确率更高，但从clean到LEO的相对退化明显更大。

## clean时间切分

|方法|day1–day3目标接收机|day4目标接收机|day4−day1–day3|
|---|---:|---:|---:|
|RIEI|55.7437（70237/126000）|49.4810（20782/42000）|-6.2627|
|DRIFT|69.4929（87561/126000）|68.1095（28606/42000）|-1.3833|

DRIFT的clean跨日保持更稳定。现有clean结构化结果将day1–day3作为一个合并测试切片、day4单列，因此不能从既有artifact中无损拆出day1、day2、day3各自的clean准确率；下表的LEO结果则保存了逐日明细。

## 逐目标接收机结果

每个接收机的clean和每种LEO场景均含24000个样本。`LEO均值`为三个场景准确率的算术平均。

### RIEI

|receiver|clean|clear weak|low-elev weak|rain weak|LEO均值|
|---|---:|---:|---:|---:|---:|
|0（1-1）|52.04|51.41|51.22|50.95|51.19|
|2（14-7）|40.75|48.30|47.43|47.41|47.72|
|5（2-1）|68.01|55.98|55.65|54.90|55.51|
|7（20-1）|40.10|33.70|34.24|34.03|33.99|
|9（7-14）|55.38|69.24|66.81|66.53|67.53|
|10（7-7）|54.18|55.96|54.84|54.29|55.03|
|11（8-8）|68.78|66.85|64.88|64.44|65.39|

### DRIFT

|receiver|clean|clear weak|low-elev weak|rain weak|LEO均值|
|---|---:|---:|---:|---:|---:|
|0（1-1）|75.61|61.62|60.59|60.81|61.01|
|2（14-7）|50.58|50.01|49.48|50.33|49.94|
|5（2-1）|81.88|66.53|64.76|64.43|65.24|
|7（20-1）|51.84|41.60|41.67|41.95|41.74|
|9（7-14）|73.67|70.33|67.84|68.02|68.73|
|10（7-7）|77.34|62.20|59.98|60.02|60.73|
|11（8-8）|73.10|69.22|67.04|67.28|67.85|

两个方法共同的最弱目标接收机是receiver7（20-1）：RIEI的LEO均值为`33.99%`，DRIFT为`41.74%`。DRIFT在7个接收机的clean和三LEO均值上均高于RIEI，但receiver2、9、11上的LEO优势较小，说明总体优势并非所有物理接收机上等幅出现。

## 三种LEO逐日结果

每个日期、每个场景均含42000个样本。

### RIEI

|日期|clear weak|low-elev weak|rain weak|LEO均值|
|---|---:|---:|---:|---:|
|day1（2021-03-01）|56.46|54.94|55.16|55.52|
|day2（2021-03-08）|55.77|55.01|54.23|55.00|
|day3（2021-03-15）|51.51|51.22|50.69|51.14|
|day4（2021-03-23）|54.22|53.15|52.81|53.40|

### DRIFT

|日期|clear weak|low-elev weak|rain weak|LEO均值|
|---|---:|---:|---:|---:|
|day1（2021-03-01）|60.78|58.90|58.91|59.53|
|day2（2021-03-08）|63.55|61.29|61.52|62.12|
|day3（2021-03-15）|58.28|57.42|57.90|57.87|
|day4（2021-03-23）|58.26|57.45|57.58|57.76|

RIEI的LEO时间最低点为day3的`51.14%`；DRIFT最低点为day4的`57.76%`，与day3仅差`0.11pp`。DRIFT在四天的LEO均值上均优于RIEI。

## 三种LEO逐发射机结果

每个发射机、每个场景均含28000个样本。

### RIEI

|transmitter|clear weak|low-elev weak|rain weak|LEO均值|
|---|---:|---:|---:|---:|
|0（14-10）|67.52|65.68|66.17|66.46|
|1（14-7）|22.11|22.55|22.46|22.37|
|2（20-15）|73.47|73.01|72.52|73.00|
|3（20-19）|52.24|51.26|49.77|51.09|
|4（6-15）|47.58|45.94|45.27|46.26|
|5（8-20）|64.01|63.05|63.14|63.40|

### DRIFT

|transmitter|clear weak|low-elev weak|rain weak|LEO均值|
|---|---:|---:|---:|---:|
|0（14-10）|69.39|67.65|67.90|68.31|
|1（14-7）|26.80|26.77|26.94|26.84|
|2（20-15）|59.18|58.03|58.59|58.60|
|3（20-19）|62.17|60.37|59.86|60.80|
|4（6-15）|63.86|60.10|61.04|61.66|
|5（8-20）|79.90|79.67|79.53|79.70|

两种方法共同的主要类别短板是transmitter1（14-7），LEO均值分别只有`22.37%`和`26.84%`。RIEI并非逐发射机都弱于DRIFT：在transmitter2（20-15）上，RIEI的LEO均值为`73.00%`，高于DRIFT的`58.60%`；DRIFT的总体优势主要来自transmitter3、4、5及多个接收机上的更高准确率。

## 最差细胞与训练健康性

- 每个`receiver×transmitter×day×scene`细胞含1000个样本。RIEI最差细胞为`receiver7（20-1）×transmitter1（14-7）×day3×leo_clear_weak`，准确率`0.30%`（3/1000）；DRIFT最差细胞为`receiver0（1-1）×transmitter1（14-7）×day2×leo_clear_weak`，准确率`1.70%`（17/1000）。这进一步定位到transmitter1的强条件性失效，不能由总体均值掩盖。
- 两行`metrics.json`均有连续200条epoch记录；日志分别有200个epoch START标记，首末为E1/E200。
- 拼接式星地辅助CE在两行均于E80首次非零，符合冻结课程；最终三种LEO测试标记各出现1次。
- RIEI验证准确率由E1的`92.4815%`到E200的`98.2370%`，全程最佳为E180的`98.5296%`；DRIFT由E1的`26.8852%`到E200的`98.2111%`，最佳为E191的`98.2481%`。
- 结构化epoch数值中未发现NaN或Inf；完整日志未发现`Traceback`、`RuntimeError`、CUDA OOM、`Killed`或`AssertionError`。

## Artifact闭合与结论边界

- 每行均存在远端`best_by_val.pt`、`metrics.json`、`satellite_detailed_metrics.csv`和完整日志。RIEI checkpoint大小16526090字节；DRIFT为16797010字节。两个卫星明细CSV均含834行，覆盖三场景、两测试切片及receiver/transmitter/day细分。
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`均已产生完整总体计数和细分结果，故本run状态从`RUNNING`推进为`ARTIFACTS_COMPLETE`并完成本报告分析，最终记为`ANALYZED`。
- 结论严格限于当前单seed（392002）、固定RIEI/DRIFT版本、相同新Phase1物理划分和全source receiver设置。当前证据支持“本行DRIFT总体优于本行RIEI”，不等价于多seed显著性结论，也不应用于目标结果反馈选种或重训。
