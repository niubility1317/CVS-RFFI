# Phase1 P1-RCRMD postfreeze v1实验报告

## 1.状态与目标

- 实验ID：`phase1_rcrmd_postfreeze_20260810_v1`
- 日期：2026-08-10
- 操作：主代理冻结方法、门和分析；唯一N607 Runner负责落地、42步执行、技术监控和小工件回收
- 当前状态：`ANALYZED / REJECT_P1_RCRMD_PERMANENT / NO_PHASE3_CAPABILITY_CLAIM`
- 训练证据：`phase1_rcrmd12_20260810_v1`已完成12/12臂技术合同，报告commit=`92646094a3da90632fb5c5dec2caadd2eb796892`
- 目标：对同一fold的C/G final checkpoint执行固定42步后冻结评估，判断P1-RCRMD是否同时满足known分类floor、LEO弱信道floor、整体不退化和source proxy连续几何双门。
- 结论边界：通过只能`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`；不得声称修复RX/day、真实unknown、多卫星协同或Phase3。

## 2.冻结实现与验证

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`e84c456049a8cd69938920923dc2e8129b578a8d`
- 独立actual-diff终裁：`P0=0/P1=0/ALLOW`，仅授权技术发布，不含性能或晋级签字。

|文件|本地SHA256|用途|
|---|---|---|
|`analysis/phase1_rcrmd_postfreeze_design_20260810.md`|`574444d497033df982b3ec09cc402b723fe5723f88dd856fbaa8741723a08339`|42步、门与证据边界|
|`code/export_phase1_rcrmd_features.py`|`a37b3cee9a5b2142e92a6d4393c74bdb85ca53b7d5df2a738ddead4429fabd04`|L/V/proxy专用clean导出|
|`code/export_phase1_rcrmd_leo_features.py`|`3d2bbc413e68c636db8349ea49b9fe196575cd169ab88114fd25d51e8cc0b96d`|三LEO source-only导出与绑定|
|`code/evaluate_phase1_rcrmd_postfreeze_pair.py`|`53ceb603454320307791fa413544c3958ca9ac7161f68ec4cfc8d345b0fc11ac`|Gaussian、pair门和F6原始工件复算|
|`code/scripts/launch_phase1_rcrmd_postfreeze_20260810.sh`|`d92883e83857f85dff00366eb6a0cd98b99aa581da8fd24a1cff9892d6cb7b73`|冻结42步launcher，Git mode100755|
|`code/tests/test_phase1_rcrmd_postfreeze.py`|`3d1f4708bea5091c7f045dd14cde80b9dfe761be360f58b3d60f1aa8f2b25284`|协议、篡改、数学与launcher测试|

本地`ssr-gpu`串行证据：

- `py_compile`：通过。
- RCRMD postfreeze focused：27/27。
- CAGM+RCRMD联合回归：65/65。
- `bash -n`：通过。
- dry-run：12 clean+12 LEO/binding+12 proxy+6 pair=42。
- source/LEO/proxy、1-row proxy、F6 summary/raw篡改、float32合法账本和material drift负测：通过。
- `git diff --check`：通过。

## 3.冻结42步与数据权限

每个12候选依次生成：

1. RCRMD clean NPZ：仅L拟合、V作为known、固定proxy作为unknown；U只重建并核hash，零forward、零persist。
2. source-only三LEO NPZ与binding：逐scenario闭合ManySig路径/SHA、source physical key及TX/RX/day。
3. 固定logits proxy JSON/CSV。
4. 每fold C/G pair；F6额外重读F1–F5原始clean/LEO/binding/proxy JSON+CSV+NPZ，核当前SHA并重算summary、delta和全部门。

Gaussian-NLL固定：

- float64 totalized-L2：正范数归一化，精确零向量映射0且保留；nonfinite fatal。
- 4类逐维ddof=1方差；class-equal pooled；`0.9*s2_c+0.1*s2_pool`；逐维floor=`1e-6`。
- 完整NLL与stable logsumexp；只用L fit，V/proxy零fit。
- 固定proxy：days=`2021_03_01,2021_03_08`；RX=`1-1,1-19,14-7,18-2,19-2,2-1`；seed=`7281148`；max/TX=400；total=400。

RCRMD特异技术绑定：receipt schema、C/G enabled+lambda、source receiver `0..6`、固定1/28、每场景28格/终态84格、共同physical/RX/class/scene n_rc与batch order、warm-start/head/class/split/新AdamW初态；C aux N/A/0，G active/loss/VJP/float32账本/terminal通过。G-only字段不得与C错误比较相等。

## 4.非补偿判定门

|门|冻结要求|
|---|---|
|clean四floor|6/6 fold，G每项≥C−2pp|
|LEO四floor|18/18 fold×scene，G每项≥C−2pp|
|fold overall|每fold三场景overall均值G−C≥0|
|global overall|全18格overall均值G−C≥0|
|proxy AUROC|每foldG−C>0，6/6|
|proxy u-gap|每fold`(proxy mean u−V mean u)`的G−C>0，6/6|

分类端点与proxy端点独立，任何一项不能补偿另一项。任一完整门失败即`REJECT_P1_RCRMD_PERMANENT`；不得调λ、seed、receiver、TX、场景、fold或重试。

## 5.N607冻结路径与命令

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd_postfreeze_20260810_v1_e84c4560`
- CWD：`<release>/code`
- training root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcrmd12_20260810_v1`
- postfreeze root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcrmd_postfreeze_20260810_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd_postfreeze_20260810_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd_postfreeze_20260810_v1_launcher.out`
- ManySig SHA：`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`

唯一启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd_postfreeze_20260810_v1_e84c4560/code && nohup env POSTFREEZE_RUN_ID=phase1_rcrmd_postfreeze_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd_postfreeze_20260810_v1_e84c4560/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcrmd12_20260810_v1 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcrmd_postfreeze_20260810_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd_postfreeze_20260810_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd_postfreeze_20260810_v1_e84c4560/code/scripts/launch_phase1_rcrmd_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd_postfreeze_20260810_v1_launcher.out 2>&1 < /dev/null &
```

Runner调用次数必须为1；调用端超时先清理本地SSH并只读确认landed，禁止重发。GPU映射沿用训练12臂：0:F1C/F5G，1:F1G/F5C，2:F2C/F6G，3:F2G/F6C，4:F3C，5:F3G，6:F4C，7:F4G；GPU7当前SCB v4已技术失败并释放，但仍不得改变映射。

## 6.技术停止、工件与分析边界

启动前必须闭合：direct preflight、release/run/log/outer ABSENT、完整archive无prefix、6成员SHA/mode、ManySig、12 final checkpoint和RCRMD receipt、py_compile、4个公开CLI `--help`、bash-n、dry-run42。

仅因错误checkout/hash/覆盖、P0/协议违反、launcher-wide确定性故障或至少2个distinct candidate同一确定性异常而停止；只停精确run-owned树并保留partial。不得按任何性能字段早停。retry=`NO`。

预期工件：

- 12 clean NPZ、12 LEO NPZ、12 LEO binding、12 proxy JSON、12 proxy CSV、6 pair JSON；
- 12 candidate日志+6 pair日志+PID表+outer；
- F6 pair含完整matrix aggregate和原始工件重算证据。

Runner只读取技术binding并回收JSON/CSV/log/manifest小工件，不下载checkpoint或NPZ，不解释性能。主代理在工件完整后读取6个pair同run结果并作唯一最终判定。

## 7.Runner落地登记

- 恢复后只读核验：release/run/log/outer/temp在启动前均ABSENT；无postfreeze进程；GPU0–7各约1MiB；12个训练final checkpoint、RCRMD terminal receipt、ManySig SHA均与训练报告§8.1闭合。
- 本地归档：`artifacts/phase1_rcrmd_postfreeze_20260810_v1_e84c4560_fulltree.tar`；无前缀、4925 members、无`code/code`重复路径；大小=`260976640`字节；SHA256=`aa2ef42e9454d2362ece03a0bed2dc92795dcce727429bf74f53afb31f3939f3`。
- 归档6成员SHA全部匹配冻结清单；launcher Git树mode=`100755`，归档/远端release可执行位保留（远端`stat`=`775`，未改内容）。远端release已解包为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd_postfreeze_20260810_v1_e84c4560`；临时归档已移除。
- 该阶段状态：`LANDED / STATIC_PENDING / NO_PERFORMANCE_RESULT`；当时尚未执行postfreeze launch。

## 8.远端静态门（launch前）

- 远端release/code：5个postfreeze Python文件`py_compile=PASS`（含test）；4个公开CLI `--help=PASS`；launcher`bash -n=PASS`。
- 冻结环境`POSTFREEZE_RUN_ID=phase1_rcrmd_postfreeze_20260810_v1`下`--dry-run`严格42条：`RCRMD_CLEAN_EXPORT=12`、`RCRMD_LEO_EXPORT_AND_BIND=12`、`FROZEN_LOGITS_PROXY_BINDING=12`、`RCRMD_PAIR_SCORE=6`；训练root引用30、postfreeze引用42；无路径创建。
- 首次静态探针仅因runner在`release/code`内重复添加`code/`前缀而失败；未改代码、未产生远端运行工件；一次机械修正后所有门通过。
- 该阶段状态：`LANDED / STATIC_PASS / NO_PERFORMANCE_RESULT`；当时exact launch调用次数=`0`，retry=`NO`。

## 9.唯一launch与技术终态

- §5 exact command调用次数=`1`；SSH调用端约124秒超时，按规则清理本地残留并只读确认已landed，未重发，retry=`NO`。
- 12候选PID/GPU登记（来自`candidate_pids.tsv`）：F1C=`859360`/0，F5G=`859361`/0，F1G=`859362`/1，F5C=`859363`/1，F2C=`859365`/2，F6G=`859366`/2，F2G=`859367`/3，F6C=`859368`/3，F3C=`859372`/4，F3G=`859376`/5，F4C=`859377`/6，F4G=`859379`/7。wrapper/launcher未在outer中持久化；自然终态核验时其进程均为0。
- 42步完整计数：clean NPZ=`12/12`，LEO NPZ=`12/12`，LEO binding=`12/12`，proxy JSON=`12/12`，proxy CSV=`12/12`，pair JSON=`6/6`；candidate日志=`12/12`，pair日志=`6/6`，PID表=`1/1`，outer=`0`字节。
- 6/6 pair技术字段均通过：schema、postfreeze root、matrix ID、training root、common training binding、C/G receipt revalidation、proxy binding和technical binding均为`true`；F6的`matrix_aggregate`存在。C/G冻结收据字段保持`C enabled=false/lambda=0`、`G enabled=true/lambda=0.02`、每场景28 cells、source receiver `0..6`。
- 18阶段日志技术异常指纹（Traceback、RuntimeError、CUDA OOM、unrecognized arguments等）计数=`0`；自然终态后目标进程=`0`，GPU compute进程=`0`，GPU0–7各约1MiB。Runner未读取或解释accuracy、floor、AUROC、u-gap或任何pair性能。

## 10.小工件回收与清理

- 远端校正bundle：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd_postfreeze_20260810_v1/phase1_rcrmd_postfreeze_20260810_v1_small_artifacts_v2.tar`；63 members（62条小工件＋manifest）、大小=`84142080`字节、SHA256=`6baf6d26c0d6570d91ad49a7b3390668460f6b63ca17308aceffd7288324bf53`。
- 远端/本地manifest：`phase1_rcrmd_postfreeze_20260810_v1_small_artifacts_v2.manifest.txt`；大小=`9071`字节、SHA256=`3e4c25a402f3defdd111f784f9374fe66fc7aa474a034b2bf7e20091e4158c66`。bundle禁含`.pth/.npz/.pt/.npy`（计数=`0`）。本地回收目录：`E:\type10-7\automation_reports\CV-SincNet\phase1_rcrmd_postfreeze_20260810_v1\artifacts\returned_small\`。
- 首个runner包`phase1_rcrmd_postfreeze_20260810_v1_small_artifacts.tar`因manifest相对路径机械错误未含manifest成员；未作为交接包使用，校正v2已独立生成并核验，训练/评估输出未受影响。
- 最终本地SSH进程=`0`、N607/bridge TCP22=`0`；远端release、run、log、checkpoint和NPZ均保留，未下载checkpoint/NPZ/特征值，未启动任何后续任务。

Runner终态：`ARTIFACTS_COMPLETE / TECHNICAL_BINDING_PASS / PAIR_JSON_READY / NO_PERFORMANCE_INTERPRETATION`。Runner未作promotion/reject；以下分析由主代理在6份pair JSON和F6原始工件重算完整后一次性完成。

## 11.主代理同run性能分析与终裁

分析输入为校正v2 bundle中的F1–F6六份`F*_C_vs_G_pair_metrics.json`。6/6技术绑定为true；F6已重开F1–F5的clean、LEO、binding、proxy JSON/CSV和NPZ，核当前SHA并重算全部摘要、delta和门。下表所有分类数值均为百分数或百分点（pp）；每行C/G/delta来自同一fold、同一scenario和同一冻结run，不使用跨run最佳值。

### 11.1 clean source-validation四floor

|fold|C overall|G overall|Δoverall|C min-class|G min-class|Δmin-class|C min-RX|G min-RX|Δmin-RX|C min-day|G min-day|Δmin-day|四floor门|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|F1|99.310|99.262|-0.048|98.738|98.881|+0.143|97.708|97.750|+0.042|99.214|99.083|-0.131|PASS|
|F2|99.262|99.042|-0.220|98.905|98.024|-0.881|96.958|96.042|-0.917|99.048|98.821|-0.226|PASS|
|F3|99.339|99.185|-0.155|98.857|98.738|-0.119|96.958|96.208|-0.750|99.238|99.012|-0.226|PASS|
|F4|99.310|99.310|0.000|99.071|99.000|-0.071|97.875|97.750|-0.125|99.262|99.250|-0.012|PASS|
|F5|98.208|97.929|-0.280|96.524|96.476|-0.048|93.167|93.000|-0.167|97.917|97.560|-0.357|PASS|
|F6|97.792|96.923|-0.869|94.762|91.429|-3.333|90.292|86.292|-4.000|97.464|96.500|-0.964|FAIL|

clean门为5/6，不满足6/6。失败集中在F6的min-class（-3.333pp）和min-RX（-4.000pp），已越过冻结的-2pp下限；其余fold不能补偿。

### 11.2 LEO三场景四floor

|fold|scene|C overall|G overall|Δoverall|C min-class|G min-class|Δmin-class|C min-RX|G min-RX|Δmin-RX|C min-day|G min-day|Δmin-day|四floor门|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|F1|clear|95.588|94.669|-0.919|90.972|89.583|-1.389|82.474|78.351|-4.124|95.342|94.595|-0.747|FAIL|
|F1|low|95.588|95.221|-0.368|92.969|89.062|-3.906|87.640|85.393|-2.247|93.750|93.269|-0.481|FAIL|
|F1|rain|95.508|95.117|-0.391|90.625|88.281|-2.344|82.278|79.747|-2.532|95.486|93.750|-1.736|FAIL|
|F2|clear|91.912|94.669|+2.757|82.639|91.406|+8.767|82.474|88.660|+6.186|91.441|94.144|+2.703|PASS|
|F2|low|88.787|91.360|+2.574|72.656|87.500|+14.844|69.663|76.404|+6.742|88.462|90.865|+2.404|PASS|
|F2|rain|89.258|91.992|+2.734|78.125|87.500|+9.375|72.152|78.481|+6.329|88.194|91.518|+3.323|PASS|
|F3|clear|94.485|95.404|+0.919|90.972|92.361|+1.389|87.629|88.660|+1.031|94.099|95.342|+1.242|PASS|
|F3|low|86.949|90.441|+3.493|75.000|80.469|+5.469|62.921|69.663|+6.742|86.905|89.423|+2.518|PASS|
|F3|rain|87.695|89.844|+2.148|79.688|82.812|+3.125|75.949|78.481|+2.532|83.333|87.500|+4.167|PASS|
|F4|clear|93.199|93.750|+0.551|86.806|88.889|+2.083|83.505|84.536|+1.031|92.236|93.168|+0.932|PASS|
|F4|low|90.074|91.360|+1.287|76.562|83.594|+7.031|68.539|74.157|+5.618|89.286|90.774|+1.488|PASS|
|F4|rain|88.867|91.211|+2.344|75.000|82.812|+7.812|85.526|86.076|+0.550|84.028|87.847|+3.819|PASS|
|F5|clear|79.412|80.699|+1.287|59.722|66.667|+6.944|53.608|57.732|+4.124|78.378|79.814|+1.435|PASS|
|F5|low|71.875|75.551|+3.676|51.562|62.500|+10.938|49.438|49.438|0.000|69.712|75.481|+5.769|PASS|
|F5|rain|67.578|73.047|+5.469|45.312|62.500|+17.188|40.789|56.579|+15.789|66.964|72.222|+5.258|PASS|
|F6|clear|80.699|84.926|+4.228|54.861|67.361|+12.500|62.687|71.134|+8.447|80.180|84.472|+4.292|PASS|
|F6|low|82.353|84.743|+2.390|71.875|71.528|-0.347|58.427|64.045|+5.618|79.327|82.692|+3.365|PASS|
|F6|rain|81.055|86.133|+5.078|60.156|76.562|+16.406|65.823|68.354|+2.532|78.819|85.069|+6.250|PASS|

LEO四floor为15/18，不满足18/18。F1三场景全部失败；F1三场景等权overall delta为-0.559130pp，因此fold overall也仅5/6。F2–F6的15格全部通过，推动全18格等权delta达到：overall=+2.180990pp、min-class=+6.438079pp、min-RX=+3.575925pp、min-day=+2.555678pp；但冻结规则明确禁止这个正总体均值补偿F1失败格。

### 11.3 source-proxy连续几何双门

|fold|C AUROC|G AUROC|ΔAUROC|C u-gap|G u-gap|Δu-gap|AUROC严格改善|u-gap严格改善|双门|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|F1|0.756060|0.805997|+0.049937|886.481|318.597|-567.884|PASS|FAIL|FAIL|
|F2|0.457251|0.343333|-0.113918|259.189|148.176|-111.013|FAIL|FAIL|FAIL|
|F3|0.916731|0.913636|-0.003095|1797.819|948.334|-849.485|FAIL|FAIL|FAIL|
|F4|0.517958|0.459608|-0.058350|2677.730|413.047|-2264.683|FAIL|FAIL|FAIL|
|F5|0.945284|0.912655|-0.032629|539.846|359.429|-180.418|FAIL|FAIL|FAIL|
|F6|0.896691|0.730596|-0.166095|1806.554|611.749|-1194.805|FAIL|FAIL|FAIL|

proxy双门为0/6。仅F1的AUROC上升，但其u-gap下降；F2–F6的AUROC和u-gap同时下降。proxy是后冻结、零fit的连续几何诊断，不能替代真实unknown评测；这里也不能被LEO分类改善补偿。

### 11.4冻结门汇总与最终决定

|门|结果|要求|判定|
|---|---|---|---|
|技术绑定|6/6|全部通过|PASS|
|clean四floor|5/6|6/6|FAIL|
|LEO四floor|15/18|18/18|FAIL|
|fold三场景overall|5/6|6/6|FAIL|
|全18格overall|+2.180990pp|≥0|PASS|
|proxy连续双门|0/6|6/6|FAIL|

最终裁决：`REJECT_P1_RCRMD_PERMANENT`。

RCRMD验证了一个局部但不稳定的现象：receiver×class等权的正margin-drop二阶矩在F2–F6显著改善LEO分类尾部，却没有形成跨fold稳定性，并在F1出现三场景一致退化；同时F6 clean尾部越界、六折source-proxy几何全部未通过双门。因此该方法不得调λ、挑fold、用全局均值补偿、重试或改名复活。它不提供真实unknown、多卫星协同或Phase3能力证据。后续研发转回已冻结的Phase3基础设施闭环；RCRMD仅作为已完成的负结果保留。

最终状态：`ANALYZED / REJECT_P1_RCRMD_PERMANENT / NO_PHASE3_CAPABILITY_CLAIM`。
