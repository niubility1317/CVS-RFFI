# Phase1 P1-RCRMD postfreeze v1实验报告

## 1.状态与目标

- 实验ID：`phase1_rcrmd_postfreeze_20260810_v1`
- 日期：2026-08-10
- 操作：主代理冻结方法、门和分析；唯一N607 Runner负责落地、42步执行、技术监控和小工件回收
- 当前状态：`ARTIFACTS_COMPLETE / TECHNICAL_BINDING_PASS / PAIR_JSON_READY / NO_PERFORMANCE_INTERPRETATION`
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

## 7.Runner落地登记（静态门待闭合）

- 恢复后只读核验：release/run/log/outer/temp在启动前均ABSENT；无postfreeze进程；GPU0–7各约1MiB；12个训练final checkpoint、RCRMD terminal receipt、ManySig SHA均与训练报告§8.1闭合。
- 本地归档：`artifacts/phase1_rcrmd_postfreeze_20260810_v1_e84c4560_fulltree.tar`；无前缀、4925 members、无`code/code`重复路径；大小=`260976640`字节；SHA256=`aa2ef42e9454d2362ece03a0bed2dc92795dcce727429bf74f53afb31f3939f3`。
- 归档6成员SHA全部匹配冻结清单；launcher归档/远端release mode=`0755`。远端release已解包为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd_postfreeze_20260810_v1_e84c4560`；临时归档已移除。
- 当前状态：`LANDED / STATIC_PENDING / NO_PERFORMANCE_RESULT`；尚未执行任何postfreeze launch。

## 8.远端静态门（launch前）

- 远端release/code：5个postfreeze Python文件`py_compile=PASS`（含test）；4个公开CLI `--help=PASS`；launcher`bash -n=PASS`。
- 冻结环境`POSTFREEZE_RUN_ID=phase1_rcrmd_postfreeze_20260810_v1`下`--dry-run`严格42条：`RCRMD_CLEAN_EXPORT=12`、`RCRMD_LEO_EXPORT_AND_BIND=12`、`FROZEN_LOGITS_PROXY_BINDING=12`、`RCRMD_PAIR_SCORE=6`；训练root引用30、postfreeze引用42；无路径创建。
- 首次静态探针仅因runner在`release/code`内重复添加`code/`前缀而失败；未改代码、未产生远端运行工件；一次机械修正后所有门通过。
- 当前状态：`LANDED / STATIC_PASS / NO_PERFORMANCE_RESULT`；exact launch调用次数=`0`，retry=`NO`。

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

当前状态：`ARTIFACTS_COMPLETE / TECHNICAL_BINDING_PASS / PAIR_JSON_READY / NO_PERFORMANCE_INTERPRETATION`。Runner不作promotion/reject，不提供性能结论；后续pair性能分析由主代理独立完成。
