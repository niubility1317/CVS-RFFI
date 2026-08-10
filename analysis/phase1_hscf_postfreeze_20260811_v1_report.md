# Phase1 P1-HSCF后冻结42步实验报告

## 1.状态与目标

- 实验ID：`phase1_hscf_postfreeze_20260811_v1`
- 日期：2026-08-11
- 当前状态：`ANALYZED / REJECT_P1_HSCF_PERMANENT / NO_PHASE3_CAPABILITY_CLAIM`
- 操作边界：主控冻结评价合同、矩阵和判定门；唯一N607 Runner只负责release落地、唯一启动、技术监控与小工件回收，不读取或解释性能字段。
- 训练输入：`phase1_hscf12_20260811_v2`，12/12臂技术闭合；训练报告SHA256=`e282a9657eaa06206d027decc88c375c512690441206ca895a4fd9f84bce356e`，Git镜像commit=`9aff4d20242ea124f96f4a979bc4bf4b0f381a58`。
- 目标：在不改变训练、fold、seed、receiver、TX、场景或阈值的前提下，对同fold C/G执行固定clean、三LEO、fixed400 proxy和连续Gaussian-NLL公平评价，产出6份pair JSON及F6矩阵聚合。
- 假设：HSCF保持跨样本head-subspace相对构型，可能在守住clean与LEO分类floor的同时改善source-only Gaussian几何的proxy连续双门；该假设仅由完整42步非补偿矩阵证伪或支持。
- 声明边界：技术完成不等于性能通过；任一非补偿门失败即`REJECT_P1_HSCF_PERMANENT`，全部通过也只能`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`，不构成unknown、真实开放集、Phase2或Phase3能力声明。

## 2.冻结版本、本地文件与独立审查

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 后冻结实现commit：`ce162329af1076c4b829715d0e027a2d8b4f5723`
- 独立actual-diff终裁：`P0=0 / P1=0 / ALLOW_LOCAL_GIT_RELEASE`
- 审查边界：ALLOW只允许技术发布与Runner交接，不包含性能结果、候选晋级或N607已执行声明。

|文件|工作树SHA256|mode|用途|
|---|---|---:|---|
|`analysis/phase1_hscf_postfreeze_design_20260811.md`|`5ff6127be1f543fd559e5ddfdaa0d914302fc7e84e4c613e2e4d3380fa4b7de4`|100644|后冻结合同、追踪与证据边界|
|`code/export_phase1_hscf_features.py`|`c169434a6ea5e3366b5b8efb5889f6cf06a6056c06f8535f23bca3330f624c5f`|100644|clean L/V/proxy专用导出|
|`code/export_phase1_hscf_leo_features.py`|`5337c8cef7016e28f5568323f0df13149530c29db29cb65ddfe8f1ec0b50a968`|100644|三LEO导出与物理绑定|
|`code/evaluate_phase1_hscf_postfreeze_pair.py`|`3633163f5c7d2e81bb0c45806ddfc60aca4e75d524d87f21665e2559182de07c`|100644|同fold C/G评分及F6原始工件聚合|
|`code/tests/test_phase1_hscf_postfreeze.py`|`fd661aa0cd96a41d1303c7cb9bbcc8a313abfa827ae7c04cbf31b03b7a3d9d57`|100644|receipt、绑定、Gaussian、floor、proxy与F6负测|
|`code/scripts/launch_phase1_hscf_postfreeze_20260811.sh`|`d13a872943a44558e4dc323a34f700da459db8c1e46a28c30cd73409ad8171c4`|100755|冻结42步launcher|

本地验证均在官方Conda hook激活`ssr-gpu`后串行完成：三脚本与测试`py_compile`通过；focused postfreeze为12 passed；HSCF core+postfreeze联合为32 passed；`bash -n`通过；dry-run严格`42=12 clean+12 LEO/binding+12 proxy+6 pair`；`git diff --check`通过。旧训练根`phase1_hscf12_20260810_v1`引用为0，当前训练根固定为`phase1_hscf12_20260811_v2`。

## 3.冻结数据、评价核与权限

- 数据：`ManySig.pkl`，预期SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- source days：`2021_03_01,2021_03_08`；source RX：`1-1,1-19,14-7,18-2,19-2,2-1`；LEO场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- `SOURCE_SAT_SEED=7281718`，`EXPORT_SEED=7281105`，每TX最多400，预期source总数1600，fixed proxy总数400。
- 每候选只用source-L clean`feat_joint`拟合float64 totalized-L2对角Gaussian；V/proxy零fit、零校准、零选模，全部行与精确零行保留，任何nonfinite fatal。
- 方差使用`ddof=1`，`0.9×class+0.1×class-equal pooled`，floor=`1e-6`；完整Gaussian-NLL及稳定logsumexp产生连续unknown量。
- HSCF训练receipt必须由当前validator重开：`B=128`、local4、固定分母512、`lambda=0.02`；C辅助N/A/0，G三scene正项与raw-unscaled VJP闭合。
- F6必须重开F1--F5原始clean/LEO NPZ、binding、proxy JSON/CSV、当前C/G checkpoint与receipt并重算，不接受prior pair自报摘要。

## 4.冻结42步矩阵与GPU映射

|顺序|候选|fold|arm|GPU|source train TX|known-val TX|proxy TX|
|---:|---|---:|---|---:|---|---|---|
|1|F1C_HSCF12|1|C|0|20-15,20-19,6-15,8-20|14-7|14-10|
|2|F5G_HSCF12|5|G|0|14-10,14-7,20-15,20-19|8-20|6-15|
|3|F1G_HSCF12|1|G|1|20-15,20-19,6-15,8-20|14-7|14-10|
|4|F5C_HSCF12|5|C|1|14-10,14-7,20-15,20-19|8-20|6-15|
|5|F2C_HSCF12|2|C|2|14-10,20-19,6-15,8-20|20-15|14-7|
|6|F6G_HSCF12|6|G|2|14-7,20-15,20-19,6-15|14-10|8-20|
|7|F2G_HSCF12|2|G|3|14-10,20-19,6-15,8-20|20-15|14-7|
|8|F6C_HSCF12|6|C|3|14-7,20-15,20-19,6-15|14-10|8-20|
|9|F3C_HSCF12|3|C|4|14-10,14-7,6-15,8-20|20-19|20-15|
|10|F3G_HSCF12|3|G|5|14-10,14-7,6-15,8-20|20-19|20-15|
|11|F4C_HSCF12|4|C|6|14-10,14-7,20-15,8-20|6-15|20-19|
|12|F4G_HSCF12|4|G|7|14-10,14-7,20-15,8-20|6-15|20-19|

每候选依次产生clean、LEO/binding和proxy三步；12候选完成后按F1--F6串行产生6个pair，共42步。候选内部串行，每GPU最多2个候选进程。

## 5.N607发布预登记

- 普通账号目标：`N607`；禁止管理员账号。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf_postfreeze_20260811_v1_ce162329`
- CWD：上述release的`code`目录。
- immutable训练根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hscf12_20260811_v2`
- postfreeze根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hscf_postfreeze_20260811_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf_postfreeze_20260811_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf_postfreeze_20260811_v1_launcher.out`
- retry：`NO`；启动所有权：唯一Runner；主控不得重复启动。

冻结唯一命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf_postfreeze_20260811_v1_ce162329/code && nohup env POSTFREEZE_RUN_ID=phase1_hscf_postfreeze_20260811_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf_postfreeze_20260811_v1_ce162329/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hscf12_20260811_v2 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hscf_postfreeze_20260811_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf_postfreeze_20260811_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf_postfreeze_20260811_v1_ce162329/code/scripts/launch_phase1_hscf_postfreeze_20260811.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf_postfreeze_20260811_v1_launcher.out 2>&1 < /dev/null &
```

Runner落地前必须完成：direct preflight；LF无prefix archive、成员SHA/mode、`code/code=0`；ManySig SHA；12个v2 final checkpoint存在、SHA记录、receipt由当前validator重开；远端py_compile/help/bash-n/dry-run42；release/run/log/outer不存在；GPU占用记录。SSH超时后先清理本地ssh/TCP22，再只读确认是否landed，禁止重发。

## 6.技术健康、停止与工件

技术停止仅限P0协议/权限/checkout/hash/输出覆盖、launcher-wide确定性故障、至少2个不同候选同一标准化异常指纹、OOM/CUDA/argparse/路径权限错误，或工件闭合失败。不得因accuracy、floor、AUROC、u-gap或任何中间性能停止、重试或调参。停止前必须绑定本run的PID/CWD/cmdline，只处理本run进程并保留部分工件；技术失败记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

预期工件：12 clean NPZ、12 LEO NPZ、12 LEO binding JSON、12 proxy JSON、12 proxy CSV、6 pair JSON、12候选日志、6 pair日志、`candidate_pids.tsv`与outer。Runner只核技术schema、输入绑定、计数、SHA、异常与F6 raw-reopen键；回收仅JSON/CSV/binding/log/PID/manifest的小bundle，排除NPZ、pth、pt和npy。性能字段由主控在技术闭合并回收后读取。

## 7.冻结非补偿门与最终结果

|冻结门|要求|当前结果|判定|
|---|---:|---:|---|
|技术绑定|6/6 pair|6/6|PASS|
|clean四floor|6/6 fold，每项Δ≥-2pp|6/6|PASS|
|LEO四floor|18/18 scene-cell，每项Δ≥-2pp|16/18；5/6 fold完整|FAIL|
|逐fold三场景overall|6/6 fold等权Δ≥0|6/6|PASS|
|全18格overall|等权Δ≥0|+1.813343pp|PASS|
|fixed400 proxy双门|6/6 fold同时ΔAUROC>0且Δu-gap>0|2/6|FAIL|

最终分析按同fold、同scene保留C/G完整行，独立重算与F6 aggregate一致。LEO四floor和proxy双门均为预注册不可补偿失败，最终状态为`ANALYZED / REJECT_P1_HSCF_PERMANENT / NO_PHASE3_CAPABILITY_CLAIM`。

## 8.Runner本地发布前技术记录（2026-08-11）

- Runner角色：`Luna/max`，唯一N607启动所有权；本阶段未启动，launch调用次数=`0`，retry=`NO`。
- direct preflight：`tools\\n607_ssh_preflight.ps1`通过；普通账号`N607`、项目根可见、8张RTX3090空闲；每次短SSH后本地`ssh.exe=0`且N607/bridge TCP22 established=`0`。
- 本地仓：`ce162329af1076c4b829715d0e027a2d8b4f5723`；工作树仅保留既有未跟踪`conversation_index/`，未修改冻结实现。
- LF无prefix完整Git archive：成员=`4964`，prefix成员=`0`，`code/code/`成员=`0`，文本CR字节=`0`；六冻结成员SHA全部匹配；launcher归档mode=`100755`。归档工件与检查JSON保存在本报告`artifacts/runner_20260811/`目录。
- 归档SHA256=`0754fc9d214c8712f1cc367afd719323a30b986e8b318364b392312055962556`，bytes=`267151360`。
- 本地静态验证：`py_compile=0`、focused pytest=`0`（12 passed）、三个`--help=0`、Git Bash`bash -n=0`、冻结dry-run=`42`行且exit=`0`；过程日志仅作技术记录。
- 远端启动前只读状态：release/run/log/outer均`ABSENT`；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；v2训练根12份`final_ssdg.pth`均存在，SHA记录于`artifacts/runner_20260811/checkpoint_sha256.tsv`；未读取性能字段。

## 9.Runner远端静态与receipt技术记录（2026-08-11）

- 唯一SCP调用次数=`1`，SCP exit=`0`；incoming archive SHA/bytes/members分别为`0754fc9d214c8712f1cc367afd719323a30b986e8b318364b392312055962556`/`267151360`/`4964`；staging到release原子落地成功，远端六成员SHA与launcher`755`匹配。
- release内静态门：`py_compile=PASS`、4个`--help=PASS`、`bash -n=PASS`、冻结dry-run=`42`行、旧候选身份行=`0`；静态命令未创建run/log输出。
- 当前release validator顺序重开12份真实terminal receipt：`terminal_contract_passed=12`、`B128/local4/denom512=12`、C零辅助=`6`、G三scene raw-unscaled VJP=`6`、C/G common binding=`6`。这些是技术闭合证据，不含性能读取或解释。
- 远端release状态=`LANDED`；尚未启动，launch调用次数仍=`0`，retry=`NO`。

## 10.Runner唯一启动、监控与技术闭合记录（2026-08-11）

- 唯一冻结启动命令调用次数=`1`，SSH返回`0`；retry=`NO`，未发生重发。启动后主PID/CWD/cmdline/run-root绑定、GPU映射与日志增长均完成短连接核验；12个候选PID均已退出，GPU计算进程已释放。
- 最终工件计数：clean NPZ=`12`、LEO NPZ=`12`、LEO binding JSON=`12`、proxy JSON=`12`、proxy CSV=`12`、pair JSON=`6`、候选/ pair阶段日志=`18`、`candidate_pids.tsv`数据行=`12`。标准化技术异常指纹扫描为`0`（未见Traceback、RuntimeError、OOM/CUDA、argparse、路径/权限类错误）。
- 6份pair技术闭合：schema=`6/6`、roots=`6/6`、common C/G binding=`6/6`、proxy C/G recomputation=`6/6`、F6 raw-reopen键=`6/6`、binding path issue=`0`；F6已重开F1--F5原始clean/LEO NPZ、binding、proxy JSON/CSV、C/G checkpoint与receipt后完成技术聚合。上述仅为工件/绑定/重开证据，不读取性能值。
- 预期outer路径`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf_postfreeze_20260811_v1/phase1_hscf_postfreeze_20260811_v1_launcher.out`最终状态=`ABSENT`。该文件不参与计算、绑定或判定；42项核心工件、18阶段日志、PID表、6份pair技术闭合均完整，故按主控决定记为非阻断缺项，不补造、不重跑。
- 机械探针曾出现一次合并闭合脚本断言失败及两次诊断脚本语法错误；均为只读探针、未改动远端状态且不影响release、唯一launch或工件，随后独立最终技术扫描通过。

## 11.小bundle回收与最终状态（2026-08-11）

- 回收范围严格为JSON/CSV/binding/log/PID/manifest；排除`.npz/.pth/.pt/.npy`。远端manifest：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf_postfreeze_20260811_v1/phase1_hscf_postfreeze_20260811_v1_small_manifest_v2.json`，SHA256=`a4530038fdc10fda0b7ca86940669823ede32ea1f238d7d588bb5eb2a6a9eeae`，bytes=`19468`；`outer_status=ABSENT`且manifest明确记录该状态，成员（不含manifest）=`61`。
- 远端bundle：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf_postfreeze_20260811_v1/phase1_hscf_postfreeze_20260811_v1_small_bundle_v2.tar`；本地回收：`E:\type10-7\automation_reports\CV-SincNet\phase1_hscf_postfreeze_20260811_v1\artifacts\retrieved_small\phase1_hscf_postfreeze_20260811_v1_small_bundle_v2.tar`，SHA256=`4dae782af6bef23c7c55d0077a04047d82330c3ff9af25b07faddae6e0fe3010`，bytes=`32542720`，成员=`62`（含manifest），禁入成员=`0`。本地manifest副本同SHA/bytes并经`inspect_bundle_v2.py`验证通过。
- 技术状态=`ARTIFACTS_COMPLETE / NO_PERFORMANCE_INTERPRETATION`；不据此晋级或拒绝候选。Runner未读取`accuracy/floor/AUROC/u-gap`等性能字段，未做任何性能解释、调参、重启或候选判定。

## 12.主控完整解析与复核边界

主控在Runner技术交接后解包最终v2小bundle到`artifacts/analysis_extract_main/`。完整读取18份阶段日志：12份候选日志各845行，6份pair日志均为完整单行JSON；Traceback、RuntimeError、OOM、CUDA、SIGSEGV、argparse、HSCF异常、NaN和Inf标记均为0。6份pair JSON的schema、matrix/training root、技术绑定、HSCF receipt重验、C/G common binding与proxy recomputation均为6/6；F6包含5份prior pair原始工件重开记录，`raw_artifacts_recomputed=true`为5/5。

主控没有采用Runner的性能判断，而是从6份pair JSON逐项独立重算clean四floor、18个LEO scene-cell四floor、每fold三场景等权overall、全18格等权overall和每foldproxy双严格门。全部数值与F6 matrix aggregate逐字义一致，F6最终verdict为`REJECT_P1_HSCF_PERMANENT`。

## 13.clean同fold结果

|fold|C overall(%)|G overall(%)|Δoverall(pp)|Δmin-class(pp)|Δmin-RX(pp)|Δmin-day(pp)|四floor|
|---|---:|---:|---:|---:|---:|---:|---|
|F1|99.273810|99.297619|+0.023810|+0.119048|+0.208333|+0.059524|PASS|
|F2|99.202381|99.184524|-0.017857|-0.071429|-0.250000|-0.011905|PASS|
|F3|99.119048|99.267857|+0.148810|+0.809524|+0.625000|+0.202381|PASS|
|F4|99.226190|99.285714|+0.059524|+0.095238|+0.083333|+0.107143|PASS|
|F5|97.494048|97.898810|+0.404762|+2.309524|+0.875000|+0.571429|PASS|
|F6|97.386905|97.607143|+0.220238|+0.976190|+0.833333|+0.357143|PASS|

clean四floor为6/6；F2的轻微负增量远高于-2pp floor，不构成失败。HSCF没有以LEO收益换取明显clean退化。

## 14.LEO三场景同fold结果

|fold|scene|C overall(%)|G overall(%)|Δoverall(pp)|Δmin-class(pp)|Δmin-RX(pp)|Δmin-day(pp)|四floor|
|---|---|---:|---:|---:|---:|---:|---:|---|
|F1|clear|96.323529|96.323529|+0.000000|+0.000000|-1.030928|+0.310559|PASS|
|F1|low-elev|95.588235|96.139706|+0.551471|-0.781250|+1.123596|+1.442308|PASS|
|F1|rain|95.312500|96.289062|+0.976562|+3.906250|+3.797468|+0.892857|PASS|
|F2|clear|92.647059|93.750000|+1.102941|+2.777778|+2.061856|+0.450450|PASS|
|F2|low-elev|89.889706|92.463235|+2.573529|+4.687500|+2.247191|+2.884615|PASS|
|F2|rain|90.625000|92.382812|+1.757812|+3.906250|+0.000000|+2.628968|PASS|
|F3|clear|91.727941|95.220588|+3.492647|+8.333333|+6.227361|+4.658385|PASS|
|F3|low-elev|85.477941|88.235294|+2.757353|+6.250000|-1.123596|+3.273810|PASS|
|F3|rain|85.156250|88.867188|+3.710938|+12.500000|+2.531646|+3.819444|PASS|
|F4|clear|93.198529|94.852941|+1.654412|+2.777778|+0.752299|+2.529237|PASS|
|F4|low-elev|88.419118|91.360294|+2.941176|+6.250000|+6.741573|+3.846154|PASS|
|F4|rain|90.039062|90.625000|+0.585938|+0.781250|+0.000000|+1.736111|PASS|
|F5|clear|80.147059|82.169118|+2.022059|+10.937500|+8.247423|+3.105590|PASS|
|F5|low-elev|71.691176|73.345588|+1.654412|+9.722222|+5.617978|-0.480769|PASS|
|F5|rain|68.359375|71.679688|+3.320312|+13.281250|+0.000000|+4.910714|PASS|
|F6|clear|78.860294|79.595588|+0.735294|-5.555556|-7.462687|-0.450450|FAIL|
|F6|low-elev|82.720588|84.742647|+2.022059|-0.781250|+6.741573|+0.480769|PASS|
|F6|rain|80.078125|80.859375|+0.781250|-4.687500|+6.329114|+1.041667|FAIL|

LEO四floor为16/18，完整fold为5/6。F6-clear的min-class与min-RX分别下降5.555556pp和7.462687pp；F6-rain的min-class下降4.687500pp。虽然这两个scene的overall仍上升，但非补偿尾部floor失败，不能由均值补偿。

逐fold三场景等权overall增量为F1=`+0.509344pp`、F2=`+1.811428pp`、F3=`+3.320312pp`、F4=`+1.727175pp`、F5=`+2.332261pp`、F6=`+1.179534pp`，达到6/6。全18格等权增量为overall=`+1.813343pp`、min-class=`+4.128086pp`、min-RX=`+2.377881pp`、min-day=`+2.060023pp`。这证明HSCF对平均LEO分类具有跨fold一致的正贡献，但不能证明所有receiver/class尾部均被保护。

## 15.fixed400 proxy连续双门

|fold|C AUROC|G AUROC|ΔAUROC|C u-gap|G u-gap|Δu-gap|双严格门|
|---|---:|---:|---:|---:|---:|---:|---|
|F1|0.797673|0.792503|-0.005170|846.024808|1403.062439|+557.037631|FAIL|
|F2|0.375816|0.436440|+0.060624|184.899931|425.374560|+240.474630|PASS|
|F3|0.940728|0.908695|-0.032032|1293.801802|1613.599518|+319.797716|FAIL|
|F4|0.475547|0.462573|-0.012974|828.357743|1119.578372|+291.220629|FAIL|
|F5|0.862321|0.876164|+0.013844|365.903380|419.142565|+53.239185|PASS|
|F6|0.740645|0.745080|+0.004435|1346.486270|836.134095|-510.352176|FAIL|

proxy双严格门仅2/6。HSCF把u-gap提高到5/6，但F1、F3、F4的AUROC轻微下降，F6则AUROC上升而u-gap明显下降；二者必须同fold同时严格为正，不能跨fold或跨指标补偿。proxy仍只是TX隔离、L-only fit的连续几何诊断，不是真实unknown能力证据。

## 16.完整门、阶段复盘与最终裁决

|候选|clean四floor|LEO四floor|完整LEO fold|全18格overall|proxy双门|最终状态|
|---|---:|---:|---:|---:|---:|---|
|ICMT|5/6|3/18|0/6|-4.309002pp|1/6|永久拒绝|
|CAGM|5/6|9/18|1/6|-0.128294pp|4/6|永久拒绝|
|RCRMD|5/6|15/18|5/6|+2.180990pp|0/6|永久拒绝|
|RCAT|6/6|10/18|3/6|+0.149357pp|2/6|永久拒绝|
|RECTE|6/6|17/18|5/6|+2.304815pp|0/6|永久拒绝|
|HSCF|6/6|16/18|5/6|+1.813343pp|2/6|永久拒绝|

HSCF的新增证据不是一次全面退化：它守住clean 6/6，使6个fold的LEO overall全部为正，并把proxy u-gap改善到5/6；但它没有同时守住F6的receiver/class尾部，也没有让proxy AUROC与u-gap在6折共同改善。相较RECTE，HSCF牺牲了一个LEO四floor cell并多恢复两个proxy双门；相较RCAT，它显著提高LEO覆盖但没有提高proxy完整通过数。这表明当前机制族仍存在“平均LEO鲁棒性、最坏cell保护、source-only proxy排序”三者不能同时闭合的结构性张力。

本节构成超过三轮探索后的记录性复盘。ICMT、CAGM、RCRMD、RCAT、RECTE和HSCF均已在同一非补偿合同下完成，不再复活、改名或以局部成功拼接；下一候选不得从F6、某receiver/day或proxy结果反向定制阈值、权重、seed或超参。若继续研发，必须先提出单一source-L-only原语，形式上同时解释平均构型、cell尾部和Gaussian几何为何不互相逃逸，再经过`DESIGN_DRAFT→FEASIBILITY_REVIEW→DESIGN_FROZEN`与独立P0/P1；postfreeze仍只作为结果门，不进入训练反馈。

F6 matrix aggregate与主控独立重算的最终verdict均为`REJECT_P1_HSCF_PERMANENT`。本轮不调lambda、不重跑、不选择F2/F5局部proxy成功，也不以6/6 fold overall或全18格正均值补偿失败。最终状态：`ANALYZED / REJECT_P1_HSCF_PERMANENT / NO_PHASE3_CAPABILITY_CLAIM`。
