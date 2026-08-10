# Phase1 P1-HSCF后冻结42步实验报告

## 1.状态与目标

- 实验ID：`phase1_hscf_postfreeze_20260811_v1`
- 日期：2026-08-11
- 当前状态：`PREREGISTERED / LOCAL_VERIFIED / P0=0 / P1=0 / NO_PERFORMANCE_INTERPRETATION`
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

## 7.冻结非补偿门与结果占位

|冻结门|要求|当前结果|判定|
|---|---:|---:|---|
|技术绑定|6/6 pair|待运行|PENDING|
|clean四floor|6/6 fold，每项Δ≥-2pp|待运行|PENDING|
|LEO四floor|18/18 scene-cell，每项Δ≥-2pp|待运行|PENDING|
|逐fold三场景overall|6/6 fold等权Δ≥0|待运行|PENDING|
|全18格overall|等权Δ≥0|待运行|PENDING|
|fixed400 proxy双门|6/6 fold同时ΔAUROC>0且Δu-gap>0|待运行|PENDING|

最终分析必须按同fold、同scene保留C/G完整行，报告overall、min-class、min-RX、min-day及proxy AUROC/u-gap；不得用不同fold或不同指标的边际最值拼接成结论。当前状态保持`PREREGISTERED / LOCAL_VERIFIED / NO_PERFORMANCE_INTERPRETATION`。

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
