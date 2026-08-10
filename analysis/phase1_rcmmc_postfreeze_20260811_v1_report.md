# Phase1 P1-RCMMC后冻结42步实验报告

## 1.状态与目标

- 实验ID：`phase1_rcmmc_postfreeze_20260811_v1`
- 日期：2026-08-11
- 当前状态：`PREREGISTERED_LOCAL_VERIFIED / WAITING_TRAINING_TECHNICAL_HANDOFF / NO_PERFORMANCE_RESULT`
- 操作边界：主控冻结评价合同、矩阵和非补偿门；唯一N607 Runner只负责release落地、唯一启动、技术监控与小工件回收，不读取或解释性能字段。
- 训练输入：`phase1_rcmmc12_20260811_v1`。12臂已自然结束且final/terminal/completion表面计数为12/12，仍须等待Runner逐臂receipt、common binding、资源收据和小bundle最终交接后，才允许后冻结启动。
- 目标：不改变训练、fold、seed、receiver、TX、场景、Gaussian或阈值，对同fold C/G执行固定clean、三LEO、fixed400 proxy和连续Gaussian-NLL公平评价，产出6份pair JSON及F6矩阵聚合。
- 假设：RCMMC对每个source RX×class cell的totalized-feature一、二阶矩做clean→LEO同物理约束，可能在保持clean和LEO分类floor的同时改善后冻结source-only Gaussian几何；该假设只能由完整42步非补偿矩阵证伪或支持。
- 声明边界：技术完成不等于性能通过；任一非补偿门失败即`REJECT_P1_RCMMC_PERMANENT`。全部通过也只能`PENDING_MAIN_REVIEW_FULL_6_FOLD`，不构成unknown、真实开放集、Phase2或Phase3能力声明。

## 2.冻结版本、本地文件与独立审查

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 后冻结实现commit：`aabd8358cb5303e34b546b6e4485afc1575fccf0`
- 独立actual-diff终裁：`P0=0 / P1=0 / ALLOW_LOCAL_GIT_RELEASE`
- 审查边界：ALLOW只允许技术发布与Runner交接，不包含性能结果、候选晋级或N607已执行声明。
- 审查中复现并修复两个实际P0：raw receiver token曾可从clean manifest/LEO binding穿透pair，也可嵌套于raw terminal receipt。最终顶层/嵌套receipt、clean manifest、LEO binding共8个攻击均fail-closed；科学公式、阈值和42步矩阵未变。

|文件|SHA256|mode|用途|
|---|---|---:|---|
|`analysis/phase1_rcmmc_postfreeze_design_20260811.md`|`43a2248f7893e1d4c3d9e187259d26566d3e7a950fdbaa0b508fe1eaa11f2304`|100644|后冻结合同、追踪与证据边界|
|`code/export_phase1_rcmmc_features.py`|`f6baea4326ed25ba6736cfdfe0809f275c9f1018b8864c9edccc597f48e5dda8`|100644|clean L/V/proxy专用导出和raw receipt重开|
|`code/export_phase1_rcmmc_leo_features.py`|`3ea743cdb50749622e04790013766853b94ef29fb893541ec7ef25464cde318a`|100644|三LEO导出及物理TX/RX/day绑定|
|`code/evaluate_phase1_rcmmc_postfreeze_pair.py`|`a1342ae7a9cb20e3895cadb0df56a5b4ee5893387cc8a075c3485fc927de519e`|100644|同fold C/G评分及F6原始工件重开|
|`code/tests/test_phase1_rcmmc_postfreeze.py`|`8d260b234b37af8741494e39a72a4d3c60cb8ddf965ac44bd273bc44e9ef1cec`|100644|receipt、绑定、Gaussian、floor、proxy、F6与raw-token负测|
|`code/scripts/launch_phase1_rcmmc_postfreeze_20260811.sh`|`d92188e2b4ce0fcc637784f3ca4ca5d7dac0be63bd7bffe742ccfe13753094c4`|100755|冻结42步launcher|

本地验证均在官方Conda hook激活`ssr-gpu`后串行完成：四个Python文件`py_compile`通过；postfreeze聚焦18 passed；RCMMC core+postfreeze联合33 passed；三个CLI help通过；`bash -n`通过；dry-run严格`42=12 clean+12 LEO/binding+12 proxy+6 pair`且旧候选identity=0；`git diff --check`通过。

## 3.冻结数据、评价核与权限

- 数据：`ManySig.pkl`，预期SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- source days：`2021_03_01,2021_03_08`；source RX：`1-1,1-19,14-7,18-2,19-2,2-1`；LEO场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- `SOURCE_SAT_SEED=7281718`，`EXPORT_SEED=7281105`，每TX最多400，预期source总数1600，fixed proxy总数400。
- 每候选只用source-L clean`feat_joint`拟合float64安全totalized-L2对角Gaussian；V/proxy零fit、零校准、零选模，全部行与精确零行保留，任何nonfinite fatal。
- 方差使用`ddof=1`，`0.9×class+0.1×class-equal pooled`，floor=`1e-6`；完整Gaussian-NLL和稳定logsumexp产生连续unknown量。
- RCMMC训练receipt必须由当前validator重开：`B=128`、`d=160`、local4、固定28格；C辅助N/A/0，G三scene正D和一次四参raw-unscaled VJP闭合。raw source receiver token只允许运行期从source split解析，receipt、manifest和binding只能持久化count/SHA。
- F6必须重开F1--F5原始clean/LEO NPZ、binding、proxy JSON/CSV、当前C/G checkpoint和receipt并重算，不接受prior pair自报摘要。

## 4.冻结42步矩阵与GPU映射

|顺序|候选|fold|arm|GPU|source train TX|known-val TX|proxy TX|
|---:|---|---:|---|---:|---|---|---|
|1|F1C_RCMMC12|1|C|0|20-15,20-19,6-15,8-20|14-7|14-10|
|2|F5G_RCMMC12|5|G|0|14-10,14-7,20-15,20-19|8-20|6-15|
|3|F1G_RCMMC12|1|G|1|20-15,20-19,6-15,8-20|14-7|14-10|
|4|F5C_RCMMC12|5|C|1|14-10,14-7,20-15,20-19|8-20|6-15|
|5|F2C_RCMMC12|2|C|2|14-10,20-19,6-15,8-20|20-15|14-7|
|6|F6G_RCMMC12|6|G|2|14-7,20-15,20-19,6-15|14-10|8-20|
|7|F2G_RCMMC12|2|G|3|14-10,20-19,6-15,8-20|20-15|14-7|
|8|F6C_RCMMC12|6|C|3|14-7,20-15,20-19,6-15|14-10|8-20|
|9|F3C_RCMMC12|3|C|4|14-10,14-7,6-15,8-20|20-19|20-15|
|10|F3G_RCMMC12|3|G|5|14-10,14-7,6-15,8-20|20-19|20-15|
|11|F4C_RCMMC12|4|C|6|14-10,14-7,20-15,8-20|6-15|20-19|
|12|F4G_RCMMC12|4|G|7|14-10,14-7,20-15,8-20|6-15|20-19|

每候选依次产生clean、LEO/binding和proxy三步；12候选完成后按F1--F6串行产生6个pair，共42步。候选内部串行，每GPU最多2个候选进程。

## 5.N607发布预登记

- 普通账号目标：`N607`；禁止管理员账号。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc_postfreeze_20260811_v1_aabd8358`
- CWD：上述release的`code`目录。
- immutable训练根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcmmc12_20260811_v1`
- postfreeze根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcmmc_postfreeze_20260811_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcmmc_postfreeze_20260811_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcmmc_postfreeze_20260811_v1_launcher.out`
- retry：`NO`；启动所有权：唯一Runner；主控不得重复启动。

冻结唯一命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc_postfreeze_20260811_v1_aabd8358/code && nohup env POSTFREEZE_RUN_ID=phase1_rcmmc_postfreeze_20260811_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc_postfreeze_20260811_v1_aabd8358/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcmmc12_20260811_v1 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcmmc_postfreeze_20260811_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcmmc_postfreeze_20260811_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc_postfreeze_20260811_v1_aabd8358/code/scripts/launch_phase1_rcmmc_postfreeze_20260811.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcmmc_postfreeze_20260811_v1_launcher.out 2>&1 < /dev/null &
```

Runner落地前必须完成：direct preflight；LF无prefix archive、六成员SHA/mode、`code/code=0`；ManySig SHA；12个RCMMC final checkpoint存在、SHA记录、receipt由当前validator重开；远端py_compile/help/bash-n/dry-run42；release/run/log/outer不存在；GPU占用记录。SSH超时后先清理本地ssh/TCP22，再只读确认是否landed，禁止重发。

## 6.技术健康、停止与工件

技术停止只限P0协议/权限/checkout/hash/输出覆盖、launcher-wide确定性故障、至少2个不同候选同一标准化异常指纹、OOM/CUDA/argparse/路径权限错误，或工件闭合失败。不得因accuracy、floor、AUROC、u-gap或任何中间性能停止、重试或调参。停止前必须绑定本run的PID/CWD/cmdline，只处理本run进程并保留部分工件；技术失败记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。

预期工件：12 clean NPZ、12 LEO NPZ、12 LEO binding JSON、12 proxy JSON、12 proxy CSV、6 pair JSON、12候选日志、6 pair日志、`candidate_pids.tsv`和outer。Runner只核技术schema、输入绑定、计数、SHA、异常与F6 raw-reopen键；回收仅JSON/CSV/binding/log/PID/manifest的小bundle，排除NPZ、pth、pt和npy。性能字段由主控在技术闭合并回收后读取。

## 7.冻结非补偿门与结果表

|冻结门|要求|当前结果|判定|
|---|---:|---:|---|
|技术绑定|6/6 pair|待运行|PENDING|
|clean四floor|6/6 fold，每项Δ≥-2pp|待运行|PENDING|
|LEO四floor|18/18 scene-cell，每项Δ≥-2pp|待运行|PENDING|
|逐fold三场景overall|6/6 fold等权Δ≥0|待运行|PENDING|
|全18格overall|等权Δ≥0|待运行|PENDING|
|fixed400 proxy双门|6/6 fold同时ΔAUROC>0且Δu-gap>0|待运行|PENDING|

最终分析必须按同fold、同scene保留C/G完整行，并由主控从6份pair JSON独立重算后与F6 aggregate核对。禁止用不同fold或不同候选的单项极值拼接结论。

## 8.已知风险与下一检查点

- 首次真实postfreeze validator可能暴露训练receipt或sealed artifact字段漂移；这是合法技术阻断，不得放宽合同或补造字段。
- F6必须重开5份prior raw artifacts；任何SHA、common binding、proxy重算或prior receipt篡改均须fail-closed。
- 运行期release可能生成`__pycache__`；只记录其为运行时副作用，不得因此修改科学工件或重跑。
- 下一检查点：训练Runner给出12/12逐臂技术合同、小bundle/report最终SHA后，主控更新本报告训练输入证据、提交mirror，并把本报告连同commit/六SHA/唯一命令交给新的单一N607 Runner。
