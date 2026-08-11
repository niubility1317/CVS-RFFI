# Phase1 P1-RCMMC后冻结42步实验报告

## 1.状态与目标

- 实验ID：`phase1_rcmmc_postfreeze_20260811_v1`
- 日期：2026-08-11
- 当前状态：`PREREGISTERED_LOCAL_VERIFIED / READY_FOR_N607_HANDOFF / NO_PERFORMANCE_RESULT`
- 操作边界：主控冻结评价合同、矩阵和非补偿门；唯一N607 Runner只负责release落地、唯一启动、技术监控与小工件回收，不读取或解释性能字段。
- 训练输入：`phase1_rcmmc12_20260811_v1`，状态=`ARTIFACTS_COMPLETE_TECHNICALLY_CLOSED / PARTIAL_LOCAL_EVIDENCE_RETRIEVAL / NO_PERFORMANCE_RESULT`。12/12 final、terminal、completion及RCMMC/resource/heldout/config收据远端齐全，逐臂checkpoint SHA、C/G合同和四参VJP已只读核验；本地仅部分小证据回收不影响远端不可变训练输入。
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
- 下一检查点：新的单一N607 Runner先做direct preflight；若direct路径不可用而身份与key合法，则按治理使用已验证bridge。它必须重新核12个checkpoint SHA和当前validator receipt、确认release/run/log/outer均不存在，再执行唯一命令一次。

## 9.训练输入技术闭合证据

- 训练报告root/mirror逐字一致，SHA256=`b676127262c1c3312c2345828dbfd9c5bfc618c2d02dcccde11dd15e6d657f8b`，Git镜像commit=`8f9310259b5cf5bf0c0000bd7c0c165fedc7471f`。
- 唯一训练launch=1、retry=NO；12/12候选自然退出，12/12 final、terminal、completion及RCMMC terminal receipt齐全，arm日志均无Traceback、RuntimeError、OOM/CUDA、argparse、权限、路径或Killed指纹，最终GPU与SSH/TCP22清零。
- 六个C臂lambda=0且RCMMC辅助计数为0；六个G臂三scene均28/28 cell、有positive-D，合计每臂1200 batches/153600 rows，四参VJP证明LEO feat_joint与shared encoder finite-nonzero、clean feat_joint与exact head None-or-zero，raw-unscaled且不触碰AMP/optimizer/RNG。
- 训练terminal的`NON_PROMOTABLE_P0_DISABLED/8`来自冻结代码`p0_mechanisms_ready=false`总guard，不是执行异常；所有训练性能声明字段均false。本轮不读取训练heldout或metrics_epoch性能。
- 本地部分技术bundle=`phase1_rcmmc12_20260811_v1_technical_bundle_partial.tar`，SHA256=`cac28754b19d37f2aeb0dab136440bbeca87297bc4b8f0fafb2945320f8e00ea`，161280B、39 files/50 tar entries；未完整回收F5G/F6小副本，12个checkpoint未下载，完整远端run/log/checkpoint仍保留。后冻结Runner必须直接以远端原件重开，不得用该部分bundle代替科学输入。

## 10.Runner发布、唯一启动与技术闭合

- Direct preflight通过：普通账号`N607`、项目根可见、GPU0--7均为0%/1MiB、身份/key有效；每次SSH/SCP完成后本地`ssh.exe`与N607 TCP22均清零。
- 首次release工程输入为六成员LF归档`phase1_rcmmc_postfreeze_20260811_v1_aabd8358_lfnorm.tar`（163840B，SHA=`1973e9e3b28269ea9482e84c5c75ab2d8fbe15d3ea9ab1d9cf70c7a01cec1924`）。它仅用于记录首个不完整incoming；曾短暂原子落地后按P0发布修复精确改名为`..._aabd8358_sixfile_incomplete_prelaunch`，未删除、未覆盖、未启动。
- 从科学实现commit=`aabd8358cb5303e34b546b6e4485afc1575fccf0`生成完整无prefix归档并只做归档层LF规范化：`phase1_rcmmc_postfreeze_20260811_v1_aabd8358_full_lfnorm.tar`，264417280B，SHA=`ebb95db7345dc6b1e362aeac242ba7bf43ff26182d0220db7019161bdaed37fb`，4976 members（4356 files、620 dirs），`code/code=0`，文本成员CR=0；六冻结成员SHA、launcher归档mode=0755及5项依赖均匹配。第二次且最后一次SCP使用`.full.incoming.tar`，累计`SCP=2`；原子stage→final后远端launcher显示0775，归档成员仍为0755，判定为tar/umask权限差异，脚本冻结命令显式使用`bash`，未chmod、未重包。
- 最后prelaunch只读门：final release存在，run/log/outer目标均ABSENT，无关联PID/CWD，8GPU均0%/1MiB；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`，远端4文件`py_compile`、4个CLI help、`bash -n`与冻结dry-run`42=12+12+12+6`均通过，12个checkpoint/current terminal receipt重开、checkpoint/terminal/status/completion/heldout SHA closure、6折C/G common binding、G三scene×28 positive和四参VJP均通过。配置receipt只核schema/method/flags/存在，不冒充terminal receipt。
- 严格逐字执行报告§5冻结命令恰1次，`launch=1`、`retry=NO`。SSH通道约34s超时后仅清理绑定本地客户端，并只读确认已落地；未重发。初始launcher PID=`1551037`、launcher bash PID=`1551038`，`candidate_pids.tsv`记录12候选及GPU映射`0,0,1,1,2,2,3,3,4,5,6,7`。
- 12候选与6个pair均自然退出；每候选日志849行，18份候选/pair日志技术错误指纹（Traceback、RuntimeError、CUDA/OOM、argparse、权限、路径、Killed）为0；终态绑定进程=0，GPU0--7均0%/1MiB，SSH/TCP22=0。远端工件计数为24 NPZ、30 JSON、12 CSV；本Runner不读取或解释任何性能字段。

## 11.终态技术重开与证据回收

- 12/12 raw clean NPZ、12/12 LEO NPZ、12/12 binding由当前release validator重开通过，12/12 proxy JSON/CSV由当前clean NPZ字节独立重算通过；每个pair的schema、postfreeze matrix ID、training/output root、C/G receipt revalidation、common binding、proxy SHA closure均通过。
- F6 pair含`rcmmc_f6_raw_reopen_required=true`及5项`matrix_aggregate.prior_pair_metrics_bindings`，全部`raw_artifacts_recomputed=true`；结合当前F6 C/G receipt/common/proxy重验，F6 raw-reopen技术门通过。该证据不构成性能结论。
- 仅回收小技术bundle，排除NPZ、pth、pt、npy、jsonl及`metrics_epoch`：最终引用`release/phase1_rcmmc_postfreeze_20260811_v1_technical_bundle_v2.tar`（84520960B，76 members=62 files+14 dirs，SHA=`787d420f9ecc1d69d6361d7c17055683c55ed8d2aa35430a84b6584df5351e24`；JSON=30、CSV=12、`.out`=19，含0B outer、`candidate_pids.tsv`=1，禁入=0）。v1首包（遗漏outer）保留为证据；v2外部manifest=`phase1_rcmmc_postfreeze_20260811_v1_technical_bundle_v2_MANIFEST.sha256`，9233B，SHA=`b5216145147c92759987c74d891bed2effeeb4d736b4a25a93cf79660c827859`。远端bundle路径仍保留。
- 当前状态：`ARTIFACTS_COMPLETE_TECHNICALLY_CLOSED / NO_PERFORMANCE_RESULT`。性能读取、解释、晋级和科学结论由主控独立完成；本Runner未因耗时、静默或任何性能字段停止。
