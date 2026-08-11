# Phase1 HNCCD后冻结42步实验报告

状态：`ARTIFACTS_COMPLETE / TECHNICALLY_CLOSED / P0=0 / P1=0 / NO_PERFORMANCE_RESULT`

## 1.实验身份与目标

- 后冻结ID：`phase1_hnccd_postfreeze_20260811_v1`
- 训练输入ID：`phase1_hnccd12_20260811_v1`
- 日期：2026-08-11
- 操作方：Codex主控；N607唯一Runner已完成技术闭合
- 后冻结实现commit：`fa30a77032e9db4acbf8efb17d4a81ab8dd37dc8`
- 训练实现commit：`b6afc5a3e19ae3146dd6afcfe8a90abff35f3cbb`
- 候选：P1-HNCCD（head-nullspace cross-covariance decorrelation）
- 目标：对12个`training_final_only`checkpoint执行固定42步，完整比较同fold C/G在clean、三scene LEO和fixed400 proxy上的连续几何与分类稳定性，并给出唯一非补偿判定
- 声明边界：本轮只检验Phase1 source-known连续几何，不训练或声明真实unknown、FAR、注册授权、Phase2、Phase3或多卫星协同能力

训练实验已由唯一Runner技术闭合：12/12 final/checkpoint和HNCCD terminal receipt完整，C辅助N/A/0，G均1200batch、三scene各400、VJP与资源逐批闭合，无failure receipt或真实技术异常。统一`NON_PROMOTABLE_P0_DISABLED/exit8`是冻结trainer的非晋级guard，不是训练失败。训练source-val仅作sanity，不替代本轮sealed后冻结门。

## 2.冻结后冻结合同

固定42步：

```text
12 clean feature/logit export
+12 single-LEO three-scene feature export and physical binding
+12 fixed400 proxy score binding
+6 same-fold C/G pair evaluation
=42
```

每个clean导出必须从当前`final_ssdg.pth`重开真实`hnccd_receipt`，验证B128、d160、local4、fixed28、source receiver计数/SHA、same-physical顺序、三scene、strict model keys、新AdamW、AMP、共同`L_base`路径与逐批资源观察。C必须aux N/A/0；G必须lambda0.02、三scene positive与raw-unscaled VJP闭合。

后冻结几何严格只在source-L clean`feat_joint`上fit：float64 totalized-L2保留精确零行，对角Gaussian使用每类`ddof=1`、`0.9×class+0.1×class-equal pooled`收缩和`1e-6`方差下限；V、LEO、proxy均只评分、绝不fit。任何feature、norm、geometry、NLL或聚合nonfinite均fail-closed。

LEO只从同一既有IQ生成`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`各一次，不增加第二view、TTA或额外model forward。binding封存TX/RX/day/class/order、scene、ManySig SHA和当前checkpoint/receipt SHA。fixed400 proxy固定days、RX、seed=`7281148`、总数400；proxy只用于连续诊断门，不得反馈训练、调参、重试、停止或候选选择。

F6必须从F1至F5的raw clean NPZ、LEO NPZ、LEO binding、proxy JSON/CSV和当前C/G checkpoint重新计算；禁止信任prior pair自报摘要。

## 3.非补偿判定

|门|冻结要求|补偿规则|
|---|---|---|
|clean四floor|每fold overall、min-class、min-RX、min-day均`G−C≥−2pp`，6/6|任一fold失败即失败|
|LEO四floor|clear/low/rain每scene四floor均`G−C≥−2pp`，18/18|scene、fold之间不可补偿|
|fold三scene overall|每fold三scene等权overall`G−C≥−2pp`，6/6|不可用clean或proxy补偿|
|global 18-cell overall|18个fold×scene等权global overall`G−C≥−2pp`|只作全矩阵总门，不覆盖单fold失败|
|fixed400 proxy|每foldAUROC改善严格正且`mean(u_proxy)-mean(u_V)`改善严格正，6/6|两项及fold之间均不可补偿|

全部门通过才可进入`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`；任一门失败则`REJECT_P1_HNCCD_PERMANENT`，不得调参、挑fold、换checkpoint、改名或拼接复活。

## 4.本地文件、SHA与验证

|文件|SHA256|Git mode|
|---|---|---|
|`analysis/phase1_hnccd_postfreeze_design_20260811.md`|`c1aba90a5abb88c9f9e523823c9513d0829e31d659e0c48f6cb973ab4125e87f`|100644|
|`code/export_phase1_hnccd_features.py`|`d1de22b56f21dfca623a4d010b36d2b9deae553e9d5eefa1528efb678fa38621`|100644|
|`code/export_phase1_hnccd_leo_features.py`|`80ba306db4808ca3102e0657ec7ba50954f78455ee33547223664b12707fed02`|100644|
|`code/evaluate_phase1_hnccd_postfreeze_pair.py`|`43caa293e45966e6ab5d6fc980b4e12d5b55ff52bedd0dfaeae1c478145557c`|100644|
|`code/tests/test_phase1_hnccd_postfreeze.py`|`afdfd3eb3e82b23d7893268968fc6808ef9e0e605880a29c282f2450c1fc9e6b`|100644|
|`code/scripts/launch_phase1_hnccd_postfreeze_20260811.sh`|`2338442dc460df725c26b3d691664c5a5d5b32ea76e5b9771640aec4dc317e3b`|100755|

本地官方Conda hook激活`ssr-gpu`后验证：

- 4个Python文件`py_compile`通过；
- HNCCD core+postfreeze联合49/49通过；
- 3个CLI`--help`通过；
- launcher`bash -n`通过；
- dry-run精确42行：clean12、LEO12、proxy12、pair6；
- 6个pair均`expected-source-count=1600`、`expected-proxy-count=400`，F6唯一携带5个prior路径；
- 嵌套raw receiver token、旧identity、nonfinite、zero-row、checkpoint/binding/proxy/F6原件篡改均有fail-closed覆盖；
- 独立actual-diff审查：`P0=0 / P1=0 / ALLOW`；
- `git diff --check`通过。

追踪卡状态：`verified=5,implemented=4,deferred=0,rejected=0,blocked=0`。真实12 checkpoint、ManySig、sealed42与F6原件已由唯一Runner在N607重开并完成技术闭合。

## 5.冻结矩阵与资源

12个候选为`F1C_HNCCD12`、`F5G_HNCCD12`、`F1G_HNCCD12`、`F5C_HNCCD12`、`F2C_HNCCD12`、`F6G_HNCCD12`、`F2G_HNCCD12`、`F6C_HNCCD12`、`F3C_HNCCD12`、`F3G_HNCCD12`、`F4C_HNCCD12`、`F4G_HNCCD12`。GPU映射固定为`0,0,1,1,2,2,3,3,4,5,6,7`；每个candidate内部依次执行clean、LEO、proxy，12个candidate完成后才串行执行F1至F6 pair。每GPU最多2个candidate进程。

固定source TX、known validation TX和proxy TX均由launcher六fold表给出；不得改变fold、TX、RX、day、scene、seed、400行配额或执行顺序。pair为CPU阶段，F6耗时不设性能停止阈值。

## 6.N607发布预登记

- 普通账号：`N607`；禁止管理员账号
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd_postfreeze_20260811_v1_fa30a770`
- code CWD：上述release的`code`目录
- 训练根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hnccd12_20260811_v1`
- postfreeze根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hnccd_postfreeze_20260811_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd_postfreeze_20260811_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd_postfreeze_20260811_v1_launcher.out`
- retry：`NO`
- 启动所有权：唯一Runner；主控不得重复启动

冻结唯一命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd_postfreeze_20260811_v1_fa30a770/code && nohup env POSTFREEZE_RUN_ID=phase1_hnccd_postfreeze_20260811_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd_postfreeze_20260811_v1_fa30a770/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hnccd12_20260811_v1 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hnccd_postfreeze_20260811_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd_postfreeze_20260811_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd_postfreeze_20260811_v1_fa30a770/code/scripts/launch_phase1_hnccd_postfreeze_20260811.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd_postfreeze_20260811_v1_launcher.out 2>&1 < /dev/null &
```

## 7.唯一Runner启动前硬门

1. direct`tools\n607_ssh_preflight.ps1`通过，普通账号、项目根和8张GPU可见；
2. 从commit`fa30a77032e9db4acbf8efb17d4a81ab8dd37dc8`生成完整LF、无prefix Git archive；验证成员数、`code/code=0`、text CR=0、6文件SHA和launcher mode；
3. release、postfreeze run、log、outer在启动前均为ABSENT；
4. ManySig SHA和训练根12个`final_ssdg.pth`完整SHA逐项记录；
5. release内4文件`py_compile`、3个help、`bash -n`、dry-run42/12/12/12/6/pair1600通过；
6. 当前release validator重开12个training checkpoint/terminal receipt，6折C/G common binding、C辅助0、G三scene/VJP/resource1200全部通过；
7. 只读真实F1C clean export与LEO binding窄smoke通过且query rows opened=0；临时smoke输出不得写入final postfreeze根；
8. GPU并发和本run PID记录后才允许唯一命令调用1次。

任一硬门失败则状态为`PRELAUNCH_BLOCKED / NO_POSTFREEZE_RESULT`，launch保持0；不得改方法、截取head、放宽validator或覆盖路径。

## 8.技术健康、停止与预期工件

预期终态：

- 12个clean NPZ；
- 12个LEO NPZ；
- 12个LEO binding JSON；
- 12个proxy JSON和12个proxy CSV；
- 6个same-fold pair JSON；
- 12个candidate日志、6个pair日志、outer和`candidate_pids.tsv`。

健康停止只允许：P0权限/路径/checkout/hash/覆盖错误、launcher-wide确定性故障、至少2个不同candidate在产生必需输出前出现同一确定性异常、OOM/CUDA/argparse/权限/SIGSEGV、validator失败或zero-artifact。停止前必须绑定本run PID/CWD/cmdline，只处理本run，保留partial并记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_POSTFREEZE_RESULT`。不得因accuracy、floor、AUROC、u-gap或中间pair表现停止、重试或选择。

launch最多1次；SSH超时先只终止并清理绑定本地客户端，再用短只读连接确认是否landed，绝不重发。每次SSH/SCP后确认本地`ssh.exe`和N607/bridge TCP22均为0。

## 9.完成后的主控分析

唯一Runner只核schema、路径、SHA、receipt/common binding、proxy raw复算、F6 raw reopen和技术异常；不得读取或解释accuracy、floor、AUROC或u-gap。Runner回收小bundle时包含JSON、CSV、binding、日志、PID和manifest，排除NPZ、pth、pt、npy；远端原件保留。

只有42步工件技术闭合后，主控才从6个pair JSON及raw绑定读取同row结果，逐项计算clean6/6、LEO18/18、fold/global overall和proxy双门，更新本报告并作唯一永久判定。

## 10.唯一Runner技术回收与终态（2026-08-11）

- 远端完整Git archive来自`fa30a77032e9db4acbf8efb17d4a81ab8dd37dc8`，无prefix、无`code/code`或`code0`，4988个成员；artifact-only LF归一化final archive为`release_git_archive_fa30a770_final_lf.tar`，bytes=`267909120`，SHA256=`260BA78A4689B596343BD486DA46001C52FDF31E98B4A828FD4483D6460DE48B`。raw archive保留为`release_git_archive_fa30a770_raw.tar`，SHA256=`46E208573A0CA5D4815392787E94D78048CC8E2F016FA66F78C8641754870DF1`；raw文本CR证据保留，未改repo。release archive直接SCP一次，远端SHA与final archive一致，atomic landing成功。
- release静态门通过：4个`py_compile`、3个`--help`、`bash -n`、dry-run精确42（clean12/LEO12/proxy12/pair6），pair source1600为6/6、proxy400为6/6、F6 prior为1；release内pycache=0。
- 当前validator技术重开通过：12/12 training checkpoint与HNCCD terminal receipt；6折C/G common binding；C auxiliary=0；G三scene positive、raw-unscaled VJP和resource1200均闭合。真实F1C clean+LEO窄smoke通过，schema/terminal/source-only/binding闭合，`query_rows_opened=0`；临时输出未写入final postfreeze根。
- 最终prelaunch门通过后唯一启动命令调用1次，SSH exit=0，retry=`NO`。12个candidate PID均按`candidate_pids.tsv`与fold/arm/GPU绑定并自然退出；GPU compute apps=0，未发生技术停止、重试或干预。
- 终态技术计数：12 clean NPZ、12 LEO NPZ、12 binding JSON、12 proxy JSON、12 proxy CSV、6 pair JSON、12 candidate log、6 pair log、outer和`candidate_pids.tsv`；schema/receipt/common binding/proxy raw复算/F6 raw reopen validator均PASS。terminal技术扫描JSON=30、CSV=12、stage logs=18，技术异常指纹为空，禁止输出扫描=0。远端完整原件保留。
- 小bundle转移SCP一次。raw SCP证据保留为`phase1_hnccd_postfreeze_20260811_v1_technical_bundle_scp_raw.tar.gz`，bytes=`13384195`，SHA256=`4C63AFBA94191361BA1836A87636B27CF8A20BF3416359C93B6C6E0BAFF307C5`。artifact-only final bundle为`phase1_hnccd_postfreeze_20260811_v1_technical_bundle.tar.gz`，bytes=`4895620`，SHA256=`E01738387D8A854B14DB7CAC0F0AFFEF23CE0B491EAC0BA35997ED2D480DA3D1`，78个tar成员（63文件、15目录、重复成员0），manifest类别为30 JSON、12 CSV、18 logs、outer、pids，forbidden=0，成员bytes/SHA审计PASS。raw SCP tar仅保留传输证据；final为本run artifact-only去重归一化，不改远端原件。
- 每次SSH/SCP后均主动断开并核验本地`ssh.exe=0`、N607/bridge TCP22 established=0；终态仍为0。
- 本节仅记录发布、启动、路径、hash、schema、receipt、binding、资源和工件技术事实；状态为`NO_PERFORMANCE_RESULT / NO_PERFORMANCE_INTERPRETATION`。不得据此读取、解释或判定accuracy、floor、AUROC、u-gap、pair数值或晋级/拒绝。
