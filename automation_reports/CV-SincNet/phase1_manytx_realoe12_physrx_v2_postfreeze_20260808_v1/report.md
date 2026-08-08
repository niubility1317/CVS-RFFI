# Phase1 ManyTx RealOE12 Postfreeze One-shot报告

目标模式：`GOAL_MODE=ACTIVE`

## 1.状态与目的

- run ID：`phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`
- 状态：`LOCAL_VERIFIED / PREREGISTERED`
- 主控：`/root`；唯一N607运行器：`/root/n607_geosat_lite_runner`
- 类型：`POSTFREEZE_ONLY / NO_TRAINING / SOURCE_ONLY_NON_CONFIRMATORY`
- exporter repair提交：`f2004a2658dfbbeeb0801550fe539f71e7ddbd1f`
- one-shot launcher提交：`222fa5e4ad2f4581e0371f46cb60c3bde297734d`
- v2训练/失败审计终态提交：`7ada1924a7b00f41ee8e31f6a61c26d66244ad6b`

根目录`E:\type10-7`不是Git仓库；本报告同时保存在Git工作树和`E:\type10-7\automation_reports\CV-SincNet\phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1\report.md`。

目的仅为读取已完成v2训练的12个final checkpoint，完成冻结的12次特征导出和24次energy-only评分。不得调用`train_ssdg.py`，不得改变或重选checkpoint、方法、fold、TX、receiver、seed、阈值或gate。

v2训练已12/12 E120、final/terminal/metrics齐全；第一次postfreeze因exporter把physical day`2021_03_01`转为`20210301`而在NPZ前统一失败。最终repair只在exporter轴解析中规定“纯十进制才是index”，保留日期和RX physical label。独立复核为`P0=0`、`P1=0`、`ALLOW_POSTFREEZE_ONESHOT_RELEASE=YES`；33 focused tests、`py_compile`、exporter help、launcher`bash -n`与DRY_RUN 12 export+24 score均通过。

## 2.冻结输入

- 训练run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_20260808_v2`。
- ManySig SHA256：`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- ManyTx SHA256：`c0319174d40eb64bc49f201743941ebedc5cc0ced284c655cab798b2bdd44275`。

|candidate|final checkpoint SHA256|
|---|---|
|F1C|`4086cafd0fa05d9621264fd37cec5f5779517777de21f099b9df2dea5460e6c3`|
|F1G|`cb019729764aea44eaf95d3d57e7d2d8ccbb3b575ddd3641001cc18dc3fec686`|
|F2C|`d21bf5f1049d6a5ef51b0851722457d9d673d5fc935eb00005a79fbd21d4a579`|
|F2G|`f541a24d8279ea3ff9c22c63852fa3728146e347ec42a5dbfd1aa4385471ceee`|
|F3C|`34cddec748ce544e7221fb4647e98aa8393da5ffcf7c1e7044810b46338a453c`|
|F3G|`f1c6789e8e959650978b385e1336d742dd702fd5c412fb428bbafb6c48765d7d`|
|F4C|`5e803c60c407b0116edb638319266654b6f3192c914b76b2ea3b038b3cef5720`|
|F4G|`57210fe000be1d3c868f72b6fa44fa39eb2551f716a2e713270044db2c29e0da`|
|F5C|`9a258f4a03b48032da58df7dc1d89f95d23899d0306f2d3108abf10f4f06c7fe`|
|F5G|`b73287aab49717f856891f12ffac48cc46206b0c71a6628ad627df020fa68e02`|
|F6C|`019bba176743a4059575a83b6f967c4de3bfba329d71f643af909d8a0b3dc505`|
|F6G|`e6b26792ffba2b7ee346565983a70ccce2b6cd87c286d6470ebf4dc1540e91d0`|

## 3.冻结入口与输出

|文件|SHA256|
|---|---|
|`code/export_spaceborne_features.py`|`89894da5a24baaf75d718b374cfea7dfeea565d24bc54f6e84d15275729bda20`|
|`code/tests/test_export_spaceborne_features_axis_labels.py`|`89d0e4f2958e333bfc768885eb2e150b24f946ce2e4f097df855b2223c01bdd2`|
|`code/scripts/launch_phase1_manytx_realoe_postfreeze_oneshot_20260808.sh`|`c15681bba47ad4c592ce118fa0519a69d629a1eedf13ccc9452e2f731837c186`|
|`code/scripts/eval_phase1_logits_open_set_reject.py`|`c97e280357b1b7316a27684d7931fbc5bd1cc1e756a32106885e2a1362b42b1a`|

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1_<COMMIT8>`。
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`。
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1.launch.out`。

```bash
cd <release>/code && nohup setsid env RUN_ID=phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_20260808_v2 bash <release>/code/scripts/launch_phase1_manytx_realoe_postfreeze_oneshot_20260808.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 4.冻结矩阵与门

12个export沿用训练GPU映射，每卡最多2个；source/held从ManySig读取，proxy20从ManyTx读取。physical days=`2021_03_01,2021_03_08`，physical RX=`1-1,1-19,14-7,18-2,19-2,2-1`，clean、equalized1、每TX 400、seed7281105。每个NPZ预期10400行：source2000、target_old400、proxy_unknown8000；三角色TX互斥，strict checkpoint load为0 missing/0 unexpected/0 skipped。

每candidate随后顺序执行proxy和fold-held两条CPU score，共24条：source正确样本校准，energy Q0.95，confidence/margin gate关闭，`unknown_far_target=0.05`。每条命令只执行一次，`retry=NO`。任何失败保留partial并终止本修复路线，不再新增repair。

运行器只核路径、输入hash、退出码、行数、角色/TX互斥、strict load、artifact hash、进程/GPU/SSH清理；不读取或解释性能。仅回收JSON、CSV、stdout、completion和manifest，不下载checkpoint/NPZ。

## 5.分析边界

主控在完整artifact返回后从score CSV逐样本`energy`重算energy AUROC，并按同fold C/G保留完整同行指标。known保护下降超过2pp时该fold G拒绝；正式`unknown FAR<=5%`不可被known精度补偿。本实验仍只提供Phase1 source-side开发证据，不构成K-shot、target unknown或Phase3正式声明。

## 6.运行器闭环证据（2026-08-08）

- 状态：远端`ARTIFACTS_COMPLETE`；本地小artifact回收在SSH服务临时拒绝后为`RETRIEVAL_PARTIAL_BLOCKED`，不影响远端run，也未重试任何export/score。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1_1e9ebcd0`；commit=`1e9ebcd02d9edfa410ac4493103ba67f49cd28c6`；本地git-archive tar SHA256=`b1a95d1e07b6b92312c2ac2fb7dcf88d7f83f41a7527cad4ceaede67ef85caba`（260884480 bytes）。归档规范哈希：exporter=`2f5ba4d2c270808b0e0ce37fba5159ffc95c91fa997d6a1ee6c26ad5035004d1`、axis test=`579fdda25e141c362e5306ad7c69f62660fd2be72368b825f859e8f1b1c59d40`、launcher=`c15681bba47ad4c592ce118fa0519a69d629a1eedf13ccc9452e2f731837c186`、scorer=`ac1cf8e45fadbc0782282625300a0c30936fdce4b2df241558dff979dafe5c04`。Windows工作树CRLF口径不同，仅记录归档字节哈希，未改远端代码。
- exact launch：上一节命令原样执行一次；caller超时后只读确认launcher已落地，未重复启动。远端最终无run-owned进程，GPU0--7均为0%/1MiB。
- 结构闭环：`export_completion.tsv`=12行、全部`exit_code=0`；12/12 NPZ各10400行，角色计数均为`source=2000,target_old=400,proxy_unknown=8000`，days均为`2021_03_01,2021_03_08`，channel=`clean`，strict load=`missing=0,unexpected=0,skipped=0`。`score_completion.tsv`=24行、全部`exit_code=0`；24 JSON、24 CSV均存在，每CSV=10401行；36 stdout日志存在；错误指纹扫描无Traceback/ValueError/RuntimeError/CUDA/ERROR/FAILED。
- 本地已回收8个candidate的完整7文件（56文件）及F5C的`proxy_metrics.json`（共57文件）；F5C/F5G/F6C/F6G其余小文件、两份completion TSV和outer log待SSH服务恢复后补传。已确认本地无残留`ssh.exe`/`scp.exe`及TCP22连接。未下载NPZ或checkpoint；不作性能结论。
- 2026-08-08 resume retrieval：单次direct N607 preflight与单次已验证lab bridge均因`Connection refused`失败；身份/本地SSH配置正常，随后确认本地SSH进程与TCP22连接均为0。未重复抓取已有文件，未改变远端状态。

## 7.F1--F4已回收同行诊断

本节只使用本地已完整回收的F1--F4八个candidate；F5/F6文件仍在远端，因此明确标记为`PARTIAL_DIAGNOSTIC`，不得据此声称完整open-world矩阵。energy AUROC由逐样本`energy`重算，正类为对应unknown role；FAR使用source正确样本Q0.95校准的energy-only gate。

|Fold|Arm|proxy energy AUROC|proxy FAR(%)|fold-held energy AUROC|fold-held FAR(%)|source known coverage(%)|
|---|---|---:|---:|---:|---:|---:|
|F1|C|0.77974|52.838|0.53623|77.50|93.85|
|F1|G|0.87886|42.437|0.62734|88.75|94.10|
|F2|C|0.75296|54.413|0.28098|94.00|94.50|
|F2|G|0.89399|39.475|0.50246|95.75|94.40|
|F3|C|0.72667|63.937|0.75954|61.25|93.95|
|F3|G|0.88028|41.800|0.62151|96.50|94.45|
|F4|C|0.75922|43.450|0.44631|83.75|94.30|
|F4|G|0.89268|32.625|0.63610|89.50|94.15|

在已回收4fold中，G相对C的proxy energy AUROC平均提高0.13180，proxy FAR平均降低14.575pp；但G的proxy FAR仍为32.625%--42.437%，远高于5%目标。同时fold-held FAR平均从79.125%恶化到92.625%（+13.50pp）。这说明固定RealOE损失学到了ManyTx proxy分布的energy分离，却没有形成可迁移的held-TX拒识边界；F3甚至出现proxy改善而fold-held FAR从61.25%升至96.50%的典型反向迁移。

结合完整6fold已知保护门仅1/6通过，本方法结论冻结为`REJECT_NO_BUNDLE_PROMOTION`。缺失的F5/F6小artifact仍需回收以补齐完整表，但不会改变这一由非补偿known门决定的晋级结论。
