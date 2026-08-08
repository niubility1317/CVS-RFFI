# P1-PAMR六折postfreeze正式评估v2

目标模式：`GOAL_MODE=ACTIVE`

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.目标、历史与冻结假设

实验ID：`phase1_pamr12_20260809_v1_postfreeze_v2`。日期：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

本实验只评估`phase1_pamr12_20260809_v1`的六折C/G final checkpoint，不重训、不调参、不选fold。PAMR以同physical clean观测的detached raw-cosine分类margin约束LEO观测正确类—最难异类angular margin，固定`lambda_pamr=0.05`、40epoch；不做显式z、RX或domain对齐，不加GRL/MMD/CORAL、EMA/外部teacher、新head或拒识阈值。proxy和LEO零fit、零校准、零选参。

v1=`phase1_pamr12_20260809_v1_postfreeze_v1`已固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`：exporter在合法satellite profile、`satellite_tta_policy=none`时逐行写运行时TTA view=`single`，而旧pair evaluator错误要求逐行`satellite`。v1只完成12 clean、12 LEO和12 proxy步骤，pair 0/6；不复用其partial，不作性能判断。

v2最小修复commit=`c9b3fb313ec979a5a1105b22312903f0437b98db`：评估器严格要求行级`channel_views=single`，同时独立强制manifest source profile=`satellite`、TTA=`none`、`simplified_leo_residual`、三scenario/seed/physical/TX/RX/order/strict/checkpoint/local4闭合；任意其他非空view、伪造profile或TTA均fail-closed。独立复核：`P0=0 / P1=0 / ALLOW`。

## 2.冻结矩阵与输入

- 训练run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1`
- 训练commit：`ee31aa3345f9e5e3251bacf7de17098377b67bc0`
- postfreeze修复commit：`c9b3fb313ec979a5a1105b22312903f0437b98db`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- source days：`2021_03_01,2021_03_08`
- source RX：`1-1,1-19,14-7,18-2,19-2,2-1`
- scenarios：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- seeds：source satellite=`7281718`；export=`7281105`；每TX上限=`400`

|Fold|train TX|known-validation TX|proxy TX|C/G GPU|
|---|---|---|---|---|
|F1|20-15,20-19,6-15,8-20|14-7|14-10|0/1|
|F2|14-10,20-19,6-15,8-20|20-15|14-7|2/3|
|F3|14-10,14-7,6-15,8-20|20-19|20-15|4/5|
|F4|14-10,14-7,20-15,8-20|6-15|20-19|6/7|
|F5|14-10,14-7,20-15,20-19|8-20|6-15|1/0|
|F6|14-7,20-15,20-19,6-15|14-10|8-20|3/2|

冻结42步：12 clean export、12三场景LEO export、12 proxy score、6 C/G pair score。GPU0=F1C+F5G，GPU1=F1G+F5C，GPU2=F2C+F6G，GPU3=F2G+F6C，GPU4=F3C，GPU5=F3G，GPU6=F4C，GPU7=F4G；每卡最多2个export。

## 3.本地放行证据

- evaluator SHA256=`f9325379b04fce80ad4d669d8f531a41ef8637f2209cb7a14670db68c89ffb3b`
- test SHA256=`6883a64fd61d7570d64807c43110f100d7c0280ead12322a4d240938a77d33aa`
- launcher未变，SHA256=`0c8959ba46f42e4cfbdd2a08adfde594093561cc9619d4ec39a75d484c9273a6`
- pycompile通过；PAMR+postfreeze focused 33 passed；bash-n通过；dry-run精确42步；diff-check通过。

## 4.N607不可覆盖路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v2_c9b3fb31`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1_postfreeze_v2`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_postfreeze_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_postfreeze_v2.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v2_c9b3fb31/code && nohup setsid env POSTFREEZE_RUN_ID=phase1_pamr12_20260809_v1_postfreeze_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v2_c9b3fb31/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1_postfreeze_v2 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_postfreeze_v2 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v2_c9b3fb31/code/scripts/launch_phase1_pamr_postfreeze_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_postfreeze_v2.launch.out 2>&1 < /dev/null & echo $!
```

唯一启动，retry=`NO`；v1 release/run/log全部不可触碰。

## 5.技术健康、停止与artifact

只因P0/路径/hash/覆盖写风险、OOM/CUDA、至少2候选同一确定性异常、零输出或预注册无进展停止；绝不因accuracy、floor、margin、FAR或AUROC停止。成功要求42步全完成、12 clean NPZ、12 LEO NPZ、12 proxy JSON/CSV、6 pair JSON及全部strict/head/checkpoint/local4/order/physical/scenario/TX/RX/channel闭合。回收仅小JSON、CSV、日志、receipt、completion、manifest；不下载NPZ/checkpoint。

## 6.预注册非补偿裁决门

1. 技术闭环全部通过。
2. clean known：六折每折overall、min-class、min-RX、min-day的`G-C>=0`。
3. LEO overall：18/18个fold×scenario的`G-C>=0`。
4. LEO floor：18/18个单元的min-class、min-RX、min-day均`G-C>=0`。
5. angular margin：18/18个单元的正确类—最难异类raw-cosine margin`G-C>=0`。

proxy AUROC/FAR与paired cosine仅诊断，不能补偿。全部通过才可标`PROMOTE_P1_PAMR_FOR_PHASE1_BUNDLE_EVIDENCE`；任一失败即`REJECT_P1_PAMR_PERMANENT`，不调lambda、不改sampler、不挑fold、不重试、不进入Phase3。

## 7.运行终态与结果

v2按冻结命令唯一启动一次，未重试、未复用或修改v1 partial。caller等待超时后清理本地SSH残留并只读确认已落地；没有重复启动。release为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v2_c9b3fb31`，CWD为`<release>/code`，postfreeze修复commit=`c9b3fb313ec979a5a1105b22312903f0437b98db`，无prefix归档SHA256=`bc571ba2e6af4e1be03f9556ecd358fb171e0e21f31d410f69c722b5fd6966b6`（261570560 bytes）。worktree与远端archive成员双hash、compile/help/bash-n、dry-run=42证据见manifest。

launcher PID=`3993`，candidate PID/GPU由`candidate_pids.tsv`固定记录：`F1C 3996/0; F5G 3997/0; F1G 3998/1; F5C 3999/1; F2C 4001/2; F6G 4002/2; F2G 4004/3; F6C 4007/3; F3C 4009/4; F3G 4010/5; F4C 4012/6; F4G 4015/7`。12个candidate pipeline均完成clean/LEO export与proxy score：clean NPZ=12（每个2400行、`channel_views=clean`）、LEO NPZ=12（每个1600行、source-only、三场景齐全、修复后行级`channel_views=single`）、proxy metrics JSON=12、proxy scores CSV=12。首个pair `F1_C_vs_G`阶段发生执行级系统故障；outer精确记录`launch_phase1_pamr_postfreeze_20260809.sh: line 196: 7853 Segmentation fault (core dumped) ... eval_phase1_pamr_pair.py`，F1 pair日志为空、pair JSON=0，launcher退出码=139并因该故障停止，F2–F6 pair未启动。该停止不涉及性能值。

|阶段|完成/总数|技术状态|
|---|---:|---|
|clean export|12/12|每NPZ 2400行；metadata clean/source+target_old+proxy_unknown|
|LEO export|12/12|每NPZ 1600行；source-only/三场景；行级channel_view=single、TTA=none、simplified_leo_residual|
|proxy score|12/12|JSON/CSV均存在；仅partial技术artifact|
|pair score|0/6|F1 pair segfault，F2–F6未启动|

partial小artifact已回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_pamr12_20260809_v1_postfreeze_v2\artifacts`：41个文件（40条manifest条目+manifest），manifest SHA256=`2ec907b13c9c35c30d0ef00f4a9071b56cae5a6e0935332504a57034bf97e975`，completion.tsv SHA256=`41cefb01d3ad08ba2309e2d18ac944344193789c4a2ef6fd871172a93cfd6c57`；逐项bytes/SHA匹配，无NPZ/checkpoint下载。传输包SHA256=`9292068a9de66e178aa79b48cc7da359e193309dc1f62a3ce10bc7a06e7c0724`，远端临时包已删除；GPU、run-owned进程、SSH/TCP22均已清理。v2状态固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，retry=`NO`。
