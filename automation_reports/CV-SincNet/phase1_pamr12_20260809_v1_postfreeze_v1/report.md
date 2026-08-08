# P1-PAMR六折postfreeze正式评估报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.目标与假设

实验ID：`phase1_pamr12_20260809_v1_postfreeze_v1`。日期：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

本实验只评估已完成的P1-PAMR六折C/G配对checkpoint，不重训、不调参、不选fold。PAMR用同physical clean观测的detached raw-cosine分类margin约束LEO观测的正确类—最难异类角margin，固定`lambda_pamr=0.05`、40epoch；无显式z对齐、RX/domain对齐、GRL/MMD/CORAL、EMA/外部teacher、新head或拒识阈值。proxy与LEO均零fit、零校准、零选参。

假设：相较同fold GeoSat-C控制臂，PAMR在不损害clean已知识别的同时，能在三个冻结LEO弱场景保持或提高四项已知floor，并让正确类—最难异类angular margin不下降。proxy仅作为source-held未知代理诊断，不能补偿LEO或clean失败。

## 2.冻结输入与矩阵

- 训练run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1`
- 训练实现commit：`ee31aa3345f9e5e3251bacf7de17098377b67bc0`
- postfreeze实现commit：`20f43b0e774cf6ca922796cc298e4fa43960e517`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- source days：`2021_03_01,2021_03_08`
- source RX：`1-1,1-19,14-7,18-2,19-2,2-1`
- LEO scenarios：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- seeds：source satellite=`7281718`；export=`7281105`
- 每TX上限：`400`

|Fold|train TX|known-validation TX|proxy TX|C/G GPU|
|---|---|---|---|---|
|F1|20-15,20-19,6-15,8-20|14-7|14-10|0/1|
|F2|14-10,20-19,6-15,8-20|20-15|14-7|2/3|
|F3|14-10,14-7,6-15,8-20|20-19|20-15|4/5|
|F4|14-10,14-7,20-15,8-20|6-15|20-19|6/7|
|F5|14-10,14-7,20-15,20-19|8-20|6-15|1/0|
|F6|14-7,20-15,20-19,6-15|14-10|8-20|3/2|

冻结42步：12个clean export、12个三场景LEO export、12个proxy score、6个C/G pair score。GPU0=F1C+F5G，GPU1=F1G+F5C，GPU2=F2C+F6G，GPU3=F2G+F6C，GPU4=F3C，GPU5=F3G，GPU6=F4C，GPU7=F4G；每卡最多2个export，CPU评分在export闭环后执行。

## 3.实现与本地放行证据

新增文件：

- `code/scripts/eval_phase1_pamr_pair.py`，SHA256=`5c51e95d02373e74e4f409243a0a4fef2ccdf5d06db508ce0ec4550768918544`
- `code/scripts/launch_phase1_pamr_postfreeze_20260809.sh`，SHA256=`0c8959ba46f42e4cfbdd2a08adfde594093561cc9619d4ec39a75d484c9273a6`
- `code/tests/test_phase1_pamr_postfreeze.py`，SHA256=`8f97cbb082613346e4babe9984199af72334cff2eb676c3e15914c366dc13fd2`

本地验证：py_compile通过；`test_phase1_pamr.py + test_phase1_pamr_postfreeze.py`共31 passed；launcher `bash -n`通过；dry-run精确42步；`git diff --check`通过。独立复核结论：`P0=0 / P1=0 / ALLOW`。评估器强绑定final checkpoint、exact `id_backbone.cls_head.head.weight`、NPZ SHA、strict-load、local4类序、row order、LEO逐行`channel_views=satellite`与manifest source profile；outer数值扰动不影响readout或pair scoring。

## 4.N607不可覆盖路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v1_20f43b0e`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1_postfreeze_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_postfreeze_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_postfreeze_v1.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<release>/code`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v1_20f43b0e/code && nohup setsid env POSTFREEZE_RUN_ID=phase1_pamr12_20260809_v1_postfreeze_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v1_20f43b0e/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1_postfreeze_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_postfreeze_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v1_20f43b0e/code/scripts/launch_phase1_pamr_postfreeze_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_postfreeze_v1.launch.out 2>&1 < /dev/null & echo $!
```

只允许上述唯一启动；retry=`NO`。若caller超时，先清理本地SSH残留并只读确认是否landed，禁止重复启动。

## 5.技术健康、artifact与停止规则

启动后核launcher/CWD/cmdline/run-root、12个候选GPU映射、日志增长与42步完成数。技术成功要求：12 clean NPZ、12 LEO NPZ、12 proxy JSON/CSV、6 pair JSON、全部completion/strict-load/checkpoint-binding/row-order/physical/scenario/TX/RX/channel闭合，且日志无OOM/CUDA/Traceback/确定性异常。

仅以下情况停止：P0协议/路径/hash/覆盖或覆盖写风险；OOM/CUDA；至少2个不同候选出现同一确定性异常；零输出或预注册无进展。绝不根据accuracy、floor、margin、FAR或AUROC中止。若停止，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，保留partial，不重启。

回收范围仅小型JSON、CSV、日志、completion、receipt、manifest；不下载checkpoint或NPZ。runner只做技术闭环，不读取或解释性能。

## 6.预注册非补偿裁决门

主控只在全部artifact返回后读取完整同排证据，并按以下冻结门逐项判定：

1. 技术闭环：全部42步及绑定/strict/row/physical/scenario证据通过。
2. clean known保护：六折中每折`G-C`的overall、min-class、min-RX、min-day均`>=0`。
3. LEO overall：18个fold×scenario单元全部`G-C>=0`。
4. LEO floor：18个单元的min-class、min-RX、min-day全部`G-C>=0`。
5. angular margin：18个单元的正确类—最难异类raw-cosine margin全部`G-C>=0`。

proxy AUROC/FAR与paired cosine只作机制诊断，不能补偿任一门。全部五门通过才可标`PROMOTE_P1_PAMR_FOR_PHASE1_BUNDLE_EVIDENCE`；任一失败即`REJECT_P1_PAMR_PERMANENT`，不调`lambda`、不改sampler、不挑fold、不重试、不进入Phase3。

## 7.运行终态与结果

本run按冻结命令唯一启动一次，未重试、未修改代码或训练输入。caller等待超时后清理本地SSH残留并以短连接确认已落地；没有重复启动。release为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr12_postfreeze_20260809_v1_20f43b0e`，CWD为`<release>/code`，postfreeze commit=`20f43b0e774cf6ca922796cc298e4fa43960e517`，无prefix归档SHA256=`4472aed7a7c346cbbd51f932370e9b6da5da081cf1c225f9dc114305891f57c8`（261560320 bytes）。远端archive成员以LF字节核验；worktree与archive的换行口径双hash及compile/help/bash-n/dry-run=42证据见manifest。

launcher PID=`4186877`，candidate PID/GPU由`candidate_pids.tsv`固定记录：`F1C 4186880/0; F5G 4186881/0; F1G 4186882/1; F5C 4186883/1; F2C 4186886/2; F6G 4186887/2; F2G 4186888/3; F6C 4186891/3; F3C 4186893/4; F3G 4186897/5; F4C 4186899/6; F4G 4186902/7`。12个candidate pipeline均完成clean/LEO export与proxy score：clean NPZ=12（每个2400行、`channel_views=clean`）、LEO NPZ=12（每个1600行、source-only、三场景齐全）、proxy metrics JSON=12、proxy scores CSV=12。首个pair `F1_C_vs_G`退出1，错误指纹为`PAMRPostfreezePairError: C LEO payload must use exactly channel_view=satellite`；只读metadata显示全部LEO NPZ的`channel_views=['single']`，而冻结pair要求`satellite`，属于launcher-wide确定性协议闭环故障。launcher因`set -e`在F1 pair后退出1，F2–F6 pair未启动；pair JSON=0、pair score CSV=0、pair日志=1。该停止不涉及性能值。

|阶段|完成/总数|技术状态|
|---|---:|---|
|clean export|12/12|每NPZ 2400行；metadata clean/source+target_old+proxy_unknown|
|LEO export|12/12|每NPZ 1600行；source-only/三场景；channel_view错误为single（预期satellite）|
|proxy score|12/12|JSON/CSV均存在；仅作partial技术artifact|
|pair score|0/6|F1首个命令确定性校验失败，F2–F6未启动|

partial小artifact已回收至`E:\type10-7\automation_reports\CV-SincNet\phase1_pamr12_20260809_v1_postfreeze_v1\artifacts`：41个文件（40条manifest条目+manifest），manifest SHA256=`baa37c7a67af497055fecc23c7d52de43c72519b961dae2101b0c98d58451e2e`，completion.tsv SHA256=`1003321f930cac72611d9eba4899854fa2922e3d20d2d3e023944eed3c37d6dd`；逐项bytes/SHA匹配，无NPZ/checkpoint下载。传输包SHA256=`39190d58823bc215c53f27c18092757b60df4873c9fc99b4260654fe8a516920`，远端临时包已删除；GPU、run-owned进程、SSH/TCP22均已清理。保留远端partial NPZ仅供主控后续技术修复审计；本run状态固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，retry=`NO`。
