# Phase1 CCPC-LEO v4 postfreeze配对评估报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`PREREGISTERED / NOT_LAUNCHED`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.实验目标与冻结假设

实验ID：`phase1_ccpc_leo12_20260809_v4_postfreeze_v1`。时间：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

对已完整训练的六折C/G final-only checkpoint执行一次冻结postfreeze矩阵，回答CCPC是否在不做RX/domain对齐、不训练拒识head、不访问proxy/held/LEO进行fit、校准或选参的前提下，保护clean known并改善LEO表示。训练run为`phase1_ccpc_leo12_20260809_v4`；12个候选均E040、技术终态完整。G相对C仅增加固定`T=0.12、lambda=0.02`的CCPC。

## 2.冻结矩阵与数据角色

共42步：12个clean导出、12个source-only LEO导出、12个source校准proxy连续评分、6个同fold C/G配对评分。每个候选使用自己的`final_ssdg.pth`；C/G同fold继承相同源checkpoint、相同TX划分与seed。clean导出角色固定为source=1600、target_old=400、proxy_unknown=400；LEO导出仅source=1600，场景固定`leo_clear_weak、leo_low_elev_weak、leo_rain_weak`。所有评分为只读，proxy、held和LEO均零fit、零校准、零选参。

GPU映射沿用训练矩阵，每卡最多2条候选pipeline：GPU0=F1C+F5G，GPU1=F1G+F5C，GPU2=F2C+F6G，GPU3=F2G+F6C，GPU4=F3C，GPU5=F3G，GPU6=F4C，GPU7=F4G。候选内部串行执行clean export→LEO export→proxy score；12条候选完成后串行执行6个pair score。

## 3.版本与本地证据

复用已落地不可变release commit：`ad261d2887d867c1993bca2f993f2d7b969000e6`；训练实现commit：`753161c9127f72498507c8bbf4d7994bc4b7e698`。postfreeze文件：

- `code/scripts/eval_phase1_ccpc_leo_pair.py`
- `code/scripts/launch_phase1_ccpc_leo_postfreeze_20260809.sh`
- `code/tests/test_phase1_ccpc_leo_postfreeze.py`

本地`ssr-gpu`验证：postfreeze focused pytest=12 passed，launcher `bash -n`通过，dry-run精确42条；独立复核`APPROVE / Critical=0 / Important=0`。本报告在启动前写入root控制面并镜像到Git承载面。

## 4.N607路径与唯一启动命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28`
- 训练输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4`
- postfreeze run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1`
- postfreeze log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1.launch.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code && nohup setsid env POSTFREEZE_RUN_ID=phase1_ccpc_leo12_20260809_v4_postfreeze_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v4_ad261d28/code/scripts/launch_phase1_ccpc_leo_postfreeze_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v4_postfreeze_v1.launch.out 2>&1 < /dev/null & echo $!
```

## 5.技术门、性能门与预期artifact

启动前核对release hash、12个final checkpoint、ManySig、目标run/log/outer不存在、GPU/活动任务。启动后核launcher/CWD/cmdline、candidate PID/GPU、日志增长、输出计数与异常指纹。仅路径/P0/覆盖错误、输出覆盖风险、OOM/CUDA、至少2条候选同一确定性异常或无进展触发技术停止；不查看性能决定停止，retry=`NO`。

成功artifact：12个clean NPZ、12个LEO NPZ、12个proxy JSON+CSV、6个pair JSON、candidate/pair日志与PID/完成回执、manifest。只回收小JSON/CSV/log/receipt，不下载NPZ/checkpoint。

五项非补偿门：①技术健康；②clean known六折全部overall/minclass/minRX/minday的G-C≥-2pp；③LEO 18个fold×scenario全部四项G-C≥-2pp且aggregate overall改善；④proxy连续排序相对C同向；⑤checkpoint SHA、strict-load、元数据与artifact闭环。任一失败即`REJECT_CCPC_LEO_NO_RETRY`，不进入Phase3。
