# Phase1 CCPC-LEO六折C/G v2监控修复报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.v1终态与唯一修复

v1在12任务均进入E001且无异常时，被runner把日志中的N/A占位`nan`误判为训练非有限数并提前终止。具体占位包括禁用domain时的`dom=nan%`、尚未执行测试时的`overall_tx=nan%(0/0)`以及未实现统计的`sat_cos=nan`；它们不是`loss_total`、`loss_ccpc`、梯度或模型输出的非有限值。v1保持partial并标`STOPPED_EARLY_MONITOR_CLASSIFICATION_ERROR / NO_PERFORMANCE_RESULT`，不恢复、不覆盖、不重启。

v2仅修正runner健康规则，代码、implementation commit、checkpoint、C/G矩阵、seed、40epoch、loss、GPU映射和输出结构全部不变。N/A占位不得触发停止；只有真实loss/梯度/模型输出非有限、CCPC fail-closed异常、Traceback/OOM/CUDA故障、进程异常退出或无进展才允许停止。

## 2.冻结方法与矩阵

G相对C只增加`P1-CCPC-LEO`：LEO anchor对detached clean bank做class-conditional paired contrastive，同TX clean为正例、batch全部TX clean为分母；固定`T=0.12、lambda=0.02`。C/G从同fold GeoSat-C checkpoint执行严格模型键warm-start，均以新AdamW/AMP状态续训40epoch，`final_only`。禁止RX/domain标签、GRL、MMD、CORAL、proxy/held训练、teacher、拒识head、阈值和扫参。

| Fold | train TX | known-validation TX | proxy TX | C/G GPU |
|---|---|---|---|---|
| F1 | 20-15,20-19,6-15,8-20 | 14-7 | 14-10 | 0/1 |
| F2 | 14-10,20-19,6-15,8-20 | 20-15 | 14-7 | 2/3 |
| F3 | 14-10,14-7,6-15,8-20 | 20-19 | 20-15 | 4/5 |
| F4 | 14-10,14-7,20-15,8-20 | 6-15 | 20-19 | 6/7 |
| F5 | 14-10,14-7,20-15,20-19 | 8-20 | 6-15 | 1/0 |
| F6 | 14-7,20-15,20-19,6-15 | 14-10 | 8-20 | 3/2 |

并发映射固定为GPU0:F1C+F5G、GPU1:F1G+F5C、GPU2:F2C+F6G、GPU3:F2G+F6C、GPU4:F3C、GPU5:F3G、GPU6:F4C、GPU7:F4G。

## 3.版本与验证

implementation commit：`e999a6c526dc676dfa0ce193b00ce11cac3d308c`。代码与v1完全相同；5文件SHA和本地10/10测试、py_compile、bash-n、12行dry-run证据见v1报告。独立代码复核为`APPROVE / Critical=0 / Important=0`。

v2发布前只需只读确认v1 release内archive/hash/compile结果仍匹配、v2 run/log/outer不存在、v1与其他训练进程为空、GPU资源满足每卡不超过2个任务；不重新构建数据或修改远端代码。

## 4.N607冻结路径与命令

- 复用只读release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v1_e999a6c5`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v2`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v2.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v1_e999a6c5/code && nohup setsid env RUN_ID=phase1_ccpc_leo12_20260809_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v1_e999a6c5/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GEOSAT_CKPT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v2 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v2 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v1_e999a6c5/code/scripts/launch_phase1_ccpc_leo12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v2.launch.out 2>&1 < /dev/null & echo $!
```

## 5.健康门、artifact与性能门

启动后核launcher PID/CWD/cmdline、12 child、GPU映射、日志增长、CONFIG-CCPC-LEO和epoch。明确忽略禁用项、未执行项或0/0统计的N/A `nan`。技术停止只接受：`train/loss_total`、`loss_ccpc_leo`、CCPC LEO feature gradient或模型输出非有限；明确异常/Traceback；OOM/CUDA故障；进程异常退出；路径/hash/overwrite/P0错误；或预注册无进展。失败不重启、不远端修码，保留partial。

每任务预期`final_ssdg.pth`、metrics CSV/JSONL、CCPC config/terminal/heldout/resource receipt；log根预期`pids.tsv`和`completion.tsv`。训练完整后执行新run ID的唯一postfreeze评估：clean known、source proxy连续排序与三场景同physical LEO C/G对照，proxy/held零fit、零校准、零选参。

五项门仍为：技术健康；6/6 clean known四项G-C≥−2pp；18/18 LEO原子格四项G-C≥−2pp且总体明确改善；source proxy连续排序相对C同向；真实checkpoint与bundle闭环。任一门失败即`REJECT_CCPC_LEO_NO_RETRY`，不进入Phase3。
