# PairBiCAD P0–P4 r2正式矩阵预登记

## 当前状态

- 状态：`RUNNING`。
- Run ID：`phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2`。
- 失败前序run：`phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1`，保持只读，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 唯一修复：正式评估恢复器按冻结row和候选协议显式重建`num_receivers/num_days/num_channels`；训练方法、loss、数据、矩阵、seed、update预算和四场景评估不变。
- Git分支：`codex/phase1-pairbicad-p4-20260831`。
- 代码冻结commit：`c25c6974e0a80108aafe6cfb9fd35f1429cfaad1`。
- N607账户：普通账户`szu2070436088`；禁止管理员账户。

## 候选与矩阵

- 候选：`P0/P1/P2/P3/P4`。
- folds：1、8。
- seeds：392001、392002、392003。
- 总行数：5×2×3=30。
- 每行预算：4000 optimizer updates。
- 训练天：day1/day2/day3。
- fold1 source receivers：3、4、6、8；fold8 source receivers：1、3、4、6。
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 物理batch48：16L+32U；clean/LEO拼接网络batch96；单次主干前向。
- 星地信道：`concat_sat_ce_only`兼容边界与`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- target/Phase2/support/query/truth：禁止访问。
- GPU0–7；每GPU最多2个本run训练进程，首波最多16行，其余排队。

## 本地验证与独立审查

- 新回归测试在旧代码上按预期RED：正式恢复器未传入维度。
- 最小修复后GREEN；`code/tests/phase1_bicad_xr`完整相关测试298/298通过。
- launcher与metrics聚焦测试39/39通过；`py_compile`通过。
- r2精确dry-run输出30/30行，候选/fold/seed/day/update/source receiver与本报告一致。
- 真实checkpoint no-query smoke：读取r1的`P4-F1-S392002`U4000 checkpoint，严格重建missing/unexpected均为0，恢复维度为4/3/2；clean和三种LEO弱场景均输出`[256,6]`有限logits；target/Phase2/support/query/truth访问均为false。
- 独立P0/P1定点审查未发现会使r2跑错、越权、覆盖输出、误杀进程、无法启动或无法产生合法prediction的问题。
- `REJECTED_EXTRA_GATE`：不新增seal、成员hash、重复审查、Phase2检查或其他白名单外门槛。

## N607发布计划

- release名称：`phase1_pairbicad_p0p4_c25c6974`。
- 本地归档：`E:\type10-7\release_archives\phase1_pairbicad_p0p4_c25c6974.tar.gz`。
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_c25c6974.tar.gz`。
- 远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_c25c6974`。
- 远端run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2`。
- 远端dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2.dispatcher.log`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`（Python3.10.19，与N607正式入口一致）。
- CWD：上述远端release根。
- 正式入口：`code/scripts/launch_phase1_pairbicad_p0p4_n607_20260831.sh`。

正式命令：

```text
PAIRBICAD_RELEASE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_c25c6974 PAIRBICAD_RUN_ID=phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r2 PAIRBICAD_MAX_JOBS_PER_GPU=2 bash code/scripts/launch_phase1_pairbicad_p0p4_n607_20260831.sh
```

## 预期artifact与停止规则

- 每行：`metrics_epoch.csv`、`metrics_epoch.jsonl`、`bicad_xr_final.pth`、`checkpoint_runtime.json`、`diagnostics.json`、clean及三种LEO评估JSON/log、`ARTIFACTS_COMPLETE.json`或`TECHNICAL_FAILURE.json`。
- run级：`plan.json`、`final_status.json`、dispatcher日志。
- 只有错误candidate/fold/receiver/day/seed/update、source-only越权、输出冲突、错误release/CWD、命令无法运行、无artifact闭合、同一确定性pre-prediction异常至少2行或进程归属不清时才能停止精确绑定的本run进程树并保留partial artifact。
- 不得因中间或最终低性能停止、重启、热补丁或选择性重跑；不得影响无关进程。
- 30/30行只有在严格checkpoint恢复及clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`全部闭合后才可标为`ARTIFACTS_COMPLETE`。
- 只按source LORO同row证据比较P0–P4；不得访问Phase2、目标接收机、support、query或truth。

## N607落地回读

- 发布前GPU0–7均为0%利用率、1MiB显存，无compute process；`/home`可用7.3TB。
- 远端archive、release根、r2 run根、dispatcher日志和PID文件发布前均不存在，不会覆盖r1或其他任务。
- release归档本地→远端SHA256一致：`ec5ca68e90fb3753b63ca19ae49fd9528e3b0d825ec9e0578d8b371f112ac09c`。
- 归档已解压到预登记release根；使用正式入口同一Python3.10.19解释器对launcher、trainer和SSDG入口执行远端`py_compile`，结果PASS。
- 首次compile探测使用了错误的预登记解释器路径并以exit127结束，未执行代码、未创建run；报告已更正为shell入口实际使用的`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

## N607正式启动回读

- 启动时间：2026-08-31 15:11 CST后；dispatcher PID`2855983`，PPID=1。
- dispatcher cmdline精确绑定release`phase1_pairbicad_p0p4_c25c6974`、r2 Run ID、P0–P4、fold1/8、3个seed、U4000和`--max-jobs-per-gpu 2`。
- 第一波16个直属主训练进程均以正式解释器运行，CWD全部为新release根，输出分别绑定r2唯一row目录。
- GPU0–7各恰好2个本run compute process；启动回读利用率86%–91%，显存约1.60–1.86GiB/卡。
- 首波覆盖P0全部6行、P1全部6行和P2前4行；其余14行由dispatcher排队，不突破每GPU2个活跃训练进程。
- run根已有30个预留row目录和`plan.json`；16个`train.log`已建立。启动早期日志为0字节，但PID/CWD/cmdline/GPU计算均健康；不得据此停止。
- 启动回读为0个`ARTIFACTS_COMPLETE`、0个`TECHNICAL_FAILURE`；尚无性能结论。
