# PairBiCAD P0–P4 r2正式矩阵预登记

## 当前状态

- 状态：`LOCAL_VERIFIED`。
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
- Python：`/home/szu2070436088/miniconda3/envs/ssr-gpu/bin/python`。
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
