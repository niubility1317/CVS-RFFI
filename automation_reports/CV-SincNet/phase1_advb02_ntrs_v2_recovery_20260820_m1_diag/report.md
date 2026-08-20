# M1历史checkpoint只读诊断

- 状态：`LOCAL_VERIFIED`
- run ID：`phase1_advb02_ntrs_v2_recovery_20260820_m1_diag`
- commit：`b8bb34ee299e984dccd52a0a06765d26b3a8419e`
- 输入checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r3/ADVB02_NTRS_LEO_WEAK_E200/final_ssdg.pth`。
- 动作：只读重评，不更新checkpoint或模型状态；输出raw/robust/fused、rescued/harmed、门控活跃率和分布。
- 场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- GPU：4。
- output/log：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_v2_recovery_20260820_m1_diag`；`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_v2_recovery_20260820_m1_diag`。
- 预期artifact：独立诊断`final_eval.json/txt`及日志。
- 停止规则：仅评估技术失败；不修改或重启历史M1。
