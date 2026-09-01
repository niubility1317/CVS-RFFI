# ADV3B02-NMFDU-GATE-V1最小预登记

- run ID：`phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1`
- Git提交：`9f5bf5a04812e07110f51314c678ff4c75bcd67f`
- 候选矩阵：M0=`ADV3B02_CORE90_SOFT_E200`历史checkpoint只评估；M1=五分支全程等权容量对照；M2=`I_b+g_null`；M3=`I/D/S/U+g_null,δ=0`；M4=`I/D/S/U+g_null+有界δ`
- 固定条件：seed=`392002`，epochs=`200`，`L_s/U_s/V=0.07/0.63/0.30`，`lambda_sat_cls=0.68`，`lambda_sat_cons=0`
- 命令：`RUN_ID=phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1 GPU_MAP=4,5,6,7 bash code/scripts/launch_phase1_adv3b02_nmfdu_gate_v1_queue_20260901.sh`
- 环境/CWD：N607普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；release根目录
- 输入：`Dataset_WigSig/ManySig.pkl`；M0 checkpoint=`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 输出：`runs/phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1/{M1,M2,M3,M4}`
- 日志：`logs/phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1/{M1,M2,M3,M4}.out`
- GPU：预登记M1/M2/M3/M4依次使用GPU4/5/6/7；启动前以N607资源preflight为准，且每GPU训练进程不超过2个
- 停止规则：仅在数据/query越权、错误split/receiver/seed/场景、输出覆盖、错误checkout、进程归属不清、无prediction闭合或同一确定性系统异常导致合法产物无法产生时停止；低性能不停止
- 预期artifact：每个训练row的最终checkpoint、训练日志、clean及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分场景评估、门控诊断与prediction；M0保存同协议评估结果
