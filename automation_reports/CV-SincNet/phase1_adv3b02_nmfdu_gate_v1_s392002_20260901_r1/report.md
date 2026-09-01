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

## 发布与启动绑定

- 代码变更：NMFDU规范激励、五分支局部门、Fisher门、训练目标/三阶段控制、M0–M4消融、真实checkpoint smoke和不可覆盖launcher
- 本地验证：聚焦协议/机制套件`72 passed`；关键Python模块`py_compile`通过；一次定点复审的四个P1均为`FIXED`
- release提交：`a2d5c05b597cd2b95fc346df345f643859d6e70d`
- release归档：`E:\type10-7\local_artifacts\releases\adv3b02_nmfdu_gate_v1_a2d5c05b.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate_v1_a2d5c05b.tar.gz`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate_v1_a2d5c05b`
- 归档SHA对照：本地与远端均为`6ce908e6121d4f02072ae83cd4ef389b778a62aa25d332258dff1fd993acd96b`
- N607验证：原生`bash -n`、launcher dry-run和远端Python编译通过
- 真实checkpoint无query smoke：`PASS`；结果=`runs/phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1_smoke/nmfdu_real_checkpoint_smoke.json`
- 实际启动命令：`env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate_v1_a2d5c05b WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl CORE90_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1 RUN_ID=phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1 GPU_MAP=4,5,6,7 bash /home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate_v1_a2d5c05b/code/scripts/launch_phase1_adv3b02_nmfdu_gate_v1_queue_20260901.sh`

## r1启动结果

- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 影响范围：M1–M4均在训练前退出；未进入数据加载、训练或性能评估
- 失败指纹：`ValueError: Phase1 source-only checkpoint selection forbids test/receiver/satellite-test best metrics; use --best_metric clean_val_tx or source_val_sat_hmean.`
- 原因：launcher沿用了历史`--best_metric joint_safe`和`enable_joint_safe_guard=true`，两者均与当前source-only保护不兼容
- 处置：保留r1全部日志和空输出目录；本地改为`source_val_sat_hmean`、关闭held-out joint guard并保留`checkpoint_selection=final_only`；回归测试`3 passed`，修复后定点复审为`FIXED`；以全新r2 run ID/release重新发布
