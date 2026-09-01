# ADV3B02-NMFDU-GATE-V1 r2最小预登记

- run ID：`phase1_adv3b02_nmfdu_gate_v1_s392002_20260902_r2`
- 修复提交：`054fcb8ec0b34d02ac206ec923cfdf5ead19d0e5`
- 候选矩阵：M0=`ADV3B02_CORE90_SOFT_E200`历史checkpoint只评估；M1=五分支全程等权容量对照；M2=`I_b+g_null`；M3=`I/D/S/U+g_null,δ=0`；M4=`I/D/S/U+g_null+有界δ`
- 固定条件：seed=`392002`，epochs=`200`，`L_s/U_s/V=0.07/0.63/0.30`，`lambda_sat_cls=0.68`，`lambda_sat_cons=0`
- r1失败指纹：source-only保护拒绝`best_metric=joint_safe`及held-out joint guard；r1无性能结果且全部产物保留
- r2兼容修复：`best_metric=source_val_sat_hmean`、`enable_joint_safe_guard=false`、`checkpoint_selection=final_only`；回归测试`3 passed`，定点复审`FIXED`
- 环境/CWD：N607普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；新release根
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`；M0 checkpoint=`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate_v1_s392002_20260902_r2/{M1,M2,M3,M4}`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate_v1_s392002_20260902_r2/{M1,M2,M3,M4}.out`
- GPU：预登记M1/M2/M3/M4依次使用GPU4/5/6/7；启动前preflight确认，且每GPU训练进程不超过2个
- 停止规则：仅在数据/query越权、错误split/receiver/seed/场景、输出覆盖、错误checkout、进程归属不清、无prediction闭合或同一确定性系统异常导致合法产物无法产生时停止；低性能不停止
- 预期artifact：每个训练row的最终checkpoint、训练日志、clean及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分场景评估、门控诊断与prediction；M0保存同协议评估结果

## r2发布状态

- release提交：`d716c735ddad657be2d2cd2e6e587397ebcb0132`
- release归档：`E:\type10-7\local_artifacts\releases\adv3b02_nmfdu_gate_v1_d716c735.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate_v1_d716c735.tar.gz`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate_v1_d716c735`
- 归档SHA对照：本地与远端均为`2249f2c72cfbd9505ff74c35e9db5ce18b08e2dead55ecb11d5d26dfacd268c7`
- N607验证：原生`bash -n`、远端Python编译和M4 dry-run通过；dry-run读回`source_val_sat_hmean/false/final_only`
- 真实checkpoint无query smoke：`PASS`；严格加载、52个NMFDU新state、23组非零梯度，query/Phase2访问均为`false`
- 当前资源：GPU4–6各有2个MARC-OT进程，GPU7有1个；未启动r2，等待满足每GPU最多2个进程的资源边界
