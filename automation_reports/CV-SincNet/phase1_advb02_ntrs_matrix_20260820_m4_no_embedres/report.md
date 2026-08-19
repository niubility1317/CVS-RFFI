# NTRS矩阵M4无嵌入有界残差实验报告

- 状态：`RUNNING`
- run ID：`phase1_advb02_ntrs_matrix_20260820_m4_no_embedres`
- candidate：`ADVB02_NTRS_NO_EMBEDRES_LEO_WEAK_E200`
- profile：`no_embed_residual`
- 实现提交：`b92648f2731ed39775a101ea74c52ecb85421371`
- GPU：4；seed：`392034`
- Phase1角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 唯一方法差异：相对M1将`alpha_max`、minimum-correction、alpha和subspace损失置0，仅关闭嵌入端有界残差；物理粗校正和其他NTRS结构保留。
- 信道：训练和最终测试仅三种LEO_WEAK，禁止`mixed_orbit`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_matrix_20260820_m4_no_embedres`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m4_no_embedres`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_advb02_ntrs_matrix_20260820_m4_no_embedres RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_matrix_20260820_m4_no_embedres LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m4_no_embedres GPU=4 SEED=392034 NTRS_PROFILE=no_embed_residual bash code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh </dev/null > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m4_no_embedres.launcher.out 2>&1 &
```

训练完成后必须保存clean和三种LEO_WEAK逐场景独立测试。仅技术/协议故障可停止，低性能不停止。

启动于`2026-08-20T03:26:55+08:00`；launcher PID=`3481852`，trainer PID=`3481935`。PID/CWD/cmdline/run root/GPU4映射和日志增长均已核对，启动前GPU4活跃训练数为0。

E003检查：`train_optimizer_step_applied=1.0`，`train_skipped_nonfinite_grad=0.0`，train TX=`2.7431%`，source val TX=`73.5794%`；当前完整日志无确定性异常标记。
