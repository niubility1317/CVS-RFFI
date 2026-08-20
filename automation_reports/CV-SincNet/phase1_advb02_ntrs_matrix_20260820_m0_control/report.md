# NTRS矩阵M0 Core90对照实验报告

- 状态：`ANALYZED`
- run ID：`phase1_advb02_ntrs_matrix_20260820_m0_control`
- candidate：`ADVB02_CORE90_LEO_WEAK_CONTROL_E200`
- profile：`control`
- 实现提交：`b92648f2731ed39775a101ea74c52ecb85421371`
- GPU：0
- seed：`392034`
- Phase1角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 唯一方法差异：相对M1关闭NTRS；Core90、训练增强、训练日程、优化器和最终测试不变。
- 训练/测试信道：仅`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，禁止`mixed_orbit`。
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_matrix_20260820_m0_control`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m0_control`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_advb02_ntrs_matrix_20260820_m0_control RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_matrix_20260820_m0_control LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m0_control GPU=0 SEED=392034 NTRS_PROFILE=control bash code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh </dev/null > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m0_control.launcher.out 2>&1 &
```

预期artifact为`final_ssdg.pth`、完整训练metrics、终态文件以及clean和三种LEO_WEAK逐场景`independent_final_eval/final_eval.json/txt`。仅协议/路径/输出碰撞/确定性技术异常或最终测试不能闭合可停止；低性能不停止。

启动于`2026-08-20T03:26:54+08:00`；launcher PID=`3481635`，trainer PID=`3481663`。PID/CWD/cmdline/run root/GPU0映射和日志增长均已核对，启动前GPU0活跃训练数为0。

E005检查：`train_optimizer_step_applied=1.0`，`train_skipped_nonfinite_grad=0.0`，train TX=`28.7153%`，source val TX=`65.7619%`；当前完整日志无确定性异常标记。

## 最终结果

- 训练与测试：E200最终checkpoint，`train_exit=0`、`eval_exit=0`、`ARTIFACTS_COMPLETE`，checkpoint加载键差异为0。
- clean总体：`87.536%`；严格unseen day/unseen RX：`81.048%`；严格RX floor：`73.542%`。
- `leo_clear_weak`：总体`72.490%`，严格`65.795%`。
- `leo_low_elev_weak`：总体`69.442%`，严格`63.032%`。
- `leo_rain_weak`：总体`69.439%`，严格`62.945%`。
- LEO均值/最差场景：`70.457%/69.439%`；严格均值/最差场景：`63.924%/62.945%`。
- 200行训练记录完整；11/9000个batch发生AMP梯度跳步，优化步执行率`99.878%`，没有非有限loss跳步或异常终止。
- checkpoint SHA256：`248fe5aff24c342d6bff8e6e7d65cb5c8b250cfc318b18c655d3e57219e93690`。
- 矩阵结论：M0在六组中clean和全部LEO核心指标均第一，保留为当前同协议基线。
