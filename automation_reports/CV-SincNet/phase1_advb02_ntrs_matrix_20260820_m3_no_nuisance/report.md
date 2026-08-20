# NTRS矩阵M3无干扰因子分解实验报告

- 状态：`ANALYZED`
- run ID：`phase1_advb02_ntrs_matrix_20260820_m3_no_nuisance`
- candidate：`ADVB02_NTRS_NO_NUISANCE_LEO_WEAK_E200`
- profile：`no_nuisance_factorization`
- 实现提交：`b92648f2731ed39775a101ea74c52ecb85421371`
- GPU：3；seed：`392034`
- Phase1角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 唯一方法差异：相对M1仅将receiver/day/channel、context-TX去泄漏、条件去相关和共享receiver损失置0。
- 信道：训练和最终测试仅三种LEO_WEAK，禁止`mixed_orbit`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_matrix_20260820_m3_no_nuisance`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m3_no_nuisance`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_advb02_ntrs_matrix_20260820_m3_no_nuisance RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_matrix_20260820_m3_no_nuisance LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m3_no_nuisance GPU=3 SEED=392034 NTRS_PROFILE=no_nuisance_factorization bash code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh </dev/null > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m3_no_nuisance.launcher.out 2>&1 &
```

训练完成后必须保存clean和三种LEO_WEAK逐场景独立测试。仅技术/协议故障可停止，低性能不停止。

启动于`2026-08-20T03:26:55+08:00`；launcher PID=`3481784`，trainer PID=`3481807`。PID/CWD/cmdline/run root/GPU3映射和日志增长均已核对，启动前GPU3活跃训练数为0。

E003检查：`train_optimizer_step_applied=1.0`，`train_skipped_nonfinite_grad=0.0`，train TX=`2.7083%`，source val TX=`73.8571%`；当前完整日志无确定性异常标记。

## 最终结果

- E200最终checkpoint独立测试闭合，`train_exit=0`、`eval_exit=0`、加载键差异为0。
- clean总体/严格/floor：`84.479%/77.912%/66.092%`。
- clear、low-elev、rain总体：`54.804%/52.182%/52.623%`；LEO均值`53.203%`。
- 三场景严格值：`47.428%/45.650%/45.782%`；严格均值`46.287%`。
- 8/9000个batch发生AMP梯度跳步，优化步执行率`99.911%`，没有非有限loss跳步或异常终止。
- checkpoint SHA256：`1ee982a4c8a7ecf7ccbf5eaacc7b193f840bd000c67a7293fab5ce3b9402d2b6`。
- 相对完整M1，LEO均值提高`1.585`个百分点；但相对M0仍下降`17.254`个百分点，不晋级。
