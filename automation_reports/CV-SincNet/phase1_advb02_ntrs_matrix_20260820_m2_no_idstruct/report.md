# NTRS矩阵M2无身份结构监督实验报告

- 状态：`ANALYZED`
- run ID：`phase1_advb02_ntrs_matrix_20260820_m2_no_idstruct`
- candidate：`ADVB02_NTRS_NO_IDSTRUCT_LEO_WEAK_E200`
- profile：`no_identity_structure`
- 实现提交：`b92648f2731ed39775a101ea74c52ecb85421371`
- GPU：2；seed：`392034`
- Phase1角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 唯一方法差异：相对M1仅将NTRS sat-KL、margin、relation和class-conditional四项身份结构损失置0。
- 信道：训练和最终测试仅三种LEO_WEAK，禁止`mixed_orbit`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_matrix_20260820_m2_no_idstruct`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m2_no_idstruct`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_matrix_20260820/b92648f2/workspace WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_advb02_ntrs_matrix_20260820_m2_no_idstruct RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_matrix_20260820_m2_no_idstruct LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m2_no_idstruct GPU=2 SEED=392034 NTRS_PROFILE=no_identity_structure bash code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh </dev/null > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_matrix_20260820_m2_no_idstruct.launcher.out 2>&1 &
```

训练完成后必须保存clean和三种LEO_WEAK逐场景独立测试。仅技术/协议故障可停止，低性能不停止。

启动于`2026-08-20T03:26:55+08:00`；launcher PID=`3481710`，trainer PID=`3481735`。PID/CWD/cmdline/run root/GPU2映射和日志增长均已核对，启动前GPU2活跃训练数为0。

E003检查：`train_optimizer_step_applied=1.0`，`train_skipped_nonfinite_grad=0.0`，train TX=`2.6736%`，source val TX=`82.6032%`；当前完整日志无确定性异常标记。

## 最终结果

- E200最终checkpoint独立测试闭合，`train_exit=0`、`eval_exit=0`、加载键差异为0。
- clean总体/严格/floor：`84.412%/77.588%/62.800%`。
- clear、low-elev、rain总体：`53.274%/50.985%/51.416%`；LEO均值`51.892%`。
- 三场景严格值：`46.140%/44.620%/44.917%`；严格均值`45.226%`。
- 8/9000个batch发生AMP梯度跳步，优化步执行率`99.911%`，没有非有限loss跳步或异常终止。
- checkpoint SHA256：`4dc05d73111c8f3cc4691552469e6b95251274e689399bbd586dd52eb8718f12`。
- 相对完整M1，LEO均值提高`0.273`个百分点；但相对M0仍下降`18.566`个百分点，不晋级。
