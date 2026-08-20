# ADVB02 NTRS LEO_WEAK Phase1实验报告（r3）

## 当前结论

- 状态：`ANALYZED`
- r3是r2确定性非有限梯度技术停止后的唯一新run；不覆盖r1/r2任何artifact。
- 仅修复NTRS零残差RMS的反向数值稳定性；候选结构、loss权重、数据角色、seed、训练日程和LEO_WEAK场景均不变。

## 最小预登记

- run ID：`phase1_advb02_ntrs_leo_weak_20260820_r3`
- candidate：`ADVB02_NTRS_LEO_WEAK_E200`
- base candidate：`ADV3B02_CORE90_SOFT_E200`
- 执行代码提交：`7c32ac84c31946a8172bf1f3ba7e4132235ea8d3`
- 单seed：`392034`
- Phase1源角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 训练增强：`concat_masked`，仅`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- 独立最终测试：clean及上述三种LEO_WEAK逐场景；不得只报告聚合均值。
- 历史`mixed_orbit`：禁止使用。
- GPU：`1`；启动前核对占用，且不超过每GPU两个训练任务。

## 修复与本地验证

r2的69个epoch中，`train_skipped_nonfinite_grad=1.0`且`train_optimizer_step_applied=0.0`始终成立。根因是NTRS的4处零残差RMS直接在0处开平方，产生无穷导数并通过零权重项污染主干梯度。r3将其替换为零值精确保持为0、零点导数有限的RMS计算。

- 新增S1零权重NTRS损失反向回归测试；修复前稳定失败，修复后通过。
- S1、S2-a、S2-b、S3真实模型反向均无非有限梯度。
- NTRS核心、模型、训练、协议负测、评测和launcher共34项聚焦测试通过；相关`py_compile`通过。
- 一次完整本地CUDA AMP试验中，GradScaler从初始高scale自动回退后在第4步开始持续得到有限梯度和成功优化步。
- 候选原独立P0/P1审查结论不变；本次仅接受针对零残差梯度问题的定点验证。

## 环境、输入与输出

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\advb02-ntrs-leo-weak-20260820`
- 计划N607 release workspace：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/7c32ac84/workspace`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 源数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 真实checkpoint冒烟输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r3`
- candidate output：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r3/ADVB02_NTRS_LEO_WEAK_E200`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r3`
- outer launcher log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r3.launcher.out`

## 精确启动命令

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/7c32ac84/workspace
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/7c32ac84/workspace WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_advb02_ntrs_leo_weak_20260820_r3 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r3 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r3 GPU=1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/7c32ac84/workspace/code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh </dev/null > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r3.launcher.out 2>&1 &
```

## 技术停止规则

仅在协议/场景/seed/数据角色错误、错误release或CWD、输出碰撞、进程归属不清、确定性同类预prediction异常至少重复两次、无法产生最终checkpoint或独立测试prediction闭合时停止本run，并只处理已核实归属于该run的进程树。低性能不触发技术停止；不干预r1、r2、CRRA或任何其他任务。

## 预期artifact

- `final_ssdg.pth`
- `metrics_epoch.csv`
- `metrics_epoch.jsonl`
- `phase1_terminal_status.json`
- `independent_final_eval/final_eval.json`
- `independent_final_eval/final_eval.txt`
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`的逐场景指标与NTRS只读遥测

## 发布映射

- 本地release归档：`E:\type10-7\local_artifacts\phase1_advb02_ntrs_leo_weak_20260820_r3\phase1_advb02_ntrs_leo_weak_20260820_r3_7c32ac84.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/7c32ac84/phase1_advb02_ntrs_leo_weak_20260820_r3_7c32ac84.tar.gz`
- 单次本地到远端SHA256：`e397c2de7ca1d3bc7d3b2b5f245a31d09a8a18d6ba133af8e17fdc1e92b3a2ad`，两端一致。

## 发布、冒烟与启动证据

- 远端release workspace完成相关Python文件编译检查，launcher完成Bash语法检查。
- GPU1上的CUDA AMP训练步冒烟通过：初始GradScaler回退后，第4步取得有限梯度并完成优化器更新，结果为`PASS_NTRS_S1_CUDA_AMP finite_step=4 final_scale=8192`。
- 真实`ADV3B02_CORE90_SOFT_E200`checkpoint与一个ManySig源样本完成无query冒烟；结果为`PASS_REAL_ADV3B02_CHECKPOINT_SOURCE_ONLY_NO_QUERY source_samples=1 query_samples=0 missing_ntrs=63 unexpected=0`。`missing_ntrs=63`是旧checkpoint不含新增NTRS参数的预期兼容加载结果。
- r3于`2026-08-20T03:01:25+08:00`在GPU1启动。launcher PID为`3466737`，trainer PID为`3466758`；二者CWD均为本报告声明的`7c32ac84/workspace`，trainer命令、run root、GPU映射和日志增长均已核对。
- 启动命令确认仅含`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，未使用`mixed_orbit`；源角色比例和seed与预登记一致。
- E001：`train_skipped_nonfinite_grad=0.0222222`，`train_optimizer_step_applied=0.9777778`，`val_tx_acc=28.6746%`。
- E002：`train_skipped_nonfinite_grad=0.0`，`train_optimizer_step_applied=1.0`，`val_tx_acc=30.2619%`，梯度遥测为`total=24.444`、`backbone=24.438`、`domain=0.846`。
- 截至E002，日志中无`Traceback`、`RuntimeError`、CUDA error、OOM、`FAIL`或`ERROR`。与r2每个epoch均100%跳过、0%优化器更新相比，零残差梯度污染已被真实训练证据消除。
- 当前只确认启动与修复有效，不构成最终性能结论。训练完成后launcher必须继续执行clean及三种LEO_WEAK逐场景独立测试，产物闭合后方可标记`ARTIFACTS_COMPLETE`。
- 纳入因果矩阵后的E039检查：`train_optimizer_step_applied=1.0`，`train_skipped_nonfinite_grad=0.0`，train TX=`61.5625%`，source val TX=`93.9683%`；当前完整日志无确定性异常标记。

## 最终结果

- 训练与测试：E200最终checkpoint，`train_exit=0`、`eval_exit=0`、`ARTIFACTS_COMPLETE`，checkpoint加载键差异为0。
- clean总体：`84.313%`；严格unseen day/unseen RX：`77.695%`；严格RX floor：`64.975%`。
- `leo_clear_weak`：总体`52.865%`，严格`45.497%`。
- `leo_low_elev_weak`：总体`50.796%`，严格`44.197%`。
- `leo_rain_weak`：总体`51.194%`，严格`44.515%`。
- LEO均值/最差场景：`51.618%/50.796%`；严格均值/最差场景：`44.736%/44.197%`。
- 200行训练记录完整；8/9000个batch发生AMP梯度跳步，优化步执行率`99.911%`，没有非有限loss跳步或异常终止。
- checkpoint SHA256：`5c124c7eca5b843cb11b16ae5f25ce9851808a4b7587b45a481f2df3e31fdef6`。
- 矩阵结论：相对M0，clean下降`3.224`个百分点，LEO均值下降`18.839`个百分点，严格LEO均值下降`19.188`个百分点；完整NTRS不晋级。
