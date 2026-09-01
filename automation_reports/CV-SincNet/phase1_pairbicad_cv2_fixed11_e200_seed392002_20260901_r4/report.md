# Phase1 PairBiCAD-CV2修复版E200正式矩阵r4

## 当前状态

- 状态：`RUNNING`。
- run ID：`phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r4`。
- 代码提交：`67c25004e10d2b07575a8bff2cd2529caee24a1b`，已push并独立核对远端OID一致。
- 旧r3已固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，partial artifact保留；r4使用新不可覆盖输出根，不复用或覆盖r3。

## r3故障与r4修复

r3两条`CV2-B0`行在200epoch完成后的final/EMA/SWAD一次`V_select`评估中重复触发`_forward_unimplemented() got an unexpected keyword argument 'y_tx'`。根因是最终候选评估传入已加载候选状态的`BiCADXRTrainer`，但该`nn.Module`包装器没有`forward`。

r4只补充`BiCADXRTrainer.forward`，把推理调用透明转发给当前`self.model`。该修复保持候选状态由trainer严格加载，同时不改变训练损失、优化器、数据划分、候选参数、LEO课程、source-only边界或200epoch预算。

## 冻结矩阵

- 12候选：`CV2-B0/B1/B2/B3/D0/D1/D2/D3/T0/T1/T2/T3`。
- fold：1、8；seed：392002；共24行。
- 每行从头训练完整200epochs；不得以6500updates、coverage、收敛状态、低性能或墙钟作为正常提前终止条件。
- ManySig day1/day2/day3；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；严格Phase1 source-only，不访问Phase2、target、support、query或truth。
- 使用现行协议`concat_sat_ce_only`、`lambda_sat_cons=0`和`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`课程。
- 11项机制保持冻结：真实LEO pair、固定200epoch、真实CoverageLedger、coverage warmup、no-early-freeze运行约束、显式双时间尺度、pair梯度5%上限、困难组30%上限、四反馈动态GRL、`V_cal/V_select`物理隔离、final/EMA/SWAD一次选模。

## 本地验证

- 行为回归测试先精确复现远端同款`_forward_unimplemented(y_tx=...)`，修复后`CV2-B0` trainer沿真实source-LORO入口完成推理，底层模型一次调用。
- `code/tests/phase1_bicad_xr`完整回归通过；仅3条既存PyTorch autocast弃用警告。
- `trainer.py`、`train_ssdg.py`和launcher均通过`py_compile`；`git diff --check`通过。
- launcher dry-run读回：r4、24行、12候选、fold1/8、seed392002、全部200epochs、source-only、GPU最大2槽。
- Luna一次独立P0/P1定点审查结论为`PASS / NO_BLOCKING_FINDINGS`：确认final/EMA/SWAD状态在评估前严格加载、选择后恢复；周期性source-LORO仍保持source-only；静态`CV2-B0`与最终`V_select`统一入口可调用。建议按预登记r4启动。

## N607发布计划

- 普通账户；禁止管理员账户。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_cv2_e200_67c25004`。
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r4`。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r4.dispatcher.log`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- GPU：2026-09-01 10:15 CST只读preflight显示GPU0—7均无计算进程，冻结容量为每卡2槽；16行并发、8行排队。
- 正式命令：`python -u code/scripts/launch_phase1_pairbicad_cv2_screen24_20260901.py --run-id phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r4 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_cv2_e200_67c25004 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --gpu-capacities 0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2`。

只读preflight已核对普通账户`szu2070436088`、项目根、8张RTX3090、ManySig和Python路径；r4 run根与release目录均不存在。未发现其他`train_ssdg.py`训练进程。

## 预期artifact与停止规则

每行必须闭合epoch200 final checkpoint、严格重建、final/EMA/SWAD与一次`V_select`选择、Coverage/LR/机制/梯度遥测、Clean和三种LEO弱场景独立JSON以及`ARTIFACTS_COMPLETE.json`。

只允许因数据/query越权、错误candidate/fold/receiver/day/seed/epoch、输出冲突、错误release/CWD、命令无法运行、同一确定性异常重复、进程归属不清或无法形成合法checkpoint/四场景artifact而停止精确run进程树。低性能和中间指标下降不得停止、重启、热补丁或选择性重跑。

## Release与smoke证据

- 单一release归档本地/远端SHA256一致：`a2b2f44d217c5d0c873b914631000acca9a21c83256fef8e221f6bd24f93d828`；远端release已解压，入口、trainer和launcher一次编译通过。
- N607环境没有pytest，因此未把缺少测试依赖误判为代码失败；改用本地先验证的无pytest定点脚本，在正式release上执行`CV2-B0 -> BiCADXRTrainer -> _evaluate_bicad_xr_source_loro`，结果`PASS`，`tx_total=2`且底层模型恰好前向一次。
- 历史真实checkpoint无query smoke在GPU0通过：严格重建missing/unexpected/shape mismatch均为0，fresh optimizer step完成，Clean与三种LEO弱场景均为有限值，`target/Phase2/support/query/truth_access=false`。

## 正式启动证据

- 启动时间：2026-09-01 10:22 CST；dispatcher PID`3441034`。
- dispatcher CWD精确绑定`phase1_pairbicad_cv2_e200_67c25004`，cmdline精确绑定r4、ManySig、普通账户Python及GPU容量`0:2,...,7:2`。
- `plan.json`独立读回：24行、seed392002、终止方式`epochs=200`、8行排队。
- 启动后检查：16个直属worker、GPU0—7各2个本run计算进程，GPU利用率81%—90%；`ARTIFACTS_COMPLETE=0`、`TECHNICAL_FAILURE=0`、确定性致命异常为空。
- 16个`train.log`已创建；启动早期文件仍为0字节，但绑定训练进程和GPU计算持续，因此按预登记规则判定健康启动，不因日志尚未写epoch行而停止或重启。

## 2026-09-01 10:39—11:03系统技术失败与停止

- `CV2-B0-F1-S392002`和`CV2-B0-F8-S392002`均完成200epoch，训练子进程返回码为0并分别写出`bicad_xr_final.pth`（10,567,139 bytes）和完整`metrics_epoch.jsonl`（4,226,205/4,226,543 bytes）。
- 两行随后以同一确定性指纹`FINAL_ARTIFACT_CLOSURE_FAILED`进入技术失败：缺少`checkpoint_runtime.json`、`diagnostics.json`及clean/三种LEO正式评估JSON和日志。该问题与性能无关，根因是r4启动器在重构后只调用`validate_artifact_closure`，没有调用已经存在的`evaluate_final_checkpoint`正式闭合流程。
- 同一指纹在两行重复，满足预登记系统性技术失败停止条件。2026-09-01 11:02 CST使用普通账户精确绑定dispatcher PID`3441034`及其全部后代；所有后代cmdline均同时包含本run ID和release ID后才停止。停止标记独立读回为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，`residual_bound_pids=[]`。
- 停止后`pgrep`未发现本run残留进程，`nvidia-smi`无计算进程。r4输出根、两个final checkpoint、metrics和全部partial artifact原样保留；未覆盖、删除、重启或热补丁。
- r4最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；不得作为科学性能结果。修复在本地Git工作树进行，并使用新release、新run ID`phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5`重新发布。
