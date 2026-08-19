# ADVB02 NTRS LEO_WEAK Phase1实验报告（r2）

## 当前结论

- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- r2是r1首batch CUDA AMP确定性技术失败后的唯一新run；不覆盖r1任何partial artifact。
- r2因确定性非有限梯度导致所有优化步被跳过，已按预登记技术停止规则停止；没有r2性能结果。

## 最小预登记

- run ID：`phase1_advb02_ntrs_leo_weak_20260820_r2`
- candidate：`ADVB02_NTRS_LEO_WEAK_E200`
- base candidate：`ADV3B02_CORE90_SOFT_E200`
- Git分支：`codex/advb02-ntrs-leo-weak-20260820`
- 执行代码提交：`e3f17fcef2e57d1c76dd9027e2e01e748393d566`
- 单seed：`392034`
- Phase1源角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 训练增强：`concat_masked`，仅`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- 独立最终测试：clean及上述三种LEO_WEAK逐场景；不得用聚合均值替代逐场景结果。
- 历史`mixed_orbit`：禁止使用。
- GPU：`1`；启动前再次执行占用与路径核对，且不超过每GPU两个训练任务。

## 方法与修复冻结

NTRS方法保持r1完整设计不变：40维分组物理描述符、fast/slow上下文、`L=3`有界广义复数校正器、source clean/LEO成对差分rank-8切空间、最大`alpha=0.20`的恒等稳健层、raw/robust双头、安全融合与回退、receiver/day/channel分解头、TX去泄漏、类别条件去相关、correctability与安全损失，以及S1/S2-a/S2-b/S3分阶段训练。PA和domain路径读取原始IQ；unknown rescue和target adapter均关闭。

r2相对r1只有一个修复：correctability概率BCE在显式float32且关闭autocast的小范围内计算，保留原概率语义和梯度；没有调整模型结构、loss权重、数据角色、增强场景、seed或训练日程。

## 环境、输入与输出

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\advb02-ntrs-leo-weak-20260820`
- N607进程CWD与release workspace：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/e3f17fce/workspace`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 源数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 真实checkpoint冒烟输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r2`
- candidate output：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r2/ADVB02_NTRS_LEO_WEAK_E200`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r2`
- outer launcher log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r2.launcher.out`

## 精确启动命令

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/e3f17fce/workspace
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/e3f17fce/workspace WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_advb02_ntrs_leo_weak_20260820_r2 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r2 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r2 GPU=1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/e3f17fce/workspace/code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh </dev/null > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r2.launcher.out 2>&1 &
```

## 技术停止规则

仅在协议/场景/seed/数据角色错误、错误release或CWD、输出碰撞、进程归属不清、确定性同类预prediction异常至少重复两次、无法产生最终checkpoint或独立测试prediction闭合时停止本run，并只处理已核实归属于该run的进程树。低性能不触发技术停止；不干预r1、CRRA或任何其他任务。

## 预期artifact

- `final_ssdg.pth`
- `metrics_epoch.csv`
- `metrics_epoch.jsonl`
- `phase1_terminal_status.json`
- `independent_final_eval/final_eval.json`
- `independent_final_eval/final_eval.txt`
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`的逐场景指标与NTRS只读遥测

## 本地验证与审查边界

- 新增CUDA AMP correctability回归测试先稳定复现r1错误，修复后由红转绿。
- NTRS核心、模型、训练、协议负测、评测、launcher及CRRA/拼接增强/checkpoint重建共16个聚焦测试模块、85项测试全部通过；相关`py_compile`通过。
- 候选已完成一次独立P0/P1审查；不重复全量审查。r2只接受针对已定位AMP问题的定点验证。
- 已知非阻断旧测试：`test_post_stage_trainers.py`两项基线陈旧断言仍要求历史`mixed_orbit`默认值及旧损失源码字面量；不会据此退回旧协议。

## N607发布验证

- release远端编译及launcher语法检查通过。
- CUDA AMP定点验证：`PASS_NTRS_CORRECTABILITY_CUDA_AMP`，loss和梯度均有限，target为`[1.0,0.0]`。
- 真实checkpoint无query冒烟：`PASS_REAL_ADV3B02_CHECKPOINT_SOURCE_ONLY_NO_QUERY`；只读取1条source样本、0条query，63个预期NTRS新增键、0个unexpected key，评估态NTRS状态修改数为0。
- 2026-08-20T01:59:15+08:00唯一启动；launcher PID=`3433901`，trainer PID=`3433919`。PID为启动时快照。
- launcher与trainer的实际CWD均为`e3f17fce/workspace`；release入口、run ID、candidate、`--use_ntrs`和三种LEO_WEAK参数匹配，`mixed_orbit`缺席。
- GPU1进程映射到trainer；训练日志从12111增长到29110字节，已出现`[EPOCH-END] E004/200`且无Traceback。

## 发布映射

- 本地唯一release归档：`E:\type10-7\local_artifacts\phase1_advb02_ntrs_leo_weak_20260820_r2\phase1_advb02_ntrs_leo_weak_20260820_r2_e3f17fce.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/e3f17fce/phase1_advb02_ntrs_leo_weak_20260820_r2_e3f17fce.tar.gz`
- SHA256：`e6d68d372abb8ba58f63488f56b4bfb12548d825a009befdfb47558cad663741`；本地与远端单次比对一致。

## 状态记录

- `LOCAL_VERIFIED`：AMP修复提交已推送，远端分支OID与本地`HEAD`一致；85项聚焦测试通过。
- `LANDED`：新release归档同步、单次SHA比对、远端编译、真实checkpoint无query冒烟和AMP定点验证均通过。
- `RUNNING`：唯一launcher已启动；PID/CWD/cmdline/GPU/log增长核对通过，观察到E004/200且无Traceback。
- `ARTIFACTS_COMPLETE`：仅在训练及clean和三种LEO_WEAK独立测试全部完成后记录。
- `ANALYZED`：仅在同row结果分析完成后记录。

## 最终技术诊断与停止记录

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 2026-08-20T02:50:19+08:00在再次核对PID、CWD、cmdline和run root归属后，仅向r2 trainer PID=`3433919`发送`SIGTERM`；launcher随后自行退出。r1、CRRA和其他任务未受影响。
- 停止时完整`metrics_epoch.csv`包含E001–E069共69行：`train_skipped_nonfinite_grad`始终为1.0，`train_optimizer_step_applied`始终为0.0，`train_tx_acc`始终为0.0%，`val_tx_acc`始终为0.238095%；这证明69个epoch的所有优化步骤均被非有限梯度保护跳过。
- 完整训练日志没有Traceback；该run不是进程崩溃，而是确定性的数值反向异常和训练无进展。由于没有任何成功优化步，r2数据不得解释为候选性能。
- launcher收尾状态为`train_exit=143`；由于按技术停止规则终止且不存在`final_ssdg.pth`，独立评测记录为`eval_exit=6 reason=final_ssdg_missing`。没有clean或三种LEO_WEAK最终测试结果，不能标记为`ARTIFACTS_COMPLETE`。
- 根因已在本地最小复现：NTRS的物理校正能量、切空间残差、预校正能量和最终校正能量直接使用`sqrt(mean(square))`。NTRS初始化和S1冻结阶段会产生精确零残差，零点处开平方导数为无穷；即使对应NTRS损失权重为0，反向图仍出现`0×Inf -> NaN`并污染主干梯度。
- 修复将上述4处改为零值仍精确为0、零点导数有限的RMS计算，并新增S1零权重NTRS损失不得污染主干梯度的回归测试；S1、S2-a、S2-b和S3真实模型反向均已验证为有限。
- r2原始日志、69行CSV/JSONL和所有partial artifacts均原路径保留，不覆盖、不删除；新验证必须使用新的run ID。
