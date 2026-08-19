# ADVB02 NTRS LEO_WEAK Phase1实验报告

## 当前结论

- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- r1已在首batch因CUDA AMP技术错误自然结束；没有checkpoint、独立测试或性能结果。
- 本实验是`ADV3B02_CORE90_SOFT_E200`上的独立新版本`ADVB02_NTRS_LEO_WEAK_E200`，不修改正在运行的CRRA实验。

## 最小预登记

- run ID：`phase1_advb02_ntrs_leo_weak_20260820_r1`
- candidate：`ADVB02_NTRS_LEO_WEAK_E200`
- base candidate：`ADV3B02_CORE90_SOFT_E200`
- Git分支：`codex/advb02-ntrs-leo-weak-20260820`
- Git提交：`11d2cfd4684de4525c760755aa56ba07f103f675`
- 单seed：`392034`
- Phase1源角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 训练增强：`concat_masked`，仅`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- 独立最终测试：clean及上述三种LEO_WEAK逐场景；禁止用聚合均值替代逐场景结果。
- 历史`mixed_orbit`：本实验禁止使用。
- GPU：`1`；2026-08-20T01:38:41+08:00只读preflight显示GPU1空闲，GPU0由既有CRRA占用。

## 方法冻结

NTRS按指导完整实现：40维分组物理描述符、fast/slow上下文、`L=3`有界广义复数校正器、source clean/LEO成对差分rank-8切空间、最大`alpha=0.20`的恒等稳健层、raw/robust双头、安全融合与回退、receiver/day/channel分解头、TX去泄漏、类别条件去相关、correctability与安全损失，以及S1/S2-a/S2-b/S3分阶段训练。PA和domain路径读取原始IQ；unknown rescue和target adapter均关闭。

## 环境、输入与输出

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\advb02-ntrs-leo-weak-20260820`
- N607发布CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/11d2cfd4/workspace`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 源数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 真实checkpoint冒烟输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r1`
- candidate output：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r1/ADVB02_NTRS_LEO_WEAK_E200`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r1`
- outer launcher log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r1.launcher.out`

## 精确启动命令

```bash
nohup env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/11d2cfd4/workspace WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_advb02_ntrs_leo_weak_20260820_r1 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_leo_weak_20260820_r1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r1 GPU=1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/11d2cfd4/workspace/code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh </dev/null > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_leo_weak_20260820_r1.launcher.out 2>&1 &
```

## 技术停止规则

仅在协议/场景/seed/数据角色错误、错误release或CWD、输出碰撞、进程归属不清、确定性同类预prediction异常至少重复两次、无法产生最终checkpoint或独立测试prediction闭合时停止本run，并只处理已核实归属于该run的进程树。低性能不触发技术停止；不干预任何其他任务。

## 预期artifact

- `final_ssdg.pth`
- `metrics_epoch.csv`
- `metrics_epoch.jsonl`
- `phase1_terminal_status.json`
- `independent_final_eval/final_eval.json`
- `independent_final_eval/final_eval.txt`
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`的逐场景指标与NTRS只读遥测

## 本地验证与独立审查

- NTRS核心、模型、训练、协议负测、评测、launcher及CRRA/拼接增强/checkpoint重建回归共16个聚焦测试模块全部通过。
- `py_compile`、launcher的`bash -n`、训练入口`--help`及launcher dry-run通过。
- 独立P0/P1审查：无P0/P1；审查员独立验证77项NTRS/CRRA回归、launcher和训练/评测入口，未修改文件。
- 已知非阻断旧测试：`test_post_stage_trainers.py`仍有两项基线陈旧断言，分别要求历史`mixed_orbit`默认值及旧损失源码字面量；`HEAD`基线已存在，不由NTRS引入。本实验不会退回`mixed_orbit`。

## N607发布验证

- release远端编译及launcher语法检查通过。
- 真实checkpoint无query冒烟：`PASS_REAL_ADV3B02_CHECKPOINT_SOURCE_ONLY_NO_QUERY`。
- 冒烟只读取1条source样本、0条query；ADV3B02 checkpoint加载到NTRS模型时有63个预期NTRS新增键、0个unexpected key；前向输出有限，评估态NTRS状态修改数为0。

## 系统技术失败与处置

- 启动时间：`2026-08-20T01:48:28+08:00`；结束时间：`2026-08-20T01:48:56+08:00`。
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- trainer在首batch进入`ntrs_correctability_loss`时，CUDA AMP拒绝概率形式`binary_cross_entropy`，报错为`binary_cross_entropy and BCELoss are unsafe to autocast`。
- `train_exit=1`；因无`final_ssdg.pth`，独立测试记`eval_exit=6`，clean和三种LEO_WEAK均无结果。
- r1进程已自然结束，没有执行进程终止；run root、训练日志、outer log和status均原地保留，不删除、不覆盖。
- 根因在本地CUDA环境同函数稳定复现；修复提交`e3f17fcef2e57d1c76dd9027e2e01e748393d566`只把该概率BCE置于显式float32、autocast关闭的小范围内，并新增CUDA AMP回归测试。正式实验转入全新r2。

## 发布映射

- 本地唯一release归档：`E:\type10-7\local_artifacts\phase1_advb02_ntrs_leo_weak_20260820_r1\phase1_advb02_ntrs_leo_weak_20260820_r1_11d2cfd4.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_leo_weak_20260820/11d2cfd4/phase1_advb02_ntrs_leo_weak_20260820_r1_11d2cfd4.tar.gz`
- SHA256：`cea74494ccb30dbe83b32227f9fa8e205dc140d131f01f5562ef3763a24168f5`；本地与远端单次比对一致。

## 状态记录

- `LOCAL_VERIFIED`：实现提交已推送，远端分支OID与本地`HEAD`一致；N607直连、路径、数据、checkpoint和GPU只读preflight通过。
- `LANDED`：release归档同步、单次SHA比对、远端编译和真实checkpoint无query冒烟均通过。
- `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`：首batch AMP异常，`train_exit=1`、`eval_exit=6`；r1不重启、不覆盖，后续使用r2。
- `ARTIFACTS_COMPLETE`：未达到。
- `ANALYZED`：仅完成技术失败归因，不存在性能分析。
