# D3旧V1结构公平学习率实验

- 状态：`RUNNING`
- run ID：`phase1_advb02_ntrs_v2_recovery_20260820_d3_v1_fairlr`
- commit：`b8bb34ee299e984dccd52a0a06765d26b3a8419e`
- profile：`v1_fair_core_lr`
- 差异：保留旧NTRS-v1双前向、独立robust head和完整损失，仅将骨干恢复为基线学习率。
- seed/角色：`392034`；`0.07/0.63/0.15/0.15`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_v2_recovery_20260820/b8bb34ee/workspace`
- output/log：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_v2_recovery_20260820_d3_v1_fairlr`；`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_v2_recovery_20260820_d3_v1_fairlr`
- GPU：2。
- 命令：`RUN_ID=phase1_advb02_ntrs_v2_recovery_20260820_d3_v1_fairlr GPU=2 SEED=392034 NTRS_PROFILE=v1_fair_core_lr bash code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh`
- 预期artifact：E200`final_ssdg.pth`、训练日志、clean及三种LEO_WEAK逐场景`final_eval.json/txt`和NTRS遥测。
- 停止规则：仅系统性技术失败；低性能不停止。
- 启动：launcher PID=`3739408`，trainer PID=`3739466`，CWD/profile/GPU/seed/输出绑定已核验，日志已增长。
