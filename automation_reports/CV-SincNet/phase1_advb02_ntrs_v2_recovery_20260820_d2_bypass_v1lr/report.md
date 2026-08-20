# D2严格旁路加旧V1学习率实验

- 状态：`LOCAL_VERIFIED`
- run ID：`phase1_advb02_ntrs_v2_recovery_20260820_d2_bypass_v1lr`
- commit：`b8bb34ee299e984dccd52a0a06765d26b3a8419e`
- profile：`v2_identity_bypass_v1_lr`
- 差异：前向严格旁路，但骨干使用旧V1分段低学习率，用于单独测量优化预算损失。
- seed/角色：`392034`；`0.07/0.63/0.15/0.15`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_v2_recovery_20260820/b8bb34ee/workspace`
- output/log：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_v2_recovery_20260820_d2_bypass_v1lr`；`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_v2_recovery_20260820_d2_bypass_v1lr`
- GPU：1。
- 命令：`RUN_ID=phase1_advb02_ntrs_v2_recovery_20260820_d2_bypass_v1lr GPU=1 SEED=392034 NTRS_PROFILE=v2_identity_bypass_v1_lr bash code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh`
- 预期artifact：E200`final_ssdg.pth`、训练日志、clean及三种LEO_WEAK逐场景`final_eval.json/txt`。
- 停止规则：仅系统性技术失败；低性能不停止。
