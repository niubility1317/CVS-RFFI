# V2-1最小共享头实验

- 状态：`RUNNING`
- run ID：`phase1_advb02_ntrs_v2_recovery_20260820_v2_min_shared`
- commit：`b8bb34ee299e984dccd52a0a06765d26b3a8419e`
- profile：`v2_min_shared_head`
- 差异：单身份前向、共享CosFace头、无LayerNorm、无独立robust head、无慢状态/物理校正/切空间/因子头，仅保留有界直接残差和四项最小训练信号。
- seed/角色：`392034`；`0.07/0.63/0.15/0.15`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_ntrs_v2_recovery_20260820/b8bb34ee/workspace`
- output/log：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_ntrs_v2_recovery_20260820_v2_min_shared`；`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_ntrs_v2_recovery_20260820_v2_min_shared`
- GPU：3。
- 命令：`RUN_ID=phase1_advb02_ntrs_v2_recovery_20260820_v2_min_shared GPU=3 SEED=392034 NTRS_PROFILE=v2_min_shared_head bash code/scripts/launch_phase1_advb02_ntrs_leo_weak_20260820.sh`
- 预期artifact：E200`final_ssdg.pth`、训练日志、clean及三种LEO_WEAK逐场景`final_eval.json/txt`和NTRS遥测。
- 停止规则：仅系统性技术失败；低性能不停止。
- 启动：launcher PID=`3739411`，trainer PID=`3739457`，CWD/profile/GPU/seed/输出绑定已核验，日志已增长。
