# ADV3B02-NMFDU Gate8 ManySig392005 r3实验预登记

- run ID：`phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3`
- 修复代码提交：`aa0eaf4ba3a63c88cae6147e542bb0d6b69e36e9`
- 失败前序：r1为CUDA Half线性求解技术失败，r2为非MUSE路径RC4遥测未初始化技术失败；二者均为`NO_PERFORMANCE_RESULT`
- 候选矩阵：E1=`equal`、E2=`i_only`、E3=`i_d`、E4=`i_d_s`、E5=`physical_fixed`、E6=`physical_full`、E7=`full_no_null`、E8=`full`
- 八行均使用`physical_gate_variant=nmfdu_v1`，不包含ADV3B02基线对比
- 数据：ManySig equalized=`1`；source RX=`1,3,4,6,8`、day=`1,2,3`；target RX=`0,2,5,7,9,10,11`、day=`0,1,2,3`；TX=`0–5`
- source协议：pool=`90000`，`L_s/U_s/V=6300/56700/27000`；split/train seed=`392005`
- target评估：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，每场景168000；target不参与训练或选模
- 训练：epochs=`200`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`、`checkpoint_selection=final_only`
- release归档：`E:\type10-7\local_artifacts\releases\adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b.tar.gz`
- release SHA256：`6d89477c7374c048d808521b8e1528eac38c6ba71e5474ad8faaff36f6afa4e0`
- release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b`
- N607 Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3/{E1,E2,E3,E4,E5,E6,E7,E8}`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3/{E1,E2,E3,E4,E5,E6,E7,E8}.out`
- GPU：E1–E8分别使用GPU0–7；用户已授权本实验谱系的`MAX_ACTIVE_PER_GPU=4`
- 启动命令：`env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3 MAX_ACTIVE_PER_GPU=4 bash /home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b/code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh`
- 停止规则：仅在数据/query越权、错误split/RX/day/seed/场景、输出覆盖、错误checkout、进程归属不清、无prediction闭合或至少两行出现同一确定性系统异常时停止；不得因低性能停止
- 预期artifact：每行最终checkpoint、训练日志、clean及三个LEO弱场景独立结果、prediction与独立scorer同row指标
- 本地验证：CUDA autocast回归、非MUSE零RC4遥测回归、关键模块`py_compile`通过；NMFDU+FastTrust聚焦套件`119 passed`；两次修复各自的一次独立P0/P1定点审查均`PASS`
- 当前状态：`LOCAL_VERIFIED / RELEASE_PREPARING / NOT_LAUNCHED`

