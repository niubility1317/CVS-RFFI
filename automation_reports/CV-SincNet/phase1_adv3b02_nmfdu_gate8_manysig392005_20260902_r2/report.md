# ADV3B02-NMFDU Gate8 ManySig392005 r2实验预登记

- run ID：`phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r2`
- 修复代码提交：`11c2cf3843875517bb5e3c0edcdc762dcb4000ab`
- 失败前序：`phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1`，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 候选矩阵：E1=`equal`、E2=`i_only`、E3=`i_d`、E4=`i_d_s`、E5=`physical_fixed`、E6=`physical_full`、E7=`full_no_null`、E8=`full`
- 八行均使用`physical_gate_variant=nmfdu_v1`，不包含ADV3B02基线对比
- 数据：ManySig equalized=`1`；source RX=`1,3,4,6,8`、day=`1,2,3`；target RX=`0,2,5,7,9,10,11`、day=`0,1,2,3`；TX=`0–5`
- source协议：pool=`90000`，`L_s/U_s/V=6300/56700/27000`，物理样本ID两两不交；split/train seed=`392005`
- target评估：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，每场景168000；target只用于最终评估，不参与训练或选模
- 训练：epochs=`200`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`、`checkpoint_selection=final_only`
- release归档：`E:\type10-7\local_artifacts\releases\adv3b02_nmfdu_gate8_manysig392005_11c2cf38.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_11c2cf38.tar.gz`
- release SHA256：`8dd447aa74e07bcfb67bc4654917dc3248a015cbd1c9b13cbcb90221c395dfd4`
- release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_11c2cf38`
- N607环境/CWD：普通账户；Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；launcher进程CWD以启动后读回为准
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r2/{E1,E2,E3,E4,E5,E6,E7,E8}`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r2/{E1,E2,E3,E4,E5,E6,E7,E8}.out`
- GPU：E1–E8分别使用GPU0–7；用户已明确授权本次将`MAX_ACTIVE_PER_GPU`提高为`4`
- 启动命令：`env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_11c2cf38 WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r2 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r2 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r2 MAX_ACTIVE_PER_GPU=4 bash /home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_11c2cf38/code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh`
- 停止规则：仅在数据/query越权、错误split/RX/day/seed/场景、输出覆盖、错误checkout、进程归属不清、无prediction闭合或同一确定性系统异常至少复现两行时停止；不得因低性能停止
- 预期artifact：每行最终checkpoint、训练日志、clean及三个LEO弱场景独立结果、prediction与后续独立scorer同row指标
- 本地验证：两个CUDA autocast回归测试先失败后通过；关键模块`py_compile`通过；完整NMFDU聚焦套件`96 passed`；完整小型NMFDU CUDA autocast前向通过；一次独立P0/P1定点审查`PASS`
- 当前状态：`LOCAL_VERIFIED / RELEASE_PREPARING / NOT_LAUNCHED`

## 发布前远端验证

- release归档本地与N607远端SHA256一致：`8dd447aa74e07bcfb67bc4654917dc3248a015cbd1c9b13cbcb90221c395dfd4`。
- 新release远端Python编译与launcher原生`bash -n`均通过。
- 真实ADV3B02 checkpoint无query smoke在N607 GPU0通过：严格加载、52个NMFDU初始化state key、23个有限非零梯度，source RX/day/equalized/split seed匹配，query truth和Phase2访问均为`false`。
- N607运行环境未安装`pytest`，因此未使用远端测试框架；改由同一远端Python/CUDA直接执行两处autocast回归，结果为`NMFDU_R2_N607_CUDA_AUTOCAST_PASS`。
- 启动前GPU0–7独立计算进程数为`1/2/2/3/2/3/3/2`；在用户授权的上限4内可分别新增一个E1–E8任务。
- 当前状态：`LANDED / READY_TO_LAUNCH / NOT_LAUNCHED`
