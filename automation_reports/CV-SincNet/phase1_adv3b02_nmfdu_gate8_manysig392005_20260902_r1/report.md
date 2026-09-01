# ADV3B02-NMFDU ManySig392005八实验最小预登记

- run ID：`phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1`
- 候选矩阵：E1=五分支等权；E2=`I`；E3=`I+D`；E4=`I+D+S`；E5=固定单位系数`I+D+S+U`；E6=可学习全局正系数`I+D+S+U`且无样本校正；E7=完整物理证据+有界样本校正但无null；E8=完整NMFDU。按用户要求不包含ADV3B02对比基线。
- Git分支：`work/adv3b02-nmfdu-gate-v1`
- Git提交：待本地验证后填写
- 命令：`bash code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh`
- 环境/CWD：N607普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；新release根
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，equalized=`1`
- 数据范围：源RX=`1,3,4,6,8`、day=`1,2,3`；目标RX=`0,2,5,7,9,10,11`、day=`0,1,2,3`；TX=`0–5`；split seed=`392005`
- 源划分：pool=`90000`，`L_s/U_s/V=6300/56700/27000`。用户给出的两个13500验证数量不具有不同方法权限，遵循`项目.md`的单一`V=0.30`协议。
- 目标测试：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，每场景168000条；仅测试，不参与训练或checkpoint选择
- 固定训练：train seed=`392005`，epochs=`200`，`lambda_sat_cls=0.68`，`lambda_sat_cons=0`，`best_metric=source_val_sat_hmean`，`checkpoint_selection=final_only`
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1/{E1,E2,E3,E4,E5,E6,E7,E8}`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1/{E1,E2,E3,E4,E5,E6,E7,E8}.out`
- GPU：E1–E8预登记依次使用GPU0–7；启动前检查资源，每GPU训练进程不超过2个
- 停止规则：仅在数据/query越权、错误split/RX/day/seed/场景、输出覆盖、错误checkout、进程归属不清、无prediction闭合或同一确定性系统异常导致合法产物无法产生时停止；低性能不停止
- 预期artifact：八行各自最终checkpoint、训练日志、clean及三种LEO弱场景评估、门控诊断与prediction；prediction完整后由独立scorer连接truth并做同row分析
- 当前状态：`LOCAL_IMPLEMENTATION_IN_PROGRESS / NOT_RELEASED / NOT_LAUNCHED`
