# ADV3B02→FCR R1-R8重新发布v2预登记

- run_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v2`
- 替代原因：v1在训练前被历史`best_metric=joint_safe`兼容检查拒绝，无性能结果；v2固定`best_metric=clean_val_tx`，但checkpoint仍只取E200最后一个epoch。
- 顺序：先训练`ADV3B02_CORE90_SOFT_E200_S392005`，其`final_ssdg.pth`生成后才启动R1-R8。
- 数据：ManySig equalized，split_mode=`tx_rx_day_1_7_2`，seed=`392005`；source receiver=`[1,3,4,6,8]`、day=`[1,2,3]`、`L_s/U_s/V=6300/56700/27000`。
- target test：receiver=`[0,2,5,7,9,10,11]`、day=`[0,1,2,3]`、TX=`[0,1,2,3,4,5]`；clean及3个LEO scenario各168000样本。
- target边界：训练期间`test_eval_policy=never`；target不参与训练、筛选或checkpoint选择。
- checkpoint：ADV3B02与R1-R8均只使用第200轮最后权重；R1-R8全部强制从本run ADV3B02 final初始化。
- prediction：正式预测进程只读无标签IQ包和稳定opaque sample_id；独立scorer最后连接分目录truth sidecar。
- GPU：先在GPU0运行ADV3B02；完成后R1-R8各映射GPU0-7。按用户授权旁路现有进程数等待，不干预既有任务。
- 技术停止：仅协议/路径/输出覆盖/确定性执行失败/无prediction闭合/scorer连接失败；低性能不停止。
- Git commit：提交后填写；release与N607路径：提交后填写。
- 当前状态：`LOCAL_VERIFIED`。

## 最终状态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

- ADV3B02在进入训练前因历史`enable_joint_safe_guard=true`被source-only保护拒绝；未生成checkpoint、prediction或性能结果。
- v2专属日志和输出保留，run树自行退出；替代run为`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v3`。
- v3显式关闭held-out joint guard；final-only、source-only、数据、seed、预算和R1-R8矩阵不变。
