# ADV3B02→FCR R1-R8重新发布v3预登记

- run_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v3`
- 替代原因：v2在训练前被历史held-out joint guard拒绝，无性能结果；v3显式`enable_joint_safe_guard=false`。
- 顺序：先训练seed392005的`ADV3B02_CORE90_SOFT_E200`，仅其E200 `final_ssdg.pth`解锁R1-R8。
- source：ManySig equalized；receiver=`[1,3,4,6,8]`、day=`[1,2,3]`；`L_s/U_s/V=6300/56700/27000`。
- target test：receiver=`[0,2,5,7,9,10,11]`、day=`[0,1,2,3]`、TX=`[0..5]`；clean及3个LEO scenario各168000。
- 选择边界：训练期间target eval为0；checkpoint固定最后epoch，`best_metric=clean_val_tx`仅满足source-only兼容检查，不执行best筛选。
- R1-R8：全部强制从本run ADV3B02 final初始化；GPU0-7各一个本批任务。
- prediction/scoring：无标签IQ包+稳定opaque ID；独立truth sidecar仅由独立scorer最后连接。
- 技术停止：仅协议/路径/输出覆盖/确定性执行失败/无prediction闭合/scorer连接失败；低性能不停止。
- 当前状态：`LOCAL_VERIFIED`。
- launcher smoke：批次`--dry-run`会真实调用训练入口的`--dry_run`，执行完整训练前参数校验但不构建数据/模型；通过后立即正式启动。
