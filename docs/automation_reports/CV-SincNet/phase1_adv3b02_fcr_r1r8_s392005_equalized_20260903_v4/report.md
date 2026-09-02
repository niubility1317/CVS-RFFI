# ADV3B02→FCR R1-R8重新发布v4预登记

- run_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`
- 替代原因：v3首轮遥测在非MUSE路径访问未初始化`rc4_route`；v4仅初始化可选遥测变量。
- 当前只启动seed392005的`ADV3B02_CORE90_SOFT_E200`；R1-R8等待其E200 `final_ssdg.pth`。
- source：ManySig equalized；receiver=`[1,3,4,6,8]`、day=`[1,2,3]`；`L_s/U_s/V=6300/56700/27000`。
- target训练边界：`test_eval_policy=never`；不训练、不筛选、不选择checkpoint；最后epoch冻结后才独立预测与评分。
- checkpoint：final-only；`best_metric=clean_val_tx`仅满足source-only兼容检查，held-out joint guard关闭。
- GPU：ADV3B02使用GPU0；不干预既有任务。R1-R8当前未启动。
- 当前状态：`LOCAL_VERIFIED`。
