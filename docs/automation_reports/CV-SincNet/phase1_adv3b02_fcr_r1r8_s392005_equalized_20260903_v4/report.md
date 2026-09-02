# ADV3B02→FCR R1-R8重新发布v4预登记

- run_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`
- 替代原因：v3首轮遥测在非MUSE路径访问未初始化`rc4_route`；v4仅初始化可选遥测变量。
- 当前只启动seed392005的`ADV3B02_CORE90_SOFT_E200`；R1-R8等待其E200 `final_ssdg.pth`。
- source：ManySig equalized；receiver=`[1,3,4,6,8]`、day=`[1,2,3]`；`L_s/U_s/V=6300/56700/27000`。
- target训练边界：`test_eval_policy=never`；不训练、不筛选、不选择checkpoint；最后epoch冻结后才独立预测与评分。
- checkpoint：final-only；`best_metric=clean_val_tx`仅满足source-only兼容检查，held-out joint guard关闭。
- GPU：ADV3B02使用GPU0；不干预既有任务。R1-R8当前未启动。
- Git固定版本：`8f1de7971853aa9650e4f83d6ad979f359c434c2`。
- release：本地`E:\type10-7\releases\phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_8f1de797.zip`；远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_8f1de797`；归档SHA256本地/远端一致：`6D8484618441CE2E8470556864C82D8DF64A3BD8E8D826A5D5713428CFE40B10`。
- 发布前验证：远端编译通过；真实checkpoint无query smoke通过；真实训练入口dry-run通过。
- 启动命令：`bash code/scripts/launch_phase1_adv3b02_fcr_r1r8_s392005_20260903.sh phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4`。
- 远端CWD：`/home/szu2070436088`；launcher PID=`381957`；ADV3B02训练PID=`382450`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200`。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200.out`。
- 启动健康检查：PID/CWD/cmdline/run-root绑定正确；PID`382450`绑定GPU0，显存约2802MiB；日志已增长至17991字节并完成E002/200，未发现Traceback。中间checkpoint标记为`NOT_SAVED_FINAL_ONLY`，R1-R8仍未启动，等待训练结束后写入的E200 `final_ssdg.pth`。
- 当前状态：`RUNNING`。
