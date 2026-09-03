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

## ADV3B02 E200基线独立测试

- eval_id：`phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4_ADV3B02_baseline_eval_v1`。
- checkpoint：`ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`，仅使用E200最终checkpoint，不进行选模。
- 数据：ManySig equalized；target receiver=`[0,2,5,7,9,10,11]`、day=`[0,1,2,3]`、TX=`[0,1,2,3,4,5]`；每场景168000条。
- 场景：`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 隔离：prepare生成独立无标签IQ package和truth sidecar；predictor仅读取IQ package及稳定opaque `sample_id`；prediction完成后由独立scorer连接truth。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/baseline_eval_v1`。
- GPU：GPU0；用户已明确允许在现有进程数上继续增加实验，不停止或干预现有任务。
- 预期artifact：`truth/truth_sidecar.json`、`inputs/iq.npy`、`inputs/manifest.json`、`prediction/predictions.json`、`prediction/score.json`及各阶段日志。
- 技术停止规则：仅在路径覆盖风险、checkpoint/数据绑定错误、prediction不完整、scorer连接错误、确定性异常、OOM或非有限输出时停止本评估；不因性能高低停止。
- 当前状态：`LOCAL_VERIFIED`。
